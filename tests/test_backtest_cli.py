"""The backtest command's funding wiring, exercised without a database.

``load_candles`` and the funding loader are patched to serve in-memory frames,
so these tests cover what the command owns -- stress parsing, the perp funding
guard, the funding column, and what it echoes -- rather than Postgres.

Funding reaches a run through two doors and the command owns both. It is a cash
flow the engine charges against held notional, and it is an *input*:
``features.flow.Crowding`` reads a per-bar ``funding_rate`` column and falls
back to a neutral 0.5 without one. The command used to open only the first door,
so a perp run of ``state_machine_v1`` silently ran three features instead of
four.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from strategy_lab import cli
from strategy_lab.features.flow import FUNDING_COLUMN
from strategy_lab.market_data.base import MarketDataIdentity
from tests.conftest import synthetic_ohlcv, synthetic_ohlcv_with_funding

runner = CliRunner()

_PERP = ["--exchange", "binance", "--market-type", "perp", "--symbol", "BTC/USDT"]
_IDENTITY = MarketDataIdentity(
    exchange="binance", market_type="perp", symbol="BTC/USDT", timeframe="4h"
)
# state_machine_v1 warms 2,192 bars, so the funding-column tests need their own
# frame rather than the 900-bar one the cost tests run on.
_MACHINE_BARS = 2400


@pytest.fixture
def candles(monkeypatch):
    df = synthetic_ohlcv(n=900, freq="4h")
    monkeypatch.setattr(cli, "load_candles", lambda **kwargs: df)
    return df


@pytest.fixture
def funding(monkeypatch, candles):
    index = pd.date_range(
        candles.index[0], candles.index[-1], freq="8h", tz="UTC", name="timestamp"
    ) + pd.Timedelta(47, unit="ms")
    frame = pd.DataFrame({"funding_rate": 0.0001}, index=index)
    monkeypatch.setattr(cli, "_funding_rates", lambda identity, df, **_: frame["funding_rate"])
    return frame


@pytest.fixture
def machine_candles(monkeypatch):
    """A perp frame long enough for ``state_machine_v1``, and its settlements.

    ``load_candles`` serves the frame with the funding column **removed**, which
    is what Postgres holds: ``market_candles`` stores raw OHLCV and nothing
    derived. A fixture that handed the column over would let the command skip
    the attachment entirely and leave every assertion below green.
    """
    df = synthetic_ohlcv_with_funding(n=_MACHINE_BARS, freq="4h")
    settlements = df[FUNDING_COLUMN][df[FUNDING_COLUMN] != 0.0]
    monkeypatch.setattr(cli, "load_candles", lambda **kwargs: df.drop(columns=FUNDING_COLUMN))
    monkeypatch.setattr(cli, "_funding_rates", lambda identity, df, **_: settlements)
    return settlements


def _invoke(tmp_path, *args):
    return runner.invoke(
        cli.app,
        [
            "backtest",
            "--report-root",
            str(tmp_path),
            "--timeframe",
            "4h",
            "--strategy",
            "donchian",
            "--exit-mode",
            "opposite_signal_only",
            *args,
        ],
    )


def _machine(tmp_path, *args):
    return runner.invoke(
        cli.app,
        [
            "backtest", "--report-root", str(tmp_path), "--timeframe", "4h",
            "--strategy", "state_machine_v1", "--exit-mode", "opposite_signal_only",
            *args,
        ],
    )


def _costs(tmp_path) -> dict:
    [report_dir] = list(tmp_path.iterdir())
    return json.loads((report_dir / "costs.json").read_text())


def _metadata(tmp_path) -> dict:
    [report_dir] = list(tmp_path.iterdir())
    return json.loads((report_dir / "config.json").read_text())["strategy_metadata"]


def test_cost_stress_renders_every_requested_multiple(candles, funding, tmp_path):
    result = _invoke(tmp_path, *_PERP, "--cost-stress", "1,2,3")

    assert result.exit_code == 0, result.output
    assert [row["multiple"] for row in _costs(tmp_path)["stress"]] == [1.0, 2.0, 3.0]
    echoed = " ".join(result.output.split())
    assert "gross" in echoed and "funding" in echoed and "net" in echoed
    assert "2x costs" in echoed and "3x costs" in echoed


def test_a_bad_cost_stress_is_rejected_with_guidance_not_a_traceback(candles, funding, tmp_path):
    """``,`` would otherwise fall back to a silent 1x run, and ``0`` would reach
    the engine as an uncaught ValueError. Both are the command's job to catch."""
    for bad in ("0", ",", "1,abc"):
        result = _invoke(tmp_path, *_PERP, "--cost-stress", bad)
        assert result.exit_code == 2, f"{bad!r} was accepted: {result.output}"
        assert "--cost-stress" in result.output
        assert result.exception is None or isinstance(result.exception, SystemExit)


def test_a_perp_run_charges_funding_by_default(candles, funding, tmp_path):
    result = _invoke(tmp_path, *_PERP)

    assert result.exit_code == 0, result.output
    assert _costs(tmp_path)["funding_applied"] is True
    # Sign follows the book, which this test does not control; what it pins is
    # that the loader was reached and something was charged.
    assert _costs(tmp_path)["stress"][0]["funding_paid"] != 0
    [report_dir] = list(tmp_path.iterdir())
    assert (report_dir / "funding.csv").exists()


def test_a_partial_funding_history_is_refused_by_name(candles, monkeypatch, tmp_path):
    """One stored settlement satisfies "some funding exists" while charging zero
    for every other one, so the run would report a net-of-funding number that is
    almost entirely gross of carry."""
    index = pd.date_range(
        candles.index[0], candles.index[-1], freq="8h", tz="UTC", name="timestamp"
    )
    stored = pd.DataFrame({"funding_rate": 0.0001}, index=index).iloc[:20]
    monkeypatch.setattr(cli, "load_candles", lambda **kwargs: candles)
    monkeypatch.setattr(
        "strategy_lab.db.funding.load_funding", lambda **kwargs: stored
    )

    result = _invoke(tmp_path, *_PERP)

    assert result.exit_code == 2, result.output
    echoed = " ".join(result.output.split())
    assert "does not cover" in echoed
    assert "fetch-funding" in echoed
    assert str(index[19].date()) in echoed


def test_no_funding_opts_out_explicitly(candles, funding, tmp_path):
    result = _invoke(tmp_path, *_PERP, "--no-funding")

    assert result.exit_code == 0, result.output
    assert _costs(tmp_path)["funding_applied"] is False


def test_an_equity_run_needs_no_funding_and_applies_none(candles, tmp_path):
    result = _invoke(
        tmp_path, "--exchange", "yahoo", "--market-type", "equity", "--symbol", "SPY"
    )

    assert result.exit_code == 0, result.output
    assert _costs(tmp_path)["funding_applied"] is False


def test_size_mode_reaches_the_engine_and_is_recorded(candles, funding, tmp_path):
    # --vol-span 20 because the estimator's warmup is 20x the span and the
    # fixture frame is 900 bars: the production default of 96 needs 1,920 and is
    # refused on it, which is the point of the flag existing.
    result = _invoke(
        tmp_path, *_PERP,
        "--size-mode", "vol-scaled-entry", "--vol-target", "0.4", "--vol-span", "20",
    )

    assert result.exit_code == 0, result.output
    [report_dir] = list(tmp_path.iterdir())
    config = json.loads((report_dir / "config.json").read_text())
    assert config["size_mode"] == "vol-scaled-entry"
    assert config["vol_target"] == 0.4
    assert config["vol_span"] == 20
    assert config["vol_warmup_bars"] == 400
    assert config["warmup_bars"] == 400


def test_the_withdrawn_vol_target_spelling_is_not_quietly_accepted(candles, funding, tmp_path):
    """The mode was renamed because "targeting" claimed continuous rebalancing that
    ``from_signals`` never performed. Silently aliasing the old spelling would keep
    that claim reachable, so it must be rejected rather than mapped."""
    result = _invoke(tmp_path, *_PERP, "--size-mode", "vol-target")

    assert result.exit_code == 2, result.output


def test_vol_scaling_a_self_sizing_strategy_exits_cleanly(candles, funding, tmp_path):
    """An incompatible pair of flags is user error, and must not surface as a traceback."""
    result = runner.invoke(
        cli.app,
        [
            "backtest", "--report-root", str(tmp_path), "--timeframe", "4h",
            "--strategy", "trend_rider_v1_deepseek_v4_pro",
            "--exit-mode", "opposite_signal_only", "--size-mode", "vol-scaled-entry", *_PERP,
        ],
    )

    assert result.exit_code == 2, result.output
    assert "--size-mode fixed" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_a_perp_run_with_no_stored_funding_stops_rather_than_reporting_gross(
    candles, monkeypatch, tmp_path
):
    """Gross-of-carry on a perp reads exactly like a net number and is not tradeable."""
    monkeypatch.setattr(
        "strategy_lab.db.funding.load_funding",
        lambda **kwargs: pd.DataFrame({"funding_rate": []}),
    )

    result = _invoke(tmp_path, *_PERP)

    assert result.exit_code != 0
    assert "fetch-funding" in result.output
    assert "--no-funding" in result.output


def test_the_funding_query_covers_the_final_bar_not_only_its_opening_stamp(monkeypatch):
    """A bar covers an interval, and settlements land up to 47 ms past a boundary.

    Bounding the query at the last candle's *opening* timestamp drops a
    settlement stamped inside the final bar, which either fails the coverage
    check on a complete history or charges that settlement as zero.
    """
    index = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC", name="timestamp")
    df = pd.DataFrame({"close": 100.0}, index=index)
    settled = index + pd.Timedelta(47, unit="ms")

    def fake_load_funding(*, start=None, end=None, **kwargs):
        # Inclusive on both bounds, matching the SQL `>= start` / `<= end`.
        frame = pd.DataFrame({"funding_rate": 0.0001}, index=settled)
        return frame.loc[pd.Timestamp(start) : pd.Timestamp(end)]

    monkeypatch.setattr("strategy_lab.db.funding.load_funding", fake_load_funding)
    identity = MarketDataIdentity(
        exchange="binance", market_type="perp", symbol="BTC/USDT", timeframe="4h"
    )

    rates = cli._funding_rates(identity, df)

    assert rates.index[-1] == settled[-1]


def test_the_funding_column_is_matched_by_containment_not_equality(monkeypatch):
    """A reindex would look right here and be wrong on the stored history.

    Binance stamps settlements up to 47 ms past the boundary -- 3,260 of BTC's
    7,559 stored settlements are off-grid -- so an equality join drops 43% of
    them and reads the survivors onto the wrong bars. Every settlement below is
    stamped late, so a column built by reindexing is entirely zero.
    """
    index = pd.date_range("2024-01-01", periods=12, freq="4h", tz="UTC", name="timestamp")
    df = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1.0}, index=index
    )
    settled = index[::2] + pd.Timedelta(47, unit="ms")
    rates = pd.Series(0.0001, index=settled, name=FUNDING_COLUMN)
    monkeypatch.setattr(cli, "_funding_rates", lambda identity, frame, **_: rates)

    framed, returned = cli._with_funding_column(_IDENTITY, df, enabled=True)

    assert returned is rates, "the settlement series still has to reach the cost ledger"
    column = framed[FUNDING_COLUMN]
    assert column.loc[index[::2]].eq(0.0001).all()
    assert column.loc[index[1::2]].eq(0.0).all()
    assert FUNDING_COLUMN not in df.columns, "the caller's frame must not be mutated"


@pytest.mark.parametrize(
    ("market_type", "enabled"), [("spot", True), ("equity", True), ("perp", False)]
)
def test_nothing_is_attached_where_there_is_no_funding_to_attach(market_type, enabled):
    """Spot and equity have no funding at all, and ``--no-funding`` is a request
    not to use what is stored. Both give the frame back untouched rather than a
    column of zeros, which ``Crowding`` would read as a measurement."""
    df = synthetic_ohlcv(n=20, freq="4h")
    identity = MarketDataIdentity(
        exchange="binance", market_type=market_type, symbol="BTC/USDT", timeframe="4h"
    )

    framed, rates = cli._with_funding_column(identity, df, enabled=enabled)

    assert framed is df
    assert rates is None


def test_a_perp_run_measures_crowding_rather_than_assuming_it(machine_candles, tmp_path):
    """**The regression.** The command loaded funding for the cost ledger and
    then let the frame go to the strategy without it, so ``crowding`` ran at its
    neutral 0.5 and the run recorded that in one word nobody reads. Measured on
    BTC/USDT perp 4h over R5's test half, trained cell: the crowding-neutral run
    returns +16.44% at Sharpe +0.801, the measured one +15.45% at +0.896 -- the
    published figure. A run whose headline is a full Sharpe point away from the
    charter is not a reproduction of it.
    """
    result = _machine(tmp_path, *_PERP)

    assert result.exit_code == 0, result.output
    assert _metadata(tmp_path)["crowding_measured"] is True


def test_the_funding_column_moves_the_fills_and_not_only_a_flag(machine_candles, tmp_path):
    """``crowding_measured`` is a claim about the signals, so check the signals.

    ``trades.csv`` is the artifact that isolates it: funding is charged as a
    post-hoc ledger against a portfolio simulated without it, so no fill can
    move because carry was or was not billed. Anything that differs between
    these two runs came through ``Crowding``.
    """
    assert _machine(tmp_path / "measured", *_PERP).exit_code == 0
    assert _machine(tmp_path / "neutral", *_PERP, "--no-funding").exit_code == 0

    assert _metadata(tmp_path / "measured")["crowding_measured"] is True
    assert _metadata(tmp_path / "neutral")["crowding_measured"] is False

    def fills(root):
        [report_dir] = list(root.iterdir())
        return (report_dir / "trades.csv").read_text()

    assert fills(tmp_path / "measured") != fills(tmp_path / "neutral")


def test_a_spot_run_of_the_same_strategy_attaches_nothing(machine_candles, tmp_path):
    """The attachment is gated on the market, not on the flag alone.

    Spot and equity frames carry no funding at all, which is the case
    ``Crowding``'s fallback exists for -- so the machine still runs there, on
    three measured features and a neutral fourth, and says so.
    """
    result = _machine(
        tmp_path, "--exchange", "binance", "--market-type", "spot", "--symbol", "BTC/USDT"
    )

    assert result.exit_code == 0, result.output
    assert _metadata(tmp_path)["crowding_measured"] is False
