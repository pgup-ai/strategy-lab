"""The backtest command's cost wiring, exercised without a database.

``load_candles`` and the funding loader are patched to serve in-memory frames,
so these tests cover what the command owns -- stress parsing, the perp funding
guard, and what it echoes -- rather than Postgres.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from strategy_lab import cli
from tests.conftest import synthetic_ohlcv

runner = CliRunner()

_PERP = ["--exchange", "binance", "--market-type", "perp", "--symbol", "BTC/USDT"]


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
    monkeypatch.setattr(cli, "_funding_rates", lambda identity, df: frame["funding_rate"])
    return frame


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


def _costs(tmp_path) -> dict:
    [report_dir] = list(tmp_path.iterdir())
    return json.loads((report_dir / "costs.json").read_text())


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
    result = _invoke(tmp_path, *_PERP, "--size-mode", "vol-target", "--vol-target", "0.4")

    assert result.exit_code == 0, result.output
    [report_dir] = list(tmp_path.iterdir())
    config = json.loads((report_dir / "config.json").read_text())
    assert config["size_mode"] == "vol-target"
    assert config["vol_target"] == 0.4


def test_vol_targeting_a_self_sizing_strategy_exits_cleanly(candles, funding, tmp_path):
    """An incompatible pair of flags is user error, and must not surface as a traceback."""
    result = runner.invoke(
        cli.app,
        [
            "backtest", "--report-root", str(tmp_path), "--timeframe", "4h",
            "--strategy", "trend_rider_v1_deepseek_v4_pro",
            "--exit-mode", "opposite_signal_only", "--size-mode", "vol-target", *_PERP,
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
