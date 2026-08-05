"""The research browser's payload, and the one claim the browser rests on.

That claim is marker parity: the entry and exit bars the payload carries are the
entry and exit bars ``run_backtest`` fills on, for the same frame and the same
exit mode. A browser that can differ from a backtest is worse than no browser --
it launders a wrong number through a trustworthy-looking chart -- so the parity
test runs the real engine over a real stored frame rather than asserting the
payload's shape.

``run_backtest`` writes a report directory, so every call here points
``report_root`` at ``tmp_path``: the browser must never put anything in
``reports/``, and neither may its tests.
"""

from __future__ import annotations

import pandas as pd
import pytest

from strategy_lab.api.analysis import (
    Contract,
    build_analysis,
    prepare_frame,
    registered_strategies,
    resolve_strategy,
)
from strategy_lab.backtests import ExitMode, run_backtest
from strategy_lab.db.funding import funding_span
from strategy_lab.features.flow import FUNDING_COLUMN
from strategy_lab.market_data.base import MarketDataIdentity
from tests.conftest import synthetic_ohlcv, synthetic_ohlcv_with_funding

_PERP = MarketDataIdentity(
    exchange="binance", market_type="perp", symbol="BTC/USDT", timeframe="4h"
)
# state_machine_v1 warms 2,192 bars, so anything exercising it needs a frame
# several times that before a single signal exists to compare.
_MACHINE_BARS = 3000
# Past BTC's permanent 40h leading funding gap, which would otherwise refuse the
# frame before a single marker was compared.
_FUNDED_START = "2022-01-01"


def _funded_end() -> str:
    """The window's right edge, taken from stored funding rather than the last bar.

    Any candle refresh fetches bars up to the present and leaves stored funding
    where it was, so an unbounded window grows past the last settlement until
    ``funding_coverage_gaps`` refuses the frame -- turning a marker-parity claim
    red while naming a strategy, when what moved was the data underneath it.
    Bounding at the last stored settlement makes the window self-determined:
    bars a later fetch adds beyond it are outside what this file tests, and the
    trailing gap stays the sub-cadence one the guard is built to tolerate.
    """
    span = funding_span(
        exchange=_PERP.exchange, market_type=_PERP.market_type, symbol=_PERP.symbol
    )
    if span is None:
        pytest.fail(
            f"no stored funding for {_PERP.exchange}/perp/{_PERP.symbol}; run "
            f"strategy-lab fetch-funding --symbol {_PERP.symbol}"
        )
    return str(span[1])


@pytest.fixture
def perp_frame(monkeypatch):
    """A synthetic perp served the way Postgres serves one: OHLCV, no funding.

    ``market_candles`` stores raw candles and nothing derived, so a fixture that
    handed the funding column over would let the payload skip the attachment and
    leave every crowding assertion below green.
    """
    df = synthetic_ohlcv_with_funding(n=_MACHINE_BARS, freq="4h")
    settlements = df[FUNDING_COLUMN][df[FUNDING_COLUMN] != 0.0]
    monkeypatch.setattr(
        "strategy_lab.api.analysis.load_candles",
        lambda **kwargs: df.drop(columns=FUNDING_COLUMN),
    )
    monkeypatch.setattr(
        "strategy_lab.backtests.funding_frame.funding_rates",
        lambda identity, frame, **_: settlements,
    )
    return df


@pytest.fixture
def spot_frame(monkeypatch):
    df = synthetic_ohlcv(n=900, freq="4h")
    monkeypatch.setattr("strategy_lab.api.analysis.load_candles", lambda **kwargs: df)
    return df


_SPOT = MarketDataIdentity(
    exchange="binance", market_type="spot", symbol="BTC/USDT", timeframe="4h"
)


# --------------------------------------------------------------------------
# The test that justifies the feature.
# --------------------------------------------------------------------------


@pytest.mark.db
@pytest.mark.parametrize(
    ("strategy_name", "exit_mode"),
    [
        # One per class of exit ingredient: an engine-side rule, an
        # engine-applied stop level, and the strategy's own signals on the one
        # strategy whose signals depend on the funding column.
        ("donchian", ExitMode.CONTINUATION_FAILURE),
        ("turnaround_v2", ExitMode.SETUP_INVALIDATION_STOP),
        ("state_machine_v1", ExitMode.OPPOSITE_SIGNAL_ONLY),
    ],
)
def test_the_payload_marks_the_bars_the_backtest_actually_trades(
    strategy_name, exit_mode, tmp_path
):
    """**The gate.** Same frame, same exit mode, same fills: bar, side, price, size.

    Compared against ``trades.csv`` rather than against the strategy's raw
    signals, because a signal is not a fill: ``from_signals`` ignores a repeated
    same-direction entry and does nothing with an exit while flat, so a payload
    that marked every signal would put arrows on bars the backtest never traded.
    ``report.py`` draws the frozen report's markers from the same trades frame,
    so this pins the browser to what the report already shows.

    Price and size are part of the claim rather than decoration, and each catches
    what the other cannot. Measured on the 2022 donchian window: reverting the
    slippage rate to its default moves every fill price by up to 0.25% and no
    bar; reverting the *fee* rate moves the price by 1.2e-16 -- a fee is not part
    of a fill price -- and the quantity by up to 14%, because there is less cash
    to deploy. A comparison on bars alone would pass against a payload priced at
    whatever cost model it liked.

    Both ends of the window come from stored data rather than from the clock --
    see :func:`_funded_end` for why the right one has to.
    """
    strategy = resolve_strategy(strategy_name).strategy
    end = _funded_end()
    prepared = prepare_frame(_PERP, strategy=strategy, start=_FUNDED_START, end=end)
    result = run_backtest(
        df=prepared.df,
        strategy=strategy,
        identity=_PERP,
        exit_mode=exit_mode,
        funding=prepared.funding,
        report_root=tmp_path,
    )
    trades = pd.read_csv(result.trades_path)

    payload = build_analysis(
        _PERP,
        strategy_name=strategy_name,
        exit_mode=exit_mode,
        start=_FUNDED_START,
        end=end,
    )

    assert not trades.empty, "a frame with no trades would pass this vacuously"
    _assert_same_fills(_entries(payload), _expected_entries(trades))
    _assert_same_fills(_exits(payload), _expected_exits(trades))


def _assert_same_fills(actual, expected) -> None:
    """Bar and side exactly; price and quantity to within the CSV's precision.

    ``trades.csv`` is the artifact a reader actually opens, so it is what the
    payload is compared against -- and pandas rounds a float on the way out
    (22685.537099999998 is written as 22685.5371), so exact equality would be
    asserting something about the writer rather than about the fills.
    """
    assert [fill[:2] for fill in actual] == [fill[:2] for fill in expected]
    for column, what in ((2, "price"), (3, "size")):
        assert [fill[column] for fill in actual] == pytest.approx(
            [fill[column] for fill in expected], rel=1e-9
        ), f"the {what} of at least one fill differs from trades.csv"


def _entries(payload) -> list[tuple[int, str, float, float]]:
    return [_fill(marker) for marker in payload.markers if marker.kind == "entry"]


def _exits(payload) -> list[tuple[int, str, float, float]]:
    return [_fill(marker) for marker in payload.markers if marker.kind == "exit"]


def _fill(marker) -> tuple[int, str, float, float]:
    return (marker.time, marker.side, marker.price, marker.size)


def _side(direction: str) -> str:
    return "long" if direction == "Long" else "short"


def _expected_entries(trades: pd.DataFrame) -> list[tuple[int, str, float, float]]:
    return sorted(
        (
            int(pd.Timestamp(row["Entry Timestamp"]).timestamp()),
            _side(row["Direction"]),
            float(row["Avg Entry Price"]),
            float(row["Size"]),
        )
        for _, row in trades.iterrows()
    )


def _expected_exits(trades: pd.DataFrame) -> list[tuple[int, str, float, float]]:
    closed = trades[trades["Status"] == "Closed"]
    return sorted(
        (
            int(pd.Timestamp(row["Exit Timestamp"]).timestamp()),
            _side(row["Direction"]),
            float(row["Avg Exit Price"]),
            float(row["Size"]),
        )
        for _, row in closed.iterrows()
    )


@pytest.mark.db
def test_parity_holds_where_every_setting_is_away_from_its_default(tmp_path):
    """Five settings the payload accepts, not one of them left at its default.

    A payload that quietly simulated at the engine's defaults would agree with a
    default backtest and disagree with every other one -- and it would disagree
    in what a fill *was* rather than in when it happened. Measured on this
    window, reverting any single one of the five to its default changes the
    prices or the sizes, while only ``position_pct`` moves a bar at all. So the
    browser's arrows can carry the wrong number while landing in exactly the
    right place, which is the quietest way for a chart to be wrong.
    """
    strategy = resolve_strategy("donchian").strategy
    prepared = prepare_frame(_PERP, strategy=strategy, start="2022-01-01")
    result = run_backtest(
        df=prepared.df,
        strategy=strategy,
        identity=_PERP,
        exit_mode=ExitMode.CONTINUATION_FAILURE,
        failure_bars=7,
        funding=prepared.funding,
        fees=0.004,
        slippage=0.003,
        cash=25_000.0,
        position_pct=0.5,
        report_root=tmp_path,
    )
    trades = pd.read_csv(result.trades_path)

    payload = build_analysis(
        _PERP,
        strategy_name="donchian",
        exit_mode=ExitMode.CONTINUATION_FAILURE,
        failure_bars=7,
        start="2022-01-01",
        fees=0.004,
        slippage=0.003,
        cash=25_000.0,
        position_pct=0.5,
    )

    assert not trades.empty
    _assert_same_fills(_entries(payload), _expected_entries(trades))
    _assert_same_fills(_exits(payload), _expected_exits(trades))


# --------------------------------------------------------------------------
# Provenance.
# --------------------------------------------------------------------------


def test_provenance_records_that_crowding_was_measured_on_a_funded_perp(perp_frame):
    """M20 in one field. Two runs of one strategy differ because one had the
    funding column, and a number shown without that context will eventually
    contradict the charter with no way to see why."""
    payload = build_analysis(
        _PERP, strategy_name="state_machine_v1", exit_mode=ExitMode.OPPOSITE_SIGNAL_ONLY
    )

    assert payload.provenance.funding_attached is True
    assert payload.provenance.crowding_measured is True
    assert payload.provenance.strategy == "state_machine_v1"
    assert payload.provenance.version == "1.0.0"
    assert payload.provenance.exit_mode == "opposite_signal_only"
    assert payload.provenance.failure_bars == 4
    assert payload.provenance.warmup_bars == 2192
    assert payload.provenance.contract == Contract.SIGNAL_SET.value
    assert payload.provenance.first_bar == str(perp_frame.index[0])
    assert payload.provenance.last_bar == str(perp_frame.index[-1])
    assert payload.provenance.bar_count == _MACHINE_BARS
    assert payload.provenance.cost_model == {
        "fee": 0.0005,
        "slippage": 0.0005,
        "cash": 10_000.0,
        "position_pct": 0.95,
    }


def test_declining_funding_is_visible_rather_than_implied(perp_frame):
    """The same strategy, the same frame, a different fourth feature.

    Measured on BTC/USDT perp 4h over R5's test half, trained cell: +16.44% at
    Sharpe +0.801 crowding-neutral against the published +15.45% at +0.896. The
    payload has to say which of those it is showing.
    """
    payload = build_analysis(
        _PERP,
        strategy_name="state_machine_v1",
        exit_mode=ExitMode.OPPOSITE_SIGNAL_ONLY,
        funding=False,
    )

    assert payload.provenance.funding_attached is False
    assert payload.provenance.crowding_measured is False


def test_a_strategy_that_reads_no_funding_says_so_without_claiming_a_measurement(
    perp_frame,
):
    """``donchian`` on a funded perp: the column is on the frame and nothing read
    it. Both facts are reported, so neither is inferred from the other."""
    payload = build_analysis(_PERP, strategy_name="donchian")

    assert payload.provenance.funding_attached is True
    assert payload.provenance.crowding_measured is False


def test_the_default_exit_mode_is_named_rather_than_left_blank(perp_frame):
    """A payload that omitted the mode would be a number computed under settings
    nobody chose, which is the failure the whole provenance block exists for."""
    payload = build_analysis(_PERP, strategy_name="donchian")

    assert payload.provenance.exit_mode == ExitMode.CONTINUATION_FAILURE.value
    assert payload.provenance.failure_bars == 4


# --------------------------------------------------------------------------
# The two contracts.
# --------------------------------------------------------------------------


def test_the_continuous_contract_returns_a_signed_level_rather_than_markers(perp_frame):
    """``BaselineSeries`` draws a signed target against a zero baseline; markers
    cannot draw one at all, which is why the dispatch is by registry."""
    payload = build_analysis(_PERP, strategy_name="state_machine_v2")

    assert payload.provenance.contract == Contract.TARGET_EXPOSURE.value
    assert payload.markers == []
    assert payload.target is not None
    assert len(payload.target) == _MACHINE_BARS
    assert max(payload.target) > 0.0 and min(payload.target) < 0.0
    assert all(abs(value) <= 1.0 for value in payload.target)
    # Warmup is a leading run of 0.0 and never None -- the inverse of the feature
    # convention, because a target says what to hold and before convergence that
    # is exactly nothing. None would reach the page as "not measurable" and be
    # drawn as a gap in the baseline rather than as flat.
    warmup = payload.target[: payload.provenance.warmup_bars]
    assert warmup and all(value == 0.0 for value in warmup)
    assert all(value is not None for value in payload.target)


def test_the_continuous_contract_reports_no_cost_model_because_it_executed_none(
    perp_frame,
):
    """The target is what the strategy asked for, not what a book did with it.
    Reporting a fee rate beside it would claim an execution that never ran."""
    payload = build_analysis(_PERP, strategy_name="state_machine_v2")

    assert payload.provenance.cost_model is None
    assert payload.provenance.exit_mode is None
    assert payload.provenance.failure_bars is None


def test_an_exit_mode_on_the_continuous_path_is_refused_rather_than_ignored(perp_frame):
    """There is no ``ExitMode`` on that contract -- a target of 0.0 *is* the exit
    -- so accepting one and dropping it would show a chart labelled with a
    setting that changed nothing."""
    with pytest.raises(ValueError, match="no exit mode"):
        build_analysis(
            _PERP,
            strategy_name="state_machine_v2",
            exit_mode=ExitMode.OPPOSITE_SIGNAL_ONLY,
        )


def test_entry_sizes_ride_along_where_a_strategy_provides_them(perp_frame):
    """``state_machine_v1``'s per-state target risk is the size at entry, and it
    is the only place that risk is visible on the boolean contract."""
    payload = build_analysis(
        _PERP, strategy_name="state_machine_v1", exit_mode=ExitMode.OPPOSITE_SIGNAL_ONLY
    )

    assert payload.position_size is not None
    assert len(payload.position_size) == _MACHINE_BARS
    assert any(value > 0.0 for value in payload.position_size)


def test_a_strategy_that_sizes_nothing_returns_no_size_series(spot_frame):
    assert build_analysis(_SPOT, strategy_name="donchian").position_size is None


# --------------------------------------------------------------------------
# The "why" layer.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("strategy_name", ["state_machine_v1", "state_machine_v2"])
def test_the_why_layer_carries_the_state_and_the_features_behind_it(
    strategy_name, perp_frame
):
    """Computed on every bar by both paths today and thrown away by both.
    Returning it is the whole point of the tool, and it needs no storage."""
    payload = build_analysis(
        _PERP, strategy_name=strategy_name, exit_mode=None
    )

    assert payload.why is not None
    assert set(payload.why.features) == {"direction", "strength", "stability", "crowding"}
    assert len(payload.why.states) == _MACHINE_BARS
    for values in payload.why.features.values():
        assert len(values) == _MACHINE_BARS
    # Warmup rows are NaN by the feature convention, and JSON has no NaN.
    assert payload.why.features["direction"][0] is None
    assert set(payload.why.states) <= {
        "compression", "breakout", "confirmed", "riding", "exhaustion", "reset",
    }


def test_a_strategy_with_no_state_to_explain_gets_no_why_layer(spot_frame):
    """Introspection rather than a hardcoded list of state machines: a strategy
    that grows a ``feature_frame`` later is covered without editing this file."""
    assert build_analysis(_SPOT, strategy_name="donchian").why is None


# --------------------------------------------------------------------------
# The registry surface.
# --------------------------------------------------------------------------


def test_every_registered_strategy_is_listed_and_labelled_by_contract():
    """Both registries, because a caller choosing a strategy has to know which
    of the two contracts it will be answered on before it asks."""
    from strategy_lab.strategies.exposure_registry import list_exposure_strategies
    from strategy_lab.strategies.registry import list_strategies

    listed = registered_strategies()
    by_name = {entry.name: entry for entry in listed}

    assert set(by_name) == set(list_strategies()) | set(list_exposure_strategies())
    assert by_name["donchian"].contract == Contract.SIGNAL_SET.value
    assert by_name["state_machine_v2"].contract == Contract.TARGET_EXPOSURE.value
    assert by_name["state_machine_v1"].warmup_bars == 2192
    assert all(entry.version for entry in listed)


def test_an_unknown_strategy_names_what_is_available():
    with pytest.raises(ValueError, match="donchian"):
        resolve_strategy("no_such_strategy")


def test_an_empty_frame_is_refused_rather_than_charted(monkeypatch):
    """An empty payload renders as a blank chart, which reads as "this strategy
    did nothing" rather than as "this dataset is not stored"."""
    empty = synthetic_ohlcv(n=5, freq="4h").iloc[:0]
    monkeypatch.setattr("strategy_lab.api.analysis.load_candles", lambda **kwargs: empty)

    with pytest.raises(ValueError, match="No candles"):
        build_analysis(_SPOT, strategy_name="donchian")
