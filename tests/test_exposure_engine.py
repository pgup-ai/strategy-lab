"""Execution, costs and funding for the continuous-exposure path.

The frames here are flat or a clean ramp rather than ``synthetic_ohlcv``'s random
walk, on purpose. A flat frame holds equity still, so the book's fraction of it
is the target and nothing else; a ramp is the smallest frame on which the two
come apart, because between decisions the book keeps its *quantity* and its
fraction drifts with the price. Both behaviours are asserted below, and a random
walk would mix them into one number that neither test could name.

The engine tracks a target **at its decision bars** and drifts between them, so
assertions about tracking are scoped to ``rebalance_target.notna()``. An
assertion over every bar would be asserting that the band does not work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from strategy_lab.backtests.costs import CostModel
from strategy_lab.backtests.exposure_engine import run_exposure_backtest
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.strategies.exposure import TargetExposure

IDENTITY = MarketDataIdentity(
    exchange="binance", market_type="perp", symbol="BTC/USDT", timeframe="4h"
)
COSTLESS = CostModel(fee=0.0, slippage=0.0)
CASH = 10_000.0

# The taper the phase exists to execute: in, held, tapered out. Six changes, so
# six orders -- against the one order from_signals gives for the same sequence.
TAPER = (0.0, 0.3, 0.7, 1.0, 1.0, 1.0, 0.55, 0.55, 0.2, 0.0)


def flat_frame(n: int, price: float = 100.0, freq: str = "4h") -> pd.DataFrame:
    """A frame whose price never moves, so every order comes from the target."""
    index = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame(
        {"open": price, "high": price, "low": price, "close": price, "volume": 1_000.0},
        index=index,
    )


def ramp_frame(n: int, start: float = 100.0, end: float = 200.0) -> pd.DataFrame:
    prices = np.linspace(start, end, n)
    index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices, "volume": 1_000.0},
        index=index,
    )


def funding_series(df: pd.DataFrame, rate: float = 0.0001) -> pd.Series:
    """8h settlements across the frame, as the venue stamps them."""
    index = pd.date_range(df.index[0], df.index[-1], freq="8h", tz="UTC", name="timestamp")
    return pd.Series(rate, index=index, dtype="float64")


@dataclass(frozen=True)
class _Scripted:
    """Replays a fixed target, so the engine is the only thing under test."""

    levels: tuple[float, ...] = TAPER
    name: str = "scripted_exposure"
    version: str = "1.0.0"
    warmup_bars: int = 0

    def compute_target(self, df: pd.DataFrame) -> TargetExposure:
        values = np.resize(np.asarray(self.levels, dtype="float64"), len(df))
        return TargetExposure(target=pd.Series(values, index=df.index))


@dataclass(frozen=True)
class _Constant:
    """Holds one level from the first bar -- the control for the taper."""

    level: float = 1.0
    name: str = "constant_exposure"
    version: str = "1.0.0"
    warmup_bars: int = 0
    metadata: dict = field(default_factory=dict)

    def compute_target(self, df: pd.DataFrame) -> TargetExposure:
        return TargetExposure(
            target=pd.Series(self.level, index=df.index, dtype="float64"),
            metadata=self.metadata,
        )


def run(strategy, df=None, **kwargs):
    return run_exposure_backtest(
        df=flat_frame(len(TAPER)) if df is None else df,
        strategy=strategy,
        identity=IDENTITY,
        cash=CASH,
        cost_model=kwargs.pop("cost_model", COSTLESS),
        **kwargs,
    )


def assert_tracks_at_decisions(result, df, *, position_pct: float) -> None:
    """The book holds what the target asked for, on the bars it was asked.

    In *currency* against initial cash rather than as a fraction of equity: that
    is what ``targetvalue`` at the repo's non-compounding anchor means, and it is
    the claim that stays true on a frame where equity moves.
    """
    decisions = result.rebalance_target.notna()
    assert decisions.any(), "no decision bars, so this asserts nothing"
    held_value = (result.position * df["close"])[decisions]
    asked = (result.target * position_pct * CASH)[decisions]
    assert held_value.to_numpy() == pytest.approx(asked.to_numpy(), abs=1e-9)


def test_a_drifting_target_produces_an_order_per_change():
    """The whole point: from_signals would give one order for this."""
    result = run(_Scripted())
    assert result.order_count == 6


def test_the_same_taper_through_the_boolean_path_produces_one_order():
    """The measurement that made this contract necessary, re-run in place.

    ``from_signals`` defaults to ``accumulate=False``: it fills on the bar a
    position opens and ignores every later size, so the taper's six changes
    collapse into a single fill. Nothing about the taper is expressible there --
    which is why this is a second contract rather than a flag on the first one.
    """
    import vectorbt as vbt

    df = flat_frame(len(TAPER))
    size = pd.Series(TAPER, index=df.index) * CASH * 0.95 / df["close"]
    pf = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=pd.Series(True, index=df.index),
        exits=pd.Series(False, index=df.index),
        size=size,
        init_cash=CASH,
        freq="4h",
    )
    assert len(pf.orders.records_readable) == 1


def test_the_position_tracks_the_target_at_every_decision_bar():
    """A taper that is not executed is a state machine whose behaviour is ignored.

    On a ramp, so that "tracks the target" has to mean *at the bars it decided
    on*: TAPER holds its level on four of ten bars, and on a moving price those
    four are exactly where the book is allowed to drift away. The same assertion
    over every bar would fail here, and would be asserting that the band does not
    work.
    """
    df = ramp_frame(len(TAPER))
    result = run(_Scripted(), df=df, position_pct=1.0)

    assert_tracks_at_decisions(result, df, position_pct=1.0)
    # Every level the taper asked for reached the book, in order, once each.
    assert result.rebalance_target.dropna().tolist() == [0.3, 0.7, 1.0, 0.55, 0.2, 0.0]
    assert result.order_count == 6


def test_position_pct_scales_the_target_rather_than_replacing_it():
    """Target 1.0 means the whole risk budget; the budget itself is position_pct.

    Flat frame, so equity never moves and the drift the band exists for is zero
    -- which is what makes an every-bar fraction meaningful here and nowhere
    else.
    """
    result = run(_Scripted(), position_pct=0.5)
    assert result.position_fraction.to_numpy() == pytest.approx(
        [level * 0.5 for level in TAPER], abs=0.01
    )


def test_a_target_crossing_zero_reverses_rather_than_flattening():
    levels = (0.0, 1.0, 0.5, 0.0, -0.5, -1.0, -0.4, 0.0)
    df = flat_frame(len(levels))
    result = run(_Scripted(levels=levels), df=df, position_pct=1.0)

    assert result.position.max() > 0
    assert result.position.min() < 0
    assert_tracks_at_decisions(result, df, position_pct=1.0)
    assert result.order_count == 7


def test_funding_is_charged_on_the_held_fraction_not_a_full_unit():
    """A 20% position pays 20% of the carry -- the taper's whole economic point.

    R2 measured BTC perp funding at +11.65%/yr paid by longs, so this factor is
    not a rounding detail: a tapered book charged full-unit carry looks worse
    than it is, and one charged none looks better.
    """
    df = flat_frame(40)
    rates = funding_series(df)
    full = run(_Constant(level=1.0), df=df, funding=rates)
    fifth = run(_Constant(level=0.2), df=df, funding=rates)

    assert full.funding_paid > 0
    assert fifth.funding_paid == pytest.approx(0.2 * full.funding_paid, rel=1e-9)


def test_a_short_receives_funding_that_a_long_pays():
    df = flat_frame(40)
    rates = funding_series(df)
    long = run(_Constant(level=1.0), df=df, funding=rates)
    short = run(_Constant(level=-1.0), df=df, funding=rates)

    assert short.funding_paid == pytest.approx(-long.funding_paid, rel=1e-9)


def test_the_equity_curve_is_net_of_funding():
    """Funding settles outside the simulation, so it appears in one place only."""
    df = flat_frame(40)
    result = run(_Constant(level=1.0), df=df, funding=funding_series(df))
    gross = result.equity - result.funding_flow.cumsum()

    assert gross.iloc[-1] == pytest.approx(CASH)
    assert result.equity.iloc[-1] == pytest.approx(CASH - result.funding_paid)
    assert result.funding_paid > 0


def test_costs_are_charged_on_every_resize_not_only_on_entry():
    """Tapering is not free; a model that ignores resize cost flatters it."""
    model = CostModel(fee=0.001, slippage=0.0)
    tapered = run(_Scripted(), cost_model=model)
    held = run(_Constant(level=1.0), cost_model=model)

    orders = tapered.orders
    entry_notional = float(orders["Size"].iloc[0] * orders["Price"].iloc[0])
    after_entry = orders.iloc[1:]

    # Every fill is charged, not merely the first: the total is the fee rate
    # against the whole traded notional.
    assert tapered.fees_paid == pytest.approx(
        float((orders["Size"] * orders["Price"]).sum()) * model.fee
    )
    assert float((after_entry["Size"] * after_entry["Price"]).sum()) * model.fee > 0
    # Measured 6.7x: an engine that charged the entry alone would land on 1.0x.
    assert tapered.fees_paid > 2 * entry_notional * model.fee
    # The control holds one level throughout, so the excess is the taper's own
    # resizing rather than anything about how a position is opened.
    assert tapered.fees_paid > held.fees_paid


def test_slippage_is_backed_out_of_the_fills_it_moved():
    """vectorbt folds slippage into the price, so an unreported one reads as zero."""
    result = run(_Scripted(), cost_model=CostModel(fee=0.001, slippage=0.001))
    assert result.slippage_paid == pytest.approx(result.fees_paid, rel=0.01)


def test_the_book_drifts_between_decisions_instead_of_tracking_every_bar():
    """The drift is the band's purpose, so it is asserted by direction.

    A test that merely *tolerated* drift would also pass an engine that never
    traded at all, so this pins both halves: across the held run the position is
    one fixed quantity whose share of a rising book **strictly rises**, and the
    decision at the end of it still moves the book to what was asked.

    A quantity that rises in value is what "trend riding" means mechanically. An
    engine that rebalanced here would be selling that winner down on every bar,
    which is why the drift is the contract rather than an error in it.
    """
    df = ramp_frame(10)
    result = run(_Scripted(levels=(0.5,) * 8 + (0.2,) * 2), df=df, position_pct=1.0)

    held = slice(1, 8)  # after the opening decision, before the taper's
    assert result.rebalance_target.notna().to_numpy().tolist() == [
        True, False, False, False, False, False, False, False, True, False
    ]
    assert result.position.iloc[held].nunique() == 1
    fraction = result.position_fraction.iloc[held]
    assert (fraction.diff().dropna() > 0).all()
    assert fraction.iloc[-1] > fraction.iloc[0]

    # ... and the band is not "never trade": the decision at bar 8 lands.
    assert result.position.iloc[8] < result.position.iloc[7]
    assert_tracks_at_decisions(result, df, position_pct=1.0)


def test_the_target_is_sized_from_initial_cash_not_from_the_equity_it_grew_to():
    """CLAUDE.md's non-compounding rule, restated for the continuous path.

    ``targetpercent`` would size from current equity, so a run that had made
    money would hold a larger notional for the same target -- and a continuous
    strategy compared against a boolean one would be measuring that sizing change
    as well as whatever it meant to measure.
    """
    df = ramp_frame(10)
    result = run(_Scripted(levels=(0.5,) * 8 + (0.8,) * 2), df=df, position_pct=1.0)

    grown = float(result.equity.iloc[8])
    assert grown > CASH * 1.2, "the ramp must have made money for this to distinguish"
    assert float(result.position.iloc[8] * df["close"].iloc[8]) == pytest.approx(0.8 * CASH)


def test_band_zero_trades_every_bar_and_fades_the_trend_it_is_riding():
    """What ``rebalance_threshold=0.0`` actually is, named rather than assumed.

    Rebalancing an unchanged target to a fixed size means selling as the price
    rises and buying as it falls -- a mean-reversion overlay, on a book whose
    strategy is trying to ride a trend. It is a usable setting and a deliberate
    one; it is not the neutral choice its number makes it look.
    """
    df = ramp_frame(6)
    result = run(_Constant(level=0.5), df=df, position_pct=1.0, rebalance_threshold=0.0)

    assert result.order_count == 6
    assert (result.position * df["close"]).to_numpy() == pytest.approx([0.5 * CASH] * 6)
    assert result.position.is_monotonic_decreasing
    # The same target under the default band is one decision and no drift trades.
    assert run(_Constant(level=0.5), df=df, position_pct=1.0).order_count == 1


def test_a_slow_taper_crosses_the_band_by_accumulating_against_the_last_decision():
    """The band's reference is the last target *submitted*, not the last one seen.

    This taper gives up 2% of the budget per bar, so **no single bar** moves it
    as far as the 5% band -- a band measured bar to bar would hold the whole
    position from the first decision to the last bar and never execute the taper
    at all. Measured against the last decision the moves accumulate, so it
    executes late rather than not at all, and "late by up to one band" is the
    whole of what a band costs.
    """
    levels = tuple(np.round(1.0 - 0.02 * np.arange(21), 2))
    df = flat_frame(len(levels))
    result = run(_Scripted(levels=levels), df=df, position_pct=1.0)

    assert result.target.diff().abs().max() < 0.05, "no single bar may cross the band"
    decisions = result.rebalance_target.dropna()
    assert 1 < len(decisions) < len(levels)
    assert (decisions.diff().dropna().abs() >= 0.05).all()

    final = float(result.position.iloc[-1] * df["close"].iloc[-1])
    assert final == pytest.approx(levels[-1] * CASH, abs=0.05 * CASH)
    assert final < 0.95 * CASH


def test_a_negative_rebalance_threshold_is_refused():
    """It would read as "wider than no band" while being satisfied by every bar."""
    with pytest.raises(ValueError, match="rebalance_threshold"):
        run(_Constant(level=1.0), rebalance_threshold=-0.01)


def test_the_engine_holds_a_strategy_to_its_declared_warmup():
    """A target that trades before its indicators converge is masked, not trusted."""
    result = run(_Constant(level=1.0, warmup_bars=4), df=flat_frame(10))

    assert result.target.iloc[:4].tolist() == [0.0] * 4
    assert result.position.iloc[:4].tolist() == [0.0] * 4
    assert result.position.iloc[4] > 0


def test_a_frame_that_is_entirely_warmup_is_refused():
    with pytest.raises(ValueError, match="warmup"):
        run(_Constant(warmup_bars=10), df=flat_frame(10))


def test_a_target_on_a_different_index_from_the_candles_is_refused():
    """Size is executed positionally against close, so a shifted index trades
    the right sizes on the wrong bars and reports a number either way."""

    @dataclass(frozen=True)
    class _Misaligned:
        name: str = "misaligned"
        version: str = "1.0.0"
        warmup_bars: int = 0

        def compute_target(self, df: pd.DataFrame) -> TargetExposure:
            shifted = df.index + pd.Timedelta("1h")
            return TargetExposure(target=pd.Series(0.5, index=shifted, dtype="float64"))

    with pytest.raises(ValueError, match="different index"):
        run(_Misaligned())


def test_an_empty_frame_is_refused():
    with pytest.raises(ValueError, match="No candles"):
        run(_Constant(), df=flat_frame(0))


def test_the_config_records_what_the_run_was():
    result = run(
        _Constant(metadata={"states": "riding"}),
        df=flat_frame(40),
        funding=funding_series(flat_frame(40)),
        position_pct=0.5,
    )
    config = result.config

    assert config["contract"] == "target_exposure"
    assert config["strategy"] == "constant_exposure"
    assert config["strategy_metadata"] == {"states": "riding"}
    assert config["funding_applied"] is True
    assert config["position_pct"] == 0.5
    assert config["rebalance_threshold"] == 0.05
    assert config["cost_model"] == {"fee": 0.0, "slippage": 0.0}
    assert config["candle_count"] == 40


def test_a_run_without_funding_charges_none():
    result = run(_Constant(level=1.0), df=flat_frame(40))
    assert result.funding_paid == 0.0
    assert result.config["funding_applied"] is False
