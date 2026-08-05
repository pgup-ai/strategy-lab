"""Execution, costs and funding for the continuous-exposure path.

The frames here are flat or a clean ramp rather than ``synthetic_ohlcv``'s random
walk, on purpose: ``size_type="targetpercent"`` is a fraction of *equity*, so a
moving price makes the book rebalance on bars where the target did not change.
That is real behaviour and one test below measures it, but every other assertion
here is about what the *target* asked for, and a random walk would mix the two.
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


def test_the_position_tracks_the_target():
    """A taper that is not executed is a state machine whose behaviour is ignored."""
    result = run(_Scripted(), position_pct=1.0)
    assert result.position_fraction.to_numpy() == pytest.approx(list(TAPER), abs=0.01)


def test_position_pct_scales_the_target_rather_than_replacing_it():
    """Target 1.0 means the whole risk budget; the budget itself is position_pct."""
    result = run(_Scripted(), position_pct=0.5)
    assert result.position_fraction.to_numpy() == pytest.approx(
        [level * 0.5 for level in TAPER], abs=0.01
    )


def test_a_target_crossing_zero_reverses_rather_than_flattening():
    levels = (0.0, 1.0, 0.5, 0.0, -0.5, -1.0, -0.4, 0.0)
    result = run(_Scripted(levels=levels), df=flat_frame(len(levels)), position_pct=1.0)

    assert result.position.max() > 0
    assert result.position.min() < 0
    assert result.position_fraction.to_numpy() == pytest.approx(list(levels), abs=0.01)
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


def test_a_constant_target_still_rebalances_as_the_price_moves():
    """A fraction of equity is not a quantity: holding it steady means trading.

    This is the drift turnover ``targetpercent`` carries, and it belongs to the
    contract rather than to any strategy's taper. A comparison of a continuous
    strategy against a boolean one that does not name it will attribute this
    cost to the taper.
    """
    df = ramp_frame(6)
    result = run(_Constant(level=0.5), df=df, position_pct=1.0)

    assert result.order_count == 6
    assert result.position_fraction.to_numpy() == pytest.approx([0.5] * 6)
    assert result.position.is_monotonic_decreasing


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
    assert config["cost_model"] == {"fee": 0.0, "slippage": 0.0}
    assert config["candle_count"] == 40


def test_a_run_without_funding_charges_none():
    result = run(_Constant(level=1.0), df=flat_frame(40))
    assert result.funding_paid == 0.0
    assert result.config["funding_applied"] is False
