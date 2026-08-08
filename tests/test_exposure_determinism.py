"""The determinism proof, for the continuous-exposure path.

``tests/test_replay_determinism.py`` proves the boolean contract's two drive
paths agree. It cannot say anything about this one: it iterates the boolean
registry and compares ``SignalSet`` fields, so a second contract without its own
proof is a second contract with no guarantee.

A continuous target is *more* exposed to that gap, not less. A boolean signal
one bar late changes when you enter; a target one bar late changes your size on
**every bar**, including all the bars where the boolean path would have emitted
nothing at all and had nothing to compare.

Three comparisons, and the three cheats below exist to show what each one
catches that the others do not:

1. **Whole history vs streaming.** Catches reading the future. Both runs start
   at bar 0, so a causal strategy passes by construction -- which is exactly
   why it is not sufficient on its own.
2. **A runner primed from mid-history.** Catches ``warmup_bars`` that is too
   small: the streamed run is handed exactly that many bars from the middle of
   history, where the whole-history run reaches the same bar carrying
   everything before it. This is the comparison PR #8 had to add after the
   state machine's convergence guarantee was broken for weeks while both
   from-bar-zero comparisons passed.
3. **Target-level equality on every bar**, not side-level. Catches a size that
   is wrong while the direction is right -- which is most of what a target
   *is*, and all of what a boolean comparison cannot see.

The strategies proved are every entry in ``strategies/exposure_registry.py``
plus two local tapers kept for shapes the registry does not currently contain.

The streaming driver is ``engine.exposure_runner.ExposureRunner`` itself, so
these three comparisons test the class a live process would run rather than a
local stand-in for it. It was a stand-in until R10e shipped the runner.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from strategy_lab.core.clock import SimClock
from strategy_lab.core.types import InstrumentId
from strategy_lab.engine.exposure_runner import ExposureRunner
from strategy_lab.feeds.base import Subscription
from strategy_lab.feeds.replay import ReplayFeed
from strategy_lab.strategies.exposure import TargetExposure
from strategy_lab.strategies.exposure_registry import (
    get_exposure_strategy,
    list_exposure_strategies,
)
from tests.conftest import synthetic_ohlcv

TIMEFRAME = "15m"
INSTRUMENT = InstrumentId("binance", "perp", "BTC/USDT")
SUB = Subscription(INSTRUMENT, TIMEFRAME)

# Bars actually compared, past each strategy's own warmup. Streaming is O(N^2) --
# every post-warmup bar re-runs compute_target over the whole buffer -- so this
# is the runtime knob, and it is bounded below by MIN_CHANGES rather than picked.
STREAM_SPAN = 400

# Bars discarded before the primed runner's warmup window starts, so the
# backtest it is compared against carries history the runner was never given.
# Zero would make this comparison identical to the first one and it would stop
# testing anything. Offset and seed are the boolean suite's measured pair.
PRIME_OFFSET = 1_500
PRIME_SEED = 3

# A target-level equality over a target that never moves is `0.0 == 0.0` on
# every bar: true, and proof of nothing. Every comparison below asserts the
# compared window actually contains this many changes first. It is the direct
# analogue of the boolean suite's MIN_SIGNALS, and it exists for the same
# reason -- that suite went vacuous once when a warmup grew past its frame.
MIN_CHANGES = 10

# The charter's per-state taper levels, which is what a target is expected to
# look like: a handful of discrete steps rather than a continuum.
LEVELS = np.array([-1.0, -0.55, -0.25, 0.0, 0.25, 0.55, 1.0])


def snap(values: np.ndarray) -> np.ndarray:
    return LEVELS[np.abs(LEVELS[None, :] - values[:, None]).argmin(axis=1)]


@dataclass(frozen=True)
class _EwmTaper:
    """Target from an ``ewm(adjust=False)`` momentum -- recursive from bar zero.

    The shape that makes comparison 2 bite: the recursion decays its seed rather
    than dropping it, so a cold start is wrong for roughly 20 spans and only a
    warmup that large makes the two paths agree.
    """

    span: int = 30
    name: str = "ewm_taper"
    version: str = "1.0.0"
    warmup_bars: int = 600

    def compute_target(self, df: pd.DataFrame) -> TargetExposure:
        trend = df["close"].ewm(span=self.span, adjust=False).mean()
        spread = df["close"].pct_change().ewm(span=self.span, adjust=False).std()
        momentum = ((df["close"] / trend - 1.0) / spread.replace(0.0, np.nan)).fillna(0.0)
        return TargetExposure(target=pd.Series(snap(np.tanh(momentum.to_numpy() / 6.0)), index=df.index))


@dataclass(frozen=True)
class _RollingTaper:
    """Target from a trailing window -- the window-safe half of the family."""

    window: int = 96
    name: str = "rolling_taper"
    version: str = "1.0.0"
    warmup_bars: int = 97

    def compute_target(self, df: pd.DataFrame) -> TargetExposure:
        from strategy_lab.features.base import rolling_percentile

        rank = rolling_percentile(df["close"].pct_change(), window=self.window)
        return TargetExposure(
            target=pd.Series(snap((rank.fillna(0.5) * 2.0 - 1.0).to_numpy()), index=df.index)
        )


# The two local tapers stay beside the registered ones rather than being
# replaced by them. They are chosen for their *shape*: ``_EwmTaper`` is
# recursive from bar zero, which is what makes comparison 2 bite, and
# ``_RollingTaper`` is exactly windowed, which is the half of the family that
# cannot fail it. A registry with one entry cannot promise to cover both.
HONEST = [
    *(get_exposure_strategy(name) for name in list_exposure_strategies()),
    _EwmTaper(),
    _RollingTaper(),
]


def frame_for(strategy, span: int = STREAM_SPAN, seed: int = 7) -> pd.DataFrame:
    return synthetic_ohlcv(n=strategy.warmup_bars + span, seed=seed)


def whole_history_targets(strategy, df: pd.DataFrame, *, first: int | None = None) -> pd.Series:
    """What a whole-history run would hold on each bar, past the warmup boundary.

    The boundary is the streaming runner's boundary exactly: it suppresses while
    ``len(buffer) <= warmup_bars``, so its first emitting bar is 0-based position
    ``warmup_bars``. One bar out here would shift the whole series and the
    equality would compare bar *t* against bar *t+1*.
    """
    boundary = strategy.warmup_bars if first is None else first
    return strategy.compute_target(df).target.iloc[boundary:]


def streamed_targets(
    strategy,
    df: pd.DataFrame,
    *,
    prime: pd.DataFrame | None = None,
    instrument: InstrumentId = INSTRUMENT,
    timeframe: str = TIMEFRAME,
) -> pd.Series:
    """What the same strategy holds when the same bars arrive one at a time.

    **This drives the production ``ExposureRunner``**, which is what this file's
    header asked for while there was none. The local driver it replaces was a
    deliberate mirror -- real ``ReplayFeed``, real ``BarBuffer``, the strategy
    over the full buffer, the last row read -- and a mirror proves the mirror.
    Every comparison below now tests the class a live process would run.

    ``rebalance_threshold=0.0`` is what turns emissions back into one value per
    post-warmup bar: the band submits whenever the move is at least the
    threshold, so zero submits always. That is the engine's own reading of a zero
    band (``exposure_engine._banded``), which is why it is the right knob rather
    than a way around the runner.

    The identity is a parameter so a caller on a different instrument or bar size
    gets bars labelled with its own -- ``scripts/r10d`` runs this on 4h perp
    frames. Measured: the label reaches ``Bar.timeframe`` and ``ts_close_ms`` and
    **nothing else**, since ``BarBuffer.frame()`` carries bar-open timestamps and
    OHLCV, so 400 bars of ``state_machine_v2`` gave an identical index and 0
    differing targets either way.
    """
    runner = ExposureRunner(
        strategy=strategy,
        instrument=instrument,
        timeframe=timeframe,
        clock=SimClock(),
        rebalance_threshold=0.0,
        record_reasons=False,
    )
    if prime is not None:
        runner.prime(prime)

    feed = ReplayFeed(frames={instrument.at(timeframe): df})
    sub = Subscription(instrument, timeframe)

    async def _run() -> pd.Series:
        held: dict[pd.Timestamp, float] = {}
        async for event in feed.stream([sub]):
            for emitted in runner.on_event(event):
                stamp = pd.Timestamp(emitted.ts_bar_ms, unit="ms", tz="UTC")
                held[stamp] = float(emitted.target_exposure)
        return pd.Series(held, dtype="float64")

    return asyncio.run(_run())


def assert_same_target(streamed: pd.Series, expected: pd.Series, *, label: str) -> None:
    """Every bar, to exact float equality, over a window that actually moves.

    ``check_exact`` is not optional. ``assert_series_equal`` defaults to a 1e-5
    relative tolerance on floats, which on a target in -1..1 would wave through
    a 0.55 held where 0.55000001 belonged -- and, worse, every difference
    smaller than that anywhere in the series. The whole value of this equality
    is that it is exact.

    ``check_freq=False`` because the streamed index is built from arriving
    timestamps and carries no inferred frequency, which is an attribute of the
    index object rather than a difference in the data.
    """
    changes = int((expected.diff().fillna(0.0) != 0).sum())
    assert changes >= MIN_CHANGES, (
        f"{label}: the compared window holds only {changes} target changes over "
        f"{len(expected)} bars, so the equality would be near-vacuous. Raise "
        f"STREAM_SPAN or use a strategy that moves."
    )
    pd.testing.assert_series_equal(
        streamed, expected, check_exact=True, check_freq=False, check_names=False
    )


@pytest.mark.parametrize("strategy", HONEST, ids=lambda s: s.name)
def test_streaming_reproduces_the_whole_history_target_on_every_bar(strategy):
    df = frame_for(strategy)
    assert_same_target(
        streamed_targets(strategy, df),
        whole_history_targets(strategy, df),
        label=strategy.name,
    )


@pytest.mark.parametrize("strategy", HONEST, ids=lambda s: s.name)
def test_a_primed_runner_reproduces_the_target_it_started_late_for(strategy):
    """The drive path a live process actually is, and the only one where *where
    you started* can show. Handed exactly ``warmup_bars`` bars from the middle
    of history, the runner has to reach the same target as a backtest that
    carried the previous 1,500 bars into the same instant."""
    df = synthetic_ohlcv(
        n=PRIME_OFFSET + strategy.warmup_bars + STREAM_SPAN, seed=PRIME_SEED
    )
    first_streamed = PRIME_OFFSET + strategy.warmup_bars

    assert_same_target(
        streamed_targets(
            strategy,
            df.iloc[first_streamed:],
            prime=df.iloc[PRIME_OFFSET:first_streamed],
        ),
        whole_history_targets(strategy, df, first=first_streamed),
        label=f"{strategy.name} primed at {PRIME_OFFSET}",
    )


# The three cheats, one per comparison.


@dataclass(frozen=True)
class _FutureReader:
    """Holds a full position when the NEXT bar closes higher -- shift(-1)."""

    name: str = "future_reader"
    version: str = "1.0.0"
    warmup_bars: int = 10

    def compute_target(self, df: pd.DataFrame) -> TargetExposure:
        ahead = (df["close"].shift(-1) > df["close"]).astype("float64")
        return TargetExposure(target=ahead)


@dataclass(frozen=True)
class _RightSideWrongSize:
    """Direction from a causal rule; *size* normalized by the full-sample maximum.

    Nothing here reads a future price to decide long or short. It reads the
    whole sample to decide how much, which is the failure a side-level
    comparison is structurally unable to see.
    """

    name: str = "right_side_wrong_size"
    version: str = "1.0.0"
    warmup_bars: int = 10

    def compute_target(self, df: pd.DataFrame) -> TargetExposure:
        returns = df["close"].pct_change().fillna(0.0)
        side = np.where(returns.to_numpy() >= 0, 1.0, -1.0)
        magnitude = 0.5 + 0.5 * returns.abs() / returns.abs().max()
        return TargetExposure(target=pd.Series(side * magnitude.to_numpy(), index=df.index))


@dataclass(frozen=True)
class _UnderDeclaredWarmup(_EwmTaper):
    """Perfectly causal, and honest about nothing else: 5 bars of declared warmup.

    Row *t* of a run over ``[0, N]`` and row *t* of a run over ``[0, t]`` are the
    same computation for this strategy, so it passes comparison 1 by
    construction. It is wrong anyway, because ``ewm(adjust=False)`` carries its
    seed for ~20 spans and a runner primed with 5 bars is holding a size derived
    from a number that has not converged.
    """

    name: str = "under_declared_warmup"
    warmup_bars: int = 5


def test_the_streaming_check_can_fail():
    """A non-causal exposure strategy must break the equality -- otherwise this
    module proves nothing."""
    cheat = _FutureReader()
    df = frame_for(cheat, span=200)
    expected = whole_history_targets(cheat, df)

    assert expected.abs().sum() > 0, "the cheat must take a position at least once"
    assert not streamed_targets(cheat, df).equals(expected)


def test_a_side_level_comparison_would_miss_a_wrong_size():
    """Comparison 3 earns its place here, and only here.

    Every bar's *direction* agrees between the two paths, so a boolean-style
    check on entry timestamps or on the sign of the position would pass this
    strategy. Every bar's *size* is wrong, because it is scaled by a maximum
    drawn from bars that have not happened.
    """
    cheat = _RightSideWrongSize()
    df = frame_for(cheat, span=200)
    expected = whole_history_targets(cheat, df)
    streamed = streamed_targets(cheat, df)

    assert np.array_equal(np.sign(streamed.to_numpy()), np.sign(expected.to_numpy()))
    assert not streamed.equals(expected)
    assert (streamed - expected).abs().max() > 0.01


def test_a_primed_runner_catches_a_warmup_that_streaming_from_bar_zero_cannot():
    """Comparison 2 earns its place: the same strategy passes 1 and fails 2.

    This is PR #8's lesson as an executable claim rather than a caution. Both
    from-bar-zero comparisons are blind to an under-declared warmup by
    construction, so a suite made only of those answers a narrower question than
    its name suggests.
    """
    cheat = _UnderDeclaredWarmup()
    df = synthetic_ohlcv(n=PRIME_OFFSET + cheat.warmup_bars + STREAM_SPAN, seed=PRIME_SEED)
    first_streamed = PRIME_OFFSET + cheat.warmup_bars

    from_bar_zero = frame_for(cheat, span=STREAM_SPAN)
    assert streamed_targets(cheat, from_bar_zero).equals(
        whole_history_targets(cheat, from_bar_zero)
    ), "the under-declared strategy is causal, so comparison 1 must pass it"

    primed = streamed_targets(
        cheat, df.iloc[first_streamed:], prime=df.iloc[PRIME_OFFSET:first_streamed]
    )
    assert not primed.equals(whole_history_targets(cheat, df, first=first_streamed))
