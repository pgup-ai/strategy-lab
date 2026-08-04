"""The phase's exit criterion: one strategy, two drive paths, identical signals.

A backtest calls ``generate_signals`` once over the whole history and reads every
row. Replay and live call it per bar against an expanding buffer and read only
the last row. Those agree only if the strategy is causal -- row *t* of its output
depends on rows <= *t* of its input and nothing later. Every strategy here is
assumed causal, the backtest keeps its bulk path because it is ~6,600x faster end
to end (the measurement is in ``StrategyRunner``'s docstring), and this module is
the check that the assumption actually holds.

A failure here is a bug in the engine or the strategy, never a reason to relax
the comparison: the whole point of the equality is that it is exact.
"""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from strategy_lab.core.clock import SimClock
from strategy_lab.core.types import InstrumentId, Side
from strategy_lab.engine import runner as runner_module
from strategy_lab.engine.runner import StrategyRunner
from strategy_lab.feeds.base import Subscription
from strategy_lab.feeds.replay import ReplayFeed
from strategy_lab.strategies.registry import get_strategy, list_strategies
from tests.conftest import synthetic_ohlcv

INSTRUMENT = InstrumentId("binance", "perp", "BTC/USDT")
SUB = Subscription(INSTRUMENT, "15m")

# Frames are sized RELATIVE TO each strategy's warmup, never as a fixed length.
# The runner emits nothing until bar warmup_bars+1, so a fixed 600-bar frame
# yields zero signals for any strategy declaring more warmup than that -- and
# every equality in this module would then be a true-but-empty `[] == []`. That
# is not hypothetical: it is what a fixed 600 did to the turnaround strategies
# the moment their warmup went to 4000.
#
# The span is the number of bars actually compared, and it is the runtime knob:
# streaming is O(N^2), because the runner re-runs generate_signals over the whole
# buffer once per post-warmup bar. Measured signal counts / streaming cost:
#   span 1000 -> v1 498, v2 40, trend_following 47, trend_rider 1258   (9.7s total)
#   span  400 -> v1 212, v2 12, trend_following 12, trend_rider  500   (3.5s total)
# turnaround_v2 is the sparse one and therefore sets the floor.
STREAM_SPAN = 1_000
REPEAT_SPAN = 400

# Every comparison below must have real signals behind it. A floor rather than
# ">0" so the tests cannot quietly decay to a single-signal comparison either.
MIN_SIGNALS = 10


def frame_for(strategy, span: int) -> pd.DataFrame:
    return synthetic_ohlcv(n=strategy.warmup_bars + span)


# Declared independently of the runner so this stays an oracle rather than a
# mirror -- if the runner relabelled or reordered its sides, an imported constant
# would relabel the expectation with it and the comparison would prove nothing.
# ``test_reference_side_order_matches_the_runner`` pins the two together instead.
_SIDE_BY_FIELD = (
    ("long_entries", Side.ENTER_LONG),
    ("long_exits", Side.EXIT_LONG),
    ("short_entries", Side.ENTER_SHORT),
    ("short_exits", Side.EXIT_SHORT),
)


def vectorized_signals(strategy, df: pd.DataFrame) -> list[tuple[int, Side]]:
    """What a whole-history backtest would produce, skipping warmup.

    The warmup boundary must be the runner's boundary exactly: the runner
    suppresses while ``len(buffer) <= warmup_bars``, so its first emitting bar is
    0-based position ``warmup_bars``. Skipping one bar more or less here would
    shift the whole list and the equality would be off by one signal.

    Timestamps use ``Timestamp.value // 1_000_000`` -- the same integer path as
    ``feeds.replay._epoch_ms``, which is what stamps ``Signal.ts_bar_ms``.
    ``int(ts.timestamp() * 1000)`` lands 1 ms low on ~0.7% of sub-second
    timestamps; candle opens are whole seconds so it would agree today, but the
    comparison must not depend on that holding.
    """
    signal_set = strategy.generate_signals(df)
    out: list[tuple[int, Side]] = []
    for position, timestamp in enumerate(df.index):
        if position < strategy.warmup_bars:
            continue
        ts_ms = timestamp.value // 1_000_000
        for field_name, side in _SIDE_BY_FIELD:
            if bool(getattr(signal_set, field_name).iloc[position]):
                out.append((ts_ms, side))
    return out


def streamed_signals(strategy, df: pd.DataFrame) -> list[tuple[int, Side]]:
    """What the event-driven runner produces from the same bars."""
    feed = ReplayFeed(frames={INSTRUMENT.at("15m"): df})
    runner = StrategyRunner(
        strategy=strategy, instrument=INSTRUMENT, timeframe="15m", clock=SimClock()
    )

    async def _run() -> list[tuple[int, Side]]:
        collected: list[tuple[int, Side]] = []
        async for event in feed.stream([SUB]):
            for signal in runner.on_event(event):
                collected.append((signal.ts_bar_ms, signal.side))
        return collected

    return asyncio.run(_run())


def test_reference_side_order_matches_the_runner():
    """List equality compares emission order, so the oracle's order is load-bearing.

    A bar can fire several sides at once (turnaround_v1 wires
    ``long_exits = short_entries``), and the runner emits them in
    ``_SIDE_BY_FIELD`` order. If that order or those labels changed, the
    per-bar comparisons above would silently be comparing different things.
    """
    assert _SIDE_BY_FIELD == runner_module._SIDE_BY_FIELD


@pytest.mark.parametrize("name", list_strategies())
def test_streaming_reproduces_vectorized_signals_exactly(name):
    strategy = get_strategy(name)
    df = frame_for(strategy, STREAM_SPAN)

    expected = vectorized_signals(strategy, df)
    actual = streamed_signals(strategy, df)

    assert len(expected) >= MIN_SIGNALS, (
        f"{name}: only {len(expected)} signals past warmup_bars="
        f"{strategy.warmup_bars} in {len(df)} bars -- the equality below would be "
        f"vacuous. Raise STREAM_SPAN."
    )
    assert actual == expected, (
        f"{name}: streaming and vectorized paths diverged. "
        f"{len(expected)} expected vs {len(actual)} actual signals."
    )


@pytest.mark.parametrize("name", list_strategies())
def test_replay_is_repeatable(name):
    """Same input, same signals -- twice. Fresh runner, fresh feed, no shared state."""
    strategy = get_strategy(name)
    df = frame_for(strategy, REPEAT_SPAN)
    first = streamed_signals(strategy, df)
    assert len(first) >= MIN_SIGNALS, (
        f"{name}: only {len(first)} signals emitted past warmup_bars="
        f"{strategy.warmup_bars}; repeatability would be vacuous. Raise REPEAT_SPAN."
    )
    assert streamed_signals(strategy, df) == first


def test_the_determinism_check_can_fail():
    """A non-causal strategy must break the equality -- otherwise this proves nothing."""
    from dataclasses import dataclass

    from strategy_lab.strategies.base import SignalSet, validate_ohlcv

    @dataclass(frozen=True)
    class _Cheat:
        name: str = "cheat"
        version: str = "1.0.0"
        warmup_bars: int = 5

        def generate_signals(self, df: pd.DataFrame) -> SignalSet:
            validate_ohlcv(df)
            longs = (df["close"].shift(-1) > df["close"]).fillna(False)
            flat = pd.Series(False, index=df.index)
            return SignalSet(longs, flat, flat, flat)

    strategy = _Cheat()
    df = synthetic_ohlcv(n=200)
    assert vectorized_signals(strategy, df), "the cheat must fire at least once"
    assert streamed_signals(strategy, df) != vectorized_signals(strategy, df)


@pytest.mark.db
def test_streaming_matches_vectorized_on_real_stored_candles():
    """The real thing: BTC/USDT 15m candles already in Postgres.

    ``turnaround_v1`` rather than ``v2`` on purpose. Both are the ewm family whose
    output depends on every prior bar, which is the case worth checking against
    real data -- but v2 fires only 126 times in the whole 83,348-bar history, so
    on a short window this test would assert ``[] == []`` and pass no matter how
    broken the engine was. v1 fires ~1,400 times per 3,000 bars, including bars
    that emit an exit and a reversal together.

    ``limit_bars`` is warmup-relative for the same reason the synthetic frames
    are: the runner suppresses the first ``warmup_bars`` of whatever it is given,
    so a fixed 3,000 leaves nothing to compare once warmup is 4,000.
    """
    strategy = get_strategy("turnaround_v1")
    instrument = InstrumentId("binance", "spot", "BTC/USDT")
    limit_bars = strategy.warmup_bars + 1_000
    feed = ReplayFeed.from_database([Subscription(instrument, "15m")], limit_bars=limit_bars)
    df = feed.frames[instrument.at("15m")]
    if df.empty:
        pytest.skip("no stored BTC/USDT 15m candles; run fetch-crypto first")

    # ``limit_bars`` takes the tail, so both paths start mid-history with no prior
    # context -- fine, because they are handed the identical truncated frame and
    # the ewm recursion therefore starts from the identical first bar. What is not
    # automatically fine is ordering: the feed reorders and de-duplicates as it
    # streams, the vectorized path reads the frame as-is, so a frame that was not
    # already sorted and unique would make the two see different data.
    assert df.index.is_monotonic_increasing and df.index.is_unique

    runner = StrategyRunner(
        strategy=strategy, instrument=instrument, timeframe="15m", clock=SimClock()
    )

    async def _run() -> list[tuple[int, Side]]:
        collected: list[tuple[int, Side]] = []
        async for event in feed.stream([Subscription(instrument, "15m")]):
            for signal in runner.on_event(event):
                collected.append((signal.ts_bar_ms, signal.side))
        return collected

    expected = vectorized_signals(strategy, df)
    assert len(expected) >= MIN_SIGNALS, (
        f"only {len(expected)} signals from {len(df)} real bars past warmup_bars="
        f"{strategy.warmup_bars}; the equality would be vacuous"
    )
    assert asyncio.run(_run()) == expected
