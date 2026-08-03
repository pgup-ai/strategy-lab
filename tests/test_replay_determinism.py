"""The phase's exit criterion: one strategy, two drive paths, identical signals.

A backtest calls ``generate_signals`` once over the whole history and reads every
row. Replay and live call it per bar against an expanding buffer and read only
the last row. Those agree only if the strategy is causal -- row *t* of its output
depends on rows <= *t* of its input and nothing later. Every strategy here is
assumed causal, the backtest keeps its bulk path because it is ~60,000x faster,
and this module is the check that the assumption actually holds.

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
            series = getattr(signal_set, field_name, None)
            if series is not None and bool(series.iloc[position]):
                out.append((ts_ms, side))
    return out


def streamed_signals(strategy, df: pd.DataFrame) -> list[tuple[int, Side]]:
    """What the event-driven runner produces from the same bars."""
    feed = ReplayFeed(frames={(INSTRUMENT, "15m"): df})
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
    df = synthetic_ohlcv(n=600)

    expected = vectorized_signals(strategy, df)
    actual = streamed_signals(strategy, df)

    assert expected, f"{name}: no signals to compare -- the equality would be vacuous"
    assert actual == expected, (
        f"{name}: streaming and vectorized paths diverged. "
        f"{len(expected)} expected vs {len(actual)} actual signals."
    )


@pytest.mark.parametrize("name", list_strategies())
def test_replay_is_repeatable(name):
    """Same input, same signals -- twice. Fresh runner, fresh feed, no shared state."""
    strategy = get_strategy(name)
    df = synthetic_ohlcv(n=400)
    first = streamed_signals(strategy, df)
    assert first, f"{name}: no signals emitted, repeatability would be vacuous"
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
    real data -- but on the last 3,000 real 15m bars v2 fires exactly zero times
    (it fires 126 times in the whole 83,348-bar history), so this test would have
    asserted ``[] == []`` and passed no matter how broken the engine was. v1 fires
    1,432 times over the same window, including bars that emit an exit and a
    reversal together.
    """
    strategy = get_strategy("turnaround_v1")
    instrument = InstrumentId("binance", "spot", "BTC/USDT")
    feed = ReplayFeed.from_database([Subscription(instrument, "15m")], limit_bars=3_000)
    df = feed.frames[(instrument, "15m")]
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
    assert expected, "expected at least one signal from 3,000 real bars"
    assert asyncio.run(_run()) == expected
