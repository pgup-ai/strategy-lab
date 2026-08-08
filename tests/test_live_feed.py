"""The live feed, checked against the only oracle it has: a delayed one.

A live feed cannot be diffed against anything at the moment it runs — that is why
the book was built first, against a frozen `trades.csv`. But a live bar and a
stored bar for the same interval are the same fact recorded twice, so the gate is
that **a live window equals a replay of the same window**: identical bars, and
identical signals from the runner driven by each.

Every check here runs offline. `LiveFeed` takes its fetch and its sleep as
parameters, so a scripted venue and a sleep that does not sleep drive the whole
class — a feed whose tests need a network is a feed whose tests do not run.
"""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from strategy_lab.core.clock import SimClock
from strategy_lab.core.types import InstrumentId, Side
from strategy_lab.engine.context import BarBuffer
from strategy_lab.engine.runner import StrategyRunner
from strategy_lab.feeds.base import Subscription
from strategy_lab.feeds.live import (
    MAX_POLL_SECONDS,
    MIN_POLL_SECONDS,
    LiveFeed,
)
from strategy_lab.feeds.replay import ReplayFeed
from strategy_lab.strategies.registry import get_strategy
from strategy_lab.timeframes import timeframe_to_millis
from tests.conftest import synthetic_ohlcv

TIMEFRAME = "4h"
INSTRUMENT = InstrumentId("binance", "perp", "BTC/USDT")
SUB = Subscription(INSTRUMENT, TIMEFRAME)
FORMING_SUB = Subscription(INSTRUMENT, TIMEFRAME, include_forming=True)


class _Stopped(Exception):
    """The venue going away, which is how a test stops an endless stream.

    ``LiveFeed.stream`` does not terminate -- the protocol says "until exhausted
    (replay) or cancelled (live)" -- so a bound has to come from outside it.
    Stopping the *venue* rather than giving the feed a poll counter keeps test
    scaffolding out of the production class, and it is also the more faithful
    shape: a live process ends because something external ended it.
    """


class _Venue:
    """A scripted exchange: hands back a growing prefix of one frame per poll.

    That is what a real poll looks like — the same window re-read, one bar longer
    each time, with the newest row still forming.

    **The forming row is perturbed, and that is load-bearing.** A venue's
    in-progress bar has a different close from the one it eventually settles at;
    a fixture that returned the final values from the first poll would make
    "superseded by the closed bar" untestable, because the two copies would be
    identical. Measured: with an unperturbed fixture, a first-wins mutant in
    ``BarBuffer.append`` survives.
    """

    def __init__(self, frame: pd.DataFrame, *, first: int, step: int = 1) -> None:
        self.frame = frame
        self.rows = first
        self.step = step
        self.calls: list[int] = []
        self.stop_after: int | None = None

    def __call__(self, identity, since_ms: int) -> pd.DataFrame:
        if self.stop_after is not None and len(self.calls) >= self.stop_after:
            raise _Stopped
        self.calls.append(since_ms)
        window = self.frame.iloc[: self.rows].copy()
        drift = 1.0 + 0.01 * len(self.calls)
        window.iloc[-1, window.columns.get_loc("close")] *= drift
        window.iloc[-1, window.columns.get_loc("high")] *= drift
        self.rows = min(self.rows + self.step, len(self.frame))
        return window


async def _noop(_seconds: float) -> None:
    return None


def drain(feed: LiveFeed, subs, *, polls: int) -> list:
    """Everything the feed yields over `polls` polls of the venue.

    Bounded by the venue rather than by the event count, because a poll that
    yields nothing new is still a poll -- and that is the steady state a live
    process spends nearly all of its time in.
    """
    feed.fetch.stop_after = polls

    async def _run():
        events = []
        try:
            async for event in feed.stream(subs):
                events.append(event)
        except _Stopped:
            pass
        return events

    return asyncio.run(_run())


@pytest.fixture
def frame() -> pd.DataFrame:
    return synthetic_ohlcv(n=40, freq=TIMEFRAME)


# --------------------------------------------------------------------------
# The two guarantees the protocol states and `isinstance` cannot check.
# --------------------------------------------------------------------------


def test_bars_ascend_and_no_interval_is_yielded_twice(frame):
    """A poll re-reads the same window every time, so both guarantees have to be
    *maintained* here rather than inherited from an exhausted iterator."""
    feed = LiveFeed(fetch=_Venue(frame, first=10), sleep=_noop)

    events = drain(feed, [SUB], polls=8)
    stamps = [event.bar.ts_open_ms for event in events]

    assert stamps == sorted(stamps)
    keys = [(e.bar.instrument.key, e.bar.timeframe, e.bar.ts_open_ms, e.bar.is_closed)
            for e in events]
    assert len(keys) == len(set(keys))


def test_the_newest_row_is_withheld_until_it_closes(frame):
    """Without `include_forming` a caller sees an interval only once the venue can
    no longer change it, so the last row of any poll is never emitted as closed on
    that poll."""
    venue = _Venue(frame, first=10)
    feed = LiveFeed(fetch=venue, sleep=_noop)

    events = drain(feed, [SUB], polls=1)

    assert all(event.bar.is_closed for event in events)
    # 10 rows fetched, the newest still forming, so 9 reach the caller.
    assert len(events) == 9


def test_a_forming_bar_is_superseded_rather_than_duplicated(frame):
    """The same interval arrives forming and then closed. `BarBuffer` replaces
    last-wins on a repeated timestamp, so one bar survives, carrying the closed
    values — and `replaced_duplicates` counts it rather than hiding it."""
    feed = LiveFeed(fetch=_Venue(frame, first=10), sleep=_noop)
    events = drain(feed, [FORMING_SUB], polls=3)

    forming = [e for e in events if not e.bar.is_closed]
    assert forming, "include_forming yielded no forming bar"

    buffer = BarBuffer()
    for event in events:
        buffer.append(event.bar)

    stamps = [e.bar.ts_open_ms for e in events]
    assert len(buffer) == len(set(stamps))
    assert buffer.replaced_duplicates == len(stamps) - len(set(stamps)) > 0

    # The surviving copy is the closed one, not the partial it replaced. Checked
    # on an interval that really arrived both ways, or this asserts nothing.
    forming_stamps = {e.bar.ts_open_ms for e in events if not e.bar.is_closed}
    closed_by_stamp = {e.bar.ts_open_ms: e.bar for e in events if e.bar.is_closed}
    superseded = sorted(forming_stamps & closed_by_stamp.keys())
    assert superseded, "no interval arrived forming and then closed"
    held = buffer.frame()
    for stamp in superseded:
        row = held.loc[pd.Timestamp(stamp, unit="ms", tz="UTC")]
        assert float(row["close"]) == pytest.approx(float(closed_by_stamp[stamp].close))


def test_include_forming_off_never_yields_an_open_bar(frame):
    feed = LiveFeed(fetch=_Venue(frame, first=10), sleep=_noop)

    events = drain(feed, [SUB], polls=5)

    assert events
    assert not any(not event.bar.is_closed for event in events)


# --------------------------------------------------------------------------
# The gate: a live window equals a replay of the same window.
# --------------------------------------------------------------------------


def test_a_live_window_yields_the_bars_a_replay_of_it_yields(frame):
    """The delayed oracle, at bar level. What the venue served live and what
    storage serves afterwards are the same fact recorded twice."""
    feed = LiveFeed(fetch=_Venue(frame, first=5), sleep=_noop)
    live = drain(feed, [SUB], polls=40)

    replay = ReplayFeed(frames={INSTRUMENT.at(TIMEFRAME): frame})

    async def _replayed():
        return [event async for event in replay.stream([SUB])]

    stored = asyncio.run(_replayed())
    # The live path cannot have emitted the still-forming final bar.
    stored = stored[: len(live)]

    assert len(live) > 10
    for got, want in zip(live, stored, strict=True):
        assert got.bar.ts_open_ms == want.bar.ts_open_ms
        assert got.bar.close == want.bar.close
        assert got.bar.is_closed == want.bar.is_closed


def test_a_live_window_yields_the_signals_a_replay_of_it_yields(frame):
    """The same oracle one level up, which is what the gate actually claims: the
    runner cannot tell the two feeds apart."""
    strategy = get_strategy("donchian")
    long_frame = synthetic_ohlcv(n=strategy.warmup_bars + 200, freq=TIMEFRAME)

    def _signals(events) -> list[tuple[int, Side]]:
        runner = StrategyRunner(
            strategy=strategy,
            instrument=INSTRUMENT,
            timeframe=TIMEFRAME,
            clock=SimClock(),
            record_reasons=False,
        )
        emitted = []
        for event in events:
            emitted.extend((s.ts_bar_ms, s.side) for s in runner.on_event(event))
        return emitted

    feed = LiveFeed(fetch=_Venue(long_frame, first=5, step=8), sleep=_noop)
    live = drain(feed, [SUB], polls=200)

    replay = ReplayFeed(frames={INSTRUMENT.at(TIMEFRAME): long_frame})

    async def _replayed():
        return [event async for event in replay.stream([SUB])]

    stored = asyncio.run(_replayed())[: len(live)]

    live_signals, stored_signals = _signals(live), _signals(stored)
    assert len(live_signals) > 10, "no signals emitted; this would compare nothing"
    assert live_signals == stored_signals


def test_a_cold_start_primes_from_backfill_and_reaches_the_replays_state(frame):
    """`backfill()` yields `Bar` and `prime_bars()` takes `Bar`, which before R10
    they did not — so a live process can warm itself from history rather than
    waiting out a 2,192-bar warmup in real time."""
    strategy = get_strategy("donchian")
    long_frame = synthetic_ohlcv(n=strategy.warmup_bars + 200, freq=TIMEFRAME)
    split = strategy.warmup_bars + 100

    feed = LiveFeed(fetch=lambda identity, since: long_frame.iloc[:split], sleep=_noop)
    warmed = StrategyRunner(
        strategy=strategy,
        instrument=INSTRUMENT,
        timeframe=TIMEFRAME,
        clock=SimClock(),
        record_reasons=False,
    )
    async def _backfilled():
        return [bar async for bar in feed.backfill(SUB, 0, 2**62)]

    bars = asyncio.run(_backfilled())

    assert len(bars) == split - 1, "backfill withheld something other than the forming bar"
    warmed.prime_bars(bars)
    assert len(warmed.buffer) == len(bars)

    # The same history through the DataFrame door has to land identically, or the
    # two priming paths disagree about what a cold start saw.
    from_frame = StrategyRunner(
        strategy=strategy,
        instrument=INSTRUMENT,
        timeframe=TIMEFRAME,
        clock=SimClock(),
        record_reasons=False,
    )
    from_frame.prime(long_frame.iloc[: split - 1])

    pd.testing.assert_frame_equal(warmed.buffer.frame(), from_frame.buffer.frame())


def test_the_poll_cadence_is_a_fraction_of_the_bar_rather_than_the_bar():
    """A feed polling once per bar sees a bar that closed at *T* only at the next
    poll — up to a whole bar late, which on 4h is four hours. Latency is bounded
    by the gap, so the gap is what decides whether "live" means anything."""
    feed = LiveFeed()
    subs = {tf: Subscription(INSTRUMENT, tf) for tf in ("15m", "1h", "4h", "1w")}

    for timeframe, sub in subs.items():
        gap = feed._interval([sub])
        bar_seconds = timeframe_to_millis(timeframe) / 1000.0
        assert gap < bar_seconds, timeframe
        assert MIN_POLL_SECONDS <= gap <= MAX_POLL_SECONDS, timeframe

    # The smallest bar in a mixed set decides, since it closes first.
    assert feed._interval(list(subs.values())) == feed._interval([subs["15m"]])


def test_an_explicit_cadence_overrides_the_derived_one():
    """The bounds above are judgements, not measurements against a rate limit —
    so an operator who has one says so."""
    assert LiveFeed(poll_seconds=1.5)._interval([Subscription(INSTRUMENT, "4h")]) == 1.5
