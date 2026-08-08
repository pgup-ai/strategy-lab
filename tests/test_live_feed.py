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
from strategy_lab.feeds.replay import ReplayFeed, _row_to_bar
from strategy_lab.features.flow import FUNDING_COLUMN
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.strategies.registry import get_strategy
from strategy_lab.timeframes import timeframe_to_millis
from tests.conftest import synthetic_ohlcv, synthetic_ohlcv_with_funding

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

    def __call__(self, identity, since_ms: int, until_ms: int | None = None) -> pd.DataFrame:
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
    fields = ("open", "high", "low", "close", "volume")
    for got, want in zip(live, stored, strict=True):
        assert got.bar.ts_open_ms == want.bar.ts_open_ms
        assert got.bar.is_closed == want.bar.is_closed
        # Every field, because "identical bars" is what the gate claims and a
        # defect in `high` or `volume` would pass a close-only comparison.
        assert [getattr(got.bar, f) for f in fields] == [getattr(want.bar, f) for f in fields]


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

    # A bounded historical request: every row in it closed long ago, so the
    # clock says so and none is withheld as forming.
    feed = LiveFeed(
        fetch=lambda identity, since, until=None: long_frame.iloc[:split],
        sleep=_noop,
        clock=SimClock(2**62),
    )
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

    assert len(bars) == split, "a bounded backfill dropped a closed bar"
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
    from_frame.prime(long_frame.iloc[:split])

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


def test_a_correction_to_the_newest_bar_reaches_the_caller(frame):
    """`lookback_bars` exists to re-read bars a venue may still correct, so
    suppressing the correction would make it decorative. The protocol forbids
    yielding *the same bar* twice; a bar whose values changed is a different bar
    under the same timestamp, and `BarBuffer` lands it last-wins."""
    venue = _Venue(frame, first=10)
    feed = LiveFeed(fetch=venue, sleep=_noop)
    first = drain(feed, [SUB], polls=2)

    # The newest bar the caller holds — the one `BarBuffer` can still replace.
    newest = feed._cursors[SUB.candle]
    at = frame.index.get_loc(pd.Timestamp(newest, unit="ms", tz="UTC"))
    corrected = frame.copy()
    corrected.iloc[at, corrected.columns.get_loc("close")] *= 1.5
    venue.frame = corrected
    again = drain(feed, [SUB], polls=3)

    assert any(e.bar.ts_open_ms == newest for e in first)
    assert any(e.bar.ts_open_ms == newest for e in again), "the correction was suppressed"


def test_a_correction_the_buffer_would_drop_is_not_re_emitted(frame):
    """The bound on the above. `BarBuffer.append` replaces on a repeated
    timestamp but drops anything *older* than its last as out-of-order —
    measured, a correction four bars back increments `dropped_out_of_order` and
    changes nothing. Emitting it would bill every consumer for an event none of
    them can apply."""
    venue = _Venue(frame, first=10)
    feed = LiveFeed(fetch=venue, sleep=_noop)
    drain(feed, [SUB], polls=2)

    newest = pd.Timestamp(feed._cursors[SUB.candle], unit="ms", tz="UTC")
    at = frame.index.get_loc(newest) - 4
    corrected = frame.copy()
    corrected.iloc[at, corrected.columns.get_loc("close")] *= 1.5
    venue.frame = corrected
    again = drain(feed, [SUB], polls=3)

    stale = int(frame.index[at].value // 10**6)
    assert again, "the venue was never re-polled, so nothing was under test"
    assert not any(e.bar.ts_open_ms == stale for e in again), (
        "a correction the buffer would drop as out-of-order was re-emitted anyway"
    )


def test_an_unchanged_re_read_is_not_re_emitted(frame):
    """The other half: re-reading the same window must not replay bars that did
    not move, or every poll would re-emit its whole lookback."""
    feed = LiveFeed(fetch=_Venue(frame, first=10, step=0), sleep=_noop)

    events = drain(feed, [SUB], polls=4)
    stamps = [e.bar.ts_open_ms for e in events]

    assert len(stamps) == len(set(stamps))


def test_one_subscription_does_not_drag_anothers_fetch_window():
    """Cursors are per `CandleId`: BTC 4h and BTC 1d advance at different rates,
    and a shared cursor would let the faster one pull the slower one's window past
    bars it had not seen.

    Asserted against the *lagging* subscription's own cursor rather than by
    comparing the two windows — two timeframes give different windows either way,
    so that comparison cannot see a shared cursor.
    """
    slow = Subscription(INSTRUMENT, "1d")
    bar_4h, bar_1d = 14_400_000, 86_400_000
    feed = LiveFeed(fetch=_Venue(pd.DataFrame(), first=0), sleep=_noop)
    feed._cursors = {SUB.candle: 5_000_000_000, slow.candle: 1_000_000_000}

    assert feed._since_ms(slow, bar_1d) == 1_000_000_000 - bar_1d * feed.lookback_bars
    assert feed._since_ms(SUB, bar_4h) == 5_000_000_000 - bar_4h * feed.lookback_bars


def test_the_real_fetch_attaches_a_perps_funding(monkeypatch):
    """The fix that matters most on this path, and the one every other test here
    cannot see because they all inject `fetch`.

    A venue's OHLCV endpoint returns no funding, so a live bar built straight from
    it carries `funding_rate=None`, the buffer materialises no column, and
    `crowding` falls back to neutral — M20 again, undoing on the live path what
    R10f closed on the replayed one.
    """
    from strategy_lab.feeds import live as live_module

    funded = synthetic_ohlcv_with_funding(n=30, freq=TIMEFRAME)
    settlements = funded[FUNDING_COLUMN][funded[FUNDING_COLUMN] != 0.0]
    bare = funded.drop(columns=[FUNDING_COLUMN])

    class _Client:
        def __init__(self, **kwargs):
            pass

        def fetch_ohlcv(self, *args, **kwargs):
            return bare

    monkeypatch.setattr("strategy_lab.market_data.binance.CryptoOhlcvClient", _Client)
    monkeypatch.setattr(
        "strategy_lab.backtests.funding_frame.funding_rates",
        lambda identity, frame, **_: settlements,
    )

    got = live_module._fetch_recent(
        MarketDataIdentity(
            exchange="binance", market_type="perp", symbol="BTC/USDT", timeframe=TIMEFRAME
        ),
        0,
    )

    assert FUNDING_COLUMN in got.columns, "a live perp bar would run crowding-neutral"
    # And it reaches the bar, which is what BarBuffer reads to decide whether the
    # column exists at all.
    bar = _row_to_bar(got.index[0], got.iloc[0], INSTRUMENT, TIMEFRAME, 14_400_000)
    assert bar.funding_rate is not None


def test_a_non_positive_or_non_finite_cadence_is_refused():
    """An operator sets this by hand: zero busy-loops, and a NaN compares false
    against every bound so it would silently mean "as fast as fetch returns"."""
    # `+inf` is the one a `> 0` guard lets through, and a feed that sleeps
    # forever has stopped without saying so.
    for bad in (0, -1.0, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="poll_seconds"):
            LiveFeed(poll_seconds=bad)


def test_a_bounded_backfill_asks_the_venue_for_the_bound():
    """Without `until` the client paginates forward to the present, so a narrow
    window far in the past would fetch years and discard nearly all of it."""
    asked: list[tuple[int, int | None]] = []

    def _fetch(identity, since_ms, until_ms=None):
        asked.append((since_ms, until_ms))
        return pd.DataFrame()

    feed = LiveFeed(fetch=_fetch, sleep=_noop)

    async def _run():
        return [bar async for bar in feed.backfill(SUB, 1_000, 2_000)]

    asyncio.run(_run())
    assert asked == [(1_000, 2_000)]


def test_a_bar_carrying_a_gap_is_still_deduplicated(frame):
    """`NaN != NaN`, so a NaN left in the dedup key would never match its own
    earlier entry: the bar would re-emit on every poll and `_seen` would grow
    without bound — a gap in the data turning into a leak and a signal storm."""
    holed = frame.copy()
    holed.iloc[2, holed.columns.get_loc("volume")] = float("nan")
    feed = LiveFeed(fetch=_Venue(holed, first=10, step=0), sleep=_noop)

    events = drain(feed, [SUB], polls=4)
    stamps = [e.bar.ts_open_ms for e in events]

    assert len(stamps) == len(set(stamps))


def test_a_cold_start_asks_for_a_lookback_not_for_1970():
    """Without a cursor the window used to open at the epoch, so the client would
    paginate from 1970 to the present before yielding a single event. History is
    `backfill`'s job; a first poll wants the same lookback every later one does."""
    now = 1_700_000_000_000
    bar_ms = 14_400_000
    feed = LiveFeed(fetch=_Venue(pd.DataFrame(), first=0), sleep=_noop, clock=SimClock(now))

    assert feed._since_ms(SUB, bar_ms) == now - bar_ms * feed.lookback_bars


def test_a_far_future_bound_means_no_bound_rather_than_a_crash(monkeypatch):
    """`backfill(sub, start, 2**62)` is how a cold start says "up to now", and
    `pd.Timestamp` raises `OutOfBoundsDatetime` past 2262 — so the sentinel has to
    reach the client as "no upper bound"."""
    from strategy_lab.feeds import live as live_module

    seen: list[str | None] = []

    class _Client:
        def __init__(self, **kwargs):
            pass

        def fetch_ohlcv(self, *args, until=None, **kwargs):
            seen.append(until)
            return pd.DataFrame()

    monkeypatch.setattr("strategy_lab.market_data.binance.CryptoOhlcvClient", _Client)
    monkeypatch.setattr(
        "strategy_lab.backtests.funding_frame.with_funding_column",
        lambda identity, frame, **kwargs: (frame, None),
    )
    identity = MarketDataIdentity(
        exchange="binance", market_type="perp", symbol="BTC/USDT", timeframe=TIMEFRAME
    )

    live_module._fetch_recent(identity, 0, 2**62)
    live_module._fetch_recent(identity, 0, 1_700_000_000_000)

    assert seen[0] is None, "a far-future sentinel should mean no bound"
    assert seen[1] is not None, "a real bound must still reach the client"


def test_a_subscription_holds_one_answer_about_funding_for_its_whole_life():
    """`with_funding_column` decides per *window*, and a feed asks about two.
    Measured before this: priming from full BTC history (which spans the venue's
    permanent ~40h head gap, so no column) and then polling the recent tail (no
    gap, so a column) crashed on the first poll, because a stream that changes its
    mind about funding is exactly what `BarBuffer` refuses (M42).

    The buffer's guard is right and stays. The feed is what must be consistent.
    """
    funded = synthetic_ohlcv_with_funding(n=60, freq=TIMEFRAME)
    head = 50

    def fetch(identity, since_ms, until_ms=None):
        if until_ms is not None:  # `backfill`; a poll passes no bound
            return funded.iloc[:head].drop(columns=[FUNDING_COLUMN])
        return funded.iloc[head - 3 :]  # the recent window, funded

    feed = LiveFeed(fetch=fetch, sleep=_noop, clock=SimClock(10**13))
    runner = StrategyRunner(
        strategy=get_strategy("donchian"),
        instrument=INSTRUMENT,
        timeframe=TIMEFRAME,
        clock=SimClock(),
        record_reasons=False,
    )

    async def _history():
        return [bar async for bar in feed.backfill(SUB, 0, 2**62)]

    runner.prime_bars(asyncio.run(_history()))
    assert not runner.buffer.carries_funding

    for event in feed._poll(SUB):
        runner.on_event(event)  # must not raise

    assert len(runner.buffer) > head, "the poll never reached the buffer"
    assert not runner.buffer.carries_funding, "the stream changed its mind after all"


def test_losing_funding_mid_stream_is_refused_where_the_cause_can_be_named():
    """The reverse of the above is not conformed away: earlier bars ran with
    funding, so continuing without it would run a different strategy than they
    did."""
    funded = synthetic_ohlcv_with_funding(n=20, freq=TIMEFRAME)
    calls = {"n": 0}

    def fetch(identity, since_ms, until_ms=None):
        calls["n"] += 1
        return funded if calls["n"] == 1 else funded.drop(columns=[FUNDING_COLUMN])

    feed = LiveFeed(fetch=fetch, sleep=_noop, clock=SimClock(10**13))
    feed._poll(SUB)

    with pytest.raises(ValueError, match="stopped carrying funding"):
        feed._poll(SUB)
