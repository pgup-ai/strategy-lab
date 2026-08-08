"""A live feed that polls, and the reasoning for why it polls.

``MarketDataFeed`` has had one implementation since Phase 1a and it reads from
Postgres. This is the second, and it is the one a paper process runs.

**It polls rather than opening a socket, and that is a decision.** The protocol
does not care where a ``BarEvent`` came from -- that is what having a protocol
buys. The venue REST client already exists and is tested, where a socket adds a
dependency, a reconnect state machine and a framing layer, all untestable without
a network. And at the timeframes this program trades, 4h and 1w, the difference
is not observable: ``browse`` already makes the same argument for its own
refresh. A socket is an optimisation for a bar size nothing here trades, and it
can land behind this same protocol later without touching the runner, the book,
or any strategy.

**The two guarantees the protocol states and cannot enforce are this class's
problem.** Bars ascend, and no ``(instrument, timeframe, ts_open_ms,
is_closed)`` is ever yielded twice. A poll re-reads the same window every time,
so both have to be *maintained* here rather than inherited -- ``_seen`` is what
does it, and it is keyed on the full quadruple because a forming bar and the
closed bar for the same interval are two legitimate events, not a duplicate.

**A forming bar is a different claim from a closed one**, which is why
``Subscription.include_forming`` finally means something. Off, the feed waits:
the caller sees an interval only once the venue can no longer change it. On, the
forming bar arrives as ``is_closed=False`` and is later **superseded** by the
closed bar for the same interval -- ``BarBuffer`` already replaces last-wins on a
repeated timestamp, so the buffer ends up with one bar carrying the closed
values, and ``replaced_duplicates`` counts it.
"""

from __future__ import annotations

import asyncio
import math
import warnings
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field, replace

import pandas as pd

from strategy_lab.core.clock import Clock, LiveClock
from strategy_lab.core.types import Bar, BarEvent, CandleId
from strategy_lab.feeds.base import FeedHealth, Subscription
from strategy_lab.feeds.replay import _epoch_ms, _ordered, _row_to_bar
from strategy_lab.features.flow import FUNDING_COLUMN
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.timeframes import timeframe_to_millis

# Bars re-read on every poll. More than one because a venue can correct the bar
# it just closed, and a feed that only ever asked for the newest would never see
# the correction -- the same reason `server.refresh_candles` reaches back rather
# than fetching the tip.
DEFAULT_LOOKBACK_BARS = 5

# Beyond this ``pd.Timestamp`` raises rather than saturating; a bound past it is
# a sentinel for "no bound" rather than a date anyone meant.
_MAX_TIMESTAMP_MS = pd.Timestamp.max.value // 1_000_000

# Settlements a perp poll's window must contain before it can be asked about
# coverage. ``funding_coverage_gaps`` certifies the cadence rather than assuming
# 8h, so it needs the interval observed twice -- three settlements -- and this
# carries one spare, since the newest may fall past the window's right edge.
# Measured on BTC/USDT perp 4h: a 5-bar window is 16h and holds two, so no
# funding column comes back at all; 8 bars holds five and is covered.
MIN_SETTLEMENTS_IN_WINDOW = 4

# The default cadence, as a fraction of the bar being polled. See ``_interval``.
POLLS_PER_BAR = 60
MIN_POLL_SECONDS = 5.0
MAX_POLL_SECONDS = 300.0


def _fetch_recent(
    identity: MarketDataIdentity, since_ms: int, until_ms: int | None = None
) -> pd.DataFrame:
    """The venue's recent candles, with a perp's stored funding attached.

    **The funding half is not optional.** The venue's OHLCV endpoint returns no
    funding, so a live bar built straight from it carries ``funding_rate=None``,
    ``BarBuffer`` materialises no column, and ``crowding`` falls back to neutral --
    which is M20 exactly, and undoes on the live path what R10f closed on the
    replayed one. Measured before this was here: a replayed perp bar carried a
    rate and a live one carried ``None``.

    Attached through ``with_funding_column`` -- the engine's own function, the
    same one ``ReplayFeed.from_database`` uses -- so the alignment rule is shared
    rather than duplicated. ``required=False`` because a live process must not
    die at the right edge: settlements land on the venue's own schedule and the
    newest bars routinely have none yet. Coverage falling short drops the column
    entirely rather than returning it partial, and what the feed does with that
    depends on whether the stream was ever funded -- see ``_conform_funding``.

    **A perp's window is widened until it can answer that question**, which is
    the reason this function and not ``_since_ms`` owns the reach-back: the
    lookback means "bars to re-read for corrections" and should keep meaning it,
    the widening belongs beside the call that needs it, and ``LiveFeed`` keeps
    the injection seam that makes its tests offline. Measured, a 5-bar window at
    4h is 16h and holds two settlements where the guard needs three, so **no
    poll at any timeframe this program trades funded itself** -- the process
    stalled on its first poll under the withholding rule, and ran ``crowding``
    neutral in silence before it. The extra bars are not waste the feed has to
    absorb: ``_seen`` already dedups a venue that returns more than it was asked
    for, which is why pruning it was rejected.
    """
    from strategy_lab.backtests.funding_frame import with_funding_column
    from strategy_lab.market_data.binance import CryptoOhlcvClient

    client = CryptoOhlcvClient(
        exchange_id=identity.exchange, market_type=identity.market_type
    )
    since = pd.Timestamp(_funded_since_ms(identity, since_ms), unit="ms", tz="UTC").isoformat()
    # ``until`` is passed through rather than filtered locally: without it the
    # client paginates forward to the present, so a bounded backfill over a narrow
    # window far in the past would fetch years and discard nearly all of it.
    # A caller's far-future sentinel means "no upper bound", not a crash:
    # ``pd.Timestamp`` raises ``OutOfBoundsDatetime`` past 2262, and
    # ``backfill(sub, start, 2**62)`` is exactly how a cold start says "up to now".
    until = None if until_ms is None or until_ms > _MAX_TIMESTAMP_MS else (
        pd.Timestamp(until_ms, unit="ms", tz="UTC").isoformat()
    )
    frame = client.fetch_ohlcv(identity.symbol, identity.timeframe, since=since, until=until)
    frame, _ = with_funding_column(identity, frame, enabled=True, required=False)
    return frame


def _funded_since_ms(identity: MarketDataIdentity, since_ms: int) -> int:
    """``since_ms``, reached back far enough for a perp to carry its own funding.

    The same rule ``server._fetch_funding`` states for the browser's refresh --
    "the earlier of the candle lookback and the last stored settlement" -- asked
    in settlements rather than hours, because the interval is per-contract.
    Unchanged off-perp, and unchanged when there are too few settlements stored
    to reach for: a window that cannot be covered is ``_conform_funding``'s
    problem, and widening toward funding that does not exist would only make the
    request slower before it got the same answer.
    """
    if identity.market_type != "perp":
        return since_ms

    from strategy_lab.db.funding import nth_newest_settlement

    floor = nth_newest_settlement(
        exchange=identity.exchange,
        market_type=identity.market_type,
        symbol=identity.symbol,
        count=MIN_SETTLEMENTS_IN_WINDOW,
    )
    if floor is None:
        return since_ms
    return min(since_ms, int(floor.value // 1_000_000))


@dataclass
class LiveFeed:
    """Polls a venue and yields bars, satisfying ``MarketDataFeed``.

    ``fetch`` and ``sleep`` are injected so the whole class is testable offline:
    every check in ``tests/test_live_feed.py`` drives it with a scripted fetch and
    a sleep that does not sleep. A feed whose tests need a network is a feed whose
    tests do not run.
    """

    name: str = "live"
    lookback_bars: int = DEFAULT_LOOKBACK_BARS
    poll_seconds: float | None = None
    fetch: Callable[..., pd.DataFrame] = _fetch_recent
    sleep: Callable[[float], object] = asyncio.sleep
    clock: Clock = field(default_factory=LiveClock)
    # Consecutive failed polls one subscription may take before the stream ends.
    fetch_retries: int = 3

    # Never pruned, deliberately. Pruning below the fetch watermark was tried and
    # reverted: it assumes the venue returns nothing older than it was asked for,
    # and a venue that over-returns would then re-emit pruned bars on every poll
    # -- a correctness property traded for a non-problem. Measured, a key is ~330
    # bytes: 0.6 MB/year at 4h and 9.5 MB/year at 15m, for a process meant to be
    # restarted rather than run forever.
    _seen: set[tuple[str, int, bool, tuple]] = field(default_factory=set, init=False)
    # Monotonic, so a stall that resolved itself is still visible afterwards.
    funding_withheld_polls: int = field(default=0, init=False)
    _funded: dict[CandleId, bool] = field(default_factory=dict, init=False, repr=False)
    _withholding: set[CandleId] = field(default_factory=set, init=False, repr=False)
    _window_floors: dict[CandleId, int] = field(default_factory=dict, init=False, repr=False)
    _cursors: dict[CandleId, int] = field(default_factory=dict, init=False, repr=False)
    _failures: dict[CandleId, int] = field(default_factory=dict, init=False, repr=False)
    _last_event_ms: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # A knob an operator sets by hand, unlike the values flowing through
        # `build_fill`: zero busy-loops, and a NaN compares false against every
        # bound so it would silently become "as fast as fetch returns".
        if self.poll_seconds is not None and not (
            math.isfinite(self.poll_seconds) and self.poll_seconds > 0
        ):
            # ``> 0`` alone lets ``inf`` through, and a feed that sleeps forever
            # has stopped without saying so -- the quietest way for a live
            # process to fail.
            raise ValueError(
                f"poll_seconds must be a finite positive number of seconds, "
                f"got {self.poll_seconds!r}"
            )

    async def stream(self, subs: Sequence[Subscription]) -> AsyncIterator[BarEvent]:
        """Poll each subscription in turn, forever, yielding what is new.

        Ordering is per-subscription rather than globally merged, unlike
        ``ReplayFeed.stream``: a live poll has no future to merge against, so
        holding one subscription's bar back to interleave it with another's would
        be inventing latency. ``MarketClock`` is what groups a live cross-section,
        exactly as it does a replayed one.

        It does not terminate. A caller stops it by breaking out of the
        iteration or cancelling the task, which is what ``until exhausted
        (replay) or cancelled (live)`` in the protocol means.
        """
        while True:
            for sub in subs:
                try:
                    events = self._poll(sub)
                except Exception as exc:
                    # Bounded and loud, not swallowed. A process meant to run for
                    # hours cannot end on one timed-out request, but a feed that
                    # retried forever would report itself healthy while producing
                    # nothing -- so the budget is per subscription, a success
                    # clears it, and spending it re-raises the original error.
                    failures = self._failures.get(sub.candle, 0) + 1
                    self._failures[sub.candle] = failures
                    if failures > self.fetch_retries:
                        raise
                    warnings.warn(
                        f"{sub.candle.key}: poll failed "
                        f"({failures}/{self.fetch_retries} before giving up): {exc}",
                        stacklevel=2,
                    )
                    continue
                self._failures.pop(sub.candle, None)
                for event in events:
                    self._last_event_ms = event.ts_event_ms
                    self._cursors[sub.candle] = max(
                        self._cursors.get(sub.candle, 0), event.bar.ts_open_ms
                    )
                    yield event
            await self.sleep(self._interval(subs))

    def _interval(self, subs: Sequence[Subscription]) -> float:
        """Seconds between polls, derived from the smallest bar in the set.

        **A fraction of the bar rather than the bar itself.** Polling once per bar
        would mean a bar that closed at *T* is not seen until the next poll, i.e.
        up to a whole bar late -- four hours, on the timeframe this program
        trades. Latency is bounded by the poll gap, so the gap is what decides
        whether "live" means anything.

        Clamped at both ends: the floor keeps a 15m subscription from hammering a
        venue, and the ceiling keeps a weekly one from being a day late. Neither
        bound is measured against a rate limit -- ``poll_seconds`` is the override
        for an operator who has one.
        """
        if self.poll_seconds is not None:
            return self.poll_seconds
        smallest = min(timeframe_to_millis(sub.timeframe) for sub in subs) / 1000.0
        return min(max(smallest / POLLS_PER_BAR, MIN_POLL_SECONDS), MAX_POLL_SECONDS)

    def _conform_funding(
        self, sub: Subscription, frame: pd.DataFrame, *, from_history: bool
    ) -> pd.DataFrame:
        """Hold one answer about funding for the life of a subscription.

        ``with_funding_column`` decides per *window*, and a feed asks about two:
        ``backfill`` over history and ``_poll`` over the recent tail. Those can
        disagree -- BTC/USDT perp's permanent ~40 h head gap means a full-history
        request finds a gap and returns no column while a recent window finds one.
        Priming unfunded and then polling funded is a stream changing its mind,
        and ``BarBuffer`` **raises** on exactly that (M42), which is right:
        measured, a cold start over full BTC history crashed on its first poll.

        The buffer's guard is not the thing to loosen; the feed is what must be
        consistent. So one answer is settled per subscription and later frames
        conform to it -- but **which frame gets to settle it matters**, because an
        absent column means two unrelated things and only one of them is a
        verdict.

        - **History says something permanent.** BTC's head gap does not close, so
          an uncovered ``backfill`` settles the stream unfunded and the tail
          conforms. It warns, because that is a process running ``crowding``
          neutral for its whole life (M20: +16.44% / 0.801 against the published
          +15.45% / 0.896) and an operator should not have to infer it.
        - **The tail says something transient.** The venue serves candles in real
          time while stored settlements advance only when a funding fetch runs, so
          the polled window outgrows its coverage every cadence. On a perp that is
          a lag, not a verdict, so an undecided subscription **withholds instead
          of deciding** -- committing there would silently pin the whole process
          unfunded on the timing of its first poll.
        - **Off-perp there is nothing to decide.** Spot never carries the column
          and never will, so it settles unfunded silently.

        Losing funding after the stream is already funded withholds too, for the
        same reason and by the same path: a raise would kill a paper run on a lag
        that resolves itself.

        Withholding loses nothing. The cursor advances only on an emitted event,
        so the next poll re-reads the same window and emits it once coverage
        catches up -- the bars are late, not gone. It is not silent either: the
        first withheld poll warns, ``funding_withheld_polls`` counts, and
        ``health().last_event_ms`` stops advancing. Turning a persistent stall
        into an alert belongs with restart and supervision, which R10
        deliberately does not own.
        """
        has_column = FUNDING_COLUMN in frame.columns
        settled = self._funded.get(sub.candle)
        undecided_tail = (
            settled is None
            and not has_column
            and not from_history
            and sub.instrument.market_type == "perp"
        )
        if undecided_tail or (settled and not has_column):
            self.funding_withheld_polls += 1
            if sub.candle not in self._withholding:
                # Once per stall rather than once per poll: at the default
                # cadence a persistent one would warn every five seconds.
                self._withholding.add(sub.candle)
                warnings.warn(
                    f"{sub.candle.key}: stored funding does not cover the polled window, "
                    f"so bars are withheld until it does. Emitting them unfunded would run "
                    f"a different strategy than the funded bars around them. Advance the "
                    f"settlements (strategy-lab fetch-funding) to resume.",
                    stacklevel=2,
                )
            return frame.iloc[:0]

        self._withholding.discard(sub.candle)
        if settled is None:
            self._funded[sub.candle] = has_column
            if not has_column and sub.instrument.market_type == "perp":
                warnings.warn(
                    f"{sub.candle.key}: stored funding does not cover this perp's history, "
                    f"so the stream runs unfunded and crowding stays neutral for the life "
                    f"of the process. BTC/USDT's ~40h leading gap is permanent and this is "
                    f"expected there; otherwise fetch funding and restart.",
                    stacklevel=2,
                )
            return frame
        if has_column and not settled:
            return frame.drop(columns=[FUNDING_COLUMN])
        return frame

    def _poll(self, sub: Subscription) -> list[BarEvent]:
        identity = _identity(sub)
        bar_ms = timeframe_to_millis(sub.timeframe)
        # No upper bound on a poll: it always wants everything up to now.
        frame = self.fetch(identity, self._since_ms(sub, bar_ms), None)
        if frame is None or frame.empty:
            return []
        frame = self._conform_funding(sub, frame, from_history=False)
        if frame.empty:
            return []

        events: list[BarEvent] = []
        ordered = _ordered(frame)
        newest = _epoch_ms(ordered.index[-1])
        for timestamp, row in ordered.iterrows():
            ts_open_ms = _epoch_ms(timestamp)
            # The newest row is the only one that can still be forming: the venue
            # has moved past every earlier interval. Deciding by *position* rather
            # than by wall-clock keeps this honest under an injected clock and
            # under a venue whose idea of "now" differs from ours.
            is_closed = ts_open_ms != newest
            if not is_closed and not sub.include_forming:
                continue
            # Keyed on the *values* as well as the identity, deliberately: a
            # venue correcting a bar it already closed is not re-sending the same
            # bar, and suppressing those would make ``lookback_bars`` decorative.
            key = (sub.candle.key, ts_open_ms, is_closed, _values(row))
            if key in self._seen:
                continue
            self._seen.add(key)
            # **Only a correction to the newest bar can actually land**, so an
            # older one is recorded as seen and dropped rather than yielded.
            # ``BarBuffer.append`` replaces on a repeated timestamp but drops
            # anything *older* than its last as out-of-order -- measured, a
            # correction four bars back increments ``dropped_out_of_order`` and
            # changes nothing. Emitting it anyway would inflate that counter, and
            # bill every downstream consumer for an event none of them can apply.
            # Such a correction is a real divergence from what storage will later
            # hold; closing it needs ``BarBuffer`` to accept an in-place
            # amendment, which is out of scope here and filed in the charter.
            if ts_open_ms < self._cursors.get(sub.candle, 0):
                continue
            bar = _row_to_bar(timestamp, row, sub.instrument, sub.timeframe, bar_ms)
            if not is_closed:
                bar = _forming(bar)
            events.append(BarEvent(bar=bar, ts_event_ms=bar.ts_close_ms, ts_recv_ms=None))
        return events

    def _since_ms(self, sub: Subscription, bar_ms: int) -> int:
        """How far back this subscription re-reads, from **its own** watermark.

        Per ``CandleId`` rather than one watermark for the feed: BTC 4h and BTC 1d
        advance at different rates, and a shared one would let the faster
        subscription drag the slower one's window past bars it had not seen. That
        is the same identity rule ``MarketSnapshot`` and ``ReplayFeed.frames``
        key on, and for the same reason.

        Once anything has been emitted the cursor is the watermark. Before that
        it is a **pinned** floor, not ``now - lookback`` recomputed each poll: a
        sliding window carries the start past bars the feed already fetched and
        withheld, and measured, 8 stalled polls at 4h moved it 7 bars, dropping 2
        out of a 5-bar lookback for good. That would make "withholding loses
        nothing" false exactly when it matters. Pinned, the window grows through
        a stall and empties in one poll when coverage returns.

        Not by seeding ``_cursors``, which means "newest *emitted*" and gates the
        correction rule -- the withheld bars would return as stale corrections
        and be dropped. And deliberately unbounded, because a cap reintroduces
        the same gap once a stall outlives it, where an unbounded window is only
        a slower poll that the withholding warning already says how to end.
        """
        newest = self._cursors.get(sub.candle)
        if newest is not None:
            return max(0, newest - bar_ms * self.lookback_bars)
        floor = self._window_floors.get(sub.candle)
        if floor is None:
            # A cold start wants the same lookback every later poll does; reaching
            # further back is ``backfill``'s job, and opening at the epoch would
            # paginate from 1970 before a single event was yielded.
            floor = max(0, self.clock.now_ms() - bar_ms * self.lookback_bars)
            self._window_floors[sub.candle] = floor
        return floor

    def resume_after(self, sub: Subscription, ts_open_ms: int) -> None:
        """Start polling from where a caller's own history ends.

        A process primes its buffer from ``backfill`` and then polls, and the two
        do not otherwise know about each other: ``backfill`` sets no cursor, so
        the first poll re-reads its whole window -- widened on a perp to cover its
        own funding, which at 15m is over a hundred bars -- and every bar older
        than the primed history is emitted for ``BarBuffer`` to drop as
        out-of-order. Nothing breaks, but the cursor exists to say where the
        caller is, and after priming the caller is here.
        """
        self._cursors[sub.candle] = max(self._cursors.get(sub.candle, 0), ts_open_ms)

    async def backfill(self, sub: Subscription, start_ms: int, end_ms: int) -> AsyncIterator[Bar]:
        """Every **closed** bar covering ``[start_ms, end_ms]``, ascending.

        This is what a cold start warms itself from, and it yields ``Bar`` because
        ``StrategyRunner.prime_bars`` takes ``Bar`` -- the two compose, which
        before R10 they did not.

        **Closed is decided by the clock, not by position.** ``_poll`` can call
        the newest row forming because the venue always serves up to now, so
        there is always a partial row at the tip. A backfill is *bounded*: a
        request ending last month gets a frame whose last row closed long ago,
        and dropping it because it happened to be last would silently warm a
        runner with one bar fewer than a replay of the same range -- an off-by-one
        that moves the first live signal.
        """
        identity = _identity(sub)
        bar_ms = timeframe_to_millis(sub.timeframe)
        frame = self.fetch(identity, start_ms, end_ms)
        if frame is None or frame.empty:
            return
        frame = self._conform_funding(sub, frame, from_history=True)
        if frame.empty:
            return
        now = self.clock.now_ms()
        for timestamp, row in _ordered(frame).iterrows():
            ts_open_ms = _epoch_ms(timestamp)
            if not start_ms <= ts_open_ms <= end_ms:
                continue
            if ts_open_ms + bar_ms > now:
                continue  # still forming
            yield _row_to_bar(timestamp, row, sub.instrument, sub.timeframe, bar_ms)

    async def server_time_ms(self) -> int:
        return self._last_event_ms or 0

    def health(self) -> FeedHealth:
        return FeedHealth(connected=True, last_event_ms=self._last_event_ms)


def _values(row: pd.Series) -> tuple[float | None, ...]:
    """What a bar says, for deciding whether a re-read is the same bar.

    Rounded, because a venue re-serving an unchanged bar can still round the last
    digit differently between responses and that is not a correction.

    ``NaN`` becomes ``None`` because ``NaN != NaN``: left as a float, a bar
    carrying one would never match its own earlier key, so it would re-emit on
    every poll and grow ``_seen`` without bound -- a gap in the data turning into
    a leak and a signal storm.
    """
    # Funding counts as something the bar says: a settlement stored after the
    # fact moves a bar's rate and leaves its OHLCV alone, so a price-only
    # fingerprint files that as unchanged and the buffer keeps a rate the record
    # will later disagree with. Only when the frame carries it, so an unfunded
    # stream keys exactly as it did.
    names = ("open", "high", "low", "close", "volume")
    if FUNDING_COLUMN in row.index:
        names += (FUNDING_COLUMN,)
    values = []
    for name in names:
        value = float(row[name])
        values.append(None if value != value else round(value, 10))
    return tuple(values)


def _identity(sub: Subscription) -> MarketDataIdentity:
    return MarketDataIdentity(
        exchange=sub.instrument.exchange,
        market_type=sub.instrument.market_type,
        symbol=sub.instrument.symbol,
        timeframe=sub.timeframe,
    )


def _forming(bar: Bar) -> Bar:
    """The same bar, marked as still open.

    ``ts_close_ms`` is left where it is -- it is the interval's end, not the time
    the data was taken, and moving it would make the same interval sort
    differently depending on when it was polled.
    """
    return replace(bar, is_closed=False)


__all__ = ["DEFAULT_LOOKBACK_BARS", "MAX_POLL_SECONDS", "MIN_POLL_SECONDS", "LiveFeed"]
