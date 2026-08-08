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
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field, replace

import pandas as pd

from strategy_lab.core.types import Bar, BarEvent
from strategy_lab.feeds.base import FeedHealth, Subscription
from strategy_lab.feeds.replay import _epoch_ms, _ordered, _row_to_bar
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.timeframes import timeframe_to_millis

# Bars re-read on every poll. More than one because a venue can correct the bar
# it just closed, and a feed that only ever asked for the newest would never see
# the correction -- the same reason `server.refresh_candles` reaches back rather
# than fetching the tip.
DEFAULT_LOOKBACK_BARS = 5

# The default cadence, as a fraction of the bar being polled. See ``_interval``.
POLLS_PER_BAR = 60
MIN_POLL_SECONDS = 5.0
MAX_POLL_SECONDS = 300.0


def _fetch_recent(identity: MarketDataIdentity, since_ms: int) -> pd.DataFrame:
    """The venue's own recent candles, through the client the fetchers already use."""
    from strategy_lab.market_data.binance import CryptoOhlcvClient

    client = CryptoOhlcvClient(
        exchange_id=identity.exchange, market_type=identity.market_type
    )
    since = pd.Timestamp(since_ms, unit="ms", tz="UTC").isoformat()
    return client.fetch_ohlcv(identity.symbol, identity.timeframe, since=since)


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
    fetch: Callable[[MarketDataIdentity, int], pd.DataFrame] = _fetch_recent
    sleep: Callable[[float], object] = asyncio.sleep

    _seen: set[tuple[str, str, int, bool]] = field(default_factory=set, init=False)
    _last_event_ms: int | None = field(default=None, init=False, repr=False)

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
                for event in self._poll(sub):
                    self._last_event_ms = event.ts_event_ms
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

    def _poll(self, sub: Subscription) -> list[BarEvent]:
        identity = _identity(sub)
        bar_ms = timeframe_to_millis(sub.timeframe)
        frame = self.fetch(identity, self._since_ms(bar_ms))
        if frame is None or frame.empty:
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
            key = (sub.instrument.key, sub.timeframe, ts_open_ms, is_closed)
            if key in self._seen:
                continue
            self._seen.add(key)
            bar = _row_to_bar(timestamp, row, sub.instrument, sub.timeframe, bar_ms)
            if not is_closed:
                bar = _forming(bar)
            events.append(BarEvent(bar=bar, ts_event_ms=bar.ts_close_ms, ts_recv_ms=None))
        return events

    def _since_ms(self, bar_ms: int) -> int:
        newest = self._last_event_ms or 0
        return max(0, newest - bar_ms * self.lookback_bars)

    async def backfill(self, sub: Subscription, start_ms: int, end_ms: int) -> AsyncIterator[Bar]:
        """Every closed bar covering ``[start_ms, end_ms]``, ascending.

        This is what a cold start warms itself from, and it yields ``Bar`` because
        ``StrategyRunner.prime_bars`` takes ``Bar`` -- the two compose, which
        before R10 they did not.
        """
        identity = _identity(sub)
        bar_ms = timeframe_to_millis(sub.timeframe)
        frame = self.fetch(identity, start_ms)
        if frame is None or frame.empty:
            return
        ordered = _ordered(frame)
        newest = _epoch_ms(ordered.index[-1])
        for timestamp, row in ordered.iterrows():
            ts_open_ms = _epoch_ms(timestamp)
            if ts_open_ms == newest or not start_ms <= ts_open_ms <= end_ms:
                continue
            yield _row_to_bar(timestamp, row, sub.instrument, sub.timeframe, bar_ms)

    async def server_time_ms(self) -> int:
        return self._last_event_ms or 0

    def health(self) -> FeedHealth:
        return FeedHealth(connected=True, last_event_ms=self._last_event_ms)


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
