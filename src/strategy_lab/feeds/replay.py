from __future__ import annotations

import heapq
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

import pandas as pd

from strategy_lab.features.flow import FUNDING_COLUMN
from strategy_lab.core.types import Bar, BarEvent, CandleId, InstrumentId
from strategy_lab.feeds.base import FeedHealth, Subscription
from strategy_lab.timeframes import timeframe_to_millis


@dataclass
class ReplayFeed:
    """Replays stored candles as BarEvents; the runner cannot tell this from a websocket.

    Duplicate timestamps are collapsed last-wins -- see :func:`_ordered`.
    Subscriptions are merged into one globally time-ordered stream -- see
    :meth:`stream`.
    """

    frames: dict[CandleId, pd.DataFrame] = field(default_factory=dict)
    name: str = "replay"
    _last_event_ms: int | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_database(
        cls,
        subscriptions: Sequence[Subscription],
        *,
        start: str | None = None,
        end: str | None = None,
        limit_bars: int | None = None,
        funding: bool = True,
        required: bool = False,
        database_url: str | None = None,
    ) -> ReplayFeed:
        """Stored candles as a feed, with a perp's funding attached to its bars.

        Funding goes on through ``backtests.funding_frame.with_funding_column`` --
        **the engine's own function, not a second implementation** -- so the
        alignment rule is shared rather than duplicated: settlements are charged
        to the bar whose interval *contains* them, because Binance stamps them up
        to 47 ms past the boundary and an equality join drops 43% of BTC's stored
        history.

        ``required`` is the caller's, and defaults to **False** for the reason the
        sweep's attachment does: a feed does not know which strategy will read it,
        and refusing every perp whose coverage is imperfect would break replaying
        ``donchian`` -- which reads no funding-derived feature -- over BTC's
        permanent 40 h leading gap. A caller that *does* know says so, and the
        ``replay`` CLI asks the strategy the same way ``sweep`` does. Uncovered
        with ``required=False`` means no column at all rather than a partial one,
        so a funding-reading strategy falls back to neutral and records it, which
        is the pre-existing behaviour rather than a silently gross number.

        Attached *after* ``limit_bars``, so the guard is asked about the range that
        will actually be replayed rather than the one that was loaded.
        """
        from strategy_lab.backtests.funding_frame import with_funding_column
        from strategy_lab.db import load_candles
        from strategy_lab.market_data.base import MarketDataIdentity

        frames: dict[CandleId, pd.DataFrame] = {}
        for sub in subscriptions:
            identity = MarketDataIdentity(
                exchange=sub.instrument.exchange,
                market_type=sub.instrument.market_type,
                symbol=sub.instrument.symbol,
                timeframe=sub.timeframe,
            )
            df = load_candles(
                exchange=identity.exchange,
                market_type=identity.market_type,
                symbol=identity.symbol,
                timeframe=identity.timeframe,
                start=start,
                end=end,
                database_url=database_url,
            )
            if limit_bars is not None:
                df = df.tail(limit_bars)
            df, _ = with_funding_column(identity, df, enabled=funding, required=required)
            frames[sub.candle] = df
        return cls(frames=frames)

    async def stream(self, subs: Sequence[Subscription]) -> AsyncIterator[BarEvent]:
        """Yield every subscription's bars as one globally time-ordered stream.

        Ties -- several candles closing at the same instant -- break on the full
        candle key: without a total order the replay/live determinism proof does not
        hold for more than one subscription. The timeframe is part of that key
        because one symbol subscribed at 4h and at 1d ties with *itself* six times a
        day, and an instrument-only key leaves those two bars unordered.
        """
        merged = heapq.merge(
            *(self._events_for(sub) for sub in subs),
            key=lambda event: (event.ts_event_ms, event.bar.candle.key),
        )
        for event in merged:
            self._last_event_ms = event.ts_event_ms
            yield event

    def _events_for(self, sub: Subscription) -> Iterator[BarEvent]:
        df = self.frames.get(sub.candle)
        if df is None or df.empty:
            return
        bar_ms = timeframe_to_millis(sub.timeframe)
        for timestamp, row in _ordered(df).iterrows():
            bar = _row_to_bar(timestamp, row, sub.instrument, sub.timeframe, bar_ms)
            yield BarEvent(bar=bar, ts_event_ms=bar.ts_close_ms, ts_recv_ms=None)

    async def backfill(self, sub: Subscription, start_ms: int, end_ms: int) -> AsyncIterator[Bar]:
        df = self.frames.get(sub.candle)
        if df is None or df.empty:
            return
        bar_ms = timeframe_to_millis(sub.timeframe)
        for timestamp, row in _ordered(df).iterrows():
            ts_open_ms = _epoch_ms(timestamp)
            if start_ms <= ts_open_ms <= end_ms:
                yield _row_to_bar(timestamp, row, sub.instrument, sub.timeframe, bar_ms)

    async def server_time_ms(self) -> int:
        return self._last_event_ms or 0

    def health(self) -> FeedHealth:
        return FeedHealth(connected=True, last_event_ms=self._last_event_ms)


def _row_to_bar(
    timestamp: pd.Timestamp,
    row: pd.Series,
    instrument: InstrumentId,
    timeframe: str,
    bar_ms: int,
) -> Bar:
    ts_open_ms = _epoch_ms(timestamp)
    # Carried only when the frame has it, which is what makes a bar's funding
    # ``None`` off a perp and a number on one -- and what ``BarBuffer`` reads to
    # decide whether ``FUNDING_COLUMN`` reaches the strategy at all.
    funding = row.get(FUNDING_COLUMN) if FUNDING_COLUMN in row.index else None
    return Bar(
        instrument=instrument,
        timeframe=timeframe,
        ts_open_ms=ts_open_ms,
        ts_close_ms=ts_open_ms + bar_ms - 1,
        open=_decimal(row["open"]),
        high=_decimal(row["high"]),
        low=_decimal(row["low"]),
        close=_decimal(row["close"]),
        volume=_decimal(row["volume"]),
        is_closed=True,
        funding_rate=None if funding is None else _decimal(funding),
    )


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    """Ascending, one row per timestamp, last occurrence winning.

    The protocol forbids yielding the same (instrument, timeframe, ts_open_ms,
    is_closed) twice, and a websocket redelivers bars after a reconnect — so honouring
    that here is what makes ReplayFeed a usable reference for the live adapters. Last
    wins because the redelivered copy is the corrected one; this matches
    ``normalize_candle_frame`` in ``db/candles.py``.

    ``kind="stable"`` is load-bearing, not decoration: ``sort_index()`` defaults to
    quicksort, which reorders equal keys, so without it "last" would mean an arbitrary
    one of the duplicates rather than the last row the caller supplied.
    """
    ordered = df.sort_index(kind="stable")
    return ordered.loc[~ordered.index.duplicated(keep="last")]


def _epoch_ms(timestamp: pd.Timestamp) -> int:
    """UTC epoch milliseconds, by integer division rather than float arithmetic.

    ``int(timestamp.timestamp() * 1000)`` is exact for whole-second timestamps — every
    candle open time we store is one — but ``Timestamp.timestamp()`` rounds to
    microseconds, so truncation lands 1 ms low for ~0.7% of sub-second timestamps
    (measured: 1,399 of 200,000 random ms-aligned values, always -1). ts_open_ms is the
    bar's identity for dedup and for replay/live equivalence, so it may not be
    approximately right.
    """
    return timestamp.value // 1_000_000


def _decimal(value) -> Decimal:
    return Decimal(str(value))
