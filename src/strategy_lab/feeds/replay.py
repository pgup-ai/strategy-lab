from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

import pandas as pd

from strategy_lab.core.types import Bar, BarEvent, InstrumentId
from strategy_lab.feeds.base import FeedHealth, Subscription
from strategy_lab.timeframes import timeframe_to_millis

FrameKey = tuple[InstrumentId, str]


@dataclass
class ReplayFeed:
    """Replays stored candles as BarEvents. Satisfies the same protocol as a live feed.

    This is the injection point that makes backtest, replay, and live share one
    strategy code path: the runner cannot tell this apart from a websocket.

    Known limitations, both pinned by tests in ``tests/test_replay_feed.py``:

    - **Subscriptions are drained sequentially, not interleaved by time.**
      ``stream([a, b])`` yields every bar of ``a`` and only then the first bar of
      ``b``. A live feed multiplexes both by arrival time, so a multi-symbol replay
      is chronologically wrong here. Phase 1a is single-symbol; fix this alongside
      the live feed rather than guessing at the merge semantics now.
    - **Duplicate index entries are replayed twice**, which the protocol forbids.
      Unreachable via :meth:`from_database` (``market_candles`` is unique on
      ``(exchange, market_type, symbol, timeframe, timestamp)``), reachable with a
      hand-built ``frames`` dict.
    """

    frames: dict[FrameKey, pd.DataFrame] = field(default_factory=dict)
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
        database_url: str | None = None,
    ) -> ReplayFeed:
        from strategy_lab.db import load_candles

        frames: dict[FrameKey, pd.DataFrame] = {}
        for sub in subscriptions:
            df = load_candles(
                exchange=sub.instrument.exchange,
                market_type=sub.instrument.market_type,
                symbol=sub.instrument.symbol,
                timeframe=sub.timeframe,
                start=start,
                end=end,
                database_url=database_url,
            )
            if limit_bars is not None:
                df = df.tail(limit_bars)
            frames[(sub.instrument, sub.timeframe)] = df
        return cls(frames=frames)

    async def stream(self, subs: Sequence[Subscription]) -> AsyncIterator[BarEvent]:
        for sub in subs:
            df = self.frames.get((sub.instrument, sub.timeframe))
            if df is None or df.empty:
                continue
            bar_ms = timeframe_to_millis(sub.timeframe)
            for timestamp, row in df.sort_index().iterrows():
                bar = _row_to_bar(timestamp, row, sub.instrument, sub.timeframe, bar_ms)
                self._last_event_ms = bar.ts_close_ms
                yield BarEvent(bar=bar, ts_event_ms=bar.ts_close_ms, ts_recv_ms=None)

    async def backfill(self, sub: Subscription, start_ms: int, end_ms: int) -> AsyncIterator[Bar]:
        df = self.frames.get((sub.instrument, sub.timeframe))
        if df is None or df.empty:
            return
        bar_ms = timeframe_to_millis(sub.timeframe)
        for timestamp, row in df.sort_index().iterrows():
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
    )


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
