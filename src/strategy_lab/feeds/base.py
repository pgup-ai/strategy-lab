"""The injection point that lets one strategy run in backtest, replay, and live.

Everything downstream (bar buffer, runner, strategies) consumes ``BarEvent``s
from a ``MarketDataFeed`` and must never learn where they came from. The
protocol is therefore deliberately thin: it says *what* to cover, never *how* to
page for it, because the venues disagree about the how.

Measured during design:

- Binance returns candles ascending; OKX returns them descending.
- Binance pages forward with ``startTime``; OKX pages backward with ``after``.
- Binance serves all history from one endpoint; OKX splits recent (last 1,440
  bars) from deep history at a different path.
- OKX defaults daily-and-above bars to UTC+8 unless a ``utc`` suffix is sent.

So ``backfill`` is specified as "cover this range" and the adapter owns ordering
and paging direction. Do not add convenience here that bakes in one venue's
shape (no ``limit``, no "next page cursor", no "advance startTime by
limit x bar_ms") — that is the Binance-shaped assumption we would have to tear
out when the second adapter lands.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from strategy_lab.core.types import Bar, BarEvent, CandleId, InstrumentId


@dataclass(frozen=True, slots=True)
class Subscription:
    instrument: InstrumentId
    timeframe: str
    include_forming: bool = False

    @property
    def candle(self) -> CandleId:
        return CandleId(self.instrument, self.timeframe)


@dataclass(frozen=True, slots=True)
class FeedHealth:
    connected: bool
    last_event_ms: int | None = None
    lag_ms: int | None = None
    reconnects: int = 0
    gaps_detected: int = 0
    weight_used: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class MarketDataFeed(Protocol):
    """Contract shared by live exchange feeds and the Postgres replay feed.

    Implementations MUST yield bars in ascending ts_open_ms order and MUST NOT
    yield the same (instrument, timeframe, ts_open_ms, is_closed) twice. The venue
    quirks listed above are the adapter's problem, not the caller's.

    Note for anyone writing the second implementation: ``isinstance(feed,
    MarketDataFeed)`` only proves the member *names* exist. It does not check
    signatures and cannot check the ordering/no-duplicate guarantees above, which
    are the parts that actually matter. Reuse the behavioural contract checks in
    ``tests/test_replay_feed.py`` instead of trusting ``isinstance``.
    """

    name: str

    def stream(self, subs: Sequence[Subscription]) -> AsyncIterator[BarEvent]:
        """Yield events until exhausted (replay) or cancelled (live)."""
        ...

    def backfill(self, sub: Subscription, start_ms: int, end_ms: int) -> AsyncIterator[Bar]:
        """Yield every closed bar covering [start_ms, end_ms], ascending."""
        ...

    async def server_time_ms(self) -> int:
        ...

    def health(self) -> FeedHealth:
        ...
