from __future__ import annotations

from strategy_lab.core.types import Bar, BarEvent, InstrumentId, MarketSnapshot


class MarketClock:
    """Groups a time-ordered event stream into complete per-timestamp snapshots.

    A timestamp is complete when an event with a **later** timestamp arrives --
    never by looking ahead, which is the same rule a live feed is bound by. The
    consequence is deliberate: a cross-sectional signal for bar *t* is available
    at *t+1*, in replay exactly as in live, so the two paths cannot diverge.

    The final timestamp has no successor, so it is released by ``flush()``.
    """

    def __init__(self) -> None:
        self._ts_event_ms: int | None = None
        self._bars: dict[InstrumentId, Bar] = {}

    def on_event(self, event: BarEvent) -> MarketSnapshot | None:
        if self._ts_event_ms is not None and event.ts_event_ms < self._ts_event_ms:
            raise ValueError(
                f"event at {event.ts_event_ms} arrived out of order after "
                f"{self._ts_event_ms}; the feed must yield a total time order"
            )

        completed = None
        if self._ts_event_ms is not None and event.ts_event_ms > self._ts_event_ms:
            completed = self._take()

        self._ts_event_ms = event.ts_event_ms
        self._bars[event.bar.instrument] = event.bar
        return completed

    def flush(self) -> MarketSnapshot | None:
        return self._take()

    def _take(self) -> MarketSnapshot | None:
        if self._ts_event_ms is None or not self._bars:
            return None
        snapshot = MarketSnapshot(ts_event_ms=self._ts_event_ms, bars=dict(self._bars))
        self._bars.clear()
        return snapshot
