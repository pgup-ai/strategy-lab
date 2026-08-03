"""Injectable time source for the event-driven engine.

The engine must never call ``time.time()`` (or any wall-clock API) directly; it
reads time only through a ``Clock``, so that a replay's clock comes from its own
events and stays reproducible. Timestamps are UTC epoch milliseconds as ``int``.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def now_ms(self) -> int:
        ...


class LiveClock:
    """Wall-clock time. The only place in the package permitted to read the system clock."""

    def now_ms(self) -> int:
        return int(time.time() * 1000)


class SimClock:
    """Deterministic clock driven by event timestamps.

    ``advance_to`` never rewinds: a websocket reconnect replays already-seen bars,
    whose timestamps are older than the clock's, and time must not follow them back.
    """

    def __init__(self, start_ms: int = 0) -> None:
        self._now_ms = start_ms

    def now_ms(self) -> int:
        return self._now_ms

    def advance_to(self, ts_ms: int) -> None:
        if ts_ms > self._now_ms:
            self._now_ms = ts_ms
