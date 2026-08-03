"""Injectable time source for the event-driven engine.

- The engine must never call ``time.time()`` (or any wall-clock API) directly;
  it reads time only through a ``Clock`` so that replays stay reproducible.
- ``LiveClock`` is the only place in this package permitted to read the
  system clock.
- ``SimClock`` is monotonic by construction: ``advance_to`` with a timestamp
  at or before the current time is a no-op, it never rewinds. This lets a
  websocket reconnect replay already-seen bars without perturbing time.
- All timestamps are UTC epoch milliseconds as ``int``, matching
  ``strategy_lab.core.types``.
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
    """Deterministic clock driven by event timestamps. Monotonic by construction."""

    def __init__(self, start_ms: int = 0) -> None:
        self._now_ms = start_ms

    def now_ms(self) -> int:
        return self._now_ms

    def advance_to(self, ts_ms: int) -> None:
        if ts_ms > self._now_ms:
            self._now_ms = ts_ms
