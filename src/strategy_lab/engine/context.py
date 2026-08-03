"""The bar history a strategy sees, and nothing else.

This is the seam between the event loop (Decimal, epoch milliseconds) and the
indicator layer (pandas, float64).
"""

from __future__ import annotations

import pandas as pd

from strategy_lab.core.types import Bar

_COLUMNS = ("open", "high", "low", "close", "volume")


class BarBuffer:
    """Accumulates closed bars and materializes the DataFrame strategies consume.

    Retains FULL history rather than a rolling window. This is load-bearing, not
    laziness: ``turnaround_v1``/``turnaround_v2`` compute ``ewm(adjust=False)``,
    which is recursive from the first element, so its value at bar *t* depends on
    every prior bar. Measured with a 60-bar window both produce different values
    than the whole-history run, i.e. live signals would silently diverge from
    backtest signals on exactly the two strategies used for crypto. Do not
    "optimize" this into a bounded window without also proving the strategies
    that run on it are window-safe.

    Values are stored as float64 -- this is the documented Decimal -> float
    boundary, the same one ``db.candles.load_candles`` crosses. Money never
    crosses back the other way: prices leaving the engine come from the ``Bar``,
    not from the frame.

    Two feed pathologies are absorbed silently, because a reconnecting websocket
    produces both routinely: a bar older than the newest one is dropped, and a
    bar repeating the newest timestamp overwrites it (the redelivered copy is the
    corrected one, matching ``ReplayFeed._ordered`` and ``normalize_candle_frame``).
    Silence would also hide a genuinely broken feed, so both are counted --
    ``dropped_out_of_order`` and ``replaced_duplicates`` are there for a health
    check to read.
    """

    def __init__(self) -> None:
        self._timestamps: list[pd.Timestamp] = []
        self._columns: dict[str, list[float]] = {name: [] for name in _COLUMNS}
        self._frame: pd.DataFrame | None = None
        self.dropped_out_of_order = 0
        self.replaced_duplicates = 0

    def __len__(self) -> int:
        return len(self._timestamps)

    def append(self, bar: Bar) -> None:
        timestamp = pd.Timestamp(bar.ts_open_ms, unit="ms", tz="UTC")

        if self._timestamps:
            last = self._timestamps[-1]
            if timestamp < last:
                self.dropped_out_of_order += 1
                return  # stale replay after a reconnect
            if timestamp == last:
                self._write(-1, bar)
                self.replaced_duplicates += 1
                self._frame = None
                return

        self._timestamps.append(timestamp)
        for name in _COLUMNS:
            self._columns[name].append(float(getattr(bar, name)))
        self._frame = None

    def frame(self) -> pd.DataFrame:
        """The full history as OHLCV float64, indexed by UTC bar-open time.

        Cached until the next ``append``; invalidating it there is what keeps a
        strategy from being handed a frame that is missing the bar it is being
        asked about. Callers must treat the result as read-only -- it is the same
        object every time until the buffer changes.
        """
        if self._frame is None:
            index = pd.DatetimeIndex(self._timestamps, name="timestamp", tz="UTC")
            self._frame = pd.DataFrame(
                {name: self._columns[name] for name in _COLUMNS}, index=index
            )
        return self._frame

    def _write(self, position: int, bar: Bar) -> None:
        for name in _COLUMNS:
            self._columns[name][position] = float(getattr(bar, name))
