"""The bar history a strategy sees, and nothing else."""

from __future__ import annotations

import pandas as pd

from strategy_lab.core.types import Bar
from strategy_lab.features.flow import FUNDING_COLUMN

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

    This is the seam between the event loop (Decimal) and the indicator layer
    (float64), the same crossing ``db.candles.load_candles`` makes. Money never
    crosses back: prices leaving the engine come from the ``Bar``, not the frame.

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
        self._funding: list[float] = []
        self._carries_funding: bool | None = None
        self._frame: pd.DataFrame | None = None
        self.dropped_out_of_order = 0
        self.replaced_duplicates = 0

    def __len__(self) -> int:
        return len(self._timestamps)

    @property
    def carries_funding(self) -> bool:
        """Whether these bars settle funding, i.e. whether ``frame()`` has the column."""
        return bool(self._carries_funding)

    def append(self, bar: Bar) -> None:
        timestamp = pd.Timestamp(bar.ts_open_ms, unit="ms", tz="UTC")

        # Discarded before validated, deliberately: a stale bar never joins the
        # history, so its funding cannot make the *frame* inconsistent, and
        # raising on one would turn a feed pathology this class exists to absorb
        # into a crash.
        if self._timestamps and timestamp < self._timestamps[-1]:
            self.dropped_out_of_order += 1
            return  # stale replay after a reconnect

        self._require_consistent_funding(bar)

        if self._timestamps and timestamp == self._timestamps[-1]:
            self._write(-1, bar)
            self.replaced_duplicates += 1
            self._frame = None
            return

        self._timestamps.append(timestamp)
        for name in _COLUMNS:
            self._columns[name].append(float(getattr(bar, name)))
        if self._carries_funding:
            self._funding.append(float(bar.funding_rate))
        self._frame = None

    def frame(self) -> pd.DataFrame:
        """The full history as float64, indexed by UTC bar-open time.

        Cached until the next ``append``; invalidating it there is what keeps a
        strategy from being handed a frame that is missing the bar it is being
        asked about. The cached object is shared, so callers must not mutate it.

        **``funding_rate`` is present only when the bars carry one**, and that is
        load-bearing rather than tidy. ``state_machine_core.build_feature_frame``
        decides whether crowding is real with ``FUNDING_COLUMN in df.columns``, so
        an always-present column full of ``NaN`` would report
        ``crowding_measured=True`` on a spot frame and feed the feature garbage --
        replacing a fallback that is *correct* off-perp with a silent wrong
        answer. Presence means measured, here as everywhere else.
        """
        if self._frame is None:
            index = pd.DatetimeIndex(self._timestamps, name="timestamp", tz="UTC")
            columns = {name: self._columns[name] for name in _COLUMNS}
            if self._carries_funding:
                columns[FUNDING_COLUMN] = self._funding
            self._frame = pd.DataFrame(columns, index=index)
        return self._frame

    def _require_consistent_funding(self, bar: Bar) -> None:
        """Every bar settles, or none does — a stream cannot change its mind.

        Dropping the column when funding stops arriving would silently run a
        *different strategy* from the one the earlier bars ran, which is M20 in a
        feed rather than a flag. Adding one mid-run is the same fault mirrored.
        """
        carries = bar.funding_rate is not None
        if self._carries_funding is None:
            self._carries_funding = carries
            return
        if carries != self._carries_funding:
            had, now = ("with", "without") if self._carries_funding else ("without", "with")
            raise ValueError(
                f"{bar.instrument.key} bar at {bar.ts_open_ms} arrived {now} funding after "
                f"{len(self._timestamps)} bars {had} it. A stream that changes its mind "
                f"silently changes what a funding-reading strategy computes."
            )

    def _write(self, position: int, bar: Bar) -> None:
        for name in _COLUMNS:
            self._columns[name][position] = float(getattr(bar, name))
        if self._carries_funding:
            self._funding[position] = float(bar.funding_rate)
