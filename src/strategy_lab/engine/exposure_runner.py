"""The event path for the continuous-exposure contract.

``StrategyRunner`` drives ``SignalSet`` strategies and refuses anything else at
construction (M40). This is its counterpart: it calls ``compute_target`` over an
expanding buffer, reads the last row, and emits an
:class:`~strategy_lab.storage.signals.ExposureSignal` on the bars a target
actually reaches the book.

**A separate class rather than a widened ``StrategyRunner``**, for the reason the
three registries exist: a class that dispatched on contract would run both and
say nothing when one half broke, which is what a shared registry was measured to
do -- an empty ``exposure_registry`` silently skipped 4 parametrized tests and
exited 0. Two classes make the boolean suites and the exposure suites cover
disjoint code, so neither can go green by not running.

**The band is the engine's rule, and it is what makes this comparable to a
backtest.** ``exposure_engine`` submits a target only once it has moved
``rebalance_threshold`` from the last one *submitted* -- never from the previous
bar's, and never from the realized position, which would define the band in terms
of its own output. That rule is sequential by construction, so a streaming runner
holds its one piece of state (``_held``) and gets it for free where the
vectorized path needs a loop.

**``side`` names the direction of the level, not an event** (M41), so a flip
from +0.5 to -0.5 is **one** row where a boolean reversal needs two:
``target=-0.5`` already says the book is short half, and ``exit_long`` alone would
not say what replaced it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from decimal import Decimal

import pandas as pd

from strategy_lab.core.clock import Clock
from strategy_lab.core.types import Bar, BarEvent, BarReason, InstrumentId, Side, Signal
from strategy_lab.engine.context import BarBuffer
from strategy_lab.engine.reasons import reason_for
from strategy_lab.feeds.replay import _row_to_bar
from strategy_lab.storage.signals import ExposureSignal
from strategy_lab.strategies.base import require_warmup_bars
from strategy_lab.timeframes import timeframe_to_millis

DEFAULT_REBALANCE_THRESHOLD = 0.05


def require_exposure_contract(strategy: object) -> None:
    """Refuse a strategy this runner cannot call, at construction.

    The mirror of ``runner.require_signal_set_contract``, which carries the
    measurement: a contract mismatch that waits for the first post-warmup bar is
    invisible for as long as the warmup (M40).
    """
    if not hasattr(strategy, "compute_target"):
        name = getattr(strategy, "name", type(strategy).__name__)
        raise TypeError(
            f"{name} has no compute_target: it is a SignalSet strategy, which "
            f"ExposureRunner cannot drive. Use engine.runner.StrategyRunner."
        )


class ExposureRunner:
    """Turns a stream of bars into ``ExposureSignal``s using a vectorized strategy.

    The same three pieces as ``StrategyRunner`` in the same order -- real
    ``BarBuffer``, the strategy called over the full buffer, the last row read --
    so the causality argument is the same one, and
    ``tests/test_exposure_determinism.py`` is what holds it.

    Cost is the same shape too: every post-warmup bar recomputes over the whole
    buffer, so replay is quadratic and live is one call per bar.
    """

    def __init__(
        self,
        *,
        strategy,
        instrument: InstrumentId,
        timeframe: str,
        clock: Clock,
        rebalance_threshold: float = DEFAULT_REBALANCE_THRESHOLD,
        allow_forming_bars: bool = False,
        record_reasons: bool = True,
    ) -> None:
        require_warmup_bars(getattr(strategy, "name", "strategy"), strategy.warmup_bars)
        require_exposure_contract(strategy)
        if rebalance_threshold < 0:
            raise ValueError(
                f"rebalance_threshold must be >= 0, not {rebalance_threshold}. A negative "
                f"band submits on every bar while claiming to damp, which is the one "
                f"reading it must never have."
            )
        self.strategy = strategy
        self.instrument = instrument
        self.timeframe = timeframe
        self.clock = clock
        self.rebalance_threshold = rebalance_threshold
        self.allow_forming_bars = allow_forming_bars
        self.record_reasons = record_reasons
        self.buffer = BarBuffer()
        self._reasons: dict[int, BarReason] = {}
        # The book starts flat, so the first target is measured against 0.0 --
        # ``exposure_engine._banded``'s own opening state. Under a non-zero band
        # that is also why the leading zeros a warmup produces cost no rows; under
        # a zero band every bar submits, here and in the engine alike.
        self._held = 0.0

    @property
    def reasons(self) -> tuple[BarReason, ...]:
        return tuple(self._reasons.values())

    @property
    def held(self) -> float:
        """The last target submitted, which is what the band measures against."""
        return self._held

    def prime_bars(self, bars: Iterable[Bar]) -> None:
        """Load warmup history without emitting, the same as ``StrategyRunner``.

        Takes ``Bar`` so it composes with ``MarketDataFeed.backfill``; ``prime``
        below is the stored-candles adapter over it.
        """
        for bar in bars:
            self.buffer.append(bar)

    def prime(self, history: pd.DataFrame) -> None:
        """Load warmup history from stored candles, without emitting."""
        bar_ms = timeframe_to_millis(self.timeframe)
        self.prime_bars(
            _row_to_bar(timestamp, row, self.instrument, self.timeframe, bar_ms)
            for timestamp, row in history.sort_index(kind="stable").iterrows()
        )

    def on_event(self, event: BarEvent) -> Sequence[ExposureSignal]:
        return self.on_bar(event.bar)

    def on_bar(self, bar: Bar) -> Sequence[ExposureSignal]:
        if not bar.is_closed and not self.allow_forming_bars:
            return ()

        if hasattr(self.clock, "advance_to"):
            self.clock.advance_to(bar.ts_close_ms)

        # A bar older than the buffer's last is dropped there and the buffer is
        # unchanged; continuing would submit a target computed from history the
        # dropped bar never joined, and stamp it with that bar's timestamp.
        dropped = self.buffer.dropped_out_of_order
        self.buffer.append(bar)
        if self.buffer.dropped_out_of_order != dropped:
            return ()

        if len(self.buffer) <= self.strategy.warmup_bars:
            return ()

        frame = self.buffer.frame()
        target = float(self.strategy.compute_target(frame).target.iloc[-1])
        if self.record_reasons:
            reason = reason_for(
                self.strategy,
                bar=bar,
                frame=frame,
                instrument=self.instrument,
                timeframe=self.timeframe,
                clock=self.clock,
            )
            if reason is not None:
                self._reasons[bar.ts_open_ms] = reason

        if abs(target - self._held) < self.rebalance_threshold:
            return ()
        previous, self._held = self._held, target
        return (self._emit(target, previous, bar),)

    def _emit(self, target: float, previous: float, bar: Bar) -> ExposureSignal:
        return ExposureSignal(
            signal=Signal(
                instrument=self.instrument,
                timeframe=self.timeframe,
                strategy_id=self.strategy.name,
                strategy_version=self.strategy.version,
                ts_bar_ms=bar.ts_open_ms,
                ts_emit_ms=self.clock.now_ms(),
                side=_side_for(target, previous),
                bar_is_closed=bar.is_closed,
                reason=f"{self.strategy.name}:target",
                entry_price=bar.close,
                stop_loss=None,
                take_profit=None,
                strength=Decimal(str(abs(target))),
                features=None,
            ),
            target_exposure=target,
        )


def _side_for(target: float, previous: float) -> Side:
    """The four-event vocabulary applied to a level.

    Zero is an exit of whatever was held, which is why ``previous`` is read: a
    target of 0.0 arriving from a short is ``EXIT_SHORT``, and the same 0.0
    arriving from a long is ``EXIT_LONG``. Everything non-zero names its own
    direction, and the level beside it carries the size.
    """
    if target > 0:
        return Side.ENTER_LONG
    if target < 0:
        return Side.ENTER_SHORT
    return Side.EXIT_SHORT if previous < 0 else Side.EXIT_LONG


__all__ = [
    "DEFAULT_REBALANCE_THRESHOLD",
    "ExposureRunner",
    "require_exposure_contract",
]
