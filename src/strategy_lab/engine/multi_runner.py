from __future__ import annotations

from collections.abc import Sequence

from strategy_lab.core.clock import Clock
from strategy_lab.core.types import BarEvent, InstrumentId, MarketSnapshot, Signal
from strategy_lab.engine.context import BarBuffer
from strategy_lab.engine.market_clock import MarketClock
from strategy_lab.engine.runner import StrategyRunner
from strategy_lab.strategies.base import Strategy


class MultiAssetRunner:
    """Drives one strategy per instrument off a single merged, time-ordered stream.

    Dispatch waits for :class:`MarketClock` to declare a timestamp complete, so a
    strategy evaluating bar *t* reads the cross-section at *t* through
    :meth:`snapshot` rather than a stale one from *t-1*. The price is that signals
    for *t* are emitted only once the first *t+1* event arrives; see ``MarketClock``
    for why that lag is the only completeness a live feed can establish.

    Each instrument gets its own ``BarBuffer`` holding full history, for the same
    reason ``StrategyRunner`` does: ``ewm(adjust=False)`` is recursive from the
    first bar, so a windowed buffer would silently change the indicator.

    There is deliberately no ``allow_forming_bars``: a snapshot is defined as the
    bars that *closed* at one event time, and a provisional price inside a
    completed cross-section is not a cross-section.
    """

    def __init__(
        self,
        *,
        strategies: dict[InstrumentId, Strategy],
        timeframe: str,
        clock: Clock,
        context: set[InstrumentId] | None = None,
    ) -> None:
        self.timeframe = timeframe
        self.clock = clock
        self._runners = {
            instrument: StrategyRunner(
                strategy=strategy, instrument=instrument, timeframe=timeframe, clock=clock
            )
            for instrument, strategy in strategies.items()
        }
        # Traded instruments overwrite context ones, so an instrument named in both
        # is buffered once -- by the runner that actually trades it.
        self._buffers: dict[InstrumentId, BarBuffer] = {
            instrument: BarBuffer() for instrument in context or ()
        }
        self._buffers.update(
            (instrument, runner.buffer) for instrument, runner in self._runners.items()
        )
        self._market_clock = MarketClock()
        self._snapshot: MarketSnapshot | None = None

    def buffer(self, instrument: InstrumentId) -> BarBuffer:
        self._require(instrument)
        return self._buffers[instrument]

    def snapshot(self) -> MarketSnapshot | None:
        """The most recently completed cross-section; None until the first completes."""
        return self._snapshot

    def on_event(self, event: BarEvent) -> Sequence[Signal]:
        if not event.bar.is_closed:
            return ()
        self._require(event.bar.instrument)

        completed = self._market_clock.on_event(event)
        return () if completed is None else self._release(completed)

    def flush(self) -> Sequence[Signal]:
        """Release the final timestamp, which has no successor to complete it."""
        completed = self._market_clock.flush()
        return () if completed is None else self._release(completed)

    def _release(self, snapshot: MarketSnapshot) -> Sequence[Signal]:
        # Publish before dispatching: that ordering is what lets a strategy read
        # its own bar's cross-section rather than the previous bar's.
        self._snapshot = snapshot
        emitted: list[Signal] = []
        for instrument, bar in snapshot.bars.items():
            runner = self._runners.get(instrument)
            if runner is None:
                self._buffers[instrument].append(bar)
            else:
                emitted.extend(runner.on_bar(bar))
        return tuple(emitted)

    def _require(self, instrument: InstrumentId) -> None:
        if instrument not in self._buffers:
            raise KeyError(f"{instrument.key} has no strategy and was not declared as context")
