from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

import pandas as pd

from strategy_lab.core.clock import Clock
from strategy_lab.core.types import Bar, BarEvent, InstrumentId, Side, Signal
from strategy_lab.engine.context import BarBuffer
from strategy_lab.feeds.replay import _row_to_bar
from strategy_lab.strategies.base import SignalSet, Strategy
from strategy_lab.timeframes import timeframe_to_millis

_ENTRY_SIDES = {Side.ENTER_LONG, Side.ENTER_SHORT}

_SIDE_BY_FIELD: tuple[tuple[str, Side], ...] = (
    ("long_entries", Side.ENTER_LONG),
    ("long_exits", Side.EXIT_LONG),
    ("short_entries", Side.ENTER_SHORT),
    ("short_exits", Side.EXIT_SHORT),
)


class StrategyRunner:
    """Turns a stream of bars into Signals using an unmodified vectorized strategy.

    On every closed bar it calls ``strategy.generate_signals(full_buffer)`` and
    reads the LAST row. Because the strategy is causal -- row *t* of its output
    depends only on rows <= *t* of its input -- that value equals what a
    whole-history backtest would produce for the same bar. That equality is the
    whole point: it is what makes backtest, replay, and live one code path
    instead of three implementations that drift.

    The adapter is therefore only as sound as the causality it assumes. A
    strategy that peeked forward (``shift(-1)``, ``bfill``, a centred rolling
    window) would still run here, just with different output than the backtest,
    and nothing in this class would notice. ``tests/test_lookahead.py`` is the
    check that keeps that honest.

    Cost, measured end to end on the real 83,348-bar BTC/USDT 15m set: the
    whole-history call takes 0.39 s, a full bar-by-bar replay of the same range
    takes ~43 minutes. Per bar the strategy call is ~4.3 ms, which is free live
    (one bar per timeframe interval) and quadratic in replay, since every bar
    re-runs the strategy over the whole buffer. That gap is why the backtest
    keeps its bulk path and why ``--limit-bars`` exists on ``replay``.
    """

    def __init__(
        self,
        *,
        strategy: Strategy,
        instrument: InstrumentId,
        timeframe: str,
        clock: Clock,
        allow_forming_bars: bool = False,
    ) -> None:
        self.strategy = strategy
        self.instrument = instrument
        self.timeframe = timeframe
        self.clock = clock
        self.allow_forming_bars = allow_forming_bars
        self.buffer = BarBuffer()

    def prime(self, history: pd.DataFrame) -> None:
        """Load warmup history without emitting signals.

        Rows become bars through the same ``_row_to_bar`` the replay feed uses,
        so a primed bar and a streamed bar are byte-identical for the same row --
        including ``ts_open_ms``, which must be the exact millisecond rather than
        approximately right: it is the bar's identity for dedup and for the
        replay/live equivalence check.

        Sorting is stable so that when the same timestamp appears twice, the last
        row the caller supplied is the one the buffer keeps -- the same last-wins
        rule as ``ReplayFeed._ordered`` and ``normalize_candle_frame``.
        """
        bar_ms = timeframe_to_millis(self.timeframe)
        for timestamp, row in history.sort_index(kind="stable").iterrows():
            self.buffer.append(_row_to_bar(timestamp, row, self.instrument, self.timeframe, bar_ms))

    def on_event(self, event: BarEvent) -> Sequence[Signal]:
        return self.on_bar(event.bar)

    def on_bar(self, bar: Bar) -> Sequence[Signal]:
        if not bar.is_closed and not self.allow_forming_bars:
            return ()

        # Duck-typed on purpose: SimClock advances from event time so a replay is
        # reproducible, LiveClock has no advance_to and keeps reading the wall
        # clock. The Clock protocol only promises now_ms, so this stays a
        # capability check rather than an isinstance test that would reject test
        # doubles and any future deterministic clock.
        if hasattr(self.clock, "advance_to"):
            self.clock.advance_to(bar.ts_close_ms)

        self.buffer.append(bar)
        # Bar N is the first to emit, where N = warmup_bars + 1: a strategy
        # declaring 4000 warmup bars is trusted from bar 4001 on. Any consumer
        # comparing runner output against a whole-history backtest has to drop
        # the same prefix.
        if len(self.buffer) <= self.strategy.warmup_bars:
            return ()

        signal_set = self.strategy.generate_signals(self.buffer.frame())
        return self._extract(signal_set, bar)

    def _extract(self, signal_set: SignalSet, bar: Bar) -> Sequence[Signal]:
        """Read the last row of every side series.

        All four are read, not the first that matches: strategies that wire
        ``long_exits = short_entries`` reverse on a single bar, and collapsing
        that into one signal would lose either the flatten or the reversal.
        """
        stop_fraction = _last_float(signal_set.setup_stop_loss)
        emitted: list[Signal] = []

        for field_name, side in _SIDE_BY_FIELD:
            series = getattr(signal_set, field_name, None)
            if series is None or not bool(series.iloc[-1]):
                continue
            emitted.append(
                Signal(
                    instrument=self.instrument,
                    timeframe=self.timeframe,
                    strategy_id=self.strategy.name,
                    strategy_version=self.strategy.version,
                    ts_bar_ms=bar.ts_open_ms,
                    ts_emit_ms=self.clock.now_ms(),
                    side=side,
                    bar_is_closed=bar.is_closed,
                    reason=f"{self.strategy.name}:{field_name}",
                    entry_price=bar.close,
                    stop_loss=_stop_price(bar.close, stop_fraction, side),
                    strength=None,
                    features=_features(signal_set),
                )
            )
        return tuple(emitted)


def _last_float(series: pd.Series | None) -> float | None:
    if series is None or series.empty:
        return None
    value = series.iloc[-1]
    return None if pd.isna(value) else float(value)


def _stop_price(close: Decimal, fraction: float | None, side: Side) -> Decimal | None:
    """``setup_stop_loss`` is a fraction of price; a Signal carries a price.

    ``setup_invalidation_stop_loss`` returns ``(close - setup_low) / close`` for a
    long and ``(setup_high - close) / close`` for a short -- both positive -- so
    the direction lives here, not in the number. Only entries carry a stop; an
    exit signal already closes the position.
    """
    if fraction is None or side not in _ENTRY_SIDES:
        return None
    offset = Decimal(str(fraction))
    if side is Side.ENTER_LONG:
        return close * (Decimal(1) - offset)
    return close * (Decimal(1) + offset)


def _features(signal_set: SignalSet) -> dict:
    """Strategy metadata, stringified for the JSONB column.

    Stringifying is what guarantees the payload is JSON-serialisable --
    ``write_signals`` raises on anything that is not -- but it is lossy: ``True``
    comes back as ``"True"`` and ``200`` as ``"200"``. Read these as labels, not
    as values to compute or branch on.
    """
    return {key: str(value) for key, value in (signal_set.metadata or {}).items()}
