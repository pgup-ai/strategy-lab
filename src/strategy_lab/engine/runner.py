from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

import pandas as pd

from strategy_lab.core.clock import Clock
from strategy_lab.core.types import Bar, BarEvent, BarReason, InstrumentId, Side, Signal
from strategy_lab.engine.context import BarBuffer
from strategy_lab.feeds.replay import _row_to_bar
from strategy_lab.strategies.base import SignalSet, Strategy, require_warmup_bars
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
    whole-history backtest would produce for the same bar, which is what makes
    backtest, replay, and live one code path instead of three that drift.

    The adapter is therefore only as sound as the causality it assumes. A strategy
    that peeked forward (``shift(-1)``, ``bfill``, a centred rolling window) would
    still run here, just disagreeing with the backtest, and nothing in this class
    would notice. ``tests/test_lookahead.py`` keeps that honest.

    Cost, measured end to end on the real 83,348-bar BTC/USDT 15m set: the
    whole-history call takes 0.39 s, a full bar-by-bar replay of the same range
    ~43 minutes. Per bar that is ~4.3 ms, free live (one bar per timeframe
    interval) but quadratic in replay, since every bar re-runs the strategy over
    the whole buffer. That gap is why the backtest keeps its bulk path and why
    ``--limit-bars`` exists on ``replay``.

    It also accumulates a :class:`BarReason` per bar for any strategy that can
    explain itself -- see :meth:`_reason_for` -- which costs a second pass over
    the buffer, because the feature frame and the state walk are recomputed
    rather than read out of the ``SignalSet``, which carries neither. Measured on
    ``state_machine_v1`` over 1,000 emitting bars of BTC/USDT perp 4h: **6.8 s
    off, 11.5 s on**, so 1.7x rather than the 2x the second pass suggests -- the
    policy and the signal derivation are in the first pass only.
    ``record_reasons=False`` is the way out for a caller that only wants signals;
    it is not the default, because the whole point of the event path is that it
    is the only one whose reasons cannot be recomputed later.
    """

    def __init__(
        self,
        *,
        strategy: Strategy,
        instrument: InstrumentId,
        timeframe: str,
        clock: Clock,
        allow_forming_bars: bool = False,
        record_reasons: bool = True,
    ) -> None:
        # At construction rather than on the first bar, so a live process
        # refuses to start rather than refusing partway through a session. The
        # backtest raises on the same claim; without this the vectorized path
        # would raise while replay traded from bar one, and the two paths
        # agreeing is what tests/test_replay_determinism.py exists to defend.
        require_warmup_bars(strategy.name, strategy.warmup_bars)
        self.strategy = strategy
        self.instrument = instrument
        self.timeframe = timeframe
        self.clock = clock
        self.allow_forming_bars = allow_forming_bars
        self.record_reasons = record_reasons
        self.buffer = BarBuffer()
        # Keyed by bar so a redelivered bar replaces its reason rather than
        # adding one, the same last-wins rule ``BarBuffer.append`` applies to the
        # bar itself: the corrected copy is the one the strategy really read.
        # ``signals`` cannot do that -- one bar legitimately emits several sides
        # -- but a bar has exactly one state, so here it is expressible and the
        # per-bar row count stays the bar count by construction.
        self._reasons: dict[int, BarReason] = {}

    @property
    def reasons(self) -> tuple[BarReason, ...]:
        """One reason per bar seen past warmup, in bar order.

        Empty for a strategy that cannot explain itself, and that is a claim
        rather than an oversight -- see :meth:`_reason_for`.
        """
        return tuple(self._reasons.values())

    def prime(self, history: pd.DataFrame) -> None:
        """Load warmup history without emitting signals.

        Rows become bars through the same ``_row_to_bar`` the replay feed uses, so
        a primed bar and a streamed bar are byte-identical for the same row --
        including ``ts_open_ms``, which is the bar's identity for dedup and so must
        be the exact millisecond rather than approximately right.

        Sorting is stable so that a repeated timestamp keeps the last row the
        caller supplied -- the same last-wins rule as ``ReplayFeed._ordered`` and
        ``normalize_candle_frame``.
        """
        bar_ms = timeframe_to_millis(self.timeframe)
        for timestamp, row in history.sort_index(kind="stable").iterrows():
            self.buffer.append(_row_to_bar(timestamp, row, self.instrument, self.timeframe, bar_ms))

    def on_event(self, event: BarEvent) -> Sequence[Signal]:
        return self.on_bar(event.bar)

    def on_bar(self, bar: Bar) -> Sequence[Signal]:
        if not bar.is_closed and not self.allow_forming_bars:
            return ()

        # A capability check, not an isinstance test: SimClock advances from event
        # time so a replay is reproducible, LiveClock has no advance_to and keeps
        # reading the wall clock. The Clock protocol only promises now_ms, so
        # isinstance here would reject test doubles and any future clock.
        if hasattr(self.clock, "advance_to"):
            self.clock.advance_to(bar.ts_close_ms)

        self.buffer.append(bar)
        # Bar warmup_bars+1 is the first to emit. Any consumer comparing runner
        # output against a whole-history backtest has to drop the same prefix.
        if len(self.buffer) <= self.strategy.warmup_bars:
            return ()

        signal_set = self.strategy.generate_signals(self.buffer.frame())
        if self.record_reasons:
            reason = self._reason_for(bar)
            if reason is not None:
                self._reasons[bar.ts_open_ms] = reason
        return self._extract(signal_set, bar)

    def _reason_for(self, bar: Bar) -> BarReason | None:
        """The state and feature values behind this bar, for a strategy that has them.

        Found by ``getattr(strategy, "feature_frame"/"machine")`` -- the same
        introspection ``api/analysis._why_layer`` uses, deliberately rather than
        matching a list of state machines, so a strategy that grows a
        ``feature_frame`` later is explained on both paths without anyone editing
        either. A strategy with neither returns ``None`` and writes **no** rows:
        an absent explanation and an empty one are different claims, and the four
        original strategies genuinely have nothing to say here.

        Recomputed rather than read off the ``generate_signals`` call above,
        because a ``SignalSet`` carries entries, exits and a size and none of the
        state behind them. The two are still one computation in the sense that
        matters -- the same frame, the same feature functions, the same machine,
        and the last row of each -- and the only alternative is a strategy-side
        return-value change, which would touch every strategy in the repo to
        serve the one path that cannot recompute.
        """
        feature_frame = getattr(self.strategy, "feature_frame", None)
        machine = getattr(self.strategy, "machine", None)
        if feature_frame is None or machine is None:
            return None

        frame, _ = feature_frame(self.buffer.frame())
        # The machine answers on every bar, warmup included: an unmeasurable row
        # is a failure to it rather than a gap, so there is no missing state.
        state = machine.run(frame).iloc[-1]
        return BarReason(
            instrument=self.instrument,
            timeframe=self.timeframe,
            strategy_id=self.strategy.name,
            strategy_version=self.strategy.version,
            ts_bar_ms=bar.ts_open_ms,
            ts_emit_ms=self.clock.now_ms(),
            bar_is_closed=bar.is_closed,
            state=str(state.value),
            features={str(name): _last_float(frame[name]) for name in frame.columns},
        )

    def _extract(self, signal_set: SignalSet, bar: Bar) -> Sequence[Signal]:
        """Read the last row of every side series.

        All four are read, not the first that matches: strategies that wire
        ``long_exits = short_entries`` reverse on a single bar, and collapsing
        that into one signal would lose either the flatten or the reversal.

        The remaining ``SignalSet`` fields -- ``trend_failure_long_exits``,
        ``trend_failure_short_exits``, ``position_size`` -- are deliberately not
        emitted. They are exit *ingredients*, and which of them fire is an
        ``ExitMode`` decision the backtest engine makes, not a property of the
        strategy: ``trend_following_deepseek_v4`` needs ``trend_structure`` while
        ``trend_rider_v1_deepseek_v4_pro`` needs ``opposite_signal_only``.
        Emitting them unconditionally here would produce a signal stream matching
        no single backtest configuration. The runner gains an ``ExitMode`` in
        Phase 1b, when signals start driving positions rather than being recorded.
        """
        stop_fraction = _last_float(signal_set.setup_stop_loss)
        emitted: list[Signal] = []

        for field_name, side in _SIDE_BY_FIELD:
            if not bool(getattr(signal_set, field_name).iloc[-1]):
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
