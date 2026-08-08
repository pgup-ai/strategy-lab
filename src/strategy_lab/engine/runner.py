from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

import pandas as pd

from strategy_lab.backtests.engine import ExitMode, _exit_signals
from strategy_lab.core.clock import Clock
from strategy_lab.core.types import Bar, BarEvent, BarReason, InstrumentId, Side, Signal
from strategy_lab.engine.context import BarBuffer
from strategy_lab.engine.reasons import reason_for
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

# The engine's own default, imported rather than retyped so a change there
# reaches the two paths together.
DEFAULT_FAILURE_BARS = 4


def require_signal_set_contract(strategy: object) -> None:
    """Refuse a strategy this runner cannot call, at construction (M40).

    Measured before this existed: ``StrategyRunner`` accepted
    ``state_machine_v2`` and survived **2,192 bars -- 365.3 days at 4h** before
    raising ``AttributeError``, because :meth:`StrategyRunner.on_bar` returns
    before touching the strategy while the buffer is inside warmup. A contract
    mismatch was therefore invisible for exactly as long as the warmup, and the
    deeper the warmup the longer the lie.

    Beside ``require_warmup_bars`` because it is the same rule: every
    precondition a runner depends on is checked here rather than at first use,
    since "first use" on a warmup-gated path is a year away.
    """
    if not hasattr(strategy, "generate_signals"):
        name = getattr(strategy, "name", type(strategy).__name__)
        raise TypeError(
            f"{name} has no generate_signals: it is a TargetExposure strategy, which "
            f"StrategyRunner cannot drive. Use engine.exposure_runner.ExposureRunner."
        )


def _resolve_exit_mode(exit_mode: ExitMode | str) -> ExitMode:
    """The mode this runner will apply, refusing the one no signal stream carries.

    ``setup_invalidation_stop`` is not expressible as a signal stream at all:
    ``engine._stop_kwargs`` hands ``from_signals`` an ``sl_stop`` series, and a
    stop that fires intrabar is not a bar-close decision any ``Signal`` encodes.
    Refused rather than approximated, because approximating it is how a replay
    quietly stops matching the backtest it claims to reproduce.
    """
    mode = ExitMode(exit_mode)
    if mode is ExitMode.SETUP_INVALIDATION_STOP:
        raise ValueError(
            "setup_invalidation_stop cannot be driven from a signal stream: the engine "
            "applies it as an intrabar sl_stop, which no bar-close Signal encodes."
        )
    return mode


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
    explain itself -- see ``engine.reasons.reason_for`` -- which costs a second pass over
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
        exit_mode: ExitMode | str | None = None,
        failure_bars: int = DEFAULT_FAILURE_BARS,
        allow_forming_bars: bool = False,
        record_reasons: bool = True,
    ) -> None:
        # At construction rather than on the first bar, so a live process
        # refuses to start rather than refusing partway through a session. The
        # backtest raises on the same claim; without this the vectorized path
        # would raise while replay traded from bar one, and the two paths
        # agreeing is what tests/test_replay_determinism.py exists to defend.
        require_warmup_bars(strategy.name, strategy.warmup_bars)
        require_signal_set_contract(strategy)
        self.strategy = strategy
        self.instrument = instrument
        self.timeframe = timeframe
        self.clock = clock
        self.exit_mode = None if exit_mode is None else _resolve_exit_mode(exit_mode)
        self.failure_bars = failure_bars
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
        rather than an oversight -- see ``engine.reasons.reason_for``.
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

        # A bar older than the buffer's last is dropped there -- a stale replay
        # after a reconnect -- and the buffer is then unchanged. Continuing would
        # compute from that unchanged history and stamp the result with the
        # dropped bar's timestamp: a signal for a bar the strategy never saw, and
        # now a persisted `bar_reasons` row describing one. Nothing arrived, so
        # nothing is decided.
        dropped = self.buffer.dropped_out_of_order
        self.buffer.append(bar)
        if self.buffer.dropped_out_of_order != dropped:
            return ()

        # Bar warmup_bars+1 is the first to emit. Any consumer comparing runner
        # output against a whole-history backtest has to drop the same prefix.
        if len(self.buffer) <= self.strategy.warmup_bars:
            return ()

        signal_set = self.strategy.generate_signals(self.buffer.frame())
        if self.record_reasons:
            reason = reason_for(
                self.strategy,
                bar=bar,
                frame=self.buffer.frame(),
                instrument=self.instrument,
                timeframe=self.timeframe,
                clock=self.clock,
            )
            if reason is not None:
                self._reasons[bar.ts_open_ms] = reason
        return self._extract(signal_set, bar)

    def _sides(self, signal_set: SignalSet) -> dict[str, pd.Series]:
        """The four side series this bar is read from, after the exit mode.

        Without an ``exit_mode`` these are the strategy's own, which is what this
        runner emitted before R10e and still emits by default: exit *ingredients*
        withheld, because which of them fire is an engine decision and emitting
        them unconditionally matches no single backtest configuration.

        With one, the exits come from ``engine._exit_signals`` -- **the engine's
        function, over the buffer, not a reimplementation of it**. That is M36
        reaching a third path: a cheaper route here is a fourth answer free to
        drift from the backtest, the browser and the board, and R10d measured
        exactly what the drift was worth (``trend_following_deepseek_v4`` emits no
        exits of its own, so 7,331 of 15,128 BTC bars).

        **``trend_structure`` can begin raising mid-run**, and that is the engine's
        behaviour rather than a defect here: it refuses a frame whose
        ``short_entries`` are non-empty, which on a growing buffer is a claim that
        can become true on any bar. It raises on the first bar a short entry
        appears, which is the earliest it is knowable.
        """
        if self.exit_mode is None:
            return {field: getattr(signal_set, field) for field, _ in _SIDE_BY_FIELD}

        long_exits, short_exits = _exit_signals(
            df=self.buffer.frame(),
            signals=signal_set,
            exit_mode=self.exit_mode,
            failure_bars=self.failure_bars,
        )
        return {
            "long_entries": signal_set.long_entries,
            "long_exits": long_exits.fillna(False),
            "short_entries": signal_set.short_entries,
            "short_exits": short_exits.fillna(False),
        }

    def _extract(self, signal_set: SignalSet, bar: Bar) -> Sequence[Signal]:
        """Read the last row of every side series.

        All four are read, not the first that matches: strategies that wire
        ``long_exits = short_entries`` reverse on a single bar, and collapsing
        that into one signal would lose either the flatten or the reversal.
        """
        stop_fraction = _last_float(signal_set.setup_stop_loss)
        sides = self._sides(signal_set)
        emitted: list[Signal] = []

        for field_name, side in _SIDE_BY_FIELD:
            if not bool(sides[field_name].iloc[-1]):
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
