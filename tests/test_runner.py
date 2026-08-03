from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal

import pandas as pd

from strategy_lab.core.clock import SimClock
from strategy_lab.core.types import InstrumentId, Side
from strategy_lab.engine.runner import StrategyRunner
from strategy_lab.feeds.replay import _row_to_bar
from strategy_lab.strategies.base import SignalSet
from strategy_lab.strategies.registry import get_strategy
from tests.conftest import synthetic_ohlcv

INSTRUMENT = InstrumentId("binance", "perp", "BTC/USDT")
BAR_MS = 15 * 60 * 1000


def bars_from(df):
    return [_row_to_bar(ts, row, INSTRUMENT, "15m", BAR_MS) for ts, row in df.iterrows()]


def make_runner(strategy, **kwargs):
    return StrategyRunner(
        strategy=strategy,
        instrument=INSTRUMENT,
        timeframe="15m",
        clock=SimClock(),
        **kwargs,
    )


def emit_past_warmup(strategy, span: int = 300):
    """Signals from ``span`` bars streamed after a primed warmup.

    Priming the warmup rather than streaming it is what keeps these tests cheap.
    The runner re-runs ``generate_signals`` over the entire buffer for every bar
    it does not suppress, so streaming a 4,000-bar warmup would buy 4,000 extra
    whole-history passes that emit nothing. ``prime`` fills the same buffer in
    bulk -- and it is what a live process does with its warmup fetch, so this is
    also closer to production than streaming the warmup ever was.

    Frames are sized from ``strategy.warmup_bars`` rather than fixed, so raising
    a strategy's warmup can never silently leave these tests with no signals.
    """
    df = synthetic_ohlcv(n=strategy.warmup_bars + span)
    runner = make_runner(strategy)
    runner.prime(df.iloc[: strategy.warmup_bars])
    emitted = []
    for bar in bars_from(df.iloc[strategy.warmup_bars :]):
        emitted.extend(runner.on_bar(bar))
    return emitted


@dataclass(frozen=True)
class _AlwaysLong:
    name: str = "always_long"
    version: str = "1.0.0"
    warmup_bars: int = 3

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        true_series = pd.Series(True, index=df.index)
        flat = pd.Series(False, index=df.index)
        return SignalSet(true_series, flat, flat, flat)


def test_runner_suppresses_signals_during_warmup():
    runner = make_runner(_AlwaysLong())
    emitted = []
    for bar in bars_from(synthetic_ohlcv(n=6)):
        emitted.extend(runner.on_bar(bar))

    # warmup_bars=3 -> bars 1,2,3 suppressed; 4,5,6 emit.
    assert len(emitted) == 3


def test_first_emitting_bar_is_the_one_after_warmup():
    """Pins the boundary itself: any consumer comparing runner output against a
    whole-history backtest must drop exactly the same prefix."""
    bars = bars_from(synthetic_ohlcv(n=6))
    runner = make_runner(_AlwaysLong())
    emitted = []
    for bar in bars:
        emitted.extend(runner.on_bar(bar))

    assert [signal.ts_bar_ms for signal in emitted] == [bar.ts_open_ms for bar in bars[3:]]


def test_runner_ignores_forming_bars_by_default():
    runner = make_runner(_AlwaysLong())
    bars = bars_from(synthetic_ohlcv(n=6))
    emitted = []
    for bar in bars:
        emitted.extend(runner.on_bar(replace(bar, is_closed=False)))

    assert emitted == []


def test_forming_bars_are_marked_and_replaced_by_the_closed_bar():
    """allow_forming_bars=True emits provisional signals; the closed bar for the
    same timestamp must overwrite the forming one rather than extend history."""
    bars = bars_from(synthetic_ohlcv(n=5))
    runner = make_runner(_AlwaysLong(), allow_forming_bars=True)
    for bar in bars[:4]:
        runner.on_bar(bar)

    forming = replace(bars[4], is_closed=False, close=Decimal("1234"))
    provisional = runner.on_bar(forming)
    final = runner.on_bar(bars[4])

    assert [signal.bar_is_closed for signal in provisional] == [False]
    assert [signal.bar_is_closed for signal in final] == [True]
    assert len(runner.buffer) == 5
    assert runner.buffer.frame()["close"].iloc[-1] == float(bars[4].close)


def test_runner_stamps_strategy_identity_and_bar_time():
    runner = make_runner(_AlwaysLong())
    bars = bars_from(synthetic_ohlcv(n=5))
    emitted = []
    for bar in bars:
        emitted.extend(runner.on_bar(bar))

    signal = emitted[0]
    assert signal.strategy_id == "always_long"
    assert signal.strategy_version == "1.0.0"
    assert signal.side is Side.ENTER_LONG
    assert signal.bar_is_closed is True
    assert signal.ts_bar_ms == bars[3].ts_open_ms
    assert signal.entry_price == bars[3].close


def test_runner_converts_stop_fraction_to_an_absolute_price():
    """turnaround_v1 reports the stop as a fraction of price; signals carry a price."""
    emitted = emit_past_warmup(get_strategy("turnaround_v1"))

    entries = [s for s in emitted if s.side is Side.ENTER_LONG and s.stop_loss is not None]
    assert entries, "expected at least one long entry with a stop"
    for signal in entries:
        assert isinstance(signal.stop_loss, Decimal)
        assert Decimal("0") < signal.stop_loss < signal.entry_price


def test_short_entry_stop_sits_above_the_entry_price():
    """The fraction is positive for both directions, so only the sign here
    distinguishes a short stop from a long one -- and an inverted sign would
    otherwise look exactly like a valid stop."""
    emitted = emit_past_warmup(get_strategy("turnaround_v1"))

    entries = [s for s in emitted if s.side is Side.ENTER_SHORT and s.stop_loss is not None]
    assert entries, "expected at least one short entry with a stop"
    for signal in entries:
        assert isinstance(signal.stop_loss, Decimal)
        assert signal.stop_loss > signal.entry_price


def test_exit_signals_carry_no_stop():
    """A stop protects an open position; an exit already closes it."""
    emitted = emit_past_warmup(get_strategy("turnaround_v1"))

    exits = [s for s in emitted if s.side in (Side.EXIT_LONG, Side.EXIT_SHORT)]
    assert exits, "expected at least one exit"
    assert all(signal.stop_loss is None for signal in exits)


def test_runner_emits_both_sides_when_a_bar_exits_long_and_enters_short():
    """turnaround_v1 wires long_exits = short_entries, so one bar can do both."""
    per_bar: dict[int, set[Side]] = {}
    for signal in emit_past_warmup(get_strategy("turnaround_v1")):
        per_bar.setdefault(signal.ts_bar_ms, set()).add(signal.side)

    assert any(
        {Side.EXIT_LONG, Side.ENTER_SHORT} <= sides for sides in per_bar.values()
    ), "expected a bar emitting both exit_long and enter_short"


def test_runner_advances_the_clock_from_event_time():
    clock = SimClock()
    runner = StrategyRunner(
        strategy=_AlwaysLong(), instrument=INSTRUMENT, timeframe="15m", clock=clock
    )
    bars = bars_from(synthetic_ohlcv(n=5))
    for bar in bars:
        runner.on_bar(bar)

    assert clock.now_ms() == bars[-1].ts_close_ms


def test_emitted_signals_are_stamped_with_the_bar_close_not_wall_time():
    runner = make_runner(_AlwaysLong())
    bars = bars_from(synthetic_ohlcv(n=5))
    emitted = []
    for bar in bars:
        emitted.extend(runner.on_bar(bar))

    assert [s.ts_emit_ms for s in emitted] == [bar.ts_close_ms for bar in bars[3:]]


def test_priming_fills_history_without_emitting():
    df = synthetic_ohlcv(n=10)
    runner = make_runner(_AlwaysLong())
    runner.prime(df)

    assert len(runner.buffer) == 10
    pd.testing.assert_frame_equal(runner.buffer.frame(), df[list(df.columns)], check_freq=False)
    # History is already past warmup, so the next streamed bar emits immediately.
    next_bar = bars_from(synthetic_ohlcv(n=11))[10]
    assert len(runner.on_bar(next_bar)) == 1


def test_priming_preserves_the_exact_millisecond():
    """int(Timestamp.timestamp() * 1000) lands 1 ms low on ~0.7% of sub-second
    timestamps; ts_open_ms is a bar's identity, so it may not be approximate."""
    off_by_one_ms = 1_082_922_440_498  # int(ts.timestamp() * 1000) gives ...497
    index = pd.DatetimeIndex([pd.Timestamp(off_by_one_ms, unit="ms", tz="UTC")], name="timestamp")
    history = pd.DataFrame(
        {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [10.0]},
        index=index,
    )

    runner = make_runner(_AlwaysLong())
    runner.prime(history)

    stored = runner.buffer.frame().index[0]
    assert stored.value // 1_000_000 == off_by_one_ms


def test_priming_then_streaming_the_same_bar_does_not_duplicate_it():
    """Warmup history and the live stream overlap in practice; the overlap has to
    collapse onto one bar or every indicator sees a repeated candle."""
    df = synthetic_ohlcv(n=10)
    runner = make_runner(_AlwaysLong())
    runner.prime(df)
    runner.on_bar(bars_from(df)[-1])

    assert len(runner.buffer) == 10
    assert runner.buffer.replaced_duplicates == 1


def test_on_event_and_on_bar_agree():
    from strategy_lab.core.types import BarEvent

    bars = bars_from(synthetic_ohlcv(n=5))
    by_bar = make_runner(_AlwaysLong())
    by_event = make_runner(_AlwaysLong())

    direct = [s for bar in bars for s in by_bar.on_bar(bar)]
    evented = [
        s
        for bar in bars
        for s in by_event.on_event(BarEvent(bar=bar, ts_event_ms=bar.ts_close_ms))
    ]
    assert direct == evented


def test_features_are_json_ready_strings():
    """The JSONB column rejects anything non-serialisable, so metadata is
    stringified -- lossily: booleans and ints come back as text."""
    import json

    emitted = emit_past_warmup(get_strategy("turnaround_v1"))
    assert emitted, "no signals emitted; nothing to inspect"

    features = emitted[0].features
    assert features["allow_shorts"] == "True"
    assert features["trend_failure_ema_span"] == "200"
    assert json.loads(json.dumps(features)) == features
