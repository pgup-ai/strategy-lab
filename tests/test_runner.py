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

    Priming rather than streaming the warmup is what keeps these tests cheap: the
    runner re-runs ``generate_signals`` over the whole buffer for every bar it does
    not suppress, so streaming a 4,000-bar warmup buys 4,000 passes that emit
    nothing. It is also what a live process does with its warmup fetch.

    Frames are sized from ``strategy.warmup_bars`` rather than fixed, so raising a
    strategy's warmup can never silently leave these tests with no signals.
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


def test_first_emitting_bar_is_the_one_after_warmup():
    """Pins the boundary itself: with warmup_bars=3, bars 1-3 are suppressed and
    bar 4 is the first to emit. Any consumer comparing runner output against a
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


def test_emitted_signals_are_stamped_with_the_bar_close_not_wall_time():
    """The runner advances SimClock from event time before emitting, which is what
    makes a replay's emission times reproducible instead of wall-clock noise."""
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


def test_a_strategy_with_no_feature_frame_records_no_reasons():
    """An absent explanation and an empty one are different claims.

    The four original strategies have no state to explain, and a row of nulls for
    each of their bars would read as "the machine looked and saw nothing" rather
    than "there is no machine". This is the same rule ``api/analysis._why_layer``
    follows by returning ``None``.
    """
    strategy = _AlwaysLong()
    runner = make_runner(strategy)
    bars = bars_from(synthetic_ohlcv(n=10))
    for bar in bars:
        runner.on_bar(bar)

    # Bars suppressed by warmup record nothing either, so without this the
    # assertion below would hold for a strategy that never reached the recording
    # block at all -- passing for the wrong reason if the warmup ever grew.
    assert len(bars) > strategy.warmup_bars
    assert runner.reasons == ()


def test_one_reason_per_bar_past_warmup():
    """The count is the emitting-bar count, and the boundary is the signal
    boundary: a reason for a bar the runner suppressed would describe a decision
    no consumer of this runner ever saw."""
    strategy = get_strategy("state_machine_v1")
    df = synthetic_ohlcv(n=strategy.warmup_bars + 20)
    runner = make_runner(strategy)
    runner.prime(df.iloc[: strategy.warmup_bars])
    for bar in bars_from(df.iloc[strategy.warmup_bars :]):
        runner.on_bar(bar)

    streamed = df.index[strategy.warmup_bars :]
    assert len(runner.reasons) == len(streamed)
    assert [r.ts_bar_ms for r in runner.reasons] == [
        ts.value // 1_000_000 for ts in streamed
    ]
    assert {r.strategy_id for r in runner.reasons} == {"state_machine_v1"}
    assert set(runner.reasons[0].features) == set(strategy.features)


def test_a_recorded_reason_matches_the_whole_history_why_layer():
    """The bar-by-bar reason equals what the research path recomputes for that bar.

    Same claim ``tests/test_replay_determinism.py`` makes for signals, and the
    same causality it rests on: the machine walks forward, so the last row of a
    run over the first *t* bars is row *t* of a run over all of them. This is
    also the harness check behind the live/research diff -- a diff that shows
    nothing is only good news if this passes, because otherwise both sides are
    just failing to look.
    """
    from strategy_lab.api.analysis import _why_layer

    strategy = get_strategy("state_machine_v1")
    df = synthetic_ohlcv(n=strategy.warmup_bars + 20)
    runner = make_runner(strategy)
    runner.prime(df.iloc[: strategy.warmup_bars])
    for bar in bars_from(df.iloc[strategy.warmup_bars :]):
        runner.on_bar(bar)

    why, _ = _why_layer(strategy, df)
    for offset, reason in enumerate(runner.reasons):
        row = strategy.warmup_bars + offset
        assert reason.state == why.states[row]
        for name, value in reason.features.items():
            assert value == why.features[name][row], f"{name} at row {row}"


def test_a_redelivered_bar_replaces_its_reason_rather_than_adding_one():
    """``BarBuffer`` takes the corrected copy of a repeated timestamp; the reason
    for that bar has to follow it, or the stored row describes a bar the strategy
    no longer holds. One bar, one state -- so unlike ``signals`` this is
    expressible, and it is what keeps the row count the bar count."""
    strategy = get_strategy("state_machine_v1")
    df = synthetic_ohlcv(n=strategy.warmup_bars + 3)
    runner = make_runner(strategy)
    runner.prime(df.iloc[: strategy.warmup_bars])
    bars = bars_from(df.iloc[strategy.warmup_bars :])
    for bar in bars:
        runner.on_bar(bar)

    before = runner.reasons[-1]
    corrected = replace(bars[-1], close=bars[-1].close * Decimal("1.05"))
    runner.on_bar(corrected)

    assert len(runner.reasons) == len(bars)
    assert len(runner.buffer) == len(df)
    assert runner.reasons[-1].ts_bar_ms == corrected.ts_open_ms
    # The timestamp is what the correction leaves alone, so it cannot tell a
    # replaced reason from a kept one -- the values are what moved.
    assert runner.reasons[-1].features != before.features


def test_record_reasons_false_skips_the_second_computation():
    """The opt-out exists because the reason layer re-runs the feature frame and
    the state walk the ``SignalSet`` does not carry, which roughly doubles the
    per-bar cost. Signals are unaffected either way."""
    strategy = get_strategy("state_machine_v1")
    df = synthetic_ohlcv(n=strategy.warmup_bars + 5)
    with_reasons = make_runner(strategy)
    without = make_runner(strategy, record_reasons=False)
    emitted = {}
    for runner in (with_reasons, without):
        runner.prime(df.iloc[: strategy.warmup_bars])
        emitted[runner] = [
            signal
            for bar in bars_from(df.iloc[strategy.warmup_bars :])
            for signal in runner.on_bar(bar)
        ]

    assert without.reasons == ()
    assert len(with_reasons.reasons) == 5
    assert emitted[with_reasons] == emitted[without]


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


def test_a_stale_bar_decides_nothing_and_records_nothing():
    """``BarBuffer`` drops a bar older than its last, leaving the history unchanged.

    Continuing past that would compute from the unchanged buffer and stamp the
    result with the dropped bar's timestamp -- a signal for a bar the strategy
    never saw, and a ``bar_reasons`` row describing one. The row is the reason
    this now returns early: signals were already wrong here, but nothing
    persisted them.
    """
    strategy = get_strategy("state_machine_v1")
    df = synthetic_ohlcv(n=strategy.warmup_bars + 4)
    runner = make_runner(strategy)
    runner.prime(df.iloc[: strategy.warmup_bars])
    bars = bars_from(df.iloc[strategy.warmup_bars :])
    for bar in bars:
        runner.on_bar(bar)

    recorded = {reason.ts_bar_ms for reason in runner.reasons}
    buffered = len(runner.buffer)

    # From the *primed* history, so its timestamp has never been recorded -- a
    # stale bar the runner has already streamed would merely overwrite its own
    # row, which cannot tell a working guard from a missing one.
    stale = bars_from(df.iloc[: strategy.warmup_bars])[0]
    assert stale.ts_open_ms not in recorded

    assert runner.on_bar(stale) == ()
    assert runner.buffer.dropped_out_of_order == 1
    assert len(runner.buffer) == buffered
    assert {reason.ts_bar_ms for reason in runner.reasons} == recorded
