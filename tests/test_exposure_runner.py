"""``ExposureRunner``'s own behaviour, beside the determinism proof that drives it.

``tests/test_exposure_determinism.py`` runs this class over real strategies and
asserts target-level equality against a whole-history run -- with the band opened
to 0.0, so it compares *every* bar. What it therefore cannot test is the band
itself, the side vocabulary, or the refusals. Those are here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from strategy_lab.core.clock import SimClock
from strategy_lab.core.types import InstrumentId, Side
from strategy_lab.engine.exposure_runner import ExposureRunner
from strategy_lab.feeds.replay import _row_to_bar
from strategy_lab.strategies.exposure import TargetExposure
from strategy_lab.strategies.registry import get_strategy
from strategy_lab.timeframes import timeframe_to_millis
from tests.conftest import synthetic_ohlcv

TIMEFRAME = "4h"
INSTRUMENT = InstrumentId("binance", "perp", "BTC/USDT")


@dataclass(frozen=True)
class _Scripted:
    """A strategy whose target is whatever the test says, bar for bar.

    The band is a rule about a *sequence*, so testing it against a market-derived
    target would test the market. Warmup is 0 because there is nothing to warm.
    """

    targets: tuple[float, ...]
    name: str = "scripted"
    version: str = "1.0.0"
    warmup_bars: int = 0

    def compute_target(self, df: pd.DataFrame) -> TargetExposure:
        values = list(self.targets[: len(df)])
        values += [values[-1] if values else 0.0] * (len(df) - len(values))
        return TargetExposure(target=pd.Series(values, index=df.index, dtype="float64"))


def drive(strategy, n: int, **kwargs) -> list:
    runner = ExposureRunner(
        strategy=strategy,
        instrument=INSTRUMENT,
        timeframe=TIMEFRAME,
        clock=SimClock(),
        record_reasons=False,
        **kwargs,
    )
    df = synthetic_ohlcv(n=n, seed=4)
    bar_ms = timeframe_to_millis(TIMEFRAME)
    emitted = []
    for timestamp, row in df.iterrows():
        bar = _row_to_bar(timestamp, row, INSTRUMENT, TIMEFRAME, bar_ms)
        emitted.extend(runner.on_bar(bar))
    return emitted


def test_the_band_measures_against_the_last_target_submitted_not_the_last_bar():
    """The engine's rule, and the reason it is that rule.

    A taper giving up 2% of the budget per bar never moves a whole 5% band in one
    bar, so a bar-to-bar reference would hold the opening position through the
    entire taper and submit **once**. Against the last submission the moves
    accumulate and it decides on every third bar.
    """
    ramp = tuple(round(1.0 - 0.02 * i, 4) for i in range(12))
    emitted = drive(_Scripted(targets=ramp), n=12, rebalance_threshold=0.05)

    levels = [float(signal.target_exposure) for signal in emitted]
    assert levels == [1.0, 0.94, 0.88, 0.82]


def test_a_zero_band_submits_on_every_bar():
    """What the determinism suite relies on to compare bar for bar, and the
    engine's own reading of ``rebalance_threshold=0.0``."""
    ramp = tuple(round(1.0 - 0.02 * i, 4) for i in range(6))
    emitted = drive(_Scripted(targets=ramp), n=6, rebalance_threshold=0.0)

    assert [float(signal.target_exposure) for signal in emitted] == list(ramp)


def test_a_move_smaller_than_the_band_reaches_the_book_on_no_bar():
    emitted = drive(_Scripted(targets=(0.5, 0.52, 0.5, 0.53)), n=4, rebalance_threshold=0.05)

    assert [float(signal.target_exposure) for signal in emitted] == [0.5]


def test_a_level_names_its_own_direction_and_zero_exits_what_was_held():
    """``side`` is the direction of the level, not an event -- and zero is the one
    value that needs the previous one to be readable at all."""
    long_then_flat = drive(_Scripted(targets=(0.6, 0.0)), n=2, rebalance_threshold=0.05)
    short_then_flat = drive(_Scripted(targets=(-0.6, 0.0)), n=2, rebalance_threshold=0.05)

    assert [s.side for s in long_then_flat] == [Side.ENTER_LONG, Side.EXIT_LONG]
    assert [s.side for s in short_then_flat] == [Side.ENTER_SHORT, Side.EXIT_SHORT]


def test_a_flip_through_zero_is_one_row_because_the_level_is_the_whole_statement():
    """Where a boolean reversal needs two signals -- flatten, then enter -- a
    target of -0.5 already says what the book holds."""
    emitted = drive(_Scripted(targets=(0.5, -0.5)), n=2, rebalance_threshold=0.05)

    assert [(s.side, float(s.target_exposure)) for s in emitted] == [
        (Side.ENTER_LONG, 0.5),
        (Side.ENTER_SHORT, -0.5),
    ]


def test_a_target_sitting_at_zero_submits_nothing():
    """The band opens against 0.0 because the book starts flat, so the leading
    zeros a warmup produces cost no rows. Only true for a non-zero band: at 0.0
    the engine submits every bar too, which is what the determinism suite uses."""
    emitted = drive(_Scripted(targets=(0.0,) * 5 + (0.8,)), n=6, rebalance_threshold=0.05)

    assert [float(s.target_exposure) for s in emitted] == [0.8]


def test_nothing_is_submitted_through_warmup_whatever_the_target_says():
    """Warmup suppression is the runner's, not the target's: a strategy asking for
    0.9 from bar zero still submits nothing until its declared warmup elapses,
    which is the boundary ``whole_history_targets`` mirrors bar for bar."""
    emitted = drive(
        _Scripted(targets=(0.9,) * 6, warmup_bars=3), n=6, rebalance_threshold=0.0
    )

    assert len(emitted) == 3


def test_a_bar_older_than_the_buffer_submits_nothing():
    """The buffer drops it and is unchanged, so continuing would submit a target
    computed from history that bar never joined, stamped with its timestamp."""
    runner = ExposureRunner(
        strategy=_Scripted(targets=(0.9,) * 4),
        instrument=INSTRUMENT,
        timeframe=TIMEFRAME,
        clock=SimClock(),
        rebalance_threshold=0.0,
        record_reasons=False,
    )
    df = synthetic_ohlcv(n=4, seed=8)
    bar_ms = timeframe_to_millis(TIMEFRAME)
    bars = [_row_to_bar(ts, row, INSTRUMENT, TIMEFRAME, bar_ms) for ts, row in df.iterrows()]

    assert runner.on_bar(bars[2])
    held = runner.held
    assert runner.on_bar(bars[0]) == ()
    assert runner.held == held


def test_a_signal_set_strategy_is_refused_at_construction():
    """The mirror of M40's check on ``StrategyRunner``: a contract mismatch that
    waits for the first post-warmup bar is invisible for as long as the warmup."""
    with pytest.raises(TypeError, match="StrategyRunner"):
        ExposureRunner(
            strategy=get_strategy("donchian"),
            instrument=INSTRUMENT,
            timeframe=TIMEFRAME,
            clock=SimClock(),
        )


def test_a_negative_band_is_refused():
    """It would submit on every bar while claiming to damp."""
    with pytest.raises(ValueError, match="rebalance_threshold"):
        ExposureRunner(
            strategy=_Scripted(targets=(0.5,)),
            instrument=INSTRUMENT,
            timeframe=TIMEFRAME,
            clock=SimClock(),
            rebalance_threshold=-0.01,
        )
