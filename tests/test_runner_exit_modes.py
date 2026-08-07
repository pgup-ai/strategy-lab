"""The runner's exit mode, and the contract check that used to wait a year.

R10d measured what the runner's exits cost while it had no mode: under each
strategy's *canonical* configuration the stream differed from the backtest on
7,331 bars for ``trend_following_deepseek_v4`` (which emits none of its own),
984 for the two turnarounds, and 0 for the other six. This file is the parity
proof that drives those to zero, and the six already at zero are what stops it
going vacuous -- a change that broke them fails the same assertion.

The runner calls ``engine._exit_signals`` rather than reimplementing it, so what
is tested here is the *wiring*: that the mode reaches the emitted stream, over
the buffer, on the right bar.
"""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from strategy_lab.backtests.engine import ExitMode, _exit_signals
from strategy_lab.core.clock import SimClock
from strategy_lab.core.types import InstrumentId, Side
from strategy_lab.engine.runner import StrategyRunner
from strategy_lab.feeds.base import Subscription
from strategy_lab.feeds.replay import ReplayFeed
from strategy_lab.strategies.exposure_registry import get_exposure_strategy
from strategy_lab.strategies.registry import get_strategy
from tests.conftest import synthetic_ohlcv

TIMEFRAME = "4h"
INSTRUMENT = InstrumentId("binance", "perp", "BTC/USDT")
FAILURE_BARS = 4

_SIDES = (
    ("long_entries", Side.ENTER_LONG),
    ("long_exits", Side.EXIT_LONG),
    ("short_entries", Side.ENTER_SHORT),
    ("short_exits", Side.EXIT_SHORT),
)

# One per class of exit ownership, which is what the parity claim ranges over:
# a strategy the engine exits for, one that owns its own exits, and one that
# provides none at all -- the case R10d measured at 7,331 differing bars.
PARITY = [
    ("donchian", ExitMode.OPPOSITE_SIGNAL_ONLY),
    ("donchian", ExitMode.CONTINUATION_FAILURE),
    ("tsmom", ExitMode.CONTINUATION_FAILURE),
    ("turnaround_v1", ExitMode.CONTINUATION_FAILURE),
    ("trend_rider_v1_deepseek_v4_pro", ExitMode.OPPOSITE_SIGNAL_ONLY),
]


def streamed(strategy, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    runner = StrategyRunner(
        strategy=strategy,
        instrument=INSTRUMENT,
        timeframe=TIMEFRAME,
        clock=SimClock(),
        record_reasons=False,
        **kwargs,
    )
    feed = ReplayFeed(frames={INSTRUMENT.at(TIMEFRAME): df})
    sub = Subscription(INSTRUMENT, TIMEFRAME)
    rows: dict[pd.Timestamp, dict[str, bool]] = {}

    async def _run() -> None:
        async for event in feed.stream([sub]):
            for signal in runner.on_event(event):
                stamp = pd.Timestamp(signal.ts_bar_ms, unit="ms", tz="UTC")
                rows.setdefault(stamp, {})[signal.side.value] = True

    asyncio.run(_run())
    # Built as booleans rather than reindexed into them. A bar that emitted only
    # an exit has no entry key, and the reindex-then-cast route leaves that cell
    # NaN -- which casts to **True**, reading every such bar as an entry. That
    # went red on a real frame in R10d before it was caught.
    index = list(rows)
    return pd.DataFrame(
        {side.value: [rows[stamp].get(side.value, False) for stamp in index] for _, side in _SIDES},
        index=index,
        dtype=bool,
    )


@pytest.mark.parametrize(("name", "mode"), PARITY, ids=lambda v: getattr(v, "value", v))
def test_the_runner_emits_the_exits_its_mode_implies(name, mode):
    """R10d's gap, driven to zero. ``turnaround_v1`` under its canonical mode was
    984 bars on BTC; the pass-through pairs were already 0 and must stay there."""
    strategy = get_strategy(name)
    df = synthetic_ohlcv(n=strategy.warmup_bars + 400, seed=11)
    got = streamed(strategy, df, exit_mode=mode, failure_bars=FAILURE_BARS)

    signals = strategy.generate_signals(df)
    long_exits, short_exits = _exit_signals(
        df=df, signals=signals, exit_mode=mode, failure_bars=FAILURE_BARS
    )
    emitting = df.index[strategy.warmup_bars :]
    expected = {
        "long_entries": signals.long_entries,
        "long_exits": long_exits.fillna(False),
        "short_entries": signals.short_entries,
        "short_exits": short_exits.fillna(False),
    }

    assert got.to_numpy().sum() > 0, "no signals emitted; this would compare nothing"
    for field, side in _SIDES:
        want = expected[field].reindex(emitting).fillna(False).astype(bool)
        assert (want != got[side.value].reindex(emitting, fill_value=False)).sum() == 0, field


def test_a_strategy_that_provides_no_exits_emits_them_once_the_mode_supplies_them():
    """The 7,331-bar case, and the one that made the stream unusable rather than
    merely different: ``trend_following_deepseek_v4`` emits no exits of its own,
    so without a mode a replay of it opens positions and never closes one."""
    strategy = get_strategy("trend_following_deepseek_v4")
    df = synthetic_ohlcv(n=strategy.warmup_bars + 400, seed=5)

    bare = streamed(strategy, df)
    with_mode = streamed(strategy, df, exit_mode=ExitMode.TREND_STRUCTURE)

    assert bare["exit_long"].sum() == 0
    assert with_mode["exit_long"].sum() > 0
    assert bare["enter_long"].sum() == with_mode["enter_long"].sum()


def test_without_a_mode_the_runner_emits_the_strategys_own_exits():
    """The pre-R10e behaviour is the default, so every stored replay predating
    this phase still describes what its runner did."""
    strategy = get_strategy("donchian")
    df = synthetic_ohlcv(n=strategy.warmup_bars + 300, seed=3)

    got = streamed(strategy, df)
    signals = strategy.generate_signals(df)
    emitting = df.index[strategy.warmup_bars :]

    want = signals.long_exits.reindex(emitting).fillna(False).astype(bool)
    assert want.sum() > 0
    assert (want != got["exit_long"].reindex(emitting, fill_value=False)).sum() == 0


def test_setup_invalidation_stop_is_refused_rather_than_approximated():
    """The engine applies it as an intrabar ``sl_stop``, which no bar-close signal
    encodes -- so a runner that accepted it would silently describe a different
    book from the backtest it claims to reproduce."""
    with pytest.raises(ValueError, match="cannot be driven from a signal stream"):
        StrategyRunner(
            strategy=get_strategy("turnaround_v1"),
            instrument=INSTRUMENT,
            timeframe=TIMEFRAME,
            clock=SimClock(),
            exit_mode=ExitMode.SETUP_INVALIDATION_STOP,
        )


def test_an_exposure_strategy_is_refused_at_construction_not_a_warmup_later():
    """M40. Measured before the check existed: ``StrategyRunner`` accepted
    ``state_machine_v2`` and survived 2,192 bars -- 365.3 days at 4h -- because
    ``on_bar`` returns before touching the strategy while inside warmup. The
    assertion is that it raises with **no** bars fed, which is what distinguishes
    the fix from the bug."""
    with pytest.raises(TypeError, match="ExposureRunner"):
        StrategyRunner(
            strategy=get_exposure_strategy("state_machine_v2"),
            instrument=INSTRUMENT,
            timeframe=TIMEFRAME,
            clock=SimClock(),
        )
