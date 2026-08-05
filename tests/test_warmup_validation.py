"""A negative ``warmup_bars`` must be refused, not arithmetically absorbed.

``warmup_bars`` is a measured claim that a strategy's indicators have not
converged, and every consumer of it does arithmetic on the number rather than
asking whether the number is a warmup at all. Each one then fails *open* --
towards trading the prefix, never towards refusing it. Measured against a stub
declaring -5:

===============================================  ==========================
``engine._warmup_bars``, ``size_mode=fixed``     returns -5, unreported
``engine._warmup_bars``, ``vol-scaled-entry``    returns 1920, absorbed
``engine._mask_warmup(-5)``                      silences 0 of 9 entries
``runner``: ``len(buffer) <= warmup_bars``       ``1 <= -5`` emits on bar 1
``sweep``: ``warmup >= len(df)``                 false, so it passes through
===============================================  ==========================

The vol-scaled row is why the check sits on ``strategy.warmup_bars`` itself
rather than on the resolved value: ``_warmup_bars`` returns
``max(declaration, estimator)``, so under vol-scaled sizing the false claim is
swallowed by the ``max`` and the run reports a perfectly ordinary 400. Guarding
the resolved number would catch the same broken strategy in one sizing mode and
not the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from strategy_lab.backtests.sweep import sweep_parameters
from strategy_lab.core.clock import SimClock
from strategy_lab.core.types import InstrumentId
from strategy_lab.engine.runner import StrategyRunner
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.strategies.base import SignalSet, require_warmup_bars
from strategy_lab.strategies.registry import get_strategy, list_strategies
from tests.conftest import synthetic_ohlcv

_IDENTITY = MarketDataIdentity(
    exchange="binance", market_type="spot", symbol="BTC/USDT", timeframe="15m"
)
_INSTRUMENT = InstrumentId("binance", "spot", "BTC/USDT")


@dataclass(frozen=True)
class _NegativeWarmup:
    """Enters on every bar, so an unguarded run is visibly one that traded."""

    name: str = "negative_warmup_stub"
    version: str = "1.0.0"
    warmup_bars: int = -5

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        always = pd.Series(True, index=df.index)
        flat = pd.Series(False, index=df.index)
        return SignalSet(always, flat.copy(), flat.copy(), flat.copy())


def test_zero_is_a_warmup_and_not_an_omission() -> None:
    """A strategy with no lookback has nothing to warm up and may say so.

    This is the whole difference from ``require_positive_span``, whose fields
    index a window and so cannot be zero.
    """
    require_warmup_bars("no_lookback", 0)


@pytest.mark.parametrize("value", [-1, True, False, 4.0, "40", None])
def test_a_warmup_that_is_not_a_non_negative_int_is_refused(value: object) -> None:
    """``bool`` is in here because it is an ``int`` subclass.

    Without the separate check ``warmup_bars=True`` passes and silently means
    one bar, which is a warmup nobody chose.
    """
    with pytest.raises(ValueError, match="warmup_bars"):
        require_warmup_bars("stub", value)


def test_the_backtest_refuses_a_negative_warmup_the_max_would_have_absorbed(tmp_path) -> None:
    """Under vol-scaled sizing the estimator's 1920 bars hide the -5 entirely.

    So this run is the one that distinguishes a guard on the declaration from a
    guard on the resolved value: without the former it completes and writes a
    report whose ``config.json`` says ``warmup_bars: 1920``, a number the
    strategy never claimed.
    """
    from strategy_lab.backtests.engine import run_backtest

    with pytest.raises(ValueError, match="warmup_bars"):
        run_backtest(
            df=synthetic_ohlcv(n=3000),
            strategy=_NegativeWarmup(),
            identity=_IDENTITY,
            exit_mode="opposite_signal_only",
            report_root=tmp_path,
            size_mode="vol-scaled-entry",
        )


def test_the_replay_runner_refuses_a_negative_warmup_at_construction() -> None:
    """At construction, so a live process declines to start rather than to continue.

    ``on_bar`` suppresses nothing when the comparison is ``1 <= -5``, so the
    unguarded runner emits a signal for the very first bar it is handed -- and
    the vectorized path would have raised on the same strategy, which is the
    two paths disagreeing about which bars exist.
    """
    with pytest.raises(ValueError, match="warmup_bars"):
        StrategyRunner(
            strategy=_NegativeWarmup(),
            instrument=_INSTRUMENT,
            timeframe="15m",
            clock=SimClock(),
        )


def test_the_sweep_refuses_a_cell_whose_warmup_is_negative() -> None:
    """``warmup_bars`` is an ordinary dataclass field, so the grid can name it.

    ``sweep_parameters`` accepts any field of the template, and the four
    strategies that do not recompute their warmup in ``__post_init__`` take the
    swept value verbatim. The whole grid then goes negative, ``warmup >=
    len(df)`` stays false, and ``returns.iloc[-5:]`` scores the surface on the
    last five bars of the frame instead of on all of them.
    """
    with pytest.raises(ValueError, match="warmup_bars"):
        sweep_parameters(
            df=synthetic_ohlcv(n=600),
            strategy_name="trend_following_deepseek_v4",
            grid={"warmup_bars": [-5, 40]},
            timeframe="15m",
        )


def test_every_registered_strategy_declares_a_non_negative_warmup() -> None:
    """Catches a new strategy at CI rather than on the run that trades it.

    Registry-wide but not sufficient on its own: this only ever sees defaults,
    while ``sweep_parameters`` rebuilds every cell with ``dataclasses.replace``
    over any field. That is why the three consumers check too.
    """
    for name in list_strategies():
        require_warmup_bars(name, get_strategy(name).warmup_bars)
