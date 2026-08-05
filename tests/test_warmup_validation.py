"""A negative ``warmup_bars`` must be refused, not arithmetically absorbed.

``warmup_bars`` is a measured claim that an indicator has not converged, and
every consumer of it does arithmetic on the number rather than asking whether
the number is a warmup at all. Each one then fails *open* -- towards using the
unconverged prefix, never towards refusing it. Both protocols declare the field
and both are covered here, because the guarantee is one guarantee.

``Strategy``, measured against a stub declaring -5:

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

``StateFeature``, measured on a 200-row frame, where the same -5 inverts every
slice it reaches rather than shortening it:

===============================================  ==========================
``features.base.mask_warmup``                    ``NaN``s 195 rows, not 5
``features.diagnostics._diagnose``               measures 5 rows, not 195
``cli._diagnosable_features``' probe frame       197 rows, not 7
===============================================  ==========================

The diagnostics row is the one that reaches a reader: coverage, IC, turnover and
the split-half comparison are computed on those five bars and printed as though
they covered the frame, and the existing ``measured.empty`` check does not trip,
because a tail slice is not empty. The CLI row is the sharper irony -- that probe
exists because a probe comparing ``NaN`` to ``NaN`` passes without testing
anything, and a negative warmup restores exactly that by another route.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from strategy_lab.backtests.sweep import sweep_parameters
from strategy_lab.cli import _diagnosable_features
from strategy_lab.core.clock import SimClock
from strategy_lab.core.types import InstrumentId
from strategy_lab.engine.runner import StrategyRunner
from strategy_lab.features.base import mask_warmup
from strategy_lab.features.diagnostics import diagnose
from strategy_lab.features.registry import get_feature, list_features
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


@dataclass(frozen=True)
class _UnmaskedFeature:
    """Declares -5 and never calls ``mask_warmup``, which is what isolates it.

    Every registered feature masks through that helper and so would raise there
    first; a stub built the same way would measure that guard a second time
    rather than the diagnostic's own. This one computes cleanly, leaving
    ``_diagnose`` as the only thing that can refuse it.
    """

    name: str = "unmasked_stub"
    version: str = "1.0.0"
    warmup_bars: int = -5

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(range(len(df)), index=df.index, dtype="float64")


@dataclass(frozen=True)
class _MaskedFeature:
    """Declares -5 and masks through ``mask_warmup``, exactly as a real feature does.

    So this one raises from inside ``compute``, which is the whole point of the
    CLI test below: the loop's ``except ValueError`` would file that raise under
    ``skipped``.
    """

    name: str = "masked_stub"
    version: str = "1.0.0"
    warmup_bars: int = -5

    def compute(self, df: pd.DataFrame) -> pd.Series:
        values = pd.Series(range(len(df)), index=df.index, dtype="float64")
        return mask_warmup(values, warmup_bars=self.warmup_bars, name=self.name)


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


def test_masking_a_warmup_refuses_the_negative_that_would_invert_it() -> None:
    """``iloc[:-5]`` is every row *but* the last five, so the mask lands inside out.

    Measured on this 200-row series without the guard: warmup 5 leaves 5 leading
    ``NaN`` and warmup -5 leaves 195 -- the unconverged prefix kept, and every
    row the feature actually measured erased.
    """
    values = pd.Series(range(200), dtype="float64")
    assert int(mask_warmup(values, warmup_bars=5, name="stub").isna().sum()) == 5
    with pytest.raises(ValueError, match="warmup_bars"):
        mask_warmup(values, warmup_bars=-5, name="stub")


def test_masking_a_warmup_makes_the_feature_name_itself() -> None:
    """``name`` has no default, so a tenth feature that forgets it fails at the call.

    A default would raise a correct-looking error naming nothing, from a helper
    with nine call sites and no other way to tell them apart.
    """
    with pytest.raises(TypeError, match="name"):
        mask_warmup(pd.Series([1.0, 2.0]), warmup_bars=1)


def test_the_feature_diagnostic_refuses_a_negative_warmup_the_empty_check_misses() -> None:
    """``measured.empty`` reads like this check and is not it.

    ``iloc[-5:]`` is 5 rows of a 200-row frame -- non-empty, so the existing
    guard waves it through and the diagnostic reports coverage, IC, turnover and
    a split-half comparison computed on five bars as though they covered the
    frame. A wrong number in the research charter, where the empty guard's own
    failure would at least have been a blank.
    """
    with pytest.raises(ValueError, match="warmup_bars"):
        diagnose(_UnmaskedFeature(), synthetic_ohlcv(n=200))


def test_the_features_cli_refuses_a_negative_warmup_rather_than_filing_a_skip(monkeypatch) -> None:
    """A guard inside that loop's ``try`` would reach the silent outcome it prevents.

    ``_diagnosable_features`` turns a ``ValueError`` from ``compute`` into a
    skip, which is the right reading for Crowding on a frame with no funding and
    the wrong one for a broken declaration -- "this feature could not be measured
    here" instead of "this feature's warmup is not a warmup". ``_MaskedFeature``
    raises from inside ``compute`` for exactly that reason, so with the guard
    moved into the ``try`` this call stops raising and returns ``([], {...})``
    with the feature recorded under ``skipped``.
    """
    monkeypatch.setattr(
        "strategy_lab.features.registry.get_feature", lambda name: _MaskedFeature()
    )
    with pytest.raises(ValueError, match="warmup_bars"):
        _diagnosable_features(["masked_stub"], synthetic_ohlcv(n=200))


def test_every_registered_feature_declares_a_non_negative_warmup() -> None:
    """The same CI catch as for strategies, over the other protocol's registry.

    Both are needed and neither generalises: the two registries are manual and
    separate, and a feature never passes through ``get_strategy``.
    """
    for name in list_features():
        require_warmup_bars(name, get_feature(name).warmup_bars)
