"""The poison probe, pointed at the continuous-exposure contract.

``tests/test_lookahead.py`` iterates ``strategies.registry.list_strategies()``
and compares eight ``SignalSet`` fields, and ``tests/test_feature_lookahead.py``
iterates the feature registry and compares one float. Neither can see an
exposure strategy: it has no ``generate_signals`` and it is deliberately in a
third registry, so without this module a registered exposure strategy would be
covered by no lookahead gate at all.

The technique, the poison profiles, the frame sizing and the probe positions are
imported rather than reinvented -- those lengths are measured (see
``PROBE_SPAN``'s comment: at 400 bars a real lookahead bug went undetected on
32.5% of seeds), and the directional profile exists because the flat one leaves
``open == close`` and so silences every candle-direction predicate. Two things
are new. Funding is poisoned as well as the prices, reusing
``test_feature_lookahead``'s profile, because ``state_machine_v2`` reads
``crowding`` and a probe that rewrote only OHLCV would hand it back a
byte-identical column. And the comparison is one signed float per bar.

**The vacuity trap is different in shape here.** The feature probe's is NaN:
``_same`` calls NaN equal to NaN, so a feature still unmeasurable at every
probed bar passes without a single real comparison. ``TargetExposure`` forbids
NaN outright, so the exposure analogue is a target that is **0.0 everywhere** --
which compares equal to itself just as happily.
:func:`test_a_flat_target_hides_lookahead_from_this_probe` demonstrates that on
a strategy that genuinely reads the future, and
:func:`test_every_registered_exposure_strategy_moves_over_its_probe_window` is
the gate that stops it counting as a pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from strategy_lab.features.flow import FUNDING_COLUMN
from strategy_lab.strategies.exposure import TargetExposure
from strategy_lab.strategies.exposure_registry import (
    get_exposure_strategy,
    list_exposure_strategies,
)
from tests.conftest import synthetic_ohlcv_with_funding
from tests.test_feature_lookahead import _poison_funding
from tests.test_lookahead import POISON_PROFILES, PROBE_SPAN, _same, probe_positions

# Target changes required past warmup before "no offenders" means anything. The
# floor, not a target: measured at 73 changes over 1,200 post-warmup bars for
# ``state_machine_v2``, on the seed below.
MIN_CHANGES = 20


def exposure_probe_frame(warm: int, seed: int = 7) -> pd.DataFrame:
    """``PROBE_SPAN`` probe-able bars past ``warm``, funding included.

    Sized ``warm + PROBE_SPAN`` for the reason ``tests/test_lookahead`` records:
    a fixed total length silently yields zero probe points as soon as a
    strategy's warmup exceeds it, and ``state_machine_v2``'s is 847.
    """
    return synthetic_ohlcv_with_funding(n=warm + PROBE_SPAN, seed=seed)


def poison_probe_target(
    strategy,
    df: pd.DataFrame,
    *,
    warm: int,
    step: int = 20,
    profiles: tuple = POISON_PROFILES,
) -> list[tuple[str, int]]:
    """Return (profile, bar index) pairs whose target changed when the FUTURE was corrupted.

    A causal strategy cannot see past bar t, so replacing bars t+1.. with
    garbage must leave row t bit-identical -- not close, identical. Every window
    contributing to bar t is built from rows <= t and no float in it has any
    business moving.
    """
    baseline = strategy.compute_target(df).target
    offenders: list[tuple[str, int]] = []
    for profile_name, poison in profiles:
        for t in probe_positions(warm, len(df), step):
            poisoned = df.copy()
            tail = poisoned.index[t + 1 :]
            poison(poisoned, tail, t)
            _poison_funding(poisoned, tail, t)
            if not _same(baseline.iloc[t], strategy.compute_target(poisoned).target.iloc[t]):
                offenders.append((profile_name, t))
    return offenders


@pytest.mark.parametrize("name", list_exposure_strategies())
def test_registered_exposure_strategies_do_not_look_ahead(name):
    strategy = get_exposure_strategy(name)
    df = exposure_probe_frame(strategy.warmup_bars)
    probed = probe_positions(strategy.warmup_bars, len(df))
    assert len(probed) >= 50, (
        f"{name}: only {len(probed)} probe points past warmup_bars="
        f"{strategy.warmup_bars}; the gate would pass without testing anything"
    )
    offenders = poison_probe_target(strategy, df, warm=strategy.warmup_bars)
    assert offenders == [], f"{name} used future data at (profile, bar index) {offenders}"


@pytest.mark.parametrize("name", list_exposure_strategies())
def test_every_registered_exposure_strategy_moves_over_its_probe_window(name):
    """The target above has to vary, or the probe compares 0.0 against 0.0.

    This is the exposure contract's version of
    ``test_every_registered_feature_is_measurable_at_its_own_warmup``: both
    exist because the comparison they guard is happy to succeed on a strategy
    that says nothing.
    """
    strategy = get_exposure_strategy(name)
    target = strategy.compute_target(exposure_probe_frame(strategy.warmup_bars)).target
    live = target.iloc[strategy.warmup_bars :]

    changes = int((live.diff().fillna(0.0) != 0).sum())
    assert changes >= MIN_CHANGES, (
        f"{name}: only {changes} target changes over {len(live)} probed bars, so the "
        f"lookahead gate above is mostly 0.0 == 0.0"
    )
    assert (live != 0.0).any(), f"{name} never held a position over the probe window"


# The probe is only worth having if it can fail. These four prove it can, and
# they are ported from the boolean and feature probes' own cheats rather than
# invented, so each one's shape is already known to be the shape that matters.


@dataclass(frozen=True)
class _FutureReaderTarget:
    """Holds a full position when the NEXT bar closes higher -- textbook shift(-1)."""

    name: str = "future_reader_target"
    version: str = "1.0.0"
    warmup_bars: int = 10

    def compute_target(self, df: pd.DataFrame) -> TargetExposure:
        return TargetExposure(target=(df["close"].shift(-1) > df["close"]).astype("float64"))


@dataclass(frozen=True)
class _FullSampleScaledTarget:
    """No shift(-1) anywhere -- but the SIZE is scaled by the full-sample spread.

    The shape a continuous contract is most exposed to and a boolean one is
    least: nothing here decides *which way* from a future bar. It decides *how
    much* from a statistic over bars that have not happened, and how much is
    most of what a target is.
    """

    name: str = "full_sample_scaled_target"
    version: str = "1.0.0"
    warmup_bars: int = 10

    def compute_target(self, df: pd.DataFrame) -> TargetExposure:
        z = (df["close"] - df["close"].mean()) / df["close"].std()
        return TargetExposure(target=np.tanh(z).fillna(0.0))


@dataclass(frozen=True)
class _CandleDirectionTarget:
    """This repo's three-candle turnaround, every candle read from the FUTURE.

    Purely candle-direction, which is the shape the flat profile is weakest
    against: a conjunction of direction predicates collapses to False on
    flat-poisoned bars, where ``open == close``.
    """

    name: str = "candle_direction_target"
    version: str = "1.0.0"
    warmup_bars: int = 10

    def compute_target(self, df: pd.DataFrame) -> TargetExposure:
        red1 = df["close"].shift(-1) < df["open"].shift(-1)
        red2 = df["close"].shift(-2) < df["open"].shift(-2)
        green = df["close"].shift(-3) > df["open"].shift(-3)
        return TargetExposure(target=(red1 & red2 & green).fillna(False).astype("float64"))


@dataclass(frozen=True)
class _NextBarFundingTarget:
    """Sizes from carry that has not settled yet.

    Here to prove the funding poison reaches this probe's comparison. Without
    it, ``state_machine_v2``'s only non-OHLCV input would be untouched and its
    ``crowding`` path would be probed by nothing.
    """

    name: str = "next_bar_funding_target"
    version: str = "1.0.0"
    warmup_bars: int = 50

    def compute_target(self, df: pd.DataFrame) -> TargetExposure:
        carry = df[FUNDING_COLUMN].shift(-1).rolling(20).mean().fillna(0.0)
        return TargetExposure(target=pd.Series(np.sign(carry), index=df.index))


@pytest.mark.parametrize(
    "cheat",
    [
        _FutureReaderTarget(),
        _FullSampleScaledTarget(),
        _CandleDirectionTarget(),
        _NextBarFundingTarget(),
    ],
    ids=lambda cheat: cheat.name,
)
def test_the_exposure_probe_detects_lookahead(cheat):
    df = exposure_probe_frame(cheat.warmup_bars)
    offenders = poison_probe_target(cheat, df, warm=cheat.warmup_bars, step=10)
    assert offenders, f"{cheat.name} smuggled future data past the probe"


@dataclass(frozen=True)
class _InertFutureReader(_FutureReaderTarget):
    """``_FutureReaderTarget``'s lookahead, scaled to nothing.

    Reads the next bar exactly as its parent does and then holds 0.0 whatever it
    finds. Non-causal, and invisible to a probe that can only compare the
    targets two runs produced.
    """

    name: str = "inert_future_reader"

    def compute_target(self, df: pd.DataFrame) -> TargetExposure:
        return TargetExposure(target=super().compute_target(df).target * 0.0)


def test_a_flat_target_hides_lookahead_from_this_probe():
    """Why the movement gate is a test rather than a comment in the docstring.

    The same lookahead the probe catches in ``_FutureReaderTarget`` walks
    straight past it once the target stops moving, because every comparison
    becomes ``0.0 == 0.0``. Nothing about the probe is wrong; it is simply not
    the check that catches this, and shipping it alone would have been a gate
    that could not fail.
    """
    inert = _InertFutureReader()
    df = exposure_probe_frame(inert.warmup_bars)

    assert poison_probe_target(inert, df, warm=inert.warmup_bars, step=10) == []
    assert poison_probe_target(
        _FutureReaderTarget(), df, warm=inert.warmup_bars, step=10
    ), "setup failed: the same rule must be caught when its target actually moves"

    # Both halves of the gate, on the slice the gate itself measures -- so this
    # tracks the gate rather than a quantity that merely resembles it. The
    # length check is not decoration: these are the gate's assertions inverted,
    # and inverting them removes the gate's own protection against an empty
    # slice, which satisfies "fewer than MIN_CHANGES changes" and "never
    # non-zero" without measuring anything.
    live = inert.compute_target(df).target.iloc[inert.warmup_bars :]
    assert len(live) == PROBE_SPAN
    assert int((live.diff().fillna(0.0) != 0).sum()) < MIN_CHANGES
    assert not (live != 0.0).any()
