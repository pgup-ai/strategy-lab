"""The poison probe, pointed at state features instead of strategies.

``tests/test_lookahead.py`` iterates ``strategies.registry.list_strategies()``, so
until this file existed no feature was covered by it -- and more than half of the
registered features are percentiles or z-scores, the one construction that reads
the future without a ``shift(-1)`` anywhere to grep for. Measured: swapping
``Energy``'s rolling percentile for ``series.rank(pct=True)`` passed the entire
412-test suite that preceded this module.

The technique, the poison profiles and the frame sizing are all imported rather
than reinvented; only two things are new. The funding column is poisoned as well
as the prices, because Crowding reads nothing else and a probe that rewrites only
OHLCV hands it back a byte-identical series. And the probe compares one float per
bar instead of eight ``SignalSet`` fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from strategy_lab.features.flow import FUNDING_COLUMN
from strategy_lab.features.registry import get_feature, list_features
from tests.conftest import synthetic_ohlcv_with_funding
from tests.test_lookahead import POISON_PROFILES, PROBE_SPAN, _same, probe_positions


def feature_probe_frame(warm: int, seed: int = 7) -> pd.DataFrame:
    """A frame with ``PROBE_SPAN`` probe-able bars past ``warm``, funding included.

    Sized ``warm + PROBE_SPAN`` for the reason ``tests/test_lookahead`` documents:
    a fixed total length silently yields zero probe points as soon as a feature's
    warmup exceeds it, and Direction's is 1920.
    """
    return synthetic_ohlcv_with_funding(n=warm + PROBE_SPAN, seed=seed)


def _poison_funding(poisoned: pd.DataFrame, tail: pd.Index, t: int) -> None:
    """Corrupt the funding column too, with both signs.

    Signs vary per bar so a feature reading future *carry direction* changes its
    answer, the same reasoning behind the directional OHLCV profile. Seeded by
    the probe index to stay deterministic while varying the pattern's phase.
    """
    if FUNDING_COLUMN not in poisoned.columns:
        return
    signs = np.where(np.random.default_rng(t).random(len(tail)) < 0.5, 1.0, -1.0)
    poisoned.loc[tail, FUNDING_COLUMN] = signs


def poison_probe_feature(
    feature,
    df: pd.DataFrame,
    *,
    warm: int,
    step: int = 20,
    profiles: tuple = POISON_PROFILES,
) -> list[tuple[str, int]]:
    """Return (profile, bar index) pairs whose value changed when the FUTURE was corrupted.

    A causal feature cannot see past bar t, so replacing bars t+1.. with garbage
    must leave row t bit-identical -- not close, identical: every rolling window
    at row t is built from rows <= t, and no float in it has any business moving.
    """
    baseline = feature.compute(df)
    offenders: list[tuple[str, int]] = []
    for profile_name, poison in profiles:
        for t in probe_positions(warm, len(df), step):
            poisoned = df.copy()
            tail = poisoned.index[t + 1 :]
            poison(poisoned, tail, t)
            _poison_funding(poisoned, tail, t)
            if not _same(baseline.iloc[t], feature.compute(poisoned).iloc[t]):
                offenders.append((profile_name, t))
    return offenders


@pytest.mark.parametrize("name", list_features())
def test_registered_features_do_not_look_ahead(name):
    feature = get_feature(name)
    df = feature_probe_frame(feature.warmup_bars)
    # "No offenders" is only meaningful if bars were actually probed.
    probed = probe_positions(feature.warmup_bars, len(df))
    assert len(probed) >= 50, (
        f"{name}: only {len(probed)} probe points past warmup_bars="
        f"{feature.warmup_bars}; the gate would pass without testing anything"
    )
    offenders = poison_probe_feature(feature, df, warm=feature.warmup_bars)
    assert offenders == [], f"{name} used future data at (profile, bar index) {offenders}"


@pytest.mark.parametrize("name", list_features())
def test_every_registered_feature_is_measurable_at_its_own_warmup(name):
    """Row ``warmup_bars`` has to carry a number, or the probe above proves nothing.

    ``_same`` treats NaN as equal to NaN, so a feature that is still NaN at every
    probed bar passes the lookahead gate without a single real comparison.
    """
    feature = get_feature(name)
    values = feature.compute(feature_probe_frame(feature.warmup_bars))
    assert values.iloc[feature.warmup_bars :].notna().all()


# The probe is only worth having if it can fail. These two prove it can, and they
# are the two shapes this phase actually risks.


@dataclass(frozen=True)
class _FullSamplePercentile:
    """The exact trap this phase is built around: no shift(-1), still non-causal."""

    name: str = "full_sample_percentile"
    version: str = "1.0.0"
    warmup_bars: int = 50

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["close"].pct_change().rolling(20).std().rank(pct=True)


@dataclass(frozen=True)
class _NextBarFunding:
    """Reads the funding that has not settled yet.

    Here to prove the funding poison does something. Without it the OHLCV
    profiles would leave Crowding's only input untouched and its probe would be
    an assertion that ``compute`` is deterministic.
    """

    name: str = "next_bar_funding"
    version: str = "1.0.0"
    warmup_bars: int = 50

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df[FUNDING_COLUMN].shift(-1).rolling(20).mean()


@pytest.mark.parametrize("cheat", [_FullSamplePercentile(), _NextBarFunding()])
def test_the_feature_probe_detects_lookahead(cheat):
    df = feature_probe_frame(cheat.warmup_bars)
    offenders = poison_probe_feature(cheat, df, warm=cheat.warmup_bars, step=10)
    assert offenders, f"{cheat.name} smuggled future data past the probe"
