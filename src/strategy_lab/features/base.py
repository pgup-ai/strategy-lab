"""The state-feature protocol, and the two windowed statistics every feature shares.

Half the state vector is a percentile or a z-score, and the obvious form of both
reads the whole series. Measured on a 200-bar ramp poisoned downward from row 121,
``series.rank(pct=True)`` at row 120 moves 0.605 -> 1.000 while the rolling form
does not move at all -- bar *t* changing its mind because of bars that had not
happened yet. That is the shape of ``_SubtleCheat`` in ``tests/test_lookahead.py``:
full-sample normalization, no ``shift(-1)`` anywhere, non-causal regardless.

So no feature hand-rolls a percentile or a z-score. Both live here, both window
strictly backwards, and ``tests/test_feature_lookahead.py`` poisons the future of
every registered feature to keep it that way.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class StateFeature(Protocol):
    """One dimension of market state, shaped exactly like ``strategies.base.Strategy``.

    Same three metadata fields and one compute method, deliberately: the poison
    probe, the manual registry and the cold-start warmup check were all written
    against that shape, so features get them without a second implementation.

    ``compute`` returns a ``pd.Series`` on the input frame's index. Signed
    features range -1..1, unsigned ones 0..1, and warmup rows are ``NaN`` -- a
    0.0 there would read as "measured, and neutral", which is a different claim
    from "not yet measurable".
    """

    name: str
    version: str
    warmup_bars: int

    def compute(self, df: pd.DataFrame) -> pd.Series:
        ...


def mask_warmup(values: pd.Series, *, warmup_bars: int) -> pd.Series:
    """``NaN`` the first ``warmup_bars`` rows, whatever the computation produced there.

    Not every indicator declines to answer early. ``rolling(n)`` yields ``NaN``
    until its window fills, but ``ewm(adjust=False)`` returns a number from bar
    zero -- a converging one, off by a decaying seed for some twenty spans. To a
    caller both are "a value", one of them silently unusable, so the warmup
    boundary is drawn here instead of being left to each indicator's own habit.
    """
    masked = values.copy()
    masked.iloc[:warmup_bars] = np.nan
    return masked


def rolling_percentile(series: pd.Series, *, window: int) -> pd.Series:
    """Where the current value sits within the trailing ``window`` observations, 0..1.

    ``NaN`` until the window fills, then the current observation's rank among the
    last ``window`` of them -- itself included -- divided by ``window``.

    Ties take the midpoint of the ranks they share rather than the top of them.
    A run of identical values reads 1.0 under the top rule, "at the very top of
    its range", about a window that has no range; on a genuinely compressed
    market that inverts ``Compression``, whose whole subject is such windows.
    """
    return series.rolling(window).rank(pct=True)


def rolling_zscore(series: pd.Series, *, window: int) -> pd.Series:
    """Deviations from the trailing ``window`` mean, in trailing standard deviations.

    ``NaN`` until the window fills. Zero trailing spread yields 0.0: the current
    value is inside its own window, so a window of identical values puts it
    exactly on the mean and the ratio is 0/0 -- pandas ``NaN``, and +/-inf the
    moment float error makes the numerator non-zero. Either one survives a
    ``tanh`` squash as a saturated feature that looks measured.
    """
    trailing = series.rolling(window)
    spread = trailing.std()
    zscore = (series - trailing.mean()) / spread
    # NaN spread (warmup) is not equal to 0, so those rows keep their NaN.
    return zscore.where(spread != 0, 0.0)
