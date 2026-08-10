"""What each state feature is, before anything is built on top of it.

The R4 gate is *each feature has a univariate diagnostic; no feature dumped in
unexamined*, and this is that diagnostic: is it computable, is it degenerate,
does it persist, what would trading it cost, does it carry any forward
information, and does that information survive being split in half.

Two decisions carry the whole module.

**The forward return starts one bar after the feature's own.** The obvious
target, ``close[t+h] / close[t] - 1``, contains no *return* from bar *t* -- which
is why it reads as safe -- but it does contain bar *t*'s *price*, as its
denominator. Every feature here is a function of that same print, so anything
rising with a high ``close[t]`` divides a high number into its own target and is
paid for it. Measured on a random walk plus an i.i.d. print error, with a feature
that reads only the current bar's deviation from its trailing mean and therefore
predicts nothing: IC -0.53 anchored at ``close[t]``, -0.01 anchored at
``close[t+1]``. The first number is entirely the anchor. Starting at *t+1* is
also the only convention that is executable -- a value known at the close of bar
*t* cannot be traded at that same close.

**The split-half ICs are reported, not the full-sample one alone.** A feature
that works in one half and not the other is a regime, not a signal, and the
average of the two hides exactly that. The full-sample number is kept beside
them, not instead of them.

Correlations are Pearson across features and IC is Spearman against returns,
deliberately: redundancy is a question about the feature vectors themselves,
where a monotone reshaping of one feature into another is *not* redundancy;
forward information is a question about ordering, where fat return tails would
otherwise let a handful of bars set the number.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy_lab.features.base import StateFeature
from strategy_lab.strategies.base import require_warmup_bars

DEFAULT_HORIZONS: tuple[int, ...] = (1, 6, 30)

# Two features correlated past this are one feature under two names -- unless
# they are one measurement deliberately exposed with both signs, which is what
# Energy and Compression are. The threshold flags the pair; only a human can say
# which of the two it is.
REDUNDANT_CORRELATION = 0.9

# Below this the Spearman estimate is mostly sampling noise, and a table full of
# confident numbers computed from a handful of bars is worse than a blank.
MIN_IC_OBSERVATIONS = 30


@dataclass(frozen=True)
class HorizonIC:
    """Forward-return information at one horizon, whole and in halves."""

    horizon: int
    ic: float
    first_half_ic: float
    second_half_ic: float
    observations: int
    first_half_observations: int
    second_half_observations: int

    @property
    def halves_agree(self) -> bool:
        """Both halves point the same way and neither is a rounding error."""
        first, second = self.first_half_ic, self.second_half_ic
        if not (np.isfinite(first) and np.isfinite(second)):
            return False
        return first * second > 0 and min(abs(first), abs(second)) >= 0.01


@dataclass(frozen=True)
class FeatureDiagnostic:
    """One feature, measured on one frame. Everything the R4 gate asks for."""

    name: str
    version: str
    warmup_bars: int
    observations: int
    coverage: float
    minimum: float
    median: float
    maximum: float
    iqr: float
    autocorrelation: float
    turnover: float
    ics: tuple[HorizonIC, ...]


@dataclass(frozen=True)
class DiagnosticSet:
    """Every feature's diagnostic plus the redundancy between them.

    Redundancy is the one question a univariate diagnostic cannot answer alone,
    so it lives here rather than on ``FeatureDiagnostic``.
    """

    horizons: tuple[int, ...]
    diagnostics: tuple[FeatureDiagnostic, ...]
    correlations: dict[str, dict[str, float]]

    def max_correlation(self, name: str) -> tuple[str, float]:
        """The feature this one tracks most closely, and the signed correlation.

        Signed rather than absolute: -1.0 and +1.0 are equally redundant and
        completely different readings, and the sign is what says which.
        """
        row = {
            other: value
            for other, value in self.correlations.get(name, {}).items()
            if other != name and np.isfinite(value)
        }
        if not row:
            return "", float("nan")
        partner = max(row, key=lambda other: abs(row[other]))
        return partner, row[partner]

    def redundant_pairs(
        self, threshold: float = REDUNDANT_CORRELATION
    ) -> list[tuple[str, str, float]]:
        """Pairs at or past ``threshold`` in absolute correlation, each listed once.

        Strongest first, so the pair most worth an argument is at the top.
        """
        names = [d.name for d in self.diagnostics]
        pairs = [
            (first, second, self.correlations[first][second])
            for index, first in enumerate(names)
            for second in names[index + 1 :]
            if np.isfinite(self.correlations.get(first, {}).get(second, np.nan))
            and abs(self.correlations[first][second]) >= threshold
        ]
        return sorted(pairs, key=lambda pair: abs(pair[2]), reverse=True)


def forward_return(close: pd.Series, *, horizon: int) -> pd.Series:
    """Return from the close of bar ``t+1`` to the close of bar ``t+1+horizon``.

    Never anchored at bar *t*'s own close. See the module docstring: that anchor
    puts the feature's own print in the denominator of its own target, and pays
    every feature that reads it.

    The last ``horizon + 1`` bars have no complete forward window and are
    ``NaN`` -- dropped by the IC rather than filled, since a shorter window is a
    different measurement wearing the same column name.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1 bar, got {horizon}")
    entry = close.shift(-1)
    exit_price = close.shift(-(horizon + 1))
    return exit_price / entry - 1.0


def forward_efficiency_ratio(close: pd.Series, *, horizon: int) -> pd.Series:
    """How *directional* the next ``horizon`` bars are, 0..1, from bar ``t+1``.

    Net displacement over the window divided by the distance price travelled to
    achieve it::

        ER[t, H] = |close[t+1+H] - close[t+1]| / sum(|close[i] - close[i-1]|)

    over ``i`` in ``(t+1, t+1+H]``. Near 1 is a clean one-way move; near 0 is a
    long path to nowhere, which is what "chop" names. It is the forward twin of
    :class:`features.trend.Strength`, which is the same ratio over the window
    that has already happened.

    **Anchored at ``close[t+1]``, exactly as :func:`forward_return` is, and here
    the anchor matters twice over.** ``close[t]`` appears in neither the
    numerator nor the path sum, so a feature that is a function of bar *t*'s
    print -- which every feature in this package is -- cannot be paid by its own
    target through either term. Anchoring at *t* would put it in both.

    A window over which price never moved has ``0 / 0``: no displacement and no
    distance is not zero efficiency, it is no measurement, and is ``NaN`` rather
    than a 0.0 that would read as perfect chop. The last ``horizon + 1`` bars
    have no complete forward window and are ``NaN`` on the same rule
    :func:`forward_return` uses.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1 bar, got {horizon}")
    entry = close.shift(-1)
    exit_price = close.shift(-(horizon + 1))
    # ``rolling(horizon)`` over the absolute steps ends on bar t+1+H and reaches
    # back to the step into t+2, which is the sum over (t+1, t+1+H] once shifted
    # onto bar t.
    path = close.diff().abs().rolling(horizon).sum().shift(-(horizon + 1))
    ratio = (exit_price - entry).abs() / path
    return ratio.where(path != 0.0)


def information_coefficient(values: pd.Series, forward: pd.Series) -> float:
    """Spearman rank correlation between a feature and a forward return.

    ``NaN`` when fewer than :data:`MIN_IC_OBSERVATIONS` bars define both, or when
    either side is constant across them -- no variation is not "no relationship",
    it is no measurement, and a 0.0 there would be read as the former.

    Computed as Pearson-on-ranks rather than through ``method="spearman"``, which
    reaches for SciPy. SciPy is present transitively and is not a declared
    dependency of this project, so a diagnostic that silently depends on it would
    break on the first clean install that resolved differently.
    """
    return _ic(_paired(values, forward))


def diagnose(
    feature: StateFeature,
    df: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> FeatureDiagnostic:
    """Measure one feature on one frame."""
    return _diagnose(feature, feature.compute(df), df["close"], horizons=horizons)


def diagnose_features(
    features: Iterable[StateFeature],
    df: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> DiagnosticSet:
    """Measure several features on one frame, and how much they duplicate each other.

    Order is preserved, so a report reads in the registry's order.
    """
    ordered = list(features)
    names = [feature.name for feature in ordered]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"Two features share the name(s) {duplicates}; the correlation matrix is "
            "keyed by name, so one would silently replace the other"
        )

    computed = {feature.name: feature.compute(df) for feature in ordered}
    diagnostics = tuple(
        _diagnose(feature, computed[feature.name], df["close"], horizons=horizons)
        for feature in ordered
    )
    # Pairwise-complete by construction: pandas drops rows where either column is
    # NaN, per pair, which is the only workable rule when Direction needs 480
    # bars of warmup and Persistence needs 96.
    matrix = pd.DataFrame(computed).corr()
    return DiagnosticSet(
        horizons=tuple(horizons),
        diagnostics=diagnostics,
        correlations={
            name: {other: float(matrix.at[name, other]) for other in names}
            for name in names
        },
    )


def to_record(result: DiagnosticSet) -> dict:
    """The whole diagnostic as JSON-safe data: ``NaN`` becomes ``null``, nothing else.

    One definition, used by both the on-disk record and the report page's
    payload, so the two cannot drift into disagreeing about the same run. NaN is
    not valid JSON, and a browser that cannot parse the payload drops the detail
    silently rather than complaining.
    """
    return {
        "horizons": list(result.horizons),
        "features": [_feature_record(diagnostic, result) for diagnostic in result.diagnostics],
        "correlations": {
            name: {other: _json_float(value) for other, value in row.items()}
            for name, row in result.correlations.items()
        },
        "redundant_pairs": [
            {"features": [first, second], "r": value}
            for first, second, value in result.redundant_pairs()
        ],
    }


def _feature_record(diagnostic: FeatureDiagnostic, result: DiagnosticSet) -> dict:
    partner, correlation = result.max_correlation(diagnostic.name)
    return {
        "name": diagnostic.name,
        "version": diagnostic.version,
        "warmup_bars": diagnostic.warmup_bars,
        "observations": diagnostic.observations,
        "coverage": _json_float(diagnostic.coverage),
        "min": _json_float(diagnostic.minimum),
        "median": _json_float(diagnostic.median),
        "max": _json_float(diagnostic.maximum),
        "iqr": _json_float(diagnostic.iqr),
        "autocorrelation": _json_float(diagnostic.autocorrelation),
        "turnover": _json_float(diagnostic.turnover),
        "max_correlation": {"feature": partner, "r": _json_float(correlation)},
        "ic": [
            {
                "horizon": entry.horizon,
                "ic": _json_float(entry.ic),
                "first_half_ic": _json_float(entry.first_half_ic),
                "second_half_ic": _json_float(entry.second_half_ic),
                "observations": entry.observations,
                "halves_agree": entry.halves_agree,
            }
            for entry in diagnostic.ics
        ],
    }


def _json_float(value: float) -> float | None:
    return None if value != value else value


def _diagnose(
    feature: StateFeature,
    values: pd.Series,
    close: pd.Series,
    *,
    horizons: Sequence[int],
) -> FeatureDiagnostic:
    """Reduce one computed feature to the numbers the R4 gate asks for.

    The warmup is checked before the slice, and the ``measured.empty`` guard
    below is not the same check however much it looks like one. A negative
    warmup turns ``iloc[warmup:]`` into a *tail* slice: measured on a 200-row
    frame, -5 leaves 5 rows rather than 195, which is non-empty and so sails
    past. Coverage, IC, turnover and the split-half comparison are then computed
    on those five bars and reported as if they covered the frame -- a wrong
    number in the research charter, which is worse than the blank the empty
    guard exists to prevent.
    """
    require_warmup_bars(feature.name, feature.warmup_bars)
    measured = values.iloc[feature.warmup_bars :]
    if measured.empty:
        raise ValueError(
            f"{feature.name} needs more than its warmup of {feature.warmup_bars} bars "
            f"and this frame has {len(values)}; every statistic would be NaN, and a "
            "table of NaNs reads as a result"
        )

    defined = measured.dropna()
    quartiles = defined.quantile([0.25, 0.75])
    return FeatureDiagnostic(
        name=feature.name,
        version=feature.version,
        warmup_bars=feature.warmup_bars,
        observations=int(len(measured)),
        coverage=float(measured.notna().mean()),
        minimum=float(defined.min()),
        median=float(defined.median()),
        maximum=float(defined.max()),
        iqr=float(quartiles.iloc[1] - quartiles.iloc[0]),
        autocorrelation=_pearson(measured, measured.shift(1)),
        # Over the measured stretch only: the step out of warmup is a jump from
        # NaN, not a bar-to-bar change anyone would trade.
        turnover=float(measured.diff().abs().mean()),
        ics=tuple(
            _horizon_ic(measured, forward_return(close, horizon=horizon), horizon=horizon)
            for horizon in horizons
        ),
    )


def _horizon_ic(values: pd.Series, forward: pd.Series, *, horizon: int) -> HorizonIC:
    paired = _paired(values, forward)
    # Split the *aligned* sample rather than the frame: the two differ by the
    # warmup at the front and the incomplete forward window at the back, and
    # splitting the frame would hand the halves uneven numbers of usable bars.
    midpoint = len(paired) // 2
    first, second = paired.iloc[:midpoint], paired.iloc[midpoint:]
    return HorizonIC(
        horizon=horizon,
        ic=_ic(paired),
        first_half_ic=_ic(first),
        second_half_ic=_ic(second),
        observations=int(len(paired)),
        first_half_observations=int(len(first)),
        second_half_observations=int(len(second)),
    )


def _paired(values: pd.Series, forward: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"feature": values, "forward": forward}).dropna()


def _ic(paired: pd.DataFrame) -> float:
    if len(paired) < MIN_IC_OBSERVATIONS:
        return float("nan")
    return _pearson(paired["feature"].rank(), paired["forward"].rank())


def _pearson(left: pd.Series, right: pd.Series) -> float:
    """Pearson correlation over rows both define, ``NaN`` if either never varies.

    The no-variation case is checked rather than left to the correlation, which
    divides by a zero standard deviation and arrives at the same ``NaN`` through
    a numpy warning.
    """
    paired = pd.DataFrame({"left": left, "right": right}).dropna()
    if paired["left"].nunique() < 2 or paired["right"].nunique() < 2:
        return float("nan")
    return float(paired["left"].corr(paired["right"]))
