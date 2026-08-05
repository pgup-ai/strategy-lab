"""The feature-and-target pipeline both state-machine adapters run.

``state_machine_v1`` and ``state_machine_v2`` are the same machine, the same
policy and the same four features, differing only in what they do with the
signed target at the end: v1 collapses it to entry/exit booleans plus an
entry-only size, v2 hands it over whole. That collapse is the *only* difference
R6 sets out to measure, so the two must not each own a pipeline -- a second copy
would drift, and a v1-vs-v2 comparison would then be measuring pipeline drift
alongside the taper with no way to tell the two apart. Everything above the
final adapter step therefore lives here and is called by both.

The ``features`` imports are function-local, which is the one piece of ugliness
here and is load-bearing. ``features.flow`` imports ``strategies.base``, and
``strategies/__init__`` imports the registry, which imports the adapters, which
import this module: a module-level ``from strategy_lab.features.flow import ...``
closes that loop and fails with a partially-initialized-module ``ImportError``
whenever the first import of the process happens to be a ``features`` one.

**Two of the four inputs are trailing ranks, not raw features.** ``strength``
and ``stability`` are fed to the machine as ``rolling_percentile`` over
``rank_window`` bars, because R4's conditioning was measured by tercile and a
tercile is a rank. ``direction`` stays raw -- ranking it would destroy the sign
that is its entire content -- and ``crowding`` is already a 0..1 axis with a
meaningful neutral at 0.5, which a rank would move.

``crowding`` is the one input that can be genuinely unavailable: it needs a
``funding_rate`` column, which only perp frames carry. Rather than refuse every
spot and equity frame, this runs without it and reports that in the flag it
returns, which each adapter puts in its own metadata. That is not the neutral
0.5 ``features.flow.Crowding`` refuses to invent -- the feature declines to
*claim* a measurement, while the flag records that the modulation was switched
off.
"""

from __future__ import annotations

import pandas as pd

from strategy_lab.state.machine import StateMachine
from strategy_lab.state.policy import target_risk_series

# The four columns the machine reads, in the order the adapters declare them.
DEFAULT_FEATURES = ("direction", "strength", "stability", "crowding")

# Features whose value is read as a trailing rank rather than as a level.
RANKED_FEATURES = ("strength", "stability")

# Neutral reading for a frame that carries no funding at all. 0.5 is the middle
# of Crowding's own axis, so it damps nothing -- which is the honest behaviour
# when there is nothing to damp with, provided the metadata says so.
NEUTRAL_CROWDING = 0.5

# Multiples of the machine's own ``convergence_bars`` to add past the deepest
# feature, so the machine has time to forget where its input began. This is why
# the derived warmup is NOT simply the max over the features: the machine is a
# recursion on top of them, so a cold start still has to re-synchronise after
# every feature it reads has converged.
#
# A multiple rather than a constant, because the cost scales with the machine
# that is actually configured -- ``StateMachine(min_dwell=1000)`` needs
# thousands of bars where the default needs a few hundred, and
# ``sweep_parameters`` reaches this through ``dataclasses.replace``.
#
# Measured with ``tests/test_strategy_metadata.py``'s own cold-start replay, 120
# probes on each of three seeds, on three machines: the default, one at
# ``min_dwell=2, cooldown=16``, and a deliberately slow ``min_dwell=8,
# cooldown=24, exhaustion_dwell=30`` (convergence bounds 34 / 36 / 80). At
# margin 0 -- warmup 1920, the features' own number -- the cold start disagrees
# on 27 to 64 of the probed bars. At 2x ``convergence_bars`` it disagrees on
# none, for any of the three, on any seed. This is four times that, because the
# margin is cheap and the failure it prevents is a live process trading a state
# the backtest never entered.
MACHINE_CONVERGENCE_MULTIPLE = 8


def build_feature_frame(
    df: pd.DataFrame, *, features: tuple[str, ...], rank_window: int
) -> tuple[pd.DataFrame, bool]:
    """The columns the machine reads, and whether crowding is real.

    Each feature is computed by ``features.registry`` rather than re-derived
    here, so the lookahead probe in ``tests/test_feature_lookahead.py`` covers
    the same code this runs.
    """
    from strategy_lab.features.base import rolling_percentile
    from strategy_lab.features.flow import FUNDING_COLUMN
    from strategy_lab.features.registry import get_feature

    columns: dict[str, pd.Series] = {}
    crowding_measured = FUNDING_COLUMN in df.columns
    for name in features:
        if name == "crowding" and not crowding_measured:
            columns[name] = pd.Series(NEUTRAL_CROWDING, index=df.index, dtype="float64")
            continue
        values = get_feature(name).compute(df)
        if name in RANKED_FEATURES:
            values = rolling_percentile(values, window=rank_window)
        columns[name] = values
    return pd.DataFrame(columns, index=df.index), crowding_measured


def signed_target(
    df: pd.DataFrame,
    *,
    machine: StateMachine,
    features: tuple[str, ...],
    rank_window: int,
    allow_shorts: bool,
) -> tuple[pd.Series, bool]:
    """The policy's signed target risk for every bar, and the crowding flag.

    This is the whole of what a state-machine strategy decides. v1 keeps
    ``np.sign`` of it for booleans and ``abs`` of it as an entry-only size; v2
    returns it unchanged. Both read the same series, which is what makes the
    difference between them exactly the collapse and nothing else.
    """
    frame, crowding_measured = build_feature_frame(
        df, features=features, rank_window=rank_window
    )
    states = machine.run(frame)
    target = target_risk_series(
        states=states,
        direction=frame["direction"],
        strength=frame["strength"],
        crowding=frame["crowding"],
    )
    if not allow_shorts:
        target = target.clip(lower=0.0)
    return target, crowding_measured


def derive_warmup_bars(
    *, machine: StateMachine, features: tuple[str, ...], rank_window: int
) -> int:
    """The deepest feature read, plus the rank on top, plus the machine.

    Not the largest declared lookback: a rank over ``rank_window`` cannot start
    until the feature underneath it has values to rank, so the two costs add
    exactly as they do in ``features.volatility.Energy``.

    The machine's share is read off the machine rather than pinned, because it
    is genuinely a function of the configuration: a cold machine with a large
    ``min_dwell`` needs that many advancing bars per lifecycle step before it
    can be where a whole-history machine already is.
    """
    from strategy_lab.features.registry import get_feature

    deepest = 0
    for name in features:
        cost = get_feature(name).warmup_bars
        if name in RANKED_FEATURES:
            cost += rank_window - 1
        deepest = max(deepest, cost)
    return deepest + MACHINE_CONVERGENCE_MULTIPLE * machine.convergence_bars


__all__ = [
    "DEFAULT_FEATURES",
    "MACHINE_CONVERGENCE_MULTIPLE",
    "NEUTRAL_CROWDING",
    "RANKED_FEATURES",
    "build_feature_frame",
    "derive_warmup_bars",
    "signed_target",
]
