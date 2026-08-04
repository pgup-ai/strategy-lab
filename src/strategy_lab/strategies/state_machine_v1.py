"""The R5 state machine, exposed through the existing ``SignalSet`` contract.

Everything this module does is plumbing between the features, the machine, the
policy and the entry/exit booleans ``vbt.Portfolio.from_signals`` understands.
The decisions live in ``state.machine`` and ``state.policy``.

Three properties a reader has to hold onto:

**``position_size`` is applied at entry only.** The engine hands it to
``from_signals``, which consumes a size on the bar that *opens* a position and
never again -- measured in R2 against the installed vectorbt. The state on the
entry bar therefore picks the size for the whole trade; a later state can close
the position but cannot scale it. See ``state.policy`` for why the charter's
exhaustion -> distribution taper is R6's problem rather than this module's.

**Two of the four inputs are trailing ranks, not raw features.** ``strength``
and ``stability`` are fed to the machine as ``rolling_percentile`` over
``rank_window`` bars, because R4's conditioning was measured by tercile and a
tercile is a rank. ``direction`` stays raw -- ranking it would destroy the sign
that is its entire content -- and ``crowding`` is already a 0..1 axis with a
meaningful neutral at 0.5, which a rank would move.

``crowding`` is the one input that can be genuinely unavailable: it needs a
``funding_rate`` column, which only perp frames carry. Rather than refuse every
spot and equity frame, the adapter runs without it and says so in
``SignalSet.metadata``. That is not the neutral 0.5 ``features.flow.Crowding``
refuses to invent -- the feature declines to *claim* a measurement, while this
records that the modulation was switched off.

The ``features`` imports are function-local, which is the one piece of ugliness
here and is load-bearing. ``features.flow`` imports ``strategies.base``, and
``strategies/__init__`` imports the registry, which imports this module: a
module-level ``from strategy_lab.features.flow import ...`` closes that loop and
fails with a partially-initialized-module ``ImportError`` whenever the first
import of the process happens to be a ``features`` one. This is the first
strategy to read a feature, so it is the first to meet the cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategy_lab.state.machine import StateMachine
from strategy_lab.state.policy import target_risk_series
from strategy_lab.strategies.base import SignalSet, require_positive_span, validate_ohlcv

# Features whose value is read as a trailing rank rather than as a level.
RANKED_FEATURES = ("strength", "stability")

# Neutral reading for a frame that carries no funding at all. 0.5 is the middle
# of Crowding's own axis, so it damps nothing -- which is the honest behaviour
# when there is nothing to damp with, provided the metadata says so.
NEUTRAL_CROWDING = 0.5

# Bars of slack past the deepest feature, for the machine itself to forget where
# its input began. This is why ``warmup_bars`` is NOT simply the max over the
# features: the machine is a recursion on top of them, so a cold start still has
# to re-synchronise after every feature it reads has converged.
#
# Measured with ``tests/test_strategy_metadata.py``'s own cold-start replay, 300
# probes on each of five seeds: at margin 0 -- warmup 1920, the features' own
# number -- the cold start disagrees on 52 to 156 of the 300 probed bars. At
# margin 60 it disagrees on none, on any seed. This is four times that, because
# the margin is cheap and the failure it prevents is a live process trading a
# state the backtest never entered.
MACHINE_CONVERGENCE_BARS = 240


@dataclass(frozen=True)
class StateMachineV1:
    """Trade the state machine's target risk, sized at entry."""

    name: str = "state_machine_v1"
    version: str = "1.0.0"
    allow_shorts: bool = True
    rank_window: int = 480
    machine: StateMachine = field(default_factory=StateMachine)
    features: tuple[str, ...] = ("direction", "strength", "stability", "crowding")
    warmup_bars: int = 0

    def __post_init__(self) -> None:
        require_positive_span(self.name, "rank_window", self.rank_window)
        object.__setattr__(self, "warmup_bars", self._warmup_bars())

    def _warmup_bars(self) -> int:
        """The deepest feature this reads, plus the rank on top, plus the machine.

        Not the largest declared lookback: a rank over ``rank_window`` cannot
        start until the feature underneath it has values to rank, so the two
        costs add exactly as they do in ``features.volatility.Energy``.
        """
        from strategy_lab.features.registry import get_feature

        deepest = 0
        for name in self.features:
            cost = get_feature(name).warmup_bars
            if name in RANKED_FEATURES:
                cost += self.rank_window - 1
            deepest = max(deepest, cost)
        return deepest + MACHINE_CONVERGENCE_BARS

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        validate_ohlcv(df)
        features, crowding_measured = self.feature_frame(df)
        states = self.machine.run(features)
        target = target_risk_series(
            states=states,
            direction=features["direction"],
            strength=features["strength"],
            crowding=features["crowding"],
        )
        if not self.allow_shorts:
            target = target.clip(lower=0.0)

        return self._signals(
            target,
            metadata={
                "allow_shorts": self.allow_shorts,
                "rank_window": self.rank_window,
                "features": list(self.features),
                "crowding_measured": crowding_measured,
                "exits": "target side change (run with --exit-mode opposite_signal_only)",
                "position_sizing": "state target risk, applied by the engine at entry only",
            },
        )

    def feature_frame(self, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """The four columns the machine reads, and whether crowding is real.

        Each feature is computed by ``features.registry`` rather than
        re-derived here, so the lookahead probe in
        ``tests/test_feature_lookahead.py`` covers the same code this runs.
        """
        from strategy_lab.features.base import rolling_percentile
        from strategy_lab.features.flow import FUNDING_COLUMN
        from strategy_lab.features.registry import get_feature

        columns: dict[str, pd.Series] = {}
        crowding_measured = FUNDING_COLUMN in df.columns
        for name in self.features:
            if name == "crowding" and not crowding_measured:
                columns[name] = pd.Series(NEUTRAL_CROWDING, index=df.index, dtype="float64")
                continue
            values = get_feature(name).compute(df)
            if name in RANKED_FEATURES:
                values = rolling_percentile(values, window=self.rank_window)
            columns[name] = values
        return pd.DataFrame(columns, index=df.index), crowding_measured

    def _signals(self, target: pd.Series, *, metadata: dict) -> SignalSet:
        """Entry and exit booleans for each change of the target's *side*.

        Only the side matters here. A target that moves 0.4 -> 0.8 while staying
        long is not an entry the engine could act on anyway -- ``from_signals``
        ignores a repeated same-direction entry under ``accumulate=False`` --
        so emitting one would put a signal in the ``signals`` table that no
        backtest ever fills.
        """
        side = np.sign(target.to_numpy(dtype="float64"))
        previous = np.concatenate([[0.0], side[:-1]])
        index = target.index

        def flags(values: np.ndarray) -> pd.Series:
            return pd.Series(values, index=index, dtype="bool")

        return SignalSet(
            long_entries=flags((side == 1.0) & (previous != 1.0)),
            long_exits=flags((side != 1.0) & (previous == 1.0)),
            short_entries=flags((side == -1.0) & (previous != -1.0)),
            short_exits=flags((side != -1.0) & (previous == -1.0)),
            position_size=target.abs(),
            metadata=metadata,
        )
