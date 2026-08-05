"""The R5 state machine, exposed through the existing ``SignalSet`` contract.

Everything this module does is plumbing between the features, the machine, the
policy and the entry/exit booleans ``vbt.Portfolio.from_signals`` understands.
The decisions live in ``state.machine`` and ``state.policy``; the feature and
target pipeline lives in ``strategies.state_machine_core``, which
``state_machine_v2`` runs too, so the only difference between the two adapters
is what each does with the target at the end.

One property a reader has to hold onto:

**``position_size`` is applied at entry only.** The engine hands it to
``from_signals``, which consumes a size on the bar that *opens* a position and
never again -- measured in R2 against the installed vectorbt. The state on the
entry bar therefore picks the size for the whole trade; a later state can close
the position but cannot scale it. That truncation is what
``state_machine_v2`` removes, on the continuous-exposure contract, without
changing a single number this adapter computes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategy_lab.state.machine import StateMachine
from strategy_lab.strategies.base import SignalSet, require_positive_span, validate_ohlcv
from strategy_lab.strategies.state_machine_core import (
    DEFAULT_FEATURES,
    RANKED_FEATURES,
    build_feature_frame,
    derive_warmup_bars,
    signed_target,
)

# ``RANKED_FEATURES`` moved to the core module with the code that reads it, and
# is re-exported because this is where a reader -- and
# ``tests/test_state_machine_strategy.py`` -- already looks for it.
__all__ = ["RANKED_FEATURES", "StateMachineV1"]


@dataclass(frozen=True)
class StateMachineV1:
    """Trade the state machine's target risk, sized at entry."""

    name: str = "state_machine_v1"
    version: str = "1.0.0"
    allow_shorts: bool = True
    rank_window: int = 480
    machine: StateMachine = field(default_factory=StateMachine)
    features: tuple[str, ...] = DEFAULT_FEATURES
    warmup_bars: int = 0

    def __post_init__(self) -> None:
        require_positive_span(self.name, "rank_window", self.rank_window)
        object.__setattr__(self, "warmup_bars", self._warmup_bars())

    def _warmup_bars(self) -> int:
        return derive_warmup_bars(
            machine=self.machine, features=self.features, rank_window=self.rank_window
        )

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        validate_ohlcv(df)
        target, crowding_measured = signed_target(
            df,
            machine=self.machine,
            features=self.features,
            rank_window=self.rank_window,
            allow_shorts=self.allow_shorts,
        )

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
        """The four columns the machine reads, and whether crowding is real."""
        return build_feature_frame(df, features=self.features, rank_window=self.rank_window)

    def _signals(self, target: pd.Series, *, metadata: dict) -> SignalSet:
        """Entry and exit booleans for each change of the target's *side*.

        Only the side matters here. A target that moves 0.4 -> 0.8 while staying
        long is not an entry the engine could act on anyway -- ``from_signals``
        ignores a repeated same-direction entry under ``accumulate=False`` --
        so emitting one would put a signal in the ``signals`` table that no
        backtest ever fills. The information in that move is not lost, only
        unusable here: ``state_machine_v2`` returns the same series whole.
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
