"""The R5 state machine again, with the collapse to booleans removed.

``state_machine_v1`` **already computes** the full signed continuous target on
every bar -- ``strategies.state_machine_core.signed_target``, the same call this
makes -- and then discards most of it, because the boolean contract has nowhere
to put a level; ``strategies.exposure`` carries the measurement that forces the
discard. The charter's per-state taper is therefore not *missing* from v1: it is
computed and thrown away.

So this is v1 with that truncation removed, and nothing else. Same machine, same
policy, same ``state.policy.STATE_TARGET_RISK`` (0.00 / 0.35 / 0.70 / 1.00 /
0.55 / 0.00 -- R5's published numbers), same four features, same warmup
derivation, and **no parameter of its own**. That is deliberate and it is what
makes the comparison against v1 legible: v1's results are already on the record,
so any degree of freedom introduced here would quietly turn "what does the
engine's truncation cost?" into a search for a v2 that wins.
``tests/test_state_machine_v2.py`` asserts both halves of that -- the field
lists match, and v2's target reproduces v1's ``position_size`` exactly.

**Exits are the target reaching 0.0**, not a separate signal. A state whose
target risk is zero -- ``COMPRESSION``, ``RESET``, or any bar the policy stands
aside on -- asks the book to hold nothing, and the engine flattens it. There is
no ``ExitMode`` on this path and no exit-mode matrix to consult: the target is
the whole instruction.

Not registered in ``strategies/registry.py``; see
``strategies/exposure_registry.py`` for why that separation is load-bearing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from strategy_lab.state.machine import StateMachine
from strategy_lab.strategies.base import require_positive_span, validate_ohlcv
from strategy_lab.strategies.exposure import TargetExposure
from strategy_lab.strategies.state_machine_core import (
    DEFAULT_FEATURES,
    build_feature_frame,
    derive_warmup_bars,
    signed_target,
)


@dataclass(frozen=True)
class StateMachineV2:
    """Hold the state machine's target risk on every bar, not just at entry."""

    name: str = "state_machine_v2"
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

    def compute_target(self, df: pd.DataFrame) -> TargetExposure:
        """The policy's signed target, handed over whole.

        No clamping or filling happens on the way out, and none is needed: the
        policy zeroes any bar whose features are unmeasurable, so warmup rows are
        already the leading run of 0.0 the contract wants rather than ``NaN``,
        and ``|state scale x damping| <= 1`` bounds the rest. ``TargetExposure``
        re-checks both, which is the point of having it check.
        """
        validate_ohlcv(df)
        target, crowding_measured = signed_target(
            df,
            machine=self.machine,
            features=self.features,
            rank_window=self.rank_window,
            allow_shorts=self.allow_shorts,
        )
        return TargetExposure(
            target=target,
            metadata={
                "allow_shorts": self.allow_shorts,
                "rank_window": self.rank_window,
                "features": list(self.features),
                "crowding_measured": crowding_measured,
                "exits": "the target reaching 0.0; there is no separate exit signal",
                "position_sizing": "state target risk, read by the engine on every bar",
            },
        )

    def feature_frame(self, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """The four columns the machine reads, and whether crowding is real.

        Byte-for-byte v1's accessor, because it is v1's pipeline. It exists so a
        reader can see *why* a target moved rather than only that it did; it
        computes nothing ``compute_target`` does not already compute and feeds
        into no decision, so the R6 comparison is untouched by it.
        """
        return build_feature_frame(df, features=self.features, rank_window=self.rank_window)


__all__ = ["StateMachineV2"]
