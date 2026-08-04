"""The market-state lifecycle, as an explicit machine over a feature frame.

Six states, walked in one direction::

    COMPRESSION -> BREAKOUT -> CONFIRMED -> RIDING -> EXHAUSTION -> RESET -> COMPRESSION

``RESET`` exists so a failed trend has somewhere to go that is not
``COMPRESSION``: the cooldown is served there, and returning straight to
``COMPRESSION`` would let the machine re-enter on the next bar of the same chop.

**The machine reads conditioners in rank space, not in feature units.** The
``strength`` column is where strength sits within its own trailing history,
0..1 -- not the raw efficiency ratio. R4's conditioning was measured by
*tercile*, and a tercile is a rank statement: on the stored BTC/USDT perp 4h
history the raw tercile boundaries are 0.063 / 0.149, and they move to
0.067 / 0.156 over the first half and 0.059 / 0.139 over the second. A
threshold written in feature units is therefore a differently sized bucket in
each era, which is the argument ``features.volatility.Energy`` already makes
for being a percentile rather than a level. ``direction`` is the exception and
stays raw: its sign is the information, and ranking a signed series destroys
it.

Three mechanisms the R5 gate names, and what each one prevents:

- **Hysteresis.** ``enter_strength`` gates every step *up* the lifecycle and
  ``exit_strength`` gates the failure that drops out of it, so between them
  lies a dead band in which nothing happens. One constant compared twice would
  make a feature hovering on it toggle the state every bar. The lifecycle is
  also strictly forward -- ``EXHAUSTION`` cannot return to ``RIDING`` -- so a
  decayed trend has to go round through ``RESET`` rather than oscillate.
- **Minimum dwell.** A step up needs its condition to have held for
  ``min_dwell`` consecutive bars *and* the current state to have lasted that
  long. Either alone is too weak: without the run length a single spike bar
  advances the machine, and without the state age a long-standing condition
  walks it through three states in three bars.
- **Cooldown.** ``RESET`` is held for ``cooldown`` bars, and ``BREAKOUT`` is
  reachable only from ``COMPRESSION``. Holding ``RESET`` *is* the "cannot
  re-enter for M bars" rule rather than a second counter beside it.

**Two hard exits bypass dwell**, because refusing to leave on a real break is
worse than churning: ``direction`` flipping against the side the move was
entered on, and ``stability`` collapsing below its floor. Both drop straight to
``RESET`` from any live state.

Every counter here **saturates**, which is load-bearing rather than incidental.
A machine is a recursion over its whole input, so without saturation the state
at bar *t* would depend on every bar back to the first one the process ever
saw, and no live process could reproduce a backtest's state: it starts cold.
Saturated, the machine is a finite automaton that forgets where it started
after any run of failing bars long enough to walk it back to ``COMPRESSION``,
which is what makes the adapter's ``warmup_bars`` mean anything at all.
``tests/test_state_machine.py::test_the_machine_forgets_where_it_started``
pins the property and ``tests/test_strategy_metadata.py`` measures how many
bars it costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from strategy_lab.strategies.base import require_positive_span

REQUIRED_COLUMNS = ("direction", "strength", "stability", "crowding")


class MarketState(Enum):
    """Where in the lifecycle of a directional move the market currently is.

    The names are the point: they are what a reader reasons about when the
    machine does something surprising, and each one is a different claim about
    what the next bars are likely to do.
    """

    COMPRESSION = "compression"
    BREAKOUT = "breakout"
    CONFIRMED = "confirmed"
    RIDING = "riding"
    EXHAUSTION = "exhaustion"
    RESET = "reset"


@dataclass(frozen=True)
class StateMachine:
    """Turns a per-bar feature frame into a per-bar :class:`MarketState`.

    Frozen and parameter-only, exactly like a ``Strategy`` or a
    ``StateFeature``: :meth:`run` starts from :attr:`INITIAL_STATE` every time
    and mutates nothing, so two runs over the same frame cannot disagree.

    ``enter_strength`` and ``exit_strength`` default to the tercile boundaries
    R4 measured against, in the rank space described in the module docstring:
    above 2/3 is the follow regime, below 1/3 is noise, and between them is the
    band where R4 found the sign inverted.
    """

    INITIAL_STATE = MarketState.COMPRESSION

    # Declared beside the machine rather than enforced inside it, on the same
    # reasoning as ``_SIDE_BY_FIELD`` in ``tests/test_replay_determinism.py``:
    # a rule the implementation checks against itself proves nothing, so this
    # stays an independent oracle for ``test_every_transition_taken_is_legal``
    # to compare the taken transitions against. Self-loops are listed because
    # staying put is the transition the machine takes on most bars.
    LEGAL_TRANSITIONS = {
        MarketState.COMPRESSION: frozenset(
            {MarketState.COMPRESSION, MarketState.BREAKOUT}
        ),
        MarketState.BREAKOUT: frozenset(
            {MarketState.BREAKOUT, MarketState.CONFIRMED, MarketState.RESET}
        ),
        MarketState.CONFIRMED: frozenset(
            {MarketState.CONFIRMED, MarketState.RIDING, MarketState.RESET}
        ),
        MarketState.RIDING: frozenset(
            {MarketState.RIDING, MarketState.EXHAUSTION, MarketState.RESET}
        ),
        MarketState.EXHAUSTION: frozenset({MarketState.EXHAUSTION, MarketState.RESET}),
        MarketState.RESET: frozenset({MarketState.RESET, MarketState.COMPRESSION}),
    }

    enter_strength: float = 2.0 / 3.0
    exit_strength: float = 1.0 / 3.0
    # A lean this small has no usable sign, so it neither starts a move nor
    # counts as having flipped against one.
    direction_floor: float = 0.10
    stability_floor: float = 0.15
    # Distance from the neutral 0.5 at which carry is extreme enough to call a
    # ride exhausted. 0.40 is the top and bottom decile of the crowding axis.
    crowding_extreme: float = 0.40
    min_dwell: int = 4
    cooldown: int = 8

    def __post_init__(self) -> None:
        for name in (
            "enter_strength",
            "exit_strength",
            "direction_floor",
            "stability_floor",
            "crowding_extreme",
        ):
            value = getattr(self, name)
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"StateMachine {name} must lie in 0..1, got {value!r}")
        if self.enter_strength <= self.exit_strength:
            raise ValueError(
                f"StateMachine enter_strength ({self.enter_strength}) must exceed "
                f"exit_strength ({self.exit_strength}); one constant compared twice "
                "is exactly the no-dead-band case hysteresis exists to avoid"
            )
        require_positive_span("StateMachine", "min_dwell", self.min_dwell)
        # Zero is a legal cooldown -- it means RESET lasts its own single bar and
        # no more -- so this cannot delegate to ``require_positive_span``.
        if isinstance(self.cooldown, bool) or not isinstance(self.cooldown, int):
            raise ValueError(f"StateMachine cooldown must be an integer, got {self.cooldown!r}")
        if self.cooldown < 0:
            raise ValueError(f"StateMachine cooldown must be >= 0, got {self.cooldown!r}")

    def run(self, features: pd.DataFrame) -> pd.Series:
        """One :class:`MarketState` per row of ``features``.

        Every column in :data:`REQUIRED_COLUMNS` must be present. A missing one
        raises rather than silently skipping whichever transition rule reads it
        -- a typo'd column name would otherwise leave a machine that still
        produces a plausible-looking state for every bar.

        A row with any feature missing is treated exactly as a failure. Warmup
        rows are ``NaN`` by construction (``features.base.mask_warmup``), and
        the alternative -- carrying the previous state across an unmeasurable
        bar -- would have the machine hold a trend it can no longer see.
        """
        _require_columns(features)
        direction = features["direction"].to_numpy(dtype="float64")
        strength = features["strength"].to_numpy(dtype="float64")
        stability = features["stability"].to_numpy(dtype="float64")
        crowding = features["crowding"].to_numpy(dtype="float64")

        measurable = (
            np.isfinite(direction)
            & np.isfinite(strength)
            & np.isfinite(stability)
            & np.isfinite(crowding)
        )
        # NaN compares False to everything, so each predicate is already False
        # on an unmeasurable bar; only ``failing`` wants the opposite.
        advancing = (strength >= self.enter_strength) & (
            np.abs(direction) >= self.direction_floor
        )
        decaying = measurable & (strength < self.enter_strength)
        failing = ~measurable | (strength < self.exit_strength)
        unstable = stability < self.stability_floor
        crowded = np.abs(crowding - 0.5) >= self.crowding_extreme

        return pd.Series(
            self._walk(
                direction=direction,
                advancing=advancing,
                decaying=decaying,
                failing=failing,
                unstable=unstable,
                crowded=crowded,
            ),
            index=features.index,
            dtype="object",
            name="state",
        )

    def _walk(
        self,
        *,
        direction: np.ndarray,
        advancing: np.ndarray,
        decaying: np.ndarray,
        failing: np.ndarray,
        unstable: np.ndarray,
        crowded: np.ndarray,
    ) -> list[MarketState]:
        """The sequential core: one pass, saturating counters, no lookahead."""
        # Counting past these changes no decision, and stopping there is what
        # makes the machine finite-state rather than a recursion with unbounded
        # memory. See the module docstring.
        age_cap = max(self.min_dwell, self.cooldown)

        state = self.INITIAL_STATE
        bars_in_state = 0
        advance_run = 0
        # +1 / -1 while a move is live: the side it was entered on, which is
        # what a later ``direction`` reading can flip *against*. A flip is only
        # meaningful relative to a commitment.
        side = 0

        states: list[MarketState] = []
        for position in range(len(direction)):
            advance_run = min(advance_run + 1, self.min_dwell) if advancing[position] else 0
            stepping_up = bars_in_state >= self.min_dwell and advance_run >= self.min_dwell

            if state is MarketState.COMPRESSION:
                following = MarketState.BREAKOUT if stepping_up else state
            elif state is MarketState.RESET:
                following = (
                    MarketState.COMPRESSION if bars_in_state >= self.cooldown else state
                )
            elif (
                failing[position]
                or unstable[position]
                or _flipped(direction[position], side, self.direction_floor)
            ):
                following = MarketState.RESET
            elif state is MarketState.RIDING and (decaying[position] or crowded[position]):
                following = MarketState.EXHAUSTION
            elif state is MarketState.BREAKOUT and stepping_up:
                following = MarketState.CONFIRMED
            elif state is MarketState.CONFIRMED and stepping_up:
                following = MarketState.RIDING
            else:
                following = state

            if following is not state:
                if following is MarketState.BREAKOUT:
                    side = 1 if direction[position] > 0 else -1
                elif following in (MarketState.RESET, MarketState.COMPRESSION):
                    side = 0
                state, bars_in_state = following, 1
            else:
                bars_in_state = min(bars_in_state + 1, age_cap)
            states.append(state)
        return states


def _flipped(direction: float, side: int, floor: float) -> bool:
    """Has ``direction`` turned far enough against the side the move was entered on?

    ``floor`` rather than a bare sign test: a lean of -0.001 against a long is
    noise, and treating it as a break would make the hard exit fire constantly
    on a trend passing through flat.
    """
    return side != 0 and direction * side <= -floor


def _require_columns(features: pd.DataFrame) -> None:
    if features.empty:
        raise ValueError("Cannot run the state machine over an empty feature frame")
    missing = [column for column in REQUIRED_COLUMNS if column not in features.columns]
    if missing:
        raise KeyError(
            f"State machine needs the column(s) {missing} and this frame has "
            f"{sorted(features.columns)}"
        )
