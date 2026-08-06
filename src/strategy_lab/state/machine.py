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
it. ``energy`` is already a percentile over that same 480-bar window, so it
arrives in the same 0..1 space without a second transformation -- ranking a rank
would be a different statistic wearing the same name.

**The entry gate has three terms, and R7b added the third.** ``advancing`` is
``strength >= enter_strength AND |direction| >= direction_floor AND energy <=
energy_ceiling``. R7 measured the first two as one gate rather than two --
``direction_floor = 0.10`` admits 82% of bars and beats ``strength`` alone by
**+0.00 pp** on the test half at every horizon -- and measured ``energy`` as the
only registered feature carrying chop information (IC vs forward efficiency
-0.0906 on BTC, -0.1521 on ETH, same sign in both halves of both).
``direction_floor`` was **joined, not replaced**: it has a second job in the
reversal test (``flipped`` below), so dropping it would change when a position
flips side, which is a different question.

**``energy_ceiling`` defaults to 1.0 and that default is inert by
construction**, which is what lets ``state_machine_v1``/``v2`` keep their
published figures rather than have them re-checked. Two facts make it so, and
both are properties of the code rather than of a particular frame:
``features.base.rolling_percentile`` is a rank over its own window, so a
measurable ``energy`` lies in ``(0, 1]`` and can never exceed a ceiling of 1.0;
and ``energy`` costs 503 warmup bars against ``direction``'s 1920, so adding it
to ``measurable`` cannot unmeasure a bar the other four had already measured.
``tests/test_state_machine.py`` pins both, and the second is pinned on a real
frame in ``tests/test_state_machine_strategy.py`` rather than only in the
abstract.

**R7c adds a second lifecycle, driven by ``energy`` instead of ``strength``,
and it is off unless both of its thresholds are set.** R7b measured the axis
this machine was built on: at H=30, ``strength >= 0.80`` lifts the trend rate
**-7.63 / +3.32** on BTC and **-1.26 / -2.52** on ETH — negative in three of
four instrument-halves, including the one R5 selected it on — while
``energy <= 0.50``, which the lifecycle does not read, lifts **+3.51 / +5.10**
and **+4.49 / +6.61**, positive in all four (M27). So the hysteresis, the dwell
and all six states sit on the weaker of the two axes. Setting ``enter_energy``
and ``exit_energy`` moves them onto the stronger one::

    advancing = (energy <= enter_energy) & (|direction| >= direction_floor)
    failing   = ~measurable | (energy > exit_energy)

Four things about that, each of which was a choice:

- **Both sides move together, or neither does.** Setting one alone is refused
  by the constructor. Leaving ``failing`` on ``strength`` would put the
  hysteresis across two features — the dead band would then be a region of a
  *plane* with no ordering between its edges, and the "one constant compared
  twice" argument below would no longer even be expressible.
- **``exit_energy > enter_energy``, the mirror of ``enter_strength >
  exit_strength``**, and enforced the same way. The inequality flips because
  the axis does: entry wants energy *low*, so the failure threshold sits
  *above* the entry one and the dead band lies between them.
- **The direction is counterintuitive and is not a sign error.** The machine
  advances on *quiet* bars. That is §2.1's own worked example — ``Strength =
  20, Energy = 95`` is "violent two-way chop" — and it reads as *the trend
  worth riding is the orderly one*.
- **``direction`` stays and ``strength`` leaves the lifecycle only.**
  ``direction`` decides side, which energy cannot, and still runs the reversal
  test. ``strength`` is still read by ``state.policy``, which conditions the
  target on it; it is the six-state walk it no longer drives.

``energy_ceiling`` keeps its meaning in either mode — it is a further ``AND``
on ``advancing``, so a machine that sets both thresholds gets their
intersection rather than one of them silently winning. At the 1.0 default it is
inert, which is why the energy-first ``advancing`` above is exactly the two
terms written there.

Three mechanisms the R5 gate names, and what each one prevents:

- **Hysteresis.** ``enter_strength`` gates every step *up* the lifecycle and
  ``exit_strength`` gates the failure that drops out of it, so between them
  lies a dead band in which the machine does not step up and does not fail.
  One constant compared twice would make a feature hovering on it toggle the
  state every bar. The dead band is where the machine declines to *climb*,
  not where nothing at all happens -- see the bounded exits below, which is
  the distinction that took a bug to find. The lifecycle is also strictly
  forward -- ``EXHAUSTION`` cannot return to ``RIDING`` -- so a decayed trend
  has to go round through ``RESET`` rather than oscillate.
- **Minimum dwell, both ways.** A step up needs ``advancing`` to have held for
  ``min_dwell`` consecutive bars *and* the current state to have lasted that
  long. Either alone is too weak: without the run length a single spike bar
  advances the machine, and without the state age a long-standing condition
  walks it through three states in three bars. The same count runs the other
  way: ``min_dwell`` consecutive *non*-advancing bars drop a ``BREAKOUT`` or
  ``CONFIRMED`` that never became a ride back to ``RESET``.
- **Cooldown.** ``RESET`` is held for ``cooldown`` bars, and ``BREAKOUT`` is
  reachable only from ``COMPRESSION``. Holding ``RESET`` *is* the "cannot
  re-enter for M bars" rule rather than a second counter beside it.

**Two hard exits bypass dwell**, because refusing to leave on a real break is
worse than churning: ``direction`` flipping against the side the move was
entered on, and ``stability`` collapsing below its floor. Both drop straight to
``RESET`` from any live state.

**Every state has a bounded exit, and that is what makes a live process able to
reproduce a backtest.** A machine is a recursion over its whole input, so the
state at bar *t* can in principle depend on every bar back to the first one the
process ever saw -- and a live process starts cold where a backtest arrives
carrying years. Counters saturating is necessary for the machine to be a finite
automaton but it is not sufficient: a state nothing routes *out of* remembers
its own arrival forever. Measured on the pre-fix machine, a persistent tail of
``direction=0.8, strength=0.5, stability=0.9, crowding=0.5`` left a cold start
in ``COMPRESSION`` and a warm one in ``EXHAUSTION``, permanently, for a target
risk of 0.0 against -0.55 -- and 14 of 48 sampled constant tails had more than
one attractor, ``BREAKOUT`` and ``CONFIRMED`` parking in the dead band the same
way. Three rules close it, and each is also the behaviour the lifecycle wanted:

- A ``BREAKOUT`` or ``CONFIRMED`` that stops advancing for ``min_dwell`` bars
  has failed as a setup and drops to ``RESET``. It had already lost every bar
  of progress toward its next step -- ``advance_run`` resets on any
  non-advancing bar -- so what it kept was a label, not a position in the
  lifecycle.
- ``RIDING`` ends on any bar that is not ``advancing``, which is the old
  "strength left the top band" rule plus the case that parked it: a lean that
  has decayed below ``direction_floor``. A ride with no direction is not a ride.
- ``EXHAUSTION`` runs out after ``exhaustion_dwell`` bars. It is the one
  transition on a plain timer, because a move that has already decayed has no
  condition left to wait on.

Together with a ``COMPRESSION -> BREAKOUT`` gate that now refuses the
conditions which would immediately end a move, this gives the invariant
``tests/test_state_machine.py::test_a_constant_tail_converges_from_any_start``
pins: **from any starting configuration, a constant tail reaches the same state
within** :attr:`StateMachine.convergence_bars` **bars.** That is what
``warmup_bars`` is denominated in; ``tests/test_strategy_metadata.py`` measures
what it costs on real-shaped input.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("direction", "strength", "stability", "crowding", "energy")


class MarketState(Enum):
    """Where in the lifecycle of a directional move the market currently is."""

    COMPRESSION = "compression"
    BREAKOUT = "breakout"
    CONFIRMED = "confirmed"
    RIDING = "riding"
    EXHAUSTION = "exhaustion"
    RESET = "reset"


@dataclass(frozen=True)
class StateMachine:
    """Turns a per-bar feature frame into a per-bar :class:`MarketState`.

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
    # to compare the taken transitions against.
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
    # Rank space over the same 480-bar window the other conditioners are ranked
    # in, so this is "how hot is realized vol against its own recent history".
    # It gates *entry to the lifecycle* -- see ``advancing`` in ``run`` -- and
    # 1.0 admits every bar, which is why it is the default: a machine that has
    # not chosen a ceiling behaves exactly as the one R5 published.
    #
    # The direction of the inequality is the one thing here that reads
    # backwards from the name. R7 measured IC(`energy`, forward efficiency) at
    # -0.0906 on BTC and -0.1521 on ETH, same sign in both halves of both: high
    # energy precedes *less* efficient forward movement, so the chop side of the
    # axis is the top of it and a ceiling is what excludes chop. §2.1's own
    # worked example is the same statement -- `Strength = 20, Energy = 95` is
    # "violent two-way chop" -- and the feature named `compression` is `1 -
    # energy`, so the state called COMPRESSION and the feature called
    # compression remain unrelated.
    energy_ceiling: float = 1.0
    # R7c's energy-first lifecycle, in the same rank space. ``None`` on both is
    # the strength-driven machine every published figure was produced by, and
    # is the only configuration in which ``energy`` does not drive the walk --
    # which is what makes the default provably inert rather than inert by
    # inspection. Setting exactly one is refused: see the module docstring.
    enter_energy: float | None = None
    exit_energy: float | None = None
    # Rank space, so this is the bottom 15% of trailing stability -- a collapse
    # relative to how cleanly this instrument usually tracks its own trend line,
    # rather than an absolute residual that means different things per market.
    stability_floor: float = 0.15
    # Distance from the neutral 0.5 at which carry is extreme enough to call a
    # ride exhausted. Crowding is a tanh, so it bunches near its rails: measured
    # on the 14,944 stored BTC/USDT perp 4h bars that carry funding, the median
    # |crowding - 0.5| is already 0.313 and a 0.40 threshold fires on 31.2% of
    # bars. 0.475 is the 90th percentile, which is what "extreme" has to mean if
    # it is going to end a ride on its own.
    crowding_extreme: float = 0.475
    min_dwell: int = 4
    cooldown: int = 8
    # Bars ``EXHAUSTION`` may last before the move is simply over. 12 is three
    # times the default ``min_dwell``: the machine gives a move as long to end
    # as it took to establish, since COMPRESSION -> BREAKOUT -> CONFIRMED ->
    # RIDING is exactly three dwell periods. A field rather than an expression
    # over ``min_dwell`` so a sweep can move it on its own; the R5 grid does not.
    exhaustion_dwell: int = 12

    def __post_init__(self) -> None:
        for name in (
            "enter_strength",
            "exit_strength",
            "direction_floor",
            "stability_floor",
            "crowding_extreme",
            "energy_ceiling",
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
        if (self.enter_energy is None) != (self.exit_energy is None):
            raise ValueError(
                "StateMachine enter_energy and exit_energy must be set together "
                f"or not at all, got {self.enter_energy!r} and {self.exit_energy!r}; "
                "one side on energy and the other on strength puts the hysteresis "
                "across two features, which is not a dead band"
            )
        if self.energy_first:
            for name in ("enter_energy", "exit_energy"):
                value = getattr(self, name)
                if not 0.0 <= float(value) <= 1.0:
                    raise ValueError(
                        f"StateMachine {name} must lie in 0..1, got {value!r}"
                    )
            if self.exit_energy <= self.enter_energy:
                raise ValueError(
                    f"StateMachine exit_energy ({self.exit_energy}) must exceed "
                    f"enter_energy ({self.enter_energy}); the inequality is the "
                    "mirror of enter_strength > exit_strength because the axis is "
                    "inverted -- entry wants energy low, so failure sits above it"
                )
        _require_bar_count("min_dwell", self.min_dwell, minimum=1)
        # Zero is a legal cooldown: it means RESET lasts its own single bar.
        _require_bar_count("cooldown", self.cooldown, minimum=0)
        # One, not zero. Both states are held for at least the bar they are
        # entered on, so 0 and 1 are the same machine either way -- but
        # ``cooldown=0`` reads as "no cooldown" and is worth being able to say,
        # while ``exhaustion_dwell=0`` reads as an EXHAUSTION that never happens,
        # which is not what it would do.
        _require_bar_count("exhaustion_dwell", self.exhaustion_dwell, minimum=1)

    @property
    def energy_first(self) -> bool:
        """Whether ``energy`` drives the lifecycle in place of ``strength``.

        Derived from the thresholds rather than carried as a separate flag, so
        there is no state in which a machine claims one mode and is configured
        for the other. ``__post_init__`` has already refused the half-set case
        by the time anything reads this.
        """
        return self.enter_energy is not None

    @property
    def convergence_bars(self) -> int:
        """Bars after which a constant tail has erased where the machine began.

        The longest walk this configuration allows between the two attractors a
        constant tail can have: ``EXHAUSTION`` runs out its dwell, ``RESET``
        serves its cooldown, and the lifecycle then climbs three dwell periods
        back to ``RIDING``. Two spare bars cover the transitions that fire on
        entry rather than on age. Measured exhaustively against an enumeration
        of tails and starting configurations in
        ``tests/test_state_machine.py``, which also asserts the bound is not
        loose by more than a factor of two.

        This is the *proven* bound and it holds for constant tails only. Real
        input is not constant, and the machine's memory on realistic input is
        longer -- which is why the strategy adapter multiplies this rather than
        using it directly.
        """
        return self.exhaustion_dwell + self.cooldown + 3 * self.min_dwell + 2

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
        energy = features["energy"].to_numpy(dtype="float64")

        measurable = (
            np.isfinite(direction)
            & np.isfinite(strength)
            & np.isfinite(stability)
            & np.isfinite(crowding)
            & np.isfinite(energy)
        )
        # NaN compares False to everything, so each predicate is already False
        # on an unmeasurable bar; only ``failing`` wants the opposite. That
        # holds on both axes: ``energy > exit_energy`` is False on a NaN energy
        # exactly as ``strength < exit_strength`` is on a NaN strength, and
        # ``~measurable`` is what turns either into a failure.
        if self.energy_first:
            entering, leaving = energy <= self.enter_energy, energy > self.exit_energy
        else:
            entering = strength >= self.enter_strength
            leaving = strength < self.exit_strength
        advancing = (
            entering
            & (np.abs(direction) >= self.direction_floor)
            & (energy <= self.energy_ceiling)
        )
        failing = ~measurable | leaving
        unstable = stability < self.stability_floor
        crowded = np.abs(crowding - 0.5) >= self.crowding_extreme

        return pd.Series(
            self._walk(
                direction=direction,
                advancing=advancing,
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
        failing: np.ndarray,
        unstable: np.ndarray,
        crowded: np.ndarray,
    ) -> list[MarketState]:
        """The sequential core: one pass, saturating counters, no lookahead."""
        # Counting past these changes no decision; saturating there is what makes
        # the machine finite-state. ``exhaustion_dwell`` belongs in the cap or a
        # dwell longer than the other two would never be reached.
        age_cap = max(self.min_dwell, self.cooldown, self.exhaustion_dwell)

        state = self.INITIAL_STATE
        bars_in_state = 0
        advance_run = 0
        stall_run = 0
        # +1 / -1 while a move is live: the side it was entered on, which is
        # what a later ``direction`` reading can flip *against*.
        side = 0

        states: list[MarketState] = []
        for position in range(len(direction)):
            if advancing[position]:
                advance_run, stall_run = min(advance_run + 1, self.min_dwell), 0
            else:
                advance_run, stall_run = 0, min(stall_run + 1, self.min_dwell)
            stepping_up = bars_in_state >= self.min_dwell and advance_run >= self.min_dwell
            stalled = stall_run >= self.min_dwell
            flipped = side != 0 and direction[position] * side <= -self.direction_floor
            # A move may only start under the conditions that would let it
            # continue. Gating the entry on less than the exits check is what
            # let an unstable or crowded tail cycle COMPRESSION -> BREAKOUT ->
            # RESET forever, at a phase a cold start could not reproduce.
            admissible = not (failing[position] or unstable[position] or crowded[position])

            if state is MarketState.COMPRESSION:
                following = MarketState.BREAKOUT if stepping_up and admissible else state
            elif state is MarketState.RESET:
                following = (
                    MarketState.COMPRESSION if bars_in_state >= self.cooldown else state
                )
            elif failing[position] or unstable[position] or flipped:
                following = MarketState.RESET
            elif state is MarketState.EXHAUSTION:
                following = (
                    MarketState.RESET if bars_in_state >= self.exhaustion_dwell else state
                )
            elif state is MarketState.RIDING:
                following = (
                    MarketState.EXHAUSTION
                    if not advancing[position] or crowded[position]
                    else state
                )
            # Only BREAKOUT and CONFIRMED reach here; the other four states are
            # each handled above, so the two branches below are exhaustive.
            elif stepping_up:
                following = (
                    MarketState.CONFIRMED
                    if state is MarketState.BREAKOUT
                    else MarketState.RIDING
                )
            elif stalled:
                following = MarketState.RESET
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


def _require_bar_count(field: str, value: object, *, minimum: int) -> None:
    """Reject a dwell or cooldown that would quietly become no rule at all.

    ``bool`` is rejected separately because it is an ``int`` subclass: without
    that, ``min_dwell=True`` passes and silently behaves as ``min_dwell=1``.
    This restates ``strategies.base.require_positive_span`` rather than
    importing it, because ``strategies`` imports ``state`` and the reverse edge
    would close the loop.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"StateMachine {field} must be an integer >= {minimum}, got {value!r}"
        )


def _require_columns(features: pd.DataFrame) -> None:
    if features.empty:
        raise ValueError("Cannot run the state machine over an empty feature frame")
    missing = [column for column in REQUIRED_COLUMNS if column not in features.columns]
    if missing:
        raise KeyError(
            f"State machine needs the column(s) {missing} and this frame has "
            f"{sorted(features.columns)}"
        )
