from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_lab.features.base import rolling_percentile
from strategy_lab.state.machine import REQUIRED_COLUMNS, MarketState, StateMachine


def frame(**columns) -> pd.DataFrame:
    """A feature frame from whatever columns are named -- and only those.

    No defaults on purpose: the machine's input contract is that every column
    it reads is present, and a helper that filled the gaps would make the
    rejection test below untestable.
    """
    n = len(next(iter(columns.values())))
    index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC", name="timestamp")
    return pd.DataFrame(columns, index=index)


def quiet(n: int, **overrides) -> pd.DataFrame:
    """A complete frame of unremarkable readings, for tests that vary one column.

    ``energy`` is 0.1 rather than a midpoint because "unremarkable" here means
    "blocks nothing": it has to sit under every ceiling any test below sets, and
    a 0.5 reading would land exactly *on* the 0.5 ceiling the energy tests use,
    where ``<=`` admits it and a reader would have to check which way the
    boundary falls to know what the test was asserting.
    """
    columns = {
        "direction": [0.0] * n,
        "strength": [0.1] * n,
        "stability": [0.9] * n,
        "crowding": [0.5] * n,
        "energy": [0.1] * n,
    }
    columns.update(overrides)
    return frame(**columns)


def wandering(n: int, *, seed: int, window: int = 200) -> np.ndarray:
    """An autocorrelated 0..1 series shaped like what the adapter actually feeds.

    The machine's inputs are trailing *ranks* of slow rolling statistics, so
    they drift rather than jump. Drawing them iid would look like a harder test
    and be a weaker one: the lifecycle needs several consecutive advancing bars
    to reach ``RIDING`` at all, and under iid draws it essentially never gets
    there, leaving the deeper states unexercised.
    """
    rng = np.random.default_rng(seed)
    walk = pd.Series(np.cumsum(rng.normal(0.0, 1.0, n + window)))
    return rolling_percentile(walk, window=window).to_numpy()[window:]


def test_the_machine_starts_flat_rather_than_guessing():
    """Bar 0 has no history behind it, so the machine may not claim a trend."""
    states = StateMachine().run(quiet(20, direction=[0.9] * 20, strength=[0.95] * 20))
    assert states.iloc[0] is MarketState.COMPRESSION


def test_a_state_is_produced_for_every_bar():
    features = quiet(20)
    states = StateMachine().run(features)
    assert len(states) == len(features)
    assert states.index.equals(features.index)


def test_an_unknown_feature_column_is_rejected_rather_than_ignored():
    """A typo'd column name must not silently disable a transition rule."""
    with pytest.raises(KeyError, match="strength"):
        StateMachine().run(
            frame(
                direction=[0.0] * 20,
                stability=[0.9] * 20,
                crowding=[0.5] * 20,
                energy=[0.1] * 20,
            )
        )


def test_thresholds_that_collapse_the_dead_band_are_rejected():
    with pytest.raises(ValueError, match="enter_strength"):
        StateMachine(enter_strength=0.30, exit_strength=0.30)


def test_hysteresis_lets_a_setup_survive_a_dip_that_a_single_threshold_would_kill():
    """What the dead band buys: a dip below ``enter_strength`` stalls, not fails.

    Two bars at 0.5 sit under the 2/3 entry and over the 1/3 exit. With the dead
    band the ``BREAKOUT`` they land on merely stops climbing for two bars and
    then goes on to ``RIDING``; with the thresholds all but collapsed
    (2/3 against 0.66) the same two bars are outright failures and the machine
    is knocked to ``RESET`` and has to serve a cooldown before it can even
    re-break out.

    Note this is a claim about *stalling*, not about nothing happening. A dip
    lasting ``min_dwell`` bars does drop the setup -- see
    ``test_a_setup_that_stops_advancing_fails_rather_than_parking``, which is
    what makes the machine converge at all.
    """
    dipping = [0.9] * 8 + [0.5] * 2 + [0.9] * 14
    features = quiet(24, direction=[0.8] * 24, strength=dipping)

    with_band = StateMachine(min_dwell=4, cooldown=8).run(features)
    without_band = StateMachine(
        enter_strength=2.0 / 3.0, exit_strength=0.66, min_dwell=4, cooldown=8
    ).run(features)

    assert MarketState.RESET not in set(with_band), "a dead-band bar knocked the setup back"
    assert with_band.iloc[-1] is MarketState.RIDING
    assert without_band.iloc[10] is MarketState.RESET, (
        "setup failed: the collapsed thresholds were supposed to fail on the dip"
    )
    assert MarketState.RIDING not in set(without_band)


def test_minimum_dwell_blocks_a_one_bar_spike_from_advancing_the_machine():
    """The bars around the spike sit in the dead band, not below ``exit_strength``.

    Putting them at 0.1 instead would make the machine reset on every one of
    them, and the state would then hold across the spike for a reason that has
    nothing to do with dwell.
    """
    machine = StateMachine(min_dwell=5)
    spike = [0.5] * 10 + [0.9] + [0.5] * 10
    states = machine.run(quiet(21, direction=[0.8] * 21, strength=spike))
    assert states.iloc[10] is states.iloc[9], "a single bar advanced the state despite dwell"


def test_dwell_paces_the_lifecycle_rather_than_letting_it_run_in_three_bars():
    """The other half of dwell: the state itself must also have lasted.

    Without it, a condition that has already been true for a while satisfies
    every step at once and the machine walks COMPRESSION to RIDING one bar at a
    time.
    """
    min_dwell = 5
    machine = StateMachine(min_dwell=min_dwell)
    states = machine.run(quiet(30, direction=[0.8] * 30, strength=[0.9] * 30))
    first_breakout = int(np.argmax((states == MarketState.BREAKOUT).to_numpy()))
    first_riding = int(np.argmax((states == MarketState.RIDING).to_numpy()))
    assert states.iloc[first_riding] is MarketState.RIDING, "setup failed: never rode"
    assert first_riding - first_breakout >= 2 * min_dwell, (
        f"BREAKOUT to RIDING took {first_riding - first_breakout} bars, which is "
        f"less than the two dwell periods of {min_dwell} bars it has to cross"
    )


def test_cooldown_prevents_immediate_re_entry_after_a_reset():
    cooldown = 8
    machine = StateMachine(cooldown=cooldown, min_dwell=1)
    pattern = [0.9] * 6 + [0.05] * 3 + [0.9] * 20        # trend, failure, immediate retry
    states = machine.run(quiet(29, direction=[0.8] * 29, strength=pattern))

    reset_bar = int(np.argmax((states == MarketState.RESET).to_numpy()))
    assert states.iloc[reset_bar] is MarketState.RESET, "setup failed: the trend never reset"
    blocked = states.iloc[reset_bar + 1 : reset_bar + 1 + cooldown]
    assert (blocked != MarketState.BREAKOUT).all(), "re-entered during cooldown"
    assert (states.iloc[reset_bar + 1 + cooldown :] == MarketState.BREAKOUT).any(), (
        "never re-entered at all, so the cooldown window above proves nothing"
    )


def test_a_direction_flip_while_riding_exits_immediately_despite_dwell():
    """Refusing to leave on a real break is worse than churning."""
    machine = StateMachine(min_dwell=4, cooldown=0)
    direction = [0.8] * 40 + [-0.8] * 10
    states = machine.run(quiet(50, direction=direction, strength=[0.9] * 50))
    assert states.iloc[39] is MarketState.RIDING, "setup failed: never reached RIDING"
    assert states.iloc[40] is MarketState.RESET


def test_a_stability_collapse_exits_immediately_despite_dwell():
    machine = StateMachine(min_dwell=4, cooldown=0)
    stability = [0.9] * 40 + [0.02] * 10
    states = machine.run(
        quiet(50, direction=[0.8] * 50, strength=[0.9] * 50, stability=stability)
    )
    assert states.iloc[39] is MarketState.RIDING, "setup failed: never reached RIDING"
    assert states.iloc[40] is MarketState.RESET


def test_extreme_crowding_ends_a_ride_while_strength_still_holds():
    """Crowding is the only non-price input the machine has.

    If an extreme reading changes no transition, the column is decoration --
    and the strength here never leaves the top tercile, so nothing else could
    have ended the ride.
    """
    machine = StateMachine(min_dwell=4, cooldown=0)
    crowding = [0.5] * 40 + [0.99] * 10
    states = machine.run(
        quiet(50, direction=[0.8] * 50, strength=[0.9] * 50, crowding=crowding)
    )
    assert states.iloc[39] is MarketState.RIDING, "setup failed: never reached RIDING"
    assert states.iloc[40] is MarketState.EXHAUSTION


def test_a_ride_whose_lean_decays_ends_even_with_strength_intact():
    """``advancing`` has two halves and both have to be able to end a ride.

    Strength stays in the top band throughout, so the old "strength left the top
    tercile" rule never fires. What goes is ``direction``: 0.02 is under the
    0.10 floor, which is too small to have a usable sign and therefore too small
    to be riding. Before this, such a tail parked ``RIDING`` forever.
    """
    machine = StateMachine(min_dwell=4, cooldown=0)
    direction = [0.8] * 40 + [0.02] * 10
    states = machine.run(quiet(50, direction=direction, strength=[0.9] * 50))
    assert states.iloc[39] is MarketState.RIDING, "setup failed: never reached RIDING"
    assert states.iloc[40] is MarketState.EXHAUSTION


def test_a_setup_that_stops_advancing_fails_rather_than_parking():
    """``min_dwell`` runs both ways: it steps the machine up and it drops it out.

    The tail sits in the dead band, so nothing fails and nothing advances. A
    ``BREAKOUT`` that keeps its label there is keeping a position in the
    lifecycle it has no claim to -- ``advance_run`` has already been reset, so
    it is no closer to ``CONFIRMED`` than a fresh ``COMPRESSION`` is.
    """
    min_dwell = 4
    machine = StateMachine(min_dwell=min_dwell, cooldown=2)
    strength = [0.9] * 5 + [0.5] * 20
    states = machine.run(quiet(25, direction=[0.8] * 25, strength=strength))
    assert states.iloc[4] is MarketState.BREAKOUT, "setup failed: never broke out"
    assert states.iloc[4 + min_dwell] is MarketState.RESET, (
        f"a stalled BREAKOUT was still held {min_dwell} bars later: {list(states)}"
    )


def test_exhaustion_ends_on_its_own_timer_rather_than_waiting_for_a_failure():
    """The one transition on a plain timer, and the one the bug lived in.

    The tail neither fails, destabilises nor flips, so every *conditional* exit
    from ``EXHAUSTION`` stays shut. Without the dwell the machine sits here for
    the rest of the input, which is a memory of where the run began that no
    ``warmup_bars`` can erase.
    """
    exhaustion_dwell = 6
    machine = StateMachine(min_dwell=4, cooldown=0, exhaustion_dwell=exhaustion_dwell)
    strength = [0.9] * 40 + [0.5] * 30
    states = machine.run(quiet(70, direction=[0.8] * 70, strength=strength))

    assert states.iloc[40] is MarketState.EXHAUSTION, "setup failed: never exhausted"
    held = int((states.iloc[40:] == MarketState.EXHAUSTION).sum())
    assert held == exhaustion_dwell, f"EXHAUSTION lasted {held} bars, not {exhaustion_dwell}"
    assert states.iloc[40 + exhaustion_dwell] is MarketState.RESET


@pytest.mark.parametrize("blocker", ["stability", "crowding"])
def test_a_breakout_is_refused_under_a_condition_that_would_end_the_move(blocker):
    """The entry gate checks what the exits check, or the machine cycles.

    Gated on ``advancing`` alone, a tail that is strong *and* unstable walks
    COMPRESSION -> BREAKOUT -> RESET -> COMPRESSION forever, at a phase that
    depends on where the run started. Both blockers are conditions the machine
    already treats as ending a live move.
    """
    values = {"stability": [0.02] * 60, "crowding": [0.99] * 60}[blocker]
    states = StateMachine(min_dwell=4, cooldown=8).run(
        quiet(60, direction=[0.8] * 60, strength=[0.95] * 60, **{blocker: values})
    )
    assert set(states) == {MarketState.COMPRESSION}, (
        f"a {blocker} blocker let the machine into {sorted({s.name for s in states})}"
    )


def test_an_unmeasurable_bar_is_treated_as_a_failure_not_as_no_news():
    """Warmup rows are NaN, and a machine that coasts through them holds a trend
    it can no longer see."""
    machine = StateMachine(min_dwell=4, cooldown=0)
    strength = [0.9] * 40 + [np.nan] * 10
    states = machine.run(quiet(50, direction=[0.8] * 50, strength=strength))
    assert states.iloc[39] is MarketState.RIDING, "setup failed: never reached RIDING"
    assert states.iloc[40] is MarketState.RESET


def test_the_default_energy_ceiling_is_provably_inert():
    """R7b's change has to cost the published figures nothing, by construction.

    ``energy`` is a ``rolling_percentile``, so a measurable reading lies in
    ``(0, 1]`` and can never exceed the 1.0 default -- which means the default
    machine's states cannot depend on the column's *values* at all. Asserted by
    driving the whole 0..1 range through it and comparing against a frame where
    every energy reading is the same number: identical states, on input that
    walks the entire lifecycle.

    This is the assertion R7b's no-op control rests on, and it is the cheap half
    of it. The expensive half is the engine reproducing +15.45% / +0.896 /
    4.67% / 73 trades on BTC, which needs Postgres and lives in
    ``scripts/r7b/step0_control.py``.
    """
    n = 2000
    varying = frame(
        direction=2.0 * wandering(n, seed=41) - 1.0,
        strength=wandering(n, seed=42),
        stability=wandering(n, seed=43),
        crowding=wandering(n, seed=44),
        energy=wandering(n, seed=45),
    )
    assert StateMachine().energy_ceiling == 1.0
    assert varying["energy"].max() > 0.99, "the probe never approached the ceiling"

    pinned = varying.assign(energy=0.0)
    states = StateMachine().run(varying)
    assert states.nunique() == len(MarketState), "the probe never walked the lifecycle"
    assert states.equals(StateMachine().run(pinned))


def test_a_lower_energy_ceiling_suppresses_entries():
    """And the machine is otherwise identical -- one field moved, nothing else.

    The same input reaches ``RIDING`` under the inert default and never leaves
    ``COMPRESSION`` once the ceiling drops below the energy it is carrying. A
    gate that changed nothing would be indistinguishable from not having added
    it, which is the failure mode a no-op default makes easy to ship.
    """
    features = quiet(60, direction=[0.8] * 60, strength=[0.95] * 60, energy=[0.90] * 60)

    assert MarketState.RIDING in set(StateMachine(min_dwell=4, cooldown=8).run(features))
    gated = StateMachine(min_dwell=4, cooldown=8, energy_ceiling=0.50).run(features)
    assert set(gated) == {MarketState.COMPRESSION}, (
        f"the energy gate let the machine into {sorted({s.name for s in gated})}"
    )


def test_the_energy_gate_binds_at_its_own_boundary_and_not_beside_it():
    """``<=``, so a reading exactly on the ceiling is admitted.

    Worth pinning because the direction of the inequality is the whole content
    of the parameter: R7 measured high ``energy`` as the chop side, so a ceiling
    excludes the top of the axis. A ``<`` here, or a floor instead of a ceiling,
    would still produce a machine that trades and a surface that ranks.
    """
    def rides(energy: float, ceiling: float) -> bool:
        features = quiet(60, direction=[0.8] * 60, strength=[0.95] * 60, energy=[energy] * 60)
        machine = StateMachine(min_dwell=4, cooldown=8, energy_ceiling=ceiling)
        return MarketState.RIDING in set(machine.run(features))

    assert rides(0.50, 0.50), "a reading exactly on the ceiling was refused"
    assert rides(0.49, 0.50)
    assert not rides(0.51, 0.50)


def test_an_unmeasurable_energy_is_a_failure_like_any_other_input():
    """``energy`` joins ``measurable``, so a NaN there is not "no news" either.

    The mirror of ``test_an_unmeasurable_bar_is_treated_as_a_failure_not_as_no
    _news`` for the new column. Without this the NaN would instead fall through
    ``energy <= ceiling`` as a plain False and merely stop the machine
    *advancing*, which reads as a stall rather than as a blind bar.
    """
    machine = StateMachine(min_dwell=4, cooldown=0)
    energy = [0.10] * 40 + [np.nan] * 10
    states = machine.run(
        quiet(50, direction=[0.8] * 50, strength=[0.9] * 50, energy=energy)
    )
    assert states.iloc[39] is MarketState.RIDING, "setup failed: never reached RIDING"
    assert states.iloc[40] is MarketState.RESET


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_an_energy_ceiling_outside_the_rank_space_is_refused(value):
    """It is a threshold on a percentile, so outside 0..1 it is not a threshold."""
    with pytest.raises(ValueError, match="energy_ceiling"):
        StateMachine(energy_ceiling=value)


# --- R7c: the energy-first lifecycle -----------------------------------------


ENERGY_FIRST = StateMachine(enter_energy=0.50, exit_energy=0.80)


def test_energy_thresholds_that_collapse_the_dead_band_are_rejected():
    """The mirror of ``enter_strength > exit_strength``, and it must be enforced.

    The inequality runs the other way because the axis is inverted -- entry
    wants energy *low*, so the failure threshold sits above the entry one. Equal
    thresholds are the no-dead-band case whichever axis they are on, and a
    machine that accepted them here would toggle state every bar on a feature
    hovering at the constant, exactly as it would on ``strength``.
    """
    with pytest.raises(ValueError, match="exit_energy"):
        StateMachine(enter_energy=0.50, exit_energy=0.50)
    with pytest.raises(ValueError, match="exit_energy"):
        StateMachine(enter_energy=0.80, exit_energy=0.50)


@pytest.mark.parametrize(
    "kwargs",
    [{"enter_energy": 0.50}, {"exit_energy": 0.80}],
    ids=["enter-only", "exit-only"],
)
def test_half_an_energy_lifecycle_is_refused(kwargs):
    """Both sides move together, or the hysteresis spans two features.

    Setting only ``enter_energy`` would advance on energy and fail on strength.
    The dead band would then be a region of a *plane* with no ordering between
    its edges -- there is no inequality to enforce between a threshold on one
    axis and a threshold on another -- so the machine would silently lose the
    property the ``enter_strength > exit_strength`` check exists to guarantee
    while still producing a plausible state for every bar.
    """
    with pytest.raises(ValueError, match="enter_energy and exit_energy"):
        StateMachine(**kwargs)


@pytest.mark.parametrize("value", [-0.01, 1.01])
@pytest.mark.parametrize("field", ["enter_energy", "exit_energy"])
def test_an_energy_lifecycle_threshold_outside_the_rank_space_is_refused(field, value):
    """Same rank space as the ceiling, so the same 0..1 refusal applies."""
    other = {"enter_energy": 0.05, "exit_energy": 0.95}[
        "exit_energy" if field == "enter_energy" else "enter_energy"
    ]
    with pytest.raises(ValueError, match=field):
        StateMachine(**{field: value, ("exit_energy" if field == "enter_energy"
                                       else "enter_energy"): other})


def test_the_energy_first_mode_is_off_by_default_and_provably_inert():
    """R7c's change has to cost the published figures nothing, by construction.

    The same argument ``test_the_default_energy_ceiling_is_provably_inert``
    makes for R7b's field, made once more for R7c's pair and in the stronger
    form the mode needs: with both thresholds ``None`` the machine does not read
    ``energy``'s *values* at all, so driving the whole 0..1 range through the
    column and pinning it to a constant give identical states over input that
    walks the entire lifecycle. That is what makes the default inert rather than
    merely untested, and it is the cheap half of control 2 -- the expensive half
    is ``scripts/r7c/step0_control.py`` reproducing all four published v1/v2
    rows bit-for-bit against ``main``.
    """
    n = 2000
    varying = frame(
        direction=2.0 * wandering(n, seed=41) - 1.0,
        strength=wandering(n, seed=42),
        stability=wandering(n, seed=43),
        crowding=wandering(n, seed=44),
        energy=wandering(n, seed=45),
    )
    assert StateMachine().enter_energy is None
    assert StateMachine().exit_energy is None
    assert not StateMachine().energy_first
    assert ENERGY_FIRST.energy_first, "the probe machine is not in the mode it names"

    states = StateMachine().run(varying)
    assert states.nunique() == len(MarketState), "the probe never walked the lifecycle"
    assert states.equals(StateMachine().run(varying.assign(energy=0.0)))
    assert not states.equals(ENERGY_FIRST.run(varying)), (
        "the energy-first machine agrees with the default on every bar, so the "
        "mode is inert where it is supposed to bite"
    )


def test_the_energy_first_mode_inverts_which_bars_advance():
    """Not a filter bolted onto the old gate -- the other axis, both ways.

    Two frames, each of which one machine rides and the other refuses. Strong
    and violent is what ``strength`` was built to enter on and is precisely what
    R7b measured as the chop side; quiet with a clean lean and no strength is
    what the energy-first machine is for. A mode that merely *narrowed* the old
    gate would ride the first frame under both machines and neither under the
    second, which is the reading a reader has to be able to rule out.
    """
    violent = quiet(60, direction=[0.8] * 60, strength=[0.95] * 60, energy=[0.90] * 60)
    orderly = quiet(60, direction=[0.8] * 60, strength=[0.05] * 60, energy=[0.10] * 60)
    fast = dict(min_dwell=4, cooldown=8)

    assert MarketState.RIDING in set(StateMachine(**fast).run(violent))
    assert MarketState.RIDING not in set(
        StateMachine(enter_energy=0.50, exit_energy=0.80, **fast).run(violent)
    )
    assert MarketState.RIDING not in set(StateMachine(**fast).run(orderly))
    assert MarketState.RIDING in set(
        StateMachine(enter_energy=0.50, exit_energy=0.80, **fast).run(orderly)
    )


def test_the_energy_dead_band_stalls_a_setup_rather_than_failing_it():
    """The hysteresis is on the new axis too, and it is what makes it hysteresis.

    Two bars at 0.65 sit above the 0.50 entry and below the 0.80 failure, so a
    ``BREAKOUT`` merely stops climbing and goes on to ride. With the thresholds
    all but collapsed (0.50 against 0.51) the same two bars are outright
    failures and the machine is knocked to ``RESET``. This is
    ``test_hysteresis_lets_a_setup_survive_a_dip_that_a_single_threshold_would
    _kill`` re-asked on the axis R7c moved the lifecycle onto -- and it is the
    assertion that would fail if ``failing`` had been left on ``strength``,
    since the strength column below never moves.
    """
    spiking = [0.10] * 8 + [0.65] * 2 + [0.10] * 14
    features = quiet(24, direction=[0.8] * 24, strength=[0.5] * 24, energy=spiking)

    with_band = StateMachine(
        enter_energy=0.50, exit_energy=0.80, min_dwell=4, cooldown=8
    ).run(features)
    without_band = StateMachine(
        enter_energy=0.50, exit_energy=0.51, min_dwell=4, cooldown=8
    ).run(features)

    assert MarketState.RESET not in set(with_band), "a dead-band bar knocked the setup back"
    assert with_band.iloc[-1] is MarketState.RIDING
    assert without_band.iloc[10] is MarketState.RESET, (
        "setup failed: the collapsed thresholds were supposed to fail on the spike"
    )
    assert MarketState.RIDING not in set(without_band)


def test_an_unmeasurable_bar_is_a_failure_in_the_energy_first_mode_too():
    """``energy > exit_energy`` is False on a NaN, so ``~measurable`` has to fire.

    The mirror of ``test_an_unmeasurable_bar_is_treated_as_a_failure_not_as_no
    _news`` on the new axis, and the one place the inverted inequality could
    have quietly changed behaviour: on the strength axis the failure predicate
    is ``<`` and on this one it is ``>``, but NaN compares False to both, so
    neither is what catches a blind bar.
    """
    machine = StateMachine(enter_energy=0.50, exit_energy=0.80, min_dwell=4, cooldown=0)
    energy = [0.10] * 40 + [np.nan] * 10
    states = machine.run(quiet(50, direction=[0.8] * 50, energy=energy))
    assert states.iloc[39] is MarketState.RIDING, "setup failed: never reached RIDING"
    assert states.iloc[40] is MarketState.RESET


def test_every_transition_taken_is_legal():
    """The lifecycle is a cycle; the machine must not jump COMPRESSION -> EXHAUSTION."""
    n = 4000
    states = StateMachine().run(
        frame(
            direction=2.0 * wandering(n, seed=5) - 1.0,
            strength=wandering(n, seed=6),
            stability=wandering(n, seed=7),
            crowding=wandering(n, seed=8),
            energy=wandering(n, seed=9),
        )
    )
    assert states.nunique() == len(MarketState), (
        f"only {states.nunique()} of {len(MarketState)} states were ever reached, so "
        "the legality check below never saw most of the lifecycle"
    )
    for previous, current in zip(states.iloc[:-1], states.iloc[1:]):
        assert current in StateMachine.LEGAL_TRANSITIONS[previous], f"{previous} -> {current}"


# Feature readings that between them trip every predicate the machine has:
# NaN (unmeasurable), each strength band, both sides of the direction floor,
# a stability collapse, an extreme carry, and both sides of an energy ceiling.
DIRECTIONS = (-0.9, -0.05, 0.0, 0.05, 0.9, np.nan)
STRENGTHS = (0.05, 0.34, 0.5, 0.95, np.nan)
STABILITIES = (0.02, 0.95)
CROWDINGS = (0.5, 0.99)
# Two values, not three: a NaN energy is unmeasurable exactly as a NaN strength
# is, and ``DIRECTIONS``/``STRENGTHS`` already carry that case into every
# combination below. What NaN would add here is runtime, not coverage.
ENERGIES = (0.10, 0.90)

BASE_TAILS = [
    dict(direction=d, strength=s, stability=st, crowding=c)
    for d in DIRECTIONS
    for s in STRENGTHS
    for st in STABILITIES
    for c in CROWDINGS
]


def constant_tails(machine: StateMachine) -> list[dict]:
    """``BASE_TAILS`` with an energy reading, crossed only where one can bite.

    A ceiling of 1.0 admits every measurable percentile, so crossing
    ``ENERGIES`` against a machine carrying the default doubles the tail set and
    cannot move a single state. Measured rather than argued: crossing all four
    parametrizations below against the full 240 tails passes, in **122 s**
    against this file's **52 s**, with identical verdicts. The saving is the
    duplicate half, not coverage.

    An energy-first machine reads the column on *both* sides of the lifecycle,
    so it always gets the full cross regardless of its ceiling -- the two
    readings straddle its entry threshold and its failure threshold, which is
    the whole of what its walk turns on.
    """
    bites = machine.energy_ceiling < 1.0 or machine.energy_first
    energies = ENERGIES if bites else ENERGIES[:1]
    return [tail | {"energy": energy} for tail in BASE_TAILS for energy in energies]


CONSTANT_TAILS = constant_tails(StateMachine())


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    return frame(**{name: [row[name] for row in rows] for name in REQUIRED_COLUMNS})


# Leads chosen to leave the machine in every one of the six states, at every
# age within each: a strong block long enough to reach RIDING, then a second
# constant block of every length up to the bound. ``test_every_state_is_reached
# _by_some_lead`` is what keeps that claim honest rather than assumed.
STRONG = dict(direction=0.9, strength=0.95, stability=0.95, crowding=0.5, energy=0.10)


def leads_for(machine: StateMachine) -> list[list[dict]]:
    climb = [STRONG] * (3 * machine.min_dwell + 2)
    spans = range(1, machine.convergence_bars + 3)
    tails = constant_tails(machine)
    second_blocks = tails[:: max(1, len(tails) // 8)]
    # Truncations of the climb walk the machine up the lifecycle one bar at a
    # time; a full climb followed by a second constant block of every length
    # reaches the states below RIDING and every age within them. Anything past
    # ``len(climb)`` would repeat the whole climb, so the first list stops there.
    return [climb[:span] for span in range(1, len(climb) + 1)] + [
        climb + [reading] * span for reading in second_blocks for span in spans
    ]


@pytest.mark.parametrize(
    "machine",
    [
        StateMachine(),
        StateMachine(enter_strength=0.80, exit_strength=1.0 / 3.0, min_dwell=2, cooldown=16),
        StateMachine(min_dwell=1, cooldown=0, exhaustion_dwell=1),
        StateMachine(energy_ceiling=0.50),
        StateMachine(enter_energy=0.50, exit_energy=0.80, min_dwell=4, cooldown=4),
    ],
    ids=["default", "trained", "fastest-legal", "energy-gated", "energy-first"],
)
def test_a_constant_tail_converges_from_any_start(machine):
    """**The invariant a live process rests on.** No finite prefix survives.

    A backtest reaches bar *t* carrying years of history; a replay or a live
    process reaches it cold. They can only agree if the state stops depending on
    where the input began, and the machine cannot promise that unless every
    state has a bounded exit. Before that landed, a tail of
    ``direction=0.8, strength=0.5, stability=0.9, crowding=0.5`` left a cold run
    in COMPRESSION and a warm one in EXHAUSTION *permanently* -- target risk 0.0
    against -0.55 -- and raising ``warmup_bars`` could not have fixed it, since
    the disagreement is unbounded.

    Every start is exercised by prefixing, which is the only way in from
    outside; the internal counters are not reachable any other way and a test
    that set them would be testing its own copy of the transition table.

    The ``energy-gated`` machine is here because ``energy_ceiling`` adds a third
    way for ``advancing`` to go false, and a bounded-exit invariant proved only
    where a predicate is inert is not proof about a machine that switches it on.
    The ``energy-first`` one is here because it re-derives *both* predicates
    from a different column, and it is the configuration R7c actually trades:
    ``warmup_bars`` is ``convergence_bars`` times a constant, so if this
    invariant does not hold in that mode then the number R7c's SOL run warms up
    by is not measuring anything. Those two are also the only parametrizations
    whose tail set carries both energy readings -- see ``constant_tails`` for
    why the other three would be paying for a duplicate.
    """
    bound = machine.convergence_bars
    tail_bars = bound + 20
    leads = leads_for(machine)

    worst = 0
    for reading in constant_tails(machine):
        cold = machine.run(rows_to_frame([reading] * tail_bars)).to_numpy()
        for lead in leads:
            warm = machine.run(rows_to_frame(lead + [reading] * tail_bars)).to_numpy()[
                len(lead) :
            ]
            disagreements = np.flatnonzero(warm != cold)
            lag = int(disagreements[-1]) + 1 if len(disagreements) else 0
            worst = max(worst, lag)
            assert lag <= bound, (
                f"a run led by {len(lead)} bars still disagreed with a cold start "
                f"{lag} bars into a constant tail {reading}, past the declared "
                f"bound of {bound}"
            )
    assert worst > bound // 2, (
        f"the worst observed lag was {worst} against a bound of {bound}; a bound "
        "that loose is not measuring the machine any more"
    )


def test_every_state_is_reached_by_some_lead():
    """Otherwise the convergence check above starts from three states, not six."""
    machine = StateMachine()
    reached = {machine.run(rows_to_frame(lead)).iloc[-1] for lead in leads_for(machine)}
    assert reached == set(MarketState), f"never started from {set(MarketState) - reached}"


def test_the_mid_band_tail_does_not_split_a_cold_run_from_a_warm_one():
    """The exact reading that was reported, kept as its own regression.

    It is the ordinary case rather than a corner: strength parked in the middle
    tercile with a clean lean and no carry pressure is what the policy calls the
    fade band, so the two runs disagreed about a *live short* rather than about
    a bookkeeping detail.
    """
    reading = dict(direction=0.8, strength=0.5, stability=0.9, crowding=0.5, energy=0.10)
    machine = StateMachine()
    tail_bars = machine.convergence_bars + 20

    cold = machine.run(rows_to_frame([reading] * tail_bars)).to_numpy()
    warm = machine.run(rows_to_frame([STRONG] * 200 + [reading] * tail_bars)).to_numpy()[200:]

    assert warm[-1] is cold[-1] is MarketState.COMPRESSION
    assert (warm[machine.convergence_bars :] == cold[machine.convergence_bars :]).all()


# Bars a cold start may take to agree with a run that has seen the whole history
# on *realistic* input. The bound proved above covers constant tails only, and
# real features drift rather than holding still, so this is the independent
# measurement rather than a restatement: over the 76 frames below the worst lag
# is 74 and 58 of them agree immediately, against a proven constant-tail bound
# of 34. The limit is set at roughly twice the worst rather than at it.
#
# R7b re-measured it after adding the ``energy`` column and a second, gated
# machine: 38 frames became 76 and the worst lag did not move off 74. A ceiling
# that bites gives the machine a further reason to leave a state, so it was
# never likely to *lengthen* the memory -- but that is an argument, and this
# number is the measurement.
MAX_CONVERGENCE_LAG = 150


@pytest.mark.parametrize(
    "machine",
    [
        StateMachine(),
        StateMachine(energy_ceiling=0.50),
        StateMachine(enter_energy=0.50, exit_energy=0.80, min_dwell=4, cooldown=4),
    ],
    ids=["default", "energy-gated", "energy-first"],
)
@pytest.mark.parametrize("seed", range(10, 200, 5))
def test_the_machine_forgets_where_it_started(seed, machine):
    """Saturating counters make this a finite automaton, not an endless recursion.

    A live process starts cold in COMPRESSION while a backtest reaches the same
    bar carrying years of history. The two agree only if the state stops
    depending on where the input began, and this measures how many bars that
    costs: cutting the frame in half and re-running the tail from scratch must
    converge on the whole-history answer, quickly and then permanently.
    """
    n = 2000
    features = frame(
        direction=2.0 * wandering(n, seed=seed) - 1.0,
        strength=wandering(n, seed=seed + 1),
        stability=wandering(n, seed=seed + 2),
        crowding=wandering(n, seed=seed + 3),
        energy=wandering(n, seed=seed + 4),
    )
    split = n // 2
    whole = machine.run(features).to_numpy()[split:]
    late = machine.run(features.iloc[split:]).to_numpy()

    disagreements = np.flatnonzero(whole != late)
    lag = int(disagreements[-1]) + 1 if len(disagreements) else 0
    assert lag <= MAX_CONVERGENCE_LAG, (
        f"a cold start still disagreed {lag} bars after the split; the machine is "
        "carrying memory of where its input began"
    )
