from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_lab.features.base import rolling_percentile
from strategy_lab.state.machine import MarketState, StateMachine


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
    """A complete frame of unremarkable readings, for tests that vary one column."""
    columns = {
        "direction": [0.0] * n,
        "strength": [0.1] * n,
        "stability": [0.9] * n,
        "crowding": [0.5] * n,
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


def changes(states: pd.Series) -> int:
    return int((states.to_numpy()[1:] != states.to_numpy()[:-1]).sum())


def test_the_machine_starts_flat_rather_than_guessing():
    """Bar 0 has no history behind it, so the machine may not claim a trend."""
    assert StateMachine.INITIAL_STATE is MarketState.COMPRESSION
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
            frame(direction=[0.0] * 20, stability=[0.9] * 20, crowding=[0.5] * 20)
        )


def test_thresholds_that_collapse_the_dead_band_are_rejected():
    with pytest.raises(ValueError, match="enter_strength"):
        StateMachine(enter_strength=0.30, exit_strength=0.30)


def test_hysteresis_stops_a_hovering_feature_from_flipping_every_bar():
    """A feature oscillating around one threshold must not toggle the state.

    0.29/0.31 straddles ``enter_strength`` and sits entirely inside the dead
    band above ``exit_strength``, so the machine may climb the lifecycle but
    must never be knocked back down it. Collapsing the two thresholds onto one
    turns every 0.29 bar into a failure and produces roughly 30 changes here
    instead of 4.
    """
    machine = StateMachine(enter_strength=0.30, exit_strength=0.20, min_dwell=1, cooldown=0)
    hovering = [0.29, 0.31] * 20
    states = machine.run(
        quiet(40, direction=[0.8] * 40, strength=hovering)
    )
    assert MarketState.RESET not in set(states), "a bar inside the dead band knocked it back"
    assert changes(states) <= 5, f"state churned {changes(states)} times on a hovering feature"


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
    crowding = [0.5] * 40 + [0.97] * 10
    states = machine.run(
        quiet(50, direction=[0.8] * 50, strength=[0.9] * 50, crowding=crowding)
    )
    assert states.iloc[39] is MarketState.RIDING, "setup failed: never reached RIDING"
    assert states.iloc[40] is MarketState.EXHAUSTION


def test_an_unmeasurable_bar_is_treated_as_a_failure_not_as_no_news():
    """Warmup rows are NaN, and a machine that coasts through them holds a trend
    it can no longer see."""
    machine = StateMachine(min_dwell=4, cooldown=0)
    strength = [0.9] * 40 + [np.nan] * 10
    states = machine.run(quiet(50, direction=[0.8] * 50, strength=strength))
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
        )
    )
    assert states.nunique() == len(MarketState), (
        f"only {states.nunique()} of {len(MarketState)} states were ever reached, so "
        "the legality check below never saw most of the lifecycle"
    )
    for previous, current in zip(states.iloc[:-1], states.iloc[1:]):
        assert current in StateMachine.LEGAL_TRANSITIONS[previous], f"{previous} -> {current}"


# Bars a cold start may take to agree with a run that has seen the whole
# history. Measured over the 38 frames below: 61 at worst, 0 on 24 of them, and
# the bound is set at roughly twice the worst rather than at it. This is the
# number the strategy adapter's warmup margin is built on, so it is a bound
# rather than a description -- transitions that push convergence past it make
# the adapter's warmup_bars wrong too.
MAX_CONVERGENCE_LAG = 120


@pytest.mark.parametrize("seed", range(10, 200, 5))
def test_the_machine_forgets_where_it_started(seed):
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
    )
    split = n // 2
    whole = StateMachine().run(features).to_numpy()[split:]
    late = StateMachine().run(features.iloc[split:]).to_numpy()

    disagreements = np.flatnonzero(whole != late)
    lag = int(disagreements[-1]) + 1 if len(disagreements) else 0
    assert lag <= MAX_CONVERGENCE_LAG, (
        f"a cold start still disagreed {lag} bars after the split; the machine is "
        "carrying memory of where its input began"
    )
