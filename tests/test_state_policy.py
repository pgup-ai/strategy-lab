from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_lab.state.machine import MarketState
from strategy_lab.state.policy import STATE_TARGET_RISK, target_risk, target_risk_series


def test_high_strength_follows_direction():
    assert target_risk(state=MarketState.RIDING, direction=+0.8, strength=0.9, crowding=0.5) > 0
    assert target_risk(state=MarketState.RIDING, direction=-0.8, strength=0.9, crowding=0.5) < 0


def test_mid_strength_fades_direction():
    """R4 measured IC -0.113 here, both halves agreeing. A monotone rule throws this away."""
    assert target_risk(state=MarketState.RIDING, direction=+0.8, strength=0.5, crowding=0.5) < 0
    assert target_risk(state=MarketState.RIDING, direction=-0.8, strength=0.5, crowding=0.5) > 0


def test_low_strength_is_flat_because_its_halves_disagreed():
    assert target_risk(state=MarketState.RIDING, direction=+0.8, strength=0.05, crowding=0.5) == 0.0


def test_extreme_crowding_shrinks_the_target_without_flipping_it():
    calm = target_risk(state=MarketState.RIDING, direction=+0.8, strength=0.9, crowding=0.5)
    crowded = target_risk(state=MarketState.RIDING, direction=+0.8, strength=0.9, crowding=0.98)
    assert 0 < crowded < calm


def test_crowding_presses_on_the_paying_side_only():
    """Crowding's measured IC is signed, so damping both sides would discard it.

    At crowding 0.98 the longs are the ones paying; a short is standing on the
    side collecting, and shrinking it too would be a symmetric reading of an
    asymmetric measurement.
    """
    short = target_risk(state=MarketState.RIDING, direction=-0.8, strength=0.9, crowding=0.98)
    undamped = target_risk(state=MarketState.RIDING, direction=-0.8, strength=0.9, crowding=0.5)
    assert short == pytest.approx(undamped)


def test_compression_and_reset_are_flat():
    for state in (MarketState.COMPRESSION, MarketState.RESET):
        assert target_risk(state=state, direction=+0.9, strength=0.9, crowding=0.5) == 0.0


def test_a_live_state_risks_more_the_further_up_the_lifecycle_it_is():
    """The size of a position is the state's whole remaining job, given that the
    engine will not resize it after entry."""
    lifecycle = [
        MarketState.BREAKOUT,
        MarketState.CONFIRMED,
        MarketState.RIDING,
    ]
    targets = [
        target_risk(state=state, direction=+0.8, strength=0.9, crowding=0.5)
        for state in lifecycle
    ]
    assert targets == sorted(targets)
    assert targets[0] > 0
    exhaustion = target_risk(
        state=MarketState.EXHAUSTION, direction=+0.8, strength=0.9, crowding=0.5
    )
    assert targets[0] < exhaustion < targets[-1]


def test_an_unmeasurable_bar_is_flat_rather_than_nan():
    """A NaN target would propagate into position_size and size an entry as NaN."""
    for missing in ("direction", "strength", "crowding"):
        inputs = {"direction": 0.8, "strength": 0.9, "crowding": 0.5, missing: np.nan}
        assert target_risk(state=MarketState.RIDING, **inputs) == 0.0


def test_the_target_never_exceeds_full_risk():
    rng = np.random.default_rng(3)
    n = 2000
    states = pd.Series(rng.choice(list(MarketState), n))
    targets = target_risk_series(
        states=states,
        direction=pd.Series(rng.uniform(-1.5, 1.5, n)),
        strength=pd.Series(rng.uniform(0.0, 1.0, n)),
        crowding=pd.Series(rng.uniform(0.0, 1.0, n)),
    )
    assert targets.abs().max() <= 1.0
    assert targets.abs().max() == pytest.approx(max(STATE_TARGET_RISK.values()), abs=0.05)


def test_the_scalar_and_vector_forms_agree():
    """They are one implementation, and this is what keeps them one."""
    rng = np.random.default_rng(4)
    n = 300
    states = pd.Series(rng.choice(list(MarketState), n))
    direction = pd.Series(rng.uniform(-1.0, 1.0, n))
    strength = pd.Series(rng.uniform(0.0, 1.0, n))
    crowding = pd.Series(rng.uniform(0.0, 1.0, n))

    vector = target_risk_series(
        states=states, direction=direction, strength=strength, crowding=crowding
    )
    scalar = [
        target_risk(
            state=states.iloc[i],
            direction=direction.iloc[i],
            strength=strength.iloc[i],
            crowding=crowding.iloc[i],
        )
        for i in range(n)
    ]
    assert vector.to_list() == pytest.approx(scalar)
