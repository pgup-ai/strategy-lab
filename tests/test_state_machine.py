from __future__ import annotations

import pandas as pd
import pytest

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
    columns = {
        "direction": [0.0] * n,
        "strength": [0.1] * n,
        "stability": [0.9] * n,
        "crowding": [0.5] * n,
    }
    columns.update(overrides)
    return frame(**columns)


def test_the_machine_starts_flat_rather_than_guessing():
    """Bar 0 has no history behind it, so the machine may not claim a trend."""
    assert StateMachine.INITIAL_STATE is MarketState.COMPRESSION
    states = StateMachine().run(
        quiet(20, direction=[0.9] * 20, strength=[0.95] * 20)
    )
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
