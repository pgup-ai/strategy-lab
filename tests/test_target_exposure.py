"""The continuous-exposure contract, and the two things it refuses.

Both refusals exist because the failure they prevent is silent. An out-of-range
target is filled at whatever cash covers instead of being rejected, and a NaN
target reaches ``from_orders`` as "no order" -- which is not "hold nothing", it
is "hold whatever you were holding". Neither shows up in the artifacts as an
error; both show up as a run that did something other than what it reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from strategy_lab.strategies.exposure import ExposureStrategy, TargetExposure
from strategy_lab.strategies.registry import get_strategy


def series(values: list[float]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="4h", tz="UTC")
    return pd.Series(values, index=index, dtype="float64")


@dataclass(frozen=True)
class _ConstantExposure:
    """The smallest object that satisfies the protocol: half the budget, always."""

    name: str = "constant_exposure"
    version: str = "1.0.0"
    warmup_bars: int = 0

    def compute_target(self, df: pd.DataFrame) -> TargetExposure:
        return TargetExposure(target=pd.Series(0.5, index=df.index, dtype="float64"))


def test_target_outside_minus_one_to_one_is_refused():
    """A target above 1 asks for leverage the book does not have; below -1, likewise."""
    with pytest.raises(ValueError, match="target"):
        TargetExposure(target=series([0.0, 1.5]))
    with pytest.raises(ValueError, match="target"):
        TargetExposure(target=series([0.0, -1.5]))


def test_full_deployment_in_either_direction_is_allowed():
    """The bound is inclusive: +/-1 is the whole risk budget, not one tick past it."""
    assert TargetExposure(target=series([1.0, -1.0])).target.tolist() == [1.0, -1.0]


def test_a_nan_target_is_refused_rather_than_read_as_flat():
    """NaN means 'not yet measurable'; 0.0 means 'measured, and hold nothing'."""
    with pytest.raises(ValueError, match="NaN"):
        TargetExposure(target=series([0.0, float("nan")]))


def test_the_refusals_name_the_row_that_caused_them():
    """A contract violation in a 15,000-bar frame is unfixable without a position."""
    with pytest.raises(ValueError, match="position 2"):
        TargetExposure(target=series([0.0, 0.0, float("nan")]))
    with pytest.raises(ValueError, match="position 1"):
        TargetExposure(target=series([0.0, 4.0, 0.0]))


def test_warmup_rows_are_expressed_as_a_leading_flat_run_not_NaN():
    exposure = TargetExposure(target=series([0.0, 0.0, 0.5]))
    assert exposure.target.iloc[0] == 0.0


def test_metadata_travels_with_the_target():
    """The engine records it in config.json exactly as it records SignalSet.metadata."""
    exposure = TargetExposure(target=series([0.0, 0.5]), metadata={"states": "riding"})
    assert exposure.metadata == {"states": "riding"}
    assert TargetExposure(target=series([0.0])).metadata == {}


def test_the_protocol_is_satisfied_by_a_minimal_implementation():
    assert isinstance(_ConstantExposure(), ExposureStrategy)


def test_a_boolean_strategy_does_not_satisfy_the_exposure_protocol():
    """Otherwise the check above passes on anything and dispatch on it is unsound.

    A ``SignalSet`` strategy carries the same name/version/warmup_bars triple, so
    only the compute method tells the two contracts apart. Task 4 registers
    exposure strategies in a registry of their own precisely because the boolean
    suites call ``generate_signals`` on everything they iterate.
    """
    assert not isinstance(get_strategy("donchian"), ExposureStrategy)
