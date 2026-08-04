from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_lab.features.base import rolling_percentile, rolling_zscore


def series(values) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="4h", tz="UTC", name="timestamp")
    return pd.Series(values, index=index, dtype="float64")


def test_rolling_percentile_is_causal_where_a_full_sample_rank_is_not():
    """A full-sample rank at bar t moves when bars after t change. Measured:
    row 120 goes 0.605 -> 1.000 under a downward poison. The rolling form does not."""
    clean = series(np.arange(200.0))
    poisoned = clean.copy()
    poisoned.iloc[121:] = -1e6

    assert clean.rank(pct=True).iloc[120] != poisoned.rank(pct=True).iloc[120]
    assert rolling_percentile(clean, window=50).iloc[120] == pytest.approx(
        rolling_percentile(poisoned, window=50).iloc[120]
    )


def test_rolling_percentile_spans_zero_to_one():
    values = rolling_percentile(series(np.random.default_rng(3).normal(size=400)), window=100)
    tail = values.iloc[100:]
    assert tail.min() >= 0.0 and tail.max() <= 1.0


def test_rolling_percentile_leaves_warmup_as_nan_not_zero():
    """NaN says 'not yet measurable'; 0.0 would say 'measured, and it is the minimum'."""
    values = rolling_percentile(series(np.arange(100.0)), window=50)
    assert values.iloc[:49].isna().all()
    assert values.iloc[49:].notna().all()


def test_rolling_percentile_puts_a_tied_window_mid_range_not_at_the_top():
    """A window of identical values has no range to sit at the top of."""
    values = rolling_percentile(series(np.full(200, 5.0)), window=50)
    assert values.iloc[50:].max() < 0.55


def test_rolling_zscore_is_causal():
    clean = series(np.arange(300.0))
    poisoned = clean.copy()
    poisoned.iloc[201:] = -1e6
    assert rolling_zscore(clean, window=100).iloc[200] == pytest.approx(
        rolling_zscore(poisoned, window=100).iloc[200]
    )


def test_rolling_zscore_of_a_flat_series_is_zero_not_infinite():
    """Zero variance would divide by zero; the guard must not produce inf."""
    values = rolling_zscore(series(np.full(200, 5.0)), window=50)
    assert np.isfinite(values.iloc[50:]).all()
