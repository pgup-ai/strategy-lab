from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_lab.features.base import rolling_percentile, rolling_zscore
from strategy_lab.features.trend import Direction, Persistence, Stability, Strength


def series(values) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="4h", tz="UTC", name="timestamp")
    return pd.Series(values, index=index, dtype="float64")


def frame(close: np.ndarray, *, half_range: float, volume: float = 500.0) -> pd.DataFrame:
    index = pd.date_range(
        "2024-01-01", periods=len(close), freq="4h", tz="UTC", name="timestamp"
    )
    return pd.DataFrame(
        {
            "open": close * (1 - half_range / 2),
            "high": close * (1 + half_range),
            "low": close * (1 - half_range),
            "close": close,
            "volume": np.full(len(close), volume),
        },
        index=index,
    )


def trending(n: int, slope: float = 0.002, noise: float = 0.0003) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    close = 100 * np.exp(np.cumsum(np.full(n, slope) + rng.normal(0, noise, n)))
    return frame(close, half_range=0.002)


def choppy(n: int, scale: float = 0.004) -> pd.DataFrame:
    rng = np.random.default_rng(12)
    close = 100 * np.exp(np.cumsum(rng.normal(0, scale, n)))
    return frame(close, half_range=0.004)


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


def test_direction_is_positive_in_an_uptrend_and_negative_in_a_downtrend():
    feature = Direction()
    up = feature.compute(trending(feature.warmup_bars + 200)).iloc[-1]
    down = feature.compute(trending(feature.warmup_bars + 200, slope=-0.002)).iloc[-1]
    assert up > 0.3 and down < -0.3


def test_direction_stays_inside_minus_one_to_one():
    feature = Direction()
    values = feature.compute(trending(feature.warmup_bars + 500, slope=0.05)).dropna()
    assert values.min() >= -1.0 and values.max() <= 1.0


def test_direction_is_not_pinned_to_its_rails_by_an_ordinary_trend():
    """A spread of a few ATRs is common, so an unscaled tanh saturates on it.

    Measured on the stored BTC/USDT perp 4h history, feeding the raw ATR ratio
    to tanh puts 51.6% of bars past |0.9|; here the same unscaled form reads
    1.0000 against 0.9884 scaled.
    """
    feature = Direction()
    values = feature.compute(trending(feature.warmup_bars + 500, slope=0.0005)).dropna()
    assert values.abs().max() < 0.99


def test_strength_separates_a_clean_trend_from_chop():
    """The whole point of the dimension: same net motion, different path."""
    feature = Strength()
    n = feature.warmup_bars + 300
    assert feature.compute(trending(n)).iloc[-1] > 0.6
    assert feature.compute(choppy(n)).iloc[-1] < 0.4


def test_persistence_is_high_on_a_straight_line_and_low_on_noise():
    """The chop side is a median, not the last bar.

    R-squared at any single bar of a random walk is a lottery -- measured across
    windows 48..200 on this seed the final bar ranges 0.24 to 0.71, straddling
    any threshold worth asserting. The median over the whole series does not.
    """
    feature = Persistence()
    n = feature.warmup_bars + 300
    assert feature.compute(trending(n, noise=0.0)).iloc[-1] > 0.95
    assert feature.compute(choppy(n)).dropna().median() < 0.7


def test_stability_falls_when_the_path_gets_ragged():
    feature = Stability()
    n = feature.warmup_bars + 300
    assert feature.compute(trending(n, noise=0.0001)).iloc[-1] > feature.compute(choppy(n)).iloc[-1]


def test_stability_is_not_persistence_wearing_another_name():
    """Normalizing residual scatter by the window's own price variance would make
    this ``1 - sqrt(1 - R^2)``, which is Persistence monotonically transformed."""
    df = choppy(Stability().warmup_bars + 800)
    pair = pd.DataFrame(
        {"stability": Stability().compute(df), "persistence": Persistence().compute(df)}
    ).dropna()
    assert abs(pair["stability"].corr(pair["persistence"])) < 0.6


@pytest.mark.parametrize("feature", [Direction(), Strength(), Persistence(), Stability()])
def test_every_trend_feature_leaves_warmup_as_nan(feature):
    values = feature.compute(trending(feature.warmup_bars + 50))
    assert values.iloc[: feature.warmup_bars].isna().all()
    assert values.iloc[feature.warmup_bars :].notna().all()


@pytest.mark.parametrize("feature", [Strength(), Persistence(), Stability()])
def test_unsigned_trend_features_stay_inside_zero_to_one(feature):
    for df in (trending(feature.warmup_bars + 400), choppy(feature.warmup_bars + 400)):
        values = feature.compute(df).dropna()
        assert values.min() >= 0.0 and values.max() <= 1.0
