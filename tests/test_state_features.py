from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_lab.features.base import rolling_percentile, rolling_zscore
from strategy_lab.features.trend import Direction, Persistence, Stability, Strength
from strategy_lab.features.volatility import Compression, CompressionRelease, Energy


def series(values) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="4h", tz="UTC", name="timestamp")
    return pd.Series(values, index=index, dtype="float64")


def frame(
    close: np.ndarray, *, half_range: float | np.ndarray, volume: float = 500.0
) -> pd.DataFrame:
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


def regime_switch(quiet: int, violent: int = 200, calm: int = 200) -> pd.DataFrame:
    """Quiet chop, then violent chop, then a calm one-way trend, in one series.

    One frame rather than three, because Energy is a percentile against the
    instrument's own recent history: two separately generated series each read
    mid-range against themselves and the contrast the charter cares about --
    same dimension, opposite regimes -- never appears.
    """
    blocks, half_ranges, level = [], [], 0.0
    for length, scale, slope, half_range, seed in (
        (quiet, 0.004, 0.0, 0.004, 1),
        (violent, 0.020, 0.0, 0.020, 2),
        (calm, 0.0005, 0.002, 0.001, 3),
    ):
        rng = np.random.default_rng(seed)
        walk = level + np.cumsum(np.full(length, slope) + rng.normal(0, scale, length))
        blocks.append(walk)
        half_ranges.append(np.full(length, half_range))
        level = walk[-1]
    return frame(
        100 * np.exp(np.concatenate(blocks)), half_range=np.concatenate(half_ranges)
    )


def test_energy_and_strength_tell_a_steady_trend_from_violent_chop():
    """The charter's own worked example, and the reason they are two dimensions.

    Strength 0.8 / Energy 0.3 is a slow steady trend; Strength 0.2 / Energy 0.95
    is violent two-way chop. Collapsed into one number both read 'active'.
    """
    energy, strength = Energy(), Strength()
    quiet = energy.warmup_bars
    df = regime_switch(quiet)
    violent_end, calm_end = quiet + 199, len(df) - 1

    assert energy.compute(df).iloc[violent_end] > 0.8
    assert strength.compute(df).iloc[violent_end] < 0.2
    assert energy.compute(df).iloc[calm_end] < 0.2
    assert strength.compute(df).iloc[calm_end] > 0.8


def test_compression_is_energy_read_from_the_other_end():
    df = regime_switch(Energy().warmup_bars)
    total = (Compression().compute(df) + Energy().compute(df)).dropna()
    assert np.allclose(total, 1.0)


def test_compression_release_is_positive_exactly_where_compression_falls():
    """The sign is the whole reason this is a named feature: differencing
    Compression by hand reports a release as a negative number."""
    df = regime_switch(Energy().warmup_bars)
    compression = Compression().compute(df)
    release = CompressionRelease().compute(df)

    falling = (compression.diff() < 0) & release.notna()
    rising = (compression.diff() > 0) & release.notna()
    assert falling.sum() > 100 and rising.sum() > 100
    assert (release[falling] > 0).all()
    assert (release[rising] < 0).all()


@pytest.mark.parametrize("feature", [Energy(), Compression(), CompressionRelease()])
def test_every_volatility_feature_leaves_warmup_as_nan(feature):
    values = feature.compute(regime_switch(feature.warmup_bars))
    assert values.iloc[: feature.warmup_bars].isna().all()
    assert values.iloc[feature.warmup_bars :].notna().all()


@pytest.mark.parametrize(
    ("feature", "floor"), [(Energy(), 0.0), (Compression(), 0.0), (CompressionRelease(), -1.0)]
)
def test_volatility_features_stay_inside_their_declared_range(feature, floor):
    values = feature.compute(regime_switch(feature.warmup_bars)).dropna()
    assert values.min() >= floor and values.max() <= 1.0
