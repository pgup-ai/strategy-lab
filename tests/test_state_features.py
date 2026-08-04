from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_lab.features.base import rolling_percentile, rolling_zscore
from strategy_lab.features.flow import Crowding, Participation, align_funding_to_bars
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


def with_funding(
    df: pd.DataFrame, *, base: float = 0.0001, recent: float | None = None, tail: int = 40
) -> pd.DataFrame:
    """Settle funding on every second bar, as an 8h venue does against 4h bars.

    ``base`` is jittered rather than flat, because a constant carry has no spread
    for a z-score to measure and every bar comes out exactly neutral -- true, but
    useless for asking whether the feature responds to anything. ``recent``
    shifts the last ``tail`` settlements to a different level.
    """
    settlements = df.index[::2]
    rates = np.random.default_rng(5).normal(base, abs(base) * 0.2, len(settlements))
    if recent is not None:
        rates[-tail:] = recent
    return df.assign(
        funding_rate=align_funding_to_bars(df.index, pd.Series(rates, index=settlements))
    )


def test_participation_ranks_a_volume_spike_above_the_quiet_bars_around_it():
    feature = Participation()
    df = choppy(feature.warmup_bars + 200)
    df.loc[df.index[-1], "volume"] = df["volume"].max() * 10
    values = feature.compute(df)
    assert values.iloc[-1] == pytest.approx(1.0)
    assert values.iloc[-2] < 0.9


def test_crowding_refuses_a_frame_with_no_funding_rather_than_calling_it_neutral():
    """Equity frames have no funding. A silent 0.5 would claim nobody is crowded."""
    feature = Crowding()
    with pytest.raises(ValueError, match="funding_rate"):
        feature.compute(choppy(feature.warmup_bars + 50))


def test_crowding_reads_above_neutral_when_longs_pay_and_below_when_shorts_do():
    feature = Crowding()
    df = choppy(feature.warmup_bars + 400)
    longs_pay = feature.compute(with_funding(df, recent=0.003)).iloc[-1]
    shorts_pay = feature.compute(with_funding(df, recent=-0.003)).iloc[-1]
    assert longs_pay > 0.9 and shorts_pay < 0.1


def test_crowding_of_an_unchanging_carry_is_neutral_not_undefined():
    """Flat funding has no spread to measure against, which is 0.5 -- not the
    NaN or the inf an unguarded z-score produces from that 0/0."""
    feature = Crowding()
    df = choppy(feature.warmup_bars + 200)
    settlements = df.index[::2]
    flat = df.assign(
        funding_rate=align_funding_to_bars(
            df.index, pd.Series(0.0001, index=settlements, dtype="float64")
        )
    )
    values = feature.compute(flat).dropna()
    assert np.isfinite(values).all()
    assert values.eq(0.5).all()


def test_funding_lands_on_the_bar_containing_it_not_the_one_it_equals():
    """Binance stamps settlements past the boundary -- 3,260 of BTC's 7,559 stored
    ones are off-grid -- so an equality join against an 8h range drops 43% of them."""
    index = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC", name="timestamp")
    late = pd.DatetimeIndex([index[2] + pd.Timedelta(milliseconds=47)])
    aligned = align_funding_to_bars(index, pd.Series([0.0001], index=late))

    assert aligned.iloc[2] == pytest.approx(0.0001)
    assert aligned.drop(index[2]).eq(0.0).all()


@pytest.mark.parametrize("crypto_only", [False, True])
def test_every_flow_feature_leaves_warmup_as_nan_and_stays_inside_zero_to_one(crypto_only):
    feature = Crowding() if crypto_only else Participation()
    df = choppy(feature.warmup_bars + 300)
    values = feature.compute(with_funding(df) if crypto_only else df)

    assert values.iloc[: feature.warmup_bars].isna().all()
    assert values.iloc[feature.warmup_bars :].notna().all()
    assert values.dropna().min() >= 0.0 and values.dropna().max() <= 1.0
