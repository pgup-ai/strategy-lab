from __future__ import annotations

import re
from dataclasses import is_dataclass

import numpy as np
import pandas as pd
import pytest

from strategy_lab.features import flow, trend, volatility
from strategy_lab.features.base import rolling_percentile, rolling_zscore
from strategy_lab.features.registry import get_feature, list_features
from strategy_lab.features.flow import Crowding, Participation, align_funding_to_bars
from strategy_lab.features.trend import Direction, Persistence, Stability, Strength
from strategy_lab.features.volatility import Compression, CompressionRelease, Energy
from tests.conftest import synthetic_ohlcv_with_funding


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


def test_direction_is_positive_in_an_uptrend_and_negative_in_a_downtrend():
    feature = Direction()
    up = feature.compute(trending(feature.warmup_bars + 200)).iloc[-1]
    down = feature.compute(trending(feature.warmup_bars + 200, slope=-0.002)).iloc[-1]
    assert up > 0.3 and down < -0.3


def test_direction_is_not_pinned_to_its_rails_by_an_ordinary_trend():
    """A spread of a few ATRs is common, so an unscaled tanh saturates on it.

    On this fixture the unscaled form reads 1.0000 against 0.9884 scaled.
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


@pytest.mark.parametrize("name", list_features())
def test_every_feature_leaves_its_warmup_as_nan(name):
    """``mask_warmup``, stated as a test.

    ``ewm(adjust=False)`` returns a number from bar zero, so without the mask
    Direction emits values from bar 23 against a declared warmup of 1,920 -- the
    convention's exact inverse. That those rows are *filled* past warmup is
    asserted in ``tests/test_feature_lookahead.py``, which needs it to be true
    before its own probe means anything.
    """
    feature = get_feature(name)
    values = feature.compute(synthetic_ohlcv_with_funding(n=feature.warmup_bars + 50, freq="4h"))
    assert values.iloc[: feature.warmup_bars].isna().all()


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
    NaN or the inf an unguarded z-score produces from that 0/0.

    Asserted over the post-warmup slice rather than ``dropna()``: dropping the
    NaNs an unguarded z-score would produce leaves an empty series, and every
    assertion below passes vacuously on one.
    """
    feature = Crowding()
    df = choppy(feature.warmup_bars + 200)
    settlements = df.index[::2]
    flat = df.assign(
        funding_rate=align_funding_to_bars(
            df.index, pd.Series(0.0001, index=settlements, dtype="float64")
        )
    )
    values = feature.compute(flat).iloc[feature.warmup_bars :]
    assert not values.empty
    assert values.eq(0.5).all()


def test_funding_lands_on_the_bar_containing_it_not_the_one_it_equals():
    """Binance stamps settlements past the boundary -- 3,260 of BTC's 7,559 stored
    ones are off-grid -- so an equality join against an 8h range drops 43% of them."""
    index = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC", name="timestamp")
    late = pd.DatetimeIndex([index[2] + pd.Timedelta(milliseconds=47)])
    aligned = align_funding_to_bars(index, pd.Series([0.0001], index=late))

    assert aligned.iloc[2] == pytest.approx(0.0001)
    assert aligned.drop(index[2]).eq(0.0).all()


# Bars probed past warmup on a cold start. Each costs one ``compute`` over a
# warmup-sized window, so this trades runtime against how many chances the check
# gets. At 120 an under-declared warmup is caught on 120/120 bars for every
# feature here, because a rolling window that is one bar short returns NaN rather
# than a slightly wrong number.
COLD_START_PROBES = 120

# Two computations of the same bar from different starting points are not
# bit-identical, and cannot be made so: pandas' rolling variance adds and removes
# observations from a running accumulator, so its rounding depends on where the
# series began. Measured over these features the disagreement tops out at 1.5e-11
# relative -- against a 0..1 feature nothing downstream resolves. It is a real
# floor, not a slack allowance: Direction at 10x its span misses by 1.4e-7 and is
# still caught, and the bit-exactness that pins its 20x is asserted separately
# below, on the EMAs themselves.
COLD_START_TOLERANCE = 1e-9


@pytest.mark.parametrize("name", list_features())
def test_declared_warmup_reproduces_whole_history_values(name):
    """What ``warmup_bars`` promises, stated as a test.

    A live process starts cold: it fetches ``warmup_bars`` bars, then computes on
    each new one. A backtest sees everything. Those two agree only if
    ``warmup_bars`` is genuinely enough history, so this replays the cold start at
    every probe point and demands the current bar come out the same.
    """
    feature = get_feature(name)
    warm = feature.warmup_bars
    df = synthetic_ohlcv_with_funding(n=warm + COLD_START_PROBES)
    whole_history = feature.compute(df)

    divergences = []
    for position in range(warm, len(df)):
        cold = feature.compute(df.iloc[position - warm : position + 1]).iloc[-1]
        expected = whole_history.iloc[position]
        if pd.isna(cold) or abs(cold - expected) > COLD_START_TOLERANCE:
            divergences.append((position, cold, expected))

    assert divergences == [], (
        f"{name} declares warmup_bars={warm}, but a cold start from exactly that "
        f"many bars disagrees with the whole-history run at {divergences[:3]} "
        f"({len(divergences)}/{COLD_START_PROBES} probed bars). Raise warmup_bars."
    )


def test_directions_ema_is_bit_exact_after_its_declared_warmup():
    """For the one recursive feature, agreeing to a tolerance is not the bar.

    ``ewm(adjust=False)`` never forgets its seed; it decays it by
    ``(1 - 2/(span+1))`` per bar, so "the values agreed" only means no probed bar
    landed outside the residual band. Measured here at span 96: the cold start is
    wrong by 3.2e-3 relative at 5x the span, 1.4e-7 at 10x and 2.1e-11 at 15x --
    all of which the tolerance above would wave through at 15x. Bit-exactness
    arrives at 18x and is where ``_EWM_WARMUP_MULTIPLE = 20`` comes from.

    Only the larger span is probed: warmup is 20x *it*, so the other EMA clears
    the same bar with room to spare and could not fail on its own.
    """
    feature = Direction()
    warm = feature.warmup_bars
    span = max(feature.fast_span, feature.slow_span)
    df = synthetic_ohlcv_with_funding(n=warm + COLD_START_PROBES)
    whole_history = df["close"].ewm(span=span, adjust=False).mean()

    inexact = [
        position
        for position in range(warm, len(df))
        if df["close"].iloc[position - warm : position + 1]
        .ewm(span=span, adjust=False)
        .mean()
        .iloc[-1]
        != whole_history.iloc[position]
    ]
    assert inexact == [], (
        f"the span-{span} EMA is not bit-exact after warmup_bars={warm} on "
        f"{len(inexact)}/{COLD_START_PROBES} probed bars; every Direction value "
        "then carries a seed the backtest did not have."
    )


@pytest.mark.parametrize("name", list_features())
def test_every_feature_declares_a_semver_version(name):
    assert re.match(r"^\d+\.\d+\.\d+$", get_feature(name).version)


def test_every_feature_defined_in_the_package_is_registered():
    """Manual registration is two places to forget, and forgetting either leaves a
    feature outside the lookahead probe -- which is the only reason the probe exists.

    Mirrors ``strategies/registry.py`` deliberately, so this catches the rot that
    pattern invites rather than trading it for an import-time side effect.
    """
    registered = {type(get_feature(name)) for name in list_features()}
    defined = {
        obj
        for module in (flow, trend, volatility)
        for obj in vars(module).values()
        if is_dataclass(obj) and getattr(obj, "__module__", None) == module.__name__
    }
    assert defined == registered, f"unregistered: {sorted(c.__name__ for c in defined - registered)}"
