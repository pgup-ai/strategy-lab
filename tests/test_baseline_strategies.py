from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_lab.strategies.registry import get_strategy, list_strategies
from tests.conftest import synthetic_ohlcv

BASELINES = ("tsmom", "ema_cross", "donchian", "multi_horizon")


def trending_frame(n: int = 600, slope: float = 0.002) -> pd.DataFrame:
    """Monotone uptrend with mild noise -- every trend baseline must go long here."""
    index = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC", name="timestamp")
    noise = np.random.default_rng(3).normal(0, 0.0003, n)
    close = 100 * np.exp(np.cumsum(np.full(n, slope) + noise))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.full(n, 500.0),
        },
        index=index,
    )


def reversal_frame(up_bars: int, down_bars: int = 300) -> pd.DataFrame:
    """An uptrend spliced onto a sustained downtrend, as one continuous price path."""
    up_leg = trending_frame(n=up_bars, slope=0.002)
    down_leg = trending_frame(n=down_bars, slope=-0.004)
    down_leg.index = pd.date_range(
        up_leg.index[-1] + pd.Timedelta("15min"),
        periods=down_bars,
        freq="15min",
        tz="UTC",
        name="timestamp",
    )
    # A single factor applied to every OHLC column, so the legs join without a
    # price gap and each bar keeps its high/low ordering. Rescaling close alone
    # would leave the other three columns at the pre-splice level.
    scale = up_leg["close"].iloc[-1] / down_leg["close"].iloc[0]
    for column in ("open", "high", "low", "close"):
        down_leg[column] = down_leg[column] * scale
    return pd.concat([up_leg, down_leg])


@pytest.mark.parametrize("name", BASELINES)
def test_baseline_is_registered(name):
    assert name in list_strategies()
    assert get_strategy(name).name == name


@pytest.mark.parametrize("name", BASELINES)
def test_disabling_shorts_does_not_disable_the_long_exit(name):
    """Leaving a long is independent of whether you act on the flip by shorting.

    The US ETF half of the program is long-only by design, so a baseline that
    never closes a position under ``--no-allow-shorts`` is buy-and-hold wearing a
    strategy's name -- it cannot be the floor the program is gated on. Three of
    these four derive ``long_exits`` from the short state, which is exactly the
    wiring that made zeroing the short state take the long exit with it.
    """
    with_shorts = get_strategy(name, allow_shorts=True)
    long_only = get_strategy(name, allow_shorts=False)
    df = reversal_frame(with_shorts.warmup_bars + 300)

    shorted_signals = with_shorts.generate_signals(df)
    long_only_signals = long_only.generate_signals(df)

    tail = slice(with_shorts.warmup_bars, None)
    assert shorted_signals.long_exits[tail].any(), (
        "this frame produces no long exits even with shorts on; it cannot test anything"
    )
    assert not long_only_signals.short_entries.any(), "long-only took a short entry"
    assert long_only_signals.long_exits[tail].any(), (
        f"{name} never closes a long when shorts are disabled"
    )
    # The stronger property the assertion above is a consequence of: the exit is
    # the trend flip itself, so disabling shorts must not move it by one bar.
    pd.testing.assert_series_equal(
        long_only_signals.long_exits, shorted_signals.long_exits, check_names=False
    )


def test_tsmom_goes_long_in_a_sustained_uptrend():
    strategy = get_strategy("tsmom")
    df = trending_frame(n=strategy.warmup_bars + 400)
    signals = strategy.generate_signals(df)

    tail = slice(strategy.warmup_bars, None)
    assert signals.long_entries[tail].any(), "expected long exposure in a clean uptrend"
    assert not signals.short_entries[tail].any(), "must not short a monotone uptrend"


def test_tsmom_reads_the_lookback_window_not_the_newest_bar():
    """One down bar inside an uptrend must not flip the position.

    This is the property that separates momentum from a one-bar direction rule,
    and it is the only thing here that notices if ``lookback`` stops being used:
    every other tsmom test passes just as happily with ``pct_change(1)``,
    because in a monotone trend the one-bar return has the trend's sign anyway.
    """
    strategy = get_strategy("tsmom")
    df = trending_frame(n=strategy.warmup_bars + 200)

    # A clean 1% down bar: opens at the previous close, closes 1% below it. Far
    # too small to turn the trailing return over ``lookback`` bars negative.
    previous_close = df["close"].iloc[-2]
    dropped = previous_close * 0.99
    df.loc[df.index[-1], ["open", "high", "low", "close"]] = [
        previous_close,
        previous_close,
        dropped,
        dropped,
    ]

    signals = strategy.generate_signals(df)
    assert signals.long_entries.iloc[-1], "a single down bar flipped a lookback-long trend"
    assert not signals.short_entries.iloc[-1]


def test_tsmom_flips_short_when_the_trend_reverses():
    strategy = get_strategy("tsmom")
    signals = strategy.generate_signals(reversal_frame(strategy.warmup_bars + 300))
    assert signals.short_entries.iloc[-100:].any(), "expected short exposure after a reversal"


def test_ema_cross_warmup_covers_ewm_convergence():
    """ewm(adjust=False) decays its seed rather than dropping it, so warmup is ~20x span."""
    strategy = get_strategy("ema_cross")
    assert strategy.warmup_bars >= 20 * strategy.slow_span


def test_ema_cross_is_long_while_fast_leads_slow():
    strategy = get_strategy("ema_cross")
    df = trending_frame(n=strategy.warmup_bars + 300)
    signals = strategy.generate_signals(df)

    tail = slice(strategy.warmup_bars, None)
    assert signals.long_entries[tail].all(), "fast EMA must lead slow throughout a clean uptrend"
    assert not signals.short_entries[tail].any()


def test_donchian_exit_channel_is_shorter_than_entry_channel():
    """The asymmetry is the point: enter slowly, leave faster, so trends can be ridden."""
    strategy = get_strategy("donchian")
    assert strategy.exit_span < strategy.entry_span


def test_donchian_breaks_out_long_on_a_new_channel_high():
    strategy = get_strategy("donchian")
    df = trending_frame(n=strategy.warmup_bars + 300)
    signals = strategy.generate_signals(df)

    tail = slice(strategy.warmup_bars, None)
    assert signals.long_entries[tail].any()
    assert not signals.short_entries[tail].any()


def test_donchian_uses_only_prior_bars_for_the_channel():
    """The channel must exclude the current bar.

    Two failure modes, opposite in shape, so both directions are pinned here.
    Drop ``shift(1)`` and the bar's own high joins the window, which -- because
    ``high >= close`` by construction -- makes ``close > channel high``
    unsatisfiable and the strategy never trades. Build the channel from
    ``close`` instead of ``high`` with no shift and the reverse happens: every
    bar ties its own window maximum and the comparison degenerates.
    """
    strategy = get_strategy("donchian")
    df = trending_frame(n=strategy.warmup_bars + 200)
    signals = strategy.generate_signals(df)

    tail = slice(strategy.warmup_bars, None)
    assert signals.long_entries[tail].any(), "channel appears to include the current bar's high"
    assert not signals.long_entries[tail].all(), "channel appears to include the current bar"


def test_donchian_exits_on_a_reverse_break_of_the_faster_channel():
    """Both exit series must actually fire, not merely be declared shorter.

    The exits are built exactly like the entries, so a dropped ``shift(1)`` on
    an exit channel is silent: ``low <= close`` within a bar makes
    ``close < exit_low`` unsatisfiable and a long would be held forever. Every
    other donchian test -- and the lookahead probe, and the determinism check --
    passes with that mutation, so this is the only thing standing behind the
    exit half of the strategy.
    """
    strategy = get_strategy("donchian")
    signals = strategy.generate_signals(reversal_frame(strategy.warmup_bars + 300))

    tail = slice(strategy.warmup_bars, None)
    assert signals.short_exits[tail].any(), "a rising close never cleared the exit channel high"
    assert signals.long_exits.iloc[-100:].any(), "a falling close never broke the exit channel low"


def test_multi_horizon_averages_several_lookbacks():
    strategy = get_strategy("multi_horizon")
    assert len(strategy.lookbacks) >= 3, "a single-lookback ensemble defeats the purpose"
    assert strategy.warmup_bars >= max(strategy.lookbacks)


def test_multi_horizon_is_long_when_every_horizon_agrees():
    strategy = get_strategy("multi_horizon")
    df = trending_frame(n=strategy.warmup_bars + 400)
    signals = strategy.generate_signals(df)

    tail = slice(strategy.warmup_bars, None)
    assert signals.long_entries[tail].any()
    assert not signals.short_entries[tail].any()


def test_multi_horizon_score_is_scale_invariant():
    """Denominating the same series in different units must not move the score."""
    strategy = get_strategy("multi_horizon")
    calm = trending_frame(n=strategy.warmup_bars + 200, slope=0.001)
    rescaled = calm.copy()
    for column in ("open", "high", "low", "close"):
        rescaled[column] = rescaled[column] * 10

    calm_score = strategy.generate_signals(calm).metadata["final_score"]
    rescaled_score = strategy.generate_signals(rescaled).metadata["final_score"]
    assert calm_score == pytest.approx(rescaled_score, rel=0.05)


def test_multi_horizon_divides_the_blend_by_realized_volatility():
    """Same trailing returns, far more noise, must not score the same.

    Scaling every price by a constant proves nothing here, because
    ``pct_change`` is scale-invariant on its own -- the blend survives that
    unchanged even with the normalization deleted. What isolates the
    denominator is holding the numerator fixed while moving volatility alone.

    An alternating per-bar multiplier does that. Every lookback is even, so bars
    ``t`` and ``t - lookback`` carry the same multiplier and the trailing return
    is unchanged to the last bit, while bar-to-bar returns pick up a ~5% zigzag
    that lifts realized volatility ~160x. Measured: the normalized score falls
    to 0.6% of the calm one; without the division the two are equal to ten
    decimal places.
    """
    strategy = get_strategy("multi_horizon")
    assert all(lookback % 2 == 0 for lookback in strategy.lookbacks), (
        "the alternating multiplier below only cancels out of an even lookback"
    )

    calm = trending_frame(n=strategy.warmup_bars + 200, slope=0.001)
    noisy = calm.copy()
    # Per-bar uniform scaling, so each bar keeps its own high/low ordering.
    zigzag = np.where(np.arange(len(calm)) % 2 == 0, 1.0, 1.05)
    for column in ("open", "high", "low", "close"):
        noisy[column] = noisy[column] * zigzag

    for lookback in strategy.lookbacks:
        assert calm["close"].pct_change(lookback).iloc[-1] == pytest.approx(
            noisy["close"].pct_change(lookback).iloc[-1], rel=1e-9
        ), f"the {lookback}-bar trailing return moved; this would no longer isolate volatility"

    calm_score = strategy.generate_signals(calm).metadata["final_score"]
    noisy_score = strategy.generate_signals(noisy).metadata["final_score"]
    assert abs(noisy_score) < 0.1 * abs(calm_score), (
        f"identical trailing returns scored {noisy_score} at high volatility versus "
        f"{calm_score} at low; the blend is not being divided by realized volatility"
    )


def test_multi_horizon_blend_is_a_t_statistic_under_the_null():
    """On a driftless random walk the blended score must sit on the order of 1.

    That is what ``/ (volatility * sqrt(lookback))`` is for: a trailing return
    grows like ``sqrt(lookback)`` under the null, so dividing it out leaves each
    horizon a unit-scale t-statistic and every horizon contributes comparably.
    Both halves of that denominator are load-bearing and the test above only
    covers one of them -- dropping ``sqrt(lookback)`` alone leaves volatility
    normalization intact, so identical-return frames still score differently and
    it passes. Measured RMS over these seeds: 0.71 correct, 6.07 with
    ``sqrt(lookback)`` dropped (the 192-bar horizon then carries ~2.2x the weight
    of the 24-bar one), 0.06 with the whole denominator dropped.
    """
    strategy = get_strategy("multi_horizon")
    scores = [
        strategy.generate_signals(
            synthetic_ohlcv(n=strategy.warmup_bars + 200, seed=seed)
        ).metadata["final_score"]
        for seed in range(30)
    ]
    rms = float(np.sqrt(np.mean(np.square(scores))))
    assert 0.25 < rms < 2.5, (
        f"blended score has RMS {rms:.3f} on a driftless random walk; a correctly "
        f"normalized t-statistic is order 1, so the denominator is wrong"
    )
