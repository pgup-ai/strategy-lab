from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_lab.strategies.registry import get_strategy, list_strategies

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
