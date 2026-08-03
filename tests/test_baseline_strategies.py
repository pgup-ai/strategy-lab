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
    up = trending_frame(n=strategy.warmup_bars + 300, slope=0.002)
    down = trending_frame(n=300, slope=-0.004)
    down.index = pd.date_range(
        up.index[-1] + pd.Timedelta("15min"),
        periods=len(down),
        freq="15min",
        tz="UTC",
        name="timestamp",
    )
    # Splice the down leg onto the end of the up leg as one continuous price
    # path. A single factor applied to every OHLC column keeps each bar's
    # high/low ordering intact; rescaling close alone would not.
    scale = up["close"].iloc[-1] / down["close"].iloc[0]
    for column in ("open", "high", "low", "close"):
        down[column] = down[column] * scale

    signals = strategy.generate_signals(pd.concat([up, down]))
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
