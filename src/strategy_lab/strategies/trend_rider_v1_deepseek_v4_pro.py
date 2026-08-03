from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from strategy_lab.filters.position_sizing import atr_position_scale
from strategy_lab.filters.regime import (
    momentum_divergence_exit,
    regime_exit,
    trend_filter,
    volatility_filter,
)
from strategy_lab.strategies.base import SignalSet, validate_ohlcv


@dataclass(frozen=True)
class TrendRiderV1DeepseekV4Pro:
    name: str = "trend_rider_v1_deepseek_v4_pro"
    version: str = "1.0.0"
    warmup_bars: int = 40
    sma_trend_span: int = 40
    roc_momentum_period: int = 26
    atr_period: int = 14
    atr_max_ratio: float = 0.10
    failure_bars: int = 3
    target_atr_ratio: float = 0.06
    min_position_scale: float = 0.3
    max_position_scale: float = 1.0
    allow_shorts: bool = False

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        validate_ohlcv(df)

        red1 = df["close"].shift(1) < df["open"].shift(1)
        red2 = df["close"].shift(2) < df["open"].shift(2)
        green = df["close"] > df["open"]

        green1 = df["close"].shift(1) > df["open"].shift(1)
        green2 = df["close"].shift(2) > df["open"].shift(2)
        red = df["close"] < df["open"]

        trend_up = trend_filter(df, span=self.sma_trend_span)
        vol_ok = volatility_filter(
            df,
            atr_period=self.atr_period,
            max_atr_ratio=self.atr_max_ratio,
        )

        long_entries = (
            red1
            & red2
            & green
            & trend_up
            & vol_ok
        )
        short_entries = (
            green1
            & green2
            & red
            & ~trend_up
            & vol_ok
        )
        if not self.allow_shorts:
            short_entries = pd.Series(False, index=df.index)

        long_entries = long_entries.fillna(False)
        short_entries = short_entries.fillna(False)

        continuation_long_exits = _continuation_failure_exits(
            df,
            failure_bars=self.failure_bars,
            direction="long",
        )
        continuation_short_exits = _continuation_failure_exits(
            df,
            failure_bars=self.failure_bars,
            direction="short",
        )
        regime_long_exits = regime_exit(df, span=self.sma_trend_span)
        regime_short_exits = ~regime_exit(df, span=self.sma_trend_span)
        momentum_long_exits = momentum_divergence_exit(
            df,
            period=self.roc_momentum_period,
        )
        momentum_short_exits = ~momentum_divergence_exit(
            df,
            period=self.roc_momentum_period,
        )

        long_exits = (
            short_entries
            | continuation_long_exits
            | regime_long_exits
            | momentum_long_exits
        )
        short_exits = (
            long_entries
            | continuation_short_exits
            | regime_short_exits
            | momentum_short_exits
        )

        position_scale = atr_position_scale(
            df,
            atr_period=self.atr_period,
            target_atr_ratio=self.target_atr_ratio,
            min_scale=self.min_position_scale,
            max_scale=self.max_position_scale,
        )

        return SignalSet(
            long_entries=long_entries,
            long_exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            position_size=position_scale,
            metadata={
                "allow_shorts": self.allow_shorts,
                "trend_sma_span": self.sma_trend_span,
                "roc_momentum_period": self.roc_momentum_period,
                "atr_period": self.atr_period,
                "atr_max_ratio": self.atr_max_ratio,
                "failure_bars": self.failure_bars,
                "target_atr_ratio": self.target_atr_ratio,
                "min_position_scale": self.min_position_scale,
                "max_position_scale": self.max_position_scale,
                "exits": "opposite_signal + continuation_failure + regime_break + momentum_divergence",
                "position_sizing": "ATR-based volatility scaling",
            },
        )


def _continuation_failure_exits(
    df: pd.DataFrame,
    *,
    failure_bars: int,
    direction: str,
) -> pd.Series:
    if failure_bars < 1:
        raise ValueError("failure_bars must be >= 1")

    close_change = df["close"].diff()
    if direction == "long":
        adverse = close_change < 0
    else:
        adverse = close_change > 0

    exits = adverse.rolling(failure_bars, min_periods=failure_bars).sum() == failure_bars
    return exits.fillna(False)
