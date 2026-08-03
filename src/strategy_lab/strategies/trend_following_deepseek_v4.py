from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from strategy_lab.strategies.base import (
    SignalSet,
    validate_ohlcv,
)


@dataclass(frozen=True)
class TrendFollowingDeepseekV4:
    name: str = "trend_following_deepseek_v4"
    version: str = "1.0.0"
    warmup_bars: int = 40
    trend_sma_span: int = 40
    max_extension: float = 1.20
    allow_shorts: bool = False

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        validate_ohlcv(df)

        red1 = df["close"].shift(1) < df["open"].shift(1)
        red2 = df["close"].shift(2) < df["open"].shift(2)
        green = df["close"] > df["open"]

        sma_trend = df["close"].rolling(self.trend_sma_span).mean()

        long_entries = (
            red1
            & red2
            & green
            & (df["close"] > sma_trend)
            & (df["close"] < sma_trend * self.max_extension)
        )
        long_entries = long_entries.fillna(False)

        return SignalSet(
            long_entries=long_entries,
            long_exits=pd.Series(False, index=df.index),
            short_entries=pd.Series(False, index=df.index),
            short_exits=pd.Series(False, index=df.index),
            setup_stop_loss=None,
            trend_failure_long_exits=None,
            trend_failure_short_exits=None,
            metadata={
                "allow_shorts": False,
                "trend_sma_span": self.trend_sma_span,
                "max_extension": self.max_extension,
            },
        )
