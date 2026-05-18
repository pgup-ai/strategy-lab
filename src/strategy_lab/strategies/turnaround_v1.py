from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from strategy_lab.strategies.base import (
    SignalSet,
    ema_trend_failure_exits,
    setup_invalidation_stop_loss,
    validate_ohlcv,
)


@dataclass(frozen=True)
class TurnaroundV1:
    name: str = "turnaround_v1"
    allow_shorts: bool = True
    trend_failure_ema_span: int = 200

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        validate_ohlcv(df)

        red1 = df["close"].shift(1) < df["open"].shift(1)
        red2 = df["close"].shift(2) < df["open"].shift(2)
        green = df["close"] > df["open"]

        green1 = df["close"].shift(1) > df["open"].shift(1)
        green2 = df["close"].shift(2) > df["open"].shift(2)
        red = df["close"] < df["open"]

        long_entries = red1 & red2 & green
        short_entries = green1 & green2 & red
        if not self.allow_shorts:
            short_entries = pd.Series(False, index=df.index)
        long_entries = long_entries.fillna(False)
        short_entries = short_entries.fillna(False)
        trend_failure_long_exits, trend_failure_short_exits = ema_trend_failure_exits(
            df,
            ema_span=self.trend_failure_ema_span,
        )

        return SignalSet(
            long_entries=long_entries,
            long_exits=short_entries,
            short_entries=short_entries,
            short_exits=long_entries,
            setup_stop_loss=setup_invalidation_stop_loss(
                df,
                long_entries=long_entries,
                short_entries=short_entries,
            ),
            trend_failure_long_exits=trend_failure_long_exits,
            trend_failure_short_exits=trend_failure_short_exits,
            metadata={
                "logic": "two opposite candles followed by a reversal candle",
                "setup_stop": "long below setup low, short above setup high",
                "trend_failure_ema_span": self.trend_failure_ema_span,
            },
        )
