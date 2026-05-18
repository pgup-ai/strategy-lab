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
class TurnaroundV2:
    name: str = "turnaround_v2"
    ema_trend_span: int = 200
    ema_extension_span: int = 20
    long_extension: float = 0.99
    short_extension: float = 1.01
    allow_shorts: bool = True

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        validate_ohlcv(df)

        red1 = df["close"].shift(1) < df["open"].shift(1)
        red2 = df["close"].shift(2) < df["open"].shift(2)
        green = df["close"] > df["open"]

        green1 = df["close"].shift(1) > df["open"].shift(1)
        green2 = df["close"].shift(2) > df["open"].shift(2)
        red = df["close"] < df["open"]

        ema_trend = df["close"].ewm(span=self.ema_trend_span, adjust=False).mean()
        ema_extension = df["close"].ewm(span=self.ema_extension_span, adjust=False).mean()

        long_entries = (
            red1
            & red2
            & green
            & (df["close"] > ema_trend)
            & (df["close"] < ema_extension * self.long_extension)
        )
        short_entries = (
            green1
            & green2
            & red
            & (df["close"] < ema_trend)
            & (df["close"] > ema_extension * self.short_extension)
        )
        if not self.allow_shorts:
            short_entries = pd.Series(False, index=df.index)
        long_entries = long_entries.fillna(False)
        short_entries = short_entries.fillna(False)
        trend_failure_long_exits, trend_failure_short_exits = ema_trend_failure_exits(
            df,
            ema_span=self.ema_trend_span,
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
                "allow_shorts": self.allow_shorts,
                "ema_trend_span": self.ema_trend_span,
                "ema_extension_span": self.ema_extension_span,
                "long_extension": self.long_extension,
                "short_extension": self.short_extension,
                "setup_stop": "long below setup low, short above setup high",
            },
        )
