from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from strategy_lab.strategies.base import SignalSet, validate_ohlcv


@dataclass(frozen=True)
class TurnaroundV1:
    name: str = "turnaround_v1"
    allow_shorts: bool = True

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

        return SignalSet(
            long_entries=long_entries.fillna(False),
            long_exits=short_entries.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=long_entries.fillna(False),
            metadata={"logic": "two opposite candles followed by a reversal candle"},
        )
