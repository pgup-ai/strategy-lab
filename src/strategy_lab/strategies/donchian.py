from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from strategy_lab.strategies.base import SignalSet, validate_ohlcv


@dataclass(frozen=True)
class Donchian:
    """Channel breakout with a faster exit channel than entry channel.

    ``shift(1)`` on every channel is load-bearing: without it the current bar is
    part of its own high/low window, and since ``high >= close`` within a bar,
    ``close > entry_high`` becomes unsatisfiable and the strategy never trades
    at all. That is not lookahead -- bar *t*'s own high is known at bar *t*'s
    close -- so the poison probe cannot see it; the breakout tests are what
    catch it.
    """

    name: str = "donchian"
    version: str = "1.0.0"
    entry_span: int = 96
    exit_span: int = 48
    allow_shorts: bool = True
    warmup_bars: int = 96

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        validate_ohlcv(df)

        entry_high = df["high"].rolling(self.entry_span).max().shift(1)
        entry_low = df["low"].rolling(self.entry_span).min().shift(1)
        exit_high = df["high"].rolling(self.exit_span).max().shift(1)
        exit_low = df["low"].rolling(self.exit_span).min().shift(1)

        long_entries = (df["close"] > entry_high).fillna(False)
        short_entries = (df["close"] < entry_low).fillna(False)
        if not self.allow_shorts:
            short_entries = pd.Series(False, index=df.index)

        return SignalSet(
            long_entries=long_entries,
            long_exits=(df["close"] < exit_low).fillna(False),
            short_entries=short_entries,
            short_exits=(df["close"] > exit_high).fillna(False),
            metadata={
                "entry_span": self.entry_span,
                "exit_span": self.exit_span,
                "allow_shorts": self.allow_shorts,
            },
        )
