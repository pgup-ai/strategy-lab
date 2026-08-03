from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from strategy_lab.strategies.base import SignalSet, validate_ohlcv

# ewm(adjust=False) is recursive from bar 0 and decays its seed rather than
# dropping it, so a span-n EMA is still wrong after n bars. Measured in Phase 1a:
# a span-200 EMA only becomes bit-exact around 4000 bars.
_EWM_WARMUP_MULTIPLE = 20


@dataclass(frozen=True)
class EmaCross:
    """Long while the fast EMA leads the slow one, short (or flat) when it doesn't."""

    name: str = "ema_cross"
    version: str = "1.0.0"
    fast_span: int = 48
    slow_span: int = 192
    allow_shorts: bool = True
    warmup_bars: int = _EWM_WARMUP_MULTIPLE * 192

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        validate_ohlcv(df)

        fast = df["close"].ewm(span=self.fast_span, adjust=False).mean()
        slow = df["close"].ewm(span=self.slow_span, adjust=False).mean()

        long_state = (fast > slow).fillna(False)
        short_state = (fast < slow).fillna(False)
        if not self.allow_shorts:
            short_state = pd.Series(False, index=df.index)

        return SignalSet(
            long_entries=long_state,
            long_exits=short_state,
            short_entries=short_state,
            short_exits=long_state,
            metadata={
                "fast_span": self.fast_span,
                "slow_span": self.slow_span,
                "allow_shorts": self.allow_shorts,
            },
        )
