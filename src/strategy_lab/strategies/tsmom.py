from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from strategy_lab.strategies.base import SignalSet, validate_ohlcv


@dataclass(frozen=True)
class Tsmom:
    """Time-series momentum: the sign of the trailing return is the position.

    The reference trend baseline. Every later model in the MDE program has to
    beat this out-of-sample to justify its complexity.

    ``pct_change`` only, so warmup is exactly the lookback -- the value at bar
    *t* reads bars *t* and *t - lookback* and nothing else, with no recursion to
    converge.
    """

    name: str = "tsmom"
    version: str = "1.0.0"
    lookback: int = 96
    warmup_bars: int = 96
    allow_shorts: bool = True

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        validate_ohlcv(df)

        trailing_return = df["close"].pct_change(self.lookback)
        long_state = (trailing_return > 0).fillna(False)
        short_state = (trailing_return < 0).fillna(False)
        if not self.allow_shorts:
            short_state = pd.Series(False, index=df.index)

        return SignalSet(
            long_entries=long_state,
            long_exits=short_state,
            short_entries=short_state,
            short_exits=long_state,
            metadata={
                "lookback": self.lookback,
                "allow_shorts": self.allow_shorts,
            },
        )
