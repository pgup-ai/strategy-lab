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

    def __post_init__(self) -> None:
        # Always recomputed, so a ``warmup_bars=`` passed by a caller is
        # overwritten: it is a measured consequence of the slow span, not a free
        # parameter. Left as a field so ``dataclasses.fields`` still reports it.
        # The slow span binds -- the fast EMA converges strictly sooner.
        object.__setattr__(self, "warmup_bars", _EWM_WARMUP_MULTIPLE * self.slow_span)

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        validate_ohlcv(df)

        fast = df["close"].ewm(span=self.fast_span, adjust=False).mean()
        slow = df["close"].ewm(span=self.slow_span, adjust=False).mean()

        long_state = (fast > slow).fillna(False)
        short_state = (fast < slow).fillna(False)

        # Only the entry is gated; ``short_state`` stays the raw crossover
        # state, because the fast EMA losing the slow one closes a long whether
        # or not shorts are enabled. Gating the state itself left a long-only
        # run with no exit at all.
        short_entries = short_state if self.allow_shorts else pd.Series(False, index=df.index)

        return SignalSet(
            long_entries=long_state,
            long_exits=short_state,
            short_entries=short_entries,
            short_exits=long_state,
            metadata={
                "fast_span": self.fast_span,
                "slow_span": self.slow_span,
                "allow_shorts": self.allow_shorts,
            },
        )
