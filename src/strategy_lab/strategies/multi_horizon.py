from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from strategy_lab.strategies.base import SignalSet, validate_ohlcv

_VOL_SPAN = 96


@dataclass(frozen=True)
class MultiHorizon:
    """Mean of volatility-normalized trailing returns across several lookbacks.

    Normalizing by realized volatility puts every horizon on the same scale, so a
    fast noisy horizon cannot dominate the blend through raw magnitude alone. The
    ensemble exists to remove the single-lookback choice, which is where trend
    backtests usually overfit -- not to beat any one lookback.

    ``rolling`` only, so warmup is the longest lookback rather than a multiple of
    it. Note that ``rolling(...).std()`` is nonetheless not bit-reproducible
    across a cold start: pandas computes it by adding and removing observations
    one at a time, and the removals leave rounding residue that depends on how
    much history preceded the window. Measured at warmup=192, the score differs
    from the whole-history value on 195 of 300 probed bars -- by at most 1.1e-15,
    against a smallest observed |score| of 5.9e-3. The signs, which are all the
    signals read, are identical on every bar.
    """

    name: str = "multi_horizon"
    version: str = "1.0.0"
    lookbacks: tuple[int, ...] = (24, 48, 96, 192)
    allow_shorts: bool = True
    warmup_bars: int = 192
    entry_threshold: float = 0.0

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        validate_ohlcv(df)

        returns = df["close"].pct_change()
        volatility = returns.rolling(_VOL_SPAN).std()

        # Dividing by ``volatility * sqrt(lookback)`` turns each raw trailing
        # return into a t-statistic, which is what makes horizons comparable:
        # a 192-bar return is naturally ~sqrt(8)x a 24-bar one, and a noisy
        # regime inflates both. Without it the blend is just an average of raw
        # returns and the longest, most volatile horizon sets the sign alone.
        scores = [
            (df["close"].pct_change(lookback) / (volatility * (lookback**0.5)))
            for lookback in self.lookbacks
        ]
        score = sum(scores) / len(scores)

        long_state = (score > self.entry_threshold).fillna(False)
        short_state = (score < -self.entry_threshold).fillna(False)
        if not self.allow_shorts:
            short_state = pd.Series(False, index=df.index)

        return SignalSet(
            long_entries=long_state,
            long_exits=short_state,
            short_entries=short_state,
            short_exits=long_state,
            metadata={
                "lookbacks": list(self.lookbacks),
                "allow_shorts": self.allow_shorts,
                # The score on the frame's LAST bar -- the current reading, which
                # is the useful one for a live or replay context.
                "final_score": float(score.iloc[-1]) if not score.empty else 0.0,
            },
        )
