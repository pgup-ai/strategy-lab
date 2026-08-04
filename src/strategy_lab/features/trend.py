"""The four state dimensions that read price alone.

Direction, Strength, Persistence and Stability answer four different questions
about the same stretch of price -- which way, how hard, how straight, how
cleanly -- and Persistence and Stability share one rolling linear fit, which is
why they live in one module.

Stability deliberately does **not** normalize residual scatter by the window's
own price variance. That quantity is ``1 - sqrt(1 - R^2)``, a monotone function
of Persistence, so the two would be one feature under two names. Scaling by the
average true range instead brings in the bar's own high/low -- information the
fit never sees -- and measured on the stored BTC/USDT perp 4h history the two
correlate at 0.04.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy_lab.features.base import mask_warmup
from strategy_lab.strategies.base import require_positive_span, validate_ohlcv

# ewm(adjust=False) recurses from bar 0 and decays its seed rather than dropping
# it, so a span-n EMA is still wrong after n bars. Measured here on the stored
# BTC/USDT perp 4h closes at span 96: a cold start disagrees with the
# whole-history value on 299/300 probed bars at 10x span and on 277/300 at 15x,
# then on 0/300 at 20x. Phase 1a measured the same 20x at span 200.
_EWM_WARMUP_MULTIPLE = 20


@dataclass(frozen=True)
class Direction:
    """Which way price is leaning, -1..1, as an ATR-normalized EMA spread.

    The spread is divided by average true range so the number reads "how many
    bars' worth of range separates the two EMAs" rather than "how many dollars",
    which is the only form comparable across instruments and across eras of one
    instrument. It is then divided again by the square root of the two EMAs'
    lag difference, because that is the scale a *driftless* series produces on
    its own: a spread grows with the square root of the span it is measured
    across whether or not anything is trending, and the dimension is supposed to
    say how convincing the lean is, not how wide the spans are.

    That second divisor is what keeps the feature off its rails. Measured on the
    stored BTC/USDT perp 4h history the raw ATR ratio has median magnitude 1.5
    and reaches 7.5, so feeding it straight to ``tanh`` -- whose implicit unit is
    then one ATR -- pins 21.6% of bars past |0.99| and 51.6% past |0.9|. Scaled,
    nothing reaches |0.86| and the 5th/95th percentiles sit at -0.46/+0.54.

    Both EMAs are ``ewm(adjust=False)``, so warmup is 20x the slower span rather
    than the span itself.
    """

    name: str = "direction"
    version: str = "1.0.0"
    fast_span: int = 24
    slow_span: int = 96
    atr_window: int = 24
    warmup_bars: int = _EWM_WARMUP_MULTIPLE * 96

    def __post_init__(self) -> None:
        require_positive_span(self.name, "fast_span", self.fast_span)
        require_positive_span(self.name, "slow_span", self.slow_span)
        require_positive_span(self.name, "atr_window", self.atr_window)
        # Recomputed rather than trusted: warmup is a consequence of the spans.
        # The larger span binds, not ``slow_span`` -- a caller is free to set
        # ``fast_span`` above it, and then the "fast" EMA is the slower recursion.
        object.__setattr__(
            self,
            "warmup_bars",
            _EWM_WARMUP_MULTIPLE * max(self.fast_span, self.slow_span),
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        validate_ohlcv(df)
        fast = df["close"].ewm(span=self.fast_span, adjust=False).mean()
        slow = df["close"].ewm(span=self.slow_span, adjust=False).mean()
        spread = (fast - slow) / df["close"]
        range_fraction = true_range_fraction(df, window=self.atr_window)
        ratio = (spread / range_fraction).where(range_fraction != 0, 0.0)
        return mask_warmup(np.tanh(ratio / self._scale()), warmup_bars=self.warmup_bars)

    def _scale(self) -> float:
        """Spread a driftless series would show anyway, in ATRs.

        ``ewm(span=s, adjust=False)`` lags its input by ``(s - 1) / 2`` bars, so
        the two EMAs sample stretches of price half that many bars apart, and a
        random walk separates them by the square root of that distance.
        """
        lag_difference = abs(self.slow_span - self.fast_span) / 2.0
        # Equal spans make the spread identically zero, so the divisor only has
        # to be finite; without this it would be a zero under a zero.
        return float(np.sqrt(lag_difference)) if lag_difference else 1.0


@dataclass(frozen=True)
class Strength:
    """Directional efficiency, 0..1: net displacement divided by distance travelled.

    1.0 is a straight line -- every bar's move went the same way. Near 0 is the
    same energy spent going nowhere. The denominator is the whole feature: drop
    it for a constant and what is left is a normalized return, which cannot tell
    a one-way move from a round trip of the same size.

    Measured on the stored BTC/USDT perp 4h history at window 96: median 0.098,
    95th percentile 0.310, max 0.624.
    """

    name: str = "strength"
    version: str = "1.0.0"
    window: int = 96
    warmup_bars: int = 96

    def __post_init__(self) -> None:
        require_positive_span(self.name, "window", self.window)
        object.__setattr__(self, "warmup_bars", self.window)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        validate_ohlcv(df)
        close = df["close"]
        net_move = (close - close.shift(self.window)).abs()
        path_length = close.diff().abs().rolling(self.window).sum()
        # A window that never moved travelled no distance either, so the ratio is
        # 0/0. No displacement is no efficiency. NaN != 0, so warmup keeps its NaN.
        efficiency = (net_move / path_length).where(path_length != 0, 0.0)
        return mask_warmup(efficiency, warmup_bars=self.warmup_bars)


@dataclass(frozen=True)
class Persistence:
    """R-squared of an OLS fit of log price on time, 0..1.

    How much of the window's price variance a single straight line accounts for.
    1.0 is a perfect exponential; a random walk is nowhere near it -- measured
    median 0.42 on the stored BTC/USDT perp 4h history at window 96.
    """

    name: str = "persistence"
    version: str = "1.0.0"
    window: int = 96
    warmup_bars: int = 96

    def __post_init__(self) -> None:
        require_positive_span(self.name, "window", self.window)
        object.__setattr__(self, "warmup_bars", self.window)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        validate_ohlcv(df)
        r_squared, _ = trend_fit(np.log(df["close"]), window=self.window)
        return mask_warmup(r_squared, warmup_bars=self.warmup_bars)


@dataclass(frozen=True)
class Stability:
    """How cleanly price tracks its own trend line, 0..1. 1.0 is a ruler.

    Residual scatter around the Persistence fit, measured in average true ranges
    and divided by ``sqrt(window)`` because a random walk's scatter grows with
    the square root of the span -- without that the ratio would be a function of
    window length rather than of the market.

    Not normalized by the window's price variance, which would make this a
    monotone function of Persistence. Measured on the stored BTC/USDT perp 4h
    history at window 96: 5th percentile 0.75, median 0.84, 95th 0.90, never
    saturating at either rail, and correlated with Persistence at 0.04.
    """

    name: str = "stability"
    version: str = "1.0.0"
    window: int = 96
    warmup_bars: int = 96

    def __post_init__(self) -> None:
        require_positive_span(self.name, "window", self.window)
        object.__setattr__(self, "warmup_bars", self.window)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        validate_ohlcv(df)
        _, residual_std = trend_fit(np.log(df["close"]), window=self.window)
        range_fraction = true_range_fraction(df, window=self.window)
        scatter = residual_std / (range_fraction * np.sqrt(self.window))
        # A window with no range at all never left its trend line.
        stability = 1.0 - scatter.where(range_fraction != 0, 0.0).clip(0.0, 1.0)
        return mask_warmup(stability, warmup_bars=self.warmup_bars)


def trend_fit(log_price: pd.Series, *, window: int) -> tuple[pd.Series, pd.Series]:
    """R-squared and residual standard deviation of an OLS fit of price on time.

    The time regressor is a **fixed** ramp centred on its own window, applied as
    one dot product per bar, rather than a running index handed to
    ``rolling.corr``. The two agree mathematically -- correlation is invariant to
    an affine shift of either variable -- and disagree in float64, because a
    running index reaches 15,000 on the stored BTC history while the window it is
    compared over spans 96, and what survives that cancellation is noise. Measured
    against a whole-history run, a cold start under the running index differed by
    4.6e-10 relative at bar 1,200 and grew with position; the fixed kernel is
    bit-identical at every length and every position. What error the features
    still show comes from ``rolling.std`` below, whose online add/remove
    accumulates differently from a different starting bar -- 7.4e-12 relative,
    measured, and not growing.

    Only R-squared needs the regressor at all: the residual is what the fit
    leaves behind, ``sd(price) * sqrt(1 - R^2)``.

    A window whose price never moved is scored ``R^2 = 0`` with zero residual --
    a line explains none of a variance that is not there. Left alone that ratio
    is 0/0 and the feature goes ``NaN`` in the middle of a live series.
    """
    values = log_price.to_numpy(dtype="float64")
    centred_time = np.arange(window, dtype="float64") - (window - 1) / 2.0
    covariance = np.full(len(values), np.nan)
    if len(values) >= window:
        covariance[window - 1 :] = np.correlate(values, centred_time, mode="valid")

    price_std = log_price.rolling(window).std()
    # Both sums of squares carry the ddof=1 convention of ``price_std``, so the
    # factor cancels in the ratio and only the constant time variance is written
    # out: sum((t - mean t)^2) over 0..window-1.
    time_variance = window * (window**2 - 1) / 12.0
    price_variance = (window - 1) * price_std**2
    explained = pd.Series(covariance**2, index=log_price.index) / (
        time_variance * price_variance
    )
    r_squared = explained.where(price_variance != 0, 0.0).clip(0.0, 1.0)
    return r_squared, price_std * np.sqrt(1.0 - r_squared)


def true_range_fraction(df: pd.DataFrame, *, window: int) -> pd.Series:
    """Average true range over ``window``, as a fraction of the current close.

    True range rather than high-low, so an overnight gap counts as the move it
    was; as a fraction, so the number is comparable across price levels.
    """
    previous_close = df["close"].shift()
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return (true_range / df["close"]).rolling(window).mean()
