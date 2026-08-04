"""How much fuel the market is burning, and whether it is coiling instead.

Energy and Compression are one measurement read from both ends, so they are
exactly redundant by construction and both exist anyway: the charter reasons
about "is there fuel" and "is it coiling" as separate questions and a state
machine written against only one of them reads backwards half the time.

The measurement is a **rolling** percentile of realized volatility, never a
full-sample one. A full-sample rank at bar *t* changes when bars after *t*
arrive, which is the trap ``features/base`` exists to close; here it would also
be self-defeating, since a percentile against a history that includes the future
is not a statement about the present at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy_lab.features.base import mask_warmup, rolling_percentile
from strategy_lab.strategies.base import require_positive_span, validate_ohlcv


@dataclass(frozen=True)
class Energy:
    """Where realized volatility sits against its own recent history, 0..1.

    High means the market is moving, with no claim about which way -- the
    charter's own example is that Strength 0.8 with Energy 0.3 is a slow steady
    trend while Strength 0.2 with Energy 0.95 is violent chop, and the two
    readings must not collapse into one number.

    A percentile rather than a level, because a 4h realized vol of 2% meant
    something different in 2019 than it does now, and a threshold on the level
    silently re-tunes itself every year.
    """

    name: str = "energy"
    version: str = "1.0.0"
    vol_window: int = 24
    percentile_window: int = 480
    warmup_bars: int = 24 + 480 - 1

    def __post_init__(self) -> None:
        require_positive_span(self.name, "vol_window", self.vol_window)
        require_positive_span(self.name, "percentile_window", self.percentile_window)
        # The vol window has to fill before the percentile window can start
        # filling with anything, so the two costs add rather than overlap.
        object.__setattr__(
            self, "warmup_bars", self.vol_window + self.percentile_window - 1
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        validate_ohlcv(df)
        return mask_warmup(
            rolling_percentile(realized_volatility(df, window=self.vol_window),
                               window=self.percentile_window),
            warmup_bars=self.warmup_bars,
        )


@dataclass(frozen=True)
class Compression:
    """How tightly the market is coiled, 0..1 -- ``1 - Energy``.

    Kept as its own feature rather than left to callers to invert, because the
    sign is exactly what gets fumbled: "compression is rising" and "vol
    percentile is rising" are opposite statements about the same series.
    """

    name: str = "compression"
    version: str = "1.0.0"
    vol_window: int = 24
    percentile_window: int = 480
    warmup_bars: int = 24 + 480 - 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "warmup_bars", self._energy().warmup_bars)

    def _energy(self) -> Energy:
        return Energy(vol_window=self.vol_window, percentile_window=self.percentile_window)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return 1.0 - self._energy().compute(df)


@dataclass(frozen=True)
class CompressionRelease:
    """Compression giving way, -1..1: positive while the coil is unwinding.

    The charter's point about compression is that the level alone is not a
    signal -- markets stay quiet for months -- and that the *derivative* is where
    the information is. This is that derivative, with the sign chosen so the
    name is the reading: positive means compression is falling, which is a
    release. Differencing ``Compression`` by hand gives the opposite.
    """

    name: str = "compression_release"
    version: str = "1.0.0"
    vol_window: int = 24
    percentile_window: int = 480
    warmup_bars: int = 24 + 480

    def __post_init__(self) -> None:
        # One bar past Compression: a difference needs a previous value, and that
        # previous value has to be past ITS own warmup, not merely present.
        object.__setattr__(self, "warmup_bars", self._compression().warmup_bars + 1)

    def _compression(self) -> Compression:
        return Compression(
            vol_window=self.vol_window, percentile_window=self.percentile_window
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return mask_warmup(
            -self._compression().compute(df).diff(), warmup_bars=self.warmup_bars
        )


def realized_volatility(df: pd.DataFrame, *, window: int) -> pd.Series:
    """Standard deviation of log returns over ``window`` bars, in return units.

    Not annualized. Everything downstream is a percentile of this against itself,
    and a constant factor cannot change a rank -- so annualizing would only
    invite the bar count into a place it cannot affect the answer.
    """
    return np.log(df["close"]).diff().rolling(window).std()
