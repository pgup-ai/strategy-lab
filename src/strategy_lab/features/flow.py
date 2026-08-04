"""Who else is here, and how expensive it has become to stand with them.

Participation is the per-instrument half of the charter's dimension; the
cross-sectional half is ``features.cross_sectional.breadth``, which reads a
snapshot across instruments rather than one series through time.

Crowding is crypto-only. Nothing about equity data carries a funding rate, so
``Crowding.compute`` refuses a frame without one rather than returning a number:
a silent 0.5 there reads "measured, and nobody is crowded", which is a claim
about a market this feature never looked at.

Funding arrives on the venue's own schedule -- 8h against 4h bars, stamped up to
47 ms past the boundary -- so :func:`align_funding_to_bars` delegates to
``backtests.costs.apply_funding`` rather than reimplementing containment. One
copy of the 47 ms rule exists; which module holds it matters less than that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy_lab.backtests.costs import apply_funding
from strategy_lab.features.base import mask_warmup, rolling_percentile, rolling_zscore
from strategy_lab.strategies.base import require_positive_span, validate_ohlcv

FUNDING_COLUMN = "funding_rate"


@dataclass(frozen=True)
class Participation:
    """Where this bar's volume sits against the instrument's own recent history, 0..1.

    A percentile, not a level: BTC's 4h volume has moved through orders of
    magnitude since 2019 and any absolute threshold on it dates immediately.

    This is one instrument answering "is anyone else here". A rising market on
    falling participation is the charter's exhaustion pattern, and it needs the
    cross-sectional half -- ``cross_sectional.breadth`` -- to be read fully.
    """

    name: str = "participation"
    version: str = "1.0.0"
    window: int = 480
    warmup_bars: int = 479

    def __post_init__(self) -> None:
        require_positive_span(self.name, "window", self.window)
        object.__setattr__(self, "warmup_bars", self.window - 1)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        validate_ohlcv(df)
        return mask_warmup(
            rolling_percentile(df["volume"], window=self.window),
            warmup_bars=self.warmup_bars,
        )


@dataclass(frozen=True)
class Crowding:
    """How extreme the recent cost of carry is against its own history, 0..1.

    0.5 is neutral. Above it longs are paying to stay long -- the crowded side is
    long -- and below it shorts are paying. Extremes on either side are the
    charter's exhaustion condition, which is why this is one signed axis mapped
    onto 0..1 rather than a magnitude that throws the side away.

    Measured over **accrued** funding rather than the raw settlement series. A
    per-bar funding column is mostly zeros by construction -- 8h settlements
    against 4h bars leave every other bar empty -- and a z-score of that measures
    the settlement schedule more than the market. Summing over ``accrual_window``
    bars first gives "what a long paid over the last day", which is defined on
    every bar and needs no assumption about the venue's interval.
    """

    name: str = "crowding"
    version: str = "1.0.0"
    accrual_window: int = 6
    zscore_window: int = 180
    warmup_bars: int = 6 + 180 - 2

    def __post_init__(self) -> None:
        require_positive_span(self.name, "accrual_window", self.accrual_window)
        require_positive_span(self.name, "zscore_window", self.zscore_window)
        # Both windows are rolling, so each costs its length less one bar.
        object.__setattr__(
            self, "warmup_bars", self.accrual_window + self.zscore_window - 2
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        validate_ohlcv(df)
        if FUNDING_COLUMN not in df.columns:
            raise ValueError(
                f"{self.name} needs a {FUNDING_COLUMN!r} column and this frame has "
                f"{sorted(df.columns)}. Attach one with align_funding_to_bars(); "
                "returning a neutral 0.5 here would claim nobody is crowded."
            )
        accrued = df[FUNDING_COLUMN].rolling(self.accrual_window).sum()
        zscore = rolling_zscore(accrued, window=self.zscore_window)
        # tanh rather than a clip: carry z-scores have long tails and the
        # interesting comparisons are between a 2-sigma and a 5-sigma extreme.
        return mask_warmup((np.tanh(zscore) + 1.0) / 2.0, warmup_bars=self.warmup_bars)


def align_funding_to_bars(index: pd.DatetimeIndex, funding: pd.Series) -> pd.Series:
    """The funding a unit long paid on each bar, zero where nothing settled.

    Delegates to ``apply_funding``, which charges each settlement to the bar whose
    interval **contains** it. That matters: Binance stamps settlements up to 47 ms
    after the 8h boundary, so an equality join against a generated ``8h``
    ``date_range`` silently drops 43% of BTC's stored history.

    ``apply_funding`` returns the cost *to* a position, so a unit long comes back
    negated; the sign is flipped once here to recover the rate itself.
    """
    unit_long = pd.Series(1.0, index=index)
    return -apply_funding(positions=unit_long, funding=funding)
