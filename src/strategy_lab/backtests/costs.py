"""Execution costs and perp funding.

Two costs with different shapes live here. Fees and slippage are per-fill
frictions and scale with how badly a fill goes, so ``CostModel.stressed``
multiplies them. Funding is a *market rate* settled between longs and shorts on
the venue's schedule; multiplying it would model a different market, not a worse
fill, so ``apply_funding`` never sees a ``CostModel`` and cannot be stressed.

Funding is the reason perp trend-following differs from futures
trend-following, and it is large: measured over the stored history, BTC/USDT
perp funding averages +11.65%/year and ETH/USDT +13.97%/year, both paid by
longs. A persistently long strategy over the 2019-2026 window pays roughly the
whole of buy-and-hold. Getting it approximately right is not good enough.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """Per-fill frictions, as one-way rates against notional.

    Defaults match the engine's historical ``fees``/``slippage`` defaults so a
    run that does not name a cost model is priced exactly as it was before this
    type existed.
    """

    fee: float = 0.0005
    slippage: float = 0.0005

    def stressed(self, multiple: float) -> CostModel:
        """The same model with execution frictions scaled by ``multiple``.

        Funding is deliberately absent from this type. A 3x cost run asks "what
        if my fills are three times worse"; it does not ask "what if the whole
        market's carry tripled", which is a different instrument.
        """
        return replace(self, fee=self.fee * multiple, slippage=self.slippage * multiple)

    @property
    def round_trip(self) -> float:
        """Fee plus slippage on both legs -- what one completed trade costs."""
        return 2.0 * (self.fee + self.slippage)


def apply_funding(*, positions: pd.Series, funding: pd.Series) -> pd.Series:
    """Funding cash flows for ``positions``, aligned to the position index.

    Returns one value per bar, in whatever units ``positions`` carries: pass
    signed notional to get currency, pass a signed weight to get a return
    fraction. Cost is ``-position x rate`` -- a long pays a positive rate, a
    short receives it, a flat position pays nothing.

    Funding is a **discrete cash flow at settlement**, not a per-bar drag. Each
    settlement is charged once, to the bar whose interval contains it, and bars
    with no settlement are exactly zero. Reindexing with ``ffill`` instead would
    smear one 8h rate across every 4h bar and roughly double the measured cost.

    Settlements are matched to bars by **containment, never equality**. Binance
    stamps them up to 47ms *after* the boundary -- 3,260 of BTC's 7,559 stored
    settlements are off-grid -- so an equality join against a generated
    ``date_range(freq="8h")`` silently drops 43% of them. Nothing here assumes
    an 8h schedule either: the assignment is derived from the timestamps
    themselves, so a contract that settles hourly, or changes interval
    mid-history, is handled without a code change.
    """
    index = _sorted_index(positions)
    values = np.zeros(len(index), dtype="float64")
    slot, _, rate = _contained(index, funding)
    if len(slot):
        held = positions.to_numpy(dtype="float64")[slot]
        # Coarse bars hold several settlements -- 8h funding under daily bars
        # settles three times a day -- so accumulate rather than assign.
        np.add.at(values, slot, -held * rate)
    return pd.Series(values, index=index, dtype="float64")


def funding_ledger(*, positions: pd.Series, funding: pd.Series) -> pd.DataFrame:
    """One row per settlement actually charged -- the audit trail behind the total.

    Same containment as :func:`apply_funding`, so the two cannot disagree about
    which settlements were counted. Funding is the number that decides whether a
    perp result is tradeable, and a single aggregate is not something a human can
    check against the venue; this is.
    """
    index = _sorted_index(positions)
    slot, settled, rate = _contained(index, funding)
    held = positions.to_numpy(dtype="float64")[slot]
    return pd.DataFrame(
        {
            "bar": index[slot],
            "notional": held,
            "rate": rate,
            "cash_flow": -held * rate,
        },
        index=pd.Index(settled, name="settled_at"),
    )


def _sorted_index(positions: pd.Series) -> pd.DatetimeIndex:
    index = positions.index
    if not index.is_monotonic_increasing:
        raise ValueError("positions must be sorted by timestamp; containment uses a binary search")
    return index


def _contained(
    index: pd.DatetimeIndex, funding: pd.Series
) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Settlements that fall inside the position window, and the bar each lands on."""
    rates = funding.dropna()
    if rates.empty or len(index) == 0:
        empty = np.empty(0, dtype="int64")
        return empty, index[:0], np.empty(0, dtype="float64")

    settled = pd.DatetimeIndex(rates.index)
    # side="right" minus one is the containing bar: the last bar whose open is
    # at or before the settlement instant. Bars are left-closed, so a settlement
    # exactly on a bar open belongs to that bar and one a millisecond earlier
    # belongs to the previous one.
    slot = index.searchsorted(settled, side="right") - 1
    inside = (slot >= 0) & (settled < _window_end(index))
    return slot[inside], settled[inside], rates.to_numpy(dtype="float64")[inside]


def _window_end(index: pd.DatetimeIndex) -> pd.Timestamp:
    """Exclusive right edge of the last bar.

    The last bar covers an interval like every other, so a settlement inside it
    is charged; one past it belongs to data we do not have. The bar span is the
    median spacing rather than the last gap, so a single missing candle does not
    redefine the window.
    """
    if len(index) < 2:
        return pd.Timestamp.max.tz_localize("UTC") if index.tz is not None else pd.Timestamp.max
    span = pd.Timedelta(int(np.median(np.diff(index.asi8))), unit="ns")
    return index[-1] + span


__all__ = ["CostModel", "apply_funding", "funding_ledger"]
