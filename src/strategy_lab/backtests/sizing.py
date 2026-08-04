"""Volatility-targeted position sizing.

Fixed fractional sizing gives a quiet market and a violent one the same
notional, which means they get wildly different *risk* -- the main reason naive
trend backtests show drawdowns nobody would actually sit through. Targeting a
constant annualized volatility instead makes the weight inversely proportional
to realized volatility, so risk is what stays constant.

The output is a **size multiplier, never a direction**: it is non-negative, and
the strategy keeps ownership of the sign. Feeding it in as ``SignalSet``'s
``position_size`` composes with the engine's non-compounding sizing without
either layer having to know about the other.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

DEFAULT_VOL_SPAN = 96


class SizeMode(str, Enum):
    """How the engine turns ``position_pct`` into a per-entry size.

    ``FIXED`` is the historical behaviour and leaves the strategy's own scale
    (if any) untouched. ``VOL_TARGET`` replaces it with
    :func:`volatility_target_weights`.
    """

    FIXED = "fixed"
    VOL_TARGET = "vol-target"


def realized_volatility(
    returns: pd.Series,
    *,
    span: int,
    bars_per_year: float,
) -> pd.Series:
    """Annualized exponentially-weighted volatility of ``returns``.

    ``ewm`` rather than ``rolling`` so the estimate reacts to a regime change
    instead of waiting for the shock to leave a fixed window, and
    ``adjust=False`` so the recursion matches what the event-driven path
    computes bar by bar.

    ``adjust=False`` is recursive from bar 0 and decays its seed rather than
    dropping it, so a span-n estimate is still wrong after n bars: a span-200
    EMA only becomes bit-exact around 4000. A strategy consuming this therefore
    needs ``warmup_bars`` of roughly 20x ``span``, which
    ``tests/test_strategy_metadata.py`` measures rather than trusts.
    """
    if bars_per_year <= 0:
        raise ValueError("bars_per_year must be > 0")

    per_bar = returns.astype("float64").ewm(span=span, adjust=False).std()
    return per_bar * np.sqrt(bars_per_year)


def volatility_target_weights(
    returns: pd.Series,
    *,
    target_annual_vol: float,
    bars_per_year: float,
    span: int = DEFAULT_VOL_SPAN,
    max_weight: float = 2.0,
) -> pd.Series:
    """Per-bar size multiplier that holds annualized risk at ``target_annual_vol``.

    ``target / realized`` is the whole idea, and the whole danger: a flat
    stretch has zero realized volatility and ``target / 0`` is infinity, which
    would silently become an infinite position rather than an error anyone
    notices. The divide is guarded and the result clipped into
    ``[0, max_weight]``, so a degenerate estimate produces a capped weight
    instead of a fictional one.

    Warmup bars have no volatility estimate yet; they get weight zero rather
    than an unbounded one, so a cold start under-trades instead of betting the
    account on three bars of history.
    """
    if target_annual_vol <= 0:
        raise ValueError("target_annual_vol must be > 0")
    if max_weight <= 0:
        raise ValueError("max_weight must be > 0")

    realized = realized_volatility(returns, span=span, bars_per_year=bars_per_year)
    weights = target_annual_vol / realized.where(realized > 0)
    return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, max_weight)


__all__ = [
    "DEFAULT_VOL_SPAN",
    "SizeMode",
    "realized_volatility",
    "volatility_target_weights",
]
