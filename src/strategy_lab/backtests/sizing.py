"""Volatility-scaled *entry* sizing.

Fixed fractional sizing gives a quiet market and a violent one the same
notional, which means they get wildly different *risk* -- the main reason naive
trend backtests show drawdowns nobody would actually sit through. Making the
weight inversely proportional to realized volatility sizes the *entry* for the
regime it is opened into.

**It does not retarget an open position, and the name says so deliberately.**
The weights below are a full per-bar series, but the engine hands them to
``vbt.Portfolio.from_signals``, which defaults to ``accumulate=False`` and
therefore ignores repeated same-direction entry signals while a position is
open. Measured on vectorbt 1.0.0 with a flat close, an entry on every bar and
``size = [1,1,1,1,5,5,5,5]``: one order, size 1.0, at the first bar, and an
assets path of ``[1]*8``. So only the weight on the bar that *opens* a position
is ever consumed, and a strategy that stays long through a calm-to-violent
regime change carries its calm-regime notional the whole way.

That is why this is ``vol-scaled-entry`` rather than volatility targeting:
targeting holds realized risk constant, and holding it constant requires
rebalancing an open position. The benefit that remains is real but narrower --
the book declines to *open* a large position in a violent regime.

Continuous rebalancing needs order-level control that ``from_signals`` cannot
express (it is signal-driven, one fill per state change). The real fix is R6 in
the charter, where the engine moves to ``from_orders`` or a custom
continuous-rebalance simulator -- open question Q4 in
``docs/research/2026-08-03-market-dynamics-engine.md``. Until then no caller
should read a constant-risk claim into these weights;
``tests/test_sizing.py::test_a_later_weight_does_not_resize_an_open_position``
pins the actual behaviour.

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

# ``ewm(adjust=False)`` is recursive from bar 0 and decays its seed rather than
# dropping it, so a span-n estimate is still wrong after n bars. Measured in
# Phase 1a on this repo's own EMAs: a span-200 recursion only becomes bit-exact
# around 4000 bars. The same multiple governs the volatility estimator below,
# which is the same recursion applied to squared deviations.
EWM_WARMUP_MULTIPLE = 20


def vol_warmup_bars(span: int) -> int:
    """Bars an ``ewm(adjust=False)`` estimate of ``span`` needs to converge.

    This is what makes a volatility-scaled run reproducible. The estimate is
    *finite* from a handful of observations -- it just is not yet a measurement
    of anything -- so without a declared warmup the weight on an early bar is a
    function of where the frame happens to start, and moving ``--start`` changes
    the size of entries over market the run has in common with its predecessor.
    """
    return EWM_WARMUP_MULTIPLE * span


class SizeMode(str, Enum):
    """How the engine turns ``position_pct`` into a per-entry size.

    ``FIXED`` is the historical behaviour and leaves the strategy's own scale
    (if any) untouched. ``VOL_SCALED_ENTRY`` replaces it with
    :func:`volatility_target_weights`, of which -- per the module docstring --
    only the value on each entry bar is ever executed.

    There is no ``vol-target`` alias. The flag has never shipped outside this
    branch, and carrying the old spelling forward would keep in circulation the
    exact word that made readers assume open positions were being retargeted.
    """

    FIXED = "fixed"
    VOL_SCALED_ENTRY = "vol-scaled-entry"


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
    """Per-bar size multiplier of ``target_annual_vol`` over realized volatility.

    The series is defined on every bar, but see the module docstring for which
    of its values the engine can actually execute: an entry bar's, and no
    other's.

    ``target / realized`` is the whole idea, and the whole danger: a flat
    stretch has zero realized volatility and ``target / 0`` is infinity, which
    would silently become an infinite position rather than an error anyone
    notices. The divide is guarded and the result clipped into
    ``[0, max_weight]``, so a degenerate estimate produces a capped weight
    instead of a fictional one.

    **Bars inside the estimator's own warmup get weight zero**, not the
    unconverged number ``ewm`` is happy to return for them. ``ewm().std()`` is
    finite after two observations, so the earlier version of this function
    handed back a full series in which the leading values were arbitrary --
    typically pinned to ``max_weight``, since a cold estimate of volatility is
    usually far too small. That made sizing depend on where the frame began: the
    same market segment was sized differently by two runs with different
    ``--start``. See :func:`vol_warmup_bars` for how long that takes.
    """
    if target_annual_vol <= 0:
        raise ValueError("target_annual_vol must be > 0")
    if max_weight <= 0:
        raise ValueError("max_weight must be > 0")

    realized = realized_volatility(returns, span=span, bars_per_year=bars_per_year)
    weights = target_annual_vol / realized.where(realized > 0)
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, max_weight)
    weights.iloc[: vol_warmup_bars(span)] = 0.0
    return weights


__all__ = [
    "DEFAULT_VOL_SPAN",
    "EWM_WARMUP_MULTIPLE",
    "SizeMode",
    "realized_volatility",
    "vol_warmup_bars",
    "volatility_target_weights",
]
