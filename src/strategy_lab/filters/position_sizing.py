from __future__ import annotations

import pandas as pd

from strategy_lab.filters.regime import compute_atr


def atr_position_scale(
    df: pd.DataFrame,
    *,
    atr_period: int = 14,
    target_atr_ratio: float = 0.04,
    min_scale: float = 0.3,
    max_scale: float = 1.0,
) -> pd.Series:
    atr = compute_atr(df, period=atr_period)
    atr_ratio = atr / df["close"]

    scale = target_atr_ratio / atr_ratio.replace(0, 1e-10)
    scale = scale.clip(lower=min_scale, upper=max_scale)
    return scale.fillna(min_scale)


def compute_entry_shares(
    capital: float,
    price: pd.Series,
    entries: pd.Series,
    *,
    position_pct: float = 0.95,
    scale: pd.Series | None = None,
) -> pd.Series:
    fraction = position_pct
    if scale is not None:
        fraction = position_pct * scale

    shares = pd.Series(0.0, index=entries.index, dtype="float64")
    shares.loc[entries] = (capital * fraction.loc[entries]) / price.loc[entries]
    return shares
