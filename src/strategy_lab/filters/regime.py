from __future__ import annotations

import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return true_range.ewm(span=period, adjust=False).mean()


def trend_filter(df: pd.DataFrame, *, span: int = 40) -> pd.Series:
    sma = df["close"].rolling(span).mean()
    return (df["close"] > sma).fillna(False)


def momentum_filter(df: pd.DataFrame, *, period: int = 26) -> pd.Series:
    roc = df["close"].pct_change(period)
    return (roc > 0).fillna(False)


def volatility_filter(
    df: pd.DataFrame,
    *,
    atr_period: int = 14,
    max_atr_ratio: float = 0.08,
) -> pd.Series:
    atr = compute_atr(df, period=atr_period)
    atr_ratio = atr / df["close"]
    return (atr_ratio < max_atr_ratio).fillna(False)


def regime_exit(df: pd.DataFrame, *, span: int = 40) -> pd.Series:
    sma = df["close"].rolling(span).mean()
    return (df["close"] < sma).fillna(False)


def momentum_divergence_exit(df: pd.DataFrame, *, period: int = 26) -> pd.Series:
    roc = df["close"].pct_change(period)
    return (roc < 0).fillna(False)
