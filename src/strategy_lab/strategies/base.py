from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalSet:
    long_entries: pd.Series
    long_exits: pd.Series
    short_entries: pd.Series
    short_exits: pd.Series
    setup_stop_loss: pd.Series | None = None
    trend_failure_long_exits: pd.Series | None = None
    trend_failure_short_exits: pd.Series | None = None
    position_size: pd.Series | None = None
    metadata: dict = field(default_factory=dict)


class Strategy(Protocol):
    name: str
    version: str
    warmup_bars: int

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        ...


def require_positive_span(strategy: str, field: str, value: object) -> None:
    """Reject a lookback pandas would turn into lookahead or into silence.

    ``pct_change(-1)`` compares each row against the *next* one, so a negative
    lookback makes a strategy non-causal -- the exact failure this repo exists to
    prevent, and one ``tests/test_lookahead.py`` cannot catch, because the poison
    probe iterates ``list_strategies()`` and so only ever sees defaults, while
    ``sweep_parameters`` rebuilds every cell with ``dataclasses.replace`` over any
    field. Zero is the quieter half: ``pct_change(0)`` and ``rolling(0)`` raise
    nothing and yield a strategy that simply never trades, which reads on a
    surface as a parameter region with no edge rather than as a broken cell.
    """
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{strategy} {field} must be a positive integer, got {value!r}")


def validate_ohlcv(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Cannot generate signals for an empty candle frame")


def setup_invalidation_stop_loss(
    df: pd.DataFrame,
    *,
    long_entries: pd.Series,
    short_entries: pd.Series,
    setup_window: int = 3,
) -> pd.Series:
    validate_ohlcv(df)

    entry_price = df["close"]
    setup_low = df["low"].rolling(setup_window).min()
    setup_high = df["high"].rolling(setup_window).max()

    stop_loss = pd.Series(np.nan, index=df.index, dtype="float64")
    long_stop = (entry_price - setup_low) / entry_price
    short_stop = (setup_high - entry_price) / entry_price

    stop_loss.loc[long_entries] = long_stop.loc[long_entries]
    stop_loss.loc[short_entries] = short_stop.loc[short_entries]
    return stop_loss.where(stop_loss > 0)


def ema_trend_failure_exits(
    df: pd.DataFrame,
    *,
    ema_span: int,
) -> tuple[pd.Series, pd.Series]:
    validate_ohlcv(df)

    ema_trend = df["close"].ewm(span=ema_span, adjust=False).mean()
    long_exits = df["close"] < ema_trend
    short_exits = df["close"] > ema_trend
    return long_exits.fillna(False), short_exits.fillna(False)
