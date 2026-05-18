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
    metadata: dict = field(default_factory=dict)


class Strategy(Protocol):
    name: str

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        ...


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
