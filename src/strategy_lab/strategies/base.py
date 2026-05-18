from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class SignalSet:
    long_entries: pd.Series
    long_exits: pd.Series
    short_entries: pd.Series
    short_exits: pd.Series
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
