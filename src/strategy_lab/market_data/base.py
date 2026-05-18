from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class MarketDataIdentity:
    exchange: str
    market_type: str
    symbol: str
    timeframe: str


class OhlcvClient(Protocol):
    def fetch_ohlcv(self, symbol: str, timeframe: str, **kwargs) -> pd.DataFrame:
        ...
