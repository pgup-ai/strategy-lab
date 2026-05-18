from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class YahooFinanceClient:
    source: str = "yahoo"

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        *,
        period: str = "2y",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        import yfinance as yf

        df = yf.download(
            symbol,
            period=None if start else period,
            start=start,
            end=end,
            interval=timeframe,
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).set_index(
                pd.DatetimeIndex([], name="timestamp")
            )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns=str.lower)
        df = df.rename(columns={"adj close": "adj_close"})
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "timestamp"

        required = ["open", "high", "low", "close", "volume"]
        return df[required].dropna().sort_index()
