from __future__ import annotations

import pandas as pd


FREQUENCY_BY_TIMEFRAME = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1d",
    "1w": "7d",
}


def timeframe_to_pandas_freq(timeframe: str) -> str:
    return FREQUENCY_BY_TIMEFRAME.get(timeframe, timeframe)


def timeframe_to_millis(timeframe: str) -> int:
    offset = pd.tseries.frequencies.to_offset(timeframe_to_pandas_freq(timeframe))
    return int(pd.Timedelta(offset).total_seconds() * 1000)
