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
    "1wk": "7d",
}


def timeframe_to_pandas_freq(timeframe: str) -> str:
    return FREQUENCY_BY_TIMEFRAME.get(timeframe, timeframe)


def timeframe_to_millis(timeframe: str) -> int:
    offset = pd.tseries.frequencies.to_offset(timeframe_to_pandas_freq(timeframe))
    return int(pd.Timedelta(offset).total_seconds() * 1000)


_YEAR_MS = 365.25 * 24 * 60 * 60 * 1000


def timeframe_to_bars_per_year(timeframe: str) -> float:
    """Bars of ``timeframe`` in a calendar year, used to annualize per-bar volatility.

    Derived from the timeframe string rather than from the frame, so the number
    is a constant known before the first bar: measuring it as ``len(df) / span``
    would let the end of the sample set the size of the first trade.

    It is calendar time, so it is exact for a 24/7 market and an overstatement
    for one that closes -- 365.25 daily bars per year against an equity
    calendar's ~252. That overstates annualized volatility by ~20% there, which
    under-sizes rather than over-sizes, and it is a fixed bias rather than a
    drift.
    """
    return _YEAR_MS / timeframe_to_millis(timeframe)
