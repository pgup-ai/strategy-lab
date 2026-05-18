from __future__ import annotations

import pandas as pd

from strategy_lab.db.candles import _batched, normalize_candle_frame


def test_normalize_candle_frame_adds_identity_fields() -> None:
    df = pd.DataFrame(
        {
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [10.0],
        },
        index=pd.DatetimeIndex(["2024-01-01T00:00:00Z"], name="timestamp"),
    )

    records = normalize_candle_frame(
        df,
        exchange="binance",
        market_type="spot",
        symbol="BTC/USDT",
        timeframe="15m",
        source="binance",
    )

    assert records == [
        {
            "exchange": "binance",
            "market_type": "spot",
            "symbol": "BTC/USDT",
            "timeframe": "15m",
            "timestamp": pd.Timestamp("2024-01-01T00:00:00Z").to_pydatetime(),
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 10.0,
            "source": "binance",
        }
    ]


def test_batched_splits_large_upserts() -> None:
    rows = [{"index": index} for index in range(5)]

    assert list(_batched(rows, 2)) == [
        [{"index": 0}, {"index": 1}],
        [{"index": 2}, {"index": 3}],
        [{"index": 4}],
    ]
