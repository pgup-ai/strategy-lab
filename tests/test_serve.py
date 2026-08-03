from __future__ import annotations

import pandas as pd
import pytest

from strategy_lab.server import build_candles_payload, parse_identity


def _frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-03", periods=3, freq="7D", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0, 102.0, 104.0],
            "high": [103.0, 105.0, 107.0],
            "low": [99.0, 101.0, 103.0],
            "close": [102.0, 104.0, 106.0],
            "volume": [1_000.0, float("nan"), 1_200.0],
        },
        index=index,
    )


def test_build_candles_payload_shape() -> None:
    payload = build_candles_payload(_frame())

    assert set(payload) == {"bars"}
    assert len(payload["bars"]) == 3
    first = payload["bars"][0]
    assert first == {
        "time": int(pd.Timestamp("2024-01-03", tz="UTC").timestamp()),
        "open": 100.0,
        "high": 103.0,
        "low": 99.0,
        "close": 102.0,
        "volume": 1_000.0,
    }
    assert payload["bars"][1]["volume"] == 0.0


def test_build_candles_payload_empty_frame() -> None:
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).set_index(
        pd.DatetimeIndex([], name="timestamp")
    )

    assert build_candles_payload(empty) == {"bars": []}


def test_parse_identity_returns_identity_and_after() -> None:
    identity, after = parse_identity(
        {
            "exchange": ["yahoo"],
            "market_type": ["equity"],
            "symbol": ["SPY"],
            "timeframe": ["1w"],
            "after": ["1700000000"],
        }
    )

    assert identity.exchange == "yahoo"
    assert identity.symbol == "SPY"
    assert after == 1_700_000_000


def test_parse_identity_rejects_missing_parameter() -> None:
    with pytest.raises(ValueError, match="symbol"):
        parse_identity(
            {"exchange": ["yahoo"], "market_type": ["equity"], "timeframe": ["1w"]}
        )
