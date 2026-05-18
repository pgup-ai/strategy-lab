from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from strategy_lab.timeframes import timeframe_to_millis


@dataclass
class CryptoOhlcvClient:
    exchange_id: str = "binance"
    market_type: str = "spot"

    def __post_init__(self) -> None:
        ccxt = _import_ccxt()

        exchange_id = self.exchange_id
        if self.exchange_id == "binance" and self.market_type in {"perp", "future", "futures"}:
            exchange_id = "binanceusdm"

        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({"enableRateLimit": True})
        self.exchange_id = exchange_id

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        *,
        since: str | None = None,
        until: str | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        since_ms = _timestamp_ms(since) if since else None
        until_ms = _timestamp_ms(until) if until else None
        frame_ms = timeframe_to_millis(timeframe)

        rows: list[list] = []
        next_since = since_ms

        while True:
            batch = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=next_since,
                limit=limit,
            )
            if not batch:
                break

            for row in batch:
                if until_ms is not None and row[0] > until_ms:
                    continue
                rows.append(row)

            last_ts = int(batch[-1][0])
            if since_ms is None or len(batch) < limit:
                break
            if until_ms is not None and last_ts >= until_ms:
                break

            next_since = last_ts + frame_ms

        return _ohlcv_to_frame(rows)


def _timestamp_ms(value: str) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp() * 1000)


def _ohlcv_to_frame(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).set_index(
            pd.DatetimeIndex([], name="timestamp")
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    return df.set_index("timestamp")


def _import_ccxt():
    repo_root = Path(__file__).resolve().parents[3]
    existing_ceiling = os.environ.get("GIT_CEILING_DIRECTORIES")
    os.environ["GIT_CEILING_DIRECTORIES"] = (
        f"{repo_root}{os.pathsep}{existing_ceiling}" if existing_ceiling else str(repo_root)
    )
    try:
        import ccxt
    finally:
        if existing_ceiling is None:
            os.environ.pop("GIT_CEILING_DIRECTORIES", None)
        else:
            os.environ["GIT_CEILING_DIRECTORIES"] = existing_ceiling

    return ccxt
