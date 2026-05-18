from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import insert

from strategy_lab.config import settings


metadata = MetaData()

candles_table = Table(
    "market_candles",
    metadata,
    Column("exchange", String(40), nullable=False),
    Column("market_type", String(40), nullable=False),
    Column("symbol", String(64), nullable=False),
    Column("timeframe", String(16), nullable=False),
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("open", Float, nullable=False),
    Column("high", Float, nullable=False),
    Column("low", Float, nullable=False),
    Column("close", Float, nullable=False),
    Column("volume", Float, nullable=False),
    Column("source", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint(
        "exchange",
        "market_type",
        "symbol",
        "timeframe",
        "timestamp",
        name="uq_market_candles_identity",
    ),
    Index("ix_market_candles_lookup", "exchange", "market_type", "symbol", "timeframe", "timestamp"),
)


def get_engine(database_url: str | None = None):
    return create_engine(database_url or settings.database_url, future=True)


def init_db(database_url: str | None = None) -> None:
    engine = get_engine(database_url)
    metadata.create_all(engine)


def normalize_candle_frame(
    df: pd.DataFrame,
    *,
    exchange: str,
    market_type: str,
    symbol: str,
    timeframe: str,
    source: str,
) -> list[dict]:
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")

    normalized = df.copy()
    if not isinstance(normalized.index, pd.DatetimeIndex):
        if "timestamp" not in normalized.columns:
            raise ValueError("Expected a DatetimeIndex or a timestamp column")
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
        normalized = normalized.set_index("timestamp")

    normalized.index = pd.to_datetime(normalized.index, utc=True)
    normalized = normalized.sort_index()
    normalized = normalized.loc[~normalized.index.duplicated(keep="last")]

    records: list[dict] = []
    for timestamp, row in normalized.iterrows():
        records.append(
            {
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": timestamp.to_pydatetime(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "source": source,
            }
        )
    return records


def upsert_candles(
    records: Iterable[dict],
    database_url: str | None = None,
    *,
    batch_size: int = 1_000,
) -> int:
    rows = list(records)
    if not rows:
        return 0

    engine = get_engine(database_url)
    with engine.begin() as conn:
        for batch in _batched(rows, batch_size):
            stmt = insert(candles_table).values(batch)
            update_columns = {
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "volume": stmt.excluded.volume,
                "source": stmt.excluded.source,
                "updated_at": func.now(),
            }
            stmt = stmt.on_conflict_do_update(
                constraint="uq_market_candles_identity",
                set_=update_columns,
            )
            conn.execute(stmt)
    return len(rows)


def _batched(rows: list[dict], batch_size: int) -> Iterable[list[dict]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for index in range(0, len(rows), batch_size):
        yield rows[index : index + batch_size]


def load_candles(
    *,
    exchange: str,
    market_type: str,
    symbol: str,
    timeframe: str,
    start: str | None = None,
    end: str | None = None,
    database_url: str | None = None,
) -> pd.DataFrame:
    engine = get_engine(database_url)
    query = select(
        candles_table.c.timestamp,
        candles_table.c.open,
        candles_table.c.high,
        candles_table.c.low,
        candles_table.c.close,
        candles_table.c.volume,
    ).where(
        candles_table.c.exchange == exchange,
        candles_table.c.market_type == market_type,
        candles_table.c.symbol == symbol,
        candles_table.c.timeframe == timeframe,
    )

    if start:
        query = query.where(candles_table.c.timestamp >= _utc_timestamp(start).to_pydatetime())
    if end:
        query = query.where(candles_table.c.timestamp <= _utc_timestamp(end).to_pydatetime())

    query = query.order_by(candles_table.c.timestamp)
    df = pd.read_sql(query, engine)
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).set_index(
            pd.DatetimeIndex([], name="timestamp")
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")


def list_candle_sets(database_url: str | None = None) -> pd.DataFrame:
    engine = get_engine(database_url)
    query = (
        select(
            candles_table.c.exchange,
            candles_table.c.market_type,
            candles_table.c.symbol,
            candles_table.c.timeframe,
            func.count().label("candles"),
            func.min(candles_table.c.timestamp).label("first_timestamp"),
            func.max(candles_table.c.timestamp).label("last_timestamp"),
        )
        .group_by(
            candles_table.c.exchange,
            candles_table.c.market_type,
            candles_table.c.symbol,
            candles_table.c.timeframe,
        )
        .order_by(
            candles_table.c.exchange,
            candles_table.c.market_type,
            candles_table.c.symbol,
            candles_table.c.timeframe,
        )
    )
    return pd.read_sql(query, engine)


def _utc_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")
