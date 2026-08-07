from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Index,
    MetaData,
    Numeric,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import insert

from strategy_lab.config import settings


OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")

metadata = MetaData()

candles_table = Table(
    "market_candles",
    metadata,
    Column("exchange", String(40), nullable=False),
    Column("market_type", String(40), nullable=False),
    Column("symbol", String(64), nullable=False),
    Column("timeframe", String(16), nullable=False),
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("open", Numeric(38, 18), nullable=False),
    Column("high", Numeric(38, 18), nullable=False),
    Column("low", Numeric(38, 18), nullable=False),
    Column("close", Numeric(38, 18), nullable=False),
    Column("volume", Numeric(38, 18), nullable=False),
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
    required = set(OHLCV_COLUMNS)
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
    # Last wins because a re-fetch's redelivered copy is the corrected one, matching
    # ``ReplayFeed._ordered``. ``kind="stable"`` is what makes "last" mean it:
    # sort_index() defaults to quicksort, which reorders equal keys above ~16 rows,
    # so without it an arbitrary duplicate survives and the correction is dropped.
    normalized = normalized.sort_index(kind="stable")
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
                "open": _to_numeric(row["open"]),
                "high": _to_numeric(row["high"]),
                "low": _to_numeric(row["low"]),
                "close": _to_numeric(row["close"]),
                "volume": _to_numeric(row["volume"]),
                "source": source,
            }
        )
    return records


def _to_numeric(value) -> Decimal:
    """Bind a price as NUMERIC, not as float8.

    SQLAlchemy's ``Numeric`` has no bind processor on a dialect with native
    decimal support, so a Python ``float`` reaches psycopg as a float8 parameter
    and Postgres applies its implicit ``float8 -> numeric`` cast on the way in --
    the *same* "%.15g" cast ``storage/migrations.py`` contorts itself to avoid,
    dropping the 16th-17th significant digits a float64 needs to round-trip.
    Measured against the stored data: 58,996 of 519,205 values (11.4%, and 73.7%
    of AAPL 1h) came back changed, and because ``upsert_candles`` is ON CONFLICT
    DO UPDATE over deliberately overlapping fetch windows, every re-fetch rewrote
    correct rows with degraded ones.

    ``float()`` first, then ``str()``: ``str`` on a float is the shortest
    representation that round-trips, which is what makes the migration's
    ``::text::numeric`` correct. ``Decimal(value)`` directly would expand to the
    binary value's full exact form (``Decimal(0.1)`` is 55 digits) and lose the
    tail to the column's scale of 18. ``float()`` also keeps ``None`` raising
    ``TypeError`` rather than being written as a plausible-looking number.
    """
    return Decimal(str(float(value)))


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
    # Storage is NUMERIC (Decimal) and every strategy and indicator does float64
    # pandas math, so the astype below is the one documented Decimal -> float64
    # boundary. coerce_float=False keeps pandas from quietly doing that conversion
    # first, as an implicit default we would then be trusting by accident.
    df = pd.read_sql(query, engine, coerce_float=False)
    if df.empty:
        # Same shape as the populated path -- float64 columns on a UTC index. A
        # bare pd.DataFrame(columns=...) hands back object dtype and a tz-naive
        # index, so an empty range would silently poison a concat or an indicator
        # that the same code handles fine when rows exist.
        return pd.DataFrame(
            {name: pd.Series(dtype="float64") for name in OHLCV_COLUMNS},
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for column in OHLCV_COLUMNS:
        df[column] = df[column].astype("float64")
    return df.set_index("timestamp")


def list_candle_sets(database_url: str | None = None) -> pd.DataFrame:
    """Every stored candle set: how far its bars reach, and when they were written.

    ``last_written`` is ``max(updated_at)``, maintained by ``upsert_candles`` on
    every conflicting row, and it is a different question from
    ``last_timestamp``. On a venue whose history only grows the two move
    together and the write time says nothing. On a **dividend-adjusted** equity
    series they come apart: the Yahoo fetcher rescales every OHLC column by
    ``adj_close / close``, so a distribution rewrites bars back to the start of
    the history rather than appending one. Measured against a fresh fetch of the
    stored SPY weekly series, **333 of 333 overlapping bars moved** (median
    0.257%, largest 0.405%, oldest 2020-01-01), and ``donchian`` -- which
    compares a close against a channel's own high and low, where a non-uniform
    rescale does not cancel -- differed on 3 of them.

    It is one more aggregate over a group-by that already runs, rather than a
    query per set, because the board asks this of every stored dataset at once.
    """
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
            func.max(candles_table.c.updated_at).label("last_written"),
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
