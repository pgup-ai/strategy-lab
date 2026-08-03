"""Storage for perp funding rates and open-interest snapshots.

Funding is a cash flow settled on the venue's own schedule, and open interest is
a point-in-time snapshot of contracts outstanding. Neither is a property of a
candle, so both get their own table rather than a column on ``market_candles``.
The DDL of record lives in :mod:`strategy_lab.storage.migrations`; the Table
objects here use their own ``MetaData`` so ``db.candles.init_db``'s
``create_all`` cannot emit them without the constraints the migrations attach.

Read the frames back the same way :func:`strategy_lab.db.load_candles` does: a
UTC ``DatetimeIndex`` named ``timestamp`` and float64 value columns, so the
Decimal -> float64 conversion happens in exactly one visible place.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from decimal import Decimal

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Index,
    MetaData,
    Numeric,
    Table,
    Text,
    UniqueConstraint,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import insert

from strategy_lab.db.candles import get_engine

metadata = MetaData()

funding_table = Table(
    "funding_rates",
    metadata,
    Column("exchange", Text, nullable=False),
    Column("market_type", Text, nullable=False),
    Column("symbol", Text, nullable=False),
    Column("funding_time_ms", BigInteger, nullable=False),
    Column("funding_rate", Numeric(38, 18), nullable=False),
    Column("mark_price", Numeric(38, 18)),
    Column("source", Text),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint(
        "exchange", "market_type", "symbol", "funding_time_ms", name="uq_funding_identity"
    ),
    Index("ix_funding_rates_lookup", "symbol", "funding_time_ms"),
)

open_interest_table = Table(
    "open_interest",
    metadata,
    Column("exchange", Text, nullable=False),
    Column("market_type", Text, nullable=False),
    Column("symbol", Text, nullable=False),
    Column("ts_ms", BigInteger, nullable=False),
    Column("open_interest", Numeric(38, 18), nullable=False),
    Column("open_interest_usd", Numeric(38, 18)),
    Column("source", Text),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint(
        "exchange", "market_type", "symbol", "ts_ms", name="uq_open_interest_identity"
    ),
    Index("ix_open_interest_lookup", "symbol", "ts_ms"),
)

# Postgres caps a statement at 65535 bound parameters and a multi-row INSERT
# binds one per column per row, so an unchunked backfill raises OperationalError
# rather than merely running slowly. Deriving the chunk from the column count
# keeps the bound correct if a column is ever added.
MAX_BOUND_PARAMETERS = 65535
MAX_FUNDING_ROWS_PER_INSERT = MAX_BOUND_PARAMETERS // len(funding_table.c)
MAX_OPEN_INTEREST_ROWS_PER_INSERT = MAX_BOUND_PARAMETERS // len(open_interest_table.c)

_FUNDING_VALUE_COLUMNS = ("funding_rate", "mark_price")
_OPEN_INTEREST_VALUE_COLUMNS = ("open_interest", "open_interest_usd")


def _to_numeric(value) -> Decimal | None:
    """Coerce a value for a ``NUMERIC`` bind without ever passing a bare float.

    SQLAlchemy's ``Numeric`` has no bind processor on a dialect with native
    decimal support, so a Python ``float`` reaches psycopg as a float8 parameter
    and Postgres applies its implicit ``float8 -> numeric`` cast -- the "%.15g"
    cast that silently drops the 16th-17th significant digits. Measured on this
    database: 88.02116722596503 stores as 88.021167225965.

    Unlike ``normalize_candle_frame``, which receives float64 out of pandas and
    must go ``Decimal(str(float(x)))``, these values arrive from the exchange as
    exact decimal strings. A ``Decimal`` is therefore bound unchanged -- routing
    it through float64 first would throw away digits the column can hold, for a
    conversion the data never needed.
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    return Decimal(str(float(value)))


def upsert_funding(
    rows: Iterable[dict],
    database_url: str | None = None,
) -> int:
    """Insert funding rows, updating any already stored. Returns the count offered.

    Last write wins, matching ``upsert_candles``: a redelivered row from an
    overlapping re-fetch is the corrected one, not a duplicate to ignore.
    """
    return _upsert(
        rows,
        table=funding_table,
        value_columns=_FUNDING_VALUE_COLUMNS,
        constraint="uq_funding_identity",
        chunk=MAX_FUNDING_ROWS_PER_INSERT,
        database_url=database_url,
    )


def upsert_open_interest(
    rows: Iterable[dict],
    database_url: str | None = None,
) -> int:
    """Insert open-interest snapshots, updating any already stored.

    Binance serves only ~30 days of OI history, so this table is filled by
    polling forward over deliberately overlapping windows; every poll re-offers
    rows the previous one stored, which is why this is an upsert.
    """
    return _upsert(
        rows,
        table=open_interest_table,
        value_columns=_OPEN_INTEREST_VALUE_COLUMNS,
        constraint="uq_open_interest_identity",
        chunk=MAX_OPEN_INTEREST_ROWS_PER_INSERT,
        database_url=database_url,
    )


def _upsert(
    rows: Iterable[dict],
    *,
    table: Table,
    value_columns: Sequence[str],
    constraint: str,
    chunk: int,
    database_url: str | None,
) -> int:
    prepared = [_prepare(row, value_columns) for row in rows]
    if not prepared:
        return 0

    with get_engine(database_url).begin() as conn:
        for start in range(0, len(prepared), chunk):
            statement = insert(table).values(prepared[start : start + chunk])
            statement = statement.on_conflict_do_update(
                constraint=constraint,
                set_={
                    name: getattr(statement.excluded, name)
                    for name in (*value_columns, "source")
                },
            )
            conn.execute(statement)
    return len(prepared)


def _prepare(row: dict, value_columns: Sequence[str]) -> dict:
    prepared = dict(row)
    prepared.setdefault("source", None)
    for name in value_columns:
        prepared[name] = _to_numeric(prepared.get(name))
    return prepared


def load_funding(
    *,
    exchange: str,
    market_type: str,
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    database_url: str | None = None,
) -> pd.DataFrame:
    """Load funding as a time series indexed by settlement time.

    ``funding_time_ms`` is kept as a column as well as an index, because
    applying funding correctly means landing it on the bar containing the exact
    settlement instant rather than resampling to a schedule.
    """
    return _load(
        table=funding_table,
        time_column="funding_time_ms",
        value_columns=_FUNDING_VALUE_COLUMNS,
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        start=start,
        end=end,
        database_url=database_url,
    )


def load_open_interest(
    *,
    exchange: str,
    market_type: str,
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    database_url: str | None = None,
) -> pd.DataFrame:
    """Load open-interest snapshots indexed by observation time."""
    return _load(
        table=open_interest_table,
        time_column="ts_ms",
        value_columns=_OPEN_INTEREST_VALUE_COLUMNS,
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        start=start,
        end=end,
        database_url=database_url,
    )


def _load(
    *,
    table: Table,
    time_column: str,
    value_columns: Sequence[str],
    exchange: str,
    market_type: str,
    symbol: str,
    start: str | None,
    end: str | None,
    database_url: str | None,
) -> pd.DataFrame:
    time_col = table.c[time_column]
    query = select(time_col, *(table.c[name] for name in value_columns)).where(
        table.c.exchange == exchange,
        table.c.market_type == market_type,
        table.c.symbol == symbol,
    )
    if start:
        query = query.where(time_col >= _timestamp_ms(start))
    if end:
        query = query.where(time_col <= _timestamp_ms(end))

    # coerce_float=False keeps pandas from quietly turning NUMERIC into float,
    # so the astype below is the one deliberate Decimal -> float64 boundary --
    # the same contract as ``load_candles``.
    df = pd.read_sql(query.order_by(time_col), get_engine(database_url), coerce_float=False)
    if df.empty:
        return _empty_frame(time_column, value_columns)

    index = pd.to_datetime(df[time_column], unit="ms", utc=True).rename("timestamp")
    df[time_column] = df[time_column].astype("int64")
    for name in value_columns:
        df[name] = df[name].astype("float64")
    return df.set_index(index)


def _empty_frame(time_column: str, value_columns: Sequence[str]) -> pd.DataFrame:
    """Same shape as the populated path, for the same reason ``load_candles`` has one.

    A bare ``DataFrame(columns=...)`` hands back object dtype and a tz-naive
    index, so an empty window would silently poison a concat or an indicator
    that the identical code handles fine when rows exist.
    """
    columns = {time_column: pd.Series(dtype="int64")}
    columns.update({name: pd.Series(dtype="float64") for name in value_columns})
    return pd.DataFrame(columns, index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))


def _timestamp_ms(value: str) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return int(timestamp.tz_convert("UTC").timestamp() * 1000)


__all__ = [
    "MAX_BOUND_PARAMETERS",
    "MAX_FUNDING_ROWS_PER_INSERT",
    "MAX_OPEN_INTEREST_ROWS_PER_INSERT",
    "funding_table",
    "load_funding",
    "load_open_interest",
    "open_interest_table",
    "upsert_funding",
    "upsert_open_interest",
]
