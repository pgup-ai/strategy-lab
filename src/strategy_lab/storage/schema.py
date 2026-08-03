"""SQLAlchemy table objects for the run/signal audit trail.

The DDL of record lives in :mod:`strategy_lab.storage.migrations` -- these Table
objects exist so application code can build typed inserts and selects, not to
create the schema. They deliberately use their own ``MetaData`` so that
``db.candles.init_db``'s ``create_all`` cannot emit these tables without the
CHECK constraints and the append-only triggers that the migrations attach.

Because these objects are for query building rather than DDL, they deliberately
mirror only what a query needs: the ``signals`` -> ``runs`` foreign key and the
``ix_signals_run`` index exist in the database but are not repeated here. Read
migrations.py, not this file, to know the real shape of the schema.
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Index,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

metadata = MetaData()

runs_table = Table(
    "runs",
    metadata,
    Column("run_id", UUID(as_uuid=True), primary_key=True),
    Column("mode", Text, nullable=False),
    Column("strategy_id", Text, nullable=False),
    Column("strategy_version", Text, nullable=False),
    Column("config", JSONB, nullable=False, server_default="{}"),
    Column("started_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("finished_at", DateTime(timezone=True)),
    Column("warmup_until_ts_ms", BigInteger),
)

signals_table = Table(
    "signals",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("run_id", UUID(as_uuid=True), nullable=False),
    Column("mode", Text, nullable=False),
    Column("strategy_id", Text, nullable=False),
    Column("strategy_version", Text, nullable=False),
    Column("exchange", Text, nullable=False),
    Column("market_type", Text, nullable=False),
    Column("symbol", Text, nullable=False),
    Column("timeframe", String(16), nullable=False),
    Column("ts_bar_ms", BigInteger, nullable=False),
    Column("ts_emit_ms", BigInteger, nullable=False),
    Column("bar_is_closed", Boolean, nullable=False),
    Column("side", Text, nullable=False),
    Column("strength", Numeric(10, 6)),
    Column("entry_price", Numeric(38, 18)),
    Column("stop_loss", Numeric(38, 18)),
    Column("take_profit", Numeric(38, 18)),
    Column("reason", Text, nullable=False),
    Column("features", JSONB, nullable=False, server_default="{}"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint(
        "run_id",
        "strategy_id",
        "strategy_version",
        "exchange",
        "symbol",
        "timeframe",
        "ts_bar_ms",
        "side",
        name="uq_signals_identity",
    ),
    Index("ix_signals_lookup", "symbol", "timeframe", "ts_bar_ms"),
)
