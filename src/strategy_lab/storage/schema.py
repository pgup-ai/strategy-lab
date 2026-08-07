"""SQLAlchemy table objects for the run/signal/reason audit trail.

The DDL of record lives in :mod:`strategy_lab.storage.migrations` -- read that,
not this file, to know the real shape of the schema. These Table objects are for
building typed inserts and selects, and deliberately use their own ``MetaData``
so ``db.candles.init_db``'s ``create_all`` cannot emit these tables without the
CHECK constraints and append-only triggers the migrations attach. They mirror
only what a query needs: the ``signals`` and ``bar_reasons`` foreign keys to
``runs``, their ``*_mode_check`` constraints and the ``ix_*_run`` indexes exist in
the database but are not repeated here.
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
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

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
    # Arrives by ALTER rather than in the CREATE, so it sits last in the real
    # table; declared here beside `strength` because both are levels and nothing
    # in a named INSERT or SELECT depends on the order.
    Column("target_exposure", Numeric(10, 6)),
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

bar_reasons_table = Table(
    "bar_reasons",
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
    Column("state", Text, nullable=False),
    # Parallel arrays, not JSONB and not a column per feature: the feature set is
    # a property of the strategy, and NUMERIC elements are what keeps the
    # `Decimal(str(float(x)))` rule reaching the stored digits. `ARRAY(Numeric)`
    # is what carries the element type to the bind -- measured against Postgres
    # 16, an all-NULL array and a 17-significant-digit Decimal both round-trip
    # through it, and a bare float in the same list does not.
    #
    # `Numeric` with no precision or scale, unlike every price column here: a
    # scale is a count of decimal *places* and a float64 needs up to 17
    # *significant* digits, so 18 places round-trips a price near 100 and
    # truncates a feature near 0. See the migration for what that cost before it
    # was found.
    Column("feature_names", ARRAY(Text), nullable=False),
    Column("feature_values", ARRAY(Numeric), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    # No `side` here, unlike `uq_signals_identity`: a bar has one state, where a
    # reversal bar has two signals. The bar *is* the identity.
    UniqueConstraint(
        "run_id",
        "strategy_id",
        "strategy_version",
        "exchange",
        "symbol",
        "timeframe",
        "ts_bar_ms",
        name="uq_bar_reasons_identity",
    ),
    Index("ix_bar_reasons_lookup", "symbol", "timeframe", "ts_bar_ms"),
)
