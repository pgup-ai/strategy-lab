from __future__ import annotations

from sqlalchemy import text

from strategy_lab.db.candles import get_engine

PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

# The USING clause is load-bearing, not decoration. Postgres' implicit
# float8 -> numeric cast formats via "%.15g" (DBL_DIG), which silently drops the
# 16th-17th significant digits that a float64 needs to round-trip. Going through
# text instead uses the shortest round-trip representation, so every stored
# float64 survives the migration exactly. Measured on this repo's data: the bare
# cast altered ~14.7k of 20.9k equity rows; the text cast alters none.
# Every statement must be safe to run repeatedly.
MIGRATIONS: tuple[str, ...] = (
    *(
        f"ALTER TABLE market_candles ALTER COLUMN {column} TYPE NUMERIC(38,18) "
        f"USING {column}::text::numeric"
        for column in PRICE_COLUMNS
    ),
    "ALTER TABLE market_candles ADD COLUMN IF NOT EXISTS ts_close_ms  BIGINT",
    "ALTER TABLE market_candles ADD COLUMN IF NOT EXISTS quote_volume NUMERIC(38,18)",
    "ALTER TABLE market_candles ADD COLUMN IF NOT EXISTS trades       INTEGER",
    "ALTER TABLE market_candles ADD COLUMN IF NOT EXISTS is_closed    BOOLEAN NOT NULL DEFAULT TRUE",
    "ALTER TABLE market_candles ADD COLUMN IF NOT EXISTS ingested_via TEXT",
)


def run_migrations(database_url: str | None = None) -> int:
    """Apply idempotent schema upgrades. Returns the number of statements executed."""
    engine = get_engine(database_url)
    with engine.begin() as conn:
        for statement in MIGRATIONS:
            conn.execute(text(statement))
    return len(MIGRATIONS)
