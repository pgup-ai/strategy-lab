from __future__ import annotations

from sqlalchemy import text

from strategy_lab.db.candles import get_engine

PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
PRICE_PRECISION = 38
PRICE_SCALE = 18


def _price_column_migration(column: str) -> str:
    """Widen one price column to NUMERIC, but only if it isn't already.

    Two things are load-bearing here.

    The ``USING`` clause: Postgres' implicit float8 -> numeric cast formats via
    "%.15g" (DBL_DIG), which silently drops the 16th-17th significant digits a
    float64 needs to round-trip. Going through text uses the shortest
    round-trip representation instead, so every stored float64 survives
    exactly. Measured on this repo's data: the bare cast altered ~14.7k of
    20.9k equity rows; the text cast alters none.

    The surrounding guard: ``USING`` forces a full table rewrite under an
    ACCESS EXCLUSIVE lock, so firing it unconditionally would make every
    ``migrate`` rewrite the whole table. That is invisible at 100k rows and a
    multi-second stall once live 1m candles arrive. The guard makes a
    re-run genuinely a no-op rather than merely a harmless one.
    """
    return f"""
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = current_schema()
      AND table_name = 'market_candles'
      AND column_name = '{column}'
      AND data_type = 'numeric'
      AND numeric_precision = {PRICE_PRECISION}
      AND numeric_scale = {PRICE_SCALE}
  ) THEN
    ALTER TABLE market_candles ALTER COLUMN {column} TYPE NUMERIC({PRICE_PRECISION},{PRICE_SCALE})
      USING {column}::text::numeric;
  END IF;
END $$
""".strip()


# Every statement must be safe to run repeatedly.
MIGRATIONS: tuple[str, ...] = (
    *(_price_column_migration(column) for column in PRICE_COLUMNS),
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
