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


# pg_trigger.tgtype bitmask: ROW=1, BEFORE=2, INSERT=4, DELETE=8, UPDATE=16,
# TRUNCATE=32. The append-only trigger is ROW|BEFORE|DELETE|UPDATE, and must stay
# clear of INSERT -- appending a signal is the one mutation that has to work.
APPEND_ONLY_TGTYPE = 1 | 2 | 8 | 16  # == 27


def _append_only_trigger_migration() -> str:
    """Install the append-only trigger, but only when it isn't already correct.

    The obvious spelling -- ``DROP TRIGGER IF EXISTS`` followed by an
    unconditional ``CREATE TRIGGER`` -- is safe but not a no-op: measured here,
    the trigger's pg_trigger.oid changes on every re-run, and the pair takes an
    ACCESS EXCLUSIVE lock on ``signals`` that Postgres holds until the migration
    transaction commits. Every later statement in that transaction therefore
    runs with all readers and writers of ``signals`` blocked -- the same trap
    the market_candles NUMERIC conversion is guarded against, and worse here
    because a live session is writing signals exactly when someone runs
    ``migrate``.

    The guard compares the function and the event mask as well as the name, so a
    trigger that is missing, points at the wrong function, or fires on the wrong
    events is still repaired; only an already-correct trigger is left alone.
    """
    return f"""
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_trigger
    WHERE tgrelid = 'signals'::regclass
      AND tgname = 'trg_signals_append_only'
      AND NOT tgisinternal
      AND tgtype = {APPEND_ONLY_TGTYPE}
      AND tgfoid = 'signals_reject_mutation'::regproc
  ) THEN
    DROP TRIGGER IF EXISTS trg_signals_append_only ON signals;
    CREATE TRIGGER trg_signals_append_only
      BEFORE UPDATE OR DELETE ON signals
      FOR EACH ROW EXECUTE FUNCTION signals_reject_mutation();
  END IF;
END $$
""".strip()


# Signals are a permanent audit trail: one table for both historical replays and
# live sessions, separated only by `mode` and `run_id`, so a replay can be diffed
# against a live session by joining on (symbol, timeframe, ts_bar_ms, side).
#
# `side` is part of uq_signals_identity on purpose. Strategies like turnaround_v1
# wire long_exits = short_entries, so one bar legitimately emits both `exit_long`
# and `enter_short` -- two distinct signals that must both persist.
SIGNAL_MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS runs (
      run_id           UUID PRIMARY KEY,
      mode             TEXT NOT NULL CHECK (mode IN ('backtest','replay','paper','live')),
      strategy_id      TEXT NOT NULL,
      strategy_version TEXT NOT NULL,
      config           JSONB NOT NULL DEFAULT '{}',
      started_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
      finished_at      TIMESTAMPTZ,
      warmup_until_ts_ms BIGINT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS signals (
      id               BIGSERIAL PRIMARY KEY,
      run_id           UUID NOT NULL REFERENCES runs(run_id),
      mode             TEXT NOT NULL CHECK (mode IN ('backtest','replay','paper','live')),
      strategy_id      TEXT NOT NULL,
      strategy_version TEXT NOT NULL,
      exchange         TEXT NOT NULL,
      market_type      TEXT NOT NULL,
      symbol           TEXT NOT NULL,
      timeframe        TEXT NOT NULL,
      ts_bar_ms        BIGINT NOT NULL,
      ts_emit_ms       BIGINT NOT NULL,
      bar_is_closed    BOOLEAN NOT NULL,
      side             TEXT NOT NULL CONSTRAINT signals_side_check
                       CHECK (side IN ('enter_long','exit_long','enter_short','exit_short')),
      strength         NUMERIC(10,6),
      entry_price      NUMERIC(38,18),
      stop_loss        NUMERIC(38,18),
      take_profit      NUMERIC(38,18),
      reason           TEXT NOT NULL,
      features         JSONB NOT NULL DEFAULT '{}',
      created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
      CONSTRAINT uq_signals_identity UNIQUE
        (run_id, strategy_id, strategy_version, exchange, symbol, timeframe, ts_bar_ms, side)
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_signals_lookup ON signals (symbol, timeframe, ts_bar_ms)",
    "CREATE INDEX IF NOT EXISTS ix_signals_run ON signals (run_id, ts_bar_ms)",
    """
    CREATE OR REPLACE FUNCTION signals_reject_mutation() RETURNS trigger AS $$
    BEGIN
      RAISE EXCEPTION 'signals is append-only; % is not permitted', TG_OP;
    END;
    $$ LANGUAGE plpgsql
    """,
    _append_only_trigger_migration(),
)


def run_migrations(database_url: str | None = None) -> int:
    """Apply idempotent schema upgrades. Returns the number of statements executed."""
    statements = MIGRATIONS + SIGNAL_MIGRATIONS
    engine = get_engine(database_url)
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
    return len(statements)
