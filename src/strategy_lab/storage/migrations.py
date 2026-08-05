from __future__ import annotations

from sqlalchemy import text

from strategy_lab.db.candles import get_engine

PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
PRICE_PRECISION = 38
PRICE_SCALE = 18


def _price_column_migration(column: str) -> str:
    """Widen one price column to NUMERIC, but only if it isn't already.

    ``USING <col>::text::numeric``: Postgres' implicit float8 -> numeric cast
    formats via "%.15g" (DBL_DIG), silently dropping the 16th-17th significant
    digits a float64 needs to round-trip. Going through text uses the shortest
    round-tripping representation instead. Measured on this repo's data, the
    bare cast altered ~14.7k of 20.9k equity rows; the text cast alters none.

    The surrounding guard: ``USING`` forces a full table rewrite under an ACCESS
    EXCLUSIVE lock, so firing it unconditionally would make every ``migrate``
    rewrite the whole table -- invisible at 100k rows, a multi-second stall once
    live 1m candles arrive.
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
# TRUNCATE=32. Both guards must stay clear of INSERT -- appending a signal is the
# one mutation that has to work.
APPEND_ONLY_TGTYPE = 1 | 2 | 8 | 16  # ROW|BEFORE|DELETE|UPDATE == 27
NO_TRUNCATE_TGTYPE = 2 | 32  # BEFORE|TRUNCATE, statement-level (no ROW bit) == 34


def _guarded_trigger_migration(name: str, tgtype: int, events: str, level: str) -> str:
    """Install one trigger on ``signals``, but only when it isn't already correct.

    ``DROP TRIGGER IF EXISTS`` plus an unconditional ``CREATE TRIGGER`` is safe
    but not a no-op: the pair takes an ACCESS EXCLUSIVE lock on ``signals`` that
    Postgres holds until the migration transaction commits, so every later
    statement runs with all readers and writers of ``signals`` blocked -- and a
    live session may be writing signals exactly when someone runs ``migrate``.

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
      AND tgname = '{name}'
      AND NOT tgisinternal
      AND tgtype = {tgtype}
      AND tgfoid = 'signals_reject_mutation'::regproc
  ) THEN
    DROP TRIGGER IF EXISTS {name} ON signals;
    CREATE TRIGGER {name}
      BEFORE {events} ON signals
      FOR EACH {level} EXECUTE FUNCTION signals_reject_mutation();
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
#
# Two triggers make `signals` append-only: a row-level one rejecting UPDATE and
# DELETE, and a statement-level one rejecting TRUNCATE, which bypasses row-level
# triggers entirely. A bad run must not be quietly rewritten to look good, so
# there is no ordinary SQL path to remove a signal. Deliberate cleanup stays
# possible, but cannot happen by reflex:
#
#     ALTER TABLE signals DISABLE TRIGGER USER;
#     DELETE FROM signals WHERE run_id = '...';
#     ALTER TABLE signals ENABLE TRIGGER USER;
#
# Re-enable them in the same transaction -- `migrate` will not notice they are
# off, because the guard above matches on the trigger's definition and not on
# whether it is enabled.
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
    # A drifting target cannot be a `side`: that column is a CHECK on four
    # discrete events, and a level moving 1.00 -> 0.55 -> 0.20 is not an event.
    # It gets its own column, nullable, so the boolean path keeps writing the
    # row it writes today with NULL here -- no backfill, and no need to decide
    # what level a strategy that never held one "really meant".
    #
    # ADD COLUMN rather than a line in the CREATE above: that CREATE is IF NOT
    # EXISTS and does nothing whatever against a database that already has the
    # table, so a column declared only there would exist on a fresh checkout and
    # nowhere else. Nullable with no default is metadata-only from Postgres 11
    # on -- the catalog gains a row and the heap is not rewritten -- and the
    # append-only triggers are untouched, since no row is updated or deleted.
    #
    # NUMERIC(10,6) matches `strength`. NUMERIC rather than float8 for the
    # reason `db/candles.py` documents: a float bound to this column arrives as
    # a float8 parameter and Postgres applies its implicit float8 -> numeric
    # cast, which formats via "%.15g". Scale 6 does not hide that. Measured
    # through the real insert path against Postgres 16.13, the float64 just
    # below 0.5499995 stores as 0.550000 bound as a float and 0.549999 bound as
    # a Decimal, because the cast lands it exactly on the 6dp half-way boundary
    # that NUMERIC then rounds away from zero.
    #
    # Guarded for the reason `_guarded_trigger_migration` is, and `IF NOT
    # EXISTS` is not that guard: ALTER TABLE takes its ACCESS EXCLUSIVE lock
    # before it looks at whether there is anything to do, and Postgres holds
    # that lock until the migration transaction commits, so a no-op still
    # blocks every reader and writer of `signals` for the rest of `migrate`.
    # Measured against a session holding only ACCESS SHARE: the bare statement
    # waits out a 2s lock_timeout and fails, the guarded one returns without
    # waiting. The guard tests presence alone -- exactly what `IF NOT EXISTS`
    # tested, and no more, since this statement never repaired a column of the
    # wrong shape either.
    """
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = current_schema()
      AND table_name = 'signals'
      AND column_name = 'target_exposure'
  ) THEN
    ALTER TABLE signals ADD COLUMN target_exposure NUMERIC(10,6);
  END IF;
END $$
""".strip(),
    "CREATE INDEX IF NOT EXISTS ix_signals_lookup ON signals (symbol, timeframe, ts_bar_ms)",
    "CREATE INDEX IF NOT EXISTS ix_signals_run ON signals (run_id, ts_bar_ms)",
    """
    CREATE OR REPLACE FUNCTION signals_reject_mutation() RETURNS trigger AS $$
    BEGIN
      RAISE EXCEPTION 'signals is append-only; % is not permitted', TG_OP
        USING HINT = 'Signals are an audit trail of what the system decided. '
                     'To remove rows deliberately: ALTER TABLE signals DISABLE '
                     'TRIGGER USER; DELETE ...; ALTER TABLE signals ENABLE '
                     'TRIGGER USER;';
    END;
    $$ LANGUAGE plpgsql
    """,
    _guarded_trigger_migration(
        "trg_signals_append_only", APPEND_ONLY_TGTYPE, "UPDATE OR DELETE", "ROW"
    ),
    _guarded_trigger_migration(
        "trg_signals_no_truncate", NO_TRUNCATE_TGTYPE, "TRUNCATE", "STATEMENT"
    ),
)


# Why these are their own tables rather than columns on `market_candles` is in
# `db.funding`. What matters here: the settlement interval is per-contract -- 8h
# for most Binance perps, but not all and not always -- so nothing below encodes
# one. `funding_time_ms` is stored as the venue reported it, and the spacing is a
# property of the data rather than a rule the schema imposes.
#
# `CREATE TABLE/INDEX IF NOT EXISTS` never rewrite an existing object, which is
# what `test_rerunning_migrations_does_not_rewrite_the_table` defends. They are
# not lock-free, though, and the two `signals` indexes above are where that
# bites: measured against a session holding ROW EXCLUSIVE -- the lock an
# ordinary INSERT takes -- `CREATE INDEX IF NOT EXISTS ix_signals_lookup` waits
# for a SHARE lock it cannot get, and waits out a 2s lock_timeout. So the
# guarded ADD COLUMN above stops `migrate` blocking a *reader* of `signals`;
# not blocking a live *writer* needs those two guarded the same way, which is
# not done here.
FUNDING_MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS funding_rates (
      exchange        TEXT NOT NULL,
      market_type     TEXT NOT NULL,
      symbol          TEXT NOT NULL,
      funding_time_ms BIGINT NOT NULL,
      funding_rate    NUMERIC(38,18) NOT NULL,
      mark_price      NUMERIC(38,18),
      source          TEXT,
      created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
      CONSTRAINT uq_funding_identity UNIQUE (exchange, market_type, symbol, funding_time_ms)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS open_interest (
      exchange           TEXT NOT NULL,
      market_type        TEXT NOT NULL,
      symbol             TEXT NOT NULL,
      ts_ms              BIGINT NOT NULL,
      open_interest      NUMERIC(38,18) NOT NULL,
      open_interest_usd  NUMERIC(38,18),
      source             TEXT,
      created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
      CONSTRAINT uq_open_interest_identity UNIQUE (exchange, market_type, symbol, ts_ms)
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_funding_rates_lookup "
    "ON funding_rates (symbol, funding_time_ms)",
    "CREATE INDEX IF NOT EXISTS ix_open_interest_lookup ON open_interest (symbol, ts_ms)",
)


def run_migrations(database_url: str | None = None) -> int:
    """Apply idempotent schema upgrades. Returns the number of statements executed."""
    statements = MIGRATIONS + SIGNAL_MIGRATIONS + FUNDING_MIGRATIONS
    engine = get_engine(database_url)
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
    return len(statements)
