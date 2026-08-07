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


def _guarded_trigger_migration(
    name: str, tgtype: int, events: str, level: str, *, table: str, function: str
) -> str:
    """Install one append-only trigger, but only when it isn't already correct.

    ``DROP TRIGGER IF EXISTS`` plus an unconditional ``CREATE TRIGGER`` is safe
    but not a no-op: the pair takes an ACCESS EXCLUSIVE lock on ``table`` that
    Postgres holds until the migration transaction commits, so every later
    statement runs with all readers and writers of that table blocked -- and a
    live session may be writing signals exactly when someone runs ``migrate``.

    The guard compares the function and the event mask as well as the name, so a
    trigger that is missing, points at the wrong function, or fires on the wrong
    events is still repaired; only an already-correct trigger is left alone.

    ``table`` and ``function`` are parameters because ``bar_reasons`` needs the
    identical pair of guards, and they are *not* defaulted to ``signals``: the
    two tables get **separate** reject functions rather than one shared
    ``TG_TABLE_NAME`` implementation, because repointing ``signals``' triggers at
    a new function is a schema change to an append-only audit trail, and the
    guard would have to fire once to make it -- taking the exclusive lock this
    exists to avoid, on the one table a live replay is writing.
    """
    return f"""
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_trigger
    WHERE tgrelid = '{table}'::regclass
      AND tgname = '{name}'
      AND NOT tgisinternal
      AND tgtype = {tgtype}
      AND tgfoid = '{function}'::regproc
  ) THEN
    DROP TRIGGER IF EXISTS {name} ON {table};
    CREATE TRIGGER {name}
      BEFORE {events} ON {table}
      FOR EACH {level} EXECUTE FUNCTION {function}();
  END IF;
END $$
""".strip()


def _guarded_index_migration(name: str, table: str, columns: str) -> str:
    """Create one index, but only when it isn't already there.

    ``CREATE INDEX IF NOT EXISTS`` takes a SHARE lock on the table before it
    checks whether there is anything to build, and SHARE conflicts with the ROW
    EXCLUSIVE an ordinary INSERT holds. Measured against a session holding ROW
    EXCLUSIVE on ``signals``, these were the only two statements in the whole
    migration set that blocked -- so an unguarded re-run stalls a live replay
    writing signals for the rest of ``migrate``, while a *reader* passes
    unharmed and makes the problem invisible.

    Presence is the whole check, unlike the trigger guard above, which compares
    the function and event mask too. An index of the same name over different
    columns is a schema change rather than drift, and repairing one silently
    would rebuild it under the lock this exists to avoid.
    """
    return f"""
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_class
    WHERE relname = '{name}'
      AND relnamespace = current_schema()::regnamespace
  ) THEN
    CREATE INDEX {name} ON {table} ({columns});
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
    _guarded_index_migration("ix_signals_lookup", "signals", "symbol, timeframe, ts_bar_ms"),
    _guarded_index_migration("ix_signals_run", "signals", "run_id, ts_bar_ms"),
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
        "trg_signals_append_only",
        APPEND_ONLY_TGTYPE,
        "UPDATE OR DELETE",
        "ROW",
        table="signals",
        function="signals_reject_mutation",
    ),
    _guarded_trigger_migration(
        "trg_signals_no_truncate",
        NO_TRUNCATE_TGTYPE,
        "TRUNCATE",
        "STATEMENT",
        table="signals",
        function="signals_reject_mutation",
    ),
)


# `signals` is keyed to *events* and reasons are the *inputs* to a decision
# rather than one, so widening it would put a row of state on every bar in a
# table whose whole shape says "something happened here". Measured on the R10a
# diff window -- BTC/USDT perp 4h, 6,048 bars -- a replay of `state_machine_v1`
# emitted 325 signal rows and 6,048 reason rows. `bar_reasons` is one row per bar
# instead, and the question it exists to answer is at least as often "why did it
# *not* trade here" as "why did it trade": that machine spent 4,124 of those
# 6,048 bars in COMPRESSION, and rows on decision bars alone cannot say so.
#
# Written by the event path only. `backtest`, `sweep` and the browser recompute
# the same values per request from immutable candles and store nothing; what is
# genuinely unrecomputable is what a *live* run saw at the moment it decided --
# funding that had not settled, a feed gap, a late bar, a revision.
#
# Same append-only pair as `signals`, for the same reason and with the same
# escape hatch (ALTER TABLE bar_reasons DISABLE TRIGGER USER; DELETE ...; ENABLE).
# A row-level trigger alone is not enough: TRUNCATE bypasses row-level triggers
# entirely.
#
# The feature values are two parallel arrays rather than a JSONB object or one
# column per feature. Per-feature columns cannot work -- the set is a property of
# the strategy, and `state_machine_v1`'s five are not another strategy's -- and
# JSONB would put the values back into the float-shaped, read-as-labels box
# `runner._features` already warns about. NUMERIC elements keep the
# `Decimal(str(float(x)))` rule intact end to end: measured through this exact
# insert path, the float64 88.02116722596503 bound as a bare float stores as
# 88.021167225965 and bound as a Decimal round-trips, because an array of Python
# floats reaches Postgres as float8[] and the implicit float8 -> numeric cast
# formats via "%.15g".
#
# **Unconstrained NUMERIC, not the NUMERIC(38,18) every price column here uses.**
# A scale is a count of decimal *places* and a float64 needs up to 17
# *significant* digits, so 18 places round-trips a price near 100 and silently
# truncates a feature near 0. That is not hypothetical: R10a's first diff run
# reported `direction`, `energy`, `stability` and `strength` disagreeing with the
# research path on 5 / 7 / 28 / 3 of 1,000 bars by exactly one ULP, and the cause
# was this column rather than either code path -- 0.0020833333333333333, a
# 1/480 rolling percentile, needs 19 places. Measured on the same values,
# `crowding`'s smallest observed reading 2.0682314349096398e-05 loses three
# significant digits at scale 18 and round-trips exactly without one. Postgres'
# unconstrained numeric stores the decimal it was given, so it is the only form
# in which "the stored value is the value the machine read" is true.
#
# A NULL element is a feature's "not yet measurable" -- warmup rows are NaN by the
# `features.base.mask_warmup` convention, and 0.0 there would read as "measured,
# and neutral", a different claim about the market. NULL rather than a stored NaN
# because Postgres NUMERIC accepts NaN without complaint and the result matches
# nothing afterwards, including itself; SQL NULL at least has `IS NULL`.
BAR_REASON_MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS bar_reasons (
      id               BIGSERIAL PRIMARY KEY,
      run_id           UUID NOT NULL REFERENCES runs(run_id),
      mode             TEXT NOT NULL CONSTRAINT bar_reasons_mode_check
                       CHECK (mode IN ('backtest','replay','paper','live')),
      strategy_id      TEXT NOT NULL,
      strategy_version TEXT NOT NULL,
      exchange         TEXT NOT NULL,
      market_type      TEXT NOT NULL,
      symbol           TEXT NOT NULL,
      timeframe        TEXT NOT NULL,
      ts_bar_ms        BIGINT NOT NULL,
      ts_emit_ms       BIGINT NOT NULL,
      bar_is_closed    BOOLEAN NOT NULL,
      state            TEXT NOT NULL,
      feature_names    TEXT[] NOT NULL,
      feature_values   NUMERIC[] NOT NULL,
      created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
      CONSTRAINT bar_reasons_features_aligned
        CHECK (cardinality(feature_names) = cardinality(feature_values)),
      CONSTRAINT uq_bar_reasons_identity UNIQUE
        (run_id, strategy_id, strategy_version, exchange, symbol, timeframe, ts_bar_ms)
    )
    """,
    _guarded_index_migration(
        "ix_bar_reasons_lookup", "bar_reasons", "symbol, timeframe, ts_bar_ms"
    ),
    _guarded_index_migration("ix_bar_reasons_run", "bar_reasons", "run_id, ts_bar_ms"),
    """
    CREATE OR REPLACE FUNCTION bar_reasons_reject_mutation() RETURNS trigger AS $$
    BEGIN
      RAISE EXCEPTION 'bar_reasons is append-only; % is not permitted', TG_OP
        USING HINT = 'Bar reasons are what a live run saw when it decided, and '
                     'nothing can recompute that afterwards. To remove rows '
                     'deliberately: ALTER TABLE bar_reasons DISABLE TRIGGER USER; '
                     'DELETE ...; ALTER TABLE bar_reasons ENABLE TRIGGER USER;';
    END;
    $$ LANGUAGE plpgsql
    """,
    _guarded_trigger_migration(
        "trg_bar_reasons_append_only",
        APPEND_ONLY_TGTYPE,
        "UPDATE OR DELETE",
        "ROW",
        table="bar_reasons",
        function="bar_reasons_reject_mutation",
    ),
    _guarded_trigger_migration(
        "trg_bar_reasons_no_truncate",
        NO_TRUNCATE_TGTYPE,
        "TRUNCATE",
        "STATEMENT",
        table="bar_reasons",
        function="bar_reasons_reject_mutation",
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
# not lock-free, though: `CREATE INDEX IF NOT EXISTS` takes a SHARE lock before
# it looks for the index, and SHARE conflicts with the ROW EXCLUSIVE an INSERT
# holds. The two `signals` indexes are guarded above for that reason -- measured
# statement by statement against a session holding ROW EXCLUSIVE on `signals`,
# they were the only two of the 22 that blocked.
#
# The four indexes below are deliberately left bare. They are on `funding_rates`
# and `open_interest`, which are batch-fetched between runs and have no live
# writer to block; the `signals` pair was worth the guard because a replay
# writes signals continuously and `migrate` is exactly what someone runs while
# one is going. Guarding these too would spend the same complexity on a session
# that does not exist -- revisit it when something writes funding live.
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


# Any bigint works; this one is arbitrary and only has to stay stable, since two
# migration runs serialize only by agreeing on it.
MIGRATION_LOCK_KEY = 8_314_070_251_063_129


def run_migrations(database_url: str | None = None) -> int:
    """Apply idempotent schema upgrades. Returns the number of statements executed.

    Every statement here is guarded or ``IF NOT EXISTS``, which makes a *repeat*
    run a no-op but does not make two *concurrent* first runs safe: each of them
    checks for the object before either creates it, so both decide to create.
    Measured on two connections with an index absent, the second raises
    ``UniqueViolation`` -- and it does so for the bare ``CREATE INDEX IF NOT
    EXISTS`` exactly as it does for the ``pg_class`` guard beside it, because
    ``IF NOT EXISTS`` is checked before the lock is taken rather than under it.

    ``pg_advisory_xact_lock`` turns that into a wait. It is transaction-scoped,
    so it is released by the commit or rollback below and never leaks a lock on
    a failed migration.
    """
    # `bar_reasons` after `signals`, since it takes the same foreign key to `runs`
    # and that table is created there.
    statements = MIGRATIONS + SIGNAL_MIGRATIONS + BAR_REASON_MIGRATIONS + FUNDING_MIGRATIONS
    engine = get_engine(database_url)
    with engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": MIGRATION_LOCK_KEY})
        for statement in statements:
            conn.execute(text(statement))
    return len(statements)
