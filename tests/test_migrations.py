from __future__ import annotations

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError

from strategy_lab.config import settings
from strategy_lab.db.candles import get_engine, list_candle_sets, load_candles
from strategy_lab.storage.migrations import (
    MIGRATION_LOCK_KEY,
    MIGRATIONS,
    PRICE_COLUMNS,
    _guarded_index_migration,
    run_migrations,
)

pytestmark = pytest.mark.db

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")

# float64 values that need 16-17 significant digits to round-trip. Postgres' bare
# float8 -> numeric cast formats via "%.15g" and mangles every one of them.
PATHOLOGICAL_FLOATS = (
    "187.6199951171875",
    "88.02116722596503",
    "305.6600036621094",
    "734.7999877929688",
    "123456789.12345679",
    "0.1",
    "105113.02",
)


def test_migrations_are_idempotent():
    run_migrations()
    run_migrations()  # second run must not raise


def test_candle_price_columns_are_numeric():
    run_migrations()
    columns = {c["name"]: c["type"] for c in inspect(get_engine()).get_columns("market_candles")}
    for name in OHLCV_COLUMNS:
        assert "NUMERIC" in str(columns[name]).upper(), f"{name} is {columns[name]}, expected NUMERIC"


def test_existing_candles_survive_the_migration():
    run_migrations()
    with get_engine().connect() as conn:
        total = conn.execute(text("SELECT count(*) FROM market_candles")).scalar_one()
    assert total > 0, "migration must preserve existing candle rows"


def test_price_migration_preserves_float64_exactly():
    """The float8 -> NUMERIC cast must not lose the last significant digits.

    Runs the *real* migration statements against a scratch table of float8
    columns and asserts every value survives the round-trip bit-for-bit. Drop
    the ``USING <col>::text::numeric`` clause and this fails on 5 of 7 values.
    """
    scratch = "_migration_cast_probe"
    columns = ", ".join(f"{column} float8" for column in PRICE_COLUMNS)
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {scratch}"))
            conn.execute(text(f"CREATE TABLE {scratch} ({columns})"))
            for value in PATHOLOGICAL_FLOATS:
                values = ", ".join([value] * len(PRICE_COLUMNS))
                conn.execute(text(f"INSERT INTO {scratch} VALUES ({values})"))

            for statement in MIGRATIONS:
                if "ALTER COLUMN" in statement:
                    conn.execute(text(statement.replace("market_candles", scratch)))

            expected = sorted(PATHOLOGICAL_FLOATS, key=float)
            for column in PRICE_COLUMNS:
                rows = conn.execute(
                    text(f"SELECT {column}::float8::text AS roundtrip FROM {scratch}")
                ).scalars().all()
                stored = sorted(rows, key=float)
                assert stored == expected, (
                    f"{column} lost precision in the float8 -> NUMERIC cast: {stored}"
                )
    finally:
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {scratch}"))


def test_rerunning_migrations_does_not_rewrite_the_table():
    """Idempotent must mean "no-op", not "harmlessly redoes the work".

    The NUMERIC conversion uses a USING cast, which rewrites the whole table
    under an ACCESS EXCLUSIVE lock. Unguarded, every migrate call pays that
    cost -- fine at 100k rows, a multi-second exclusive-lock stall once live
    1m candles arrive. A table rewrite allocates a new relfilenode, so an
    unchanged relfilenode proves nothing was rewritten.
    """
    run_migrations()  # reach the migrated state first
    with get_engine().connect() as conn:
        before = conn.execute(text("SELECT pg_relation_filenode('market_candles')")).scalar_one()
    run_migrations()
    with get_engine().connect() as conn:
        after = conn.execute(text("SELECT pg_relation_filenode('market_candles')")).scalar_one()

    assert before == after, (
        "migrate rewrote market_candles on a re-run; the NUMERIC conversion is "
        "firing unconditionally instead of checking the column type first"
    )


def test_rerunning_migrations_does_not_block_a_reader_of_signals():
    """A no-op statement is still not a lock-free one.

    ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` takes ACCESS EXCLUSIVE *before*
    it checks whether there is a column to add, and Postgres holds it until the
    migration transaction commits -- so an unguarded re-run stalls every reader
    and writer of ``signals``, live sessions included, for the rest of
    ``migrate``. The test above measures rewrites, which that statement does not
    do; this is the cost it does have.

    Asserted with a lock rather than a stopwatch. The second connection holds
    ACCESS SHARE, what a plain ``SELECT`` takes: it conflicts with ACCESS
    EXCLUSIVE and with nothing else the migration needs, so only the offending
    statement can trip it. ``lock_timeout`` on the migrating connection alone
    turns "would have waited" into a failure instead of a hung suite.

    A reader is the narrower claim, and it isolates this statement: ACCESS SHARE
    conflicts with ACCESS EXCLUSIVE and with nothing else the migration needs.
    The writer case is
    :func:`test_rerunning_migrations_does_not_block_a_writer_of_signals`, which
    covers the two index statements instead.
    """
    run_migrations()  # reach the migrated state first
    impatient = (
        make_url(settings.database_url)
        .update_query_dict({"options": "-c lock_timeout=2s"})
        .render_as_string(hide_password=False)
    )

    with get_engine().begin() as reader:
        reader.execute(text("LOCK TABLE signals IN ACCESS SHARE MODE"))
        # Returning is the assertion: unguarded, this raises LockNotAvailable
        # once the timeout expires, naming the ADD COLUMN as the waiting
        # statement.
        run_migrations(impatient)


def test_rerunning_migrations_does_not_block_a_writer_of_signals():
    """The same cost, one lock level up, from a different pair of statements.

    ``CREATE INDEX IF NOT EXISTS`` asks for a SHARE lock before it looks for the
    index, and SHARE conflicts with the ROW EXCLUSIVE an ordinary INSERT holds.
    A reader never sees it -- ACCESS SHARE and SHARE are compatible -- so the
    test above passes against the unguarded statements and this one does not,
    which is why both exist rather than one widened.

    ROW EXCLUSIVE is what makes this specific: measured statement by statement
    against a session holding it, ``ix_signals_lookup`` and ``ix_signals_run``
    were the only two of the 22 that blocked. The live session this protects is
    a replay writing signals while someone runs ``migrate``.
    """
    run_migrations()
    impatient = (
        make_url(settings.database_url)
        .update_query_dict({"options": "-c lock_timeout=2s"})
        .render_as_string(hide_password=False)
    )

    with get_engine().begin() as writer:
        # LOCK rather than a real INSERT: `signals` is append-only and the row
        # would outlive the test. The lock is what an INSERT would hold anyway.
        writer.execute(text("LOCK TABLE signals IN ROW EXCLUSIVE MODE"))
        run_migrations(impatient)


def test_migrations_serialize_against_each_other():
    """Idempotent on re-run is a different claim from safe when run twice at once.

    Every statement checks for its object before creating it, and none of those
    checks is taken under a lock that would make it binding, so two runs
    starting with the object absent both decide to create. Measured directly on
    a scratch index, with the second connection held until the first had built
    it: ``UniqueViolation`` -- and the bare ``CREATE INDEX IF NOT EXISTS`` fails
    identically, so this is inherent to check-then-create rather than something
    the ``pg_class`` guard introduced.

    Asserted as "the lock is taken", not as a reproduced collision, because
    **there is no reproduction to write through** ``run_migrations``: every
    object its guards look for is also declared in ``storage/schema.py`` and so
    created by ``init_db``'s ``create_all``, leaving the guards false on any
    database this code produces. Measured on a throwaway database -- two
    concurrent first runs raise nothing with the lock removed. The exposure is
    a future migration whose object is not in ``create_all``, which is what the
    lock is here for; a test claiming to race one today would be a test that
    cannot fail.
    """
    run_migrations()
    impatient = (
        make_url(settings.database_url)
        .update_query_dict({"options": "-c lock_timeout=2s"})
        .render_as_string(hide_password=False)
    )

    with get_engine().begin() as holder:
        holder.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": MIGRATION_LOCK_KEY})
        with pytest.raises(OperationalError, match="lock timeout"):
            run_migrations(impatient)


def test_the_index_guard_still_builds_an_index_that_is_missing():
    """The guard has to skip an existing index *and* create an absent one.

    Every other test here runs against an already-migrated database, where a
    guard that silently never fires is indistinguishable from one that works --
    including the lock tests above, which a statement that does nothing passes
    most easily of all. On a scratch table because that is the only place the
    absent case exists.
    """
    scratch, index = "_index_guard_probe", "ix_index_guard_probe"
    statement = _guarded_index_migration(index, scratch, "side")
    oid = text("SELECT oid FROM pg_class WHERE relname = :n")
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(text(f"CREATE TABLE {scratch} (id bigserial primary key, side text)"))

            conn.execute(text(statement))
            built = conn.execute(
                text("SELECT indexdef FROM pg_indexes WHERE indexname = :n"), {"n": index}
            ).scalar_one_or_none()
            before = conn.execute(oid, {"n": index}).scalar_one()

            conn.execute(text(statement))
            after = conn.execute(oid, {"n": index}).scalar_one()
    finally:
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {scratch}"))

    assert built is not None, "the guard skipped an index that did not exist"
    assert "(side)" in built, f"built over the wrong columns: {built}"
    # An unchanged OID is the no-op: a guard that dropped and rebuilt would
    # satisfy every other assertion here while taking the lock it exists to avoid.
    assert before == after


def test_load_candles_returns_float64_not_decimal():
    """The Decimal -> float64 boundary: strategies do float64 pandas math.

    Postgres NUMERIC comes back from pd.read_sql as object-dtype Decimal. If any
    column leaks through uncoerced, every strategy in the repo breaks, so assert
    all five columns land as float64.
    """
    run_migrations()
    sets = list_candle_sets()
    if sets.empty:
        pytest.skip("no candle sets stored; nothing to coerce")

    largest = sets.sort_values("candles", ascending=False).iloc[0]
    df = load_candles(
        exchange=largest["exchange"],
        market_type=largest["market_type"],
        symbol=largest["symbol"],
        timeframe=largest["timeframe"],
    )
    assert not df.empty

    for name in OHLCV_COLUMNS:
        assert df[name].dtype == "float64", f"{name} is {df[name].dtype}, expected float64"
