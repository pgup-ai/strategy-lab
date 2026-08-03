from __future__ import annotations

import pytest
from sqlalchemy import inspect, text

from strategy_lab.db.candles import get_engine, list_candle_sets, load_candles
from strategy_lab.storage.migrations import MIGRATIONS, PRICE_COLUMNS, run_migrations

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
