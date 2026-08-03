from __future__ import annotations

from decimal import Decimal
from urllib.parse import quote

import pandas as pd
import pytest
from sqlalchemy import text

from strategy_lab.config import settings
from strategy_lab.db.candles import (
    _batched,
    get_engine,
    init_db,
    load_candles,
    normalize_candle_frame,
    upsert_candles,
)
from tests.test_migrations import PATHOLOGICAL_FLOATS

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def test_normalize_candle_frame_adds_identity_fields() -> None:
    df = pd.DataFrame(
        {
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [10.0],
        },
        index=pd.DatetimeIndex(["2024-01-01T00:00:00Z"], name="timestamp"),
    )

    records = normalize_candle_frame(
        df,
        exchange="binance",
        market_type="spot",
        symbol="BTC/USDT",
        timeframe="15m",
        source="binance",
    )

    assert records == [
        {
            "exchange": "binance",
            "market_type": "spot",
            "symbol": "BTC/USDT",
            "timeframe": "15m",
            "timestamp": pd.Timestamp("2024-01-01T00:00:00Z").to_pydatetime(),
            "open": Decimal("1.0"),
            "high": Decimal("2.0"),
            "low": Decimal("0.5"),
            "close": Decimal("1.5"),
            "volume": Decimal("10.0"),
            "source": "binance",
        }
    ]
    # Equality alone would not pin this down: Decimal("1.5") == 1.5 is True, so the
    # dict comparison above passes whether the values are Decimal or float. The
    # binding type is the whole point of the fix, so assert it directly.
    assert all(isinstance(records[0][name], Decimal) for name in OHLCV_COLUMNS)


def test_price_fields_carry_the_float64_shortest_repr() -> None:
    """Decimal must be built from ``str(float)``, not from the float directly.

    ``Decimal(0.1)`` expands the binary value to its full 55-digit exact form;
    ``Decimal(str(0.1))`` gives ``0.1``. Both compare equal to the float, so only
    the digits themselves distinguish the two, and only the ``str`` form survives
    NUMERIC(38,18)'s 18-digit scale.
    """
    value = 0.12345678901234568
    df = pd.DataFrame(
        {name: [value] for name in OHLCV_COLUMNS},
        index=pd.DatetimeIndex(["2024-01-01T00:00:00Z"], name="timestamp"),
    )

    record = normalize_candle_frame(
        df,
        exchange="binance",
        market_type="spot",
        symbol="BTC/USDT",
        timeframe="15m",
        source="binance",
    )[0]

    for name in OHLCV_COLUMNS:
        assert record[name] == Decimal("0.12345678901234568"), name
        assert float(record[name]) == value, name


def test_nan_prices_stay_nan_rather_than_becoming_garbage() -> None:
    """Preserves the pre-fix behaviour: ``float('nan')`` went in, NaN came out.

    The Yahoo fetcher drops NaN rows, so this should never fire in practice --
    but a silent 0.0 or a crash-free wrong number would be worse than either
    NaN or an exception, so pin what actually happens.
    """
    df = pd.DataFrame(
        {name: [float("nan")] for name in OHLCV_COLUMNS},
        index=pd.DatetimeIndex(["2024-01-01T00:00:00Z"], name="timestamp"),
    )

    record = normalize_candle_frame(
        df,
        exchange="binance",
        market_type="spot",
        symbol="BTC/USDT",
        timeframe="15m",
        source="binance",
    )[0]

    for name in OHLCV_COLUMNS:
        assert isinstance(record[name], Decimal), name
        assert record[name].is_nan(), name


def test_missing_prices_fail_loudly() -> None:
    """A None price must raise, not be coerced into some plausible-looking number."""
    df = pd.DataFrame(
        {name: [None] for name in OHLCV_COLUMNS},
        index=pd.DatetimeIndex(["2024-01-01T00:00:00Z"], name="timestamp"),
    )

    with pytest.raises(TypeError):
        normalize_candle_frame(
            df,
            exchange="binance",
            market_type="spot",
            symbol="BTC/USDT",
            timeframe="15m",
            source="binance",
        )


def test_batched_splits_large_upserts() -> None:
    rows = [{"index": index} for index in range(5)]

    assert list(_batched(rows, 2)) == [
        [{"index": 0}, {"index": 1}],
        [{"index": 2}, {"index": 3}],
        [{"index": 4}],
    ]


# --- The write path against a real Postgres, end to end. ---

SCRATCH_SCHEMA = "_candle_writepath_probe"


@pytest.fixture
def scratch_database_url():
    """A URL whose ``search_path`` points at an empty throwaway schema.

    Unqualified ``market_candles`` then resolves to a clone inside that schema,
    so ``upsert_candles``/``load_candles`` run byte-identical SQL against a table
    that is not the 100k-row research dataset. Retargeting through libpq rather
    than by monkeypatching the SQLAlchemy metadata keeps this an end-to-end test
    of the shipping code path.
    """
    url = f"{settings.database_url}?options={quote(f'-csearch_path={SCRATCH_SCHEMA}')}"
    admin = get_engine()
    with admin.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCRATCH_SCHEMA} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {SCRATCH_SCHEMA}"))
    try:
        init_db(url)
        yield url
    finally:
        with admin.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {SCRATCH_SCHEMA} CASCADE"))


@pytest.mark.db
def test_written_prices_round_trip_bit_for_bit(scratch_database_url) -> None:
    """Every float64 a fetcher produces must come back out of Postgres unchanged.

    ``storage/migrations.py`` goes through ``col::text::numeric`` precisely
    because Postgres' implicit ``float8 -> numeric`` cast formats with "%.15g"
    and drops the last two significant digits. Binding a Python ``float`` to a
    NUMERIC column re-applies that same lossy cast on every insert, which -- with
    ``ON CONFLICT DO UPDATE`` and overlapping fetch windows -- rewrites correct
    stored values with degraded ones on every re-fetch. Values must therefore
    reach the driver as ``Decimal``.

    Measured before the fix: 5 of these 7 values came back changed.
    """
    values = [float(value) for value in PATHOLOGICAL_FLOATS]
    index = pd.date_range("2024-01-01", periods=len(values), freq="15min", tz="UTC")
    df = pd.DataFrame(
        {name: values for name in OHLCV_COLUMNS},
        index=pd.DatetimeIndex(index, name="timestamp"),
    )
    identity = dict(exchange="binance", market_type="spot", symbol="BTC/USDT", timeframe="15m")

    records = normalize_candle_frame(df, source="probe", **identity)
    upsert_candles(records, database_url=scratch_database_url)
    stored = load_candles(database_url=scratch_database_url, **identity)

    for name in OHLCV_COLUMNS:
        assert list(stored[name]) == values, (
            f"{name} lost precision in the write path: {list(stored[name])} != {values}"
        )


@pytest.mark.db
def test_refetching_does_not_degrade_already_correct_rows(scratch_database_url) -> None:
    """The regression that makes this critical rather than cosmetic.

    Fetch windows overlap by design and ``upsert_candles`` is ON CONFLICT DO
    UPDATE, so an already-stored bar is rewritten every time it is fetched
    again. The rows in ``market_candles`` are already correct -- the NUMERIC
    migration put them there through ``col::text::numeric``. A lossy write path
    therefore does not merely fail to improve them, it *destroys* them on the
    next re-fetch.

    So this seeds the table the way the migration leaves it (exact NUMERIC
    literals, not floats bound as parameters) and then re-fetches. Nothing may
    change.
    """
    identity = dict(exchange="binance", market_type="spot", symbol="BTC/USDT", timeframe="15m")
    engine = get_engine(scratch_database_url)
    with engine.begin() as conn:
        for offset, literal in enumerate(PATHOLOGICAL_FLOATS):
            prices = ", ".join([literal] * len(OHLCV_COLUMNS))
            conn.execute(
                text(
                    "INSERT INTO market_candles "
                    "(exchange, market_type, symbol, timeframe, timestamp, "
                    f" {', '.join(OHLCV_COLUMNS)}, source) "
                    "VALUES (:exchange, :market_type, :symbol, :timeframe, :timestamp, "
                    f" {prices}, 'seed')"
                ),
                {**identity, "timestamp": pd.Timestamp("2024-01-01", tz="UTC")
                 + pd.Timedelta(minutes=15 * offset)},
            )

    before = load_candles(database_url=scratch_database_url, **identity)
    # A re-fetch hands the same bars back to the write path; ON CONFLICT rewrites them.
    upsert_candles(
        normalize_candle_frame(before, source="probe", **identity),
        database_url=scratch_database_url,
    )
    after = load_candles(database_url=scratch_database_url, **identity)

    # check_exact is mandatory: the default rtol=1e-5 is ~11 orders of magnitude
    # looser than the corruption being tested and passes straight through it.
    pd.testing.assert_frame_equal(before, after, check_exact=True)
