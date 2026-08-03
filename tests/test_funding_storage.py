"""Storage for funding rates and open-interest snapshots.

Funding is a settlement cash flow and open interest is a point-in-time snapshot,
so neither rides on ``market_candles``. What these tests mostly defend is the
number path: Binance sends exact decimal *strings*, and every place this repo has
lost data before was a ``float`` sneaking into a ``NUMERIC`` bind.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest
from sqlalchemy import text

from strategy_lab.db.candles import get_engine
from strategy_lab.db.funding import (
    MAX_FUNDING_ROWS_PER_INSERT,
    funding_table,
    load_funding,
    load_open_interest,
    upsert_funding,
    upsert_open_interest,
)
from strategy_lab.storage.migrations import run_migrations

pytestmark = pytest.mark.db

IDENTITY = dict(exchange="binance", market_type="perp", symbol="TEST/USDT")


@pytest.fixture(autouse=True)
def clean_slate():
    """Both tables are ordinary upsert targets, so rows survive the process that
    wrote them and the next run inherits them.

    Without this, an assertion can be satisfied by a *previous* run's row rather
    than by the one the test just wrote -- which is how a test passes against a
    deliberately broken implementation. Measured: with rows left behind,
    replacing ``ON CONFLICT DO UPDATE`` with ``DO NOTHING`` kept all 12 tests
    green. Only the ``TEST``-prefixed symbols are touched; real perp data is
    stored under ``BTC/USDT`` and ``ETH/USDT``.
    """
    run_migrations()
    _delete_test_symbols()
    yield
    _delete_test_symbols()


def _delete_test_symbols() -> None:
    with get_engine().begin() as conn:
        for table in ("funding_rates", "open_interest"):
            conn.execute(text(f"DELETE FROM {table} WHERE symbol LIKE 'TEST%'"))


def _rows(count: int, start_ms: int = 1_700_000_000_000):
    return [
        {
            **IDENTITY,
            "funding_time_ms": start_ms + i * 28_800_000,
            "funding_rate": Decimal("0.0001") * (i + 1),
            "mark_price": Decimal("60000.5"),
        }
        for i in range(count)
    ]


def test_funding_round_trips_as_decimal():
    upsert_funding(_rows(1))
    loaded = load_funding(**IDENTITY)
    assert not loaded.empty
    assert loaded["funding_rate"].iloc[0] == pytest.approx(0.0001)


def test_refetching_funding_does_not_duplicate():
    rows = _rows(3, start_ms=1_710_000_000_000)
    upsert_funding(rows)
    before = len(load_funding(**IDENTITY))
    upsert_funding(rows)
    assert len(load_funding(**IDENTITY)) == before


def test_funding_loads_in_time_order():
    rows = _rows(4, start_ms=1_720_000_000_000)
    upsert_funding(list(reversed(rows)))
    loaded = load_funding(**IDENTITY)
    assert loaded.index.is_monotonic_increasing


def test_open_interest_round_trips():
    upsert_open_interest([
        {**IDENTITY, "ts_ms": 1_730_000_000_000,
         "open_interest": Decimal("108899.067"),
         "open_interest_usd": Decimal("6933046705.78")},
    ])
    loaded = load_open_interest(**IDENTITY)
    assert not loaded.empty
    assert loaded["open_interest"].iloc[0] == pytest.approx(108899.067)


def test_refetching_open_interest_does_not_duplicate():
    """OI is polled forward over deliberately overlapping windows, so every poll
    re-offers rows the previous one already stored."""
    rows = [
        {**IDENTITY, "ts_ms": 1_731_000_000_000 + i * 14_400_000,
         "open_interest": Decimal("100.5"), "open_interest_usd": Decimal("6000000.25")}
        for i in range(3)
    ]
    upsert_open_interest(rows)
    before = len(load_open_interest(**IDENTITY))
    upsert_open_interest(rows)
    assert len(load_open_interest(**IDENTITY)) == before


def test_a_large_price_survives_the_numeric_bind():
    """A bare ``float`` bind reaches Postgres as float8 and is cast via "%.15g".

    Measured on this database: bound as a float, 88.02116722596503 comes back
    88.021167225965 and 6933046705.789517 comes back 6933046705.78952. Note the
    funding *rate* cannot show this -- NUMERIC(38,18) at a magnitude of 1e-4
    holds only ~14 significant digits, which "%.15g" never truncates -- so the
    two large-magnitude columns are where the mangling is observable.
    """
    observed_ms = 1_740_000_000_000
    upsert_funding([
        {**IDENTITY, "funding_time_ms": observed_ms,
         "funding_rate": Decimal("0.0001"), "mark_price": Decimal("88.02116722596503")},
    ])
    upsert_open_interest([
        {**IDENTITY, "ts_ms": observed_ms,
         "open_interest": Decimal("187.6199951171875"),
         "open_interest_usd": Decimal("6933046705.789517")},
    ])

    observed = str(pd.Timestamp(observed_ms, unit="ms", tz="UTC"))
    funding = load_funding(**IDENTITY, start=observed, end=observed)
    interest = load_open_interest(**IDENTITY, start=observed, end=observed)

    assert funding["mark_price"].iloc[0] == 88.02116722596503
    assert interest["open_interest"].iloc[0] == 187.6199951171875
    assert interest["open_interest_usd"].iloc[0] == 6933046705.789517


def test_an_exchange_decimal_is_stored_without_a_float_round_trip():
    """Binance sends exact decimal strings, so nothing here needs float64 at all.

    ``Decimal(str(float(x)))`` is the right coercion for a value that *arrives*
    as a float (``normalize_candle_frame``, out of pandas), but routing an
    already-exact Decimal through float64 would discard digits the column can
    hold. Read back over raw SQL, because ``load_funding`` deliberately coerces
    to float64 and would hide the difference.
    """
    exact = Decimal("1.000000000000000001")  # 1.0 exactly, once through float64
    assert float(exact) == 1.0
    upsert_funding([
        {**IDENTITY, "funding_time_ms": 1_750_000_000_000,
         "funding_rate": Decimal("0.0001"), "mark_price": exact},
    ])

    with get_engine().connect() as conn:
        stored = conn.execute(
            text(
                "SELECT mark_price FROM funding_rates "
                "WHERE symbol = :symbol AND funding_time_ms = :ts"
            ),
            {"symbol": IDENTITY["symbol"], "ts": 1_750_000_000_000},
        ).scalar_one()

    assert stored == exact, f"mark_price went through float64: {stored}"


def test_a_refetch_corrects_a_revised_rate():
    """Last write wins, matching ``upsert_candles``: a redelivered row is the
    corrected one, not a duplicate to ignore."""
    settled_ms = 1_760_000_000_000
    row = {**IDENTITY, "funding_time_ms": settled_ms,
           "funding_rate": Decimal("0.0001"), "mark_price": Decimal("60000.5")}
    upsert_funding([row])
    upsert_funding([{**row, "funding_rate": Decimal("-0.0003")}])

    settled = pd.Timestamp(settled_ms, unit="ms", tz="UTC")
    loaded = load_funding(**IDENTITY, start=str(settled), end=str(settled))
    assert len(loaded) == 1
    assert loaded["funding_rate"].iloc[0] == pytest.approx(-0.0003)


def test_funding_loads_as_float64_on_a_utc_index():
    """The one deliberate Decimal -> float64 boundary, matching ``load_candles``.

    NUMERIC arrives from psycopg as object-dtype ``Decimal``; every consumer
    downstream does float64 pandas math.
    """
    upsert_funding(_rows(2, start_ms=1_770_000_000_000))
    loaded = load_funding(**IDENTITY)

    assert loaded["funding_rate"].dtype == "float64"
    assert loaded["mark_price"].dtype == "float64"
    assert loaded["funding_time_ms"].dtype == "int64"
    assert str(loaded.index.tz) == "UTC"
    assert loaded.index.name == "timestamp"


def test_an_empty_range_has_the_same_shape_as_a_populated_one():
    """A bare ``DataFrame(columns=...)`` hands back object dtype and a tz-naive
    index, so an empty window would poison a concat or an indicator that the
    identical code handles fine when rows exist."""
    upsert_funding(_rows(1, start_ms=1_780_000_000_000))
    populated = load_funding(**IDENTITY)
    empty = load_funding(**IDENTITY, start="1999-01-01", end="1999-02-01")

    assert empty.empty
    assert list(empty.columns) == list(populated.columns)
    assert empty.dtypes.to_dict() == populated.dtypes.to_dict()
    assert str(empty.index.tz) == "UTC"
    assert empty.index.name == "timestamp"


def test_start_and_end_bound_the_returned_window():
    start_ms = 1_790_000_000_000
    upsert_funding(_rows(6, start_ms=start_ms))
    first = pd.Timestamp(start_ms, unit="ms", tz="UTC")

    windowed = load_funding(
        **IDENTITY,
        start=str(first + pd.Timedelta(hours=8)),
        end=str(first + pd.Timedelta(hours=24)),
    )

    assert len(windowed) == 3
    assert windowed.index.min() == first + pd.Timedelta(hours=8)
    assert windowed.index.max() == first + pd.Timedelta(hours=24)


def test_a_write_larger_than_one_statement_is_chunked():
    """Postgres caps a statement at 65535 bound parameters and a multi-row INSERT
    binds one per column per row. Unchunked, this raises rather than running
    slowly -- this repo already hit that ceiling once, above 3,640 rows.
    """
    assert MAX_FUNDING_ROWS_PER_INSERT * len(funding_table.c) <= 65535

    count = 65535 // len(funding_table.c) + 500
    rows = [
        {**IDENTITY, "symbol": "TESTCHUNK/USDT",
         "funding_time_ms": 1_800_000_000_000 + i * 28_800_000,
         "funding_rate": Decimal("0.0001"), "mark_price": Decimal("60000.5")}
        for i in range(count)
    ]
    upsert_funding(rows)

    loaded = load_funding(exchange="binance", market_type="perp", symbol="TESTCHUNK/USDT")
    assert len(loaded) == count
