from __future__ import annotations

import uuid
from contextlib import contextmanager
from decimal import Decimal

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine

from strategy_lab.core.types import InstrumentId, Mode, Side, Signal
from strategy_lab.storage.migrations import run_migrations
from strategy_lab.storage.schema import signals_table
from strategy_lab.storage.signals import (
    MAX_BOUND_PARAMETERS,
    MAX_ROWS_PER_INSERT,
    create_run,
    load_signals,
    write_signals,
)

pytestmark = pytest.mark.db

INSTRUMENT = InstrumentId("binance", "perp", "BTC/USDT")


def make_signal(ts_bar_ms: int, side: Side = Side.ENTER_LONG, **overrides) -> Signal:
    fields = {
        "instrument": INSTRUMENT,
        "timeframe": "15m",
        "strategy_id": "turnaround_v2",
        "strategy_version": "1.0.0",
        "ts_bar_ms": ts_bar_ms,
        "ts_emit_ms": ts_bar_ms + 900_000,
        "side": side,
        "bar_is_closed": True,
        "reason": "2 red then green",
        "entry_price": Decimal("63128.00"),
        "stop_loss": Decimal("62740.10"),
        "features": {"ema200": "62110.4"},
    }
    return Signal(**(fields | overrides))


@contextmanager
def count_signal_inserts():
    """Collect every INSERT issued against ``signals`` inside the block.

    Listens on the ``Engine`` class because ``get_engine`` hands back a fresh
    engine per call, so there is no instance to attach to.
    """
    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("INSERT INTO SIGNALS"):
            statements.append(statement)

    event.listen(Engine, "before_cursor_execute", record)
    try:
        yield statements
    finally:
        event.remove(Engine, "before_cursor_execute", record)


@pytest.fixture
def run_id():
    run_migrations()
    return create_run(
        run_id=uuid.uuid4(),
        mode=Mode.REPLAY,
        strategy_id="turnaround_v2",
        strategy_version="1.0.0",
        config={"source": "test"},
    )


def test_opposite_sides_on_the_same_bar_both_persist(run_id):
    """turnaround_v2 exits long and enters short on one bar — both must survive."""
    ts = 1_785_723_300_000
    written = write_signals(
        run_id,
        Mode.REPLAY,
        [make_signal(ts, Side.EXIT_LONG), make_signal(ts, Side.ENTER_SHORT)],
    )
    assert written == 2
    # The id tiebreak in load_signals keeps a reversal bar's exit ahead of its
    # entry; on ts_bar_ms alone the two rows tie and the order is unspecified.
    assert [s.side for s in load_signals(run_id=run_id)] == [Side.EXIT_LONG, Side.ENTER_SHORT]


def test_signals_load_in_bar_order(run_id):
    write_signals(
        run_id,
        Mode.REPLAY,
        [make_signal(1_785_724_200_000), make_signal(1_785_723_300_000)],
    )
    loaded = load_signals(run_id=run_id)
    assert [s.ts_bar_ms for s in loaded] == [1_785_723_300_000, 1_785_724_200_000]


def test_a_partially_overlapping_rewrite_inserts_only_the_new_signals(run_id):
    """A resumed replay re-emits bars it already stored plus the new ones.

    Reporting the offered count rather than the inserted count would make a
    resumed run look like it discovered signals it merely re-derived.
    """
    first, second, third = 1_785_723_300_000, 1_785_724_200_000, 1_785_725_100_000
    assert write_signals(run_id, Mode.REPLAY, [make_signal(first), make_signal(second)]) == 2

    overlapping = [make_signal(first), make_signal(second), make_signal(third)]
    assert write_signals(run_id, Mode.REPLAY, overlapping) == 1
    assert write_signals(run_id, Mode.REPLAY, overlapping) == 0
    assert [s.ts_bar_ms for s in load_signals(run_id=run_id)] == [first, second, third]


def test_a_duplicate_inside_one_batch_collapses_rather_than_raising(run_id):
    """Two identical signals in one call conflict with each other, not with a
    stored row -- the ON CONFLICT DO UPDATE form raises "cannot affect row a
    second time" on exactly this input."""
    ts = 1_785_723_300_000
    assert write_signals(run_id, Mode.REPLAY, [make_signal(ts), make_signal(ts)]) == 1
    assert len(load_signals(run_id=run_id)) == 1


def test_the_whole_signal_survives_the_round_trip(run_id):
    """Field-by-field equality, not just the price.

    Task 12 diffs a replay against a re-replay by comparing Signals, so any
    field the round trip mangles turns into a phantom determinism failure.
    """
    original = Signal(
        instrument=INSTRUMENT,
        timeframe="15m",
        strategy_id="turnaround_v2",
        strategy_version="1.0.0",
        ts_bar_ms=1_785_723_300_000,
        ts_emit_ms=1_785_724_200_000,
        side=Side.ENTER_SHORT,
        bar_is_closed=False,
        reason="2 green then red",
        entry_price=Decimal("63128.00"),
        stop_loss=Decimal("62740.10"),
        take_profit=Decimal("64000"),
        strength=Decimal("0.75"),
        features={"ema200": "62110.4"},
    )
    write_signals(run_id, Mode.REPLAY, [original])
    assert load_signals(run_id=run_id) == [original]


@pytest.mark.parametrize(
    "price",
    [
        pytest.param(Decimal("0.000000000000000001"), id="smallest-representable-18dp"),
        pytest.param(Decimal("99999999999999999999.123456789012345678"), id="full-38-digits"),
    ],
)
def test_prices_at_the_edge_of_the_column_precision_round_trip_exactly(run_id, price):
    """NUMERIC(38,18) holds 20 integer and 18 fractional digits. Everything
    inside that envelope must return the same value -- these feed PnL."""
    write_signals(run_id, Mode.REPLAY, [make_signal(1_785_723_300_000, entry_price=price)])
    assert load_signals(run_id=run_id)[0].entry_price == price


def test_features_round_trip_and_a_missing_dict_becomes_empty(run_id):
    """The column is NOT NULL DEFAULT '{}', so ``None`` in comes back as ``{}``."""
    first, second = 1_785_723_300_000, 1_785_724_200_000
    write_signals(
        run_id,
        Mode.REPLAY,
        [
            make_signal(first, features={"ema200": "62110.4", "atr": 41.5, "long": True}),
            make_signal(second, features=None),
        ],
    )
    loaded = load_signals(run_id=run_id)
    assert loaded[0].features == {"ema200": "62110.4", "atr": 41.5, "long": True}
    assert loaded[1].features == {}


def test_a_decimal_in_features_fails_loudly_and_writes_nothing(run_id):
    """``features`` is JSONB and Decimal is not JSON-serialisable.

    Decimal is this codebase's default numeric type, so a strategy stashing one
    in features is a natural mistake. The trap would be "fixing" it with a str
    coercion, which round-trips a Decimal back as a string and breaks the next
    comparison silently.
    """
    with pytest.raises(TypeError, match="Decimal is not JSON serializable"):
        write_signals(
            run_id, Mode.REPLAY, [make_signal(1_785_723_300_000, features={"atr": Decimal("1.5")})]
        )
    assert load_signals(run_id=run_id) == []


def test_writing_nothing_touches_no_database(run_id):
    """An unreachable URL proves the early return happens before any connection,
    so a bar that produced no signals costs nothing."""
    unreachable = "postgresql+psycopg://nobody@127.0.0.1:1/nothing"
    assert write_signals(run_id, Mode.REPLAY, [], database_url=unreachable) == 0


def test_the_insert_batch_stays_under_the_bound_parameter_limit():
    """The configured chunk size must actually fit inside Postgres' 65535 bound
    parameter cap -- the chunking test below monkeypatches the size, so nothing
    else checks the shipped value. Checked arithmetically because proving it
    against the server means appending thousands of rows to an append-only table.
    """
    assert MAX_ROWS_PER_INSERT >= 1
    assert MAX_ROWS_PER_INSERT * len(signals_table.c) <= MAX_BOUND_PARAMETERS


def test_a_batch_larger_than_one_statement_is_written_whole(run_id, monkeypatch):
    """Chunking must not lose, duplicate, or reorder rows across statements.

    The statement count is asserted, not just the result: five rows in one
    statement also loads back correctly, so without it this passes just as
    happily when the chunking is gone -- and the bug it prevents only appears
    above 3276 rows, far past what a test should append to an append-only table.
    """
    monkeypatch.setattr("strategy_lab.storage.signals.MAX_ROWS_PER_INSERT", 2)
    base = 1_785_723_300_000
    timestamps = [base + i * 900_000 for i in range(5)]

    with count_signal_inserts() as statements:
        assert write_signals(run_id, Mode.REPLAY, [make_signal(ts) for ts in timestamps]) == 5

    assert len(statements) == 3, "5 rows at 2 per statement must be split into 3 inserts"
    assert [s.ts_bar_ms for s in load_signals(run_id=run_id)] == timestamps


def test_load_signals_can_filter_by_symbol_and_timeframe(run_id):
    write_signals(run_id, Mode.REPLAY, [make_signal(1_785_723_300_000)])
    assert len(load_signals(run_id=run_id, symbol="BTC/USDT", timeframe="15m")) == 1
    assert load_signals(run_id=run_id, symbol="ETH/USDT") == []
    assert load_signals(run_id=run_id, timeframe="1h") == []
