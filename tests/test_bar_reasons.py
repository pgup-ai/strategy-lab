"""The per-bar reason trail: its schema guarantees and its write path.

``bar_reasons`` is the one table the event path writes that no other path can
reconstruct, so the guarantees it needs are ``signals``' guarantees: append-only
through two triggers, one identity per bar, and a ``run_id`` that makes a second
replay a second run rather than a rewrite of the first.
"""

from __future__ import annotations

import re
import uuid
from decimal import Decimal

import pytest
from sqlalchemy import text

from strategy_lab.core.types import BarReason, InstrumentId, Mode
from strategy_lab.db.candles import get_engine
from strategy_lab.storage.bar_reasons import (
    MAX_BOUND_PARAMETERS,
    MAX_ROWS_PER_INSERT,
    load_bar_reasons,
    write_bar_reasons,
)
from strategy_lab.storage.migrations import run_migrations
from strategy_lab.storage.schema import bar_reasons_table
from strategy_lab.storage.signals import create_run

pytestmark = pytest.mark.db

INSTRUMENT = InstrumentId("binance", "perp", "BTC/USDT")


def make_reason(ts_bar_ms: int, **overrides) -> BarReason:
    fields = {
        "instrument": INSTRUMENT,
        "timeframe": "4h",
        "strategy_id": "state_machine_v1",
        "strategy_version": "1.0.0",
        "ts_bar_ms": ts_bar_ms,
        "ts_emit_ms": ts_bar_ms + 14_400_000,
        "bar_is_closed": True,
        "state": "compression",
        "features": {"direction": 0.62, "strength": 0.11, "crowding": 0.5},
    }
    return BarReason(**(fields | overrides))


@pytest.fixture
def run_id():
    run_migrations()
    return create_run(
        run_id=uuid.uuid4(),
        mode=Mode.REPLAY,
        strategy_id="state_machine_v1",
        strategy_version="1.0.0",
        config={},
    )


def _insert_raw(conn, run_id, *, ts_bar_ms: int) -> None:
    conn.execute(
        text(
            "INSERT INTO bar_reasons (run_id, mode, strategy_id, strategy_version, exchange, "
            "market_type, symbol, timeframe, ts_bar_ms, ts_emit_ms, bar_is_closed, state, "
            "feature_names, feature_values) VALUES (:run_id, 'replay', 'state_machine_v1', "
            "'1.0.0', 'binance', 'perp', 'BTC/USDT', '4h', :ts, :ts, TRUE, 'compression', "
            "ARRAY['direction'], ARRAY[0.5]::numeric[])"
        ),
        {"run_id": run_id, "ts": ts_bar_ms},
    )


# --------------------------------------------------------------------------
# Schema guarantees
# --------------------------------------------------------------------------


def test_appending_a_reason_is_never_blocked(run_id):
    """The triggers fire BEFORE UPDATE OR DELETE. If either ever catches INSERT
    the live path stops recording the one thing it alone can record, so assert
    the append explicitly rather than relying on other tests' inserts."""
    with get_engine().begin() as conn:
        _insert_raw(conn, run_id, ts_bar_ms=1_785_723_600_000)
        count = conn.execute(
            text("SELECT count(*) FROM bar_reasons WHERE run_id = :r"), {"r": run_id}
        ).scalar_one()
    assert count == 1


def test_bar_reasons_are_append_only(run_id):
    engine = get_engine()
    with engine.begin() as conn:
        _insert_raw(conn, run_id, ts_bar_ms=1_785_723_400_000)

    with pytest.raises(Exception, match="append-only"):
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE bar_reasons SET state = 'riding' WHERE run_id = :r"), {"r": run_id}
            )

    with pytest.raises(Exception, match="append-only"):
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM bar_reasons WHERE run_id = :r"), {"r": run_id})


def test_truncate_is_rejected(run_id):
    """TRUNCATE bypasses row-level triggers entirely.

    Without the statement-level guard, ``TRUNCATE bar_reasons`` erases the whole
    trail without firing trg_bar_reasons_append_only and without any warning --
    and this table carries far more rows than ``signals`` over the same range --
    6,048 against 325 on R10a's diff window -- so "let me just clear that out" is
    a *more* normal thing for someone to type here, not less.
    """
    engine = get_engine()
    with engine.begin() as conn:
        _insert_raw(conn, run_id, ts_bar_ms=1_785_723_700_000)

    with pytest.raises(Exception, match="append-only"):
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE bar_reasons"))

    with engine.connect() as conn:
        survived = conn.execute(
            text("SELECT count(*) FROM bar_reasons WHERE run_id = :r"), {"r": run_id}
        ).scalar_one()
    assert survived == 1, "TRUNCATE was rejected but the row is gone anyway"


def test_reasons_can_be_deleted_only_by_deliberately_disabling_the_triggers(run_id):
    """The documented escape hatch must actually work, exactly as for ``signals``."""
    engine = get_engine()
    with engine.begin() as conn:
        _insert_raw(conn, run_id, ts_bar_ms=1_785_723_800_000)

    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE bar_reasons DISABLE TRIGGER USER"))
        conn.execute(text("DELETE FROM bar_reasons WHERE run_id = :r"), {"r": run_id})
        conn.execute(text("ALTER TABLE bar_reasons ENABLE TRIGGER USER"))

    with engine.connect() as conn:
        remaining = conn.execute(
            text("SELECT count(*) FROM bar_reasons WHERE run_id = :r"), {"r": run_id}
        ).scalar_one()
        enabled = conn.execute(
            text(
                "SELECT count(*) FROM pg_trigger WHERE tgrelid = 'bar_reasons'::regclass "
                "AND NOT tgisinternal AND tgenabled <> 'D'"
            )
        ).scalar_one()
    assert remaining == 0, "the deliberate cleanup path no longer works"
    assert enabled == 2, "both triggers must be left enabled after the cleanup"


def test_rerunning_migrations_does_not_recreate_either_trigger():
    """Idempotent must mean "no-op", not "harmlessly redoes the work".

    ``DROP TRIGGER`` + ``CREATE TRIGGER`` takes an ACCESS EXCLUSIVE lock the
    migration transaction holds to the end, so every later statement runs with
    this table's writers blocked. Recreating a trigger allocates a new
    pg_trigger.oid, so unchanged oids prove both were left alone.
    """
    run_migrations()
    query = text(
        "SELECT tgname, oid FROM pg_trigger WHERE tgrelid = 'bar_reasons'::regclass "
        "AND NOT tgisinternal ORDER BY tgname"
    )
    with get_engine().connect() as conn:
        before = conn.execute(query).all()
    run_migrations()
    with get_engine().connect() as conn:
        after = conn.execute(query).all()

    assert len(before) == 2, f"expected both triggers, found {[r.tgname for r in before]}"
    assert before == after, f"migrate recreated a trigger on a re-run ({before} -> {after})"


def test_mode_check_matches_the_core_enum():
    """The SQL CHECK and ``core.types.Mode`` are two copies of one vocabulary.

    ``signals`` has the same pairing and ``tests/test_signals_schema.py`` guards
    it there; this is the second copy, and a table that silently refuses a mode
    the engine emits fails at runtime, in live, mid-session.
    """
    run_migrations()
    with get_engine().connect() as conn:
        definition = conn.execute(
            text(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conname = 'bar_reasons_mode_check' AND conrelid = 'bar_reasons'::regclass"
            )
        ).scalar_one()
    allowed = set(re.findall(r"'([a-z_]+)'::text", definition))
    assert allowed == {member.value for member in Mode}


def test_a_row_cannot_carry_more_names_than_values(run_id):
    """Two parallel arrays are one mapping only while they stay the same length.

    A row that lost the alignment reads back as a different feature's value under
    a feature's name -- silently, and only for the features past the truncation.
    """
    with pytest.raises(Exception, match="bar_reasons_features_aligned"):
        with get_engine().begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO bar_reasons (run_id, mode, strategy_id, strategy_version, "
                    "exchange, market_type, symbol, timeframe, ts_bar_ms, ts_emit_ms, "
                    "bar_is_closed, state, feature_names, feature_values) VALUES "
                    "(:r, 'replay', 's', '1.0.0', 'binance', 'perp', 'BTC/USDT', '4h', 1, 1, "
                    "TRUE, 'compression', ARRAY['a','b'], ARRAY[0.5]::numeric[])"
                ),
                {"r": run_id},
            )


# --------------------------------------------------------------------------
# The write path
# --------------------------------------------------------------------------


def test_one_row_per_bar_and_the_count_is_the_bar_count(run_id):
    """Success check 1 of the plan, at the storage layer.

    A per-bar table whose row count is not the bar count is answering a different
    question from the one it was built for.
    """
    bars = [1_785_720_000_000 + i * 14_400_000 for i in range(50)]
    written = write_bar_reasons(run_id, Mode.REPLAY, [make_reason(ts) for ts in bars])

    stored = load_bar_reasons(run_id=run_id)
    assert written == 50
    assert len(stored) == 50
    assert [reason.ts_bar_ms for reason in stored] == bars


def test_re_offering_a_bar_within_a_run_writes_nothing(run_id):
    """Within a run the first row for a bar wins.

    A stored record of what a run saw that a later pass can overwrite is not a
    record, so the re-offer has to be free rather than an error *and* has to
    change nothing -- note the second reason below carries a different state.
    """
    ts = 1_785_720_000_000
    assert write_bar_reasons(run_id, Mode.REPLAY, [make_reason(ts)]) == 1
    assert write_bar_reasons(run_id, Mode.REPLAY, [make_reason(ts, state="riding")]) == 0

    [stored] = load_bar_reasons(run_id=run_id)
    assert stored.state == "compression"


def test_a_second_replay_of_one_range_adds_a_second_run_s_rows():
    """Success check 5: ``run_id`` semantics match ``signals``.

    Idempotency is *within* a run, not across runs -- two runs saw the range at
    two different moments, and on the live path that difference is the entire
    reason the table exists. Replaying twice is expected to double the rows.
    """
    run_migrations()
    bars = [1_785_720_000_000 + i * 14_400_000 for i in range(5)]
    runs = [
        create_run(
            run_id=uuid.uuid4(),
            mode=Mode.REPLAY,
            strategy_id="state_machine_v1",
            strategy_version="1.0.0",
            config={},
        )
        for _ in range(2)
    ]
    for one in runs:
        assert write_bar_reasons(one, Mode.REPLAY, [make_reason(ts) for ts in bars]) == 5

    assert [len(load_bar_reasons(run_id=one)) for one in runs] == [5, 5]


def test_a_warmup_value_round_trips_as_none_rather_than_zero(run_id):
    """``None`` is "not yet measurable"; ``0.0`` is "measured, and neutral".

    They are different claims about the market, and a table that collapses them
    puts a measured mid-range reading on a bar nothing could be measured on.
    """
    write_bar_reasons(
        run_id,
        Mode.REPLAY,
        [make_reason(1_785_720_000_000, features={"direction": None, "strength": 0.0})],
    )

    [stored] = load_bar_reasons(run_id=run_id)
    assert stored.features == {"direction": None, "strength": 0.0}
    with get_engine().connect() as conn:
        values = conn.execute(
            text("SELECT feature_values FROM bar_reasons WHERE run_id = :r"), {"r": run_id}
        ).scalar_one()
    assert values[0] is None, "a warmup value became a stored number"


@pytest.mark.parametrize(
    "value",
    [
        # The float8 -> numeric cast: bound as a bare float this stores as
        # 88.021167225965, because a list of Python floats reaches Postgres as
        # float8[] and that cast formats via "%.15g".
        88.02116722596503,
        # The column's *scale*, which is the half the R10a diff found rather than
        # anyone predicting. This is a 1/480 rolling percentile and its shortest
        # round-tripping decimal needs 19 places; at NUMERIC(38,18) it comes back
        # one ULP low, and four features then read as disagreeing with the
        # research path on bars where nothing disagreed.
        0.0020833333333333333,
        # `crowding`'s smallest reading over the diff window: three significant
        # digits gone at scale 18.
        2.0682314349096398e-05,
        -0.659906,
        1.0,
    ],
)
def test_a_feature_value_keeps_every_float64_digit(run_id, value):
    """A stored feature value is the value the machine read, to the last bit.

    Both halves of that: ``Decimal(str(float(x)))`` at the bind, and a column
    with no scale to truncate it afterwards. A scale counts decimal *places*
    while a float64 needs up to 17 *significant* digits, and the two only
    coincide above ~0.1 -- which is where every price in this repo lives and no
    feature does.
    """
    write_bar_reasons(
        run_id, Mode.REPLAY, [make_reason(1_785_720_000_000, features={"direction": value})]
    )

    [stored] = load_bar_reasons(run_id=run_id)
    assert stored.features["direction"] == value
    assert repr(stored.features["direction"]) == repr(value)


def test_a_non_finite_feature_value_is_refused(run_id):
    """Postgres NUMERIC accepts NaN without complaint and nothing matches it
    afterwards, including itself. A warmup row is ``None``; a NaN reaching here
    means something upstream stopped distinguishing the two."""
    with pytest.raises(ValueError, match="finite or None"):
        write_bar_reasons(
            run_id,
            Mode.REPLAY,
            [make_reason(1_785_720_000_000, features={"direction": float("nan")})],
        )


def test_features_round_trip_by_name_not_by_position(run_id):
    """The stored arrays are sorted, so a caller that built its dict in another
    order still reads its own values back under its own names."""
    features = {"strength": 0.25, "crowding": 0.75, "direction": -0.5}
    write_bar_reasons(
        run_id, Mode.REPLAY, [make_reason(1_785_720_000_000, features=features)]
    )

    [stored] = load_bar_reasons(run_id=run_id)
    assert stored.features == features
    assert stored.instrument == INSTRUMENT
    assert stored.timeframe == "4h"
    assert stored.state == "compression"


def test_writing_nothing_touches_no_connection():
    """An empty write must not open a transaction, so a caller can offer whatever
    it collected without first asking whether it collected anything."""
    assert write_bar_reasons(uuid.uuid4(), Mode.REPLAY, []) == 0


def test_the_insert_chunk_stays_under_the_bind_parameter_cap():
    """Postgres refuses a statement past 65535 bound parameters, and an array
    counts as one however many features it carries -- so the bound is a function
    of the column count alone, which is why it is derived rather than written
    down. A long replay writes one row per bar and reaches the cap easily."""
    assert MAX_ROWS_PER_INSERT * len(bar_reasons_table.c) <= MAX_BOUND_PARAMETERS
    assert MAX_ROWS_PER_INSERT >= 1


def test_a_stored_reason_reads_back_as_float_not_decimal(run_id):
    """NUMERIC arrives from the driver as ``Decimal``; every reader of a feature
    value computes in float64. Leaving it a Decimal makes the first comparison
    against a recomputed value a type error instead of a diff."""
    write_bar_reasons(
        run_id, Mode.REPLAY, [make_reason(1_785_720_000_000, features={"direction": 0.5})]
    )

    [stored] = load_bar_reasons(run_id=run_id)
    assert isinstance(stored.features["direction"], float)
    assert not isinstance(stored.features["direction"], Decimal)
