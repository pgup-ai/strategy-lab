from __future__ import annotations

import re
import uuid

import pytest
from sqlalchemy import text

from strategy_lab.core.types import Mode, Side
from strategy_lab.db.candles import get_engine
from strategy_lab.storage.migrations import run_migrations

pytestmark = pytest.mark.db


def _insert_run(conn) -> uuid.UUID:
    run_id = uuid.uuid4()
    conn.execute(
        text(
            "INSERT INTO runs (run_id, mode, strategy_id, strategy_version, config) "
            "VALUES (:run_id, 'replay', 'turnaround_v2', '1.0.0', '{}'::jsonb)"
        ),
        {"run_id": run_id},
    )
    return run_id


def _insert_signal(conn, run_id, *, ts_bar_ms: int) -> None:
    conn.execute(
        text(
            "INSERT INTO signals (run_id, mode, strategy_id, strategy_version, exchange, "
            "market_type, symbol, timeframe, ts_bar_ms, ts_emit_ms, bar_is_closed, side, reason) "
            "VALUES (:run_id, 'replay', 'turnaround_v2', '1.0.0', 'binance', 'perp', "
            "'BTC/USDT', '15m', :ts, :ts, TRUE, 'enter_long', 'test')"
        ),
        {"run_id": run_id, "ts": ts_bar_ms},
    )


@pytest.mark.parametrize(
    ("constraint", "enum"),
    [("signals_side_check", Side), ("signals_mode_check", Mode)],
)
def test_check_constraints_match_the_core_enums(constraint, enum):
    """The SQL CHECK lists and the StrEnums in core.types are two copies of one
    vocabulary. Add a member to either alone and the engine emits a signal the
    database silently refuses -- at runtime, in live, on a real trade."""
    run_migrations()
    with get_engine().connect() as conn:
        definition = conn.execute(
            text(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conname = :n AND conrelid = 'signals'::regclass"
            ),
            {"n": constraint},
        ).scalar_one()
    allowed = set(re.findall(r"'([a-z_]+)'::text", definition))
    assert allowed == {member.value for member in enum}, (
        f"{constraint} allows {sorted(allowed)} but {enum.__name__} defines "
        f"{sorted(member.value for member in enum)}"
    )


def test_one_bar_can_emit_both_an_exit_and_an_opposite_entry():
    """turnaround_v1 wires long_exits = short_entries, so one bar emits two signals.

    Drop ``side`` from uq_signals_identity and the second insert collides with
    the first: the reversal bar silently loses half its audit trail.
    """
    run_migrations()
    ts = 1_785_723_500_000
    with get_engine().begin() as conn:
        run_id = _insert_run(conn)
        for side in ("exit_long", "enter_short"):
            conn.execute(
                text(
                    "INSERT INTO signals (run_id, mode, strategy_id, strategy_version, exchange, "
                    "market_type, symbol, timeframe, ts_bar_ms, ts_emit_ms, bar_is_closed, side, "
                    "reason) VALUES (:run_id, 'replay', 'turnaround_v1', '1.0.0', 'binance', "
                    "'perp', 'BTC/USDT', '15m', :ts, :ts, TRUE, :side, 'reversal')"
                ),
                {"run_id": run_id, "ts": ts, "side": side},
            )
        stored = conn.execute(
            text("SELECT side FROM signals WHERE run_id = :r ORDER BY side"), {"r": run_id}
        ).scalars().all()
    assert stored == ["enter_short", "exit_long"]


def test_appending_a_signal_is_never_blocked():
    """The trigger fires BEFORE UPDATE OR DELETE. If it ever catches INSERT the
    whole system stops recording, so assert the append path explicitly rather
    than relying on the other tests' inserts happening to work."""
    run_migrations()
    with get_engine().begin() as conn:
        run_id = _insert_run(conn)
        _insert_signal(conn, run_id, ts_bar_ms=1_785_723_600_000)
        count = conn.execute(
            text("SELECT count(*) FROM signals WHERE run_id = :r"), {"r": run_id}
        ).scalar_one()
    assert count == 1


def test_truncate_is_rejected():
    """TRUNCATE bypasses row-level triggers entirely.

    Without a statement-level guard, ``TRUNCATE signals`` erases the whole audit
    trail without firing trg_signals_append_only and without any warning -- and
    "let me just clear out the test rows" is a normal thing for someone to type.
    """
    run_migrations()
    engine = get_engine()
    with engine.begin() as conn:
        run_id = _insert_run(conn)
        _insert_signal(conn, run_id, ts_bar_ms=1_785_723_700_000)

    with pytest.raises(Exception, match="append-only"):
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE signals"))

    with engine.connect() as conn:
        survived = conn.execute(
            text("SELECT count(*) FROM signals WHERE run_id = :r"), {"r": run_id}
        ).scalar_one()
    assert survived == 1, "TRUNCATE was rejected but the row is gone anyway"


def test_signals_can_be_deleted_only_by_deliberately_disabling_the_triggers():
    """The documented escape hatch must actually work.

    Append-only is enforced, not absolute: rows accumulate forever, so a real
    cleanup path has to exist -- it just has to require obvious intent. This is
    the sequence documented in SIGNAL_MIGRATIONS and in the trigger's error hint.
    """
    run_migrations()
    engine = get_engine()
    with engine.begin() as conn:
        run_id = _insert_run(conn)
        _insert_signal(conn, run_id, ts_bar_ms=1_785_723_800_000)

    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE signals DISABLE TRIGGER USER"))
        conn.execute(text("DELETE FROM signals WHERE run_id = :r"), {"r": run_id})
        conn.execute(text("ALTER TABLE signals ENABLE TRIGGER USER"))

    with engine.connect() as conn:
        remaining = conn.execute(
            text("SELECT count(*) FROM signals WHERE run_id = :r"), {"r": run_id}
        ).scalar_one()
        enabled = conn.execute(
            text(
                "SELECT count(*) FROM pg_trigger WHERE tgrelid = 'signals'::regclass "
                "AND NOT tgisinternal AND tgenabled <> 'D'"
            )
        ).scalar_one()
    assert remaining == 0, "the deliberate cleanup path no longer works"
    assert enabled == 2, "both triggers must be left enabled after the cleanup"


def test_rerunning_migrations_does_not_recreate_either_trigger():
    """Idempotent must mean "no-op", not "harmlessly redoes the work".

    ``DROP TRIGGER`` + ``CREATE TRIGGER`` takes an ACCESS EXCLUSIVE lock on
    signals that Postgres holds until the migration transaction commits, so
    every later statement runs with signal writers blocked -- precisely when a
    live session is appending. Recreating a trigger allocates a new
    pg_trigger.oid, so unchanged oids prove both were left alone.
    """
    run_migrations()
    query = text(
        "SELECT tgname, oid FROM pg_trigger WHERE tgrelid = 'signals'::regclass "
        "AND NOT tgisinternal ORDER BY tgname"
    )
    with get_engine().connect() as conn:
        before = conn.execute(query).all()
    run_migrations()
    with get_engine().connect() as conn:
        after = conn.execute(query).all()

    assert len(before) == 2, f"expected both triggers, found {[r.tgname for r in before]}"
    assert before == after, (
        f"migrate recreated a trigger on a re-run ({before} -> {after}); both should "
        "be guarded so an already-correct trigger takes no lock at all"
    )


def test_signals_are_append_only():
    run_migrations()
    engine = get_engine()
    with engine.begin() as conn:
        run_id = _insert_run(conn)
        _insert_signal(conn, run_id, ts_bar_ms=1_785_723_400_000)

    with pytest.raises(Exception, match="append-only"):
        with engine.begin() as conn:
            conn.execute(text("UPDATE signals SET reason = 'tampered' WHERE run_id = :r"),
                         {"r": run_id})

    with pytest.raises(Exception, match="append-only"):
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM signals WHERE run_id = :r"), {"r": run_id})
