"""Storage for a continuously drifting target exposure.

Marked per test rather than with a module-level ``pytestmark``, unlike the other
storage suites: the four tests under the first heading below are pure Python and
must run whether or not Postgres is up. Three of them guard that a bare float
never reaches a NUMERIC bind, and a guard that skips silently is not a guard;
the fourth is arithmetic on the column count, which needs no server either.
"""

from __future__ import annotations

import copy
import uuid
from decimal import Decimal

import pytest
from sqlalchemy import text

from strategy_lab.core.types import InstrumentId, Mode, Side, Signal
from strategy_lab.db.candles import get_engine
from strategy_lab.storage.migrations import SIGNAL_MIGRATIONS, run_migrations
from strategy_lab.storage.schema import signals_table
from strategy_lab.storage.signals import (
    MAX_BOUND_PARAMETERS,
    MAX_ROWS_PER_INSERT,
    ExposureSignal,
    create_run,
    load_signals,
    write_signals,
)

INSTRUMENT = InstrumentId("binance", "perp", "BTC/USDT")

# The float64 immediately below 0.5499995 -- half a micro-unit under the
# charter's Exhaustion target of 0.55. Its shortest round-tripping repr needs 16
# significant digits, so Postgres' "%.15g" float8 -> numeric cast lands it
# exactly on the 6dp half-way boundary, which NUMERIC(10,6) then rounds away
# from zero. Measured through this insert path against Postgres 16.13:
#   bound as a Decimal -> 0.549999      (what the strategy decided)
#   bound as a float   -> 0.550000      (a level it never asked for)
# It is not an exotic value: the probe that found it disagreed on 4,817 of
# 12,844 float64s sampled near 6dp boundaries in -1..1.
TRUNCATING_TARGET = 0.5499994999999999
TARGET_AS_DECIMAL_BIND = Decimal("0.549999")
TARGET_AS_FLOAT_BIND = Decimal("0.550000")

# 21 significant digits -- more than float64 carries and more than the column
# stores, which is what makes it the one value that separates those two claims.
# It reaches the dataclass intact and the database rounded, and there is a test
# for each.
EXACT_21_DIGITS = Decimal("0.123456789012345678901")
STORED_AT_COLUMN_SCALE = Decimal("0.123457")


def make_signal(ts_bar_ms: int, side: Side = Side.ENTER_LONG, **overrides) -> Signal:
    fields = {
        "instrument": INSTRUMENT,
        "timeframe": "4h",
        "strategy_id": "state_machine_v2",
        "strategy_version": "1.0.0",
        "ts_bar_ms": ts_bar_ms,
        "ts_emit_ms": ts_bar_ms + 14_400_000,
        "side": side,
        "bar_is_closed": True,
        "reason": "riding",
        "entry_price": Decimal("63128.00"),
        # Explicit, because the column is NOT NULL DEFAULT '{}': leaving it None
        # would store {} and break the round-trip comparisons below on a
        # difference that has nothing to do with the target.
        "features": {},
    }
    return Signal(**(fields | overrides))


@pytest.fixture
def run_id():
    run_migrations()
    return create_run(
        run_id=uuid.uuid4(),
        mode=Mode.REPLAY,
        strategy_id="state_machine_v2",
        strategy_version="1.0.0",
        config={"source": "test"},
    )


# --- the conversion, which needs no database -------------------------------


def test_a_float_target_is_normalised_to_the_shortest_round_tripping_decimal():
    """The bind can only be as safe as the type, so the type does the coercion.

    A ``float`` left on the dataclass would reach psycopg as a float8 parameter
    -- SQLAlchemy's ``Numeric`` has no bind processor on a dialect with native
    decimal support -- and Postgres' implicit float8 -> numeric cast formats via
    "%.15g". Converting at construction means no call site can forget.
    """
    entry = ExposureSignal(make_signal(0), TRUNCATING_TARGET)
    assert isinstance(entry.target_exposure, Decimal), "a float survived onto the dataclass"
    assert entry.target_exposure == Decimal("0.5499994999999999")


def test_an_incoming_decimal_is_not_routed_through_float():
    """``db/funding.py``'s rule: an exact decimal must not detour via float64.

    This value needs 21 significant digits. Through ``float()`` it would come
    back as 0.12345678901234568 -- and the point of accepting a Decimal at all
    is that the caller already had digits worth keeping.

    In memory, which is as far as this claim goes: the column keeps six of
    those digits, and what that costs is pinned below.
    """
    assert ExposureSignal(make_signal(0), EXACT_21_DIGITS).target_exposure == EXACT_21_DIGITS


@pytest.mark.parametrize(
    "value",
    [pytest.param(float("nan"), id="float-nan"), pytest.param(Decimal("NaN"), id="decimal-nan")],
)
def test_a_nan_target_is_refused_rather_than_stored(value):
    """Postgres NUMERIC accepts NaN. Measured: it inserts without complaint and
    comes back as ``Decimal('NaN')``, which matches nothing -- not even itself
    -- so an audit row holding one is unfindable and unfalsifiable. Infinity
    needs no guard here; it overflows NUMERIC(10,6) and the server rejects it.
    """
    with pytest.raises(ValueError, match="finite"):
        ExposureSignal(make_signal(0), value)


def test_the_chunk_size_moved_with_the_new_column():
    """The bound-parameter cap is arithmetic on the column count, not a constant.

    A multi-row INSERT binds one parameter per column per row against Postgres'
    65535 cap. Before ``target_exposure`` that allowed 3276 rows per statement;
    with it, 3120. A hardcoded 3276 would still read as correct and would sit
    4% past the cap -- failing only on a replay long enough to fill a chunk.
    """
    assert "target_exposure" in signals_table.c
    assert MAX_ROWS_PER_INSERT == MAX_BOUND_PARAMETERS // len(signals_table.c)
    assert MAX_ROWS_PER_INSERT == 3120, "the cap did not move when the column was added"


def test_a_wrapped_signals_fields_are_readable_without_unwrapping_it():
    """The wrapper forwards, so a mixed result needs no isinstance at the call site.

    ``load_signals`` returns a union, and a caller that has to unwrap one arm of
    it before reading ``.side`` would be correct today only because nothing
    writes a level yet.
    """
    signal = make_signal(1_785_723_300_000, Side.ENTER_SHORT)
    wrapped = ExposureSignal(signal, Decimal("-0.550000"))

    assert wrapped.side is Side.ENTER_SHORT
    assert wrapped.ts_bar_ms == signal.ts_bar_ms
    assert wrapped.instrument.symbol == "BTC/USDT"
    assert wrapped.entry_price == signal.entry_price
    # Its own fields still win; forwarding fires only when lookup has failed.
    assert wrapped.target_exposure == Decimal("-0.550000")
    assert wrapped.signal is signal

    with pytest.raises(AttributeError):
        wrapped.no_such_field


def test_forwarding_a_missing_signal_stops_rather_than_looping():
    """``__getattr__`` must terminate when ``signal`` itself is what is missing.

    ``__new__`` without ``__init__`` is the half-built state itself, and it has
    to be built that way: on a *finished* instance ``self.signal`` resolves, so
    forwarding terminates whether or not the guard is there and a test written
    against one cannot fail. Without the guard every attribute here raises
    ``RecursionError`` instead.

    Copying does not reach this state, which is the part worth knowing:
    ``slots=True`` makes dataclasses generate a real ``__setstate__``, so the
    copy protocol finds one rather than probing for it. ``deepcopy`` is asserted
    below because it is what a reader expects the guard to be about -- it
    round-trips with the guard removed, and recurses once ``slots`` is dropped.
    """
    half_built = ExposureSignal.__new__(ExposureSignal)
    for name in ("side", "signal"):
        with pytest.raises(AttributeError):
            getattr(half_built, name)

    wrapped = ExposureSignal(make_signal(1_785_723_300_000), Decimal("0.550000"))
    duplicate = copy.deepcopy(wrapped)
    assert duplicate == wrapped
    assert duplicate.side is wrapped.side


# --- the column -------------------------------------------------------------


def _column_shape(conn, table: str):
    return conn.execute(
        text(
            "SELECT data_type, numeric_precision, numeric_scale, is_nullable, column_default "
            "FROM information_schema.columns WHERE table_schema = current_schema() "
            "AND table_name = :t AND column_name = 'target_exposure'"
        ),
        {"t": table},
    ).one_or_none()


@pytest.mark.db
def test_the_column_is_a_nullable_numeric_10_6():
    """Nullable and defaultless are both load-bearing. A NOT NULL would break
    every boolean strategy, and a DEFAULT would assert a level for rows whose
    strategy never held one.

    This asserts the state of the *deployed* database, which is why it is not
    on its own: a database that already has the column passes it no matter what
    the migration says. The scratch-table test below is what pins the statement.
    """
    run_migrations()
    with get_engine().connect() as conn:
        column = _column_shape(conn, "signals")

    assert column is not None, "run_migrations did not add signals.target_exposure"
    assert (column.data_type, column.numeric_precision, column.numeric_scale) == ("numeric", 10, 6)
    assert column.is_nullable == "YES"
    assert column.column_default is None


@pytest.mark.db
def test_the_migration_adds_a_nullable_column_without_rewriting_the_rows():
    """What the statement *does*, on a table that does not yet have the column.

    Two claims, both unobservable against the real ``signals`` now that it is
    migrated. First the shape it creates -- widen it to NUMERIC(38,18) or bolt
    on a NOT NULL DEFAULT and this fails while the test above still passes.
    Second that it is metadata-only: nullable with no default means the catalog
    gains a row and the heap is left alone, where a rewrite would hold an ACCESS
    EXCLUSIVE lock for the length of the table. A rewrite allocates a new
    relfilenode, so an unchanged one is the proof.
    """
    statements = [s for s in SIGNAL_MIGRATIONS if "target_exposure" in s]
    assert len(statements) == 1, f"expected one target_exposure migration, found {len(statements)}"

    scratch = "_target_exposure_probe"
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {scratch}"))
            conn.execute(text(f"CREATE TABLE {scratch} (id bigserial primary key, side text)"))
            for side in ("enter_long", "exit_long", "enter_short"):
                conn.execute(text(f"INSERT INTO {scratch} (side) VALUES (:s)"), {"s": side})
            before = conn.execute(text(f"SELECT pg_relation_filenode('{scratch}')")).scalar_one()

            conn.execute(text(statements[0].replace("signals", scratch)))

            after = conn.execute(text(f"SELECT pg_relation_filenode('{scratch}')")).scalar_one()
            surviving = conn.execute(text(f"SELECT count(*) FROM {scratch}")).scalar_one()
            column = _column_shape(conn, scratch)
    finally:
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {scratch}"))

    assert column is not None, "the migration statement added no target_exposure column"
    assert (column.data_type, column.numeric_precision, column.numeric_scale) == ("numeric", 10, 6)
    assert column.is_nullable == "YES"
    assert column.column_default is None
    assert before == after, "ADD COLUMN rewrote the table; it must be catalog-only"
    assert surviving == 3


@pytest.mark.db
def test_rerunning_the_migration_leaves_the_column_untouched():
    """Idempotent has to mean "no-op", not "harmlessly redoes the work".

    Dropping and re-adding the column would lose every stored level in silence.
    A re-added column gets a fresh attnum, and any rewrite a fresh relfilenode,
    so both being unchanged is the proof that the second run did nothing.
    """
    run_migrations()
    query = text(
        "SELECT a.attnum, a.atttypmod, pg_relation_filenode('signals') AS filenode, "
        "(SELECT count(*) FROM pg_attribute WHERE attrelid = 'signals'::regclass "
        " AND attnum > 0 AND NOT attisdropped) AS columns "
        "FROM pg_attribute a WHERE a.attrelid = 'signals'::regclass "
        "AND a.attname = 'target_exposure' AND NOT a.attisdropped"
    )
    with get_engine().connect() as conn:
        before = conn.execute(query).one()
    run_migrations()
    with get_engine().connect() as conn:
        after = conn.execute(query).one()

    assert before == after, f"the second migrate changed the column ({before} -> {after})"


# --- what gets stored -------------------------------------------------------


@pytest.mark.db
def test_a_boolean_signal_stores_null_and_loads_back_unchanged(run_id):
    """The whole point of a nullable column: the boolean path writes what it
    always wrote. A 0.0 here would read as "measured, and hold nothing", which
    is a claim four registered strategies never make.
    """
    signal = make_signal(1_785_723_300_000)
    assert write_signals(run_id, Mode.REPLAY, [signal]) == 1

    with get_engine().connect() as conn:
        stored = conn.execute(
            text("SELECT target_exposure FROM signals WHERE run_id = :r"), {"r": run_id}
        ).scalar_one()
    assert stored is None

    loaded = load_signals(run_id=run_id)
    assert loaded == [signal]
    assert type(loaded[0]) is Signal, "a boolean row came back claiming to carry a level"


@pytest.mark.db
@pytest.mark.parametrize(
    "target",
    [
        pytest.param(Decimal("1.000000"), id="fully-long"),
        pytest.param(Decimal("-1.000000"), id="fully-short"),
        pytest.param(Decimal("0.000001"), id="smallest-representable"),
        pytest.param(Decimal("0.123456"), id="six-decimal-places"),
        pytest.param(Decimal("0.550000"), id="the-exhaustion-taper"),
    ],
)
def test_a_target_round_trips_at_the_full_column_precision(run_id, target):
    """NUMERIC(10,6) holds six decimals either side of zero, and a taper that
    stores 0.55 and reads back 0.6 is a taper the audit trail cannot check.
    """
    entry = ExposureSignal(make_signal(1_785_723_300_000), target)
    assert write_signals(run_id, Mode.REPLAY, [entry]) == 1

    loaded = load_signals(run_id=run_id)
    assert loaded == [entry]
    assert loaded[0].target_exposure == target


@pytest.mark.db
def test_a_target_finer_than_the_column_comes_back_rounded_and_unequal(run_id):
    """Surviving in memory and surviving the column are separate claims.

    The pure-Python test above pins the first; this pins what the second costs.
    NUMERIC(10,6) rounds on write, so an ``ExposureSignal`` carrying more than
    six decimals is **not equal** to the one that comes back -- pinned here so
    that the obvious write-then-load equality assertion is a documented trap
    rather than a discovery.

    Not an argument for widening the column. A millionth of the risk budget is
    finer than any sizing decision this book makes, and widening a live NUMERIC
    is the operation that rewrote ~14,700 stored candles in this repo and needed
    a pg_dump restore.
    """
    entry = ExposureSignal(make_signal(1_785_723_300_000), EXACT_21_DIGITS)
    assert write_signals(run_id, Mode.REPLAY, [entry]) == 1

    loaded = load_signals(run_id=run_id)
    assert loaded[0].target_exposure == STORED_AT_COLUMN_SCALE
    assert loaded[0].target_exposure != entry.target_exposure
    assert loaded != [entry]


@pytest.mark.db
def test_a_float_target_is_bound_as_numeric_not_as_float8(run_id):
    """The failure this column exists to avoid, end to end.

    Earlier in this program the same implicit cast silently rewrote ~14,700
    stored candles and needed a pg_dump restore. Here it is quieter and no less
    wrong: a target of 0.5499994999999999 bound as a float stores as 0.550000,
    so the audit trail records a level the strategy never chose.

    Read back with raw SQL rather than through ``load_signals``, so the claim is
    about what the *column* holds. Going through the loader would put the same
    conversion on both sides of the assertion, and a conversion that is wrong in
    both directions still round-trips.
    """
    entry = ExposureSignal(make_signal(1_785_723_300_000), TRUNCATING_TARGET)
    write_signals(run_id, Mode.REPLAY, [entry])

    with get_engine().connect() as conn:
        stored = conn.execute(
            text("SELECT target_exposure FROM signals WHERE run_id = :r"), {"r": run_id}
        ).scalar_one()

    assert stored != TARGET_AS_FLOAT_BIND, "the float8 -> numeric cast rewrote the target"
    assert stored == TARGET_AS_DECIMAL_BIND


@pytest.mark.db
@pytest.mark.parametrize("exposure_first", [True, False], ids=["exposure-first", "boolean-first"])
def test_a_batch_mixing_both_contracts_writes_both(run_id, exposure_first):
    """A multi-row INSERT takes its column list from the first row offered, so
    the two orders are genuinely different statements. Both must work: a
    reversal bar can emit a boolean exit and a levelled entry together.
    """
    first, second = 1_785_723_300_000, 1_785_724_200_000
    boolean = make_signal(first, Side.EXIT_LONG)
    levelled = ExposureSignal(make_signal(second, Side.ENTER_SHORT), Decimal("-0.550000"))
    batch = [levelled, boolean] if exposure_first else [boolean, levelled]

    assert write_signals(run_id, Mode.REPLAY, batch) == 2
    loaded = load_signals(run_id=run_id)
    assert [type(item) for item in loaded] == [Signal, ExposureSignal]
    assert loaded[1].target_exposure == Decimal("-0.550000")

    # The union is readable without asking which arm each element is. This is
    # the shape every existing consumer already uses -- `[s.side for s in
    # load_signals(...)]` -- and it worked before only because nothing wrote a
    # level, so the first run that did would have raised AttributeError here.
    assert [item.side for item in loaded] == [Side.EXIT_LONG, Side.ENTER_SHORT]
    assert [item.ts_bar_ms for item in loaded] == [first, second]
    assert {item.instrument.symbol for item in loaded} == {"BTC/USDT"}


@pytest.mark.db
def test_re_offering_a_signal_within_a_run_is_free_and_never_rewrites_its_level(run_id):
    """``uq_signals_identity`` idempotency has to survive the new column.

    ``target_exposure`` is deliberately outside that constraint, which makes the
    second assertion the interesting one: a re-offer carrying a *different*
    level still writes nothing. ON CONFLICT DO NOTHING quietly becoming DO
    UPDATE would let a later pass overwrite a recorded decision, which is the
    one thing the append-only triggers exist to prevent.
    """
    ts = 1_785_723_300_000
    original = ExposureSignal(make_signal(ts), Decimal("0.550000"))
    assert write_signals(run_id, Mode.REPLAY, [original]) == 1
    assert write_signals(run_id, Mode.REPLAY, [original]) == 0

    revised = ExposureSignal(make_signal(ts), Decimal("0.200000"))
    assert write_signals(run_id, Mode.REPLAY, [revised]) == 0

    loaded = load_signals(run_id=run_id)
    assert len(loaded) == 1
    assert loaded[0].target_exposure == Decimal("0.550000")
