"""Write and read the run/signal audit trail.

``signals`` is append-only (see :mod:`strategy_lab.storage.migrations`), so an
insert is the only supported mutation, and re-recording a range that was already
stored has to be free rather than an error: every write goes through ``ON
CONFLICT DO NOTHING`` on ``uq_signals_identity``.

One table serves both strategy contracts. A boolean strategy writes a
:class:`~strategy_lab.core.types.Signal` and leaves ``target_exposure`` NULL,
exactly as it did before that column existed; a continuous one writes an
:class:`ExposureSignal`, which is a signal plus the position *level* it asks
for. Which contract a stored row came from is therefore a type distinction on
the way back out, not a ``None`` a reader has to remember to check.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from strategy_lab.core.types import InstrumentId, Mode, Side, Signal
from strategy_lab.db.candles import get_engine
from strategy_lab.storage.schema import runs_table, signals_table

# Postgres caps a statement at 65535 bound parameters and a multi-row INSERT
# binds one per column per row, so an unchunked write of a long replay's signals
# raises OperationalError rather than merely running slowly. Deriving the chunk
# from the column count keeps the bound correct if a column is ever added -- as
# one now has been: `target_exposure` took this from 3276 rows to 3120, which a
# hardcoded 3276 would instead have moved 4% past the cap.
MAX_BOUND_PARAMETERS = 65535
MAX_ROWS_PER_INSERT = MAX_BOUND_PARAMETERS // len(signals_table.c)


@dataclass(frozen=True, slots=True)
class ExposureSignal:
    """A signal and the position *level* it asks the book to hold.

    ``core.types.Signal`` describes an event -- one of four sides -- and stays
    that way: the level belongs to one strategy contract out of two, so putting
    it on ``Signal`` would give every boolean strategy a field it can only ever
    leave ``None``. Pairing the two here instead is what lets the boolean path
    keep writing exactly the row it writes today.

    ``target_exposure`` is normalised to ``Decimal`` on construction, so no
    float can reach the ``NUMERIC(10,6)`` bind by any route -- see
    :func:`_to_numeric` for what that cast costs. ``Bar`` refuses a non-Decimal
    price rather than coercing one, and the difference is where the number comes
    from: an exchange sends a price as an exact decimal string, so a float there
    is a bug worth raising on, while a target is a float64 out of a pandas
    series by construction. Refusing it would only move ``Decimal(str(float(x)))``
    out to every call site, which is precisely where it gets forgotten.

    Reads of the wrapped signal's fields are **forwarded**, so a caller can
    iterate a mixed result from :func:`load_signals` and read ``.side`` or
    ``.ts_bar_ms`` off every element without first asking which contract wrote
    it. Without that the union is a trap rather than an answer, and one that
    hides until the first run that stores a level: the type is there for a
    caller who wants to know, and stays out of the way of one who does not.
    """

    signal: Signal
    target_exposure: float | Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_exposure", _to_numeric(self.target_exposure))

    def __getattr__(self, name: str):
        """Read the wrapped signal's fields as if they were this type's own.

        Runs only when ordinary lookup fails, so ``signal`` and
        ``target_exposure`` resolve normally, and a field added to either would
        win over the same name on ``Signal`` rather than being shadowed by it.

        The ``signal`` guard is the base case of an unbounded recursion, not a
        defensive check. On an instance whose fields are not assigned yet, any
        forwarded read asks for ``self.signal``, which is missing for the same
        reason, which forwards again. Measured on
        ``ExposureSignal.__new__(ExposureSignal)``: without the guard every
        attribute raises ``RecursionError``, with it every attribute raises
        ``AttributeError``.

        What does *not* reach that state, contrary to the obvious guess, is
        copying. ``slots=True`` makes dataclasses generate a real
        ``__setstate__``, so the copy protocol finds one by ordinary lookup
        rather than probing a half-built instance for it -- measured,
        ``deepcopy`` round-trips with the guard removed, and recurses as soon as
        ``slots`` is dropped. The guard is what keeps that keyword from being
        load-bearing at a distance.
        """
        if name == "signal":
            raise AttributeError(name)
        return getattr(self.signal, name)


def create_run(
    *,
    run_id: uuid.UUID,
    mode: Mode,
    strategy_id: str,
    strategy_version: str,
    config: dict,
    warmup_until_ts_ms: int | None = None,
    database_url: str | None = None,
) -> uuid.UUID:
    """Record the run header. Signals reference it, so this must happen first."""
    with get_engine(database_url).begin() as conn:
        conn.execute(
            insert(runs_table).values(
                run_id=run_id,
                mode=str(mode),
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                config=config,
                warmup_until_ts_ms=warmup_until_ts_ms,
            )
        )
    return run_id


def write_signals(
    run_id: uuid.UUID,
    mode: Mode,
    signals: Iterable[Signal | ExposureSignal],
    *,
    database_url: str | None = None,
) -> int:
    """Insert signals, skipping duplicates. Returns the count actually inserted.

    ``RETURNING`` after ``ON CONFLICT DO NOTHING`` emits a row only for a tuple
    that was really inserted, so a resumed replay re-offering a range it already
    stored reports 0 instead of claiming it discovered those signals again.
    ``target_exposure`` is deliberately outside ``uq_signals_identity``: a
    re-offer carrying a *different* level is still a duplicate and still writes
    nothing, because a stored decision a later pass can overwrite is not an
    audit trail.

    A ``Decimal`` in ``Signal.features`` raises ``TypeError`` before anything is
    written -- deliberately, since coercing it to a string would round-trip back
    as a string and break the next comparison silently. Chunks share one
    transaction, so a failure part-way through rolls the whole call back.
    """
    rows = [_to_row(run_id, mode, signal) for signal in signals]
    if not rows:
        return 0

    with get_engine(database_url).begin() as conn:
        inserted = 0
        for start in range(0, len(rows), MAX_ROWS_PER_INSERT):
            statement = (
                insert(signals_table)
                .values(rows[start : start + MAX_ROWS_PER_INSERT])
                .on_conflict_do_nothing(constraint="uq_signals_identity")
                .returning(signals_table.c.id)
            )
            inserted += len(conn.execute(statement).fetchall())
        return inserted


def load_signals(
    *,
    run_id: uuid.UUID | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    database_url: str | None = None,
) -> list[Signal | ExposureSignal]:
    """Read signals ordered by ``(ts_bar_ms, id)`` -- bar order, then insertion
    order within a bar, so a reversal bar's exit stays ahead of its entry.

    A row that stored a level comes back as the :class:`ExposureSignal` that
    stored it, and a row that did not comes back as a bare ``Signal``; see
    :func:`_from_row`.

    ``signals`` is append-only and never pruned, so calling this with no filters
    selects an ever-growing table into memory; pass at least ``run_id`` outside
    of ad-hoc inspection.
    """
    query = select(signals_table).order_by(signals_table.c.ts_bar_ms, signals_table.c.id)
    if run_id is not None:
        query = query.where(signals_table.c.run_id == run_id)
    if symbol is not None:
        query = query.where(signals_table.c.symbol == symbol)
    if timeframe is not None:
        query = query.where(signals_table.c.timeframe == timeframe)

    with get_engine(database_url).connect() as conn:
        return [_from_row(row) for row in conn.execute(query).mappings()]


def _to_numeric(value: float | Decimal) -> Decimal:
    """Convert a target exposure to the exact decimal the NUMERIC column holds.

    Two conventions already in this repo, one rule underneath: never let a bare
    ``float`` reach a NUMERIC bind. ``db/candles.py`` emits
    ``Decimal(str(float(x)))`` because its input is float64 out of pandas and
    ``str`` on a float is the shortest representation that round-trips;
    ``db/funding.py`` passes an incoming ``Decimal`` through untouched, because
    routing an exact decimal through float64 would discard digits the column can
    hold. A target arrives either way, so both branches live here.

    SQLAlchemy's ``Numeric`` has no bind processor on a dialect with native
    decimal support, so a float reaches psycopg as a float8 parameter and
    Postgres applies its implicit ``float8 -> numeric`` cast, which formats via
    "%.15g". A scale of 6 does not hide that. Measured through this exact insert
    path against Postgres 16.13, ``0.5499994999999999`` -- the float64
    immediately below the taper's 0.55 -- stores as ``0.550000`` bound as a
    float and ``0.549999`` bound as a Decimal: the cast rounds it onto the 6dp
    half-way boundary, which NUMERIC then rounds away from zero.

    NaN is refused rather than stored, because Postgres NUMERIC *accepts* it
    (measured: it comes back as ``Decimal('NaN')``, which no later comparison
    matches -- including a comparison with itself). It is the one non-finite
    value the column will not catch on its own; ``Infinity`` overflows
    NUMERIC(10,6) and the server rejects it.
    """
    numeric = value if isinstance(value, Decimal) else Decimal(str(float(value)))
    if not numeric.is_finite():
        raise ValueError(
            f"target_exposure must be finite, got {value!r}. Postgres NUMERIC "
            f"stores NaN without complaint, so an unguarded one becomes an audit "
            f"row that nothing will ever match."
        )
    return numeric


def _to_row(run_id: uuid.UUID, mode: Mode, entry: Signal | ExposureSignal) -> dict:
    is_exposure = isinstance(entry, ExposureSignal)
    signal = entry.signal if is_exposure else entry
    return {
        "run_id": run_id,
        "mode": str(mode),
        "strategy_id": signal.strategy_id,
        "strategy_version": signal.strategy_version,
        "exchange": signal.instrument.exchange,
        "market_type": signal.instrument.market_type,
        "symbol": signal.instrument.symbol,
        "timeframe": signal.timeframe,
        "ts_bar_ms": signal.ts_bar_ms,
        "ts_emit_ms": signal.ts_emit_ms,
        "bar_is_closed": signal.bar_is_closed,
        "side": str(signal.side),
        "strength": signal.strength,
        "entry_price": signal.entry_price,
        "stop_loss": signal.stop_loss,
        "take_profit": signal.take_profit,
        "reason": signal.reason,
        "features": signal.features or {},
        # Always present, NULL for a boolean signal. A multi-row INSERT takes
        # its column list from the first row offered, so a key that appears only
        # on some rows either raises or silently writes the wrong shape,
        # depending on which contract happened to be first in the batch.
        "target_exposure": entry.target_exposure if is_exposure else None,
    }


def _from_row(row) -> Signal | ExposureSignal:
    signal = Signal(
        instrument=InstrumentId(row["exchange"], row["market_type"], row["symbol"]),
        timeframe=row["timeframe"],
        strategy_id=row["strategy_id"],
        strategy_version=row["strategy_version"],
        ts_bar_ms=row["ts_bar_ms"],
        ts_emit_ms=row["ts_emit_ms"],
        side=Side(row["side"]),
        bar_is_closed=row["bar_is_closed"],
        reason=row["reason"],
        # NUMERIC columns arrive from psycopg as Decimal already.
        entry_price=row["entry_price"],
        stop_loss=row["stop_loss"],
        take_profit=row["take_profit"],
        strength=row["strength"],
        features=row["features"],
    )
    target = row["target_exposure"]
    # NULL is the boolean path's row, and the *type* is the answer: a caller
    # holding an ExposureSignal knows a level was stored, where a Signal with a
    # None field is something the caller can read straight past.
    return signal if target is None else ExposureSignal(signal, target)


__all__ = [
    "MAX_BOUND_PARAMETERS",
    "MAX_ROWS_PER_INSERT",
    "ExposureSignal",
    "create_run",
    "write_signals",
    "load_signals",
]
