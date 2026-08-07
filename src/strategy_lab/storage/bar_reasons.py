"""Write and read the per-bar reason trail.

``bar_reasons`` is append-only through the same two triggers as ``signals`` (see
:mod:`strategy_lab.storage.migrations`), so an insert is the only supported
mutation and re-offering a range that was already stored has to be free rather
than an error: every write goes through ``ON CONFLICT DO NOTHING`` on
``uq_bar_reasons_identity``.

**Why this exists when the research path deliberately does not store it.**
``api/analysis._why_layer`` computes the same values per request and returns them
rather than persisting them, because a stored copy of a deterministic function of
immutable candles is a second record to drift. That reasoning holds. What it does
not cover is what a *live* run saw at the moment it decided -- funding that had
not settled yet, a feed gap, a bar that arrived late, a revision -- none of which
is recoverable from ``market_candles`` afterwards. So the research path keeps
recomputing, the event path persists, and the diff between the two is the gate.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from strategy_lab.core.types import BarReason, InstrumentId, Mode
from strategy_lab.db.candles import get_engine
from strategy_lab.storage.schema import bar_reasons_table

# The same cap and the same derivation as ``storage.signals``: Postgres binds one
# parameter per column per row and refuses a statement past 65535 of them. An
# array counts as one parameter however many features it carries, so the bound is
# a function of the column count alone -- which is why it is derived from the
# table rather than written down.
MAX_BOUND_PARAMETERS = 65535
MAX_ROWS_PER_INSERT = MAX_BOUND_PARAMETERS // len(bar_reasons_table.c)


def write_bar_reasons(
    run_id: uuid.UUID,
    mode: Mode,
    reasons: Iterable[BarReason],
    *,
    database_url: str | None = None,
) -> int:
    """Insert reason rows, skipping duplicates. Returns the count actually inserted.

    ``RETURNING`` after ``ON CONFLICT DO NOTHING`` emits a row only for a tuple
    that was really inserted, so a resumed replay re-offering a range it already
    stored reports 0 instead of claiming it discovered those bars again.

    The identity is ``(run_id, strategy, instrument, timeframe, bar)``. Within a
    run the first row for a bar wins and a later one writes nothing -- a stored
    record of what a run saw that a later pass can overwrite is not a record.
    *Across* runs there is no dedup at all: a second replay of the same range is
    expected to add a second run's worth of rows, exactly as ``signals`` does,
    because two runs saw the range at two different moments and that difference
    is the whole reason this table exists.

    Chunks share one transaction, so a failure part-way through rolls the whole
    call back.
    """
    rows = [_to_row(run_id, mode, reason) for reason in reasons]
    if not rows:
        return 0

    with get_engine(database_url).begin() as conn:
        inserted = 0
        for start in range(0, len(rows), MAX_ROWS_PER_INSERT):
            statement = (
                insert(bar_reasons_table)
                .values(rows[start : start + MAX_ROWS_PER_INSERT])
                .on_conflict_do_nothing(constraint="uq_bar_reasons_identity")
                .returning(bar_reasons_table.c.id)
            )
            inserted += len(conn.execute(statement).fetchall())
        return inserted


def load_bar_reasons(
    *,
    run_id: uuid.UUID | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    database_url: str | None = None,
) -> list[BarReason]:
    """Read reason rows in bar order, then insertion order within a bar.

    ``bar_reasons`` is append-only and never pruned, and it carries one row per
    bar rather than per decision -- measured on the R10a diff window, 6,048 rows
    against ``signals``' 325 over the same range -- so calling this with no
    filters selects an ever-growing table into memory. Pass at least ``run_id``
    outside of ad-hoc inspection.
    """
    query = select(bar_reasons_table).order_by(
        bar_reasons_table.c.ts_bar_ms, bar_reasons_table.c.id
    )
    if run_id is not None:
        query = query.where(bar_reasons_table.c.run_id == run_id)
    if symbol is not None:
        query = query.where(bar_reasons_table.c.symbol == symbol)
    if timeframe is not None:
        query = query.where(bar_reasons_table.c.timeframe == timeframe)

    with get_engine(database_url).connect() as conn:
        return [_from_row(row) for row in conn.execute(query).mappings()]


def _to_numeric(value: float | Decimal | None) -> Decimal | None:
    """One feature value as the exact decimal the NUMERIC element holds.

    ``None`` stays ``None`` and becomes SQL NULL, which is the only honest
    spelling of a warmup row: the feature convention is that an unmeasurable bar
    is ``NaN``, and ``0.0`` there would read as "measured, and neutral". NaN
    itself is refused for the reason ``storage.signals._to_numeric`` documents --
    Postgres NUMERIC *accepts* it and the stored value then matches nothing
    afterwards, including itself -- so a caller must have already mapped it to
    ``None``.

    ``Decimal(str(float(x)))`` is not decoration. Measured through this exact
    insert path against Postgres 16, the float64 ``88.02116722596503`` bound as a
    bare float inside the array stores as ``88.021167225965`` and bound as a
    ``Decimal`` round-trips: a list of Python floats reaches the server as
    ``float8[]`` and the implicit ``float8 -> numeric`` cast formats via "%.15g".

    The column is *unconstrained* ``NUMERIC`` for the other half of the same
    problem, which the R10a diff found rather than anyone predicting: a scale is
    a count of decimal *places*, ``str`` on a float gives up to 17 *significant*
    digits, and the two only coincide above ~0.1. A 1/480 rolling percentile is
    ``0.0020833333333333333`` and needs 19 places, so at ``NUMERIC(38,18)`` it
    came back one ULP out and the diff reported four features disagreeing with
    the research path on bars where nothing had actually disagreed.
    """
    if value is None:
        return None
    numeric = value if isinstance(value, Decimal) else Decimal(str(float(value)))
    if not numeric.is_finite():
        raise ValueError(
            f"feature values must be finite or None, got {value!r}. A warmup row "
            f"is None; Postgres NUMERIC stores NaN without complaint, so an "
            f"unguarded one becomes a stored value nothing will ever match."
        )
    return numeric


def _to_row(run_id: uuid.UUID, mode: Mode, reason: BarReason) -> dict:
    # Sorted so two rows for the same strategy carry their features in the same
    # order whatever a dict happened to be built in, which is what lets a reader
    # compare `feature_names` between rows instead of zipping every time.
    names = sorted(reason.features)
    return {
        "run_id": run_id,
        "mode": str(mode),
        "strategy_id": reason.strategy_id,
        "strategy_version": reason.strategy_version,
        "exchange": reason.instrument.exchange,
        "market_type": reason.instrument.market_type,
        "symbol": reason.instrument.symbol,
        "timeframe": reason.timeframe,
        "ts_bar_ms": reason.ts_bar_ms,
        "ts_emit_ms": reason.ts_emit_ms,
        "bar_is_closed": reason.bar_is_closed,
        "state": reason.state,
        "feature_names": names,
        "feature_values": [_to_numeric(reason.features[name]) for name in names],
    }


def _from_row(row) -> BarReason:
    return BarReason(
        instrument=InstrumentId(row["exchange"], row["market_type"], row["symbol"]),
        timeframe=row["timeframe"],
        strategy_id=row["strategy_id"],
        strategy_version=row["strategy_version"],
        ts_bar_ms=row["ts_bar_ms"],
        ts_emit_ms=row["ts_emit_ms"],
        bar_is_closed=row["bar_is_closed"],
        state=row["state"],
        # Back to float64, the dtype every reader of these values computes in --
        # the same crossing ``load_candles`` and ``BarBuffer`` make for prices,
        # in the one direction prices are never allowed to travel.
        features={
            name: None if value is None else float(value)
            for name, value in zip(row["feature_names"], row["feature_values"], strict=True)
        },
    )


__all__ = [
    "MAX_BOUND_PARAMETERS",
    "MAX_ROWS_PER_INSERT",
    "load_bar_reasons",
    "write_bar_reasons",
]
