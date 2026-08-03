"""Write and read the run/signal audit trail.

``signals`` is append-only (see :mod:`strategy_lab.storage.migrations`), so the
only supported mutation is an insert. Re-running a replay over a range that was
already recorded must therefore be free rather than an error: every write goes
through ``ON CONFLICT DO NOTHING`` on ``uq_signals_identity`` and reports how
many rows were genuinely new.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from strategy_lab.core.types import InstrumentId, Mode, Side, Signal
from strategy_lab.db.candles import get_engine
from strategy_lab.storage.schema import runs_table, signals_table

# Postgres' wire protocol caps a single statement at 65535 bound parameters, and
# a multi-row INSERT binds one per column per row. Exceeding it is not a slow
# path but an OperationalError, so a long replay flushing its signals in one
# call would simply crash. Deriving the chunk from the table's column count
# keeps the bound correct if a column is ever added; using every column rather
# than just the ones _to_row writes leaves headroom on purpose.
MAX_BOUND_PARAMETERS = 65535
MAX_ROWS_PER_INSERT = MAX_BOUND_PARAMETERS // len(signals_table.c)


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
    signals: Iterable[Signal],
    *,
    database_url: str | None = None,
) -> int:
    """Insert signals, skipping duplicates. Returns the count actually inserted.

    ``RETURNING`` after ``ON CONFLICT DO NOTHING`` emits a row only for a tuple
    that was really inserted, so the return value is the number of *new*
    signals, not the number offered. A resumed replay that re-emits a range it
    already stored therefore reports 0 and changes nothing.

    Every ``Signal.features`` value must be JSON-serialisable. A ``Decimal``
    there raises ``TypeError`` before anything is written -- deliberately, since
    the alternative is coercing it to a string that would come back as a string
    and quietly break the next comparison.

    Large batches are split across statements but share one transaction, so a
    failure part-way through rolls the whole call back.
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
) -> list[Signal]:
    """Read signals ordered by ``(ts_bar_ms, id)`` -- bar order, insertion order
    within a bar, so a reversal bar's exit and entry keep the order they were
    emitted in.

    Every filter is optional, but calling this with none of them selects the
    whole table into memory. ``signals`` is append-only and never pruned, so
    that set only ever grows; pass at least ``run_id`` outside of ad-hoc
    inspection.
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


def _to_row(run_id: uuid.UUID, mode: Mode, signal: Signal) -> dict:
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
    }


def _from_row(row) -> Signal:
    return Signal(
        instrument=InstrumentId(row["exchange"], row["market_type"], row["symbol"]),
        timeframe=row["timeframe"],
        strategy_id=row["strategy_id"],
        strategy_version=row["strategy_version"],
        ts_bar_ms=row["ts_bar_ms"],
        ts_emit_ms=row["ts_emit_ms"],
        side=Side(row["side"]),
        bar_is_closed=row["bar_is_closed"],
        reason=row["reason"],
        entry_price=_as_decimal(row["entry_price"]),
        stop_loss=_as_decimal(row["stop_loss"]),
        take_profit=_as_decimal(row["take_profit"]),
        strength=_as_decimal(row["strength"]),
        features=row["features"],
    )


def _as_decimal(value) -> Decimal | None:
    """Normalise a NUMERIC column to ``Decimal``.

    Under psycopg this is already a no-op -- NUMERIC arrives as ``Decimal``, so
    no test can distinguish this from returning the value untouched. It stays as
    a guard for the day a column changes type or another driver hands back a
    float or a string, and it routes through ``str`` because
    ``Decimal(float)`` would expose the binary expansion.
    """
    return None if value is None else Decimal(str(value))


__all__: Sequence[str] = [
    "MAX_BOUND_PARAMETERS",
    "MAX_ROWS_PER_INSERT",
    "create_run",
    "write_signals",
    "load_signals",
]
