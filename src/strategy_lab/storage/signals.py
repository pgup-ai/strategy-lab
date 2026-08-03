"""Write and read the run/signal audit trail.

``signals`` is append-only (see :mod:`strategy_lab.storage.migrations`), so an
insert is the only supported mutation, and re-recording a range that was already
stored has to be free rather than an error: every write goes through ``ON
CONFLICT DO NOTHING`` on ``uq_signals_identity``.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from strategy_lab.core.types import InstrumentId, Mode, Side, Signal
from strategy_lab.db.candles import get_engine
from strategy_lab.storage.schema import runs_table, signals_table

# Postgres caps a statement at 65535 bound parameters and a multi-row INSERT
# binds one per column per row, so an unchunked write of a long replay's signals
# raises OperationalError rather than merely running slowly. Deriving the chunk
# from the column count keeps the bound correct if a column is ever added.
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
    that was really inserted, so a resumed replay re-offering a range it already
    stored reports 0 instead of claiming it discovered those signals again.

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
) -> list[Signal]:
    """Read signals ordered by ``(ts_bar_ms, id)`` -- bar order, then insertion
    order within a bar, so a reversal bar's exit stays ahead of its entry.

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
        # NUMERIC columns arrive from psycopg as Decimal already.
        entry_price=row["entry_price"],
        stop_loss=row["stop_loss"],
        take_profit=row["take_profit"],
        strength=row["strength"],
        features=row["features"],
    )


__all__ = [
    "MAX_BOUND_PARAMETERS",
    "MAX_ROWS_PER_INSERT",
    "create_run",
    "write_signals",
    "load_signals",
]
