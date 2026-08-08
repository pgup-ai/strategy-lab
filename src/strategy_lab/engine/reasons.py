"""The per-bar explanation both runners record, in one place.

``StrategyRunner`` and ``ExposureRunner`` drive different contracts but owe the
same row: what state the strategy was in on this bar and what the features behind
it were. Shared rather than written twice because the two copies would drift on
the first field added to ``BarReason``, and a live path whose explanation is
subtly different per contract is worse than one with none.
"""

from __future__ import annotations

import pandas as pd

from strategy_lab.core.clock import Clock
from strategy_lab.core.types import Bar, BarReason, InstrumentId


def reason_for(
    strategy,
    *,
    bar: Bar,
    frame: pd.DataFrame,
    instrument: InstrumentId,
    timeframe: str,
    clock: Clock,
) -> BarReason | None:
    """The state and feature values behind this bar, for a strategy that has them.

    Found by ``getattr(strategy, "feature_frame"/"machine")`` -- the same
    introspection ``api/analysis._why_layer`` uses, deliberately rather than
    matching a list of state machines, so a strategy that grows a
    ``feature_frame`` later is explained on every path without anyone editing
    them. A strategy with neither returns ``None`` and writes **no** rows: an
    absent explanation and an empty one are different claims, and the four
    original strategies genuinely have nothing to say here.

    Recomputed rather than read off the strategy's own return value, because
    neither ``SignalSet`` nor ``TargetExposure`` carries the state behind it. The
    two are still one computation in the sense that matters -- the same frame, the
    same feature functions, the same machine, and the last row of each -- and the
    only alternative is a contract change touching every strategy in the repo to
    serve the one path that cannot recompute.

    **A raise here is fatal to the bar, deliberately, and it costs a decision the
    strategy had already produced.** The strategy has returned by the time this
    runs, so catching would let the decision through and drop only its
    explanation. It is not caught because this calls the strategy's *own*
    ``feature_frame`` and ``machine``: for a state machine, a raise means it could
    not compute the state it trades on, and a book that does not know its own
    state should stop rather than trade on the half of the computation that
    happened to succeed. That is the rule ``Crowding`` already follows by raising
    instead of returning a neutral 0.5. A caller that wants decisions without
    explanations has ``record_reasons=False`` rather than a swallowed exception.
    **Worth revisiting when a live feed exists** -- losing a live decision to a
    diagnostic failure is a different trade-off from losing a replayed one, and
    these paths are replay-only today.
    """
    feature_frame = getattr(strategy, "feature_frame", None)
    machine = getattr(strategy, "machine", None)
    if feature_frame is None or machine is None:
        return None

    features, _ = feature_frame(frame)
    state = machine.run(features).iloc[-1]
    return BarReason(
        instrument=instrument,
        timeframe=timeframe,
        strategy_id=strategy.name,
        strategy_version=strategy.version,
        ts_bar_ms=bar.ts_open_ms,
        ts_emit_ms=clock.now_ms(),
        bar_is_closed=bar.is_closed,
        state=state.value,
        features={str(name): float(features[name].iloc[-1]) for name in features.columns},
    )


__all__ = ["reason_for"]
