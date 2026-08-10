"""What state the market is in, and nothing about trading it.

The instrument view answers "what did this strategy do and why". This answers
only the second half, for the case where the question *is* the regime: candles,
volume, the machine's state per bar and the features behind it. No fills, no
trades, no cost model, no exit mode -- not hidden, absent, because none of them
were computed.

**It exists because the half nobody wants costs 92% of the time.** Measured on
BTC/USDT spot 4h, 18,842 bars: loading candles 181 ms, ``generate_signals``
25 ms, the state and all five features 45 ms, and
``vbt.Portfolio.from_signals`` **2,951 ms** of a 3,202 ms total. Watching one
symbol tolerates that; watching a dozen does not.

**It is a slice of the same derivation rather than a cheaper route to the same
answer.** ``prepare_frame`` and ``_why_layer`` are imported from
:mod:`api.analysis` rather than reimplemented, for the reason the board calls
``build_analysis`` instead of computing a tile the fast way: a second path is
free to drift, and the drift surfaces as a monitor quietly disagreeing with the
chart it links to.

**And it has to refuse a short frame itself.** Every other path reaches
``engine._warmup_bars``, which raises when warmup covers the frame. This one
never runs the engine, and without its own check the failure is silent and
specific: the machine answers on every bar, an unmeasurable row reads to it as
*failing*, and failing renders as ``COMPRESSION``. Measured on BTC/USDT spot 1d
with ``state_machine_v1`` -- 3,060 bars against a 2,192-bar warmup -- the
machine reports ``compression`` on **2,114 of the 2,192 warmup bars**. A view
whose whole purpose is "when is this chopping" would answer "chopping" over
exactly the range where it knows nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


from strategy_lab.api.analysis import (
    DatasetUnavailable,
    WhyLayer,
    _has_state,
    _why_layer,
    prepare_frame,
    resolve_strategy,
)
from strategy_lab.api.board import board_window
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.server import build_candles_payload


class StateUnavailable(ValueError):
    """This frame cannot carry a state, and the message says why.

    Separate from :class:`DatasetUnavailable`, which means there are no candles.
    Here there are candles and they are not enough -- a distinction the page
    needs, because one is fixed by fetching and the other never is.
    """


@dataclass(frozen=True)
class StateProvenance:
    """What was true of this reading. The analysis payload's provenance minus
    everything about execution, which is the whole point of this path.

    ``measurable_bars`` is the number a reader should judge the view by: a frame
    can be long enough to produce a state and still show only a handful of them.
    """

    identity: dict[str, str]
    strategy: str
    version: str
    warmup_bars: int
    bar_count: int
    measurable_bars: int
    first_bar: str
    measurable_from: str
    last_bar: str
    reads_crowding: bool
    crowding_measured: bool
    funding_attached: bool
    generated_at: str


@dataclass(frozen=True)
class StatePayload:
    bars: list[dict[str, float]]
    why: WhyLayer
    provenance: StateProvenance


def build_state(
    identity: MarketDataIdentity,
    *,
    strategy_name: str,
    start: str | None = None,
    end: str | None = None,
    funding: bool = True,
) -> StatePayload:
    """Candles and the machine's reading of them, over stored history."""
    resolved = resolve_strategy(strategy_name)
    strategy = resolved.strategy
    warmup = int(strategy.warmup_bars)
    # The same predicate `/api/strategies` publishes as `has_state`, so what the
    # page offers and what this accepts cannot drift apart.
    if not _has_state(strategy):
        raise StateUnavailable(
            f"{strategy_name} computes no feature frame, so it has no state to "
            f"show; pick a state machine"
        )

    # A perp with no bounds asked of it is asked over its whole stored candle
    # span, and a funding-reading machine is then refused wherever the stored
    # settlements fall short of that -- measured on the flagship dataset, BTC/USDT
    # perp 4h candles start 2019-09-08 16:00 against a first settlement at
    # 2019-09-10 08:00, so the state view 409'd on its own default view of it.
    # The instrument view never hit this because a tile hands it funded edges;
    # nothing hands the state view anything. Bounded by ``board_window``, which
    # is the board's own function rather than a second rule that could disagree
    # with it: perp-only, from stored funding, and ``WHOLE_HISTORY`` elsewhere.
    window = board_window(identity)
    prepared = prepare_frame(
        identity,
        strategy=strategy,
        start=start or window.start,
        end=end or window.end,
        funding=funding,
    )
    measurable = len(prepared.df) - warmup
    if measurable < 1:
        raise StateUnavailable(
            f"{strategy_name} needs {warmup:,} bars before its first state and "
            f"this frame has {len(prepared.df):,}. Every bar would be inside "
            f"warmup, where the machine reports compression because its inputs "
            f"are not measurable yet — which is not a claim about the market."
        )

    why, crowding_measured = _why_layer(strategy, prepared.df)
    return StatePayload(
        bars=build_candles_payload(prepared.df)["bars"],
        why=why,
        provenance=StateProvenance(
            identity={
                "exchange": identity.exchange,
                "market_type": identity.market_type,
                "symbol": identity.symbol,
                "timeframe": identity.timeframe,
            },
            strategy=strategy.name,
            version=strategy.version,
            warmup_bars=warmup,
            bar_count=len(prepared.df),
            measurable_bars=measurable,
            first_bar=str(prepared.df.index.min()),
            # The bar the ribbon starts on, resolved here rather than on the
            # page: an index into `bars` would make the page's rendering depend
            # on it having the same frame, which a refresh can change under it.
            measurable_from=str(prepared.df.index[warmup]),
            last_bar=str(prepared.df.index.max()),
            reads_crowding="crowding" in getattr(strategy, "features", ()),
            crowding_measured=crowding_measured,
            funding_attached=prepared.funding_attached,
            generated_at=datetime.now(UTC).isoformat(),
        ),
    )


__all__ = [
    "DatasetUnavailable",
    "StatePayload",
    "StateProvenance",
    "StateUnavailable",
    "build_state",
]
