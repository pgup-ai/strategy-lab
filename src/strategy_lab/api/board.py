"""Several instruments at once, by slicing the single view rather than redoing it.

**The board computes nothing of its own.** Every row is
:func:`api.analysis.build_analysis` over that dataset's own frame, reduced to the
handful of values a tile shows. The charter's standing rule is that the browser
is not free to disagree with a backtest -- which is why its markers are fills off
the engine's ``from_signals`` call rather than raw signals -- and a board that
derived "current state" by a cheaper route would be a **third** path free to
disagree with the other two. Measured, one warm analysis is 330-400 ms on the
stored perp frames against 5.5 s for the first one in a process; that is the
price of not having a second answer, and it is paid once per closed bar.

Four things follow, all of them measured.

**A frame's bound is shaped by its market type, and a perp's is its own funding
span.** BTC/USDT perp candles begin 40 h before the venue's first settlement, so
an unbounded request raises ``FundingUnavailable`` permanently and correctly --
and a board that asked for "every dataset" would blank on the first instrument.
The rule here is ``tests/test_state_machine_gate.py``'s and ``scripts/r7d``'s:
open at the first stored settlement, close at the last. The right bound costs at
most one bar (ETH's funding lands 4 h behind its newest candle) and is what keeps
a candle fetch that ran without funding from turning the whole board into
refusals. ``as_of`` and ``dataset_last_bar`` are both on the row so that lag is
readable rather than inferred.

**Anything that is not a perp settles nothing, so it is not asked about
settlements.** An equity runs to its newest stored bar and *no*
``funding_span`` query is issued for one -- not as an optimisation but because
asking is how a coverage guard gets invented for a market that has none, and a
window derived from an empty table would bound a frame by an absence. What can
go stale there is the other end entirely: the Yahoo fetcher rescales all history
by adjusted close, so a dividend **restates** past bars rather than appending to
them -- measured, 333 of 333 stored SPY weekly bars moved against a fresh fetch
(median 0.257%) and ``donchian`` differed on 3. ``last_written``
(``max(updated_at)``, one aggregate on the enumeration that already runs) is what
a tile shows for that, and it is a different claim from ``as_of`` rather than a
second spelling of it.

**A dataset that still cannot be answered reports why in its row.** A
``ValueError`` -- no candles in the funded window, funding that does not cover
it, a frame shorter than the strategy's warmup -- is *data about that
instrument*, not a failure of the board: one instrument's permanent leading gap
must not blank the other fifteen. Anything that is not a ``ValueError`` is a bug
and propagates, because a grey tile is a worse place to hide one than a
traceback.

**Every request recomputes, and that is the browser's contract rather than an
oversight.** An earlier version of this module memoised each row against the
newest stored candle and the funding window. Review killed it and was right to:
``POST /api/refresh`` upserts *overlapping* recent candles by design, so a
corrective refresh that rewrites the last few bars without adding one leaves
that stamp unmoved -- and a tile would then quietly contradict the chart it
links to, which is the single failure M36 exists to prevent. A cache whose
correctness depends on enumerating every writer is fine until one is missed.
What makes that affordable is the streaming below, not a memo: the first row
lands in tens of milliseconds and the rest fill in.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace

from strategy_lab.api.analysis import Marker, Provenance, build_analysis
from strategy_lab.backtests import ExitMode
from strategy_lab.db import list_candle_sets
from strategy_lab.db.funding import funding_span
from strategy_lab.market_data.base import MarketDataIdentity

# Enough tail to read a trend off a tile and far short of a chart. The full
# frame is 15,124 bars and 694 markers -- roughly 40x what a sparkline needs --
# which is the measurement that made the board a slice rather than a fan-out of
# /api/analysis.
DEFAULT_SPARK_BARS = 120

# The most any row will carry. The full frame is 15,124 floats, and a board is
# many rows at once, so the tail is bounded here rather than at the caller.
# ``models.BoardQuery`` takes its upper bound from here so the two cannot
# disagree about what a request may ask for.
MAX_SPARK_BARS = 500


@dataclass(frozen=True)
class BoardWindow:
    """The bounds a dataset's frame is asked for, and why it has them.

    ``None`` on both sides is the whole stored history, which is right for
    anything that is not a perp. On a perp both sides come from stored funding:
    the head because a leading gap can be permanent, the tail because a candle
    fetch that did not advance funding leaves the coverage guard refusing a
    frame the board was showing a moment ago.
    """

    start: str | None
    end: str | None


# The whole stored history, which is the honest window for an instrument that
# settles nothing. Named rather than spelled twice so the two callers below
# cannot drift into meaning different things by it.
WHOLE_HISTORY = BoardWindow(start=None, end=None)


@dataclass(frozen=True)
class BoardStamp:
    """What the answer depends on, as cheaply as it can be established.

    ``dataset_last_bar`` is the plan's ``last_bar_ts``; the window is here
    beside it because funding moves independently of candles and moving it
    changes the frame. Both are aggregate queries -- no candle row is read to
    build one.
    """

    dataset_last_bar: str
    window: BoardWindow


@dataclass(frozen=True)
class BoardRow:
    """One tile: what this strategy is doing on this instrument, as of which bar.

    ``unavailable`` and the rest are mutually exclusive -- a row either carries
    an answer or the reason there is none, and a reader never has to work out
    which by looking for nulls.
    """

    identity: dict[str, str]
    strategy: str
    contract: str | None
    state: str | None
    features: dict[str, float | None] | None
    latest_fill: Marker | None
    target: float | None
    as_of: str | None
    dataset_last_bar: str
    last_written: str
    closes: list[float]
    unavailable: str | None
    provenance: Provenance | None


@dataclass(frozen=True)
class DatasetRef:
    """A stored candle set, its newest bar, and when it was last written.

    All three come from the one enumeration query. ``last_bar`` and
    ``last_written`` answer different questions and only look alike on a venue
    whose history grows: a dividend-adjusted equity series is *restated* on a
    distribution, so its bars can all move without its newest one changing.
    """

    identity: MarketDataIdentity
    last_bar: str
    last_written: str


def stored_datasets(*, market_type: str | None = None) -> list[DatasetRef]:
    """Every stored candle set, on storage's own four-part identity.

    One aggregate query for the whole board, and it is also the freshness probe
    -- both halves of it. ``last_timestamp`` is the newest bar each set holds,
    which is what a perp tile shows beside its ``as of`` so a funding-bounded lag
    is visible rather than two; ``last_written`` is when those bars were last
    upserted, which is what an equity tile shows because its whole history is
    rewritten on a dividend rather than extended.
    """
    sets = list_candle_sets()
    if market_type is not None:
        sets = sets[sets["market_type"] == market_type]
    return [
        DatasetRef(
            identity=MarketDataIdentity(
                exchange=str(row["exchange"]),
                market_type=str(row["market_type"]),
                symbol=str(row["symbol"]),
                timeframe=str(row["timeframe"]),
            ),
            last_bar=str(row["last_timestamp"]),
            last_written=str(row["last_written"]),
        )
        for _, row in sets.iterrows()
    ]


def board_window(identity: MarketDataIdentity) -> BoardWindow:
    """The bounds this dataset's frame is asked for, chosen by its market type.

    The dispatch is the point. A perp is bounded by its own stored settlements
    (below); anything else runs to its newest stored bar and is **not asked
    about funding at all**. That second half is a claim rather than an
    optimisation: an equity settles nothing, so a ``funding_span`` query for one
    can only ever return ``None``, and a window derived from an empty table is a
    coverage guard invented for a market that has none. Asserted by statement in
    ``tests/test_api_board.py`` rather than left to the absence of an error,
    because the failure it prevents is silent.
    """
    if identity.market_type == "perp":
        return funding_window(identity)
    return WHOLE_HISTORY


def funding_window(identity: MarketDataIdentity) -> BoardWindow:
    """The bounds a *perp* frame should be asked for, from its own stored funding.

    Not a fallback and not a guess: on a perp the funded span *is* the window a
    funding-reading strategy can be run over, and outside it the coverage guard
    refuses -- correctly, since every unstored settlement would be charged as
    zero on an instrument where R2 measured carry at roughly the size of
    buy-and-hold. A perp with no stored funding at all gets the unbounded window
    and the refusal that follows, which names the fetch command; inventing a
    window there would hide the missing history rather than report it.

    Perp only, and it refuses anything else rather than answering
    ``WHOLE_HISTORY`` for it. Answering would make this look like a function of
    every market type while quietly querying ``funding_rates`` for instruments
    that have none -- the caller above is where the market type is decided.
    """
    if identity.market_type != "perp":
        raise ValueError(
            f"funding_window is for a perp, not {identity.market_type!r}: "
            f"{identity.exchange}/{identity.symbol} settles nothing, so a funding "
            f"span for it would be an empty table read as a window. Use "
            f"board_window, which dispatches on the market type."
        )
    span = funding_span(
        exchange=identity.exchange,
        market_type=identity.market_type,
        symbol=identity.symbol,
    )
    if span is None:
        return WHOLE_HISTORY
    return BoardWindow(start=str(span[0]), end=str(span[1]))


def stream_board(
    datasets: Sequence[DatasetRef],
    *,
    strategies: Sequence[str],
    exit_mode: ExitMode | None = None,
    spark_bars: int = DEFAULT_SPARK_BARS,
) -> Iterator[BoardRow]:
    """One row per (dataset, strategy), yielded as each is finished.

    A generator rather than a list because the total is what it is: 330-400 ms
    per row, serially, and **parallelism does not fix it** -- measured
    on the three stored perp frames, four threads returned 1.10x, since the work
    is pandas and vectorbt under the GIL rather than waiting on anything. What a
    caller can have is the first row early, so the board fills in instead of
    arriving as one blob.

    The funding probe is per *contract*, not per dataset: funding is keyed
    without a timeframe, so BTC 4h and BTC 1h share one window and one query.
    Datasets that are not perps memoise too, and memoise a query that was never
    issued -- ``board_window`` returns ``WHOLE_HISTORY`` for them without
    touching the database.
    """
    windows: dict[tuple[str, str, str], BoardWindow] = {}
    for dataset in datasets:
        identity = dataset.identity
        contract = (identity.exchange, identity.market_type, identity.symbol)
        if contract not in windows:
            windows[contract] = board_window(identity)
        stamp = BoardStamp(dataset_last_bar=dataset.last_bar, window=windows[contract])
        for strategy in strategies:
            yield _row(
                dataset,
                strategy=strategy,
                stamp=stamp,
                exit_mode=exit_mode,
                spark_bars=spark_bars,
            )


def _row(
    dataset: DatasetRef,
    *,
    strategy: str,
    stamp: BoardStamp,
    exit_mode: ExitMode | None,
    spark_bars: int,
) -> BoardRow:
    row = _compute(dataset, strategy=strategy, stamp=stamp, exit_mode=exit_mode)
    return replace(row, closes=row.closes[-spark_bars:])


def _compute(
    dataset: DatasetRef,
    *,
    strategy: str,
    stamp: BoardStamp,
    exit_mode: ExitMode | None,
) -> BoardRow:
    """One analysis, sliced to the handful of values a tile shows.

    A ``ValueError`` becomes the row's ``unavailable`` -- "this frame is shorter
    than the strategy's warmup" is a fact about the stored bars, and one
    instrument's refusal must not blank the rest of the board. Anything else
    propagates.
    """
    identity = dataset.identity
    try:
        payload = build_analysis(
            identity,
            strategy_name=strategy,
            exit_mode=exit_mode,
            start=stamp.window.start,
            end=stamp.window.end,
        )
    except ValueError as exc:
        return _empty_row(dataset, strategy=strategy, unavailable=str(exc))

    return BoardRow(
        identity=_identity(identity),
        strategy=strategy,
        contract=payload.provenance.contract,
        state=None if payload.why is None else payload.why.states[-1],
        features=(
            None
            if payload.why is None
            else {name: values[-1] for name, values in payload.why.features.items()}
        ),
        latest_fill=payload.markers[-1] if payload.markers else None,
        target=None if payload.target is None else payload.target[-1],
        as_of=payload.provenance.last_bar,
        dataset_last_bar=dataset.last_bar,
        last_written=dataset.last_written,
        closes=[float(bar["close"]) for bar in payload.bars[-MAX_SPARK_BARS:]],
        unavailable=None,
        provenance=payload.provenance,
    )


def _empty_row(dataset: DatasetRef, *, strategy: str, unavailable: str) -> BoardRow:
    """A refusal, plus the two facts about the *stored data* that survive it.

    ``dataset_last_bar`` and ``last_written`` come off the enumeration rather
    than off the run, so they are still true when there is no run: "this frame
    is shorter than the strategy's warmup" is a fact about bars that exist, and
    when they were last written is exactly what a reader needs in order to
    decide whether refreshing would change it.
    """
    return BoardRow(
        identity=_identity(dataset.identity),
        strategy=strategy,
        contract=None,
        state=None,
        features=None,
        latest_fill=None,
        target=None,
        as_of=None,
        dataset_last_bar=dataset.last_bar,
        last_written=dataset.last_written,
        closes=[],
        unavailable=unavailable,
        provenance=None,
    )


def _identity(identity: MarketDataIdentity) -> dict[str, str]:
    return {
        "exchange": identity.exchange,
        "market_type": identity.market_type,
        "symbol": identity.symbol,
        "timeframe": identity.timeframe,
    }


__all__ = [
    "DEFAULT_SPARK_BARS",
    "MAX_SPARK_BARS",
    "WHOLE_HISTORY",
    "BoardRow",
    "BoardStamp",
    "BoardWindow",
    "DatasetRef",
    "board_window",
    "funding_window",
    "stored_datasets",
    "stream_board",
]
