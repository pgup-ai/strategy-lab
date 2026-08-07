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

Three things follow, all of them measured.

**Each frame is bounded by its own funding span.** BTC/USDT perp candles begin
40 h before the venue's first settlement, so an unbounded request raises
``FundingUnavailable`` permanently and correctly -- and a board that asked for
"every dataset" would blank on the first instrument. The rule here is
``tests/test_state_machine_gate.py``'s and ``scripts/r7d``'s: open at the first
stored settlement, close at the last. The right bound costs at most one bar
(ETH's funding lands 4 h behind its newest candle) and is what keeps a candle
fetch that ran without funding from turning the whole board into refusals.
``as_of`` and ``dataset_last_bar`` are both on the row so that lag is readable
rather than inferred.

**A dataset that still cannot be answered reports why in its row.** A
``ValueError`` -- no candles in the funded window, funding that does not cover
it, a frame shorter than the strategy's warmup -- is *data about that
instrument*, not a failure of the board: one instrument's permanent leading gap
must not blank the other fifteen. Anything that is not a ``ValueError`` is a bug
and propagates, because a grey tile is a worse place to hide one than a
traceback.

**The cache is keyed on the bar, never on wall-clock time.** A full-history
recompute changes only when a bar closes or a settlement lands, so an entry
carries the stamp it was computed under -- the newest stored candle plus the
funding window -- and is served whenever that stamp still holds. At 4 h bars
that is valid for hours; a time-to-live would expire while the answer had not
moved. The stamp is what the two cheap probes below establish, and they are the
only queries a fully cached board issues: you cannot know a cache is valid
without asking whether a bar arrived. One entry per (dataset, strategy, exit
mode), replaced rather than added to, so the cache cannot grow with the
calendar. It is in-process and dies with ``browse``; **nothing here writes** --
no report directory, no ``signals`` row, no ``bar_reasons`` row, no schema.

The one input change the stamp cannot see is a settlement *rewritten in place*
between the span's own endpoints, which a re-fetch of the same range does not
produce and a corrected history would. `POST /api/refresh` moves the end and is
seen.
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

# The most any row will carry, and therefore the most a cached one holds. A slot
# is (dataset, strategy, exit mode), so a process someone flips through leaves
# one entry per combination -- bounded, but four figures of them once every exit
# mode has been tried, and 15,124 floats apiece would be the browser quietly
# holding the candle table in memory. ``models.BoardQuery`` takes its upper
# bound from here so the two cannot disagree about what is cached.
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
class BoardSlot:
    """What a cache entry is *for*: one pair, at one exit mode.

    Keyed on the four-part candle identity rather than on the instrument, for
    the reason ``CandleId`` exists: BTC 4h and BTC 1h are different datasets and
    a new 4h bar says nothing about the 1h one.
    """

    exchange: str
    market_type: str
    symbol: str
    timeframe: str
    strategy: str
    exit_mode: str | None


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
    closes: list[float]
    unavailable: str | None
    provenance: Provenance | None


@dataclass(frozen=True)
class DatasetRef:
    """A stored candle set and the newest bar in it, from the enumeration query."""

    identity: MarketDataIdentity
    last_bar: str


# One entry per slot, replaced when its stamp moves. In-process, and it dies
# with the server: a cache that outlived the process would be a second copy of
# the research record with no way to tell it had gone stale.
_CACHE: dict[BoardSlot, tuple[BoardStamp, BoardRow]] = {}


def clear_board_cache() -> None:
    """Forget everything held. Only a test should need this."""
    _CACHE.clear()


def stored_datasets(*, market_type: str | None = None) -> list[DatasetRef]:
    """Every stored candle set, on storage's own four-part identity.

    One aggregate query for the whole board. It is also the freshness probe:
    ``last_timestamp`` is what tells a cached row whether a bar has closed since
    it was computed, so enumerating and validating the cache are the same query
    rather than two.
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
        )
        for _, row in sets.iterrows()
    ]


def funding_window(identity: MarketDataIdentity) -> BoardWindow:
    """The bounds this frame should be asked for, from its own stored funding.

    Not a fallback and not a guess: on a perp the funded span *is* the window a
    funding-reading strategy can be run over, and outside it the coverage guard
    refuses -- correctly, since every unstored settlement would be charged as
    zero on an instrument where R2 measured carry at roughly the size of
    buy-and-hold. A perp with no stored funding at all gets the unbounded window
    and the refusal that follows, which names the fetch command; inventing a
    window there would hide the missing history rather than report it.
    """
    if identity.market_type != "perp":
        return BoardWindow(start=None, end=None)
    span = funding_span(
        exchange=identity.exchange,
        market_type=identity.market_type,
        symbol=identity.symbol,
    )
    if span is None:
        return BoardWindow(start=None, end=None)
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
    per uncached row, serially, and **parallelism does not fix it** -- measured
    on the three stored perp frames, four threads returned 1.10x, since the work
    is pandas and vectorbt under the GIL rather than waiting on anything. What a
    caller can have is the first row early, so the board fills in instead of
    arriving as one blob, and a fully cached board arrives at once.

    The funding probe is per *contract*, not per dataset: funding is keyed
    without a timeframe, so BTC 4h and BTC 1h share one window and one query.
    """
    windows: dict[tuple[str, str, str], BoardWindow] = {}
    for dataset in datasets:
        identity = dataset.identity
        contract = (identity.exchange, identity.market_type, identity.symbol)
        if contract not in windows:
            windows[contract] = funding_window(identity)
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
    identity = dataset.identity
    slot = BoardSlot(
        exchange=identity.exchange,
        market_type=identity.market_type,
        symbol=identity.symbol,
        timeframe=identity.timeframe,
        strategy=strategy,
        exit_mode=None if exit_mode is None else ExitMode(exit_mode).value,
    )
    cached = _CACHE.get(slot)
    if cached is not None and cached[0] == stamp:
        # Returned with the spark tail this request asked for. Everything else
        # is a function of the frame and does not depend on how much of it the
        # caller wants to draw.
        return replace(cached[1], closes=cached[1].closes[-spark_bars:])

    row = _compute(dataset, strategy=strategy, stamp=stamp, exit_mode=exit_mode)
    _CACHE[slot] = (stamp, row)
    return replace(row, closes=row.closes[-spark_bars:])


def _compute(
    dataset: DatasetRef,
    *,
    strategy: str,
    stamp: BoardStamp,
    exit_mode: ExitMode | None,
) -> BoardRow:
    """One analysis, sliced to the handful of values a tile shows.

    A ``ValueError`` becomes the row's ``unavailable`` and is cached like any
    other answer: "this frame is shorter than the strategy's warmup" is as much
    a function of the stored bars as a state is, and recomputing it every poll
    would make the cheapest rows the only uncached ones. Anything else
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
        closes=[float(bar["close"]) for bar in payload.bars[-MAX_SPARK_BARS:]],
        unavailable=None,
        provenance=payload.provenance,
    )


def _empty_row(dataset: DatasetRef, *, strategy: str, unavailable: str) -> BoardRow:
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
    "BoardRow",
    "BoardSlot",
    "BoardStamp",
    "BoardWindow",
    "DatasetRef",
    "clear_board_cache",
    "funding_window",
    "stored_datasets",
    "stream_board",
]
