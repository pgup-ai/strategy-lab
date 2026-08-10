"""One request's worth of research: candles, what a strategy did, and why.

**The design move everything else follows from.** Signals are computed here, per
request, by the same whole-history vectorized call ``run_backtest`` makes over
the same stored candles -- never by joining to the ``signals`` table and never
from the event path. That is what makes this incapable of disagreeing with a
backtest, and it is the entire justification for the tool. The cost is that the
browser is not live in the tick sense; at 4h bars, poll-and-recompute is
indistinguishable from real time and is honest about what it is.

**Markers are fills, not signals.** ``from_signals`` ignores a repeated
same-direction entry under ``accumulate=False`` and does nothing with an exit
while the book is flat, so a payload built from the raw ``SignalSet`` would put
arrows on bars no backtest ever traded. The markers below come from the same
``pf.trades.records_readable`` frame ``backtests/report.py`` draws the frozen
report's markers from, and ``tests/test_api_analysis.py`` pins them to
``run_backtest``'s ``trades.csv`` on a real stored frame.

Reaching them means repeating the engine's ``from_signals`` call, which is the
one duplication in this module and is deliberate: the *decisions* around it --
warmup, exit-mode resolution, stop levels, entry sizing -- are the engine's own
functions rather than re-derived here, and the parity test is what holds the
call itself in line. Extracting the call would mean editing ``run_backtest``,
whose four original strategies have byte-identical results of record.

Nothing here writes anything. No report directory, no ``signals`` row, no
migration -- the "why" layer is computed on every bar by both existing paths and
discarded by both, and returning it rather than storing it is what keeps this
tool unable to add a new way for the research record to be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from strategy_lab.backtests import ExitMode
from strategy_lab.backtests.funding_frame import with_funding_column
from strategy_lab.backtests.sizing import DEFAULT_VOL_SPAN, SizeMode
from strategy_lab.db import load_candles
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.server import build_candles_payload
from strategy_lab.strategies.base import SignalSet, Strategy
from strategy_lab.strategies.exposure import ExposureStrategy
from strategy_lab.strategies.exposure_registry import (
    get_exposure_strategy,
    list_exposure_strategies,
)
from strategy_lab.strategies.registry import get_strategy, list_strategies
from strategy_lab.timeframes import timeframe_to_pandas_freq

DEFAULT_FEES = 0.0005
DEFAULT_SLIPPAGE = 0.0005
DEFAULT_CASH = 10_000.0
DEFAULT_POSITION_PCT = 0.95
DEFAULT_FAILURE_BARS = 4

# The feature that cannot be computed without a funding column. A strategy
# reading it on a perp needs the attachment to succeed rather than to fall back,
# which is the difference between R5's published figures and a neighbouring set.
FUNDING_DERIVED_FEATURE = "crowding"


class DatasetUnavailable(ValueError):
    """No stored candles for the requested identity and window."""


class Contract(str, Enum):
    """Which strategy contract answered the request.

    Not cosmetic: it decides whether the payload carries markers or a signed
    level, and a caller has to know before it draws anything.
    """

    SIGNAL_SET = "signal_set"
    TARGET_EXPOSURE = "target_exposure"


@dataclass(frozen=True)
class StrategyInfo:
    """What a caller needs to pick a strategy without loading it.

    ``has_state`` is what ``/api/state`` can answer for, and it is introspected
    from ``feature_frame`` rather than matched against a list of names -- the
    same rule ``_why_layer`` follows, so a strategy that grows one is offered
    without anyone editing a second place.
    """

    name: str
    contract: str
    version: str
    warmup_bars: int
    has_state: bool


@dataclass(frozen=True)
class ResolvedStrategy:
    strategy: Strategy | ExposureStrategy
    contract: Contract


@dataclass(frozen=True)
class PreparedFrame:
    """The frame a strategy should see, and what had to be true to build it.

    ``funding`` is the settlement series the cost ledger charges, on the venue's
    own stamps; ``funding_attached`` says whether the per-bar column reached the
    frame. They move together, and both are here so a caller handing this frame
    to ``run_backtest`` gives it exactly what this module gave the strategy.
    """

    df: pd.DataFrame
    funding: pd.Series | None
    funding_attached: bool


@dataclass(frozen=True)
class Marker:
    """One fill: when, which way, at what price, and how much.

    ``size`` is the trade's quantity, and it is here because it is the only part
    of a fill the cost model visibly moves. Measured on the 2022 donchian window,
    raising the fee rate from 0.0005 to 0.004 changes the entry price by 1.2e-16
    -- a fee is not part of the price -- and the quantity by up to 14%, because
    the cash left to deploy is smaller. A marker without it would describe two
    materially different books identically.
    """

    time: int
    kind: str
    side: str
    price: float
    size: float


@dataclass(frozen=True)
class Trade:
    """A round trip and what it made, sliced from the same ``Portfolio`` the
    arrows come from.

    The engine already computes this and ``_markers`` throws all but the
    timestamps away. Recomputing PnL from the markers would be a second
    derivation free to disagree with the chart it sits under -- the board's rule
    (M36), applied to the one number a reader is most likely to quote.

    **``pnl`` on an open trade is a mark, not money**, so ``status`` travels with
    it and the two are summed separately. vectorbt reports an open trade against
    the last close, which moves on the next bar and on the range you asked for;
    adding that into a realised total would report an unrealised number under a
    realised name.
    """

    entry_time: int
    exit_time: int | None
    direction: str
    size: float
    entry_price: float
    exit_price: float | None
    fees: float
    pnl: float
    return_pct: float
    status: str


@dataclass(frozen=True)
class WhyLayer:
    """The state a strategy was in on each bar and the feature values behind it.

    Both existing paths compute this and throw it away. It is returned rather
    than persisted: no schema change, no migration, and no second copy of the
    research record to fall out of step with the first.
    """

    states: list[str]
    features: dict[str, list[float | None]]


@dataclass(frozen=True)
class Provenance:
    """What was true of this run, carried in every response.

    M20 is the reason it is not a detail panel: two runs of one strategy differed
    because one had the funding column, and the number that moved was published
    before anyone noticed. A figure shown without this context will eventually
    contradict the charter with no way to see why.
    """

    identity: dict[str, str]
    strategy: str
    version: str
    contract: str
    exit_mode: str | None
    failure_bars: int | None
    warmup_bars: int
    allow_shorts: bool
    # A pair. Read either alone and it says the wrong thing -- see ``_provenance``.
    reads_crowding: bool
    crowding_measured: bool
    funding_attached: bool
    cost_model: dict[str, float] | None
    first_bar: str
    last_bar: str
    bar_count: int
    generated_at: str


@dataclass(frozen=True)
class AnalysisPayload:
    bars: list[dict[str, float]]
    markers: list[Marker]
    trades: list[Trade]
    position_size: list[float | None] | None
    target: list[float | None] | None
    why: WhyLayer | None
    provenance: Provenance


def _has_state(strategy: Strategy | ExposureStrategy) -> bool:
    """Whether this strategy has a state to show, on ``_why_layer``'s own test."""
    return (
        getattr(strategy, "feature_frame", None) is not None
        and getattr(strategy, "machine", None) is not None
    )


def registered_strategies() -> list[StrategyInfo]:
    """Every strategy either registry knows, labelled by the contract it answers on."""
    entries = [
        StrategyInfo(
            name=name,
            contract=Contract.SIGNAL_SET.value,
            version=get_strategy(name).version,
            warmup_bars=int(get_strategy(name).warmup_bars),
            has_state=_has_state(get_strategy(name)),
        )
        for name in list_strategies()
    ]
    entries.extend(
        StrategyInfo(
            name=name,
            contract=Contract.TARGET_EXPOSURE.value,
            version=get_exposure_strategy(name).version,
            warmup_bars=int(get_exposure_strategy(name).warmup_bars),
            has_state=_has_state(get_exposure_strategy(name)),
        )
        for name in list_exposure_strategies()
    )
    return entries


def resolve_strategy(name: str, *, allow_shorts: bool = True) -> ResolvedStrategy:
    """The strategy object and its contract, from whichever registry holds it.

    The two registries are disjoint by construction -- a strategy implements one
    contract or the other -- so which one answers is what tells this module
    whether to call ``generate_signals`` or ``compute_target``.
    """
    if name in list_strategies():
        return ResolvedStrategy(
            strategy=get_strategy(name, allow_shorts=allow_shorts),
            contract=Contract.SIGNAL_SET,
        )
    if name in list_exposure_strategies():
        return ResolvedStrategy(
            strategy=get_exposure_strategy(name, allow_shorts=allow_shorts),
            contract=Contract.TARGET_EXPOSURE,
        )
    available = ", ".join(sorted(set(list_strategies()) | set(list_exposure_strategies())))
    raise ValueError(f"Unknown strategy {name!r}. Available: {available}")


def prepare_frame(
    identity: MarketDataIdentity,
    *,
    strategy: Strategy | ExposureStrategy,
    start: str | None = None,
    end: str | None = None,
    funding: bool = True,
) -> PreparedFrame:
    """Stored candles, with the funding column a perp strategy reads.

    The attachment rule is ``backtests/funding_frame.with_funding_column`` -- the
    same one ``backtest`` and ``sweep`` use, perp only, matched by containment
    rather than by a reindex. Whether missing funding is fatal follows the
    sweep's precedent: only a strategy that actually reads a funding-derived
    feature is refused, so browsing ``donchian`` over an unfunded perp keeps
    working, and the payload says the column was absent either way.
    """
    df = load_candles(
        exchange=identity.exchange,
        market_type=identity.market_type,
        symbol=identity.symbol,
        timeframe=identity.timeframe,
        start=start,
        end=end,
    )
    if df.empty:
        raise DatasetUnavailable(
            f"No candles stored for {identity.exchange}/{identity.market_type}/"
            f"{identity.symbol}/{identity.timeframe}"
            + (f" over {start} -> {end}" if start or end else "")
        )

    df = df.sort_index()
    needs_funding = FUNDING_DERIVED_FEATURE in getattr(strategy, "features", ())
    df, rates = with_funding_column(
        identity, df, enabled=funding, required=needs_funding
    )
    return PreparedFrame(df=df, funding=rates, funding_attached=rates is not None)


def build_analysis(
    identity: MarketDataIdentity,
    *,
    strategy_name: str,
    exit_mode: ExitMode | str | None = None,
    start: str | None = None,
    end: str | None = None,
    failure_bars: int = DEFAULT_FAILURE_BARS,
    fees: float = DEFAULT_FEES,
    slippage: float = DEFAULT_SLIPPAGE,
    cash: float = DEFAULT_CASH,
    position_pct: float = DEFAULT_POSITION_PCT,
    funding: bool = True,
    allow_shorts: bool = True,
) -> AnalysisPayload:
    """Everything one view of one strategy over one candle set needs.

    ``exit_mode`` is optional and resolves to the engine's own default on the
    boolean contract, recorded in the provenance block so the reader is never
    left guessing which mode produced the arrows. On the continuous contract
    there is no exit mode at all -- a target of 0.0 *is* the exit -- so one
    passed there is refused rather than accepted and dropped.

    ``failure_bars`` is a parameter rather than a constant for the same reason:
    it decides how many adverse closes ``continuation_failure`` and
    ``trend_structure`` exit on, so pinning it here would show one exit rule
    while a backtest at another setting shows a different one.
    """
    resolved = resolve_strategy(strategy_name, allow_shorts=allow_shorts)
    if resolved.contract is Contract.TARGET_EXPOSURE and exit_mode is not None:
        raise ValueError(
            f"{strategy_name} runs on the continuous-exposure contract, which has "
            f"no exit mode: a target of 0.0 is the exit. Accepting {exit_mode!r} "
            f"here would label the view with a setting that changed nothing."
        )
    prepared = prepare_frame(
        identity, strategy=resolved.strategy, start=start, end=end, funding=funding
    )
    if resolved.contract is Contract.TARGET_EXPOSURE:
        return _exposure_payload(identity, resolved, prepared, allow_shorts=allow_shorts)

    mode = ExitMode.CONTINUATION_FAILURE if exit_mode is None else ExitMode(exit_mode)
    return _signal_payload(
        identity,
        resolved,
        prepared,
        exit_mode=mode,
        failure_bars=failure_bars,
        fees=fees,
        slippage=slippage,
        cash=cash,
        position_pct=position_pct,
        allow_shorts=allow_shorts,
    )


def _signal_payload(
    identity: MarketDataIdentity,
    resolved: ResolvedStrategy,
    prepared: PreparedFrame,
    *,
    exit_mode: ExitMode,
    failure_bars: int,
    fees: float,
    slippage: float,
    cash: float,
    position_pct: float,
    allow_shorts: bool,
) -> AnalysisPayload:
    strategy, df = resolved.strategy, prepared.df
    signals = strategy.generate_signals(df)
    trades = _fills(
        df,
        strategy=strategy,
        signals=signals,
        identity=identity,
        exit_mode=exit_mode,
        failure_bars=failure_bars,
        fees=fees,
        slippage=slippage,
        cash=cash,
        position_pct=position_pct,
    )
    return AnalysisPayload(
        bars=build_candles_payload(df)["bars"],
        markers=_markers(trades),
        trades=_trades(trades),
        position_size=_values(signals.position_size),
        target=None,
        why=_why_layer(strategy, df)[0],
        provenance=_provenance(
            identity,
            resolved,
            prepared,
            metadata=signals.metadata,
            exit_mode=exit_mode.value,
            failure_bars=failure_bars,
            allow_shorts=allow_shorts,
            cost_model={
                "fee": fees,
                "slippage": slippage,
                "cash": cash,
                "position_pct": position_pct,
            },
        ),
    )


def _exposure_payload(
    identity: MarketDataIdentity,
    resolved: ResolvedStrategy,
    prepared: PreparedFrame,
    *,
    allow_shorts: bool,
) -> AnalysisPayload:
    from strategy_lab.backtests.exposure_engine import _flat_through_warmup

    strategy, df = resolved.strategy, prepared.df
    exposure = strategy.compute_target(df)
    # Flattened by the engine's own function, not returned raw. A strategy may
    # emit a non-zero target inside its declared warmup -- measured,
    # ``state_machine_v2`` does so on 62 of its 847 warmup rows -- and
    # ``run_exposure_backtest`` zeroes those before executing anything. Drawing
    # the raw target would put exposure on the page over bars a backtest holds
    # nothing on, which is the continuous contract's version of marking signals
    # instead of fills.
    target = _flat_through_warmup(
        exposure.target, warmup_bars=strategy.warmup_bars, strategy=strategy
    )
    return AnalysisPayload(
        bars=build_candles_payload(df)["bars"],
        markers=[],
        trades=[],
        position_size=None,
        target=_values(target),
        why=_why_layer(strategy, df)[0],
        provenance=_provenance(
            identity,
            resolved,
            prepared,
            metadata=exposure.metadata,
            exit_mode=None,
            failure_bars=None,
            allow_shorts=allow_shorts,
            # Nothing was executed: this is the level the strategy asked for, and
            # a fee rate reported beside it would claim a book that never ran.
            # Which is also why the trade table above is empty rather than
            # missing -- ``run_exposure_backtest`` would produce one, and this
            # path deliberately does not call it.
            cost_model=None,
        ),
    )


def _fills(
    df: pd.DataFrame,
    *,
    strategy: Strategy,
    signals: SignalSet,
    identity: MarketDataIdentity,
    exit_mode: ExitMode,
    failure_bars: int,
    fees: float,
    slippage: float,
    cash: float,
    position_pct: float,
) -> pd.DataFrame:
    """The trades ``run_backtest`` would fill on this frame at this exit mode.

    Every decision here is the engine's own function, imported rather than
    re-derived -- warmup, the exit-mode ingredients, the stop levels and the
    entry sizes. Only the ``from_signals`` call is repeated, and
    ``tests/test_api_analysis.py`` compares its trades against ``trades.csv``
    from a real ``run_backtest`` so the repetition cannot drift unnoticed.

    Sizing is fixed-fractional, matching the engine's default. There is no
    volatility-scaling knob on this path: that mode masks a further 20x its span
    of bars, so offering it here would quietly blank the front of a chart.
    """
    import vectorbt as vbt

    from strategy_lab.backtests.engine import (
        _compute_entry_sizes,
        _exit_signals,
        _mask_warmup,
        _stop_kwargs,
        _warmup_bars,
    )

    warmup = _warmup_bars(strategy, df, size_mode=SizeMode.FIXED, vol_span=DEFAULT_VOL_SPAN)
    long_exits, short_exits = _exit_signals(
        df=df, signals=signals, exit_mode=exit_mode, failure_bars=failure_bars
    )
    stop_kwargs = _stop_kwargs(df, signals.setup_stop_loss, exit_mode)
    signals, long_exits, short_exits = _mask_warmup(signals, long_exits, short_exits, warmup)
    size = _compute_entry_sizes(
        long_entries=signals.long_entries,
        short_entries=signals.short_entries,
        close=df["close"],
        cash=cash,
        position_pct=position_pct,
        position_scale=signals.position_size,
    )

    portfolio = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=signals.long_entries,
        exits=long_exits,
        short_entries=signals.short_entries,
        short_exits=short_exits,
        size=size,
        init_cash=cash,
        fees=fees,
        slippage=slippage,
        freq=timeframe_to_pandas_freq(identity.timeframe),
        **stop_kwargs,
    )
    return portfolio.trades.records_readable


def _markers(trades: pd.DataFrame) -> list[Marker]:
    """One arrow per fill, in the shape ``backtests/report.py`` already draws.

    An open trade contributes an entry and no exit: it has not left the book, so
    marking one would draw a fill that has not happened.
    """
    markers: list[Marker] = []
    for _, trade in trades.iterrows():
        side = "long" if trade["Direction"] == "Long" else "short"
        size = float(trade["Size"])
        markers.append(
            Marker(
                time=_epoch(trade["Entry Timestamp"]),
                kind="entry",
                side=side,
                price=float(trade["Avg Entry Price"]),
                size=size,
            )
        )
        if trade["Status"] == "Closed":
            markers.append(
                Marker(
                    time=_epoch(trade["Exit Timestamp"]),
                    kind="exit",
                    side=side,
                    price=float(trade["Avg Exit Price"]),
                    size=size,
                )
            )
    markers.sort(key=lambda marker: (marker.time, marker.kind))
    return markers


def _trades(trades: pd.DataFrame) -> list[Trade]:
    """The same round trips the arrows are drawn from, carrying their result."""
    rows: list[Trade] = []
    for _, trade in trades.iterrows():
        closed = trade["Status"] == "Closed"
        rows.append(
            Trade(
                entry_time=_epoch(trade["Entry Timestamp"]),
                exit_time=_epoch(trade["Exit Timestamp"]) if closed else None,
                direction="long" if trade["Direction"] == "Long" else "short",
                size=float(trade["Size"]),
                entry_price=float(trade["Avg Entry Price"]),
                exit_price=float(trade["Avg Exit Price"]) if closed else None,
                # Both legs, so the table's PnL column visibly nets what the cost
                # model charged rather than leaving a reader to infer it.
                fees=float(trade["Entry Fees"]) + float(trade["Exit Fees"]),
                pnl=float(trade["PnL"]),
                return_pct=float(trade["Return"]),
                status="closed" if closed else "open",
            )
        )
    return rows


def _why_layer(
    strategy: Strategy | ExposureStrategy, df: pd.DataFrame
) -> tuple[WhyLayer | None, bool]:
    """Per-bar state and features, and whether crowding in them is real.

    Introspected rather than matched against a list of state machines, following
    ``sweep_command``'s ``getattr(strategy, "features", ())``: a strategy that
    grows a ``feature_frame`` later is explained without anyone remembering to
    edit this function, and one that has no state to explain gets nothing rather
    than an empty shell that reads as an absence of state.

    The flag rides along because ``build_feature_frame`` returns it here and the
    signal paths get the same value from their run metadata. ``build_state`` has
    no run, and inferring it from the values -- a constant crowding column means
    the neutral fallback -- would be a second derivation of a fact this call
    already holds.
    """
    feature_frame = getattr(strategy, "feature_frame", None)
    machine = getattr(strategy, "machine", None)
    if feature_frame is None or machine is None:
        return None, False

    frame, crowding_measured = feature_frame(df)
    # The machine answers on every bar, warmup included: an unmeasurable row is
    # a failure to it rather than a gap, so there is no missing state to encode.
    # It is also why a warmup bar reports COMPRESSION -- failing input, not a
    # quiet market -- and why every view has to draw the ribbon from
    # ``warmup_bars`` rather than from bar zero.
    states = machine.run(frame)
    return (
        WhyLayer(
            states=[state.value for state in states],
            features={str(name): _values(frame[name]) for name in frame.columns},
        ),
        crowding_measured,
    )


def _provenance(
    identity: MarketDataIdentity,
    resolved: ResolvedStrategy,
    prepared: PreparedFrame,
    *,
    metadata: dict,
    exit_mode: str | None,
    failure_bars: int | None,
    allow_shorts: bool,
    cost_model: dict[str, float] | None,
) -> Provenance:
    """``crowding_measured`` comes from the strategy's own metadata, not from the
    funding flag: a strategy that reads no funding-derived feature measured no
    crowding however well funded the frame was, and the two facts are reported
    separately so neither is inferred from the other.

    ``reads_crowding`` is what makes that pair legible, and it comes from a
    **third** source: the strategy's declared features rather than the run's
    metadata. It is a fact about the *configuration*, true before a bar is
    computed, so introspecting it follows ``_why_layer`` above -- a strategy that
    grows a crowding feature is described correctly without anyone remembering to
    republish a metadata key, which is the failure mode a silent warning has.

    Together they are the only honest predicate for the M20 condition: reads and
    not measured is a machine running with ``crowding`` pinned to
    ``NEUTRAL_CROWDING``, while neither is ``donchian``, where nothing is wrong
    at all. A market-type gate cannot tell those apart.
    """
    strategy = resolved.strategy
    return Provenance(
        identity={
            "exchange": identity.exchange,
            "market_type": identity.market_type,
            "symbol": identity.symbol,
            "timeframe": identity.timeframe,
        },
        strategy=strategy.name,
        version=strategy.version,
        contract=resolved.contract.value,
        exit_mode=exit_mode,
        failure_bars=failure_bars,
        warmup_bars=int(strategy.warmup_bars),
        allow_shorts=allow_shorts,
        # The literal is ``state_machine_core.build_feature_frame``'s, which is the
        # code that does the pinning.
        reads_crowding="crowding" in getattr(strategy, "features", ()),
        crowding_measured=bool(metadata.get("crowding_measured", False)),
        funding_attached=prepared.funding_attached,
        cost_model=cost_model,
        first_bar=str(prepared.df.index.min()),
        last_bar=str(prepared.df.index.max()),
        bar_count=len(prepared.df),
        generated_at=datetime.now(UTC).isoformat(),
    )


def _epoch(timestamp: Any) -> int:
    return int(pd.Timestamp(timestamp).timestamp())


def _values(series: pd.Series | None) -> list[float | None] | None:
    """A float series as JSON, with NaN as ``null``.

    ``NaN`` is a feature's "not yet measurable" and JSON has no spelling for it;
    ``0.0`` would read as "measured, and neutral", which is a different claim
    about the market.
    """
    if series is None:
        return None
    values = series.to_numpy(dtype="float64")
    return [None if np.isnan(value) else float(value) for value in values]


__all__ = [
    "AnalysisPayload",
    "Contract",
    "DatasetUnavailable",
    "Marker",
    "PreparedFrame",
    "Provenance",
    "ResolvedStrategy",
    "StrategyInfo",
    "WhyLayer",
    "build_analysis",
    "prepare_frame",
    "registered_strategies",
    "resolve_strategy",
]
