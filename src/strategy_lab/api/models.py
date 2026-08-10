"""The browser's wire contract: what may be asked for, and what comes back.

**Pydantic is here for the inbound half.** ``server.py`` hand-parses five query
parameters -- a ``required()`` closure raising on empty strings, and
``int(values[0]) if values else None`` with no bounds check. This surface is
roughly double that, and every hand-parsed parameter is a place where a wrong or
missing value silently becomes a default. That is M20 exactly: a funding column
silently absent changed a published figure, and it took a second-asset
replication to notice. An API that quietly defaults ``exit_mode`` reproduces the
failure precisely -- a plausible number computed under settings nobody chose --
so every query model below refuses what it does not recognise and names the
field it refused.

**The outbound half is for the schema, not for safety.** This repo already
enforces required fields with frozen dataclasses and validating ``__post_init__``
(``TargetExposure``, ``SignalSet``, ``Bar``, ``Signal``), and ``api.analysis``
follows that idiom; Pydantic adds nothing there the house style does not already
do. The response models exist so the shape is documented and stable -- with one
exception that *is* about safety: ``extra="forbid"``. A response model silently
drops fields it has not declared, so a provenance field added upstream and
forgotten here would vanish from the wire with nothing raised. Forbidding extras
turns that omission into an error instead of a quiet loss.
"""

from __future__ import annotations

import re

from typing import Annotated, Literal

import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from strategy_lab.api.analysis import (
    DEFAULT_CASH,
    DEFAULT_FAILURE_BARS,
    DEFAULT_FEES,
    DEFAULT_POSITION_PCT,
    DEFAULT_SLIPPAGE,
    Contract,
    resolve_strategy,
)
from strategy_lab.api.board import DEFAULT_SPARK_BARS, MAX_SPARK_BARS
from strategy_lab.backtests import ExitMode
from strategy_lab.timeframes import timeframe_to_millis

# Symbols carry a slash and sometimes a colon (BTC/USDT:USDT); venue and
# timeframe ids do not. Bounded and character-limited because both reach a SQL
# equality filter and a filesystem-free identity string, and an unbounded query
# parameter is a habit rather than a requirement.
_VENUE_PATTERN = r"^[A-Za-z0-9_.-]+$"
_SYMBOL_PATTERN = r"^[A-Za-z0-9/:._-]+$"

# Storage keys candles on exactly three market types, and both the identity and
# the board's filter answer for the same set. Written once so a fourth reaches
# both or neither -- a filter offering a value the identity refuses is a control
# whose only outcome is a 422.
MarketType = Literal["spot", "perp", "equity"]


class _Strict(BaseModel):
    """Refuse an unrecognised field rather than ignoring it.

    On a query model that means a misspelled parameter is a 422 naming it,
    instead of a run at the default the caller thought they had overridden.
    """

    model_config = ConfigDict(extra="forbid")


class IdentityQuery(_Strict):
    """The four fields that identify a candle set, exactly as storage keys it.

    The timeframe is a literal string here for the same reason it is one in
    ``market_candles``: ``1w`` and ``1wk`` are distinct datasets, and normalising
    them would silently serve one where the other was asked for.
    """

    exchange: Annotated[str, Field(min_length=1, max_length=32, pattern=_VENUE_PATTERN)]
    market_type: MarketType
    symbol: Annotated[str, Field(min_length=1, max_length=32, pattern=_SYMBOL_PATTERN)]
    timeframe: Annotated[str, Field(min_length=1, max_length=8, pattern=_VENUE_PATTERN)]

    @field_validator("timeframe")
    @classmethod
    def _parseable(cls, value: str) -> str:
        try:
            timeframe_to_millis(value)
        except ValueError as exc:
            raise ValueError(f"{value!r} is not a timeframe this lab can parse") from exc
        return value


class BoundedQuery(IdentityQuery):
    """An identity plus the window asked of it.

    Shared by every endpoint that slices a frame, because the two bounds have to
    *mean* the same thing everywhere. They did not: ``/api/state`` shipped with
    bare ``start``/``end`` strings and none of this, so the page sent one
    ``<input type="date">`` value to both endpoints and got two different frames
    back — the state view ending the evening before the instrument view, which
    is exactly the disagreement `/api/state` exists not to have.
    """

    start: str | None = None
    end: str | None = None

    @field_validator("start", "end")
    @classmethod
    def _timestamp(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            pd.Timestamp(value)
        except ValueError as exc:
            raise ValueError(f"{value!r} is not a timestamp") from exc
        return value

    @field_validator("end")
    @classmethod
    def _end_of_the_named_day(cls, value: str | None) -> str | None:
        """A bare date as ``end`` means all of that day, not its first instant.

        ``load_candles`` filters ``timestamp <= end`` and ``"2023-10-31"`` parses
        as midnight, so on a 4h frame that returned one bar of the named day and
        dropped the other five -- while ``start`` included the whole of its first
        day, since that side compares ``>=``. The page's ``<input type="date">``
        sends exactly this shape, so a user picked a day and the chart ended the
        evening before, which reads as stale data rather than as a boundary.

        Extended here rather than in the page so any client gets the same
        meaning, and only for a value carrying no time of day -- an explicit
        ``2023-10-31 00:00`` still means that instant.
        """
        if value is None:
            return None
        parsed = pd.Timestamp(value)
        if parsed == parsed.normalize() and len(value.strip()) <= 10:
            return str(parsed + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
        return value


class AnalysisQuery(BoundedQuery):
    """Everything that can change the numbers, and nothing that cannot.

    ``exit_mode`` is deliberately optional rather than defaulted here: the
    continuous-exposure contract has no exit mode at all, so a model that
    defaulted one would have to drop it silently for half the registry.
    """

    strategy: Annotated[str, Field(min_length=1, max_length=64)]
    exit_mode: ExitMode | None = None
    # Zero adverse closes would exit on every bar; the engine refuses it with a
    # ValueError several layers down, which reaches a caller as a 400 rather
    # than as the field it belongs to.
    failure_bars: Annotated[int, Field(ge=1)] = DEFAULT_FAILURE_BARS
    fees: Annotated[float, Field(ge=0.0, le=1.0)] = DEFAULT_FEES
    slippage: Annotated[float, Field(ge=0.0, le=1.0)] = DEFAULT_SLIPPAGE
    cash: Annotated[float, Field(gt=0.0)] = DEFAULT_CASH
    position_pct: Annotated[float, Field(ge=0.01, le=1.0)] = DEFAULT_POSITION_PCT
    funding: bool = True
    allow_shorts: bool = True

    @field_validator("strategy")
    @classmethod
    def _registered(cls, value: str) -> str:
        resolve_strategy(value)
        return value

    @model_validator(mode="after")
    def _exit_mode_belongs_to_the_contract(self) -> AnalysisQuery:
        """An exit mode is meaningless on the continuous contract, so it is
        refused there rather than accepted and dropped -- a chart labelled with a
        setting that changed nothing is the failure this whole model guards."""
        if self.exit_mode is None:
            return self
        if resolve_strategy(self.strategy).contract is Contract.TARGET_EXPOSURE:
            raise ValueError(
                f"{self.strategy} runs on the continuous-exposure contract, which "
                f"has no exit mode: a target of 0.0 is the exit"
            )
        return self


class BoardQuery(_Strict):
    """Which pairs the board covers. No dataset here: it covers what is stored.

    ``strategies`` is comma-separated, the shape the CLI already uses for
    ``--symbols`` and ``--horizons``, and every name in it is resolved against
    the registries at the boundary -- a misspelling is a 422 naming the field
    rather than a row-shaped hole in the middle of a stream that has already
    started.

    There is no cost model, no ``allow_shorts`` and no ``failure_bars``: the
    board runs every row at the engine's defaults, which each row's own
    provenance states. Every row is a full recompute, so widening the query
    would buy a tile settings it cannot show at the cost of running the board
    again per combination.
    """

    strategies: Annotated[str, Field(min_length=1, max_length=512)]
    # ``None`` is "every market", which is a real answer here rather than a
    # missing filter: the board covers what is stored.
    market_type: MarketType | None = None
    exit_mode: ExitMode | None = None
    # Bounded above by ``board.MAX_SPARK_BARS``, which is where a row's tail is
    # actually cut: asking for more would serve a shorter one than the number
    # requested, silently.
    spark_bars: Annotated[int, Field(ge=2, le=MAX_SPARK_BARS)] = DEFAULT_SPARK_BARS

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(part.strip() for part in self.strategies.split(",") if part.strip())

    @field_validator("strategies")
    @classmethod
    def _registered(cls, value: str) -> str:
        names = [part.strip() for part in value.split(",") if part.strip()]
        if not names:
            raise ValueError("name at least one strategy")
        if len(set(names)) != len(names):
            # Two identical tiles per dataset, and the second computed from
            # scratch to say what the first already said. Refused rather than
            # deduplicated: the caller asked for something that cannot be drawn,
            # and silently narrowing it is the habit this module exists to
            # break.
            raise ValueError(f"duplicate strategy in {value!r}")
        for name in names:
            resolve_strategy(name)
        return ",".join(names)

    @model_validator(mode="after")
    def _exit_mode_belongs_to_every_contract(self) -> BoardQuery:
        """One exit mode covers every row, so one continuous strategy refuses it.

        Applying it to the boolean rows and dropping it for the continuous ones
        would label half the board with a setting that changed nothing there,
        which is exactly what ``AnalysisQuery`` refuses one pair at a time.
        """
        if self.exit_mode is None:
            return self
        continuous = [
            name
            for name in self.names
            if resolve_strategy(name).contract is Contract.TARGET_EXPOSURE
        ]
        if continuous:
            raise ValueError(
                f"{', '.join(continuous)} runs on the continuous-exposure contract, "
                f"which has no exit mode: a target of 0.0 is the exit"
            )
        return self


# A trailing `Z`, `+hh:mm` or `-hh:mm` after the time part.
_HAS_TIMEZONE = re.compile(r"(?:[zZ]|[+-]\d{2}:?\d{2})$")
_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class RefreshQuery(IdentityQuery):
    """``after`` is an epoch-second cursor: the caller already has bars up to it.

    Bounded below because a negative cursor reads as a timestamp before 1970 and
    would return the whole history to a caller that asked for the tail.
    """

    after: Annotated[int, Field(ge=0)] | None = None
    # Only for a timeframe with nothing stored yet -- see `refresh_candles`.
    # A real datetime rather than a string, so an unparseable one is a 422 from
    # the boundary instead of a traceback from `pd.Timestamp` inside the route,
    # and a naive one is made UTC rather than raising later against an aware
    # comparison deep in the fetch.
    since: AwareDatetime | None = None

    @field_validator("since", mode="before")
    @classmethod
    def _assume_utc(cls, value: object) -> object:
        if not isinstance(value, str) or not value:
            return value
        # A bare date is what `AnalysisQuery` already accepts for its own bounds,
        # so refusing it here would make one date mean two things depending on
        # the endpoint. Midnight, because a start is the beginning of its day.
        if _DATE_ONLY.match(value):
            return value + "T00:00:00+00:00"
        if not _HAS_TIMEZONE.search(value):
            return value + "+00:00"
        return value


class DatasetModel(_Strict):
    exchange: str
    market_type: str
    symbol: str
    timeframe: str
    candles: int
    first_timestamp: str
    last_timestamp: str
    # `None` for most: Yahoo publishes no stream and `1wk` is its spelling of a
    # week. The page shows a live control only where this is set.
    stream: str | None = None


class StateQuery(BoundedQuery):
    """What a state reading depends on, which is markedly less than an analysis.

    No exit mode, no fees, no cash: none of them move a feature or a state, and
    accepting one would imply this path executed something. The *bounds* are the
    one thing it must share with ``AnalysisQuery`` exactly, which is why they
    come from a common base rather than being restated here.
    """

    strategy: Annotated[str, Field(min_length=1, max_length=64)]
    funding: bool = True


class StrategyModel(_Strict):
    name: str
    contract: str
    version: str
    warmup_bars: int
    has_state: bool


class BarModel(_Strict):
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarkerModel(_Strict):
    time: int
    kind: Literal["entry", "exit"]
    side: Literal["long", "short"]
    price: float
    size: float


class TradeModel(_Strict):
    entry_time: int
    exit_time: int | None
    direction: Literal["long", "short"]
    size: float
    entry_price: float
    exit_price: float | None
    fees: float
    pnl: float
    return_pct: float
    status: Literal["closed", "open"]


class WhyModel(_Strict):
    states: list[str]
    features: dict[str, list[float | None]]


class ProvenanceModel(_Strict):
    """Non-optional, and every field of it. See ``analysis.Provenance``."""

    identity: dict[str, str]
    strategy: str
    version: str
    contract: str
    exit_mode: str | None
    failure_bars: int | None
    warmup_bars: int
    allow_shorts: bool
    reads_crowding: bool
    crowding_measured: bool
    funding_attached: bool
    cost_model: dict[str, float] | None
    first_bar: str
    last_bar: str
    bar_count: int
    generated_at: str


class AnalysisModel(_Strict):
    bars: list[BarModel]
    markers: list[MarkerModel]
    trades: list[TradeModel]
    position_size: list[float | None] | None
    target: list[float | None] | None
    why: WhyModel | None
    provenance: ProvenanceModel


class StateProvenanceModel(_Strict):
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


class StateModel(_Strict):
    bars: list[BarModel]
    why: WhyModel
    provenance: StateProvenanceModel


class BoardRowModel(_Strict):
    """One board tile on the wire, forbidding extras for the reason above.

    The board streams, so this model is applied per line rather than by
    FastAPI's ``response_model`` -- same guarantee, same place a field added to
    ``BoardRow`` and forgotten here becomes an error instead of a quiet loss.

    ``unavailable`` is the whole of a refused row and ``provenance`` is
    ``None`` there: nothing was computed, so there is no run to describe, and a
    provenance block full of defaults would describe one that never happened.

    ``dataset_last_bar`` and ``last_written`` are non-optional and survive a
    refusal, because both are facts about the stored candles rather than about
    the run.
    """

    identity: dict[str, str]
    strategy: str
    contract: str | None
    state: str | None
    features: dict[str, float | None] | None
    latest_fill: MarkerModel | None
    target: float | None
    as_of: str | None
    dataset_last_bar: str
    last_written: str
    closes: list[float]
    unavailable: str | None
    provenance: ProvenanceModel | None


class RefreshModel(_Strict):
    """The bars, and what the refresh actually wrote to get them.

    ``funding_upserted`` is ``None`` on anything that is not a perp, because no
    settlements were sought there -- a different fact from a contract that
    settled none, and the whole point of the counts is that "3 candles and 0
    settlements" is visible as drift rather than inferred from a chart that
    later refuses to load.
    """

    bars: list[BarModel]
    candles_upserted: int
    funding_upserted: int | None


__all__ = [
    "AnalysisModel",
    "AnalysisQuery",
    "BarModel",
    "BoardQuery",
    "BoardRowModel",
    "DatasetModel",
    "IdentityQuery",
    "MarkerModel",
    "MarketType",
    "ProvenanceModel",
    "RefreshModel",
    "RefreshQuery",
    "StrategyModel",
    "WhyModel",
]
