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

from typing import Annotated, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class AnalysisQuery(IdentityQuery):
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
    start: str | None = None
    end: str | None = None
    fees: Annotated[float, Field(ge=0.0, le=1.0)] = DEFAULT_FEES
    slippage: Annotated[float, Field(ge=0.0, le=1.0)] = DEFAULT_SLIPPAGE
    cash: Annotated[float, Field(gt=0.0)] = DEFAULT_CASH
    position_pct: Annotated[float, Field(ge=0.01, le=1.0)] = DEFAULT_POSITION_PCT
    funding: bool = True
    allow_shorts: bool = True

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
    provenance states. Widening the query would multiply the cache key by
    settings a tile cannot show.
    """

    strategies: Annotated[str, Field(min_length=1, max_length=512)]
    # ``None`` is "every market", which is a real answer here rather than a
    # missing filter: the board covers what is stored.
    market_type: MarketType | None = None
    exit_mode: ExitMode | None = None
    # Bounded above by what a cached row holds: asking for more would serve a
    # shorter tail than the number requested, silently.
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
            # Two identical rows per dataset, one of them shadowing the other in
            # the cache. Refused rather than deduplicated: the caller asked for
            # something that cannot be drawn, and silently narrowing it is the
            # habit this whole module exists to break.
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


class RefreshQuery(IdentityQuery):
    """``after`` is an epoch-second cursor: the caller already has bars up to it.

    Bounded below because a negative cursor reads as a timestamp before 1970 and
    would return the whole history to a caller that asked for the tail.
    """

    after: Annotated[int, Field(ge=0)] | None = None


class DatasetModel(_Strict):
    exchange: str
    market_type: str
    symbol: str
    timeframe: str
    candles: int
    first_timestamp: str
    last_timestamp: str


class StrategyModel(_Strict):
    name: str
    contract: str
    version: str
    warmup_bars: int


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
    position_size: list[float | None] | None
    target: list[float | None] | None
    why: WhyModel | None
    provenance: ProvenanceModel


class BoardRowModel(_Strict):
    """One board tile on the wire, forbidding extras for the reason above.

    The board streams, so this model is applied per line rather than by
    FastAPI's ``response_model`` -- same guarantee, same place a field added to
    ``BoardRow`` and forgotten here becomes an error instead of a quiet loss.

    ``unavailable`` is the whole of a refused row and ``provenance`` is
    ``None`` there: nothing was computed, so there is no run to describe, and a
    provenance block full of defaults would describe one that never happened.
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
