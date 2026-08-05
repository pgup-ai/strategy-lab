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
from strategy_lab.backtests import ExitMode
from strategy_lab.timeframes import timeframe_to_millis

# Symbols carry a slash and sometimes a colon (BTC/USDT:USDT); venue and
# timeframe ids do not. Bounded and character-limited because both reach a SQL
# equality filter and a filesystem-free identity string, and an unbounded query
# parameter is a habit rather than a requirement.
_VENUE_PATTERN = r"^[A-Za-z0-9_.-]+$"
_SYMBOL_PATTERN = r"^[A-Za-z0-9/:._-]+$"


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
    market_type: Literal["spot", "perp", "equity"]
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


class RefreshModel(_Strict):
    bars: list[BarModel]


__all__ = [
    "AnalysisModel",
    "AnalysisQuery",
    "BarModel",
    "DatasetModel",
    "IdentityQuery",
    "MarkerModel",
    "ProvenanceModel",
    "RefreshModel",
    "RefreshQuery",
    "StrategyModel",
    "WhyModel",
]
