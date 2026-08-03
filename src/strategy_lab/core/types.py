from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

_PRICE_FIELDS = ("open", "high", "low", "close", "volume")


class Side(StrEnum):
    ENTER_LONG = "enter_long"
    EXIT_LONG = "exit_long"
    ENTER_SHORT = "enter_short"
    EXIT_SHORT = "exit_short"

    @property
    def opposite_exit(self) -> Side:
        if self is Side.ENTER_LONG:
            return Side.EXIT_LONG
        if self is Side.ENTER_SHORT:
            return Side.EXIT_SHORT
        return self


class Mode(StrEnum):
    BACKTEST = "backtest"
    REPLAY = "replay"
    PAPER = "paper"
    LIVE = "live"


@dataclass(frozen=True, slots=True)
class InstrumentId:
    exchange: str
    market_type: str
    symbol: str

    @property
    def key(self) -> str:
        return f"{self.exchange}:{self.market_type}:{self.symbol}"


@dataclass(frozen=True, slots=True)
class Bar:
    instrument: InstrumentId
    timeframe: str
    ts_open_ms: int
    ts_close_ms: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    is_closed: bool
    quote_volume: Decimal | None = None
    trades: int | None = None

    def __post_init__(self) -> None:
        for field_name in _PRICE_FIELDS:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                raise TypeError(f"{field_name} must be Decimal, got {type(value).__name__}")
        if self.ts_close_ms <= self.ts_open_ms:
            raise ValueError("ts_close_ms must be after ts_open_ms")
        if self.high < self.low:
            raise ValueError("high must be >= low")


@dataclass(frozen=True, slots=True)
class BarEvent:
    bar: Bar
    ts_event_ms: int
    ts_recv_ms: int | None = None

    @property
    def instrument(self) -> InstrumentId:
        return self.bar.instrument

    @property
    def is_closed(self) -> bool:
        return self.bar.is_closed


@dataclass(frozen=True, slots=True)
class Signal:
    instrument: InstrumentId
    timeframe: str
    strategy_id: str
    strategy_version: str
    ts_bar_ms: int
    ts_emit_ms: int
    side: Side
    bar_is_closed: bool
    reason: str
    entry_price: Decimal | None = None
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    strength: Decimal | None = None
    features: dict | None = None
