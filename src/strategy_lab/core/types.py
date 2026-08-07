"""Shared vocabulary types for the event-driven engine.

- All timestamps are UTC epoch milliseconds as ``int`` — never naive
  datetimes, never local time.
- Prices and quantities are ``Decimal`` in core and storage; float64 appears
  only later, inside the pandas indicator layer.
- This package is stdlib-only: no I/O, no network, no third-party imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

_DECIMAL_FIELDS = ("open", "high", "low", "close", "volume")


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
    """What is traded. Not what is sampled -- see :class:`CandleId`."""

    exchange: str
    market_type: str
    symbol: str

    @property
    def key(self) -> str:
        return f"{self.exchange}:{self.market_type}:{self.symbol}"

    def at(self, timeframe: str) -> CandleId:
        return CandleId(self, timeframe)


@dataclass(frozen=True, slots=True)
class CandleId:
    """The full candle identity: an instrument sampled at one timeframe.

    An instrument is what a position is held in; a timeframe is one sampling of
    it, and the two are not interchangeable -- ``Signal`` carries both as separate
    fields because a 4h signal and a 1d signal on BTC trade the same book. But
    anything that *stores bars* must key on the pair: BTC at 4h and BTC at 1d
    close at the same instant six times a day, so a dict keyed by instrument alone
    silently keeps whichever arrived last. The timeframe is a literal string --
    ``1w`` and ``1wk`` are distinct datasets, not synonyms.
    """

    instrument: InstrumentId
    timeframe: str

    @property
    def key(self) -> str:
        return f"{self.instrument.key}:{self.timeframe}"


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

    @property
    def candle(self) -> CandleId:
        return CandleId(self.instrument, self.timeframe)

    def __post_init__(self) -> None:
        for field_name in _DECIMAL_FIELDS:
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
class MarketSnapshot:
    """Every bar that closed at one event time.

    Breadth over five coins is meaningless unless all five bars describe the same
    instant. Only candles that actually have a bar at ``ts_event_ms`` are held
    -- crypto trades around the clock and equities do not -- so a partial universe
    is the normal case and ``absent`` must never be read as ``unchanged``.

    Keyed by :class:`CandleId`, not ``InstrumentId``: one symbol subscribed at two
    timeframes closes both bars at the same instant, and an instrument-keyed dict
    resolves that by dropping one.
    """

    ts_event_ms: int
    bars: dict[CandleId, Bar]

    def __getitem__(self, candle: CandleId) -> Bar:
        return self.bars[candle]

    def __contains__(self, candle: object) -> bool:
        return candle in self.bars

    def __len__(self) -> int:
        return len(self.bars)

    def get(self, candle: CandleId) -> Bar | None:
        return self.bars.get(candle)

    @property
    def candles(self) -> tuple[CandleId, ...]:
        return tuple(self.bars)


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


@dataclass(frozen=True, slots=True)
class BarReason:
    """What a strategy saw on one bar, whether or not it acted on it.

    :class:`Signal` is an *event*; this is the row underneath every bar, whether
    or not one fired. Measured on R10a's diff window -- ``state_machine_v1`` over
    6,048 bars of BTC/USDT perp 4h -- that is 325 signals against 6,048 reasons,
    and 4,124 of those bars were spent in ``COMPRESSION``. The question a
    dashboard exists to answer is at least as often "why did it **not** trade
    here" as "why did it trade", and rows on decision bars alone cannot say.

    ``features`` holds ``float``, not ``Decimal``, which is the one place this
    module departs from its own header rule. That rule is about *prices and
    quantities*; a feature value is neither. It is float64 out of the pandas
    indicator layer by construction, so demanding a ``Decimal`` here would only
    move ``Decimal(str(float(x)))`` out to every call site -- precisely the
    argument ``storage.signals.ExposureSignal`` makes for a target level, and
    precisely where the cast gets forgotten. The cast happens once, at the
    storage bind.

    ``None`` is "not yet measurable", never "measured, and neutral": warmup rows
    are ``NaN`` by the ``features.base.mask_warmup`` convention and a ``0.0``
    there would be a different claim about the market. ``state`` has no such
    hole -- the machine answers on every bar, treating an unmeasurable row as a
    failure rather than a gap -- so it is a plain label.
    """

    instrument: InstrumentId
    timeframe: str
    strategy_id: str
    strategy_version: str
    ts_bar_ms: int
    ts_emit_ms: int
    bar_is_closed: bool
    state: str
    features: dict[str, float | None]
