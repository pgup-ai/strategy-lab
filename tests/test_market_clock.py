from __future__ import annotations

from decimal import Decimal

import pytest

from strategy_lab.core.types import Bar, BarEvent, InstrumentId, MarketSnapshot
from strategy_lab.engine.market_clock import MarketClock

BTC = InstrumentId("binance", "perp", "BTC/USDT")
ETH = InstrumentId("binance", "perp", "ETH/USDT")


def bar(instrument: InstrumentId, ts_open_ms: int, close: str = "100") -> Bar:
    return Bar(
        instrument=instrument,
        timeframe="4h",
        ts_open_ms=ts_open_ms,
        ts_close_ms=ts_open_ms + 14_400_000 - 1,
        open=Decimal(close),
        high=Decimal(close),
        low=Decimal(close),
        close=Decimal(close),
        volume=Decimal("1"),
        is_closed=True,
    )


def event(instrument: InstrumentId, ts_open_ms: int) -> BarEvent:
    b = bar(instrument, ts_open_ms)
    return BarEvent(bar=b, ts_event_ms=b.ts_close_ms, ts_recv_ms=None)


def test_snapshot_reports_a_missing_instrument_rather_than_inventing_one():
    """Absent must never read as unchanged -- instruments list, delist, and halt."""
    snapshot = MarketSnapshot(ts_event_ms=1_000, bars={BTC.at("4h"): bar(BTC, 0)})
    assert ETH.at("4h") not in snapshot
    assert snapshot.get(ETH.at("4h")) is None
    with pytest.raises(KeyError):
        snapshot[ETH.at("4h")]


def test_a_timestamp_is_complete_only_once_a_later_event_arrives():
    """Completeness is established causally: t is done when t+1 shows up."""
    clock = MarketClock()
    assert clock.on_event(event(BTC, 0)) is None
    assert clock.on_event(event(ETH, 0)) is None

    snapshot = clock.on_event(event(BTC, 14_400_000))
    assert snapshot is not None
    assert set(snapshot.candles) == {BTC.at("4h"), ETH.at("4h")}
    assert snapshot.ts_event_ms == 14_400_000 - 1


def test_flush_releases_the_final_timestamp():
    """Nothing arrives after the last bar, so it needs an explicit flush."""
    clock = MarketClock()
    clock.on_event(event(BTC, 0))
    clock.on_event(event(ETH, 0))

    snapshot = clock.flush()
    assert snapshot is not None and len(snapshot) == 2
    assert clock.flush() is None, "flushing twice must not replay the snapshot"


def test_a_partial_universe_is_emitted_as_is():
    """ETH is halted; the snapshot reports BTC only rather than stalling."""
    clock = MarketClock()
    clock.on_event(event(BTC, 0))
    snapshot = clock.on_event(event(BTC, 14_400_000))
    assert set(snapshot.candles) == {BTC.at("4h")}


def test_an_out_of_order_event_is_rejected_not_silently_reordered():
    clock = MarketClock()
    clock.on_event(event(BTC, 14_400_000))
    with pytest.raises(ValueError, match="out of order"):
        clock.on_event(event(ETH, 0))


def test_a_duplicate_instrument_at_one_timestamp_keeps_the_last():
    """A reconnect can redeliver a bar; last wins, matching the feed's dedup."""
    clock = MarketClock()
    clock.on_event(event(BTC, 0))
    clock.on_event(BarEvent(bar=bar(BTC, 0, close="999"), ts_event_ms=14_399_999, ts_recv_ms=None))
    snapshot = clock.flush()
    assert snapshot[BTC.at("4h")].close == Decimal("999")
