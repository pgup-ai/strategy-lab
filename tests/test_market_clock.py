from __future__ import annotations

from decimal import Decimal

import pytest

from strategy_lab.core.types import Bar, InstrumentId, MarketSnapshot

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


def test_snapshot_exposes_its_bars_by_instrument():
    snapshot = MarketSnapshot(ts_event_ms=1_000, bars={BTC: bar(BTC, 0), ETH: bar(ETH, 0)})
    assert snapshot[BTC].instrument == BTC
    assert set(snapshot.instruments) == {BTC, ETH}
    assert len(snapshot) == 2


def test_snapshot_reports_a_missing_instrument_rather_than_inventing_one():
    """Absent must never read as unchanged -- instruments list, delist, and halt."""
    snapshot = MarketSnapshot(ts_event_ms=1_000, bars={BTC: bar(BTC, 0)})
    assert ETH not in snapshot
    assert snapshot.get(ETH) is None
    with pytest.raises(KeyError):
        snapshot[ETH]


def test_snapshot_is_frozen():
    snapshot = MarketSnapshot(ts_event_ms=1_000, bars={BTC: bar(BTC, 0)})
    with pytest.raises(AttributeError):
        snapshot.ts_event_ms = 2_000
