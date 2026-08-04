from __future__ import annotations

from decimal import Decimal

import pytest

from strategy_lab.core.types import Bar, InstrumentId, MarketSnapshot
from strategy_lab.features.cross_sectional import breadth, confirms

BTC = InstrumentId("binance", "perp", "BTC/USDT")
ETH = InstrumentId("binance", "perp", "ETH/USDT")
SOL = InstrumentId("binance", "perp", "SOL/USDT")


def bar(instrument, open_: str, close: str) -> Bar:
    return Bar(
        instrument=instrument, timeframe="4h", ts_open_ms=0, ts_close_ms=14_399_999,
        open=Decimal(open_), high=Decimal(max(open_, close, key=float)),
        low=Decimal(min(open_, close, key=float)), close=Decimal(close),
        volume=Decimal("1"), is_closed=True,
    )


def snapshot(**bars) -> MarketSnapshot:
    mapping = {BTC: bars.get("btc"), ETH: bars.get("eth"), SOL: bars.get("sol")}
    return MarketSnapshot(ts_event_ms=14_399_999,
                          bars={k: v for k, v in mapping.items() if v is not None})


def test_breadth_is_the_fraction_advancing():
    snap = snapshot(btc=bar(BTC, "100", "110"), eth=bar(ETH, "100", "105"),
                    sol=bar(SOL, "100", "90"))
    assert breadth(snap) == pytest.approx(2 / 3)


def test_breadth_over_a_partial_universe_uses_only_present_instruments():
    """A halted instrument must not count as flat -- it is absent, not unchanged."""
    snap = snapshot(btc=bar(BTC, "100", "110"), eth=bar(ETH, "100", "90"))
    assert breadth(snap) == pytest.approx(0.5)


def test_breadth_of_an_empty_snapshot_is_undefined_rather_than_zero():
    with pytest.raises(ValueError, match="no instruments"):
        breadth(MarketSnapshot(ts_event_ms=0, bars={}))


def test_breadth_of_a_lone_instrument_is_refused_as_not_a_cross_section():
    """Mixed timeframes make this the common case, and 0.0/1.0 there is just direction.

    A 1d close coincides with a 4h close only at day boundaries, so most snapshots
    over a mixed-timeframe universe hold one instrument. Dividing by instruments
    present would return a well-formed number carrying no cross-sectional content.
    """
    snap = snapshot(btc=bar(BTC, "100", "110"))
    with pytest.raises(ValueError, match="min_instruments"):
        breadth(snap)
    assert breadth(snap, min_instruments=1) == pytest.approx(1.0)


def test_confirms_requires_the_leader_and_a_quorum_of_followers():
    snap = snapshot(btc=bar(BTC, "100", "110"), eth=bar(ETH, "100", "105"),
                    sol=bar(SOL, "100", "90"))
    assert confirms(snap, leader=BTC, quorum=0.5) is True
    assert confirms(snap, leader=BTC, quorum=0.9) is False


def test_confirms_is_false_when_the_leader_is_absent():
    """No leader bar means no confirmation claim -- not a default True."""
    snap = snapshot(eth=bar(ETH, "100", "105"))
    assert confirms(snap, leader=BTC, quorum=0.5) is False


def test_confirms_follows_the_leader_down_as_well_as_up():
    """Confirmation is agreement with the leader's direction, not a bullish test."""
    snap = snapshot(btc=bar(BTC, "100", "90"), eth=bar(ETH, "100", "95"),
                    sol=bar(SOL, "100", "110"))
    assert confirms(snap, leader=BTC, quorum=0.5) is True


def test_confirms_is_false_when_the_leader_has_no_direction_to_confirm():
    snap = snapshot(btc=bar(BTC, "100", "100"), eth=bar(ETH, "100", "105"))
    assert confirms(snap, leader=BTC, quorum=0.5) is False
