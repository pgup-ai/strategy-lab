from __future__ import annotations

from decimal import Decimal

import pytest

from strategy_lab.core.types import Bar, BarEvent, InstrumentId, Mode, Side


def make_bar(**overrides) -> Bar:
    defaults = dict(
        instrument=InstrumentId("binance", "perp", "BTC/USDT"),
        timeframe="15m",
        ts_open_ms=1_785_723_300_000,
        ts_close_ms=1_785_724_199_999,
        open=Decimal("63205.31"),
        high=Decimal("63286.00"),
        low=Decimal("63100.00"),
        close=Decimal("63128.00"),
        volume=Decimal("96.3039"),
        is_closed=True,
    )
    defaults.update(overrides)
    return Bar(**defaults)


def test_instrument_id_is_hashable_and_renders_a_stable_key():
    instrument = InstrumentId("binance", "perp", "BTC/USDT")
    assert {instrument: 1}[instrument] == 1
    assert instrument.key == "binance:perp:BTC/USDT"


def test_bar_rejects_non_decimal_prices():
    with pytest.raises(TypeError, match="open must be Decimal"):
        make_bar(open=63205.31)


def test_bar_rejects_close_time_before_open_time():
    with pytest.raises(ValueError, match="ts_close_ms must be after ts_open_ms"):
        make_bar(ts_close_ms=1_785_723_299_999)


def test_bar_rejects_high_below_low():
    with pytest.raises(ValueError, match="high must be >= low"):
        make_bar(high=Decimal("1"), low=Decimal("2"))


def test_bar_is_frozen():
    bar = make_bar()
    with pytest.raises(AttributeError):
        bar.close = Decimal("1")


def test_bar_event_exposes_bar_timestamp():
    event = BarEvent(bar=make_bar(), ts_event_ms=1_785_724_200_140, ts_recv_ms=None)
    assert event.ts_event_ms == 1_785_724_200_140
    assert event.instrument == InstrumentId("binance", "perp", "BTC/USDT")


def test_side_and_mode_are_string_enums():
    assert Side.ENTER_LONG == "enter_long"
    assert Mode.REPLAY == "replay"
    assert Side.ENTER_LONG.opposite_exit == Side.EXIT_LONG
    assert Side.ENTER_SHORT.opposite_exit == Side.EXIT_SHORT
