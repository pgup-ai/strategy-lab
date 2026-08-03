from __future__ import annotations

from decimal import Decimal

import pytest

from strategy_lab.core import Bar, InstrumentId, Side


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
    assert {instrument: 1}[instrument] == 1  # ReplayFeed keys its frames on this
    assert instrument.key == "binance:perp:BTC/USDT"


@pytest.mark.parametrize("field_name", ["open", "high", "low", "close", "volume"])
def test_bar_rejects_non_decimal_prices(field_name):
    with pytest.raises(TypeError, match=f"{field_name} must be Decimal"):
        make_bar(**{field_name: 1.0})


def test_bar_rejects_close_time_before_open_time():
    with pytest.raises(ValueError, match="ts_close_ms must be after ts_open_ms"):
        make_bar(ts_close_ms=1_785_723_299_999)


def test_bar_rejects_zero_length_bar():
    """The boundary that separates `<=` from `<` in the validation."""
    with pytest.raises(ValueError, match="ts_close_ms must be after ts_open_ms"):
        make_bar(ts_close_ms=1_785_723_300_000)


def test_bar_rejects_high_below_low():
    with pytest.raises(ValueError, match="high must be >= low"):
        make_bar(high=Decimal("1"), low=Decimal("2"))


def test_side_opposite_exit_maps_entries_and_passes_exits_through():
    assert Side.ENTER_LONG.opposite_exit is Side.EXIT_LONG
    assert Side.ENTER_SHORT.opposite_exit is Side.EXIT_SHORT
    assert Side.EXIT_LONG.opposite_exit is Side.EXIT_LONG
