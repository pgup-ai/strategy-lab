from __future__ import annotations

import pandas as pd

from strategy_lab.core.types import InstrumentId
from strategy_lab.engine.context import BarBuffer
from strategy_lab.feeds.replay import _row_to_bar
from tests.conftest import synthetic_ohlcv

INSTRUMENT = InstrumentId("binance", "perp", "BTC/USDT")
BAR_MS = 15 * 60 * 1000


def bars_from(df: pd.DataFrame):
    return [_row_to_bar(ts, row, INSTRUMENT, "15m", BAR_MS) for ts, row in df.iterrows()]


def test_buffer_starts_empty():
    assert len(BarBuffer()) == 0


def test_buffer_frame_matches_the_source_dataframe_exactly():
    df = synthetic_ohlcv(n=30)
    buffer = BarBuffer()
    for bar in bars_from(df):
        buffer.append(bar)

    frame = buffer.frame()
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    pd.testing.assert_frame_equal(frame, df[frame.columns], check_freq=False)


def test_buffer_index_is_utc_and_named_timestamp():
    df = synthetic_ohlcv(n=5)
    buffer = BarBuffer()
    for bar in bars_from(df):
        buffer.append(bar)

    frame = buffer.frame()
    assert frame.index.name == "timestamp"
    assert str(frame.index.tz) == "UTC"


def test_empty_buffer_still_yields_a_utc_ohlcv_frame():
    """An empty DatetimeIndex defaults to tz-naive, which would make the frame's
    dtype depend on how many bars had arrived."""
    frame = BarBuffer().frame()

    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert str(frame.index.tz) == "UTC"
    assert frame.empty


def test_appending_the_same_bar_twice_replaces_rather_than_duplicates():
    """A websocket reconnect can resend the last closed bar."""
    df = synthetic_ohlcv(n=3)
    bars = bars_from(df)
    buffer = BarBuffer()
    for bar in bars:
        buffer.append(bar)
    buffer.append(bars[-1])

    assert len(buffer) == 3


def test_replacing_a_bar_overwrites_every_column():
    """The resend is the corrected copy, so its values must win -- not just its slot."""
    from dataclasses import replace
    from decimal import Decimal

    bars = bars_from(synthetic_ohlcv(n=3))
    buffer = BarBuffer()
    for bar in bars:
        buffer.append(bar)

    corrected = replace(
        bars[-1],
        open=Decimal("11"),
        high=Decimal("14"),
        low=Decimal("9"),
        close=Decimal("13"),
        volume=Decimal("777"),
    )
    buffer.append(corrected)

    row = buffer.frame().iloc[-1]
    assert row.to_dict() == {
        "open": 11.0,
        "high": 14.0,
        "low": 9.0,
        "close": 13.0,
        "volume": 777.0,
    }
    assert buffer.replaced_duplicates == 1


def test_out_of_order_bar_is_rejected():
    df = synthetic_ohlcv(n=3)
    bars = bars_from(df)
    buffer = BarBuffer()
    buffer.append(bars[2])
    buffer.append(bars[0])
    assert len(buffer) == 1


def test_dropped_and_replaced_bars_are_counted():
    """Both are silent by design; a broken feed must still be countable."""
    bars = bars_from(synthetic_ohlcv(n=3))
    buffer = BarBuffer()
    buffer.append(bars[2])
    buffer.append(bars[0])
    buffer.append(bars[1])
    buffer.append(bars[2])

    assert len(buffer) == 1
    assert buffer.dropped_out_of_order == 2
    assert buffer.replaced_duplicates == 1


def test_frame_is_cached_until_the_next_append():
    df = synthetic_ohlcv(n=4)
    bars = bars_from(df)
    buffer = BarBuffer()
    for bar in bars[:3]:
        buffer.append(bar)

    first = buffer.frame()
    assert buffer.frame() is first
    buffer.append(bars[3])
    assert buffer.frame() is not first


def test_replacing_a_bar_invalidates_the_cached_frame():
    """A replacement keeps the length -- only the cache invalidation makes it visible."""
    from dataclasses import replace
    from decimal import Decimal

    bars = bars_from(synthetic_ohlcv(n=3))
    buffer = BarBuffer()
    for bar in bars:
        buffer.append(bar)

    stale = buffer.frame()
    buffer.append(replace(bars[-1], close=Decimal("12345")))
    fresh = buffer.frame()

    assert fresh is not stale
    assert fresh["close"].iloc[-1] == 12345.0
