from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pandas as pd

from strategy_lab.core.types import InstrumentId
from strategy_lab.engine.context import BarBuffer
from strategy_lab.feeds.replay import _row_to_bar
from tests.conftest import synthetic_ohlcv

INSTRUMENT = InstrumentId("binance", "perp", "BTC/USDT")
BAR_MS = 15 * 60 * 1000


def bars_from(df: pd.DataFrame):
    return [_row_to_bar(ts, row, INSTRUMENT, "15m", BAR_MS) for ts, row in df.iterrows()]


def test_buffer_frame_matches_the_source_dataframe_exactly():
    df = synthetic_ohlcv(n=30)
    buffer = BarBuffer()
    for bar in bars_from(df):
        buffer.append(bar)

    frame = buffer.frame()
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    pd.testing.assert_frame_equal(frame, df[frame.columns], check_freq=False)


def test_buffer_retains_every_bar_rather_than_a_rolling_window():
    """The full-history invariant, asserted directly rather than via its symptoms.

    Nothing else in the suite catches a bounded buffer. A 250-bar cap produces 0
    signal mismatches out of 8,000 sampled comparisons in the determinism suite
    (relative ema200 error 6.6e-4) -- so a plausible "memory optimization" ships
    green while silently making live signals disagree with backtest signals on the
    ``ewm(adjust=False)`` strategies, which are recursive from bar 0. Asserting
    unboundedness fails for a cap of *any* size and cannot rot when strategy
    parameters change.
    """
    bars = 5000
    df = synthetic_ohlcv(n=bars)
    buffer = BarBuffer()
    for bar in bars_from(df):
        buffer.append(bar)

    frame = buffer.frame()
    assert len(buffer) == bars
    assert len(frame) == bars
    assert frame.index[0] == df.index[0]


def test_empty_buffer_still_yields_a_utc_ohlcv_frame():
    """An empty DatetimeIndex defaults to tz-naive, which would make the frame's
    dtype depend on how many bars had arrived."""
    frame = BarBuffer().frame()

    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert str(frame.index.tz) == "UTC"
    assert frame.empty


def test_repeating_the_newest_bar_overwrites_it_rather_than_appending():
    """A websocket reconnect resends the last closed bar, and the resend is the
    corrected copy -- so its values must win, in place."""
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

    assert len(buffer) == 3
    assert buffer.frame().iloc[-1].to_dict() == {
        "open": 11.0,
        "high": 14.0,
        "low": 9.0,
        "close": 13.0,
        "volume": 777.0,
    }
    assert buffer.replaced_duplicates == 1


def test_out_of_order_and_repeated_bars_are_absorbed_and_counted():
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
    bars = bars_from(synthetic_ohlcv(n=4))
    buffer = BarBuffer()
    for bar in bars[:3]:
        buffer.append(bar)

    first = buffer.frame()
    assert buffer.frame() is first
    buffer.append(bars[3])
    assert buffer.frame() is not first


def test_replacing_a_bar_invalidates_the_cached_frame():
    """A replacement keeps the length -- only the cache invalidation makes it visible."""
    bars = bars_from(synthetic_ohlcv(n=3))
    buffer = BarBuffer()
    for bar in bars:
        buffer.append(bar)

    stale = buffer.frame()
    buffer.append(replace(bars[-1], close=Decimal("12345")))
    fresh = buffer.frame()

    assert fresh is not stale
    assert fresh["close"].iloc[-1] == 12345.0
