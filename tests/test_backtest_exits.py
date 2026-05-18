from __future__ import annotations

import pandas as pd

from strategy_lab.backtests.engine import _continuation_failure_exits, _sma_break_exits


def test_continuation_failure_exits_after_consecutive_adverse_closes() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "close": [100, 101, 100, 99, 98, 99, 100, 101],
        },
        index=index,
    )

    long_exits, short_exits = _continuation_failure_exits(df, failure_bars=3)

    assert long_exits.tolist() == [False, False, False, False, True, False, False, False]
    assert short_exits.tolist() == [False, False, False, False, False, False, False, True]


def test_sma_break_exits_when_close_crosses_below() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="W", tz="UTC")
    df = pd.DataFrame({"close": [100, 102, 101, 103, 98, 97]}, index=index)

    long_exits, short_exits = _sma_break_exits(df, sma_span=3)

    assert not long_exits.iloc[0]
    assert not long_exits.iloc[1]
    assert not long_exits.iloc[2]
    assert not long_exits.iloc[3]
    assert long_exits.iloc[4]
    assert long_exits.iloc[5]
    assert not short_exits.any()
