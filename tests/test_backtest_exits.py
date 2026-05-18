from __future__ import annotations

import pandas as pd

from strategy_lab.backtests.engine import _continuation_failure_exits


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
