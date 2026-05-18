from __future__ import annotations

import pandas as pd

from strategy_lab.strategies.turnaround_v1 import TurnaroundV1
from strategy_lab.strategies.turnaround_v2 import TurnaroundV2


def _frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100, 100, 98, 96, 101, 103, 105, 104],
            "high": [101, 101, 99, 103, 104, 106, 106, 105],
            "low": [99, 97, 95, 95, 100, 102, 103, 100],
            "close": [99, 98, 96, 102, 103, 105, 104, 101],
            "volume": [1_000] * 8,
        },
        index=index,
    )


def test_turnaround_v1_detects_base_reversals() -> None:
    signals = TurnaroundV1().generate_signals(_frame())

    assert signals.long_entries.iloc[3]
    assert signals.short_entries.iloc[6]
    assert signals.long_exits.equals(signals.short_entries)
    assert signals.short_exits.equals(signals.long_entries)
    assert signals.setup_stop_loss is not None
    assert signals.trend_failure_long_exits is not None
    assert signals.trend_failure_short_exits is not None
    assert signals.setup_stop_loss.iloc[3] == (102 - 95) / 102
    assert signals.setup_stop_loss.iloc[6] == (106 - 104) / 104


def test_turnaround_v2_keeps_signal_shape() -> None:
    signals = TurnaroundV2(ema_trend_span=2, ema_extension_span=2).generate_signals(_frame())

    assert signals.long_entries.index.equals(_frame().index)
    assert signals.short_entries.dtype == bool
