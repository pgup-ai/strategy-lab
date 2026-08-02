from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from strategy_lab.backtests.engine import _continuation_failure_exits, _sma_break_exits, run_backtest
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.strategies.base import SignalSet


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


@dataclass(frozen=True)
class _ShortOnlyStrategy:
    name: str = "short_only_stub"

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        no_signal = pd.Series(False, index=df.index)
        short_entries = no_signal.copy()
        short_entries.iloc[2] = True
        short_exits = no_signal.copy()
        short_exits.iloc[7] = True
        return SignalSet(
            long_entries=no_signal.copy(),
            long_exits=no_signal.copy(),
            short_entries=short_entries,
            short_exits=short_exits,
        )


def _ohlcv_frame() -> pd.DataFrame:
    closes = [100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0, 98.0, 99.0]
    index = pd.date_range("2024-01-01", periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1_000.0] * len(closes),
        },
        index=index,
    )


_IDENTITY = MarketDataIdentity(
    exchange="test",
    market_type="spot",
    symbol="TEST/USDT",
    timeframe="1d",
)


def test_run_backtest_opens_short_trades(tmp_path) -> None:
    result = run_backtest(
        df=_ohlcv_frame(),
        strategy=_ShortOnlyStrategy(),
        identity=_IDENTITY,
        exit_mode="opposite_signal_only",
        report_root=tmp_path,
    )

    trades = pd.read_csv(result.trades_path)
    short_trades = trades[trades["Direction"] == "Short"]
    assert not short_trades.empty, "short entry produced no short trades"


def test_trend_structure_rejects_short_entries(tmp_path) -> None:
    with pytest.raises(ValueError, match="short"):
        run_backtest(
            df=_ohlcv_frame(),
            strategy=_ShortOnlyStrategy(),
            identity=_IDENTITY,
            exit_mode="trend_structure",
            report_root=tmp_path,
        )


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
