"""The engine must not trade on indicators the strategy declares unconverged.

``warmup_bars`` is a measured claim, not a formality: for ``ewm(adjust=False)``
it runs to roughly 20x the span, because the recursion decays its seed rather
than dropping it. The sweep slices its returns at the warmup and
``StrategyRunner`` emits nothing until the buffer is past it; ``run_backtest``
did neither, so ``ema_cross`` on the canonical 15,118-bar perp frame traded
3,840 bars -- 25.4% of the run -- on EMAs it declares are still wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
import pytest

from strategy_lab.backtests.engine import _mask_warmup, run_backtest
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.strategies.base import SignalSet


_IDENTITY = MarketDataIdentity(
    exchange="binance", market_type="spot", symbol="BTC/USDT", timeframe="1d"
)

_WARMUP = 5
_EARLY_ENTRY = 2
_LATE_ENTRY = 8


@dataclass(frozen=True)
class _TwoEntryStrategy:
    """Enters once inside the warmup and once outside it.

    Both entries are hand-placed, so the only thing that can remove the first is
    the mask -- there is no indicator here whose NaNs could be doing the work.
    """

    name: str = "two_entry_stub"
    version: str = "1.0.0"
    warmup_bars: int = _WARMUP

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        flat = pd.Series(False, index=df.index)
        long_entries = flat.copy()
        long_exits = flat.copy()
        for entry in (_EARLY_ENTRY, _LATE_ENTRY):
            if entry + 2 < len(df):
                long_entries.iloc[entry] = True
                long_exits.iloc[entry + 2] = True
        return SignalSet(
            long_entries=long_entries,
            long_exits=long_exits,
            short_entries=flat.copy(),
            short_exits=flat.copy(),
        )


def _frame(n: int = 14) -> pd.DataFrame:
    closes = [100.0 + i for i in range(n)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1_000.0] * n,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC", name="timestamp"),
    )


def test_no_trade_lands_before_the_declared_warmup(tmp_path) -> None:
    df = _frame()
    result = run_backtest(
        df=df,
        strategy=_TwoEntryStrategy(),
        identity=_IDENTITY,
        exit_mode="opposite_signal_only",
        report_root=tmp_path,
    )

    trades = pd.read_csv(result.trades_path)
    assert not trades.empty, (
        "the post-warmup entry produced no trade either, so this test would "
        "pass on an engine that simply never trades"
    )
    entered = pd.to_datetime(trades["Entry Timestamp"], utc=True)
    assert entered.min() == df.index[_LATE_ENTRY], (
        f"first trade opened at {entered.min()}, but bar {_EARLY_ENTRY} is "
        f"inside the {_WARMUP}-bar warmup and must not have been traded"
    )


def test_a_frame_that_is_entirely_warmup_is_refused(tmp_path) -> None:
    """A fully masked run reads as a strategy that declined to trade.

    Empty trades, a flat curve and zeroed statistics are indistinguishable from
    a real result, which is why this raises instead of writing them.
    """
    with pytest.raises(ValueError, match="warmup") as raised:
        run_backtest(
            df=_frame(n=_WARMUP),
            strategy=_TwoEntryStrategy(),
            identity=_IDENTITY,
            exit_mode="opposite_signal_only",
            report_root=tmp_path,
        )
    message = str(raised.value)
    assert "two_entry_stub" in message and str(_WARMUP) in message, (
        f"error names neither the strategy nor the budget: {message}"
    )


def test_the_warmup_is_recorded_in_the_run_config(tmp_path) -> None:
    result = run_backtest(
        df=_frame(),
        strategy=_TwoEntryStrategy(),
        identity=_IDENTITY,
        exit_mode="opposite_signal_only",
        report_root=tmp_path,
    )
    config = json.loads((result.report_dir / "config.json").read_text())
    assert config["warmup_bars"] == _WARMUP, (
        "config.json is the reproducibility record; a run whose masked prefix "
        "is not written down cannot be reproduced from it"
    )


def test_every_side_is_masked_not_only_the_entries() -> None:
    """Both sides, matching ``StrategyRunner``, which emits nothing at all.

    Only the entries can change the P&L -- ``signals_to_size_nb`` does nothing
    with an exit while the position is flat -- so an entries-only mask would
    reach the same equity while leaving the two execution paths describing
    different signal sets.
    """
    index = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC", name="timestamp")
    always = pd.Series(True, index=index)
    signals = SignalSet(
        long_entries=always.copy(),
        long_exits=always.copy(),
        short_entries=always.copy(),
        short_exits=always.copy(),
    )

    masked, long_exits, short_exits = _mask_warmup(signals, always.copy(), always.copy(), 4)

    sides = {
        "long_entries": masked.long_entries,
        "long_exits": masked.long_exits,
        "short_entries": masked.short_entries,
        "short_exits": masked.short_exits,
        "derived long_exits": long_exits,
        "derived short_exits": short_exits,
    }
    for name, series in sides.items():
        assert series.tolist() == [False, False, False, False, True, True], (
            f"{name} was not masked through the 4-bar warmup: {series.tolist()}"
        )
