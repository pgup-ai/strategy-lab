from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.strategies.base import SignalSet, Strategy
from strategy_lab.timeframes import timeframe_to_pandas_freq


class ExitMode(str, Enum):
    CONTINUATION_FAILURE = "continuation_failure"
    OPPOSITE_SIGNAL_ONLY = "opposite_signal_only"
    SETUP_INVALIDATION_STOP = "setup_invalidation_stop"
    TREND_FAILURE = "trend_failure"
    TREND_STRUCTURE = "trend_structure"


@dataclass(frozen=True)
class BacktestResult:
    report_dir: Path
    stats_path: Path
    trades_path: Path
    equity_curve_path: Path
    plot_path: Path


def run_backtest(
    *,
    df: pd.DataFrame,
    strategy: Strategy,
    identity: MarketDataIdentity,
    fees: float = 0.0005,
    slippage: float = 0.0005,
    cash: float = 10_000.0,
    exit_mode: ExitMode | str = ExitMode.CONTINUATION_FAILURE,
    failure_bars: int = 4,
    position_pct: float = 0.95,
    report_root: Path = Path("reports"),
) -> BacktestResult:
    try:
        import vectorbt as vbt
    except ImportError as exc:
        raise RuntimeError("vectorbt is required to run backtests. Install with `pip install -e .`.") from exc

    if df.empty:
        raise ValueError(f"No candles loaded for {identity}")

    df = df.sort_index()
    signals = strategy.generate_signals(df)
    exit_mode = ExitMode(exit_mode)
    long_exits, short_exits = _exit_signals(
        df=df,
        signals=signals,
        exit_mode=exit_mode,
        failure_bars=failure_bars,
    )
    stop_kwargs = _stop_kwargs(df, signals.setup_stop_loss, exit_mode)

    size = _compute_entry_sizes(
        entries=signals.long_entries,
        close=df["close"],
        cash=cash,
        position_pct=position_pct,
        position_scale=signals.position_size,
    )

    pf = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=signals.long_entries,
        exits=long_exits,
        short_entries=signals.short_entries,
        short_exits=short_exits,
        size=size,
        init_cash=cash,
        fees=fees,
        slippage=slippage,
        freq=timeframe_to_pandas_freq(identity.timeframe),
        **stop_kwargs,
    )

    report_dir = _build_report_dir(report_root, identity, strategy.name)
    report_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "identity": asdict(identity),
        "strategy": strategy.name,
        "strategy_metadata": signals.metadata,
        "fees": fees,
        "slippage": slippage,
        "cash": cash,
        "exit_mode": exit_mode.value,
        "failure_bars": failure_bars,
        "position_pct": position_pct,
        "data_start": str(df.index.min()),
        "data_end": str(df.index.max()),
        "candle_count": int(len(df)),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    _write_json(report_dir / "config.json", config)

    stats = pf.stats()
    stats_path = report_dir / "stats.json"
    _write_json(stats_path, stats.to_dict())

    trades_path = report_dir / "trades.csv"
    trades = pf.trades.records_readable
    trades.to_csv(trades_path, index=False)

    equity_curve_path = report_dir / "equity_curve.csv"
    equity = pf.value()
    equity.to_frame("equity").to_csv(equity_curve_path)

    plot_path = report_dir / "plot.html"
    pf.plot().write_html(plot_path)

    return BacktestResult(
        report_dir=report_dir,
        stats_path=stats_path,
        trades_path=trades_path,
        equity_curve_path=equity_curve_path,
        plot_path=plot_path,
    )


def _exit_signals(
    *,
    df: pd.DataFrame,
    signals: SignalSet,
    exit_mode: ExitMode,
    failure_bars: int,
) -> tuple[pd.Series, pd.Series]:
    if exit_mode == ExitMode.OPPOSITE_SIGNAL_ONLY or exit_mode == ExitMode.SETUP_INVALIDATION_STOP:
        return signals.long_exits, signals.short_exits

    if exit_mode == ExitMode.CONTINUATION_FAILURE:
        continuation_long_exits, continuation_short_exits = _continuation_failure_exits(
            df,
            failure_bars=failure_bars,
        )
        return (
            signals.long_exits | continuation_long_exits,
            signals.short_exits | continuation_short_exits,
        )

    if exit_mode == ExitMode.TREND_FAILURE:
        if signals.trend_failure_long_exits is None or signals.trend_failure_short_exits is None:
            raise ValueError("Strategy did not provide trend failure exits")

        return (
            signals.long_exits | signals.trend_failure_long_exits,
            signals.short_exits | signals.trend_failure_short_exits,
        )

    if exit_mode == ExitMode.TREND_STRUCTURE:
        sma_span = signals.metadata.get("trend_sma_span", 40)
        sma_break_long, _ = _sma_break_exits(df, sma_span=sma_span)
        continuation_long_exits, _ = _continuation_failure_exits(
            df,
            failure_bars=failure_bars,
        )
        return (
            sma_break_long.fillna(False) | continuation_long_exits,
            pd.Series(False, index=df.index),
        )

    raise ValueError(f"Unsupported exit mode: {exit_mode}")


def _continuation_failure_exits(
    df: pd.DataFrame,
    *,
    failure_bars: int,
) -> tuple[pd.Series, pd.Series]:
    if failure_bars < 1:
        raise ValueError("failure_bars must be >= 1")

    close_change = df["close"].diff()
    lower_close = close_change < 0
    higher_close = close_change > 0
    long_exits = lower_close.rolling(failure_bars, min_periods=failure_bars).sum() == failure_bars
    short_exits = higher_close.rolling(failure_bars, min_periods=failure_bars).sum() == failure_bars
    return (
        long_exits.fillna(False),
        short_exits.fillna(False),
    )


def _sma_break_exits(
    df: pd.DataFrame,
    *,
    sma_span: int,
) -> tuple[pd.Series, pd.Series]:
    sma = df["close"].rolling(sma_span).mean()
    long_exits = df["close"] < sma
    return long_exits.fillna(False), pd.Series(False, index=df.index)


def _stop_kwargs(
    df: pd.DataFrame,
    setup_stop_loss: pd.Series | None,
    exit_mode: ExitMode,
) -> dict[str, Any]:
    if exit_mode in {
        ExitMode.CONTINUATION_FAILURE,
        ExitMode.OPPOSITE_SIGNAL_ONLY,
        ExitMode.TREND_FAILURE,
        ExitMode.TREND_STRUCTURE,
    }:
        return {}
    if setup_stop_loss is None:
        raise ValueError("Strategy did not provide setup invalidation stop levels")

    return {
        "price": df["close"],
        "open": df["open"],
        "high": df["high"],
        "low": df["low"],
        "sl_stop": setup_stop_loss,
        "sl_trail": False,
        "stop_entry_price": "Price",
        "stop_exit_price": "StopMarket",
        "upon_stop_exit": "Close",
    }


def _build_report_dir(report_root: Path, identity: MarketDataIdentity, strategy_name: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parts = [
        timestamp,
        identity.exchange,
        identity.market_type,
        identity.symbol,
        identity.timeframe,
        strategy_name,
    ]
    return report_root / _slug("_".join(parts))


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def _compute_entry_sizes(
    *,
    entries: pd.Series,
    close: pd.Series,
    cash: float,
    position_pct: float,
    position_scale: pd.Series | None,
) -> pd.Series:
    size = pd.Series(0.0, index=entries.index, dtype="float64")

    if position_scale is not None:
        fraction = position_pct * position_scale
        size.loc[entries] = (cash * fraction.loc[entries]) / close.loc[entries]
    else:
        size.loc[entries] = (cash * position_pct) / close.loc[entries]

    return size


def _json_safe(value: Any) -> Any:
    if value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
