from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.strategies.base import Strategy
from strategy_lab.timeframes import timeframe_to_pandas_freq


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

    pf = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=signals.long_entries,
        exits=signals.long_exits,
        short_entries=signals.short_entries,
        short_exits=signals.short_exits,
        init_cash=cash,
        fees=fees,
        slippage=slippage,
        freq=timeframe_to_pandas_freq(identity.timeframe),
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


def _json_safe(value: Any) -> Any:
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
