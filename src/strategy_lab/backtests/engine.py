from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy_lab.backtests.costs import CostModel, apply_funding, funding_ledger
from strategy_lab.backtests.report import render_report_html
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
    costs_path: Path
    funding_path: Path | None = None


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
    cost_model: CostModel | None = None,
    funding: pd.Series | None = None,
    cost_stress: Sequence[float] = (1.0,),
) -> BacktestResult:
    """Simulate ``strategy`` over ``df`` and write one run's artifacts.

    ``cost_model`` supersedes the scalar ``fees``/``slippage`` pair, which stays
    for callers that predate it; both describe the same per-fill frictions.

    ``funding`` is a rate series indexed by the venue's settlement instants. When
    supplied, it is charged against the notional held *into* each settlement bar
    and ``equity_curve.csv`` becomes the net-of-funding curve -- the tradeable
    one. When absent (equities, spot) the file is byte-identical to what this
    function produced before costs existed, and ``stats.json`` gains no keys,
    because a funding column on a market that has no funding is noise.

    ``cost_stress`` lists the fee/slippage multiples to compare. 1.0 is always
    present and is the headline run. Funding is never stressed: it is a market
    rate, so tripling it models a different instrument rather than a worse fill.
    """
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
        long_entries=signals.long_entries,
        short_entries=signals.short_entries,
        close=df["close"],
        cash=cash,
        position_pct=position_pct,
        position_scale=signals.position_size,
    )

    model = cost_model if cost_model is not None else CostModel(fee=fees, slippage=slippage)
    multiples = _stress_multiples(cost_stress)

    def _simulate(costs: CostModel):
        return vbt.Portfolio.from_signals(
            close=df["close"],
            entries=signals.long_entries,
            exits=long_exits,
            short_entries=signals.short_entries,
            short_exits=short_exits,
            size=size,
            init_cash=cash,
            fees=costs.fee,
            slippage=costs.slippage,
            freq=timeframe_to_pandas_freq(identity.timeframe),
            **stop_kwargs,
        )

    portfolios = {multiple: _simulate(model.stressed(multiple)) for multiple in multiples}
    pf = portfolios[1.0]

    flows = {
        multiple: _funding_flow(portfolio, df, funding)
        for multiple, portfolio in portfolios.items()
    }
    breakdown = [
        _cost_breakdown(
            multiple=multiple,
            pf=portfolios[multiple],
            flow=flows[multiple],
            costs=model.stressed(multiple),
            cash=cash,
        )
        for multiple in multiples
    ]
    headline = breakdown[multiples.index(1.0)]
    funding_applied = funding is not None and not funding.empty
    ledger = (
        funding_ledger(positions=_funding_notional(pf, df), funding=funding)
        if funding_applied
        else None
    )

    report_dir = build_report_dir(report_root, identity, strategy.name)
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
        "cost_model": asdict(model),
        "cost_stress": list(multiples),
        "funding_applied": funding_applied,
        "funding_settlements": int(len(ledger)) if ledger is not None else 0,
        "data_start": str(df.index.min()),
        "data_end": str(df.index.max()),
        "candle_count": int(len(df)),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    _write_json(report_dir / "config.json", config)

    costs_path = report_dir / "costs.json"
    _write_json(costs_path, {"funding_applied": funding_applied, "stress": breakdown})

    stats = pf.stats().to_dict()
    if funding_applied:
        stats["Funding Paid"] = headline["funding_paid"]
        stats["Net Return [%]"] = headline["net_return_pct"]
    stats_path = report_dir / "stats.json"
    _write_json(stats_path, stats)

    trades_path = report_dir / "trades.csv"
    trades = pf.trades.records_readable
    trades.to_csv(trades_path, index=False)

    funding_path = None
    if ledger is not None:
        funding_path = report_dir / "funding.csv"
        ledger.to_csv(funding_path)

    equity_curve_path = report_dir / "equity_curve.csv"
    equity = pf.value() + flows[1.0].cumsum()
    equity.to_frame("equity").to_csv(equity_curve_path)

    plot_path = report_dir / "plot.html"
    plot_path.write_text(
        render_report_html(
            df=df,
            trades=trades,
            equity=equity,
            config=config,
            stats=stats,
            costs={"funding_applied": funding_applied, "stress": breakdown},
        ),
        encoding="utf-8",
    )

    return BacktestResult(
        report_dir=report_dir,
        stats_path=stats_path,
        trades_path=trades_path,
        equity_curve_path=equity_curve_path,
        plot_path=plot_path,
        costs_path=costs_path,
        funding_path=funding_path,
    )


def _stress_multiples(cost_stress: Sequence[float]) -> list[float]:
    """Sorted, de-duplicated multiples, always including the 1.0 headline run.

    Without 1.0 the stress table would have nothing to be stressed *against*, and
    a reader comparing a 3x row to no baseline learns nothing.
    """
    multiples = {1.0}
    for multiple in cost_stress:
        value = float(multiple)
        if value <= 0:
            raise ValueError(f"cost stress multiples must be > 0, got {multiple!r}")
        multiples.add(value)
    return sorted(multiples)


def _funding_notional(pf, df: pd.DataFrame) -> pd.Series:
    """Signed notional held *into* each bar, valued at that bar's open.

    Fills land at the bar's close, so ``assets()`` at bar *t* is the position
    held over bar *t+1* -- the shift is what makes the charge causal rather than
    settling funding against a position taken after the settlement happened.

    The open is the mark at the instant a settlement on a bar boundary occurs,
    which is every settlement when bars divide the funding interval (4h bars, 8h
    funding). Bars coarser than the interval carry several settlements and mark
    them all at the bar's open; that approximation moves a charge by a fraction
    of a percent of itself and is not worth a mark-price series that is NULL for
    60% of stored history.
    """
    return (pf.assets().shift(1) * df["open"]).fillna(0.0)


def _funding_flow(pf, df: pd.DataFrame, funding: pd.Series | None) -> pd.Series:
    if funding is None or funding.empty:
        return pd.Series(0.0, index=df.index, dtype="float64")
    return apply_funding(positions=_funding_notional(pf, df), funding=funding)


def _cost_breakdown(
    *,
    multiple: float,
    pf,
    flow: pd.Series,
    costs: CostModel,
    cash: float,
) -> dict[str, float]:
    """What the run earned before costs, what each cost took, and what is left.

    Sizing is non-compounding -- entries are sized from *initial* cash, never
    from current equity -- so every cost is a flat cash deduction and gross is
    exactly net plus the three costs back. That identity is what makes this a
    reconciliation rather than an estimate.
    """
    fees_paid = float(pf.stats()["Total Fees Paid"])
    slippage_paid = _slippage_paid(pf, costs.slippage)
    funding_paid = float(-flow.sum())
    net_final = float(pf.value().iloc[-1]) + float(flow.sum())
    gross_final = net_final + fees_paid + slippage_paid + funding_paid
    return {
        "multiple": multiple,
        "fee_rate": costs.fee,
        "slippage_rate": costs.slippage,
        "gross_return_pct": (gross_final / cash - 1.0) * 100.0,
        "fees_paid": fees_paid,
        "slippage_paid": slippage_paid,
        "funding_paid": funding_paid,
        "net_return_pct": (net_final / cash - 1.0) * 100.0,
        "net_final_equity": net_final,
    }


def _slippage_paid(pf, slippage: float) -> float:
    """Currency lost to slippage, recovered from the fill prices it moved.

    vectorbt folds slippage into the fill rather than reporting it, so it is
    invisible next to ``Total Fees Paid`` unless it is backed out: a buy filled
    at ``reference * (1 + slippage)`` and a sell at ``reference * (1 -
    slippage)``, always against the trader.
    """
    if slippage == 0.0:
        return 0.0
    orders = pf.orders.records_readable
    if orders.empty:
        return 0.0

    direction = np.where(orders["Side"].to_numpy() == "Buy", 1.0, -1.0)
    reference = orders["Price"].to_numpy(dtype="float64") / (1.0 + direction * slippage)
    return float((orders["Size"].to_numpy(dtype="float64") * reference * slippage).sum())


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
        if signals.short_entries.any():
            raise ValueError(
                "trend_structure provides no short exits; "
                "disable shorts or use a long-only strategy"
            )
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


def build_report_dir(report_root: Path, identity: MarketDataIdentity, label: str) -> Path:
    """Timestamped, identity-stamped output directory for one run's artifacts.

    Shared with the parameter sweep so both write into the same, sortable
    ``reports/`` layout; ``label`` is what distinguishes them.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parts = [
        timestamp,
        identity.exchange,
        identity.market_type,
        identity.symbol,
        identity.timeframe,
        label,
    ]
    return report_root / _slug("_".join(parts))


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def _compute_entry_sizes(
    *,
    long_entries: pd.Series,
    short_entries: pd.Series,
    close: pd.Series,
    cash: float,
    position_pct: float,
    position_scale: pd.Series | None,
) -> pd.Series:
    entries = long_entries | short_entries
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
