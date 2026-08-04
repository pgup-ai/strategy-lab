from __future__ import annotations

import json
import math
import re
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy_lab.backtests.costs import CostModel, apply_funding, funding_ledger
from strategy_lab.backtests.report import render_report_html
from strategy_lab.backtests.sizing import (
    DEFAULT_VOL_SPAN,
    SizeMode,
    vol_warmup_bars,
    volatility_target_weights,
)
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.strategies.base import SignalSet, Strategy
from strategy_lab.timeframes import timeframe_to_bars_per_year, timeframe_to_pandas_freq


_COSTLESS = CostModel(fee=0.0, slippage=0.0)


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
    size_mode: SizeMode | str = SizeMode.FIXED,
    vol_target: float = 0.30,
    max_weight: float = 2.0,
    vol_span: int = DEFAULT_VOL_SPAN,
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

    ``size_mode`` selects between fixed-fractional entries and volatility-scaled
    ones; ``vol_target``, ``max_weight`` and ``vol_span`` configure the latter
    and are ignored under ``fixed``, which is bit-for-bit the behaviour that
    predates them. ``vol-scaled-entry`` scales the *entry* and nothing after it:
    ``from_signals`` fills once per state change, so an open position keeps the
    weight it was opened with however far volatility moves. See
    ``backtests/sizing.py`` for the measurement and for where the real fix lives.
    """
    try:
        import vectorbt as vbt
    except ImportError as exc:
        raise RuntimeError("vectorbt is required to run backtests. Install with `pip install -e .`.") from exc

    if df.empty:
        raise ValueError(f"No candles loaded for {identity}")

    df = df.sort_index()
    exit_mode = ExitMode(exit_mode)
    size_mode = SizeMode(size_mode)
    signals = strategy.generate_signals(df)
    signals = _vol_scaled_entry_weights(
        signals,
        df=df,
        strategy=strategy,
        timeframe=identity.timeframe,
        size_mode=size_mode,
        vol_target=vol_target,
        max_weight=max_weight,
        vol_span=vol_span,
        position_pct=position_pct,
    )
    # After the sizing check, so an incompatible pair of flags is reported as
    # such rather than as whichever data problem the frame happens to have too.
    warmup = _warmup_bars(strategy, df, size_mode=size_mode, vol_span=vol_span)
    long_exits, short_exits = _exit_signals(
        df=df,
        signals=signals,
        exit_mode=exit_mode,
        failure_bars=failure_bars,
    )
    stop_kwargs = _stop_kwargs(df, signals.setup_stop_loss, exit_mode)

    signals, long_exits, short_exits = _mask_warmup(signals, long_exits, short_exits, warmup)

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
    # One costless simulation serves every stress row -- scaling a rate that is
    # already zero changes nothing -- and a run priced at zero already is its
    # own gross, so the second portfolio is built only when it can differ.
    costless = pf if model == _COSTLESS else _simulate(_COSTLESS)
    gross_final = float(costless.value().iloc[-1])

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
            gross_final=gross_final,
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
        "warmup_bars": warmup,
        "fees": fees,
        "slippage": slippage,
        "cash": cash,
        "exit_mode": exit_mode.value,
        "failure_bars": failure_bars,
        "position_pct": position_pct,
        "size_mode": size_mode.value,
        **_sizing_config(
            size_mode,
            timeframe=identity.timeframe,
            vol_target=vol_target,
            max_weight=max_weight,
            vol_span=vol_span,
            position_pct=position_pct,
        ),
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

    costs = {"funding_applied": funding_applied, "stress": breakdown}
    costs_path = report_dir / "costs.json"
    _write_json(costs_path, costs)

    equity = pf.value() + flows[1.0].cumsum()

    stats = pf.stats().to_dict()
    if funding_applied:
        stats = _split_by_curve(
            stats,
            gross_equity=pf.value(),
            net_equity=equity,
            cash=cash,
            freq=timeframe_to_pandas_freq(identity.timeframe),
        )
        stats["Funding Paid"] = headline["funding_paid"]
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
    equity.to_frame("equity").to_csv(equity_curve_path)

    plot_path = report_dir / "plot.html"
    plot_path.write_text(
        render_report_html(
            df=df,
            trades=trades,
            equity=equity,
            config=config,
            stats=stats,
            costs=costs,
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


def _warmup_bars(
    strategy: Strategy,
    df: pd.DataFrame,
    *,
    size_mode: SizeMode,
    vol_span: int,
) -> int:
    """Leading bars nothing may trade on, from whichever estimator is slowest.

    Two claims compete and the deeper one binds. The strategy's own
    ``warmup_bars`` is one. The other belongs to the sizing layer: under
    ``vol-scaled-entry`` the weight comes from an ``ewm(adjust=False)``
    volatility estimate, which needs roughly 20x its span before it means
    anything. Sizing an entry off the shorter of the two is the same defect in
    a different layer -- an entry taken on a number that has not converged.

    Refused rather than clamped when the frame is entirely warmup: a run that
    masks every bar produces an empty ``trades.csv``, a flat equity curve and a
    ``stats.json`` full of zeros, none of which reads as "there was no data" --
    it reads as a strategy that declined to trade. The sweep already refuses the
    same case for the same reason.
    """
    budgets = {f"{strategy.name} declares {strategy.warmup_bars}": int(strategy.warmup_bars)}
    if size_mode is not SizeMode.FIXED:
        estimator = vol_warmup_bars(vol_span)
        budgets[f"the span-{vol_span} volatility estimator needs {estimator}"] = estimator

    reason, warmup = max(budgets.items(), key=lambda item: item[1])
    if warmup >= len(df):
        raise ValueError(
            f"{reason} warmup bars but the frame has {len(df)}; every bar would "
            f"be masked and the run would report a flat curve rather than an "
            f"absence of data"
        )
    return warmup


def _mask_warmup(
    signals: SignalSet,
    long_exits: pd.Series,
    short_exits: pd.Series,
    warmup: int,
) -> tuple[SignalSet, pd.Series, pd.Series]:
    """Silence every entry and exit until the declared warmup has elapsed.

    ``warmup_bars`` is a measured claim that an indicator has not converged yet
    -- for ``ewm(adjust=False)`` roughly 20x the span, because the recursion
    decays its seed rather than dropping it. The sweep and the replay runner
    both act on that claim; the engine did not, so ``ema_cross`` on the
    canonical 15,118-bar perp frame traded **3,840 bars, 25.4% of the run**, on
    EMAs it declares are still wrong.

    Both sides are masked, matching ``StrategyRunner.on_event``, which emits
    nothing at all until ``len(buffer) > warmup_bars``. Only the entries change
    the simulation -- ``signals_to_size_nb`` does nothing with an exit while the
    position is flat, and a stop cannot fire on a position that was never opened
    -- so masking the exits is what makes the two paths describe the same signal
    set rather than merely reach the same P&L.
    """
    mask = np.arange(len(signals.long_entries)) >= warmup

    def masked(series: pd.Series) -> pd.Series:
        return series & mask

    return (
        replace(
            signals,
            long_entries=masked(signals.long_entries),
            long_exits=masked(signals.long_exits),
            short_entries=masked(signals.short_entries),
            short_exits=masked(signals.short_exits),
        ),
        masked(long_exits),
        masked(short_exits),
    )


def _vol_scaled_entry_weights(
    signals: SignalSet,
    *,
    df: pd.DataFrame,
    strategy: Strategy,
    timeframe: str,
    size_mode: SizeMode,
    vol_target: float,
    max_weight: float,
    vol_span: int,
    position_pct: float,
) -> SignalSet:
    """``signals`` with ``position_size`` replaced by inverse-volatility weights.

    Only the weight on each entry bar is executed -- ``from_signals`` does not
    resize a position it has already opened -- so this scales entries and does
    not target risk continuously. ``backtests/sizing.py`` has the measurement.

    A strategy that already ships a ``position_size`` is refused rather than
    overridden or multiplied. ``trend_rider_v1_deepseek_v4_pro`` is the case
    that exists: its scale is ``0.06 / (ATR/close)``, itself an inverse-vol
    weight, so multiplying the two would size on ``1 / vol**2`` and land nowhere
    near ``vol_target`` -- the run would still be labelled for 30% while sizing
    entries at neither 30% nor anything stable. Overriding instead would
    silently delete a documented part of the strategy. Neither failure is
    visible in the artifacts, so the combination is rejected at the boundary.
    """
    if size_mode is SizeMode.FIXED:
        return signals
    if signals.position_size is not None:
        raise ValueError(
            f"{strategy.name} already sizes its own positions, and stacking "
            f"volatility scaling on top sizes entries at neither its scale nor "
            f"{vol_target:.0%} annualized volatility. Run it with "
            f"--size-mode fixed, or vol-scale a strategy that leaves sizing to "
            f"the engine."
        )

    executable = _executable_max_weight(max_weight, position_pct)
    if executable < max_weight:
        warnings.warn(
            f"max_weight {max_weight:g} is above the {executable:.4g} this book "
            f"can fill at position_pct {position_pct:g}: an entry is sized as "
            f"cash x position_pct x weight and there is no leverage, so a calm "
            f"regime would request more long notional than initial cash covers "
            f"and be filled short of it. Weights are capped at {executable:.4g}, "
            f"recorded as max_weight_effective in config.json.",
            stacklevel=3,
        )

    weights = volatility_target_weights(
        df["close"].pct_change(),
        target_annual_vol=vol_target,
        bars_per_year=timeframe_to_bars_per_year(timeframe),
        span=vol_span,
        max_weight=executable,
    )
    return replace(signals, position_size=weights)


def _executable_max_weight(max_weight: float, position_pct: float) -> float:
    """``max_weight`` clipped to the weight the book's own cash can fill.

    Without this the run silently does something other than what it advertises:
    vectorbt fills what the cash covers rather than rejecting the order, so a
    ``--max-weight 2.0`` run at the default 95% deployment sizes for 30% vol on
    paper and holds whatever fit in practice, with ``config.json`` recording the
    number that was asked for.
    """
    return min(max_weight, 1.0 / position_pct)


def _sizing_config(
    size_mode: SizeMode,
    *,
    timeframe: str,
    vol_target: float,
    max_weight: float,
    vol_span: int,
    position_pct: float,
) -> dict[str, Any]:
    """The vol-scaling settings, recorded only on runs that actually used them.

    Both caps are recorded: ``max_weight`` is what was asked for and
    ``max_weight_effective`` is what the book could fill, so a run stays
    reproducible even when the two disagree.
    """
    if size_mode is SizeMode.FIXED:
        return {}
    return {
        "vol_target": vol_target,
        "max_weight": max_weight,
        "max_weight_effective": _executable_max_weight(max_weight, position_pct),
        "vol_span": vol_span,
        "vol_warmup_bars": vol_warmup_bars(vol_span),
        "bars_per_year": timeframe_to_bars_per_year(timeframe),
    }


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


_PATH_STATS = (
    "Total Return [%]",
    "End Value",
    "Max Drawdown [%]",
    "Max Drawdown Duration",
    "Annualized Return [%]",
    "Annualized Volatility [%]",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Calmar Ratio",
    "Omega Ratio",
)


def _equity_risk(equity: pd.Series, *, cash: float, freq: str) -> dict[str, Any]:
    """Risk statistics for one equity path, on vectorbt's own definitions.

    Reusing the returns accessor rather than hand-rolling a Sharpe is what makes
    the gross and net columns comparable: the same estimator sees both curves,
    so any difference between them is funding and nothing else.

    The first bar's return is measured against ``cash`` rather than dropped,
    because ``pf.stats()`` counts it and a curve scored over one fewer
    observation would differ from the simulated book for a reason that has
    nothing to do with funding.
    """
    returns = equity.pct_change()
    returns.iloc[0] = equity.iloc[0] / cash - 1.0
    acc = returns.vbt.returns(freq=freq)
    return {
        "Total Return [%]": float(acc.total()) * 100.0,
        "End Value": float(equity.iloc[-1]),
        "Max Drawdown [%]": -float(acc.max_drawdown()) * 100.0,
        "Max Drawdown Duration": acc.drawdowns.max_duration(),
        "Annualized Return [%]": float(acc.annualized()) * 100.0,
        "Annualized Volatility [%]": float(acc.annualized_volatility()) * 100.0,
        "Sharpe Ratio": float(acc.sharpe_ratio()),
        "Sortino Ratio": float(acc.sortino_ratio()),
        "Calmar Ratio": float(acc.calmar_ratio()),
        "Omega Ratio": float(acc.omega_ratio()),
    }


def _split_by_curve(
    stats: dict[str, Any],
    *,
    gross_equity: pd.Series,
    net_equity: pd.Series,
    cash: float,
    freq: str,
) -> dict[str, Any]:
    """``stats`` with every path statistic named for the curve it describes.

    Funding is settled outside the simulation, so ``pf.stats()`` measures
    drawdown, Sharpe and the rest on a curve the report does not plot. On the
    BTC perp that gap is not cosmetic -- funding runs to a third of initial
    capital -- so on a funded run no path statistic is left bare: each appears
    once per curve, and the reader is never asked which one they are holding.
    Trade-level statistics are untouched, because funding is a carry on the
    book rather than a cost attributable to any trade.
    """
    split = {key: value for key, value in stats.items() if key not in _PATH_STATS}
    for curve, equity in (("gross", gross_equity), ("net", net_equity)):
        for key, value in _equity_risk(equity, cash=cash, freq=freq).items():
            split[f"{key} ({curve} of funding)"] = value
    split["Net Return [%]"] = split.pop("Total Return [%] (net of funding)")
    return split


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
    gross_final: float,
) -> dict[str, float]:
    """What the run earned before costs, what each cost took, and what is left.

    ``gross_final`` comes from a *second simulation* priced at zero, not from
    adding the costs back onto net. The two differ whenever a cost changed a
    fill: at 95% deployment there is not enough cash to buy the same size at a
    worse price, so the cost-bearing book holds less and its P&L is scaled down
    on top of the fee it paid. ``size_effect`` is exactly that remainder -- a
    drag on a profitable run, a credit on a losing one -- and naming it keeps
    the waterfall an identity rather than an approximation that quietly absorbs
    the difference into the gross figure.
    """
    fees_paid = float(pf.stats()["Total Fees Paid"])
    slippage_paid = _slippage_paid(pf, costs.slippage)
    funding_paid = float(-flow.sum())
    net_final = float(pf.value().iloc[-1]) + float(flow.sum())
    return {
        "multiple": multiple,
        "fee_rate": costs.fee,
        "slippage_rate": costs.slippage,
        "gross_return_pct": (gross_final / cash - 1.0) * 100.0,
        "fees_paid": fees_paid,
        "slippage_paid": slippage_paid,
        "funding_paid": funding_paid,
        "size_effect": gross_final - net_final - fees_paid - slippage_paid - funding_paid,
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
