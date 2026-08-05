"""Execution for the continuous-exposure contract.

``run_backtest`` drives ``vbt.Portfolio.from_signals``, which fills once per
state change and consumes a size only on the bar that opens a position. This
module drives ``from_orders`` with ``size_type="targetpercent"`` instead, which
reads the size on *every* bar and issues whatever order moves the book to it.
Measured against the installed vectorbt (1.0.0) on a flat 10-bar frame:

    target  : [0.0, 0.3, 0.7, 1.0, 1.0, 1.0, 0.55, 0.55, 0.2, 0.0]
    position: [0.0, 30., 70., 100., 100., 100., 55., 55., 20., 0.0]
    orders  : 6      -- the boolean path gives 1 for the same input

The two paths are siblings, not successors: the four original strategies keep
``from_signals`` and their results of record, and nothing here changes them.

**Two consequences of ``targetpercent`` a reader has to carry.**

*It is a fraction of current equity, so it compounds*, where ``run_backtest``
sizes every entry from *initial* cash. A profitable continuous run therefore
grows its notional and a boolean one does not, which is a difference between the
two paths that has nothing to do with the taper. Any comparison of a continuous
strategy against a boolean one is measuring both effects at once and should say
so.

*A target that never changes still trades.* Equity moves, so holding a constant
fraction of it means trading to stay there. Price is the obvious mover --
measured, a constant 0.5 target on a six-bar ramp from 100 to 200 issues six
orders, selling into strength -- but **costs move it too**: the same 10-bar
taper that issues 6 orders costlessly issues 9 at a 10 bps fee, the three extras
being dust rebalances after each fee shrank the equity the fraction is taken of.

That is not a rounding effect at scale. Measured on the stored 15,128-bar
BTC/USDT 4h perp history with a target snapped to seven levels: **4,996 target
changes, 14,404 orders** -- 96% of live bars trade -- and at 5 bps fee plus 5 bps
slippage the fills cost 18,850 against 10,000 of initial capital, taking a
costless 21,932 final equity down to 557. Tracking itself is exact (the position
matches the target to 5.6e-16 costlessly, 1.5e-3 at 10 bps), so this is the
price of *holding a fraction rather than a quantity*, not an execution defect.
An order count here is therefore not a count of a strategy's decisions, and
turnover on this path is not comparable to a boolean path's trade count.

Funding, fees and slippage all come from ``backtests/costs.py`` -- the same
containment matching and the same held-notional convention the boolean path
uses, so the two can be compared without wondering whether their cost models
agree.

No report directory is written. This is the execution path; ``config`` on the
result is the reproducibility record, and the artifacts land when a registered
strategy needs them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from strategy_lab.backtests.costs import (
    CostModel,
    apply_funding,
    held_notional,
    slippage_paid,
)
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.strategies.exposure import ExposureStrategy, TargetExposure
from strategy_lab.timeframes import timeframe_to_pandas_freq


@dataclass(frozen=True)
class ExposureBacktestResult:
    """One continuous-exposure run, in memory.

    ``equity`` is net of funding -- the tradeable curve, matching what
    ``run_backtest`` writes to ``equity_curve.csv``. ``funding_flow`` is the
    per-bar cash flow behind it, so the gross curve is
    ``equity - funding_flow.cumsum()`` and no reader has to guess which curve a
    number describes.

    ``position_fraction`` is the signed value of the position over equity, i.e.
    what the target asked for, measured on the book that resulted. Comparing the
    two is the phase's gate.
    """

    target: pd.Series
    position: pd.Series
    position_fraction: pd.Series
    orders: pd.DataFrame
    order_count: int
    equity: pd.Series
    funding_flow: pd.Series
    fees_paid: float
    slippage_paid: float
    funding_paid: float
    config: dict[str, Any]


def run_exposure_backtest(
    *,
    df: pd.DataFrame,
    strategy: ExposureStrategy,
    identity: MarketDataIdentity,
    cash: float = 10_000.0,
    position_pct: float = 0.95,
    cost_model: CostModel | None = None,
    funding: pd.Series | None = None,
) -> ExposureBacktestResult:
    """Track ``strategy``'s target exposure over ``df``, charging costs and funding.

    Mirrors ``run_backtest``'s signature where the two paths share a concept, so
    a caller moving between them is not surprised: ``cash`` and ``position_pct``
    mean what they mean there, and ``position_pct`` scales the target, so a
    target of 1.0 at the 0.95 default asks for 95% of equity.

    The scalar ``fees``/``slippage`` pair is deliberately absent -- it exists on
    ``run_backtest`` only for callers that predate ``CostModel``, and this path
    has none. A cost-stress table is likewise not built in: call this once per
    ``cost_model.stressed(multiple)``, which is the same thing without a second
    stress implementation.

    ``funding`` is settled outside the simulation exactly as in the boolean
    path, so the position is unaffected by the carry it pays and ``equity`` is
    the only place funding appears.
    """
    try:
        import vectorbt as vbt
    except ImportError as exc:
        raise RuntimeError(
            "vectorbt is required to run backtests. Install with `pip install -e .`."
        ) from exc

    if df.empty:
        raise ValueError(f"No candles loaded for {identity}")

    df = df.sort_index()
    exposure = strategy.compute_target(df)
    _validate_alignment(exposure, df, strategy)
    target = _flat_through_warmup(
        exposure.target, warmup_bars=strategy.warmup_bars, strategy=strategy
    )

    model = cost_model if cost_model is not None else CostModel()
    pf = vbt.Portfolio.from_orders(
        close=df["close"],
        size=target * position_pct,
        size_type="targetpercent",
        init_cash=cash,
        fees=model.fee,
        slippage=model.slippage,
        freq=timeframe_to_pandas_freq(identity.timeframe),
    )

    orders = pf.orders.records_readable
    gross_equity = pf.value()
    notional = held_notional(pf.assets(), df["open"])
    flow = (
        apply_funding(positions=notional, funding=funding)
        if funding is not None and not funding.empty
        else pd.Series(0.0, index=df.index, dtype="float64")
    )

    return ExposureBacktestResult(
        target=target,
        position=pf.assets(),
        position_fraction=pf.assets() * df["close"] / gross_equity,
        orders=orders,
        order_count=int(len(orders)),
        equity=gross_equity + flow.cumsum(),
        funding_flow=flow,
        fees_paid=float(pf.stats()["Total Fees Paid"]),
        slippage_paid=slippage_paid(orders, model.slippage),
        funding_paid=float(-flow.sum()),
        config={
            "identity": asdict(identity),
            "strategy": strategy.name,
            "strategy_version": strategy.version,
            "strategy_metadata": exposure.metadata,
            "contract": "target_exposure",
            "warmup_bars": int(strategy.warmup_bars),
            "cash": cash,
            "position_pct": position_pct,
            "cost_model": asdict(model),
            "funding_applied": bool(funding is not None and not funding.empty),
            "data_start": str(df.index.min()),
            "data_end": str(df.index.max()),
            "candle_count": int(len(df)),
        },
    )


def _validate_alignment(exposure: TargetExposure, df: pd.DataFrame, strategy) -> None:
    """Refuse a target that is not one value per candle, on the candles' own index.

    ``from_orders`` takes ``size`` positionally against ``close``, so a target
    that is merely the right *length* but a different index executes silently
    against the wrong bars. Length alone is the check that would let a strategy
    reindexed to a funding schedule, or one that dropped its warmup rows instead
    of flattening them, run to completion and report a number.
    """
    if not exposure.target.index.equals(df.index):
        raise ValueError(
            f"{strategy.name} returned a target on a different index from the "
            f"candles ({len(exposure.target)} rows against {len(df)}); it is "
            f"executed positionally against close, so a misaligned target would "
            f"trade the right sizes on the wrong bars"
        )


def _flat_through_warmup(target: pd.Series, *, warmup_bars: int, strategy) -> pd.Series:
    """Hold nothing until the declared warmup has elapsed.

    The same claim ``engine._mask_warmup`` enforces for the boolean path, and
    enforced here for the same reason: ``warmup_bars`` is a measured statement
    that the strategy's indicators have not converged, and a run that trades
    before it is trading on numbers the strategy itself says are wrong. The
    strategy is expected to emit 0.0 there already -- the contract forbids NaN --
    so this is the engine holding it to its own declaration rather than a
    conversion.

    Refused rather than clamped when the frame is entirely warmup, matching
    ``engine._warmup_bars``: a run flat on every bar produces an empty order
    book and a flat curve, which reads as a strategy that declined to trade
    rather than as an absence of data.
    """
    if warmup_bars >= len(target):
        raise ValueError(
            f"{strategy.name} declares {warmup_bars} warmup bars but the frame has "
            f"{len(target)}; every bar would be flat and the run would report a "
            f"flat curve rather than an absence of data"
        )
    flattened = target.copy()
    flattened.iloc[:warmup_bars] = 0.0
    return flattened


__all__ = ["ExposureBacktestResult", "run_exposure_backtest"]
