"""Execution for the continuous-exposure contract.

``run_backtest`` drives ``vbt.Portfolio.from_signals``, which fills once per
state change and consumes a size only on the bar that opens a position. This
module drives ``from_orders`` with ``size_type="targetvalue"`` instead, which
reads a size on *every* bar rather than only at an entry, and issues whatever
order moves the book to it. Measured against the installed vectorbt (1.0.0) on a
flat 10-bar frame:

    target  : [0.0, 0.3, 0.7, 1.0, 1.0, 1.0, 0.55, 0.55, 0.2, 0.0]
    position: [0.0, 30., 70., 100., 100., 100., 55., 55., 20., 0.0]
    orders  : 6      -- the boolean path gives 1 for the same input

The two paths are siblings, not successors: the four original strategies keep
``from_signals`` and their results of record, and nothing here changes them.

**The target is a fraction of the risk budget, and the budget is initial cash.**
The order size is ``target x position_pct x cash``, which is CLAUDE.md's rule for
the boolean path -- entries sized from *initial* cash, never from current equity
-- restated for this one. ``targetpercent`` is the obvious alternative and the
wrong one: it is a fraction of *current* equity, so a profitable run silently
grows its notional, and a continuous strategy compared against a boolean one
would be measuring a change of sizing model alongside whatever it meant to
measure. One consequence of the anchor is worth carrying: a target of 1.0 asks
for ``position_pct x cash`` however far equity has fallen, so a deep enough
drawdown leaves the book unable to afford it and it fills what cash covers.
Measured on the history below, that bound bit on 1 decision bar in 4,531.

**The rebalance band is a model choice, not a cost optimisation.** A target
reaches the engine through ``rebalance_threshold``: a value is submitted only
once it has moved at least that far from the last value *submitted*, and every
other bar is ``NaN``, which ``from_orders`` reads as "no order, hold what you
held". Between decisions the book therefore holds a fixed *quantity*, and its
fraction of equity **rises with a winner and falls with a loser** -- measured
below, a median drift of 2.5 percentage points of equity away from the target,
and a maximum of 26.0. That is why ``position_fraction`` is what the book drifted
to rather than what the target asked for.

Rebalancing an unchanged target on every bar does the opposite -- to a fixed
notional here, to a fixed fraction under ``targetpercent``, and either way it
trims winners and adds to losers: a mean-reversion overlay bolted onto a strategy
whose whole thesis is riding a trend. ``rebalance_threshold=0.0`` is exactly that
overlay, since it submits on every bar -- a usable setting, and the honest name
for what it does, but not a neutral default. Nor is it a cost story: costless
and funding-free, the same target ends at 20,742.99 at the 0.05 default against
20,261.47 at band 0.0, so the overlay moves the result before a single fee is
charged.

Measured on the stored 15,128-bar BTC/USDT 4h perp history, funding applied, at
the default cash and cost model, with the ``_EwmTaper`` from
``tests/test_exposure_determinism.py`` as the target -- an ``ewm(span=30)``
momentum snapped to the charter's seven levels:

    band    decisions   orders   final equity
    0.05        4,531    4,531       6,648.55   <- the default: one order each
    0.00        4,531   12,442       5,991.58

An order count is still not a count of a strategy's decisions: it is a count of
decisions *at the band the run used*, and at band 0.0 those same 4,531 decisions
produce 12,442 fills, the other 7,911 being drift rebalances nothing asked for.
Turnover here is comparable to a boolean path's trade count only through that
number. Tracking itself is exact -- costlessly the position matches the submitted
target at every decision bar to 3.8e-16.

The band does not touch a strategy's *own* turnover, and should not. The taper
above is a determinism fixture rather than a strategy and decides every 3.3 bars;
the 6,297.43 of fees and 6,297.43 of slippage that costs, against 10,000 of
initial capital, is why it ends below where it started while a costless run of
the same decisions ends at 20,742.99.

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

import numpy as np
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

    ``target`` is what the strategy asked for on every bar;
    ``rebalance_target`` is what the band actually submitted -- the same value on
    a decision bar, ``NaN`` on a bar the band held. ``rebalance_target.notna()``
    is therefore the decision-bar mask, and it is the series to compare a target
    against, because between decisions the book is deliberately left alone.

    ``position_fraction`` is the signed value of the position over equity, which
    equals the target only *at* a decision bar. Between decisions the book holds
    a fixed quantity, so its fraction drifts with the price -- up in a winner,
    down in a loser. That drift is the contract, not an execution error, and
    reading this series as "what the target asked for" inverts what the band is
    for.
    """

    target: pd.Series
    rebalance_target: pd.Series
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
    rebalance_threshold: float = 0.05,
    cost_model: CostModel | None = None,
    funding: pd.Series | None = None,
) -> ExposureBacktestResult:
    """Track ``strategy``'s target exposure over ``df``, charging costs and funding.

    Mirrors ``run_backtest``'s signature where the two paths share a concept, so
    a caller moving between them is not surprised: ``cash`` and ``position_pct``
    mean what they mean there, and ``position_pct`` scales the target, so a
    target of 1.0 at the 0.95 default asks for 95% of *initial cash*.

    ``rebalance_threshold`` is in the target's own units -- 0.05 is five percent
    of the risk budget -- so it is unchanged by ``cash`` and ``position_pct``.
    ``0.0`` submits every bar, which is the mean-reversion overlay the module
    docstring describes rather than a neutral setting; a negative one is
    refused.

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
    if rebalance_threshold < 0:
        raise ValueError(
            f"rebalance_threshold must be >= 0, not {rebalance_threshold}. A negative "
            f"band is satisfied by every bar, which is what 0.0 already means, so it "
            f"would read as 'wider than none' while doing the opposite."
        )

    df = df.sort_index()
    exposure = strategy.compute_target(df)
    _validate_alignment(exposure, df, strategy)
    target = _flat_through_warmup(
        exposure.target, warmup_bars=strategy.warmup_bars, strategy=strategy
    )

    submitted = _banded(target, threshold=rebalance_threshold)

    model = cost_model if cost_model is not None else CostModel()
    pf = vbt.Portfolio.from_orders(
        close=df["close"],
        # A currency value against *initial* cash, never current equity: the
        # repo's non-compounding sizing rule, which ``targetpercent`` silently
        # broke here.
        size=submitted * position_pct * cash,
        size_type="targetvalue",
        init_cash=cash,
        fees=model.fee,
        slippage=model.slippage,
        freq=timeframe_to_pandas_freq(identity.timeframe),
    )

    orders = pf.orders.records_readable
    assets = pf.assets()
    gross_equity = pf.value()
    notional = held_notional(assets, df["open"])
    flow = (
        apply_funding(positions=notional, funding=funding)
        if funding is not None and not funding.empty
        else pd.Series(0.0, index=df.index, dtype="float64")
    )

    return ExposureBacktestResult(
        target=target,
        rebalance_target=submitted,
        position=assets,
        position_fraction=assets * df["close"] / gross_equity,
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
            "rebalance_threshold": rebalance_threshold,
            "cost_model": asdict(model),
            "funding_applied": bool(funding is not None and not funding.empty),
            "data_start": str(df.index.min()),
            "data_end": str(df.index.max()),
            "candle_count": int(len(df)),
        },
    )


def _validate_alignment(
    exposure: TargetExposure, df: pd.DataFrame, strategy: ExposureStrategy
) -> None:
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


def _banded(target: pd.Series, *, threshold: float) -> pd.Series:
    """The target on bars that move it far enough to act on, ``NaN`` elsewhere.

    ``NaN`` is ``from_orders``' "no order", i.e. hold whatever was held, which is
    the reading a target series must never carry (see ``strategies/exposure.py``)
    and exactly the one an order series wants: between decisions the book keeps
    its quantity and its fraction of equity drifts.

    **The comparison is against the last target submitted, never against the
    realized position fraction.** The realized fraction depends on fills, fills
    depend on this band, and a band that read them back would be defined in terms
    of its own output -- not something a vectorized path can precompute, and a
    feedback loop rather than a rule. It also matters that the reference is the
    last *submitted* target rather than the previous bar's: a taper that gives up
    2% of the budget per bar never moves a whole 5% band in one bar, so a
    bar-to-bar reference would hold the opening position through the entire
    taper. Against the last submission the moves accumulate and it decides on
    every third bar -- late by up to one band, which is what a band costs.

    Sequential by construction for that reason -- each decision is measured
    against the previous surviving one, so there is no cumulative form to
    vectorize.
    """
    values = target.to_numpy(dtype="float64")
    submitted = np.full(len(values), np.nan, dtype="float64")
    held = 0.0  # the book starts flat, so the first target is measured against 0
    for position, value in enumerate(values):
        if abs(value - held) >= threshold:
            submitted[position] = value
            held = value
    return pd.Series(submitted, index=target.index, dtype="float64")


def _flat_through_warmup(
    target: pd.Series, *, warmup_bars: int, strategy: ExposureStrategy
) -> pd.Series:
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
