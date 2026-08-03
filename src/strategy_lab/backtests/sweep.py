from __future__ import annotations

import dataclasses
import itertools
import math
import statistics
from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy_lab.strategies.base import SignalSet, Strategy
from strategy_lab.strategies.registry import get_strategy
from strategy_lab.timeframes import timeframe_to_millis

_MILLIS_PER_YEAR = 365 * 24 * 3600 * 1000


@dataclass(frozen=True)
class SweepPoint:
    params: dict
    total_return: float
    sharpe: float
    max_drawdown: float
    trades: int


def bars_per_year(timeframe: str) -> float:
    """Annualization factor for a timeframe, so Sharpe is comparable across them.

    Hardcoding a bar count would silently report a 15m number for 4h perps and
    daily ETFs alike -- the same returns annualizing 4x and 22x wrong.
    """
    return _MILLIS_PER_YEAR / timeframe_to_millis(timeframe)


def sweep_parameters(
    *,
    df: pd.DataFrame,
    strategy_name: str,
    grid: dict[str, list],
    timeframe: str,
) -> list[SweepPoint]:
    """Evaluate a strategy across a parameter grid on one frame.

    Deliberately vectorbt-free: a sweep runs hundreds of configurations, and the
    point is the *shape* of the surface, not the exact PnL of any one cell.
    Costs and funding live in the R2 layer, which the single-run backtest uses.

    Every cell is evaluated over the same bar range -- the template's declared
    ``warmup_bars``, not a per-cell one -- because a surface whose cells cover
    different samples compares nothing.
    """
    template = get_strategy(strategy_name)
    valid = {f.name for f in dataclasses.fields(template)}
    unknown = set(grid).difference(valid)
    if unknown:
        raise ValueError(
            f"{strategy_name} does not accept parameter(s) {sorted(unknown)}; "
            f"available: {sorted(valid)}"
        )

    warmup = template.warmup_bars
    if warmup >= len(df):
        raise ValueError(
            f"{strategy_name} declares {warmup} warmup bars but the frame has "
            f"{len(df)}; every cell would score 0.0 and the surface would read "
            f"as a flat plateau"
        )

    names = list(grid)
    points: list[SweepPoint] = []
    for combination in itertools.product(*(grid[name] for name in names)):
        params = dict(zip(names, combination))
        strategy = dataclasses.replace(template, **params)
        points.append(_evaluate(df, strategy, params, warmup, timeframe))
    return points


def _evaluate(
    df: pd.DataFrame,
    strategy: Strategy,
    params: dict,
    warmup: int,
    timeframe: str,
) -> SweepPoint:
    signals = strategy.generate_signals(df)
    position = positions_from_signals(signals)

    returns = (df["close"].pct_change() * position).iloc[warmup:].fillna(0.0)
    equity = (1 + returns).cumprod()

    drawdown = (equity / equity.cummax() - 1).min()
    volatility = returns.std()
    sharpe = 0.0
    if math.isfinite(volatility) and volatility > 0:
        sharpe = float(returns.mean() / volatility * bars_per_year(timeframe) ** 0.5)

    return SweepPoint(
        params=params,
        total_return=float(equity.iloc[-1] - 1) if len(equity) else 0.0,
        sharpe=sharpe,
        max_drawdown=float(drawdown) if len(equity) else 0.0,
        # Bars on which the position changed. A flip counts once, not twice --
        # this is a turnover comparison between cells, not a fill count.
        trades=int(position.diff().abs().gt(0).sum()),
    )


def positions_from_signals(signals: SignalSet) -> pd.Series:
    """Net position in {-1, 0, +1}, one bar behind the signal that produced it.

    Two independent books summed, exactly as the engine's ``from_signals`` call
    treats ``entries``/``exits`` against ``short_entries``/``short_exits``.
    Deriving the position from entries alone makes every exit ingredient
    invisible: measured on 83,348 BTC 15m bars, all four donchian ``exit_span``
    values then scored bit-identically -- a whole grid axis reporting "this
    parameter does not matter" when what did not matter was the sweep.

    Shifted one bar because a signal computed from bar *t*'s close can only be
    traded from bar *t + 1*. Without the shift the sweep reports lookahead
    returns and every number on the surface is fiction.
    """
    position = _book(signals.long_entries, signals.long_exits, 1.0) + _book(
        signals.short_entries, signals.short_exits, -1.0
    )
    return position.shift(1).fillna(0.0)


def _book(entries: pd.Series, exits: pd.Series, held: float) -> pd.Series:
    """One side's book: open on an entry, flatten on an exit, else hold.

    Kept per side because an exit only speaks for its own side -- a long exit
    must not flatten an open short. Entries are written after exits so an entry
    wins a same-bar conflict: an entry states a direction, an exit only says the
    previous one has expired. No registered strategy can produce that conflict
    today, so both properties are pinned by unit test rather than by any live
    surface.
    """
    book = pd.Series(np.nan, index=entries.index, dtype="float64")
    book[exits] = 0.0
    book[entries] = held
    return book.ffill().fillna(0.0)


def stability_score(points: list[SweepPoint]) -> float:
    """How much of the surface works, penalized by how much it varies.

    A broad plateau of modest results scores above a single spectacular cell
    surrounded by losses -- that spike is an overfit, and reporting its number as
    "the" result is the most common way a trend backtest lies.

    Spread alone is not enough to say that. A surface of two strong cells and
    six dead ones has both a higher mean and a small enough spread to beat a
    plateau on ``mean / (1 + spread)``; multiplying by the fraction of cells
    that are positive at all is what puts the plateau back on top.
    """
    if len(points) < 2:
        return 0.0
    sharpes = [p.sharpe for p in points]
    positive_fraction = sum(1 for s in sharpes if s > 0) / len(sharpes)
    spread = statistics.pstdev(sharpes)
    return float(statistics.fmean(sharpes) * positive_fraction / (1 + spread))
