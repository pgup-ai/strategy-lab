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

    Every cell is evaluated over the same bar range -- the *deepest* cell's
    ``warmup_bars``, not the template's -- because a surface whose cells cover
    different samples compares nothing, and a cell's spans are swept, so the
    template's warmup can be far too short for the larger cells. The deepest is
    the only value that is both common to every cell and sufficient for each.
    """
    template = get_strategy(strategy_name)
    valid = {f.name for f in dataclasses.fields(template)}
    unknown = set(grid).difference(valid)
    if unknown:
        raise ValueError(
            f"{strategy_name} does not accept parameter(s) {sorted(unknown)}; "
            f"available: {sorted(valid)}"
        )

    names = list(grid)
    empty = [name for name in names if not grid[name]]
    if empty:
        raise ValueError(
            f"{strategy_name} grid parameter(s) {sorted(empty)} have no values; "
            f"their product is empty, so there is no surface to score"
        )

    cells = [
        (params, dataclasses.replace(template, **params))
        for params in (
            dict(zip(names, combination, strict=True))
            for combination in itertools.product(*(grid[name] for name in names))
        )
    ]

    deepest_params, deepest = max(cells, key=lambda cell: cell[1].warmup_bars)
    warmup = deepest.warmup_bars
    if warmup >= len(df):
        raise ValueError(
            f"{strategy_name} cell {deepest_params} needs {warmup} warmup bars but "
            f"the frame has {len(df)}; every cell would score 0.0 and the surface "
            f"would read as a flat plateau"
        )

    return [_evaluate(df, strategy, params, warmup, timeframe) for params, strategy in cells]


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

    # Clamped to 1.0 because the curve starts at par, before the first bar's
    # return: an unclamped cummax takes the already-reduced first value as the
    # peak, so a cell that loses 10% on bar one and then goes flat reports no
    # drawdown at all.
    drawdown = (equity / equity.cummax().clip(lower=1.0) - 1).min()
    volatility = returns.std()
    sharpe = 0.0
    if math.isfinite(volatility) and volatility > 0:
        sharpe = float(returns.mean() / volatility * bars_per_year(timeframe) ** 0.5)

    return SweepPoint(
        params=params,
        total_return=float(equity.iloc[-1] - 1) if len(equity) else 0.0,
        sharpe=sharpe,
        max_drawdown=float(drawdown) if len(equity) else 0.0,
        # Bars on which the position changed, over the same window ``returns``
        # covers -- a reader multiplies this by a cost assumption to sanity-check
        # a cell, so counting transitions whose returns were excluded overstates
        # the cost of the returns actually reported. A flip counts once, not
        # twice: this is a turnover comparison between cells, not a fill count.
        trades=int(position.diff().abs().gt(0).iloc[warmup:].sum()),
    )


def positions_from_signals(signals: SignalSet) -> pd.Series:
    """Net position in {-1, 0, +1}, one bar behind the signal that produced it.

    **One net position, resolved the way the engine resolves it** -- not two
    independent books summed. The books were the earlier model and they disagree
    with ``vbt.Portfolio.from_signals`` on exactly one event, reversal: when an
    opposite entry fires before the same-side exit does, the engine reverses
    (``upon_opposite_entry`` defaults to ``ReverseReduce``, and with
    ``accumulate=False`` that lands in ``signals_to_size_nb``'s "reverse the
    position" branch) while two books cancel to flat -- and then stay wrong,
    because the stale side keeps its holding until its own exit finally arrives.
    ``donchian`` with ``exit_span > entry_span`` produces that event on live
    data: the exit channel is the wider one, so ``close < entry_low`` fires
    before ``close < exit_low``. Three cells of the published R0 grid are in
    that configuration.

    The resolution order below is vectorbt's own, read off
    ``portfolio/nb.py`` under this repo's settings (``accumulate=False``, all
    three conflict modes ``ignore``, ``upon_opposite_entry=ReverseReduce``):

    1. An entry and an exit on the *same* side cancel each other; so do a long
       entry and a short entry. ``ConflictMode.Ignore`` drops **both** signals,
       so neither "entry wins" nor "exit wins" -- the bar does nothing.
    2. From a long, a short entry outranks a long exit (reverse, not close);
       from a short, a long entry outranks a short exit. A same-side entry while
       already in that direction is a no-op, because ``accumulate=False``.

    vectorbt guards step 1 with ``if is_long_entry or is_short_entry``. That
    guard is not reproduced because it is provably inert -- every branch inside
    it already requires an entry -- and a mutation test confirmed it: adding it
    kills no mutant, so it would be untestable code.

    Deriving the position from entries alone would instead make every exit
    ingredient invisible: measured on 83,348 BTC 15m bars, all four donchian
    ``exit_span`` values then scored bit-identically -- a whole grid axis
    reporting "this parameter does not matter" when what did not matter was the
    sweep.

    Shifted one bar because a signal computed from bar *t*'s close can only be
    traded from bar *t + 1*. Without the shift the sweep reports lookahead
    returns and every number on the surface is fiction.
    """
    held = _net_position(
        signals.long_entries.to_numpy(dtype=bool),
        signals.long_exits.to_numpy(dtype=bool),
        signals.short_entries.to_numpy(dtype=bool),
        signals.short_exits.to_numpy(dtype=bool),
    )
    position = pd.Series(held, index=signals.long_entries.index, dtype="float64")
    return position.shift(1).fillna(0.0)


def _net_position(
    long_entries: np.ndarray,
    long_exits: np.ndarray,
    short_entries: np.ndarray,
    short_exits: np.ndarray,
) -> np.ndarray:
    """The state machine itself: one position, updated bar by bar.

    Sequential rather than vectorized because the outcome of a bar depends on
    the position carried into it -- a short entry reverses a long but opens a
    short from flat, and those are different states, not different signals.
    """
    held = np.zeros(len(long_entries), dtype="float64")
    position = 0.0
    for i, (long_entry, long_exit, short_entry, short_exit) in enumerate(
        zip(
            long_entries.tolist(),
            long_exits.tolist(),
            short_entries.tolist(),
            short_exits.tolist(),
            strict=True,
        )
    ):
        if long_entry and long_exit:
            long_entry = long_exit = False
        if short_entry and short_exit:
            short_entry = short_exit = False
        if long_entry and short_entry:
            long_entry = short_entry = False

        if position > 0:
            if short_entry:
                position = -1.0
            elif long_exit:
                position = 0.0
        elif position < 0:
            if long_entry:
                position = 1.0
            elif short_exit:
                position = 0.0
        else:
            if long_entry:
                position = 1.0
            elif short_entry:
                position = -1.0

        held[i] = position
    return held


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
