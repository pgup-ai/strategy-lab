from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_lab.backtests import sweep as sweep_module
from strategy_lab.backtests.sweep import (
    SweepPoint,
    positions_from_signals,
    stability_score,
    sweep_parameters,
)
from strategy_lab.strategies.base import SignalSet
from strategy_lab.strategies.registry import get_strategy
from tests.conftest import synthetic_ohlcv


def test_sweep_returns_one_point_per_parameter_combination():
    df = synthetic_ohlcv(n=900)
    points = sweep_parameters(
        df=df,
        strategy_name="donchian",
        grid={"entry_span": [24, 48], "exit_span": [12, 24]},
        timeframe="15m",
    )
    assert len(points) == 4
    assert {tuple(sorted(p.params.items())) for p in points} == {
        (("entry_span", 24), ("exit_span", 12)),
        (("entry_span", 24), ("exit_span", 24)),
        (("entry_span", 48), ("exit_span", 12)),
        (("entry_span", 48), ("exit_span", 24)),
    }


def test_sweep_refuses_a_frame_that_is_all_warmup():
    """Otherwise every cell scores 0.0 and the surface reads as a flat plateau.

    ``ema_cross`` declares 3840 warmup bars, so a 900-bar frame leaves nothing
    to evaluate. Returning zeros there would be the sweep lying in exactly the
    direction ``stability_score`` exists to prevent.
    """
    with pytest.raises(ValueError, match="warmup"):
        sweep_parameters(
            df=synthetic_ohlcv(n=900),
            strategy_name="ema_cross",
            grid={"fast_span": [24, 48]},
            timeframe="15m",
        )


def test_the_sweep_warms_up_every_cell_for_the_deepest_grid_entry(monkeypatch):
    """A cell's own span can exceed the template's warmup, and usually does.

    Every cell must still cover the same bars or the surface compares nothing,
    so there is one warmup for the whole grid -- but taking it from the template
    picks a value that is too *small* for the larger cells, which is the unsafe
    direction. The deepest cell's warmup is the only choice that is both common
    to every cell and sufficient for each.
    """
    df = synthetic_ohlcv(n=900)
    deepest = 600
    assert get_strategy("tsmom").warmup_bars < deepest < len(df), (
        "the deep cell must out-warm the template and still fit the frame"
    )

    alone = sweep_parameters(
        df=df, strategy_name="tsmom", grid={"lookback": [12]}, timeframe="15m"
    )[0]

    warmups: list[int] = []
    evaluate = sweep_module._evaluate

    def record(frame, strategy, params, warmup, timeframe):
        warmups.append(warmup)
        return evaluate(frame, strategy, params, warmup, timeframe)

    monkeypatch.setattr(sweep_module, "_evaluate", record)
    points = sweep_parameters(
        df=df, strategy_name="tsmom", grid={"lookback": [12, deepest]}, timeframe="15m"
    )

    assert warmups == [deepest, deepest], (
        f"cells were evaluated at warmups {warmups}; every cell must use the "
        f"deepest cell's {deepest}"
    )
    shallow = next(p for p in points if p.params["lookback"] == 12)
    assert shallow.total_return != alone.total_return, (
        "the shallow cell scored identically with and without a deeper cell "
        "beside it, so the deepened warmup never reached the returns"
    )


def test_sweep_names_the_cell_whose_warmup_exceeds_the_frame():
    """The template fits the frame here; only one grid entry blows the budget.

    Reporting the strategy alone would leave the user re-deriving which of a
    hundred cells was too deep.
    """
    with pytest.raises(ValueError, match="warmup") as raised:
        sweep_parameters(
            df=synthetic_ohlcv(n=900),
            strategy_name="donchian",
            grid={"entry_span": [24, 4000]},
            timeframe="15m",
        )
    assert "4000" in str(raised.value), (
        f"error does not name the offending cell: {raised.value}"
    )


def _alternating_frame(n: int = 400) -> pd.DataFrame:
    """Strictly alternating up/down closes, so bar *t* never predicts bar *t+1*.

    A one-bar-lookback momentum rule is right about the *current* bar by
    construction and wrong about the *next* one by construction, which is what
    makes this frame separate a sweep from a lookahead machine.
    """
    index = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC", name="timestamp")
    close = 100.0 * np.where(np.arange(n) % 2 == 0, 1.0, 1.02)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(n, 500.0),
        },
        index=index,
    )


def test_the_sweep_trades_the_bar_after_the_signal_not_the_signal_bar():
    """A signal from bar *t*'s close can only be traded from bar *t+1*.

    On an alternating series a one-bar momentum rule buys every up bar and sells
    every down bar -- so trading it on the signal bar itself captures every move
    and returns a spectacular profit, while trading it one bar later is wrong on
    every single bar and must lose. Dropping the ``.shift(1)`` flips the sign of
    this number, which is the whole difference between a sweep and fiction.
    """
    point = sweep_parameters(
        df=_alternating_frame(),
        strategy_name="tsmom",
        grid={"lookback": [1]},
        timeframe="15m",
    )[0]

    assert point.total_return < 0, (
        f"a rule that is wrong on every bar returned {point.total_return:+.2%}; "
        f"the sweep is trading on the signal bar rather than the one after it"
    )


def test_the_sweep_honours_the_exit_signals_not_only_the_entries():
    """A position model built from entries alone makes exit parameters invisible.

    ``donchian`` is the case that exposes it: its exits are a separate reverse
    channel rather than the inverse of the entry state, so deriving the position
    from entries alone scored all four ``exit_span`` values bit-identically on
    83,348 real BTC 15m bars -- a whole grid axis reporting "this parameter does
    not matter" when what did not matter was the sweep.
    """
    df = synthetic_ohlcv(n=900)
    points = sweep_parameters(
        df=df,
        strategy_name="donchian",
        grid={"entry_span": [96], "exit_span": [12, 96]},
        timeframe="15m",
    )

    fast_exit, slow_exit = points
    assert fast_exit.params["exit_span"] == 12 and slow_exit.params["exit_span"] == 96
    assert fast_exit.total_return != slow_exit.total_return, (
        "both exit channels produced identical equity; the sweep is ignoring long_exits"
    )
    assert fast_exit.trades != slow_exit.trades, (
        "a 12-bar exit channel must turn over more often than a 96-bar one"
    )


def _signals(*, long_entries, long_exits, short_entries, short_exits) -> SignalSet:
    """A hand-built SignalSet, for cases no registered strategy can produce."""
    index = pd.date_range(
        "2024-01-01", periods=len(long_entries), freq="15min", tz="UTC", name="timestamp"
    )
    return SignalSet(
        *(
            pd.Series([bool(f) for f in flags], index=index, dtype=bool)
            for flags in (long_entries, long_exits, short_entries, short_exits)
        )
    )


def test_an_exit_only_flattens_its_own_side():
    """A long exit must not close an open short, or the two books contaminate.

    Unreachable through any registered strategy -- ``tsmom`` and friends fire
    ``long_exits`` on exactly the bars they enter short, so an entry immediately
    overwrites the erroneous flatten, and ``donchian`` re-asserts its short on
    almost every such bar. Left to a live surface this would be a near-equivalent
    mutant that no test could kill, so it is pinned here instead.
    """
    position = positions_from_signals(
        _signals(
            long_entries=[0, 0, 0, 0],
            long_exits=[0, 1, 0, 0],  # fires while a short is open, with no entry
            short_entries=[1, 0, 0, 0],
            short_exits=[0, 0, 0, 1],
        )
    )
    assert list(position) == [0.0, -1.0, -1.0, -1.0], (
        "a long exit closed an open short; the books are not independent"
    )


def test_an_entry_wins_a_same_bar_conflict_with_its_own_exit():
    """An entry states a direction; an exit only says the previous one expired."""
    position = positions_from_signals(
        _signals(
            long_entries=[1, 0, 0],
            long_exits=[1, 0, 1],
            short_entries=[0, 0, 0],
            short_exits=[0, 0, 0],
        )
    )
    assert list(position) == [0.0, 1.0, 1.0], (
        "the same-bar exit won over the entry, so the long never opened"
    )


def test_a_position_is_never_leveraged_by_the_two_books():
    """Summing the books must stay in {-1, 0, +1} for every registered baseline."""
    df = synthetic_ohlcv(n=900)
    for name in ("tsmom", "ema_cross", "donchian", "multi_horizon"):
        position = positions_from_signals(get_strategy(name).generate_signals(df))
        assert set(position.unique()).issubset({-1.0, 0.0, 1.0}), (
            f"{name} produced positions {sorted(set(position.unique()))}"
        )


def test_sharpe_annualizes_from_the_timeframe_not_a_hardcoded_bar_count():
    """A 4h bar is 16 of 15m bars, so the same returns annualize 4x differently.

    The R1 perp data is 4h and the ETF data is weekly; a Sharpe hardcoded to
    ``(365 * 24 * 4) ** 0.5`` reports a 15m number for both.
    """
    df = synthetic_ohlcv(n=900)
    grid = {"lookback": [24]}

    fast = sweep_parameters(df=df, strategy_name="tsmom", grid=grid, timeframe="15m")[0]
    slow = sweep_parameters(df=df, strategy_name="tsmom", grid=grid, timeframe="4h")[0]

    assert abs(fast.sharpe) > 0.1, "a Sharpe of ~0 would make the ratio below vacuous"
    assert fast.total_return == pytest.approx(slow.total_return), (
        "the timeframe must only change annualization, never the trades"
    )
    assert fast.sharpe == pytest.approx(4.0 * slow.sharpe), (
        f"15m Sharpe {fast.sharpe:.4f} vs 4h {slow.sharpe:.4f}; expected exactly "
        f"sqrt(16) = 4x, so the annualization factor is not derived from the timeframe"
    )


def test_stability_score_rewards_a_broad_plateau_over_a_lone_spike():
    """The gate for R0: neighbouring parameters must behave similarly.

    The spike wins on ``max(sharpes)`` and on a plain mean, both asserted below,
    so any summary reducing to either would rank the overfit first.
    """
    plateau = [
        SweepPoint({"n": n}, total_return=0.2, sharpe=1.0, max_drawdown=-0.1, trades=10)
        for n in range(5)
    ]
    spike = [
        SweepPoint(
            {"n": n},
            total_return=2.0,
            sharpe=(9.0 if n == 2 else 0.1),
            max_drawdown=-0.1,
            trades=10,
        )
        for n in range(5)
    ]

    assert max(p.sharpe for p in spike) > max(p.sharpe for p in plateau)
    assert (
        sum(p.sharpe for p in spike) / len(spike) > sum(p.sharpe for p in plateau) / len(plateau)
    )
    assert stability_score(plateau) > stability_score(spike)


def test_stability_score_counts_how_much_of_the_surface_works():
    """Two good cells in a dead field must lose to a wholly-positive plateau.

    This isolates the ``positive_fraction`` term, which the spike tests above do
    not: they are already decided by the spread penalty alone. Here the narrow
    surface has *twice* the mean Sharpe and a spread small enough that
    ``mean / (1 + spread)`` ranks it first (0.1745 vs 0.1200) -- only counting
    how many cells are positive at all puts the plateau back on top (0.1200 vs
    0.0436). Sharpe exactly 0.0 is what a cell that never trades really returns,
    so the dead field is the shape of a channel too wide to ever break.
    """
    plateau = [
        SweepPoint({"n": n}, total_return=0.05, sharpe=0.12, max_drawdown=-0.1, trades=10)
        for n in range(8)
    ]
    two_good_cells = [
        SweepPoint(
            {"n": n},
            total_return=(0.4 if n < 2 else 0.0),
            sharpe=(1.0 if n < 2 else 0.0),
            max_drawdown=-0.1,
            trades=(10 if n < 2 else 0),
        )
        for n in range(8)
    ]

    narrow_sharpes = [p.sharpe for p in two_good_cells]
    plateau_sharpes = [p.sharpe for p in plateau]
    assert np.mean(narrow_sharpes) > np.mean(plateau_sharpes)
    assert np.std(narrow_sharpes) > np.std(plateau_sharpes)

    assert stability_score(plateau) > stability_score(two_good_cells)
