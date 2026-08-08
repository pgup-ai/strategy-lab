"""The book against the backtest it must agree with.

``trades.csv`` is the oracle: a frozen artifact this repo already trusts, and the
reason the book was built before the live feed. A book that reproduces one is
correct in the only sense available here; a live feed can be diffed against
nothing.

The db-marked comparison below is the phase's real check. The synthetic tests
around it pin the individual rules it rests on, which is what says *why* a future
failure happened rather than only that it did.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from strategy_lab.backtests import ExitMode, run_backtest
from strategy_lab.backtests.costs import CostModel as EngineCosts
from strategy_lab.backtests.engine import _exit_signals, _mask_warmup, _warmup_bars
from strategy_lab.backtests.exposure_engine import (
    _banded,
    _flat_through_warmup,
    run_exposure_backtest,
)
from strategy_lab.backtests.funding_frame import with_funding_column
from strategy_lab.backtests.sizing import DEFAULT_VOL_SPAN, SizeMode
from strategy_lab.core.types import Side
from strategy_lab.db import load_candles
from strategy_lab.db.funding import funding_span
from strategy_lab.engine.book import PaperBook
from strategy_lab.engine.exposure_book import ExposureBook
from strategy_lab.engine.fills import CostModel, Direction
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.strategies.exposure_registry import get_exposure_strategy
from strategy_lab.strategies.registry import get_strategy
from tests.conftest import synthetic_ohlcv

FEE, SLIP, CASH, PCT = 0.0005, 0.0005, 10_000.0, 0.95
_PERP = MarketDataIdentity(
    exchange="binance", market_type="perp", symbol="BTC/USDT", timeframe="4h"
)
COSTS = CostModel(fee=FEE, slippage=SLIP)


def _signals_at(book_sides, close, ts, book, scale=1.0):
    return book.on_bar(
        [SimpleNamespace(side=side) for side in book_sides],
        close=close,
        ts_bar_ms=ts,
        scale=scale,
    )


# --------------------------------------------------------------------------
# The rules, on frames a reader can hold in their head.
# --------------------------------------------------------------------------


def test_a_repeated_entry_on_the_same_side_opens_nothing_more():
    """``from_signals`` defaults to ``accumulate=False``: a strategy signalling
    *enter long* on ten bars opens one position, not ten."""
    book = PaperBook(cash=CASH, position_pct=PCT, costs=COSTS)

    first = _signals_at([Side.ENTER_LONG], 100.0, 0, book)
    again = _signals_at([Side.ENTER_LONG], 110.0, 1, book)

    assert len(first) == 1
    assert again == ()
    assert book.position == pytest.approx(CASH * PCT / 100.0)


def test_an_opposite_entry_reverses_without_an_exit_signal_and_both_legs_are_sells():
    """The rule a book gets wrong by guessing. A short entry while long closes the
    long *and* opens the short on one bar, and both legs are sells, so both take
    the sell-side price."""
    book = PaperBook(cash=CASH, position_pct=PCT, costs=COSTS)
    _signals_at([Side.ENTER_LONG], 100.0, 0, book)

    fills = _signals_at([Side.ENTER_SHORT], 120.0, 1, book)

    assert [fill.direction for fill in fills] == [Direction.SELL, Direction.SELL]
    assert {fill.price for fill in fills} == {120.0 * (1 - SLIP)}
    assert book.position < 0
    assert len(book.trades) == 1, "the closed long should be recorded as one trade"


def test_an_entry_is_sized_off_initial_cash_however_the_account_has_moved():
    """Non-compounding, which is the rule this repo has named twice. The second
    entry asks for the same quantity-at-price as the first even after a large
    win, because the request is against *initial* cash."""
    book = PaperBook(cash=CASH, position_pct=PCT, costs=COSTS)
    _signals_at([Side.ENTER_LONG], 100.0, 0, book)
    _signals_at([Side.EXIT_LONG], 200.0, 1, book)  # roughly doubles the account
    assert book.balance > CASH * 1.5

    _signals_at([Side.ENTER_LONG], 100.0, 2, book)

    assert book.position == pytest.approx(CASH * PCT / 100.0)


def test_a_fill_the_account_cannot_pay_for_is_clipped_rather_than_refused():
    """Found by the oracle rather than by reasoning: one trade of 19 on a real BTC
    range differed, at 9,158.96 of a requested 9,500 notional. The engine does not
    refuse an unaffordable entry, it fills ``balance / (price x (1 + fee))``."""
    book = PaperBook(cash=CASH, position_pct=PCT, costs=COSTS)
    _signals_at([Side.ENTER_LONG], 100.0, 0, book)
    _signals_at([Side.EXIT_LONG], 60.0, 1, book)  # a large loss
    affordable = book.balance
    requested = CASH * PCT / 60.0
    assert affordable < CASH * PCT

    _signals_at([Side.ENTER_LONG], 60.0, 2, book)

    price = 60.0 * (1 + SLIP)
    assert book.position == pytest.approx(affordable / (price * (1 + FEE)))
    assert book.position < requested, "the clip has to actually bind here"


def test_a_target_that_does_not_move_issues_no_order():
    """The band lives in the runner, so an unchanged *submitted* target reaching
    the book still has to issue nothing -- applying a band here too would apply
    it twice."""
    book = ExposureBook(cash=CASH, position_pct=PCT, costs=COSTS)

    assert book.on_target(0.5, close=100.0, ts_bar_ms=0) is not None
    assert book.on_target(0.5, close=100.0, ts_bar_ms=1) is None


def test_a_target_crossing_zero_is_one_order_through_the_flat_point():
    book = ExposureBook(cash=CASH, position_pct=PCT, costs=COSTS)
    book.on_target(0.5, close=100.0, ts_bar_ms=0)

    fill = book.on_target(-0.5, close=100.0, ts_bar_ms=1)

    assert fill is not None
    assert fill.direction is Direction.SELL
    assert fill.quantity == pytest.approx(2 * CASH * PCT * 0.5 / 100.0)
    assert book.position < 0


def test_the_exposure_book_sizes_against_initial_cash_not_current_equity():
    """``targetpercent`` would compound here; ``targetvalue`` against initial cash
    is what the engine uses and what keeps the two paths comparable."""
    book = ExposureBook(cash=CASH, position_pct=PCT, costs=COSTS)
    book.on_target(1.0, close=100.0, ts_bar_ms=0)
    held = book.position

    book.on_target(0.0, close=400.0, ts_bar_ms=1)
    book.on_target(1.0, close=100.0, ts_bar_ms=2)

    assert book.position == pytest.approx(held)


# --------------------------------------------------------------------------
# The oracle.
# --------------------------------------------------------------------------


def _funded_frame(tail: int):
    span = funding_span(exchange="binance", market_type="perp", symbol="BTC/USDT")
    if span is None:
        pytest.skip("no stored funding for BTC/USDT perp")
    df = load_candles(
        exchange="binance",
        market_type="perp",
        symbol="BTC/USDT",
        timeframe="4h",
        start=span[0],
        end=span[1],
    )
    df, funding = with_funding_column(_PERP, df, enabled=True)
    return df.iloc[-tail:], funding


@pytest.mark.db
@pytest.mark.parametrize(
    ("name", "mode", "tail"),
    [
        ("donchian", ExitMode.OPPOSITE_SIGNAL_ONLY, 3000),
        ("turnaround_v1", ExitMode.CONTINUATION_FAILURE, 6000),
    ],
)
def test_the_book_reproduces_the_backtests_own_trades(name, mode, tail, tmp_path):
    """The phase's gate. Same entry bar, exit bar, direction, size and both fill
    prices as ``trades.csv``, on every closed trade.

    The engine reports the still-open position as a trade with ``Status: Open``
    and the book holds it as a position, so the comparison is against the closed
    ones -- counting an open position as a trade would compare something the
    engine does not.
    """
    df, funding = _funded_frame(tail)
    strategy = get_strategy(name)
    result = run_backtest(
        df=df,
        strategy=strategy,
        identity=_PERP,
        exit_mode=mode,
        fees=FEE,
        slippage=SLIP,
        cash=CASH,
        position_pct=PCT,
        funding=funding,
        report_root=tmp_path,
    )
    engine = pd.read_csv(
        result.trades_path, parse_dates=["Entry Timestamp", "Exit Timestamp"]
    )
    engine = engine[engine["Status"] == "Closed"].reset_index(drop=True)

    signals = strategy.generate_signals(df)
    warmup = _warmup_bars(strategy, df, size_mode=SizeMode.FIXED, vol_span=DEFAULT_VOL_SPAN)
    long_exits, short_exits = _exit_signals(
        df=df, signals=signals, exit_mode=mode, failure_bars=4
    )
    masked, long_exits, short_exits = _mask_warmup(
        signals, long_exits.fillna(False), short_exits.fillna(False), warmup
    )

    book = PaperBook(cash=CASH, position_pct=PCT, costs=COSTS)
    scale = signals.position_size
    for position, timestamp in enumerate(df.index):
        sides = [
            side
            for side, series in (
                (Side.ENTER_LONG, masked.long_entries),
                (Side.EXIT_LONG, long_exits),
                (Side.ENTER_SHORT, masked.short_entries),
                (Side.EXIT_SHORT, short_exits),
            )
            if bool(series.iloc[position])
        ]
        if sides:
            _signals_at(
                sides,
                float(df["close"].iloc[position]),
                int(timestamp.value // 10**6),
                book,
                scale=1.0 if scale is None else float(scale.iloc[position]),
            )

    assert len(book.trades) == len(engine) > 10
    # A reversal is the rule most likely to be wrong, so the range has to contain
    # one or this check has not exercised it.
    reversals = int(
        (engine["Entry Timestamp"].values[1:] == engine["Exit Timestamp"].values[:-1]).sum()
    )
    assert reversals > 0, "no reversal in the compared range"

    for index, trade in enumerate(book.trades):
        row = engine.iloc[index]
        assert pd.Timestamp(trade.entry_ts_ms, unit="ms", tz="UTC") == row["Entry Timestamp"]
        assert pd.Timestamp(trade.exit_ts_ms, unit="ms", tz="UTC") == row["Exit Timestamp"]
        assert trade.direction == row["Direction"].lower()
        assert trade.quantity == pytest.approx(row["Size"])
        assert trade.entry_price == pytest.approx(row["Avg Entry Price"])
        assert trade.exit_price == pytest.approx(row["Avg Exit Price"])


@pytest.mark.db
def test_the_exposure_book_reproduces_the_exposure_engines_orders():
    """The same gate for the continuous contract, against the orders
    ``from_orders`` actually issued rather than against a target series."""
    df, funding = _funded_frame(2600)
    strategy = get_exposure_strategy("state_machine_v2")
    result = run_exposure_backtest(
        df=df,
        strategy=strategy,
        identity=_PERP,
        cash=CASH,
        position_pct=PCT,
        rebalance_threshold=0.05,
        cost_model=EngineCosts(fee=FEE, slippage=SLIP),
        funding=funding,
    )
    target = _flat_through_warmup(
        strategy.compute_target(df).target, warmup_bars=strategy.warmup_bars, strategy=strategy
    )
    submitted = _banded(target, threshold=0.05)

    book = ExposureBook(cash=CASH, position_pct=PCT, costs=COSTS)
    filled = [
        (timestamp, fill)
        for timestamp, value in submitted.dropna().items()
        if (
            fill := book.on_target(
                float(value),
                close=float(df.loc[timestamp, "close"]),
                ts_bar_ms=int(timestamp.value // 10**6),
            )
        )
        is not None
    ]

    engine = result.orders
    assert len(filled) == len(engine) > 5
    for (timestamp, fill), (_, row) in zip(filled, engine.iterrows(), strict=True):
        assert timestamp == pd.Timestamp(row["Timestamp"])
        assert fill.quantity == pytest.approx(row["Size"])
        assert (fill.direction is Direction.BUY) == (row["Side"] == "Buy")


def test_the_synthetic_probe_that_derived_the_rules_still_holds(tmp_path):
    """The 12-bar run in ``engine.fills``' docstring, re-run.

    Those numbers are the justification for every rule in this module, so a
    vectorbt upgrade that moved them would leave the docstring asserting
    something false -- and this file agreeing with it, since both would move
    together in the db test above.
    """

    @dataclass(frozen=True)
    class _Probe:
        name: str = "probe"
        version: str = "1.0.0"
        warmup_bars: int = 0

        def generate_signals(self, frame):
            from strategy_lab.strategies.base import SignalSet

            def at(positions):
                series = pd.Series(False, index=frame.index)
                series.iloc[positions] = True
                return series

            return SignalSet(at([1, 2, 6]), at([4]), at([8]), at([10]))

    close = pd.Series(np.arange(100.0, 112.0))
    index = pd.date_range("2024-01-01", periods=12, freq="4h", tz="UTC")
    df = pd.DataFrame(
        {"open": close.values, "high": close.values + 1, "low": close.values - 1,
         "close": close.values, "volume": 1.0},
        index=index,
    )
    result = run_backtest(
        df=df,
        strategy=_Probe(),
        identity=MarketDataIdentity(
            exchange="x", market_type="spot", symbol="P/Q", timeframe="4h"
        ),
        exit_mode=ExitMode.OPPOSITE_SIGNAL_ONLY,
        fees=0.001,
        slippage=0.002,
        cash=10_000.0,
        position_pct=0.95,
        report_root=tmp_path,
    )
    trades = pd.read_csv(result.trades_path)

    assert trades["Size"].iloc[0] == pytest.approx(10_000 * 0.95 / 101)
    assert trades["Avg Entry Price"].iloc[0] == pytest.approx(101 * 1.002)
    assert trades["Avg Exit Price"].iloc[0] == pytest.approx(104 * 0.998)
    assert trades["Entry Fees"].iloc[0] == pytest.approx(
        0.001 * trades["Size"].iloc[0] * trades["Avg Entry Price"].iloc[0]
    )
    # The reversal: the long's exit and the short's entry share a bar and a price.
    assert trades["Exit Timestamp"].iloc[1] == trades["Entry Timestamp"].iloc[2]
    assert trades["Avg Exit Price"].iloc[1] == pytest.approx(trades["Avg Entry Price"].iloc[2])


def test_a_book_over_a_flat_frame_trades_nothing():
    """The non-vacuity guard for everything above: a book that emitted fills with
    no signals would make every comparison pass by accident."""
    book = PaperBook(cash=CASH, position_pct=PCT, costs=COSTS)
    frame = synthetic_ohlcv(n=50)

    for position, timestamp in enumerate(frame.index):
        book.on_bar([], close=float(frame["close"].iloc[position]),
                    ts_bar_ms=int(timestamp.value // 10**6))

    assert book.fills == []
    assert book.position == 0.0
