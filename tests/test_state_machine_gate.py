"""The R5 gate: does ``state_machine_v1`` beat the R0 baseline out of sample?

This file is the executable form of the phase's own protocol, and the protocol
is the point rather than the numbers:

- **Parameters were chosen on the first 60% of bars and nowhere else.** 54
  configurations were evaluated there; the cell pinned below is the one with the
  highest net-of-funding Sharpe on that half. The R0 baseline gets the same
  treatment -- ``donchian`` 40/10 is the best of the R0 gate's own 16-cell
  surface *on the same training half*, so both sides are tuned by the same rule
  on the same bars.
- **The last 40% was evaluated once.** Nothing below was adjusted after seeing
  it. The 2x/3x cost rows and the 16-cell baseline surface were measured in the
  same pass, not in a second one after a disappointing first.
- **A run trades exactly the bars its half owns.** Each strategy's frame starts
  at ``split - warmup_bars`` so the engine's own mask lands on the boundary:
  the machine warms 2,160 bars and donchian 40, and comparing runs over
  different tradeable bars is a defect the charter has already corrected once.

The window is pinned with an explicit ``end`` rather than reading to the end of
the table. ``market_candles`` accumulates, and a gate whose split moves every
time someone fetches a candle is not a record of anything.

The assertions are the verdict, not the digits: the headline figures carry loose
tolerances because they must survive a vectorbt or pandas point release, while
the sample shape -- which bars, how many, how many trades -- is exact, because a
silent change there means the two sides stopped being comparable.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from strategy_lab.backtests.costs import funding_coverage_gaps, window_end
from strategy_lab.backtests.engine import ExitMode, run_backtest
from strategy_lab.features.flow import FUNDING_COLUMN, align_funding_to_bars
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.state.machine import StateMachine
from strategy_lab.strategies.donchian import Donchian
from strategy_lab.strategies.registry import get_strategy

pytestmark = pytest.mark.db

IDENTITY = MarketDataIdentity(
    exchange="binance", market_type="perp", symbol="BTC/USDT", timeframe="4h"
)
# Binance settled no funding for the contract's first 40 hours, so the window
# opens at the first stored settlement rather than the first candle.
WINDOW_START = "2019-09-10 08:00:00"
WINDOW_END = "2026-08-03 20:00:00"
EXPECTED_BARS = 15_118
TRAIN_FRACTION = 0.6
SPLIT_AT = pd.Timestamp("2023-10-31 00:00:00", tz="UTC")
TEST_BARS = 6_048

# Chosen on the first 60% of bars, out of the 54 declared in the R5 progress
# log. Not a default: the default machine was one of the 54 and is weaker
# in-sample (Sharpe +0.131 against +0.769) and out of sample (+0.738 against
# +0.938).
TRAINED_MACHINE = StateMachine(
    enter_strength=0.80, exit_strength=1.0 / 3.0, min_dwell=2, cooldown=16
)
# The best cell of the R0 gate's 16-cell donchian surface on that same first
# 60%, at Sharpe +1.433 -- so the baseline is not being handed a configuration
# the machine's own selection rule would have rejected.
BASELINE = Donchian(entry_span=40, exit_span=10)


@pytest.fixture(scope="module")
def frame() -> tuple[pd.DataFrame, pd.Series]:
    from strategy_lab.db.candles import load_candles
    from strategy_lab.db.funding import load_funding

    df = load_candles(
        exchange=IDENTITY.exchange,
        market_type=IDENTITY.market_type,
        symbol=IDENTITY.symbol,
        timeframe=IDENTITY.timeframe,
        start=WINDOW_START,
        end=WINDOW_END,
    )
    if df.empty:
        pytest.skip("no stored BTC/USDT perp 4h candles; run fetch-perp first")
    rates = load_funding(
        exchange=IDENTITY.exchange,
        market_type=IDENTITY.market_type,
        symbol=IDENTITY.symbol,
    )
    if rates.empty:
        pytest.skip("no stored BTC/USDT funding; run fetch-funding first")
    rates = rates["funding_rate"]
    return df.assign(**{FUNDING_COLUMN: align_funding_to_bars(df.index, rates)}), rates


def out_of_sample(strategy, frame, tmp_path) -> dict:
    """One engine run over the test half, trading from ``SPLIT_AT`` exactly."""
    df, rates = frame
    split = int(TRAIN_FRACTION * len(df))
    window = df.iloc[split - strategy.warmup_bars :]
    funding = rates[
        (rates.index >= window.index[0]) & (rates.index < window_end(window.index))
    ]
    assert not funding_coverage_gaps(funding=funding, index=window.index)

    result = run_backtest(
        df=window,
        strategy=strategy,
        identity=IDENTITY,
        exit_mode=ExitMode.OPPOSITE_SIGNAL_ONLY,
        funding=funding,
        cost_stress=(1.0, 2.0, 3.0),
        report_root=tmp_path / strategy.name,
    )
    stats = json.loads(result.stats_path.read_text())
    config = json.loads((result.report_dir / "config.json").read_text())
    stress = json.loads(result.costs_path.read_text())["stress"]
    return {
        "first_tradeable": window.index[config["warmup_bars"]],
        "tradeable_bars": len(window) - config["warmup_bars"],
        "sharpe": stats["Sharpe Ratio (net of funding)"],
        "net_return_pct": stats["Net Return [%]"],
        "max_drawdown_pct": stats["Max Drawdown [%] (net of funding)"],
        "trades": int(stats["Total Trades"]),
        "net_by_stress": {row["multiple"]: row["net_return_pct"] for row in stress},
    }


@pytest.fixture(scope="module")
def machine_run(frame, tmp_path_factory) -> dict:
    strategy = replace(get_strategy("state_machine_v1"), machine=TRAINED_MACHINE)
    return out_of_sample(strategy, frame, tmp_path_factory.mktemp("machine"))


@pytest.fixture(scope="module")
def baseline_run(frame, tmp_path_factory) -> dict:
    return out_of_sample(BASELINE, frame, tmp_path_factory.mktemp("baseline"))


def test_the_split_is_the_one_the_parameters_were_chosen_on(frame):
    """A moved split silently turns the out-of-sample claim into something else."""
    df, _ = frame
    assert len(df) == EXPECTED_BARS
    split = int(TRAIN_FRACTION * len(df))
    assert df.index[split] == SPLIT_AT
    assert len(df) - split == TEST_BARS


def test_both_sides_trade_the_same_out_of_sample_bars(machine_run, baseline_run):
    """The comparison is only a comparison if the two runs cover one window.

    The machine warms 2,160 bars and donchian 40, so a shared frame start would
    hand donchian 2,120 bars the machine cannot see.
    """
    for run in (machine_run, baseline_run):
        assert run["first_tradeable"] == SPLIT_AT
        assert run["tradeable_bars"] == TEST_BARS


def test_neither_side_is_inert(machine_run, baseline_run):
    """Non-vacuity: a strategy that never trades cannot lose, and cannot win."""
    assert machine_run["trades"] > 50
    assert baseline_run["trades"] > 50


def test_the_machine_beats_the_r0_baseline_out_of_sample(machine_run, baseline_run):
    """**The gate.** Risk-adjusted return, drawdown and total return, all three.

    Measured 2026-08-04: Sharpe +0.938 against +0.072, max drawdown 8.24%
    against 43.86%, net +23.29% against -6.64%. The margin is wide enough that
    the tolerances below are about surviving a dependency bump, not about the
    verdict being close.
    """
    assert machine_run["sharpe"] > baseline_run["sharpe"]
    assert machine_run["max_drawdown_pct"] < baseline_run["max_drawdown_pct"]
    assert machine_run["net_return_pct"] > baseline_run["net_return_pct"]

    assert machine_run["sharpe"] == pytest.approx(0.938, abs=0.05)
    assert machine_run["max_drawdown_pct"] == pytest.approx(8.24, abs=0.5)
    assert machine_run["net_return_pct"] == pytest.approx(23.29, abs=1.5)
    assert baseline_run["sharpe"] == pytest.approx(0.072, abs=0.05)
    assert baseline_run["max_drawdown_pct"] == pytest.approx(43.86, abs=0.5)
    assert baseline_run["net_return_pct"] == pytest.approx(-6.64, abs=1.5)


def test_the_win_survives_the_cost_stress_that_r2_gates_on(machine_run, baseline_run):
    """R2's standard, and the machine's median hold is 3 bars against 23.

    Trading that much more often per unit of exposure is exactly the profile a
    fee assumption can flatter, so the win has to hold at 2x and 3x as well as
    at 1x. It does -- but the machine's own edge does not survive 3x, where it
    lands at -0.61%. That is a real limit and the test states it rather than
    stopping at "beats the baseline everywhere".
    """
    for multiple in (1.0, 2.0, 3.0):
        assert machine_run["net_by_stress"][multiple] > baseline_run["net_by_stress"][multiple]
    assert machine_run["net_by_stress"][2.0] > 0.0
    assert machine_run["net_by_stress"][3.0] < 1.0
