from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_lab.features.flow import FUNDING_COLUMN, align_funding_to_bars
from strategy_lab.features.registry import get_feature
from strategy_lab.state.machine import MarketState
from strategy_lab.state.policy import STATE_TARGET_RISK
from strategy_lab.strategies.registry import get_strategy, list_strategies
from strategy_lab.strategies.state_machine_v1 import RANKED_FEATURES
from tests.conftest import synthetic_ohlcv, synthetic_ohlcv_with_funding

NAME = "state_machine_v1"

# Enough bars past warmup for the machine to walk the lifecycle several times.
PROBE_SPAN = 800


def probe_frame(strategy, *, funding: bool = False, seed: int = 7) -> pd.DataFrame:
    build = synthetic_ohlcv_with_funding if funding else synthetic_ohlcv
    return build(n=strategy.warmup_bars + PROBE_SPAN, seed=seed)


def test_it_is_registered_and_covered_by_the_safety_suites():
    """Registration is what enrols it in test_lookahead and test_replay_determinism."""
    assert NAME in list_strategies()
    assert get_strategy(NAME).name == NAME


def test_warmup_is_the_deepest_feature_it_reads_plus_the_rank_on_top():
    """Not a new number, and not the largest declared lookback either.

    ``direction`` alone declares 1920, and a trailing rank cannot start until
    the feature underneath it has values to rank -- so the two costs add, the
    way they already do inside ``features.volatility.Energy``.
    """
    strategy = get_strategy(NAME)
    deepest = max(
        get_feature(name).warmup_bars + (strategy.rank_window - 1 if name in RANKED_FEATURES else 0)
        for name in strategy.features
    )
    assert deepest == get_feature("direction").warmup_bars
    # And then some, because the machine is a recursion on top of the features
    # and has its own cold start. Measured at warmup 1920 -- exactly `deepest` --
    # tests/test_strategy_metadata.py finds 52 to 156 divergences out of 300
    # probes depending on the seed; it needs about 60 more bars to find none.
    assert strategy.warmup_bars > deepest, "no room left for the machine to converge"


def test_no_signal_fires_before_every_feature_it_reads_is_measurable():
    """Not "no signal inside warmup" -- this repo lets the *engine* mask that.

    ``turnaround_v1`` emits 1016 entries inside its own declared warmup and
    ``ema_cross`` 3839; ``tests/test_backtest_warmup.py`` is what pins the
    masking. The claim that belongs to the strategy is narrower and stronger:
    nothing fires while the machine cannot see anything at all.
    """
    strategy = get_strategy(NAME)
    signals = strategy.generate_signals(probe_frame(strategy))
    blind = max(get_feature(name).warmup_bars for name in strategy.features)
    for field in ("long_entries", "long_exits", "short_entries", "short_exits"):
        assert not getattr(signals, field).iloc[:blind].any(), f"{field} fired while blind"
    assert (signals.position_size.iloc[:blind] == 0.0).all()


def test_it_trades_often_enough_for_the_safety_suites_to_mean_something():
    """Both safety suites compare signal *lists*; an inert strategy passes them empty."""
    strategy = get_strategy(NAME)
    signals = strategy.generate_signals(probe_frame(strategy))
    fired = sum(
        int(getattr(signals, field).iloc[strategy.warmup_bars :].sum())
        for field in ("long_entries", "long_exits", "short_entries", "short_exits")
    )
    assert fired >= 20, f"only {fired} signals in {PROBE_SPAN} bars past warmup"


def test_position_size_is_present_and_bounded():
    strategy = get_strategy(NAME)
    size = strategy.generate_signals(probe_frame(strategy)).position_size
    assert size is not None
    assert size.dropna().between(0.0, 1.0).all()


def test_position_size_carries_the_state_target_rather_than_a_constant():
    """The engine applies it at entry only, so the entry bar's state is the size.

    Without funding there is nothing to damp with, so every non-flat size must
    land exactly on one of the policy's state constants.
    """
    strategy = get_strategy(NAME)
    size = strategy.generate_signals(probe_frame(strategy)).position_size
    live = set(np.round(size[size > 0].unique(), 12))
    assert live, "never took a position"
    assert live <= set(STATE_TARGET_RISK.values())
    assert len(live) >= 2, "only one state ever sized a position; the table is decoration"


def test_the_flags_describe_a_consistent_position_path():
    """Replaying the flags from flat must never re-enter a side or exit a flat book.

    A 0.35 -> 1.00 target move while already long is not an entry the engine can
    fill -- ``from_signals`` ignores a repeated same-direction entry under
    ``accumulate=False`` -- so emitting one would write a signal into the
    append-only table that no backtest ever acts on.
    """
    strategy = get_strategy(NAME)
    signals = strategy.generate_signals(probe_frame(strategy))
    long_entries = signals.long_entries.to_numpy()
    long_exits = signals.long_exits.to_numpy()
    short_entries = signals.short_entries.to_numpy()
    short_exits = signals.short_exits.to_numpy()

    side = 0
    for position in range(len(long_entries)):
        # Exits before entries, because a reversal fires both on one bar.
        if long_exits[position]:
            assert side == 1, f"bar {position}: exited a long the book was not holding"
            side = 0
        if short_exits[position]:
            assert side == -1, f"bar {position}: exited a short the book was not holding"
            side = 0
        if long_entries[position]:
            assert side == 0, f"bar {position}: entered long while already on {side}"
            side = 1
        if short_entries[position]:
            assert side == 0, f"bar {position}: entered short while already on {side}"
            side = -1


def test_a_frame_without_funding_runs_without_crowding_and_records_that():
    """Crowding needs a funding column that only perps carry.

    ``features.flow.Crowding`` refuses such a frame outright, which is right for
    a feature declining to claim a measurement -- but a strategy that refused
    every spot and equity frame could not be compared against anything.
    """
    strategy = get_strategy(NAME)
    signals = strategy.generate_signals(probe_frame(strategy))
    assert signals.metadata["crowding_measured"] is False


def test_funding_actually_reaches_the_signals():
    """The other half: when crowding IS available it must change something."""
    strategy = get_strategy(NAME)
    with_funding = strategy.generate_signals(probe_frame(strategy, funding=True))
    without = strategy.generate_signals(probe_frame(strategy))

    assert with_funding.metadata["crowding_measured"] is True
    assert not with_funding.position_size.equals(without.position_size)


def test_disabling_shorts_leaves_the_long_side_untouched():
    long_only = get_strategy(NAME, allow_shorts=False)
    both = get_strategy(NAME)
    frame = probe_frame(both)

    restricted = long_only.generate_signals(frame)
    unrestricted = both.generate_signals(frame)

    assert not restricted.short_entries.any()
    assert unrestricted.short_entries.any(), "setup failed: it never shorted anyway"
    assert restricted.long_entries.sum() > 0


def test_the_machine_and_the_policy_are_reachable_from_the_adapter():
    """A state the adapter can never produce is a rule nobody can trade."""
    strategy = get_strategy(NAME)
    features, _ = strategy.feature_frame(probe_frame(strategy))
    states = set(strategy.machine.run(features))
    assert states == set(MarketState), f"unreachable states: {set(MarketState) - states}"


@pytest.mark.db
def test_it_runs_on_the_stored_perp_history_with_real_funding():
    """The frame this was designed against: BTC/USDT perp 4h with its own carry."""
    from strategy_lab.db.candles import load_candles
    from strategy_lab.db.funding import load_funding

    df = load_candles(
        exchange="binance", market_type="perp", symbol="BTC/USDT", timeframe="4h"
    )
    if df.empty:
        pytest.skip("no stored BTC/USDT perp 4h candles; run fetch-perp first")
    funding = load_funding(exchange="binance", market_type="perp", symbol="BTC/USDT")
    if funding.empty:
        pytest.skip("no stored BTC/USDT funding; run fetch-perp first")
    df = df.assign(
        **{FUNDING_COLUMN: align_funding_to_bars(df.index, funding["funding_rate"])}
    )

    strategy = get_strategy(NAME)
    signals = strategy.generate_signals(df)
    assert signals.metadata["crowding_measured"] is True

    fired = sum(
        int(getattr(signals, field).iloc[strategy.warmup_bars :].sum())
        for field in ("long_entries", "long_exits", "short_entries", "short_exits")
    )
    assert fired >= 20, f"only {fired} signals over {len(df)} real bars"
    assert signals.position_size.between(0.0, 1.0).all()
