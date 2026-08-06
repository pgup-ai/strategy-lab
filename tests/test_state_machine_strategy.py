from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from strategy_lab.features.flow import FUNDING_COLUMN, align_funding_to_bars
from strategy_lab.features.registry import get_feature
from strategy_lab.state.machine import MarketState, StateMachine
from strategy_lab.state.policy import STATE_TARGET_RISK
from strategy_lab.strategies.registry import get_strategy, list_strategies
from strategy_lab.strategies.state_machine_v1 import RANKED_FEATURES, StateMachineV1
from strategy_lab.strategies.state_machine_v2 import StateMachineV2
from tests.conftest import synthetic_ohlcv, synthetic_ohlcv_with_funding

NAME = "state_machine_v1"

# Enough bars past warmup for the machine to walk the lifecycle several times.
PROBE_SPAN = 800

# Bars probed by the cold-start replay below, and the fields it compares --
# both narrower than ``tests/test_strategy_metadata.py``'s, which owns the same
# check for every registered strategy. This one exists only for machines that
# are not registered, so it is scoped to what this adapter actually emits.
COLD_START_PROBES = 200
SIGNAL_FIELDS = ("long_entries", "long_exits", "short_entries", "short_exits", "position_size")


def probe_frame(strategy, *, funding: bool = False, seed: int = 7) -> pd.DataFrame:
    build = synthetic_ohlcv_with_funding if funding else synthetic_ohlcv
    return build(n=strategy.warmup_bars + PROBE_SPAN, seed=seed)


def test_it_is_registered_and_covered_by_the_safety_suites():
    """Registration is what enrols it in test_lookahead and test_replay_determinism."""
    assert NAME in list_strategies()


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


def test_funding_reaches_the_signals_and_its_absence_is_recorded():
    """``features.flow.Crowding`` refuses a frame with no funding column outright,
    which is right for a feature declining to claim a measurement -- but a
    strategy that refused every spot and equity frame could not be compared
    against anything. So it runs without crowding and says so, and when crowding
    IS available it has to change something.
    """
    strategy = get_strategy(NAME)
    with_funding = strategy.generate_signals(probe_frame(strategy, funding=True))
    without = strategy.generate_signals(probe_frame(strategy))

    assert with_funding.metadata["crowding_measured"] is True
    assert without.metadata["crowding_measured"] is False
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


def test_energy_never_unmeasures_a_bar_the_other_inputs_had_measured():
    """The second half of R7b's no-op, and the half that is about warmups.

    ``StateMachine.run`` treats any missing input as a failure, so joining
    ``energy`` to ``measurable`` could in principle knock the machine to
    ``RESET`` on bars it used to walk -- which would move the published figures
    while ``energy_ceiling`` sat at its inert 1.0. It cannot, because ``Energy``
    costs 503 warmup bars against ``Direction``'s 1920 and neither has interior
    gaps, so ``energy``'s NaN prefix is strictly inside the frame's. Asserted on
    the adapter's own pipeline rather than on the arithmetic, because the
    arithmetic is what a future feature change would quietly invalidate.
    """
    strategy = get_strategy(NAME)
    features, _ = strategy.feature_frame(probe_frame(strategy))
    others = [name for name in strategy.features if name != "energy"]
    measurable = features[others].notna().all(axis=1)
    blinded = measurable & features["energy"].isna()
    assert not blinded.any(), (
        f"energy is the only unmeasurable input on {int(blinded.sum())} bars, so "
        "adding it to the machine's measurable set changed which bars it walks"
    )
    assert measurable.any(), "setup failed: no bar was measurable at all"


def test_the_machine_and_the_policy_are_reachable_from_the_adapter():
    """A state the adapter can never produce is a rule nobody can trade."""
    strategy = get_strategy(NAME)
    features, _ = strategy.feature_frame(probe_frame(strategy))
    states = set(strategy.machine.run(features))
    assert states == set(MarketState), f"unreachable states: {set(MarketState) - states}"


def test_the_declared_warmup_scales_with_the_machine_it_holds():
    """A pinned margin is wrong for any machine but the one it was measured on.

    ``sweep_parameters`` rebuilds every cell with ``dataclasses.replace``, so a
    grid over ``min_dwell`` is a grid over convergence cost: a cold machine
    needing 1,000 consecutive advancing bars per lifecycle step cannot be where
    a whole-history machine already is until it has had them. The adapter
    therefore spends ``StateMachine.convergence_bars`` rather than a constant,
    and what is asserted here is the consequence -- the margin past the features
    is never below the machine's own proven bound, and a slower machine is
    handed strictly more of it.
    """
    features_only = max(
        get_feature(name).warmup_bars
        + (get_strategy(NAME).rank_window - 1 if name in RANKED_FEATURES else 0)
        for name in get_strategy(NAME).features
    )
    slower = None
    for config in (StateMachine(min_dwell=1), StateMachine(), StateMachine(min_dwell=1_000)):
        strategy = replace(get_strategy(NAME), machine=config)
        margin = strategy.warmup_bars - features_only
        assert margin >= config.convergence_bars, (
            f"min_dwell={config.min_dwell} declares {margin} bars past its features, "
            f"under its own convergence bound of {config.convergence_bars}"
        )
        assert slower is None or margin > slower, (
            f"min_dwell={config.min_dwell} was handed {margin} bars, no more than the "
            f"{slower} a faster machine got"
        )
        slower = margin


@pytest.mark.parametrize(
    "machine",
    [
        StateMachine(min_dwell=8, cooldown=24, exhaustion_dwell=30),
        StateMachine(min_dwell=16, cooldown=32, exhaustion_dwell=48),
    ],
    ids=["slow", "slower"],
)
def test_a_configured_machine_gets_a_warmup_that_covers_it(machine):
    """The empirical half: is the derived margin actually enough?

    ``tests/test_strategy_metadata.py`` replays the cold start for the
    *registered* strategy only, which is the default machine. These two are
    reached the way ``sweep_parameters`` reaches a cell, and neither is
    registered anywhere.

    Both are chosen for being slow enough to have a materially different
    convergence cost (80 and 130 bars against the default's 34) *and* fast
    enough to still take positions inside the probed window -- the R5 trained
    cell is deliberately not here, because it holds nothing at all over 200 bars
    of this frame and every comparison would be ``0.0 == 0.0``. The non-vacuity
    assertion below is what keeps that from happening silently.
    """
    strategy = replace(get_strategy(NAME), machine=machine)
    warm = strategy.warmup_bars
    df = synthetic_ohlcv(n=warm + COLD_START_PROBES)
    whole = strategy.generate_signals(df)

    live = int((whole.position_size.iloc[warm:] > 0).sum())
    assert live >= 10, (
        f"only {live} of the {COLD_START_PROBES} probed bars hold a position, so the "
        "comparison below is mostly flat against flat. Raise COLD_START_PROBES."
    )

    divergences = []
    for position in range(warm, len(df)):
        cold = strategy.generate_signals(df.iloc[position - warm : position + 1])
        for field in SIGNAL_FIELDS:
            expected, actual = getattr(whole, field).iloc[position], getattr(cold, field).iloc[-1]
            if not (pd.isna(expected) and pd.isna(actual)) and expected != actual:
                divergences.append((position, field))
                break

    assert divergences == [], (
        f"a machine with min_dwell={machine.min_dwell}, cooldown={machine.cooldown}, "
        f"exhaustion_dwell={machine.exhaustion_dwell} declares warmup_bars={warm} and a "
        f"cold start from exactly that many bars disagrees at {divergences[:5]}"
    )


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


@pytest.mark.parametrize(
    "machine",
    [
        StateMachine(energy_ceiling=0.50),
        StateMachine(enter_energy=0.50, exit_energy=0.80),
    ],
    ids=["energy-gated", "energy-first"],
)
def test_an_energy_threshold_refuses_a_rank_window_it_is_not_ranked_over(machine):
    """``enter_strength`` and the energy thresholds must share a window.

    ``enter_strength`` is a threshold on a rank over ``rank_window``; the energy
    thresholds are thresholds on ``Energy``'s own ``percentile_window``. They
    coincide at the defaults, which is the only reason the two have ever been
    comparable -- and ``rank_window`` is a live field, so nothing else stops
    them drifting. Measured on the R5 frame, ``strength >= 0.80`` admits 23.8% /
    21.6% / 20.9% of bars at ``rank_window`` 240 / 480 / 960 while
    ``energy <= 0.35`` stays at 37.1%, so the *relative* selectivity of two
    gates moves with a parameter only one reads. That is M29, and it is refused
    rather than documented.
    """
    window = get_feature("energy").percentile_window
    for adapter in (StateMachineV1, StateMachineV2):
        with pytest.raises(ValueError, match="not comparable"):
            adapter(rank_window=window // 2, machine=machine)
        adapter(rank_window=window, machine=machine)  # the matched window is fine


def test_a_rank_window_change_is_still_allowed_without_an_energy_threshold():
    """The guard is about comparability, so it must not fire where nothing compares.

    Both energy thresholds are inert by default, and a ``rank_window`` sweep
    that never sets one is a legitimate experiment the guard has no business
    refusing.
    """
    window = get_feature("energy").percentile_window
    for adapter in (StateMachineV1, StateMachineV2):
        assert adapter(rank_window=window // 2).rank_window == window // 2
