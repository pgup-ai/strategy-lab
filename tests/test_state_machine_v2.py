"""``state_machine_v2`` is ``state_machine_v1`` with the collapse removed.

The central claim of R6 Task 4, and the thing every assertion here is arranged
around: v1 computes the full signed continuous target on every bar and then
keeps only ``np.sign`` of it plus an ``abs`` the engine reads once, at entry. v2
returns that same series whole. **Nothing else differs** -- not the machine, not
the policy, not ``STATE_TARGET_RISK``, not the features, not the warmup.

That matters beyond tidiness. v1's numbers are published (R5 gate), so Task 6
reads a v1-vs-v2 difference as the cost of the engine's truncation. Any
parameter v2 introduced, or any drift between two copies of the feature
pipeline, would silently convert that measurement into a search. Two tests below
are what license the reading: :func:`test_v2_recovers_exactly_the_target_v1_discards`
(the series are the same to float equality) and
:func:`test_v2_introduces_no_parameter_of_its_own` (the dataclasses carry the
same knobs).
"""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest

from strategy_lab.backtests.exposure_engine import run_exposure_backtest
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.state.machine import StateMachine
from strategy_lab.state.policy import STATE_TARGET_RISK
from strategy_lab.strategies.exposure_registry import (
    get_exposure_strategy,
    list_exposure_strategies,
)
from strategy_lab.strategies.registry import get_strategy, list_strategies
from strategy_lab.strategies.state_machine_v1 import StateMachineV1
from strategy_lab.strategies.state_machine_v2 import StateMachineV2
from tests.conftest import synthetic_ohlcv, synthetic_ohlcv_with_funding

NAME = "state_machine_v2"
TWIN = "state_machine_v1"

# Bars past warmup. 900 is where the machine walks the lifecycle often enough
# for the non-vacuity floors below to hold with margin on both frame kinds --
# measured at 21-45 target changes over 400 bars, so this is roughly double what
# the assertions need rather than tuned to them.
PROBE_SPAN = 900

# Floors that stop every equality here from being ``0.0 == 0.0`` -- same
# reasoning as ``MIN_CHANGES`` in tests/test_exposure_determinism.py.
MIN_CHANGES = 20
MIN_LEVELS = 3

FRAMES = (synthetic_ohlcv, synthetic_ohlcv_with_funding)
FRAME_IDS = ("no funding", "funding")

IDENTITY = MarketDataIdentity(
    exchange="binance", market_type="perp", symbol="BTC/USDT", timeframe="15m"
)


def frame(build, strategy, *, seed: int = 7) -> pd.DataFrame:
    return build(n=strategy.warmup_bars + PROBE_SPAN, seed=seed)


def assert_the_window_actually_moves(target: pd.Series, warm: int, *, label: str) -> None:
    live = target.iloc[warm:]
    changes = int((live.diff().fillna(0.0) != 0).sum())
    levels = len(set(np.round(live[live != 0.0].to_numpy(), 12)))
    assert changes >= MIN_CHANGES, f"{label}: only {changes} target changes past warmup"
    assert levels >= MIN_LEVELS, f"{label}: only {levels} distinct non-zero levels past warmup"
    assert (live > 0).any() and (live < 0).any(), f"{label}: the target never took both sides"


def test_it_is_registered_where_the_continuous_suites_look_and_nowhere_else():
    """Registration is what enrols it; the *wrong* registration is worse than none.

    Six parametrized tests iterate ``list_strategies()`` and every one calls
    ``generate_signals``, which this strategy does not have. Listing it there
    would error at best and skip silently at worst -- and a silent skip reads as
    a passing safety gate.
    """
    assert NAME in list_exposure_strategies()
    assert NAME not in list_strategies()
    assert TWIN in list_strategies(), "v2 is a sibling; v1 must stay registered"
    assert TWIN not in list_exposure_strategies()


def test_an_unknown_name_is_refused_with_the_available_ones():
    with pytest.raises(ValueError, match="state_machine_v2"):
        get_exposure_strategy("state_machine_v3")


@pytest.mark.parametrize("build", FRAMES, ids=FRAME_IDS)
def test_v2_recovers_exactly_the_target_v1_discards(build):
    """The phase's central assertion, on the default ``allow_shorts=True``.

    ``check_exact`` is not optional: the 1e-5 default tolerance would wave
    through any drift smaller than itself, and the claim is that the two series
    hold the same number, not a close one.

    Both frame kinds run because they exercise different halves of the policy:
    without funding every non-zero target is exactly a ``STATE_TARGET_RISK``
    constant, with funding the crowding damping makes it continuous.
    """
    v1, v2 = get_strategy(TWIN), get_exposure_strategy(NAME)
    df = frame(build, v2)

    signals = v1.generate_signals(df)
    exposure = v2.compute_target(df)

    assert_the_window_actually_moves(exposure.target, v2.warmup_bars, label=NAME)
    pd.testing.assert_series_equal(
        exposure.target.abs(), signals.position_size, check_exact=True, check_names=False
    )


@pytest.mark.parametrize("build", FRAMES, ids=FRAME_IDS)
def test_the_sign_of_v2s_target_reproduces_v1s_entry_and_exit_pattern(build):
    """The other half of "the same information": v1's booleans are recoverable.

    The rule is restated here rather than imported, on the same reasoning as
    ``StateMachine.LEGAL_TRANSITIONS``: a rule checked against its own
    implementation proves nothing. v1 enters long on a bar whose target is
    positive after a bar whose target was not, and exits on the reverse -- so
    every boolean it emits is a function of the series v2 hands over.
    """
    v1, v2 = get_strategy(TWIN), get_exposure_strategy(NAME)
    df = frame(build, v2)

    signals = v1.generate_signals(df)
    target = v2.compute_target(df).target
    # The book starts flat, so bar 0 is measured against a target of 0.
    previous = target.shift(1).fillna(0.0)

    long_now, long_before = target > 0, previous > 0
    short_now, short_before = target < 0, previous < 0

    assert int(signals.long_entries.sum() + signals.short_entries.sum()) >= MIN_CHANGES // 2
    for expected, actual in (
        (long_now & ~long_before, signals.long_entries),
        (~long_now & long_before, signals.long_exits),
        (short_now & ~short_before, signals.short_entries),
        (~short_now & short_before, signals.short_exits),
    ):
        pd.testing.assert_series_equal(
            actual, expected, check_exact=True, check_names=False
        )


@pytest.mark.parametrize("build", FRAMES, ids=FRAME_IDS)
def test_disabling_shorts_clips_the_target_exactly_as_v1_clips_its_size(build):
    """``allow_shorts=False`` must mean the same thing on both contracts.

    v1 clips the target at 0.0 before collapsing it, so a long-only v2 that
    merely refused to *report* shorts would still differ from v1 wherever the
    policy leaned short. The unrestricted assertion is what keeps this from
    passing on a frame that never shorted anyway.
    """
    v1 = get_strategy(TWIN, allow_shorts=False)
    v2 = get_exposure_strategy(NAME, allow_shorts=False)
    df = frame(build, v2)

    restricted = v2.compute_target(df).target
    assert (restricted >= 0.0).all()
    assert (get_exposure_strategy(NAME).compute_target(df).target < 0).any(), (
        "setup failed: the unrestricted target never went short, so the clip is untested"
    )
    pd.testing.assert_series_equal(
        restricted.abs(),
        v1.generate_signals(df).position_size,
        check_exact=True,
        check_names=False,
    )


def test_v2_introduces_no_parameter_of_its_own():
    """A new knob here would turn Task 6 from a measurement into a search.

    The plan originally listed 0.25 / 0.55 for breakout / confirmed; the shipped
    ``STATE_TARGET_RISK`` is 0.00 / 0.35 / 0.70 / 1.00 / 0.55 / 0.00 and those
    are R5's *published* numbers. Writing a table here instead would be tuning
    v2 against a v1 whose results are already on the record. Field-by-field
    rather than by eye, because the failure this guards is somebody adding a
    plausible-looking ``taper_floor`` later.
    """
    v1_fields = {f.name for f in fields(StateMachineV1)}
    v2_fields = {f.name for f in fields(StateMachineV2)}
    assert v1_fields == v2_fields

    v1, v2 = StateMachineV1(), StateMachineV2()
    for name in sorted(v1_fields - {"name", "version"}):
        assert getattr(v1, name) == getattr(v2, name), f"{name} differs between v1 and v2"


def test_the_target_carries_the_policy_table_rather_than_a_table_of_its_own():
    """Every level reachable without funding is a ``STATE_TARGET_RISK`` constant.

    Undamped frame only: with funding the crowding damping scales those
    constants continuously, so membership is not the claim there. The bound the
    two frames share -- ``|target| <= 1``, no NaN -- is ``TargetExposure``'s own
    and is asserted where it is enforced, in tests/test_target_exposure.py.
    """
    v2 = get_exposure_strategy(NAME)
    target = v2.compute_target(frame(synthetic_ohlcv, v2)).target

    live = set(np.round(target[target != 0.0].abs().unique(), 12))
    assert live <= set(STATE_TARGET_RISK.values())
    assert len(live) >= MIN_LEVELS, "one state ever sized a position; the table is decoration"


def test_warmup_matches_v1s_and_still_scales_with_the_machine_it_holds():
    """Same derivation, not a copied constant.

    A pinned number would be wrong for any machine but the default, and
    ``dataclasses.replace`` is how a sweep reaches a different one.
    """
    assert get_exposure_strategy(NAME).warmup_bars == get_strategy(TWIN).warmup_bars

    slower = None
    for config in (StateMachine(min_dwell=1), StateMachine(), StateMachine(min_dwell=1_000)):
        v1 = replace(get_strategy(TWIN), machine=config)
        v2 = replace(get_exposure_strategy(NAME), machine=config)
        assert v2.warmup_bars == v1.warmup_bars
        assert slower is None or v2.warmup_bars > slower
        slower = v2.warmup_bars


def test_funding_reaches_the_target_and_its_absence_is_recorded():
    """``crowding`` is the one input a spot or equity frame cannot supply.

    Same contract as v1's: run without it and say so, rather than refuse the
    frame or invent the neutral 0.5 that ``features.flow.Crowding`` declines to
    claim.
    """
    v2 = get_exposure_strategy(NAME)
    with_funding = v2.compute_target(frame(synthetic_ohlcv_with_funding, v2))
    without = v2.compute_target(frame(synthetic_ohlcv, v2))

    assert with_funding.metadata["crowding_measured"] is True
    assert without.metadata["crowding_measured"] is False
    assert not with_funding.target.equals(without.target)


def test_the_target_is_a_valid_exposure_on_a_frame_that_is_entirely_warmup():
    """Warmup is a leading run of 0.0, which the contract accepts and NaN is not.

    The policy zeroes any bar whose features are unmeasurable, so this holds
    without v2 filling anything -- and ``TargetExposure`` is what would raise if
    that ever stopped being true.
    """
    exposure = get_exposure_strategy(NAME).compute_target(synthetic_ohlcv(n=200))
    assert (exposure.target == 0.0).all()


def test_the_taper_resizes_a_position_the_boolean_path_could_only_open():
    """What the second contract is *for*, as an executable claim.

    v1 emits one entry per side change and the engine sizes it once; v2's target
    steps between states while the side is unchanged, and every such step is an
    order. If this ever stopped holding, v2 would be a slower spelling of v1.
    """
    v1, v2 = get_strategy(TWIN), get_exposure_strategy(NAME)
    df = frame(synthetic_ohlcv_with_funding, v2)

    signals = v1.generate_signals(df)
    result = run_exposure_backtest(df=df, strategy=v2, identity=IDENTITY)

    target = result.target.to_numpy()
    side = np.sign(target)
    previous_side = np.concatenate([[0.0], side[:-1]])
    previous_size = np.concatenate([[0.0], np.abs(target)[:-1]])
    held_and_resized = (side != 0.0) & (side == previous_side) & (np.abs(target) != previous_size)

    entries = int(signals.long_entries.sum() + signals.short_entries.sum())
    assert entries > 0, "setup failed: v1 never opened a position on this frame"
    assert int(held_and_resized.sum()) >= entries, (
        f"only {int(held_and_resized.sum())} in-side resizes against {entries} v1 entries; "
        "the taper is not doing anything the boolean path could not"
    )
    assert result.order_count > entries
