"""The comparison R10h's gate is read off.

`scripts/r10h/delayed_oracle.py` is measurement code rather than library code, and
it went out with two correctness bugs that review found: a hardcoded feature list
that checked a name the strategy does not produce while never comparing one it
does, and a one-directional diff that read a replay-only row as agreement. Both
had the same consequence — a comparison that could not fail, printing as one that
passed.

That is a strong reason for measurement code to be tested even though it runs
once: the figure it produces is the phase's claim, and nothing downstream would
notice it being wrong.

Loaded by path because `scripts/` is not a package; it belongs with its phase.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from strategy_lab.core.types import BarReason, InstrumentId

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "r10h" / "delayed_oracle.py"


@pytest.fixture(scope="module")
def oracle():
    spec = importlib.util.spec_from_file_location("r10h_delayed_oracle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reason(ts: int, state: str = "RIDING", **features) -> BarReason:
    return BarReason(
        instrument=InstrumentId("binance", "perp", "BTC/USDT"),
        timeframe="15m",
        strategy_id="state_machine_v1",
        strategy_version="1.0.0",
        ts_bar_ms=ts,
        ts_emit_ms=ts,
        bar_is_closed=True,
        state=state,
        features=features,
    )


def test_matching_rows_agree(oracle):
    rows = [_reason(1, energy=0.4), _reason(2, energy=0.5)]

    compared, counts = oracle._compare_reasons(rows, list(rows))

    assert (compared, counts) == (2, {})


def test_an_unmeasurable_value_on_both_sides_is_agreement(oracle):
    """Storage refuses NaN and requires the caller to map it to `None`, so `None`
    is the spelling of "not measurable on this bar" — and the same absence on
    both paths is exactly what agreement looks like, not a difference."""
    compared, counts = oracle._compare_reasons(
        [_reason(1, crowding=None, energy=0.4)],
        [_reason(1, crowding=None, energy=0.4)],
    )

    assert (compared, counts) == (1, {})


def test_a_value_present_on_one_side_only_is_a_difference(oracle):
    """The bound on the above: `None` against a number is a real divergence, and
    it is what a live path running a feature neutral would look like."""
    _, counts = oracle._compare_reasons(
        [_reason(1, crowding=None)], [_reason(1, crowding=0.5)]
    )

    assert counts == {"crowding": 1}


def test_a_feature_name_on_one_side_only_is_a_difference(oracle):
    """The bug that hid `stability`: a name nobody carries is not in the union at
    all, but a name one side carries must never be skipped."""
    _, counts = oracle._compare_reasons(
        [_reason(1, energy=0.4)], [_reason(1, energy=0.4, stability=0.7)]
    )

    assert counts == {"stability": 1}


def test_a_moved_value_and_a_moved_state_are_both_counted(oracle):
    _, counts = oracle._compare_reasons(
        [_reason(1, energy=0.4)], [_reason(1, state="RESET", energy=0.9)]
    )

    assert counts == {"state": 1, "energy": 1}


def test_rows_on_one_side_only_are_counted_in_both_directions(oracle):
    """Iterating only `live` left every count at zero when the replay produced an
    extra row, which prints as agreement."""
    _, only_in_replay = oracle._compare_reasons([_reason(1)], [_reason(1), _reason(2)])
    _, only_in_live = oracle._compare_reasons([_reason(1), _reason(2)], [_reason(1)])

    assert only_in_replay == {"only in replay": 1}
    assert only_in_live == {"only in the live run": 1}


def test_funding_agrees_only_when_both_sides_say_the_same_thing(oracle):
    """The log writes an empty field for a bar with no rate, which pandas reads
    as `NaN` — and `NaN != NaN`, so "both absent" needs its own branch."""
    column = oracle.FUNDING_COLUMN
    absent, present, moved = {column: float("nan")}, {column: 0.0001}, {column: 0.0009}

    assert oracle._funding_agrees(absent, absent)
    assert oracle._funding_agrees(present, dict(present))
    assert not oracle._funding_agrees(absent, present)
    assert not oracle._funding_agrees(present, absent)
    assert not oracle._funding_agrees(present, moved)
