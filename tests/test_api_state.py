"""The state reading, and the refusal it has to make for itself.

Every other path reaches ``engine._warmup_bars``, which raises when warmup
covers the frame. ``build_state`` runs no engine, so without its own check the
failure is silent rather than loud: the machine answers on every bar, reads
unmeasurable input as *failing*, and failing renders as ``COMPRESSION``. A view
whose whole purpose is "when is this chopping" would then answer "chopping" over
exactly the range where it knows nothing.

That is what most of this file is about. The rest pins that the reading is a
*slice* of the analysis path rather than a second derivation of it — the board's
rule (M36), which is the only thing keeping a monitor from quietly disagreeing
with the chart it links to.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from strategy_lab.api.analysis import build_analysis, registered_strategies
from strategy_lab.api.state import StateUnavailable, build_state
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.strategies.registry import get_strategy
from tests.conftest import synthetic_ohlcv

_SPOT = MarketDataIdentity(
    exchange="binance", market_type="spot", symbol="BTC/USDT", timeframe="4h"
)
_MACHINE = "state_machine_v1"
# Derived, not pinned: R12 moved it 2,192 -> 847 and a literal here would fail
# for a reason that says nothing about what these tests are checking.
_WARMUP = get_strategy(_MACHINE).warmup_bars

# `board_window`'s answer off-perp: no bounds, and no funding question asked.
_WHOLE = SimpleNamespace(start=None, end=None)


@pytest.fixture
def deep_frame(monkeypatch):
    """Long enough to clear ``state_machine_v1``'s warmup with room to read."""
    df = synthetic_ohlcv(n=_WARMUP + 400, freq="4h")
    monkeypatch.setattr("strategy_lab.api.analysis.load_candles", lambda **_: df)
    return df


@pytest.fixture
def shallow_frame(monkeypatch):
    df = synthetic_ohlcv(n=_WARMUP - 1, freq="4h")
    monkeypatch.setattr("strategy_lab.api.analysis.load_candles", lambda **_: df)
    return df


def test_a_frame_inside_warmup_is_refused_with_the_reason(shallow_frame):
    """**The gate.** Not "no data" — the frame is one bar short of the warmup.
    The machine would answer on every one of those bars, and the answer would be
    compression."""
    with pytest.raises(StateUnavailable) as refused:
        build_state(_SPOT, strategy_name=_MACHINE)

    message = str(refused.value)
    assert f"{_WARMUP:,}" in message and f"{_WARMUP - 1:,}" in message
    assert "compression" in message, (
        "the refusal did not say what would have been drawn instead"
    )


def test_one_bar_past_warmup_is_enough_to_answer(monkeypatch):
    """The bound on the refusal above, so it cannot quietly become
    ``> warmup + k`` and refuse frames that can answer."""
    df = synthetic_ohlcv(n=_WARMUP + 1, freq="4h")
    monkeypatch.setattr("strategy_lab.api.analysis.load_candles", lambda **_: df)

    payload = build_state(_SPOT, strategy_name=_MACHINE)

    assert payload.provenance.measurable_bars == 1
    assert payload.provenance.measurable_from == str(df.index[_WARMUP])


def test_the_reading_carries_every_feature_the_machine_reads(deep_frame):
    payload = build_state(_SPOT, strategy_name=_MACHINE)

    assert set(payload.why.features) == {
        "crowding", "direction", "energy", "stability", "strength"
    }
    assert len(payload.why.states) == len(payload.bars) == len(deep_frame)
    for name, values in payload.why.features.items():
        assert len(values) == len(deep_frame), name


def test_warmup_rows_are_null_features_rather_than_zeros(deep_frame):
    """A 0.0 there reads as "measured, and neutral", which is a different claim
    about the market from "not measurable yet"."""
    payload = build_state(_SPOT, strategy_name=_MACHINE)
    direction = payload.why.features["direction"]

    assert direction[0] is None
    assert any(value is not None for value in direction[_WARMUP:]), (
        "nothing was measurable after warmup, so the null check above proves nothing"
    )


@pytest.mark.db
def test_a_perp_is_bounded_by_its_own_funding_before_being_asked_for(monkeypatch):
    """**The bug this test exists for**, on the dataset it broke.

    BTC/USDT perp 4h candles begin 2019-09-08 16:00 and the venue's first
    settlement lands 2019-09-10 08:00 — the documented ~40h leading gap, which is
    a fact about the venue rather than something to fetch. Asked unbounded, a
    crowding-reading machine is refused over that head, and the state view had no
    tile to hand it funded edges the way the instrument view does. So its default
    view of the flagship perp was a 409.

    Bounded by ``board_window`` — the board's own function, not a second rule —
    so the frame starts at the first settlement instead.
    """
    from strategy_lab.api.board import board_window
    from strategy_lab.market_data.base import MarketDataIdentity

    identity = MarketDataIdentity(
        exchange="binance", market_type="perp", symbol="BTC/USDT", timeframe="4h"
    )
    window = board_window(identity)
    if window.start is None:
        pytest.skip("no stored funding for BTC/USDT perp; nothing to bound by")

    payload = build_state(identity, strategy_name=_MACHINE)

    assert payload.provenance.first_bar >= window.start, (
        "the frame reaches behind the first stored settlement, which is the "
        "range the coverage guard refuses"
    )
    assert payload.provenance.crowding_measured is True


def test_a_spot_frame_is_not_bounded_by_funding_it_does_not_have(deep_frame, monkeypatch):
    """The other half of the dispatch. ``board_window`` returns the whole history
    off-perp and never asks about funding — an instrument that settles nothing
    must not have a window invented for it."""
    seen = []
    monkeypatch.setattr(
        "strategy_lab.api.state.board_window",
        lambda identity: (seen.append(identity.market_type) or _WHOLE),
    )

    payload = build_state(_SPOT, strategy_name=_MACHINE)

    assert seen == ["spot"]
    assert payload.provenance.bar_count == len(deep_frame)


def test_switching_funding_off_asks_for_the_whole_frame(deep_frame, monkeypatch):
    """Bounding a perp by its stored funding exists to keep the coverage guard
    happy. With `funding=False` no funding is loaded and no guard runs, so the
    bound would only narrow the window the caller asked for — silently, and after
    a span query for a run that ignores it."""
    seen = []
    monkeypatch.setattr(
        "strategy_lab.api.state.board_window",
        lambda identity: (seen.append(identity) or _WHOLE),
    )

    build_state(_SPOT, strategy_name=_MACHINE, funding=False)

    assert seen == [], "a funding span was fetched for a run that loads no funding"


def test_a_strategy_with_no_state_is_refused_by_name(deep_frame):
    with pytest.raises(StateUnavailable, match="no feature frame"):
        build_state(_SPOT, strategy_name="donchian")


def test_what_the_state_view_offers_is_what_it_accepts():
    """``has_state`` is published so the page can filter, and read back by
    ``build_state`` so the filter and the refusal cannot drift apart.

    Asserted against the predicate rather than a list of names, which would fail
    the day a third state machine is registered for a reason saying nothing.
    """
    from strategy_lab.api.analysis import _has_state, resolve_strategy

    offered = {entry.name for entry in registered_strategies() if entry.has_state}

    assert offered, "nothing offers a state, so the filter below proves nothing"
    for entry in registered_strategies():
        assert entry.has_state == _has_state(resolve_strategy(entry.name).strategy)


def test_the_reading_agrees_with_the_analysis_path_bar_for_bar(deep_frame):
    """**The oracle.** Same frame, same strategy, same states and features —
    which is the whole claim: this is ``build_analysis`` with the book left out,
    not a cheaper answer to the same question. A second derivation would be free
    to drift, and the drift would surface as a monitor contradicting the chart
    it links to."""
    state = build_state(_SPOT, strategy_name=_MACHINE)
    analysis = build_analysis(_SPOT, strategy_name=_MACHINE)

    assert state.bars == analysis.bars
    assert state.why.states == analysis.why.states
    assert state.why.features == analysis.why.features
    assert state.provenance.warmup_bars == analysis.provenance.warmup_bars
    assert state.provenance.crowding_measured == analysis.provenance.crowding_measured
    assert state.provenance.funding_attached == analysis.provenance.funding_attached
