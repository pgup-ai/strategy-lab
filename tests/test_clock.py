from __future__ import annotations

from strategy_lab.core.clock import Clock, LiveClock, SimClock


def test_sim_clock_starts_at_zero_and_advances_only_when_told():
    clock = SimClock()
    assert clock.now_ms() == 0
    clock.advance_to(1_785_724_200_000)
    assert clock.now_ms() == 1_785_724_200_000
    clock.advance_to(1_785_724_200_000)
    assert clock.now_ms() == 1_785_724_200_000


def test_sim_clock_never_goes_backwards():
    clock = SimClock(start_ms=1_000)
    clock.advance_to(500)
    assert clock.now_ms() == 1_000


def test_sim_clock_ignores_stale_replay_after_advancing_forward():
    # Mirrors a websocket reconnect: the feed replays bars already seen, whose
    # event timestamps are older than the latest bar the clock advanced to.
    clock = SimClock()
    clock.advance_to(2_000)
    clock.advance_to(1_000)
    assert clock.now_ms() == 2_000
    # Time must still be able to move forward again afterwards.
    clock.advance_to(3_000)
    assert clock.now_ms() == 3_000


def test_live_clock_returns_millisecond_epoch():
    now = LiveClock().now_ms()
    assert isinstance(now, int)
    # Sanity band: after 2020-01-01 and before 2100-01-01, in ms.
    assert 1_577_836_800_000 < now < 4_102_444_800_000


def test_live_clock_and_sim_clock_satisfy_the_clock_protocol():
    assert isinstance(LiveClock(), Clock)
    assert isinstance(SimClock(), Clock)
