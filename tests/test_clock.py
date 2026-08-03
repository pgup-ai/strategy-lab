from __future__ import annotations

from strategy_lab.core.clock import LiveClock, SimClock


def test_sim_clock_advances_forward_only():
    """A websocket reconnect replays already-seen bars, whose timestamps are older than
    the clock's. Rewinding there would make a replay depend on reconnect timing."""
    clock = SimClock(start_ms=1_000)
    clock.advance_to(500)
    assert clock.now_ms() == 1_000
    clock.advance_to(3_000)
    assert clock.now_ms() == 3_000
    assert SimClock().now_ms() == 0


def test_live_clock_returns_millisecond_epoch():
    now = LiveClock().now_ms()
    assert isinstance(now, int)
    # Sanity band: after 2020-01-01 and before 2100-01-01, in ms — catches a seconds return.
    assert 1_577_836_800_000 < now < 4_102_444_800_000
