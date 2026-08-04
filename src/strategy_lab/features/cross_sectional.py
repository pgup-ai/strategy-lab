"""Features read across instruments at one instant, rather than along one series.

Both read a single :class:`MarketSnapshot` and nothing else -- no history, no
lookahead -- and both assume one shared timeframe across the universe. A snapshot
groups bars by close time, so a daily bar meets a 4h bar only at day boundaries;
over a mixed-timeframe universe most snapshots hold a single instrument, which is
why ``breadth`` refuses one rather than dividing by whatever happened to show up.
"""

from __future__ import annotations

from strategy_lab.core.types import Bar, CandleId, MarketSnapshot


def breadth(snapshot: MarketSnapshot, *, min_instruments: int = 2) -> float:
    """Fraction of the instruments present whose bar closed above its open.

    An absent instrument is absent, not flat: crypto trades around the clock and
    equities do not, and scoring a closed session as "did not advance" turns every
    weekend into a bear signal. Below ``min_instruments`` the quotient is
    arithmetically fine and semantically empty -- breadth over one instrument is
    only ever 0.0 or 1.0, that instrument's direction wearing a cross-sectional
    name -- so it raises; pass ``min_instruments=1`` to opt out deliberately.
    """
    present = len(snapshot)
    if present == 0:
        raise ValueError("breadth over no instruments is undefined, not 0.0")
    if present < min_instruments:
        raise ValueError(
            f"breadth over {present} instruments is not a cross-section "
            f"(min_instruments={min_instruments}, snapshot at {snapshot.ts_event_ms})"
        )
    advancing = sum(1 for bar in snapshot.bars.values() if _direction(bar) > 0)
    return advancing / present


def confirms(snapshot: MarketSnapshot, *, leader: CandleId, quorum: float) -> bool:
    """Whether at least ``quorum`` of the followers moved the leader's way.

    The leader is excluded from its own vote, so this measures agreement rather
    than rewarding a large leader move. A missing leader, an unchanged leader and
    an empty field of followers are all False: there is no confirmation to claim,
    and True would let a one-instrument snapshot wave a trade through.

    ``quorum`` is a fraction, so it is range-checked rather than trusted: a
    negative one makes ``agreeing / followers >= quorum`` true for a field that
    unanimously disagreed, which reads as confirmation and is the opposite of one.
    """
    if not 0.0 <= quorum <= 1.0:
        raise ValueError(f"quorum must be a fraction in [0.0, 1.0], got {quorum}")

    leader_bar = snapshot.get(leader)
    if leader_bar is None:
        return False
    direction = _direction(leader_bar)
    if direction == 0:
        return False

    followers = [bar for candle, bar in snapshot.bars.items() if candle != leader]
    if not followers:
        return False
    agreeing = sum(1 for bar in followers if _direction(bar) == direction)
    return agreeing / len(followers) >= quorum


def _direction(bar: Bar) -> int:
    if bar.close > bar.open:
        return 1
    if bar.close < bar.open:
        return -1
    return 0
