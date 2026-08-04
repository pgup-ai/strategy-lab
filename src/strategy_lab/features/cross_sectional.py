"""Features read across instruments at one instant, rather than along one series.

Both functions read a single :class:`MarketSnapshot` and nothing else -- no
history, no lookahead. The snapshot they are handed is complete in the only sense
a live feed can establish: a later event has already proved that nothing more is
coming for that timestamp.

**These features assume one shared timeframe across the universe.** A snapshot
groups bars by close time, so a daily bar meets a 4h bar only at day boundaries;
over a mixed-timeframe universe most snapshots therefore hold a single
instrument. That is why ``breadth`` refuses to score a snapshot below
``min_instruments`` instead of dividing by whatever happens to be present.
"""

from __future__ import annotations

from strategy_lab.core.types import Bar, InstrumentId, MarketSnapshot


def breadth(snapshot: MarketSnapshot, *, min_instruments: int = 2) -> float:
    """Fraction of the instruments present whose bar closed above its open.

    Only instruments actually in the snapshot are counted. An absent instrument is
    absent, not flat: crypto trades around the clock and equities do not, and
    scoring a closed session as "did not advance" turns every weekend into a bear
    signal.

    ``min_instruments`` is the smallest universe worth calling a cross-section.
    Below it the answer is arithmetically fine and semantically empty -- breadth
    over one instrument is only ever 0.0 or 1.0, which is that instrument's
    direction wearing a cross-sectional name. Pass ``min_instruments=1``
    deliberately if that is what you want.
    """
    present = len(snapshot)
    if present == 0:
        raise ValueError(
            "breadth over no instruments is undefined; returning 0.0 would report "
            "a halted or unlisted universe as one where nothing advanced"
        )
    if present < min_instruments:
        raise ValueError(
            f"breadth over {present} of a required {min_instruments} instruments is "
            f"not a cross-section (min_instruments); snapshot at {snapshot.ts_event_ms}"
        )
    advancing = sum(1 for bar in snapshot.bars.values() if _direction(bar) > 0)
    return advancing / present


def confirms(snapshot: MarketSnapshot, *, leader: InstrumentId, quorum: float) -> bool:
    """Whether at least ``quorum`` of the followers moved the leader's way.

    The leader is excluded from its own vote, so this measures agreement rather
    than rewarding a large leader move. False when the leader is missing, when it
    closed unchanged (there is no direction to agree with), and when nothing else
    is present -- in every case there is no confirmation to claim, and defaulting
    to True would let a one-instrument snapshot wave a trade through.
    """
    leader_bar = snapshot.get(leader)
    if leader_bar is None:
        return False
    direction = _direction(leader_bar)
    if direction == 0:
        return False

    followers = [bar for instrument, bar in snapshot.bars.items() if instrument != leader]
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
