"""What vectorbt does when one bar carries contradictory signals.

Read off ``portfolio/nb.py`` under this repo's settings -- ``accumulate=False``,
all three conflict modes ``ignore``, ``upon_opposite_entry=ReverseReduce`` -- and
kept in one place because **two paths that resolve conflicts differently disagree
about trades neither of them reports as an error**. ``backtests.sweep`` reduces a
signal set to a net position and ``engine.book`` fills one; both call this.

The rule is that ``ConflictMode.Ignore`` drops **both** signals rather than
picking a winner, so a bar carrying an entry and an exit on the same side, or
both entry directions, does *nothing*.

What is deliberately **not** here is the second half of the resolution -- that
from a long a short entry outranks a long exit -- because the two callers need it
in different shapes: the sweep wants a position in {-1, 0, +1} and the book wants
fills at prices. Sharing the part that is a pure function of four booleans, and
not the part that is a state machine over a position, is what keeps this a shared
*rule* rather than a shared *implementation* neither caller quite fits.
"""

from __future__ import annotations

from typing import NamedTuple


class ResolvedSignals(NamedTuple):
    long_entry: bool
    long_exit: bool
    short_entry: bool
    short_exit: bool


def resolve_conflicts(
    long_entry: bool, long_exit: bool, short_entry: bool, short_exit: bool
) -> ResolvedSignals:
    """Cancel the pairs vectorbt cancels, leaving the rest untouched.

    vectorbt guards this with ``if is_long_entry or is_short_entry``. That guard
    is not reproduced because it is provably inert -- every branch inside it
    already requires an entry -- and a mutation test confirmed it: adding it
    kills no mutant, so it would be untestable code.
    """
    if long_entry and long_exit:
        long_entry = long_exit = False
    if short_entry and short_exit:
        short_entry = short_exit = False
    if long_entry and short_entry:
        long_entry = short_entry = False
    return ResolvedSignals(long_entry, long_exit, short_entry, short_exit)


__all__ = ["ResolvedSignals", "resolve_conflicts"]
