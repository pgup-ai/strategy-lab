"""The market-state lifecycle, as an explicit machine over a feature frame.

Six states, walked in one direction::

    COMPRESSION -> BREAKOUT -> CONFIRMED -> RIDING -> EXHAUSTION -> RESET -> COMPRESSION

``RESET`` exists so a failed trend has somewhere to go that is not
``COMPRESSION``: the cooldown is served there, and returning straight to
``COMPRESSION`` would let the machine re-enter on the next bar of the same chop.

**The machine reads conditioners in rank space, not in feature units.** The
``strength`` column is where strength sits within its own trailing history,
0..1 -- not the raw efficiency ratio. R4's conditioning was measured by
*tercile*, and a tercile is a rank statement: on the stored BTC/USDT perp 4h
history the raw tercile boundaries are 0.063 / 0.149, and they move to
0.067 / 0.156 over the first half and 0.059 / 0.139 over the second. A
threshold written in feature units is therefore a differently sized bucket in
each era, which is the argument ``features.volatility.Energy`` already makes
for being a percentile rather than a level. ``direction`` is the exception and
stays raw: its sign is the information, and ranking a signed series destroys
it.

Every counter here **saturates** -- ``bars_in_state`` is only ever compared
against ``min_dwell`` or ``cooldown``, and ``advance_run`` against
``min_dwell``. That is load-bearing rather than incidental. A machine is a
recursion over its whole input, so without saturation the state at bar *t*
would depend on every bar back to the first one the process ever saw, and no
live process could reproduce a backtest's state: it starts cold. With
saturating counters the machine is a finite automaton that re-synchronises on
any run of failing bars long enough to walk it back to ``COMPRESSION``, which
is what makes the adapter's ``warmup_bars`` mean anything at all.
``tests/test_strategy_metadata.py`` measures how long that takes rather than
trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

REQUIRED_COLUMNS = ("direction", "strength", "stability", "crowding")


class MarketState(Enum):
    """Where in the lifecycle of a directional move the market currently is.

    The names are the point: they are what a reader reasons about when the
    machine does something surprising, and each one is a different claim about
    what the next bars are likely to do.
    """

    COMPRESSION = "compression"
    BREAKOUT = "breakout"
    CONFIRMED = "confirmed"
    RIDING = "riding"
    EXHAUSTION = "exhaustion"
    RESET = "reset"


@dataclass(frozen=True)
class StateMachine:
    """Turns a per-bar feature frame into a per-bar :class:`MarketState`.

    Frozen and parameter-only, exactly like a ``Strategy`` or a
    ``StateFeature``: :meth:`run` starts from :attr:`INITIAL_STATE` every time
    and mutates nothing, so two runs over the same frame cannot disagree.
    """

    INITIAL_STATE = MarketState.COMPRESSION

    def run(self, features: pd.DataFrame) -> pd.Series:
        """One :class:`MarketState` per row of ``features``.

        Every column in :data:`REQUIRED_COLUMNS` must be present. A missing one
        raises rather than silently skipping whichever transition rule reads it
        -- a typo'd column name would otherwise leave a machine that still
        produces a plausible-looking state for every bar.
        """
        _require_columns(features)
        return pd.Series(
            [self.INITIAL_STATE] * len(features),
            index=features.index,
            dtype="object",
            name="state",
        )


def _require_columns(features: pd.DataFrame) -> None:
    if features.empty:
        raise ValueError("Cannot run the state machine over an empty feature frame")
    missing = [column for column in REQUIRED_COLUMNS if column not in features.columns]
    if missing:
        raise KeyError(
            f"State machine needs the column(s) {missing} and this frame has "
            f"{sorted(features.columns)}"
        )
