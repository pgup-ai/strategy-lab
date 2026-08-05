"""State plus conditioning to a signed target risk. Applied at entry, once.

**The engine consumes a target only on the bar that opens a position.**
``vbt.Portfolio.from_signals`` defaults to ``accumulate=False``, and R2 measured
what that means against the installed vectorbt: with ``size = [1,1,1,1,5,5,5,5]``
and an entry on every bar, the result is one order of size 1.0 and a position
that never resizes. So the state on the *entry* bar picks the size, later states
can close the position but cannot scale it, and the charter's exhaustion ->
distribution taper is deferred to R6 where the continuous-exposure contract
lands. Building the taper here would ship a state machine whose defining
behaviour the engine silently ignores -- the failure R2 found in "volatility
targeting" that was really entry scaling. A reader who assumes rebalancing will
mis-read every result this policy produces.

**The conditioning is not monotone, and a threshold rule is the wrong shape.**
R4 measured the IC of ``direction`` against the ``[t+1, t+31]`` return on 13,167
BTC/USDT perp 4h bars, split by ``strength`` tercile:

===========  ========  ==========================  =====
tercile      IC@30b    first half / second half    n
===========  ========  ==========================  =====
low          +0.0022   -0.0361 / +0.0468           4,389
**mid**      -0.1128   -0.1201 / -0.1101           4,389
high         +0.1314   +0.1953 / +0.0621           4,389
===========  ========  ==========================  =====

Unconditional IC is +0.0385. The middle tercile carries a *larger absolute* IC
than the unconditional signal, with the opposite sign and both halves agreeing:
middling trend quality is a mean-reversion regime, and it is the most usable
thing R4 produced. "Trade when strength is high" would discard the better of
the two live regimes. The low tercile is where the halves disagree in sign --
noise, and the one band that must be flat.

That makes ``state_machine_v1`` a hybrid rather than a trend follower, and the
mid band is where the machine spends much of its ``EXHAUSTION`` time: ``RIDING``
is left on any bar that stops advancing, which includes strength dropping out of
the top tercile but also a lean decaying below ``direction_floor``.

``crowding`` modulates size rather than direction. It was consistently negative
at every horizon R4 measured -- high carry, lower forward return -- so it
shrinks a position on the side that is *paying* and leaves the other side
alone. Shrinking both symmetrically would throw away the sign of the only
measurement this input has.

**``direction`` sets the sign and nothing else.** Scaling the target by
``abs(direction)`` is the obvious next step and is deliberately not taken, for
two reasons. R4 measured direction's information *coefficient*, which is a
claim about which way, not about how much; sizing by its magnitude is a second
claim nothing has measured. And it would make the size a continuous function of
a feature built on pandas rolling means, whose online accumulation is only
~1e-12 reproducible from a cold start -- measured here as a one-ULP
disagreement that no amount of ``warmup_bars`` removes, because it is not a
convergence problem. State scales are a small table of constants instead, so
the size is exactly reproducible wherever the state is.

The one continuous term left is the crowding damping, which inherits that same
~1e-12 rolling noise. It only bites on frames that carry funding, and it moves
a position size by less than a millionth of a basis point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_lab.state.machine import MarketState

# Fraction of full risk each state is worth. The charter's §2.4 table, minus
# the taper it cannot express here: compression flat, riding full, exhaustion
# a little over half.
STATE_TARGET_RISK: dict[MarketState, float] = {
    MarketState.COMPRESSION: 0.00,
    MarketState.BREAKOUT: 0.35,
    MarketState.CONFIRMED: 0.70,
    MarketState.RIDING: 1.00,
    MarketState.EXHAUSTION: 0.55,
    MarketState.RESET: 0.00,
}

# Tercile boundaries, in the rank space ``state.machine`` documents. These are
# R4's measured split, not a tuned pair.
FOLLOW_BAND = 2.0 / 3.0
NOISE_BAND = 1.0 / 3.0

# How much of the target the most extreme carry reading may remove. Half rather
# than all: crowding is one weak input, and a rule that could zero a position on
# its own would be a direction call in a size modifier's clothing.
CROWDING_PENALTY = 0.5


def target_risk(
    *,
    state: MarketState,
    direction: float,
    strength: float,
    crowding: float,
    follow_band: float = FOLLOW_BAND,
    noise_band: float = NOISE_BAND,
    crowding_penalty: float = CROWDING_PENALTY,
) -> float:
    """Signed target risk for one bar, in -1..1. Positive is long.

    A thin wrapper over :func:`target_risk_series` rather than a second
    implementation: the vectorized form is what the strategy adapter runs, and
    two copies of a rule this fiddly would drift.
    """
    series = target_risk_series(
        states=pd.Series([state]),
        direction=pd.Series([direction], dtype="float64"),
        strength=pd.Series([strength], dtype="float64"),
        crowding=pd.Series([crowding], dtype="float64"),
        follow_band=follow_band,
        noise_band=noise_band,
        crowding_penalty=crowding_penalty,
    )
    return float(series.iloc[0])


def target_risk_series(
    *,
    states: pd.Series,
    direction: pd.Series,
    strength: pd.Series,
    crowding: pd.Series,
    follow_band: float = FOLLOW_BAND,
    noise_band: float = NOISE_BAND,
    crowding_penalty: float = CROWDING_PENALTY,
) -> pd.Series:
    """:func:`target_risk` over a whole frame, on ``states``' index."""
    if not 0.0 <= noise_band < follow_band <= 1.0:
        raise ValueError(
            f"need 0 <= noise_band ({noise_band}) < follow_band ({follow_band}) <= 1; "
            "collapsing them removes the mid band, which is where R4 measured the "
            "larger absolute IC"
        )
    # Outside [0, 1] the damping stops damping: at 2.0 a fully crowded long comes
    # out negative -- a short wearing a long's target -- and at -1.0 the target
    # exceeds full risk. Both are silent, so they are refused rather than clipped.
    if not 0.0 <= crowding_penalty <= 1.0:
        raise ValueError(
            f"crowding_penalty ({crowding_penalty}) must lie in [0, 1]; outside it "
            "the damping flips the side or pushes the target past full risk"
        )

    scale = states.map(STATE_TARGET_RISK).to_numpy(dtype="float64")
    lean = direction.to_numpy(dtype="float64")
    quality = strength.to_numpy(dtype="float64")
    carry = crowding.to_numpy(dtype="float64")

    # High follows the lean, mid fades it, low stands aside -- and an
    # unmeasurable bar stands aside too, rather than inheriting a band from a
    # NaN comparison that quietly reads False.
    side = np.where(quality >= follow_band, np.sign(lean), -np.sign(lean))
    live = np.isfinite(lean) & np.isfinite(quality) & np.isfinite(carry)
    side = np.where(live & (quality >= noise_band), side, 0.0)

    damping = 1.0 - crowding_penalty * _carry_pressure_against(side, carry)

    target = side * scale * damping
    # A flat bar is flat, whatever NaN the arithmetic above carried into it.
    return pd.Series(np.where(side == 0.0, 0.0, target), index=states.index, dtype="float64")


def _carry_pressure_against(side: np.ndarray, crowding: np.ndarray) -> np.ndarray:
    """How hard the carry leans on ``side`` specifically, 0..1.

    ``crowding`` above 0.5 means longs are paying to stay long, so it presses on
    a long and not on a short. R4 measured the IC as consistently negative at
    every horizon, which is a statement about the side, not about the magnitude.
    """
    extremity = np.minimum(2.0 * np.abs(crowding - 0.5), 1.0)
    return np.where(side == np.sign(crowding - 0.5), extremity, 0.0)


__all__ = [
    "CROWDING_PENALTY",
    "FOLLOW_BAND",
    "NOISE_BAND",
    "STATE_TARGET_RISK",
    "target_risk",
    "target_risk_series",
]
