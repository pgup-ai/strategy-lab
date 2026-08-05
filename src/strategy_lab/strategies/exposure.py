"""The continuous-exposure contract: a position level, read on every bar.

The second strategy contract in this repo, beside ``strategies.base.SignalSet``,
and it exists because of one measured limit. ``vbt.Portfolio.from_signals``
defaults to ``accumulate=False`` and consumes ``position_size`` on the bar that
*opens* a position and never again -- measured in R2 against the installed
vectorbt, ``size = [1,1,1,1,5,5,5,5]`` with an entry every bar yields one order
of size 1.0. A boolean signal set can therefore say *enter*, *exit* and *how big
to start*, but it cannot say *hold 55% now, having held 100% a bar ago*. That
sentence is the charter's per-state taper, and it is the thing no ``SignalSet``
can express.

So this contract carries a **level**, not an event: ``target[t]`` is the whole
of what the book should hold over bar *t*, whatever it held over *t-1*.
``backtests/exposure_engine.py`` executes it through
``Portfolio.from_orders(size_type="targetvalue")``, which turns each change in
that level into an order -- each change past its rebalance band, since between
decisions that engine deliberately leaves the book alone.

The two contracts coexist deliberately. The four original strategies keep
``SignalSet`` and their byte-identical results of record; a strategy that needs
to scale a position it already holds implements this one instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetExposure:
    """The signed fraction of risk capital to hold on each bar, in -1..1.

    ``+1`` is the whole risk budget long, ``-1`` the whole of it short, ``0``
    flat. It is a fraction of the *budget*, not of equity: the engine turns it
    into a percentage of the book by multiplying through ``position_pct``, so a
    target of 1.0 at ``position_pct=0.95`` asks for 95% of equity.

    **Warmup is a leading run of 0.0, and NaN is refused.** This inverts the
    convention in ``features/base.py``, on purpose. A feature's warmup rows are
    ``NaN`` because "not yet measurable" and "measured, and neutral" are
    different claims about the market, and a 0.0 there would assert the second
    while meaning the first. A *target* carries no such ambiguity: it says what
    to hold, and before an indicator has converged the answer is to hold
    nothing, which is exactly 0.0. A reader who knows the feature convention
    should read the asymmetry as deliberate rather than as an oversight -- a
    NaN target admitted here would reach ``from_orders`` as "no order", which is
    silently "keep whatever you were holding", the one reading that is wrong.

    Beyond +/-1 is refused because the book has no leverage: an order for 150%
    of a fully deployed account is filled at whatever the cash covers, so the
    run would quietly execute something other than the strategy it reports.
    """

    target: pd.Series
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_target(self.target)


@runtime_checkable
class ExposureStrategy(Protocol):
    """Shaped exactly like ``strategies.base.Strategy``, with one method changed.

    Same three metadata fields, one compute method, so the tooling written
    against the boolean protocol -- the manual registry, the cold-start warmup
    check, the poison probe -- transfers by construction rather than by a second
    implementation.

    ``runtime_checkable`` so a caller can ask whether an object satisfies this
    before dispatching it down the continuous path. That check is structural
    only: it confirms the four names exist and says nothing about whether
    ``compute_target`` is causal. The poison probe is what answers that.
    """

    name: str
    version: str
    warmup_bars: int

    def compute_target(self, df: pd.DataFrame) -> TargetExposure:
        ...


def _validate_target(target: pd.Series) -> None:
    values = target.to_numpy(dtype="float64")

    # NaN first: ``abs(nan) > 1`` is False, so the range check below would pass a
    # NaN through and report the wrong problem -- or no problem at all.
    missing = np.flatnonzero(np.isnan(values))
    if len(missing):
        raise ValueError(
            f"target holds NaN at {len(missing)} of {len(values)} rows, first at "
            f"position {int(missing[0])}. A target has no 'not yet measurable' "
            f"state the way a feature does: before warmup the position to hold is "
            f"nothing, which is 0.0. NaN would reach the engine as 'no order', "
            f"i.e. hold whatever was held before."
        )

    outside = np.flatnonzero(np.abs(values) > 1.0)
    if len(outside):
        position = int(outside[0])
        raise ValueError(
            f"target must lie in -1..1; position {position} is "
            f"{values[position]:g} ({len(outside)} of {len(values)} rows outside). "
            f"This book has no leverage, so a target beyond full deployment is "
            f"filled at whatever cash covers rather than refused."
        )


__all__ = ["ExposureStrategy", "TargetExposure"]
