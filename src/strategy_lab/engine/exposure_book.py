"""What the account holds, driven by the ``ExposureSignal``s the event path emits.

The continuous counterpart of ``engine.book.PaperBook``, and a separate class for
the reason there are two engines and two runners: a class dispatching on contract
runs both and says nothing when one half breaks.

**The band has already been applied.** ``ExposureRunner`` emits only on the bars a
target moved far enough to act on, which is ``exposure_engine._banded``'s job in
the vectorized path. So this converts a *submitted* target into an order and does
not decide whether to submit -- putting the band here too would apply it twice.

**A target is a currency value against initial cash**, matching
``from_orders(size_type="targetvalue")``: ``target x position_pct x cash``, never
a fraction of current equity. ``targetpercent`` would compound, which is the rule
this repo has broken once and named for it.

The order is the *difference* between the value held and the value wanted, so a
target that does not move issues nothing and a target crossing zero issues one
order through the flat point rather than two.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from strategy_lab.engine.fills import CostModel, Direction, Fill, build_fill, fill_price


@dataclass
class ExposureBook:
    """A position held to a signed target value.

    ``position`` is a signed quantity, so a short is negative -- unlike
    ``PaperBook``'s round trips, which carry an unsigned quantity and a direction,
    because a level has no round trip to name.
    """

    cash: float = 10_000.0
    position_pct: float = 0.95
    costs: CostModel = field(default_factory=CostModel)

    balance: float = field(default=0.0, init=False)
    position: float = field(default=0.0, init=False)
    fills: list[Fill] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.balance = self.cash

    def on_target(self, target: float, *, close: float, ts_bar_ms: int) -> Fill | None:
        """Move the book to ``target``, and return the order that did it.

        ``None`` when the held quantity already matches, which is not the same as
        the band declining to submit: the band lives in the runner, and this is
        the case where a submitted target happens to need no trade.
        """
        wanted_value = target * self.position_pct * self.cash
        # Priced at the raw close, the same denominator ``from_orders`` uses to
        # turn a target *value* into a target *quantity* -- the slippage is a
        # cost of the resulting order, not an input to how big it is.
        wanted_quantity = wanted_value / close
        delta = wanted_quantity - self.position
        if delta == 0.0:
            return None

        direction = Direction.BUY if delta > 0 else Direction.SELL
        quantity = abs(delta)
        if direction is Direction.BUY:
            # The same clip ``PaperBook`` applies, and for the same reason: this
            # book's whole claim is that it agrees with ``from_orders``, whose own
            # module docstring says a drawdown deep enough "fills what cash
            # covers". Measured, it does not currently bind -- 938 of 938 orders
            # over the full 15,128-bar stored history match either way -- so this
            # buys agreement *by construction* rather than by the data never
            # having drawn down far enough.
            price = fill_price(close, direction, self.costs.slippage)
            quantity = max(0.0, min(quantity, self.balance / (price * (1.0 + self.costs.fee))))
            if quantity == 0.0:
                return None
        fill = build_fill(
            ts_bar_ms=ts_bar_ms,
            direction=direction,
            quantity=quantity,
            close=close,
            costs=self.costs,
        )
        self.balance += (
            -(fill.notional + fill.fee)
            if direction is Direction.BUY
            else fill.notional - fill.fee
        )
        self.position += quantity if direction is Direction.BUY else -quantity
        self.fills.append(fill)
        return fill


__all__ = ["ExposureBook"]
