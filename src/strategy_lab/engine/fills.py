"""What a decision costs when it reaches the book.

The rules below are **derived from the installed vectorbt**, not chosen: a book
that priced fills its own way would be a second answer to a question
``run_backtest`` already answers, and the whole point of a paper book is that it
agrees with the backtest it will be compared against.

Measured with a controlled 12-bar signal set -- entries at bars 1, 2 and 6, an
exit at 4, a short entry at 8, a short exit at 10, ``fees=0.001``,
``slippage=0.002``, ``cash=10_000``, ``position_pct=0.95``:

======================================  ==========================================
rule                                    evidence
======================================  ==========================================
size = cash x pct x scale / close[t]    94.059406 at close 101 = 10000x0.95/101
buy fills at close x (1 + slippage)     101 -> 101.202, 110 -> 110.220
sell fills at close x (1 - slippage)    104 -> 103.792, 108 -> 107.784
fee = rate x size x fill price          9.519 / (94.059406 x 101.202) = 0.001
======================================  ==========================================

**Size is computed off the raw close, not the filled price.** The two differ by
the slippage, and using the filled one would buy slightly less than the engine
does on every entry -- a small, permanent, one-directional divergence, which is
the worst kind to have between two paths that are supposed to be identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Direction(str, Enum):
    """Which way the order goes, which is all the pricing depends on.

    Not the *position* side: closing a long and opening a short are both
    ``SELL`` and both take the sell-side price, which is why a reversal fills
    both of its legs at the same number.
    """

    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class Fill:
    """One execution: what moved, at what price, and what it cost.

    ``quantity`` is unsigned and ``direction`` carries the sign, matching the
    engine's order records rather than inventing a signed convention the
    comparison would then have to undo.
    """

    ts_bar_ms: int
    direction: Direction
    quantity: float
    price: float
    fee: float

    @property
    def notional(self) -> float:
        return self.quantity * self.price


@dataclass(frozen=True, slots=True)
class CostModel:
    """The two rates a fill is priced with.

    A separate type from ``backtests.costs.CostModel`` on purpose: that one also
    carries ``cash`` and ``position_pct``, which are a *backtest's* configuration,
    while a book is handed its cash once and sizes from it. Sharing the class
    would put fields on this path that mean nothing here.
    """

    fee: float = 0.0005
    slippage: float = 0.0005


def fill_price(close: float, direction: Direction, slippage: float) -> float:
    """The price a market order at this bar's close actually gets.

    Slippage is charged *against* the order in both directions -- a buy pays up,
    a sell receives less -- which is what makes it a cost rather than a sign
    convention.
    """
    return close * (1.0 + slippage) if direction is Direction.BUY else close * (1.0 - slippage)


def entry_quantity(*, close: float, cash: float, position_pct: float, scale: float = 1.0) -> float:
    """How much an entry buys, off **initial** cash and the raw close.

    Non-compounding, which is the repo's standing rule: a book that has doubled
    its equity sizes the next entry exactly as it sized the first. ``cash`` is
    therefore the account's *starting* cash and never its current value.
    """
    return cash * position_pct * scale / close


def build_fill(
    *,
    ts_bar_ms: int,
    direction: Direction,
    quantity: float,
    close: float,
    costs: CostModel,
) -> Fill:
    if quantity < 0:
        # ``Fill.quantity`` is unsigned by contract and ``direction`` carries the
        # sign. A negative one settles as a credit and moves the position the
        # wrong way, which is silent -- so it is refused where it can still be
        # attributed to its caller.
        raise ValueError(f"quantity must be unsigned, got {quantity!r}; direction carries the sign")
    price = fill_price(close, direction, costs.slippage)
    return Fill(
        ts_bar_ms=ts_bar_ms,
        direction=direction,
        quantity=quantity,
        price=price,
        fee=costs.fee * quantity * price,
    )


__all__ = ["CostModel", "Direction", "Fill", "build_fill", "entry_quantity", "fill_price"]
