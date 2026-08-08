"""What the account holds, driven by the ``Signal``s the event path emits.

``StrategyRunner`` turns bars into decisions; this turns decisions into a
position. Until R10g the event path stopped at the first half -- signals were
written to Postgres and nothing consumed them -- which is the gap
``StrategyRunner._extract`` has named since Phase 1a.

**Its correctness claim is that it agrees with ``run_backtest``**, and that is
checkable rather than asserted: ``tests/test_paper_book.py`` feeds it the same
``SignalSet`` the engine resolved, over a stored range, and compares every closed
trade against the backtest's own ``trades.csv``. It does **not** drive the book
through ``ReplayFeed`` and ``StrategyRunner`` -- that the streamed and vectorized
signals are identical is ``tests/test_replay_determinism.py``'s claim, and
restating it here would test that suite rather than this book. Where the two
could differ, the engine wins by definition -- see ``engine.fills`` for the
pricing rules, which are derived from the installed vectorbt rather than chosen.

Four rules carry the behaviour, and three of them are easy to get wrong:

0. **Contradictory signals on one bar cancel.** An entry and an exit on the
   same side, or both entry directions, leave the bar doing nothing -- vectorbt's
   ``ConflictMode.Ignore`` drops *both* rather than picking a winner. Shared with
   ``backtests.sweep`` through ``backtests.conflicts`` so the two paths cannot
   resolve a contradictory bar differently.
1. **A repeated same-side entry does nothing.** ``from_signals`` defaults to
   ``accumulate=False``, so a strategy signalling *enter long* on ten consecutive
   bars opens one position. A book that added to it would report a size the
   backtest never held.
2. **An opposite entry reverses without needing an exit.** A short entry while
   long closes the long and opens the short on the same bar, as two fills. Both
   are sells, so both take the sell-side price.
3. **Size is fixed at entry and never revisited.** The strategy's per-bar scale
   is read on the bar that opens the position and ignored afterwards, matching
   what R6 measured of ``from_signals``.
4. **Sizing is non-compounding in the *request* and clipped by cash in the
   *fill*.** The requested quantity is ``cash x position_pct x scale / close``
   off *initial* cash, so a book that has doubled its equity still asks for the
   same size. But an account that has *lost* money cannot pay for it, and
   ``from_signals`` fills what it can afford rather than refusing: measured,
   ``filled = min(requested, balance / (price x (1 + fee)))``, with the balance
   moved by ``-(qty x price + fee)`` on a buy and ``+(qty x price - fee)`` on a
   sell. This is the only reason the book carries a cash balance, and it was
   found by the oracle rather than by reasoning -- one trade of 19 on a real BTC
   range differed (M43).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from strategy_lab.backtests.conflicts import resolve_conflicts
from strategy_lab.core.types import Side, Signal
from strategy_lab.engine.fills import (
    CostModel,
    Direction,
    Fill,
    build_fill,
    entry_quantity,
    fill_price,
)

_OPENS = {Side.ENTER_LONG: 1, Side.ENTER_SHORT: -1}
_CLOSES = {Side.EXIT_LONG: 1, Side.EXIT_SHORT: -1}


@dataclass(frozen=True, slots=True)
class Trade:
    """A round trip, in the shape ``trades.csv`` reports one.

    Held closed-only: an open position is the book's ``position``, not a trade
    with a missing half, and reporting it as one would make a comparison against
    ``trades.csv`` count something the engine does not.
    """

    direction: str
    quantity: float
    entry_ts_ms: int
    entry_price: float
    entry_fee: float
    exit_ts_ms: int
    exit_price: float
    exit_fee: float


@dataclass
class PaperBook:
    """A position, and the fills that got it there.

    ``cash`` is the account's *starting* cash and never moves: it is what every
    entry sizes against, which is the non-compounding rule. ``balance`` is the
    running one, and exists for a narrower purpose than it looks -- it is what
    bounds a fill the account cannot pay for, and nothing else reads it.
    """

    cash: float = 10_000.0
    position_pct: float = 0.95
    costs: CostModel = field(default_factory=CostModel)

    balance: float = field(default=0.0, init=False)
    position: float = field(default=0.0, init=False)
    fills: list[Fill] = field(default_factory=list, init=False)
    trades: list[Trade] = field(default_factory=list, init=False)
    _entry: Fill | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.balance = self.cash

    @property
    def is_flat(self) -> bool:
        return self.position == 0.0

    def on_bar(
        self,
        signals: Sequence[Signal],
        *,
        close: float,
        ts_bar_ms: int,
        scale: float = 1.0,
    ) -> Sequence[Fill]:
        """Apply one bar's decisions and return what filled.

        Exits are applied before entries, which is what makes a reversal one
        bar's work rather than two: the closing leg has to be priced and recorded
        before the opening leg can size against a flat book. ``from_signals``
        resolves the same bar the same way -- measured, a short entry while long
        produces two trades sharing a timestamp.
        """
        emitted: list[Fill] = []
        sides = {signal.side for signal in signals}
        # Contradictory pairs cancel *before* anything is decided, through the
        # engine's own rule rather than a second reading of it: a bar carrying an
        # entry and an exit on one side, or both entry directions, does nothing.
        # Without this the set below is also unordered, so a bar carrying both
        # entries would have opened whichever side iteration happened to yield.
        resolved = resolve_conflicts(
            Side.ENTER_LONG in sides,
            Side.EXIT_LONG in sides,
            Side.ENTER_SHORT in sides,
            Side.EXIT_SHORT in sides,
        )
        opening = (
            Side.ENTER_LONG
            if resolved.long_entry
            else Side.ENTER_SHORT
            if resolved.short_entry
            else None
        )
        closing_by_exit = not self.is_flat and (
            (resolved.long_exit and self._sign > 0) or (resolved.short_exit and self._sign < 0)
        )
        # An opposite-side entry closes what is held without an exit signal.
        reversing = opening is not None and not self.is_flat and _OPENS[opening] != self._sign

        if closing_by_exit or reversing:
            emitted.append(self._close(close=close, ts_bar_ms=ts_bar_ms))

        if opening is not None and self.is_flat:
            opened = self._open(_OPENS[opening], close=close, ts_bar_ms=ts_bar_ms, scale=scale)
            if opened is not None:
                emitted.append(opened)

        self.fills.extend(emitted)
        return tuple(emitted)

    @property
    def _sign(self) -> int:
        return 1 if self.position > 0 else -1 if self.position < 0 else 0

    def _open(self, sign: int, *, close: float, ts_bar_ms: int, scale: float) -> Fill | None:
        direction = Direction.BUY if sign > 0 else Direction.SELL
        quantity = entry_quantity(
            close=close, cash=self.cash, position_pct=self.position_pct, scale=scale
        )
        if direction is Direction.BUY:
            # Requested off *initial* cash, filled out of the *balance*. See the
            # class docstring: the engine does not refuse an unaffordable entry,
            # it fills what the account can pay for. Floored at zero because a
            # balance can go negative -- a short bought back far above where it
            # was sold -- and a negative "quantity" would settle as a credit and
            # open the *opposite* side of the one the signal asked for.
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
        self._settle(fill)
        self.position = sign * quantity
        self._entry = fill
        return fill

    def _close(self, *, close: float, ts_bar_ms: int) -> Fill:
        quantity = abs(self.position)
        fill = build_fill(
            ts_bar_ms=ts_bar_ms,
            # Closing a long is a sell and closing a short is a buy, so the
            # direction inverts the position rather than following it.
            direction=Direction.SELL if self.position > 0 else Direction.BUY,
            quantity=quantity,
            close=close,
            costs=self.costs,
        )
        self._settle(fill)
        entry = self._entry
        if entry is not None:
            self.trades.append(
                Trade(
                    direction="long" if self.position > 0 else "short",
                    quantity=quantity,
                    entry_ts_ms=entry.ts_bar_ms,
                    entry_price=entry.price,
                    entry_fee=entry.fee,
                    exit_ts_ms=ts_bar_ms,
                    exit_price=fill.price,
                    exit_fee=fill.fee,
                )
            )
        self.position = 0.0
        self._entry = None
        return fill

    def _settle(self, fill: Fill) -> None:
        """Move the cash balance by one fill, the way the engine's ledger does."""
        if fill.direction is Direction.BUY:
            self.balance -= fill.notional + fill.fee
        else:
            self.balance += fill.notional - fill.fee


__all__ = ["PaperBook", "Trade"]
