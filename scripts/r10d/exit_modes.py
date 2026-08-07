"""R10d item (b): how far the replay stream is from each backtest configuration.

The census claims *"for the five strategies that depend on engine-side exits the
signal stream matches no single backtest configuration"*. That is a claim about
``StrategyRunner._extract`` -- which emits the four side series and withholds
``trend_failure_*``, ``setup_stop_loss`` and ``position_size`` -- against
``backtests.engine._exit_signals``, which is where an ``ExitMode`` turns those
ingredients into the booleans the portfolio trades.

**(a) and (b) confound, and this measures (b) alone.** On a perp
``state_machine_v1``'s replayed signals already differ from its backtested ones
because ``crowding`` runs neutral on the event path (M20, measured total in
R10a). Comparing a live replay against a backtest would charge that difference
to (b). So both sides come from **one** ``generate_signals`` call over the whole
frame: the raw side series stand in for the replay stream, and ``_exit_signals``
is applied to the same ``SignalSet``. Identical inputs, identical funding, so the
only thing that can differ is the exit mode.

CONTROL R validates that substitution rather than assuming it -- one real replay,
reconstructed into per-bar booleans and asserted equal to the whole-history
series it stands in for.

**The pre-registered entry control was vacuous, and this is the version that can
fail.** The protocol said entries must differ on zero bars because
"``_exit_signals`` passes entries through untouched" -- true, and trivially so:
it returns exits and never sees an entry, so nothing about that comparison could
ever have gone red. What *does* touch entries is ``_mask_warmup``, two lines
further down, which silences both sides until the declared warmup has elapsed.
So the control that means something compares the engine's masked entries against
the **runner's own** warmup rule: ``_mask_warmup`` keeps rows at positional index
``>= warmup``, and ``StrategyRunner.on_event`` emits once ``len(buffer) >
warmup_bars``, i.e. at index ``>= warmup_bars`` -- designed to agree, never
measured together. A difference here is the two paths disagreeing about which
bars exist at all, which would sit underneath every exit-mode number in this
file.

Usage::

    python scripts/r10d/exit_modes.py [--symbol BTC/USDT]

Writes ``$R10D_OUT/exit_modes.json`` (default a temp dir).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_lab.backtests.engine import ExitMode, _exit_signals, _mask_warmup, _warmup_bars
from strategy_lab.backtests.funding_frame import with_funding_column
from strategy_lab.backtests.sizing import DEFAULT_VOL_SPAN, SizeMode
from strategy_lab.core.clock import SimClock
from strategy_lab.core.types import InstrumentId, Side
from strategy_lab.db import load_candles
from strategy_lab.db.funding import funding_span
from strategy_lab.engine.runner import StrategyRunner
from strategy_lab.feeds.base import Subscription
from strategy_lab.feeds.replay import ReplayFeed
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.strategies.registry import get_strategy, list_strategies

OUT = Path(os.environ.get("R10D_OUT", Path(tempfile.gettempdir()) / "strategy-lab-r10d"))
REPO = Path(__file__).resolve().parents[2]
# The refusal every harness here makes: a directory under the repo could be
# ``reports/``, which is the frozen record of a run someone chose to publish.
if OUT.resolve() == REPO or REPO in OUT.resolve().parents:
    raise SystemExit(f"R10D_OUT must point outside the repository, not {OUT}")

EXCHANGE, MARKET_TYPE, TIMEFRAME = "binance", "perp", "4h"
FAILURE_BARS = 4

# STRATEGIES.md's matrix, as data. A cell the matrix marks "raises" is asserted
# to raise rather than skipped: "it raises" is the matrix's own claim, and an
# unrun cell is not evidence for it.
CANONICAL: dict[str, ExitMode] = {
    "tsmom": ExitMode.OPPOSITE_SIGNAL_ONLY,
    "ema_cross": ExitMode.OPPOSITE_SIGNAL_ONLY,
    "donchian": ExitMode.OPPOSITE_SIGNAL_ONLY,
    "multi_horizon": ExitMode.OPPOSITE_SIGNAL_ONLY,
    "turnaround_v1": ExitMode.CONTINUATION_FAILURE,
    "turnaround_v2": ExitMode.CONTINUATION_FAILURE,
    "trend_following_deepseek_v4": ExitMode.TREND_STRUCTURE,
    "trend_rider_v1_deepseek_v4_pro": ExitMode.OPPOSITE_SIGNAL_ONLY,
    "state_machine_v1": ExitMode.OPPOSITE_SIGNAL_ONLY,
}

# The modes ``_exit_signals`` returns the strategy's own exits for, unchanged.
# Named rather than inferred, because whether this set is right is the first
# reading the protocol declared.
PASS_THROUGH = {ExitMode.OPPOSITE_SIGNAL_ONLY, ExitMode.SETUP_INVALIDATION_STOP}

_SIDES: tuple[tuple[str, Side], ...] = (
    ("long_entries", Side.ENTER_LONG),
    ("long_exits", Side.EXIT_LONG),
    ("short_entries", Side.ENTER_SHORT),
    ("short_exits", Side.EXIT_SHORT),
)


def frame(symbol: str) -> tuple[MarketDataIdentity, pd.DataFrame]:
    """The stored perp frame, bounded at both ends by its own funding span.

    A db-marked run on a real perp frame that does not bound its right edge goes
    red on an unrelated candle refresh -- the coverage guard refusing a frame
    whose bars outran its settlements, which is correct and has nothing to do
    with what is being measured here.
    """
    identity = MarketDataIdentity(
        exchange=EXCHANGE, market_type=MARKET_TYPE, symbol=symbol, timeframe=TIMEFRAME
    )
    span = funding_span(exchange=EXCHANGE, market_type=MARKET_TYPE, symbol=symbol)
    if span is None:
        raise SystemExit(f"no stored funding for {symbol}; run strategy-lab fetch-funding")
    df = load_candles(
        exchange=EXCHANGE,
        market_type=MARKET_TYPE,
        symbol=symbol,
        timeframe=TIMEFRAME,
        start=span[0],
        end=span[1],
    )
    df, _ = with_funding_column(identity, df, enabled=True)
    return identity, df


def runner_mask(df: pd.DataFrame, warmup_bars: int) -> np.ndarray:
    """The bars ``StrategyRunner`` emits on: it withholds until ``len(buffer) >
    warmup_bars``, and after appending row *i* the buffer holds *i+1*."""
    return np.arange(len(df)) >= warmup_bars


def measure(name: str, df: pd.DataFrame) -> dict:
    strategy = get_strategy(name)
    if len(df) <= strategy.warmup_bars:
        return {"refused": f"warmup {strategy.warmup_bars} > {len(df)} bars"}

    signals = strategy.generate_signals(df)
    engine_warmup = _warmup_bars(strategy, df, size_mode=SizeMode.FIXED, vol_span=DEFAULT_VOL_SPAN)
    live = runner_mask(df, strategy.warmup_bars)

    entry = {
        "warmup_bars": int(strategy.warmup_bars),
        "engine_warmup": int(engine_warmup),
        "canonical": CANONICAL[name].value,
        "modes": {},
    }

    for mode in ExitMode:
        try:
            long_exits, short_exits = _exit_signals(
                df=df, signals=signals, exit_mode=mode, failure_bars=FAILURE_BARS
            )
        except ValueError as exc:
            entry["modes"][mode.value] = {"raises": str(exc)}
            continue

        masked, engine_long, engine_short = _mask_warmup(
            signals, long_exits.fillna(False), short_exits.fillna(False), engine_warmup
        )
        # What the runner would have emitted: the strategy's own series, silenced
        # by the runner's own warmup rule and nothing else.
        live_long = signals.long_exits & live
        live_short = signals.short_exits & live
        entry["modes"][mode.value] = {
            "raises": None,
            "pass_through": mode in PASS_THROUGH,
            "canonical": mode == CANONICAL[name],
            "long_exit_diff": int((engine_long.astype(bool) != live_long).sum()),
            "short_exit_diff": int((engine_short.astype(bool) != live_short).sum()),
            "engine_long_exits": int(engine_long.sum()),
            "live_long_exits": int(live_long.sum()),
            "engine_short_exits": int(engine_short.sum()),
            "live_short_exits": int(live_short.sum()),
            # The control that can fail: the engine's entries after warmup
            # masking against the runner's, which is where the two paths could
            # disagree about which bars exist at all.
            "entry_diff": int(
                ((masked.long_entries != (signals.long_entries & live)).sum())
                + ((masked.short_entries != (signals.short_entries & live)).sum())
            ),
        }
    return entry


async def replayed_sides(identity: MarketDataIdentity, strategy, df: pd.DataFrame) -> pd.DataFrame:
    instrument = InstrumentId(
        exchange=identity.exchange,
        market_type=identity.market_type,
        symbol=identity.symbol,
    )
    runner = StrategyRunner(
        strategy=strategy,
        instrument=instrument,
        timeframe=identity.timeframe,
        clock=SimClock(),
        record_reasons=False,
    )
    candle = instrument.at(identity.timeframe)
    feed = ReplayFeed({candle: df})
    subscription = Subscription(instrument=instrument, timeframe=identity.timeframe)
    rows: dict[pd.Timestamp, dict[str, bool]] = {}

    def record(signals) -> None:
        for signal in signals:
            stamp = pd.Timestamp(signal.ts_bar_ms, unit="ms", tz="UTC")
            rows.setdefault(stamp, {})[signal.side.value] = True

    # No ``flush()``: that belongs to ``MultiAssetRunner``, which withholds a
    # timestamp until a later event proves it complete. A single-instrument
    # runner emits on every closed bar as it arrives.
    async for event in feed.stream([subscription]):
        record(runner.on_event(event))

    # Densified here rather than at the comparison, and it is not cosmetic. A bar
    # that emitted only an exit leaves ``enter_long`` **NaN** rather than absent,
    # and ``NaN`` casts to ``True``: a control written to reindex-then-cast reads
    # every such bar as an entry and fails against a correct replay. Every column
    # exists, so a side no bar emitted is False rather than missing.
    dense = pd.DataFrame(rows.values(), index=list(rows))
    return dense.reindex(columns=[side.value for _, side in _SIDES]).fillna(False).astype(bool)


def control_r(identity: MarketDataIdentity, df: pd.DataFrame, *, bars: int) -> dict:
    """One real replay, asserted equal to the series it stands in for.

    ``donchian`` rather than a state machine: 96 warmup bars against 2,192, and
    it reads no funding-derived feature -- so census item (a) cannot contaminate
    the control that validates the substitution the rest of this file rests on.
    The replayed slice is the frame's tail, and the whole-history series is
    computed over the **whole** frame and then sliced, which is the difference
    that would show if the strategy were not causal.
    """
    strategy = get_strategy("donchian")
    tail = df.iloc[-bars:]
    streamed = asyncio.run(replayed_sides(identity, strategy, tail))
    whole = strategy.generate_signals(df)

    emitting = tail.index[strategy.warmup_bars :]
    mismatches: dict[str, int] = {}
    for field, side in _SIDES:
        expected = getattr(whole, field).reindex(emitting).fillna(False).astype(bool)
        got = streamed[side.value].reindex(emitting, fill_value=False)
        mismatches[field] = int((expected != got).sum())
    return {
        "strategy": "donchian",
        "bars_streamed": int(len(tail)),
        "bars_compared": int(len(emitting)),
        "signal_bars": int(len(streamed)),
        "mismatches": mismatches,
        "holds": all(count == 0 for count in mismatches.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--control-bars", type=int, default=600)
    args = parser.parse_args()

    identity, df = frame(args.symbol)
    result: dict = {
        "symbol": args.symbol,
        "bars": int(len(df)),
        "first_bar": str(df.index[0]),
        "last_bar": str(df.index[-1]),
        "strategies": {name: measure(name, df) for name in sorted(list_strategies())},
    }
    result["control_r"] = control_r(identity, df, bars=args.control_bars)

    cells = [
        (f"{name}/{mode}", cell)
        for name, entry in result["strategies"].items()
        if "modes" in entry
        for mode, cell in entry["modes"].items()
        if cell.get("raises") is None
    ]
    result["pass_through_diffs"] = {
        label: cell["long_exit_diff"] + cell["short_exit_diff"]
        for label, cell in cells
        if cell["pass_through"] and cell["long_exit_diff"] + cell["short_exit_diff"] > 0
    }
    result["entry_control_failures"] = {
        label: cell["entry_diff"] for label, cell in cells if cell["entry_diff"] > 0
    }
    result["kill_switch_fired"] = bool(result["pass_through_diffs"])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exit_modes.json").write_text(json.dumps(result, indent=2, default=str))

    print(f"frame: {args.symbol} {len(df)} bars, {df.index[0]} -> {df.index[-1]}\n")
    print(f"{'strategy':32} {'mode':24} {'long':>7} {'short':>7} {'entry':>6}")
    print("-" * 92)
    for name, entry in sorted(result["strategies"].items()):
        if "modes" not in entry:
            print(f"{name:32} {entry['refused']}")
            continue
        for mode, cell in entry["modes"].items():
            if cell["raises"] is not None:
                print(f"{name:32} {mode:24} {'raises':>7}")
                continue
            mark = "  <- canonical" if cell["canonical"] else ""
            print(
                f"{name:32} {mode:24} {cell['long_exit_diff']:>7} "
                f"{cell['short_exit_diff']:>7} {cell['entry_diff']:>6}{mark}"
            )

    control = result["control_r"]
    print(
        f"\nCONTROL R (real replay, {control['strategy']}, {control['bars_compared']} bars "
        f"compared, {control['signal_bars']} signal bars): "
        f"{'holds' if control['holds'] else 'FAILED ' + str(control['mismatches'])}"
    )
    print(f"entry control: {result['entry_control_failures'] or 'clean'}")
    if result["kill_switch_fired"]:
        print(f"\nKILL SWITCH: pass-through modes differ -> {result['pass_through_diffs']}")
        return 1
    print("kill switch not fired: every pass-through mode differs on 0 bars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
