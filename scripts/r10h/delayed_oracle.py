"""R10h's gate: a live window against the record the venue serves for it later.

**A live feed has no oracle at the moment it runs.** That is why R10g built the
book first -- ``trades.csv`` is frozen and a book either reproduces it or does
not -- and it is why this phase exists at all. A poll produces a bar nothing can
be diffed against while it is arriving.

It has a *delayed* one, and that is the gate:

    Run the feed over a window. Later, fetch the same range historically and
    replay it. The bars must be identical, and so must the signals.

A live bar and a stored bar for the same interval are the same fact recorded
twice. Anything the live path does differently -- a partial candle taken as
final, a duplicate after a reconnect, a timestamp off by a millisecond, funding
attached from a window too narrow to carry it -- shows up as a difference against
what the venue itself will serve tomorrow.

**Read from the database, not from memory**, as R10a's diff is: the comparison is
against what ``load_signals`` and ``load_bar_reasons`` return, so the
``Decimal(str(float(x))) -> NUMERIC(38,18) -> float`` round-trip is inside the
thing being tested rather than beside it.

**The paper process writes no candles to ``market_candles``.** The stored bars it
is checked against come from a separate ``fetch-perp`` after the fact, which is
what makes this an oracle instead of an echo. Its ``--bars-csv`` log is the other
half and not a contradiction: the live bars must be written down somewhere or
there is nothing to hold the later fetch against, and a file beside the run is
not the record of a dataset.

**The four readings, declared in the plan before this ran**
(docs/plans/2026-08-08-r10h-running-the-paper-process.md):

1. Everything matches -> the gate passes, and R10's delayed oracle is performed
   rather than simulated.
2. Bars differ -> either the feed's closed-bar rule is wrong or the venue revised
   what it served. Section [1] distinguishes them: a revision moves OHLCV under a
   matching timestamp, a closed-bar error moves the timestamps themselves.
3. Signals differ while bars agree -> the two drivers disagree on identical
   input, which ``tests/test_replay_determinism.py`` says cannot happen, so it
   would be a gap in that suite found the way R10f's was.
4. **Nothing to compare** -> zero live bars, or a window so short the strategy
   never emitted. This is a **failed run, not a pass**, and it is called out
   because an empty diff prints the same "0 differences" as a perfect one. Every
   section prints its own count first, and a comparison with nothing in it exits
   non-zero rather than reporting agreement.

Usage::

    python scripts/r10h/delayed_oracle.py <paper-run-id> [live-bars.csv]
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from strategy_lab.core.clock import SimClock
from strategy_lab.core.types import InstrumentId
from strategy_lab.db.candles import get_engine, load_candles
from strategy_lab.engine.runner import StrategyRunner
from strategy_lab.feeds.base import Subscription
from strategy_lab.feeds.replay import ReplayFeed
from strategy_lab.storage.bar_reasons import load_bar_reasons
from strategy_lab.storage.schema import runs_table
from strategy_lab.storage.signals import load_signals
from strategy_lab.strategies.registry import get_strategy
from strategy_lab.timeframes import timeframe_to_millis

FEATURES = ("energy", "direction", "strength", "persistence", "crowding")
OHLCV = ("open", "high", "low", "close", "volume")

# Prices round-trip through NUMERIC(38,18) on one side and a csv on the other, so
# equality is to a tolerance rather than to the bit. Well below a tick on any
# instrument here, and far above either representation's error.
PRICE_TOLERANCE = 1e-9


def _run_header(run_id: uuid.UUID) -> dict:
    with get_engine().connect() as conn:
        row = (
            conn.execute(select(runs_table).where(runs_table.c.run_id == run_id))
            .mappings()
            .first()
        )
    if row is None:
        raise SystemExit(f"no run {run_id}")
    return dict(row)


def _replay(header: dict, window_start_ms: int, window_end_ms: int):
    """Drive the same strategy over the same range, from stored candles.

    Reaching back a full warmup before the window and filtering afterwards, for
    the reason the paper process primed: a replay starting at the window's left
    edge emits nothing until its warmup is behind it, so the comparison would be
    against a runner that had not woken up yet.
    """
    config = header["config"]
    strategy = get_strategy(header["strategy_id"])
    instrument = InstrumentId(config["exchange"], config["market_type"], config["symbol"])
    subscription = Subscription(instrument, config["timeframe"])
    bar_ms = timeframe_to_millis(config["timeframe"])
    start = pd.Timestamp(
        window_start_ms - bar_ms * strategy.warmup_bars, unit="ms", tz="UTC"
    )
    end = pd.Timestamp(window_end_ms, unit="ms", tz="UTC")

    feed = ReplayFeed.from_database(
        [subscription],
        start=str(start),
        end=str(end),
        funding=True,
        required="crowding" in getattr(strategy, "features", ()),
    )
    runner = StrategyRunner(
        strategy=strategy,
        instrument=instrument,
        timeframe=config["timeframe"],
        clock=SimClock(),
        exit_mode=config.get("exit_mode"),
    )

    async def _drive():
        emitted = []
        async for event in feed.stream([subscription]):
            emitted.extend(runner.on_event(event))
        return emitted

    signals = asyncio.run(_drive())
    window = range(window_start_ms, window_end_ms + 1)
    return (
        [s for s in signals if s.ts_bar_ms in window],
        [r for r in runner.reasons if r.ts_bar_ms in window],
    )


def _compare_bars(header: dict, bars_csv: Path) -> tuple[int, dict[str, int]]:
    """Every logged live bar against the stored candle for the same interval."""
    config = header["config"]
    stored = load_candles(
        exchange=config["exchange"],
        market_type=config["market_type"],
        symbol=config["symbol"],
        timeframe=config["timeframe"],
    )
    by_ts = {int(ts.value // 10**6): row for ts, row in stored.iterrows()}

    live = pd.read_csv(bars_csv)
    live = live[live["is_closed"]]  # a forming bar has no stored counterpart yet
    counts: dict[str, int] = defaultdict(int)
    compared = 0
    for _, row in live.iterrows():
        stored_row = by_ts.get(int(row["ts_open_ms"]))
        if stored_row is None:
            counts["absent from storage"] += 1
            continue
        compared += 1
        for name in OHLCV:
            if abs(float(row[name]) - float(stored_row[name])) > PRICE_TOLERANCE:
                counts[name] += 1
    return compared, dict(counts)


def _compare_signals(live, replayed) -> tuple[int, int]:
    live_keys = sorted((s.ts_bar_ms, str(s.side)) for s in live)
    replay_keys = sorted((s.ts_bar_ms, str(s.side)) for s in replayed)
    return len(live_keys), len(set(live_keys) ^ set(replay_keys))


def _compare_reasons(live, replayed) -> tuple[int, dict[str, int]]:
    by_ts = {r.ts_bar_ms: r for r in replayed}
    counts: dict[str, int] = defaultdict(int)
    compared = 0
    for reason in live:
        other = by_ts.get(reason.ts_bar_ms)
        if other is None:
            counts["absent from replay"] += 1
            continue
        compared += 1
        if reason.state != other.state:
            counts["state"] += 1
        for name in FEATURES:
            mine = (reason.features or {}).get(name)
            theirs = (other.features or {}).get(name)
            if mine is None and theirs is None:
                continue
            if mine is None or theirs is None or abs(float(mine) - float(theirs)) > 1e-9:
                counts[name] += 1
    return compared, dict(counts)


def main() -> int:
    if not 2 <= len(sys.argv) <= 3:
        raise SystemExit("usage: delayed_oracle.py <paper-run-id> [live-bars.csv]")
    run_id = uuid.UUID(sys.argv[1])
    bars_csv = Path(sys.argv[2]) if len(sys.argv) == 3 else None

    header = _run_header(run_id)
    config = header["config"]
    print(
        f"paper run {run_id}: {config['symbol']} {config['timeframe']} "
        f"{header['strategy_id']} v{header['strategy_version']}"
    )

    live_signals = load_signals(run_id=run_id)
    live_reasons = load_bar_reasons(run_id=run_id)
    print(f"  live: {len(live_signals)} signals, {len(live_reasons)} reason rows")
    if not live_reasons:
        print(
            "\nREADING 4: nothing to compare. The run recorded no bars, so every "
            "diff below would read 0 differences for the wrong reason."
        )
        return 2

    start_ms, end_ms = live_reasons[0].ts_bar_ms, live_reasons[-1].ts_bar_ms
    print(
        f"  live window: {pd.Timestamp(start_ms, unit='ms', tz='UTC')} -> "
        f"{pd.Timestamp(end_ms, unit='ms', tz='UTC')}"
    )

    failed = False
    exercised = {"bars", "signals", "reasons"}

    print("\n[1] live bars against the stored candles fetched afterwards")
    if bars_csv is None:
        print("  NOT PERFORMED: no --bars-csv log from the run.")
        print("  Reading 2 is undecidable without it: a venue revision and a")
        print("  closed-bar error both surface only as a derived difference.")
        exercised.remove("bars")
    else:
        compared, counts = _compare_bars(header, bars_csv)
        print(f"  {compared} closed bars compared")
        for name, count in sorted(counts.items()):
            print(f"    {name}: {count} differing")
        if not counts:
            print("    0 differing on every OHLCV field")
        failed = failed or bool(counts) or compared == 0

    print("\n[2] replaying the same range from storage")
    replay_signals, replay_reasons = _replay(header, start_ms, end_ms)
    print(f"  replay: {len(replay_signals)} signals, {len(replay_reasons)} reason rows")

    print("\n[3] signals")
    count, differing = _compare_signals(live_signals, replay_signals)
    print(f"  {count} live signals, {differing} differing")
    if not count and not replay_signals:
        # Reading 4 applied to one section rather than to the run. Over a window
        # this short a real strategy emits nothing -- `state_machine_v1` fired
        # 325 times in 6,048 bars -- so "0 differing" here is the absence of a
        # comparison, not the result of one, and section [4] is what carries the
        # gate: it compares the features and the state that *produce* signals, on
        # every bar rather than on the rare ones that acted.
        print("  NOT EXERCISED: neither path emitted over this window.")
        exercised.remove("signals")
    failed = failed or bool(differing)

    print("\n[4] per-bar reasons")
    compared, counts = _compare_reasons(live_reasons, replay_reasons)
    print(f"  {compared} bars compared")
    for name in ("state", *FEATURES, "absent from replay"):
        if name in counts:
            print(f"    {name}: {counts[name]} differing")
    if not counts:
        print("    0 differing on the state and on every feature")
    failed = failed or bool(counts) or compared == 0

    if failed:
        print("\nA reading other than 1 applies -- see the counts above.")
        return 1
    if exercised != {"bars", "signals", "reasons"}:
        skipped = ", ".join(sorted({"bars", "signals", "reasons"} - exercised))
        print(
            f"\nREADING 1 on what ran, but NOT on everything: {skipped} was not "
            f"exercised. Nothing below disagreed; that is a weaker claim than the "
            f"gate, and reporting it as the gate is exactly what reading 4 warns "
            f"about."
        )
        return 3
    print("\nREADING 1: the gate passes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
