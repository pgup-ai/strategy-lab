"""R10a's gate: stored-live reasons against the browser's recomputed why-layer.

R10's roadmap gate is *"research and live code paths produce identical output"*.
This makes it checkable **per bar** rather than per signal: one replay of a stored
perp range writes ``bar_reasons``, ``api.analysis.build_analysis`` recomputes the
same range through the research path, and the two are compared bar by bar, per
feature and for the state.

**Read from the database, not from memory.** The comparison is against what
``load_bar_reasons`` returns, so the ``Decimal(str(float(x))) -> NUMERIC(38,18)
-> float`` round-trip is inside the thing being tested. An in-memory diff would
pass over a storage layer that silently rounded.

**Three readings, declared in the plan before this ran** (docs/plans/
2026-08-06-r10a-why-layer.md):

1. ``crowding`` differs, the other four agree, and the state differs only where
   crowding moved a transition -> census item (a) is the whole of the gap for
   this strategy.
2. Anything else differs -> an unenumerated gap, found here rather than in
   production, and that is the phase's real result.
3. Nothing differs -> **suspect the harness**. The determinism suite is already
   blind to (a) by comparing crowding-neutral against crowding-neutral on
   synthetic frames that carry no funding (census item (e)); a green diff for the
   same reason is the same failure in a different hat.

Reading 1's second clause is checked directly rather than argued: CONTROL C
re-runs the research path's own machine over the research path's own features
with **crowding replaced by the neutral constant the live path used**, and asks
whether that reproduces the stored states exactly. If it does, crowding is the
only cause of every state difference -- which is a stronger claim than counting
transitions and squinting at them.

Usage::

    python scripts/r10a/diff_why_layer.py [--bars N]

Writes ``$R10A_OUT/diff_why_layer.json`` (default a temp dir).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

from strategy_lab.api.analysis import build_analysis, prepare_frame
from strategy_lab.backtests import ExitMode
from strategy_lab.core.clock import SimClock
from strategy_lab.core.types import InstrumentId, Mode
from strategy_lab.db import load_candles
from strategy_lab.db.funding import funding_span
from strategy_lab.engine.runner import StrategyRunner
from strategy_lab.feeds.base import Subscription
from strategy_lab.feeds.replay import ReplayFeed
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.storage.bar_reasons import load_bar_reasons, write_bar_reasons
from strategy_lab.storage.migrations import run_migrations
from strategy_lab.storage.signals import create_run
from strategy_lab.strategies.registry import get_strategy
from strategy_lab.strategies.state_machine_core import NEUTRAL_CROWDING

OUT = Path(os.environ.get("R10A_OUT", Path(tempfile.gettempdir()) / "strategy-lab-r10a"))
REPO = Path(__file__).resolve().parents[2]
# The same refusal ``scripts/r7b/r7blib.py`` makes: a directory under the repo
# could be ``reports/``, which is the frozen record of a run someone chose to
# publish. This is a diff, not a run of record.
if OUT.resolve() == REPO or REPO in OUT.resolve().parents:
    raise SystemExit(f"R10A_OUT must point outside the repository, not {OUT}")

EXCHANGE, MARKET_TYPE, SYMBOL, TIMEFRAME = "binance", "perp", "BTC/USDT", "4h"
STRATEGY = "state_machine_v1"
IDENTITY = MarketDataIdentity(
    exchange=EXCHANGE, market_type=MARKET_TYPE, symbol=SYMBOL, timeframe=TIMEFRAME
)
INSTRUMENT = InstrumentId(EXCHANGE, MARKET_TYPE, SYMBOL)
DEFAULT_BARS = 3192  # state_machine_v1's 2,192-bar warmup plus 1,000 measured bars


def mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def window(bars: int) -> tuple[str, str, int]:
    """A start/end pair both paths resolve through the same ``load_candles``.

    The right edge is bounded by the last stored funding settlement, per the
    coverage rule in CLAUDE.md: ``refresh_candles`` advances candles past the
    venue's last settlement, and ``funding_coverage_gaps`` then refuses the whole
    frame -- so a window whose right edge is "the newest candle" is one candle
    fetch away from failing for a reason that has nothing to do with this diff.
    """
    df = load_candles(
        exchange=EXCHANGE, market_type=MARKET_TYPE, symbol=SYMBOL, timeframe=TIMEFRAME
    )
    span = funding_span(exchange=EXCHANGE, market_type=MARKET_TYPE, symbol=SYMBOL)
    if span is None:
        raise SystemExit("no stored funding for this contract; the diff needs a funded perp")
    covered = df[df.index <= span[1]]
    frame = covered.iloc[-bars:]
    return str(frame.index[0]), str(frame.index[-1]), len(frame)


def replay(start: str, end: str) -> tuple[uuid.UUID, int, float]:
    """One real event-path run over the window, persisted like any other replay."""
    strategy = get_strategy(STRATEGY)
    subscription = Subscription(INSTRUMENT, TIMEFRAME)
    feed = ReplayFeed.from_database([subscription], start=start, end=end)
    runner = StrategyRunner(
        strategy=strategy, instrument=INSTRUMENT, timeframe=TIMEFRAME, clock=SimClock()
    )

    async def _run() -> list:
        collected = []
        async for event in feed.stream([subscription]):
            collected.extend(runner.on_event(event))
        return collected

    started = time.time()
    signals = asyncio.run(_run())
    elapsed = time.time() - started

    run_id = create_run(
        run_id=uuid.uuid4(),
        mode=Mode.REPLAY,
        strategy_id=strategy.name,
        strategy_version=strategy.version,
        config={
            "exchange": EXCHANGE,
            "market_type": MARKET_TYPE,
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "start": start,
            "end": end,
            "limit_bars": None,
            "warmup_bars": strategy.warmup_bars,
            "purpose": "R10a why-layer diff",
        },
    )
    written = write_bar_reasons(run_id, Mode.REPLAY, runner.reasons)
    if written != len(runner.reasons):
        raise SystemExit(f"wrote {written} of {len(runner.reasons)} reasons")
    return run_id, len(signals), elapsed


def compare(stored: list[float | None], recomputed: list[float | None]) -> dict:
    """Bar-by-bar equality for one feature, plus how far apart the two ever got.

    ``None`` is "not yet measurable" on both sides and two of them agree. A
    ``None`` opposite a number does not, and is reported separately from a
    numeric disagreement because it is a different failure: one path could not
    measure a bar the other could.
    """
    differing, presence, deltas = 0, 0, []
    for left, right in zip(stored, recomputed, strict=True):
        if left is None and right is None:
            continue
        if left is None or right is None:
            differing += 1
            presence += 1
            continue
        if left != right:
            differing += 1
            deltas.append(abs(left - right))
    return {
        "bars": len(stored),
        "differing": differing,
        "presence_mismatches": presence,
        "max_abs_delta": max(deltas) if deltas else 0.0,
        "mean_abs_delta": float(np.mean(deltas)) if deltas else 0.0,
    }


def declared_reading(
    per_feature: dict, features: list[str], state_differing: int, crowding_explains_state: bool
) -> str:
    """Which of the plan's three rows this run landed in.

    Written as the plan wrote them, so the result cannot be read backwards. Note
    that row 1 needs the state clause too: ``crowding`` differing while some
    *other* cause moved a transition would satisfy the feature half of row 1 and
    still be row 2, which is the case worth catching.
    """
    crowding_moved = per_feature["crowding"]["differing"] > 0
    others_agree = all(
        per_feature[name]["differing"] == 0 for name in features if name != "crowding"
    )
    if crowding_moved and others_agree and crowding_explains_state:
        return "1 -- (a) is the whole of the live/research gap for this strategy"
    if not crowding_moved and state_differing == 0:
        return "3 -- NOTHING DIFFERS: suspect the harness before celebrating"
    return "2 -- AN UNENUMERATED GAP: something other than crowding moved"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", type=int, default=DEFAULT_BARS)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    run_migrations()
    strategy = get_strategy(STRATEGY)

    # A window at or below warmup produces no reason rows, and the per-feature
    # shares below then divide by zero. `--bars 0` is worse than useless rather
    # than merely empty: `iloc[-0:]` is `iloc[0:]`, so it silently replays the
    # whole history instead of nothing.
    if args.bars <= strategy.warmup_bars:
        raise SystemExit(
            f"--bars must exceed the strategy's warmup of {strategy.warmup_bars}; "
            f"got {args.bars}, which would compare zero bars"
        )
    start, end, frame_bars = window(args.bars)
    if frame_bars <= strategy.warmup_bars:
        raise SystemExit(
            f"the stored window resolved to {frame_bars} bars, at or below the "
            f"warmup of {strategy.warmup_bars}; nothing would be compared"
        )
    print(f"{STRATEGY} v{strategy.version} on {EXCHANGE}/{MARKET_TYPE}/{SYMBOL}/{TIMEFRAME}")
    print(f"window: {start} -> {end}  ({frame_bars} bars, warmup {strategy.warmup_bars})\n")

    print("replaying (event path, writes bar_reasons) ...")
    run_id, signal_count, elapsed = replay(start, end)
    rows = load_bar_reasons(run_id=run_id)
    print(f"  run {run_id}: {len(rows)} reason rows, {signal_count} signals, {elapsed:.1f}s")

    print("recomputing (research path, writes nothing) ...")
    started = time.time()
    payload = build_analysis(
        IDENTITY,
        strategy_name=STRATEGY,
        exit_mode=ExitMode.OPPOSITE_SIGNAL_ONLY,
        start=start,
        end=end,
    )
    # The same frame the browser handed the strategy, reloaded by the same rule,
    # for the bar timestamps and for CONTROL C. Reading them off the payload's
    # bars would go through `int(ts.timestamp())`, which is exact at 4h and is
    # still a second copy of the index rather than the index.
    prepared = prepare_frame(IDENTITY, strategy=strategy, start=start, end=end)
    print(f"  {len(payload.bars)} bars, why-layer over {len(payload.why.states)} rows, "
          f"{time.time() - started:.1f}s")

    # harness
    # Reading 3's trap, checked before anything is compared: if the research side
    # ran crowding-neutral too, this diff proves neutral == neutral and nothing
    # else -- exactly the blindness census item (e) records in the determinism
    # suite. A green diff would then be worthless and must not be reported.
    crowding_measured = payload.provenance.crowding_measured
    print(f"\nHARNESS -- the research path measured crowding: {crowding_measured} "
          f"(funding attached: {payload.provenance.funding_attached})")
    if not crowding_measured:
        raise SystemExit(
            "the research path ran crowding-neutral, so this diff compares neutral "
            "to neutral and can only be green for the wrong reason -- stop"
        )

    # Row `warmup + i` of the recomputed layer is the bar reason row `i` covers:
    # the runner suppresses everything up to and including its warmup, and both
    # paths resolved the window through the same `load_candles` call.
    offset = frame_bars - len(rows)
    ts = [stamp.value // 1_000_000 for stamp in prepared.df.index]
    aligned = [reason.ts_bar_ms for reason in rows] == ts[offset:]
    print(f"ALIGNMENT -- {len(rows)} stored rows against rows {offset}.. of "
          f"{frame_bars}: {mark(aligned)}")
    if not aligned:
        raise SystemExit("the two paths did not see the same bars; nothing below is comparable")

    # diff
    features = sorted(strategy.features)
    per_feature = {
        name: compare(
            [reason.features[name] for reason in rows],
            payload.why.features[name][offset:],
        )
        for name in features
    }
    stored_states = [reason.state for reason in rows]
    research_states = payload.why.states[offset:]
    state_differing = sum(a != b for a, b in zip(stored_states, research_states, strict=True))

    print(f"\n{'feature':<12} {'bars':>7} {'differing':>10} {'%':>7} "
          f"{'max |delta|':>13} {'mean |delta|':>13}")
    for name, result in per_feature.items():
        share = 100.0 * result["differing"] / result["bars"]
        print(f"{name:<12} {result['bars']:>7} {result['differing']:>10} {share:>6.1f}% "
              f"{result['max_abs_delta']:>13.6f} {result['mean_abs_delta']:>13.6f}")
    print(f"{'state':<12} {len(rows):>7} {state_differing:>10} "
          f"{100.0 * state_differing / len(rows):>6.1f}%")

    print("\nstate distribution (stored live / research recomputed)")
    labels = sorted(set(stored_states) | set(research_states))
    for label in labels:
        print(f"  {label:<12} {stored_states.count(label):>6} / "
              f"{research_states.count(label):>6}")

    # controls
    print("\nCONTROL A -- the live path really ran crowding at the neutral constant")
    stored_crowding = {reason.features["crowding"] for reason in rows}
    control_a = stored_crowding == {NEUTRAL_CROWDING}
    print(f"  stored crowding values: {sorted(stored_crowding)[:4]}"
          f"{' ...' if len(stored_crowding) > 4 else ''}  ->  {mark(control_a)}")

    print("CONTROL B -- the research path really measured something else")
    research_crowding = payload.why.features["crowding"][offset:]
    moved = sum(1 for value in research_crowding if value != NEUTRAL_CROWDING)
    control_b = moved > 0
    print(f"  {moved} of {len(rows)} bars away from {NEUTRAL_CROWDING}  ->  {mark(control_b)}")

    print("CONTROL C -- crowding is the *only* cause of the state differences")
    print("  (research features, research machine, crowding pinned to the live value)")
    frame, _ = strategy.feature_frame(prepared.df)
    frame["crowding"] = NEUTRAL_CROWDING
    neutralised = [state.value for state in strategy.machine.run(frame)][offset:]
    residual = sum(a != b for a, b in zip(neutralised, stored_states, strict=True))
    control_c = residual == 0
    print(f"  {residual} of {len(rows)} states still differ  ->  {mark(control_c)}")

    reading = declared_reading(per_feature, features, state_differing, control_c)
    print(f"\nDECLARED READING {reading}")

    payload_out = {
        "window": {"start": start, "end": end, "bars": frame_bars, "offset": offset},
        "strategy": {"name": STRATEGY, "version": strategy.version,
                     "warmup_bars": strategy.warmup_bars, "features": features},
        "run_id": str(run_id),
        "replay": {"reason_rows": len(rows), "signals": signal_count, "seconds": elapsed},
        "research": {"crowding_measured": crowding_measured,
                     "funding_attached": payload.provenance.funding_attached},
        "per_feature": per_feature,
        "state": {
            "differing": state_differing,
            "bars": len(rows),
            "stored_distribution": {k: stored_states.count(k) for k in labels},
            "research_distribution": {k: research_states.count(k) for k in labels},
        },
        "controls": {"a_live_neutral": control_a, "b_research_measured": control_b,
                     "c_crowding_explains_state": control_c, "c_residual_states": residual},
        "reading": reading,
    }
    path = OUT / "diff_why_layer.json"
    path.write_text(json.dumps(payload_out, indent=2, default=str))
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
