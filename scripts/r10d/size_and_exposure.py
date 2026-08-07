"""R10d items (c) and (d): size on the wire, and ``TargetExposure`` on the event path.

**(c) Size.** ``Signal`` has no size field and ``_extract`` withholds
``position_size``, so the census calls size "engine-side" and a live chart unable
to show how big. The question that decides whether that is a *gap* is whether the
number is **recomputable**: under M35 what can be recomputed is not persisted, and
``position_size`` is returned by ``generate_signals`` over the same buffer the
runner already holds. So this measures the scale at the bars where it is actually
consumed -- entries, and only entries, because ``from_signals`` defaults to
``accumulate=False`` and a position never resizes after it opens (R6) -- and
checks that a cold buffer reproduces it.

**(d) Exposure.** ``StrategyRunner`` calls ``generate_signals``, which an
exposure strategy does not have. The failure is recorded exactly rather than
described, and then the question is how far the path is from working:
``tests/test_exposure_determinism.py`` already streams these strategies through a
driver its own header calls a deliberate mirror of ``StrategyRunner``, on
``synthetic_ohlcv``. **That driver is imported rather than copied** -- a second
copy would measure a different thing -- and pointed at a real perp frame, which
is the frame it has never run on.

**The verdict pair is crowding-neutral on both sides.** ``BarBuffer``
materialises OHLCV and nothing else, so a streamed run is *always* crowding-
neutral; comparing it against a funded whole-history run would charge census item
(a) to (d). So the verdict compares streamed against whole-history over the
**unfunded** frame, and the funded comparison is reported beside it as (a)'s size
on the second contract -- which nothing has measured before.

Usage::

    python scripts/r10d/size_and_exposure.py [--symbol BTC/USDT] [--stream-bars N]

Writes ``$R10D_OUT/size_and_exposure.json`` (default a temp dir).
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_lab.core.clock import SimClock
from strategy_lab.core.types import BarEvent, InstrumentId
from strategy_lab.engine.runner import StrategyRunner
from strategy_lab.feeds.replay import _row_to_bar
from strategy_lab.timeframes import timeframe_to_millis
from strategy_lab.strategies.exposure_registry import (
    get_exposure_strategy,
    list_exposure_strategies,
)
from strategy_lab.strategies.registry import get_strategy, list_strategies

from tests.test_exposure_determinism import streamed_targets, whole_history_targets

OUT = Path(os.environ.get("R10D_OUT", Path(tempfile.gettempdir()) / "strategy-lab-r10d"))
REPO = Path(__file__).resolve().parents[2]
if OUT.resolve() == REPO or REPO in OUT.resolve().parents:
    raise SystemExit(f"R10D_OUT must point outside the repository, not {OUT}")


def size_on_the_wire(df: pd.DataFrame) -> dict:
    """(c): the scale at the bars that consume it, and whether it survives a cold start."""
    out: dict[str, dict] = {}
    for name in sorted(list_strategies()):
        strategy = get_strategy(name)
        if len(df) <= strategy.warmup_bars:
            out[name] = {"refused": f"warmup {strategy.warmup_bars} > {len(df)} bars"}
            continue
        signals = strategy.generate_signals(df)
        scale = signals.position_size
        if scale is None:
            out[name] = {"emits_position_size": False}
            continue

        entries = (signals.long_entries | signals.short_entries).to_numpy()
        at_entry = pd.Series(scale.to_numpy()[entries], index=df.index[entries]).dropna()
        out[name] = {
            "emits_position_size": True,
            "entry_bars": int(entries.sum()),
            "measurable_at_entry": int(len(at_entry)),
            "entries_not_one": int((~np.isclose(at_entry.to_numpy(), 1.0)).sum()),
            "min": float(at_entry.min()) if len(at_entry) else None,
            "median": float(at_entry.median()) if len(at_entry) else None,
            "max": float(at_entry.max()) if len(at_entry) else None,
            "distinct": int(at_entry.round(6).nunique()) if len(at_entry) else 0,
        }
    return out


def size_is_recomputable(df: pd.DataFrame, *, name: str, tail: int) -> dict:
    """The M35 question for (c): can a consumer holding only stored candles get it back?

    A cold buffer over the last ``tail`` bars, against the whole-history value on
    the same bars. Equal means the live path *loses* nothing by not sending it --
    the number is a function of bars the consumer already has -- so (c) is a
    reporting choice rather than a data-loss gap.
    """
    strategy = get_strategy(name)
    signals = strategy.generate_signals(df)
    if signals.position_size is None:
        return {"strategy": name, "emits_position_size": False}

    window = df.iloc[-(tail + strategy.warmup_bars) :]
    cold = strategy.generate_signals(window).position_size
    compared = df.index[-tail:]
    whole = signals.position_size.reindex(compared)
    got = cold.reindex(compared)
    both = whole.notna() & got.notna()
    return {
        "strategy": name,
        "emits_position_size": True,
        "bars_compared": int(both.sum()),
        "differing": int((~np.isclose(whole[both].to_numpy(), got[both].to_numpy())).sum()),
        "nan_disagreements": int((whole.notna() != got.notna()).sum()),
    }


def runner_refusal(identity, df: pd.DataFrame) -> dict:
    """(d), first: *where* the event path fails when handed an exposure strategy.

    Recorded as the exception and its position rather than described, because
    "cannot run at all" is the census's wording and *when* it fails is the finding.
    ``require_warmup_bars`` sits in ``__init__`` precisely so a live process
    refuses to start rather than partway through a session; whether the contract
    mismatch gets the same treatment is what this asks, so construction and the
    first bar are probed separately.
    """
    instrument = InstrumentId(
        exchange=identity.exchange, market_type=identity.market_type, symbol=identity.symbol
    )
    bar_ms = timeframe_to_millis(identity.timeframe)

    out: dict[str, dict] = {}
    for name in sorted(list_exposure_strategies()):
        strategy = get_exposure_strategy(name)
        cell: dict = {
            "has_generate_signals": hasattr(strategy, "generate_signals"),
            "warmup_bars": int(strategy.warmup_bars),
        }
        try:
            runner = StrategyRunner(
                strategy=strategy,
                instrument=instrument,
                timeframe=identity.timeframe,
                clock=SimClock(),
                record_reasons=False,
            )
            cell["constructs"] = True
        except Exception as exc:  # noqa: BLE001 -- the type is the measurement
            out[name] = {**cell, "constructs": False, "fails_on_bar": 0,
                         "error": f"{type(exc).__name__}: {exc}"}
            continue

        # Fed until it raises rather than probed once, because *when* is the
        # measurement: ``on_bar`` returns before touching the strategy while the
        # buffer is inside warmup, so a single-bar probe reports a clean start
        # for a runner that will die a warmup later.
        probe = df.iloc[: strategy.warmup_bars + 2]
        out[name] = {**cell, "fails_on_bar": None, "error": None}
        for index, (timestamp, row) in enumerate(probe.iterrows()):
            bar = _row_to_bar(timestamp, row, instrument, identity.timeframe, bar_ms)
            try:
                runner.on_event(BarEvent(bar=bar, ts_event_ms=bar.ts_close_ms))
            except Exception as exc:  # noqa: BLE001
                out[name] = {
                    **cell,
                    "fails_on_bar": index,
                    "bars_survived": index,
                    "days_survived_at_timeframe": round(index * bar_ms / 86_400_000, 1),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                break
    return out


def exposure_streams(df_unfunded: pd.DataFrame, df_funded: pd.DataFrame, *, bars: int) -> dict:
    """(d), second: does the proven driver reproduce whole history on a *real* frame?

    The verdict pair is unfunded on both sides. The funded row beside it is census
    item (a) on the continuous contract, which nothing has measured.
    """
    out: dict[str, dict] = {}
    for name in sorted(list_exposure_strategies()):
        strategy = get_exposure_strategy(name)
        need = strategy.warmup_bars + bars
        if len(df_unfunded) < need:
            out[name] = {"refused": f"needs {need} bars, frame has {len(df_unfunded)}"}
            continue
        window_unfunded = df_unfunded.iloc[-need:]
        window_funded = df_funded.iloc[-need:]

        streamed = streamed_targets(strategy, window_unfunded)
        neutral = whole_history_targets(strategy, window_unfunded)
        funded = whole_history_targets(strategy, window_funded)

        aligned = streamed.reindex(neutral.index)
        out[name] = {
            "warmup_bars": int(strategy.warmup_bars),
            "bars_compared": int(len(neutral)),
            # The verdict: both sides crowding-neutral, so only a streaming
            # defect can move it.
            "streamed_vs_neutral_differing": int(
                (~np.isclose(aligned.to_numpy(), neutral.to_numpy(), equal_nan=True)).sum()
            ),
            # Context, not verdict: census item (a) on this contract.
            "neutral_vs_funded_differing": int(
                (~np.isclose(neutral.to_numpy(), funded.to_numpy(), equal_nan=True)).sum()
            ),
            "target_moves": int(neutral.round(9).nunique()),
            "neutral_mean_abs": float(np.abs(neutral.to_numpy()).mean()),
            "funded_mean_abs": float(np.abs(funded.to_numpy()).mean()),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--stream-bars", type=int, default=400)
    parser.add_argument("--recompute-tail", type=int, default=600)
    args = parser.parse_args()

    from exit_modes import frame  # same funding-bounded frame item (b) used

    identity, funded = frame(args.symbol)
    unfunded = funded.drop(columns=["funding_rate"], errors="ignore")

    result = {
        "symbol": args.symbol,
        "bars": int(len(funded)),
        "c_size": size_on_the_wire(funded),
        "c_recomputable": [
            size_is_recomputable(funded, name=name, tail=args.recompute_tail)
            for name in sorted(list_strategies())
            if get_strategy(name).warmup_bars + args.recompute_tail < len(funded)
        ],
        "d_runner": runner_refusal(identity, funded),
        "d_streams": exposure_streams(unfunded, funded, bars=args.stream_bars),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "size_and_exposure.json").write_text(json.dumps(result, indent=2, default=str))

    print(f"frame: {args.symbol} {len(funded)} bars\n")
    print("(c) position_size at entry bars")
    print(f"  {'strategy':32} {'emits':>6} {'entries':>8} {'!=1.0':>7} {'min':>8} {'max':>8}")
    for name, cell in result["c_size"].items():
        if "refused" in cell:
            print(f"  {name:32} {cell['refused']}")
        elif not cell["emits_position_size"]:
            print(f"  {name:32} {'no':>6}")
        else:
            print(
                f"  {name:32} {'yes':>6} {cell['entry_bars']:>8} {cell['entries_not_one']:>7} "
                f"{cell['min']:>8.3f} {cell['max']:>8.3f}"
            )
    print("\n(c) recomputable from a cold buffer")
    for cell in result["c_recomputable"]:
        if not cell["emits_position_size"]:
            continue
        print(
            f"  {cell['strategy']:32} {cell['bars_compared']:>6} bars, "
            f"{cell['differing']} differing, {cell['nan_disagreements']} NaN disagreements"
        )

    print("\n(d) StrategyRunner with an exposure strategy")
    for name, cell in result["d_runner"].items():
        if cell.get("fails_on_bar") is None:
            print(f"  {name:32} survives {cell['warmup_bars'] + 2} bars without raising")
            continue
        print(
            f"  {name:32} constructs, survives {cell['bars_survived']} bars "
            f"({cell['days_survived_at_timeframe']} days), then:"
        )
        print(f"  {'':32}   {cell['error']}")
    print("\n(d) streamed vs whole history, on a real frame")
    for name, cell in result["d_streams"].items():
        if "refused" in cell:
            print(f"  {name:32} {cell['refused']}")
            continue
        print(
            f"  {name:32} {cell['bars_compared']:>5} bars  "
            f"streamed vs neutral: {cell['streamed_vs_neutral_differing']:>5}   "
            f"neutral vs funded: {cell['neutral_vs_funded_differing']:>5}   "
            f"distinct targets: {cell['target_moves']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
