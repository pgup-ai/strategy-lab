"""R9 part 5 and two loose ends.

(a) **Purging and embargo.** The plan states rather than performs them, on the
    grounds that ``state_machine_v1`` fits nothing, has no labels, and that its
    one boundary effect -- machine state carried across a split -- was measured
    by R5's frame-start invariance check. Both halves of that are re-checked
    here against the code as it stands: the strategy is enrolled in the
    causality suite, and the whole-history and test-window runs are compared
    entry timestamp by entry timestamp.

(b) The ``exit_strength`` perturbation reported identical headline figures to
    six decimals. Identical numbers are either the same book or a harness
    error, so the two equity curves are compared bar by bar.

(c) The deflated Sharpe at bar frequency inherits a kurtosis of ~110, because
    the machine is out of the market on most bars and those returns are exactly
    zero. The same statistic on daily-aggregated returns says whether that is
    what drives the answer.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import r9lib as R
from step2_dsr import deflated_sharpe, expected_max_sharpe

OUT = R.OUT
OUT.mkdir(parents=True, exist_ok=True)


def entries(report: Path) -> pd.Series:
    trades = pd.read_csv(report / "trades.csv")
    return pd.to_datetime(trades["Entry Timestamp"], utc=True)


def main() -> None:
    frame = R.load_frame()
    df, _ = frame
    split = int(R.gate.TRAIN_FRACTION * len(df))
    strategy = R.gate.machine(R.gate.TRAINED_MACHINE)

    from strategy_lab.strategies.registry import list_strategies

    registered = list(list_strategies())
    print(f"(a) state_machine_v1 in list_strategies(): "
          f"{'state_machine_v1' in registered} (of {len(registered)}) "
          f"-- so tests/test_lookahead.py and test_replay_determinism.py both cover it")

    whole = R.run(strategy, frame, OUT / "inv" / "whole",
                  first_tradeable=strategy.warmup_bars, stop=len(df), keep=True)
    R.run(strategy, frame, OUT / "inv" / "window",
          first_tradeable=split, stop=len(df), keep=True)
    whole_report = next((OUT / "inv" / "whole" / strategy.name).glob("*Z_*"))
    window_report = next((OUT / "inv" / "window" / strategy.name).glob("*Z_*"))

    split_at = df.index[split]
    whole_entries = entries(whole_report)
    whole_entries = whole_entries[whole_entries >= split_at].reset_index(drop=True)
    window_entries = entries(window_report).reset_index(drop=True)
    same = len(whole_entries) == len(window_entries) and bool((whole_entries == window_entries).all())
    print(f"(a) frame-start invariance: whole-history run has {len(whole_entries)} entries at or "
          f"after {split_at}, test-window run has {len(window_entries)}; identical: {same}")
    print(f"    whole-history run over the test half: {whole['trades']} trades total over "
          f"{whole['tradeable_bars']} tradeable bars")

    # (b)
    alternative = R.gate.machine(replace(R.gate.TRAINED_MACHINE, exit_strength=0.20))
    left = R.run(strategy, frame, OUT / "same" / "winner", first_tradeable=split, stop=len(df))
    right = R.run(alternative, frame, OUT / "same" / "exit020", first_tradeable=split, stop=len(df))
    equal = bool(np.array_equal(left["equity"].to_numpy(), right["equity"].to_numpy()))
    print(f"(b) exit_strength 1/3 vs 0.20 on the test half: equity curves identical on all "
          f"{len(left['equity'])} bars: {equal}; "
          f"max abs difference {np.max(np.abs(left['equity'].to_numpy() - right['equity'].to_numpy())):.3e}")

    # (c)
    rows = json.loads((OUT / "training_surface.json").read_text())
    returns = np.load(OUT / "training_returns.npz")
    winner = max(rows, key=lambda row: row["sharpe_tradeable"])
    sample = pd.Series(returns[winner["cell"]])
    per_bar = float(sample.mean() / sample.std(ddof=1))
    factor = (winner["sharpe_tradeable"] / per_bar) ** 2

    daily = (1.0 + sample).groupby(np.arange(len(sample)) // 6).prod() - 1.0
    trials = np.array([row["sharpe_tradeable"] for row in rows], dtype="float64")
    sigma_daily = float(np.std(trials, ddof=1)) / math.sqrt(factor / 6.0)
    observed_daily = float(daily.mean() / daily.std(ddof=1))
    benchmark_daily = expected_max_sharpe(sigma=sigma_daily, trials=len(trials))
    stats = deflated_sharpe(
        observed=observed_daily, benchmark=benchmark_daily, returns=daily.to_numpy()
    )
    print(f"(c) daily aggregation ({len(daily)} days of 6 bars): observed SR/day "
          f"{observed_daily:+.5f}, E[max] {benchmark_daily:+.5f}, skew {stats['skew']:+.3f}, "
          f"excess kurtosis {stats['excess_kurtosis']:+.3f}, z {stats['z']:+.4f}, "
          f"DSR {stats['dsr']:.6f}")
    fraction = float((sample == 0.0).mean())
    print(f"    bars with exactly zero return in the winner's training curve: {fraction:.1%}")


if __name__ == "__main__":
    main()
