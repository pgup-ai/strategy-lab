"""Step 5 (plan §4): the ``enter_strength`` extension -- R9's lead.

``enter_strength`` is the only axis stable across all nine of R9's walk-forward
folds, it wins at the most selective value ever tried, and it has sat on the
swept range's edge in every study that has touched it (R5, the ETH replication,
R9). Either the grid stopped short of the real optimum, or 0.80 is an interior
optimum of a wider grid nobody has drawn.

The axis is extended to **{0.85, 0.90, 0.95}**, holding the other three at R5's
trained values, scored on the **training half only** by M21's scalar -- Sharpe
over the bars a cell can trade, recomputed from the run's own equity curve with
the engine's own estimator.

**This is a grid extension, not a re-selection.** No winner is re-derived, no
published figure moves, and nothing here is pinned anywhere (M22). Its whole
purpose is to say whether R9's ridge is a boundary artifact.

**Every cell trades the same bars.** ``first_tradeable`` is the deepest warmup
in R5's own 54-cell grid (bar 2,352), not each cell's own, because M21 is
precisely the finding that a cell scored over a longer flat lead-in is scored on
a different window rather than measured better. The trained cell is re-run here
under the identical rule as the control: it must reproduce +1.3974.

**Declared threshold:** 0.80 was a boundary artifact if any of {0.85, 0.90,
0.95} beats **+1.3974**.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import replace

import r7lib as R

EXTENSION = (0.85, 0.90, 0.95)


def _sharpe(value: float | None) -> str:
    """A Sharpe that is ``None`` is a book that never traded, not a zero."""
    return "null" if value is None else f"{value:+.4f}"


def main() -> None:
    started = time.time()
    frame = R.load_frame()
    df, _ = frame
    split = R.split_index(df)

    # The 54-cell grid's deepest warmup, read off the grid rather than pinned,
    # so this cannot drift from the gate it is being compared against.
    grid = [R.gate.machine(config) for config in R.gate.declared_machines()]
    first_tradeable = max(strategy.warmup_bars for strategy in grid)
    print(f"frame: {len(df)} bars; training half ends at bar {split} "
          f"({df.index[split - 1]})")
    print(f"first tradeable bar {first_tradeable} ({df.index[first_tradeable]}), "
          f"{split - first_tradeable} tradeable bars -- the deepest warmup of R5's 54")

    cells = [("control 0.80 (R5's trained cell)", R.TRAINED)] + [
        (f"extension {value:.2f}", replace(R.TRAINED, enter_strength=value))
        for value in EXTENSION
    ]

    print(f"\n{'cell':>34} {'warmup':>7} {'M21 sharpe':>11} {'whole-frame':>12} "
          f"{'net %':>9} {'maxDD %':>8} {'trades':>7} {'3x costs %':>11}")
    rows = []
    root = R.OUT / "enter_strength"
    for name, config in cells:
        strategy = R.gate.machine(config)
        row = R.gate_run(strategy, frame, root / name.replace(" ", "_"),
                         first_tradeable=first_tradeable, stop=split)
        row["label"] = name
        row["enter_strength"] = config.enter_strength
        rows.append(row)
        tradeable = _sharpe(row["sharpe_tradeable"])
        whole = _sharpe(row["sharpe_whole_frame"])
        stress = row["net_by_stress"].get(3.0, row["net_by_stress"].get("3.0", float("nan")))
        print(f"{name:>34} {row['warmup_bars']:>7} {tradeable:>11} {whole:>12} "
              f"{row['net_return_pct']:>+9.2f} {row['max_drawdown_pct']:>8.2f} "
              f"{row['trades']:>7} {stress:>+11.2f}")
    shutil.rmtree(root, ignore_errors=True)

    control = rows[0]
    print(f"\nCONTROL: the trained cell's M21 scalar is "
          f"{control['sharpe_tradeable']:+.4f}; R9 published +{R.ENTER_STRENGTH_BAR:.4f}")
    control_ok = abs(control["sharpe_tradeable"] - R.ENTER_STRENGTH_BAR) <= 5e-4
    print(f"  {R.mark(control_ok)}  the extension is scored on the same rule R9 "
          f"scored the incumbent on")

    print("\n" + "=" * 78)
    print(f"DECLARED THRESHOLD -- 0.80 was a boundary artifact if any of "
          f"{{{', '.join(f'{v:.2f}' for v in EXTENSION)}}}")
    print(f"beats +{R.ENTER_STRENGTH_BAR:.4f} (M21 scalar, training half).")
    print(f"\n{'cell':>34} {'M21 sharpe':>11} {'vs +1.3974':>11} {'verdict':>8}")
    verdicts = {}
    for row in rows[1:]:
        sharpe = row["sharpe_tradeable"]
        passed = sharpe is not None and sharpe > R.ENTER_STRENGTH_BAR
        verdicts[row["label"]] = passed
        gap = "null" if sharpe is None else f"{sharpe - R.ENTER_STRENGTH_BAR:+.4f}"
        print(f"{row['label']:>34} {_sharpe(sharpe):>11} {gap:>11} {R.mark(passed):>8}")

    cleared = [name for name, ok in verdicts.items() if ok]
    print(f"\n{len(cleared)}/{len(verdicts)} extension cells beat the incumbent: "
          f"{cleared or 'none'}")
    print("Read this as a statement about the *ridge*, not as a selection: "
          "nothing here is pinned,")
    print("and R5's published cell is unchanged whichever way it came out (M22).")

    R.write("step5_enter_strength.json", {
        "verdicts": verdicts,
        "control_reproduces": control_ok,
        "first_tradeable": first_tradeable,
        "cells": [R.slim_row(row) for row in rows],
    })
    print(f"\nelapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
