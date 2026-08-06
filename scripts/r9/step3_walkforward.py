"""R9 part 2: walk-forward, geometry fixed by the pre-registration.

Expanding training window from bar 2,352 (the deepest warmup in the grid),
minimum 3,000 training bars, non-overlapping 1,008-bar test blocks stepping to
the end of the frame. On each fold the winner is re-derived from the 54 cells by
the declared scalar -- M21's, tradeable bars only -- and evaluated **once** on
that fold's test block.

R5's own scalar (whole window) is ranked in parallel and never used to select
anything: where the two disagree the disagreement is recorded, not resolved.
"""

from __future__ import annotations

import json
import time

import r9lib as R

OUT = R.OUT
OUT.mkdir(parents=True, exist_ok=True)

FIRST_TRADEABLE = 2_352
MIN_TRAIN_BARS = 3_000
TEST_BLOCK = 1_008
SCALAR = "sharpe_tradeable"
R5_SCALAR = "sharpe_whole_frame"


def folds(total: int) -> list[tuple[int, int, int]]:
    """``(train_start, train_stop, test_stop)`` per fold. Test blocks are exact."""
    out = []
    test_start = FIRST_TRADEABLE + MIN_TRAIN_BARS
    while test_start + TEST_BLOCK <= total:
        out.append((FIRST_TRADEABLE, test_start, test_start + TEST_BLOCK))
        test_start += TEST_BLOCK
    return out


def main() -> None:
    started = time.time()
    frame = R.load_frame()
    df, _ = frame
    cells = R.gate.declared_machines()
    strategies = [R.gate.machine(config) for config in cells]
    deepest = max(strategy.warmup_bars for strategy in strategies)
    assert deepest == FIRST_TRADEABLE, f"deepest warmup {deepest}, plan says {FIRST_TRADEABLE}"

    plan = folds(len(df))
    leftover = len(df) - (plan[-1][2] if plan else 0)
    print(f"{len(plan)} folds; {leftover} bars past the last test block are not evaluated")

    results = []
    for number, (train_start, train_stop, test_stop) in enumerate(plan, start=1):
        test_start = train_stop
        surface = []
        for position, (config, strategy) in enumerate(zip(cells, strategies)):
            row = R.run(
                strategy,
                frame,
                OUT / "wf" / f"f{number}" / f"cell{position:02d}",
                first_tradeable=train_start,
                stop=train_stop,
            )
            row["cell"] = R.cell_key(config)
            row["label"] = R.describe(config)
            surface.append((config, R.slim(row)))

        winner, best = max(surface, key=lambda item: item[1][SCALAR])
        r5_winner, r5_best = max(surface, key=lambda item: item[1][R5_SCALAR])
        agree = winner == r5_winner

        chosen = R.run(
            R.gate.machine(winner),
            frame,
            OUT / "wf" / f"f{number}" / "test",
            first_tradeable=test_start,
            stop=test_stop,
        )
        alternative = None
        if not agree:
            alternative = R.slim(
                R.run(
                    R.gate.machine(r5_winner),
                    frame,
                    OUT / "wf" / f"f{number}" / "test_r5",
                    first_tradeable=test_start,
                    stop=test_stop,
                )
            )
            alternative["label"] = R.describe(r5_winner)

        close = df["close"]
        hold = float(close.iloc[test_stop - 1] / close.iloc[test_start] - 1.0) * 100.0
        fold = {
            "fold": number,
            "train_bars": train_stop - train_start,
            "train_start": str(df.index[train_start]),
            "train_end": str(df.index[train_stop - 1]),
            "test_start": str(df.index[test_start]),
            "test_end": str(df.index[test_stop - 1]),
            "test_bars": test_stop - test_start,
            "buy_and_hold_pct": hold,
            "winner": R.cell_key(winner),
            "winner_label": R.describe(winner),
            "winner_train_scalar": best[SCALAR],
            "winner_train_whole_frame": best[R5_SCALAR],
            "runner_up_train_scalar": sorted(row[SCALAR] for _, row in surface)[-2],
            "r5_winner_label": R.describe(r5_winner),
            "r5_winner_train_scalar": r5_best[R5_SCALAR],
            "scalars_agree": agree,
            "train_positive_cells": sum(1 for _, row in surface if row[SCALAR] > 0),
            "test": R.slim(chosen),
            "test_r5_rule_winner": alternative,
            "surface": [row for _, row in surface],
        }
        results.append(fold)
        print(
            f"fold {number}: train {fold['train_bars']:>5} bars -> {fold['winner_label']} "
            f"({best[SCALAR]:+.4f})  |  test {fold['test_start'][:10]}..{fold['test_end'][:10]} "
            f"sharpe {chosen['sharpe_tradeable']:+.4f} (whole {chosen['sharpe_whole_frame']:+.4f}) "
            f"net {chosen['net_return_pct']:+.2f}% trades {chosen['trades']} "
            f"B&H {hold:+.1f}%  agree={agree}"
        )
        (OUT / "walkforward.json").write_text(json.dumps(results, indent=2, default=str))

    print(f"elapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
