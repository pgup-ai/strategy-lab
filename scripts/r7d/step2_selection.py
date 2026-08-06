"""Steps 2-3 (plan §2 then §3): the surface, the selection, and the second switch.

The plan numbers the tradeability switch after the selection because it is a
statement about **the selected cell's** trade count, so one script does both and
prints the switch's verdict before anything downstream could use it.

- **Selection** is the highest Sharpe over **tradeable bars** (M21) across the
  three declared cells, on **BTC's training half only**, with the full surface
  reported. Nothing else is re-derived, on any frame (M22).
- **The second kill switch** is R7c's, unchanged: the selected cell must trade
  between 26 and 231 round trips on that same half -- R5's 77 within a factor of
  three either way. **If it fires, the SOL holdout is not spent and the phase
  stops.**

The three cells are the **coverage** targets of step 1, and their ``energy``
values are read from ``step1_lift.json`` rather than re-derived here, so the
machine that trades is provably the one whose derivation the previous step
printed.

R5's own trained cell is run over the identical bars as a **reference, not a
candidate**: it is where the 77 in the band comes from, and a band whose anchor
is never re-measured is a number copied out of a document. Its Sharpe here is
**not** R9's published +1.3974 and is not meant to be -- R9 scored the 54-cell
grid from that grid's deepest warmup (2,352 bars) while every cell here warms
2,160, which is R7c's own caveat 2. The round-trip count the band is built on is
77 either way.
"""

from __future__ import annotations

import shutil
import time

import r7dlib as R

R7 = R.R7


def main() -> None:
    started = time.time()
    derivation = R.read("step1_lift.json")
    assert derivation["any_target_clears"], (
        "step 1's kill switch fired at every coverage target; the phase stops there"
    )

    frame = R7.load_frame()
    df, _ = frame
    split = R7.split_index(df)
    print(f"frame: {len(df)} bars; training half is bars [0, {split}) "
          f"ending {df.index[split]}")

    cells = {
        row["cell"]["label"]: row["cell"]
        for row in derivation["frames"]["BTC"]["gates"].values()
        if row["kind"] == "declared"
    }
    assert len(cells) == len(R.ENTER_COVERAGE_TARGETS), (
        f"expected {len(R.ENTER_COVERAGE_TARGETS)} declared cells, got {sorted(cells)}"
    )

    print("\nthe declared grid, in coverage, with BTC's own derived values:")
    for label, cell in cells.items():
        print(f"  {label:>34}: enter_energy {cell['enter']['energy_value']:.4f} "
              f"(covers {cell['enter']['realised_coverage']:.2%}), exit_energy "
              f"{cell['exit']['energy_value']:.4f} "
              f"(covers {cell['exit']['realised_coverage']:.2%})")

    strategies = {label: R.strategy_for(cell) for label, cell in cells.items()}
    reference_label = "R5 trained (reference, not a candidate)"
    strategies[reference_label] = R7.gate.machine(R7.TRAINED)
    warmup = R.shared_warmup(strategies)
    print(f"\nevery cell warms {warmup} bars, so the training half scores "
          f"{split - warmup} tradeable bars")

    root = R.OUT / "selection"
    rows = R.run_over(strategies, frame, root, first_tradeable=warmup, stop=split)

    print(f"\n{R.header()}")
    for label in strategies:
        print(R.line(label, rows[label]))

    scored = {
        label: rows[label] for label in cells
        if rows[label]["sharpe_tradeable"] is not None
    }
    assert scored, "no candidate cell produced a Sharpe at all"
    winner = max(scored, key=lambda label: scored[label]["sharpe_tradeable"])
    ranked = sorted(row["sharpe_tradeable"] for row in scored.values())
    runner_up = f"{ranked[-2]:+.4f}" if len(ranked) > 1 else "none -- only one cell scored"
    print(f"\nSELECTED  {winner}  Sharpe {scored[winner]['sharpe_tradeable']:+.4f} "
          f"(runner-up {runner_up})")

    reference = rows[reference_label]
    print(f"\nR5's trained cell over the identical bars: {reference['trades']} round "
          f"trips, Sharpe {reference['sharpe_tradeable']:+.4f}")
    print(f"  the declared band is {R.KILL_SWITCH_MIN}-{R.KILL_SWITCH_MAX}, a factor "
          f"of three either side of the plan's {R.R5_TRAINING_TRADES}")

    trades = rows[winner]["trades"]
    fired = not (R.KILL_SWITCH_MIN <= trades <= R.KILL_SWITCH_MAX)
    inside = {
        label: R.KILL_SWITCH_MIN <= rows[label]["trades"] <= R.KILL_SWITCH_MAX
        for label in cells
    }
    print("\n" + "=" * 78)
    print("THE SECOND KILL SWITCH -- tradeability, before any out-of-sample return")
    print(f"  the selected cell trades {trades} round trips on BTC's training half")
    print(f"  {R.mark(not fired)}  "
          + ("OUTSIDE the band: the phase stops here and SOL is NOT spent"
             if fired else "inside the band: the phase continues to evaluation"))
    print(f"  every cell's count: "
          f"{ {label: rows[label]['trades'] for label in cells} }")
    print(f"  inside the band: { {label: ok for label, ok in inside.items()} }")

    # The first switch's verdict at the *selected* target, which is the form the
    # plan's threshold table states it in ("at the selected coverage target").
    # Step 1 read it at every target and all three cleared, so this can only
    # confirm -- but a phase that reports the switch only in the form that is
    # easiest to pass is not reporting it.
    target = cells[winner]["enter"]["target_coverage"]
    first_switch = derivation["verdicts"][f"{target:.2f} H={R.HORIZON}"]
    print(f"\nthe first switch at the selected target ({target:.0%}): "
          f"mean {first_switch['mean_pp']:+.2f} pp, all four positive "
          f"{first_switch['all_positive']}  {R.mark(first_switch['clears'])}")

    R.write("step2_selection.json", {
        "split_at": str(df.index[split]),
        "frame_bars": len(df),
        "warmup_bars": warmup,
        "tradeable_bars": rows[winner]["tradeable_bars"],
        "cells": cells,
        "surface": {label: R.slim(row) for label, row in rows.items()},
        "selected": {
            "label": winner,
            "target_coverage": target,
            "enter_energy": cells[winner]["enter"]["energy_value"],
            "exit_energy": cells[winner]["exit"]["energy_value"],
        },
        "selected_sharpe_tradeable": scored[winner]["sharpe_tradeable"],
        "runner_up_sharpe_tradeable": runner_up,
        "r5_reference_trades": reference["trades"],
        "r5_reference_sharpe_tradeable": reference["sharpe_tradeable"],
        "kill_switch": {
            "trades": trades,
            "band": [R.KILL_SWITCH_MIN, R.KILL_SWITCH_MAX],
            "fired": fired,
            "cells_inside_band": inside,
        },
        "first_switch_at_selected_target": first_switch,
    })
    shutil.rmtree(root, ignore_errors=True)
    print(f"\nelapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
