"""Steps 1-2 (plan §2 then §1): the surface, the selection, and the kill switch.

The plan numbers the kill switch before the selection because it is *read*
first -- before any out-of-sample return exists. Mechanically the selection has
to happen first: the switch is a statement about **the selected cell's** trade
count. So this one script does both, in that order, and prints the switch's
verdict before anything downstream could use it.

- **Selection** is the highest Sharpe over **tradeable bars** (M21) across the
  declared grid, on **BTC's training half only**. One selection, reported with
  its full surface.
- **The kill switch** is tradeability: the selected cell must trade between
  ``KILL_SWITCH_MIN`` and ``KILL_SWITCH_MAX`` round trips on that same half --
  R5's 77, within a factor of three either way. Below it the P&L is a handful of
  draws; above it the run is being scored on costs rather than on state. **If it
  fires, the SOL holdout is not spent and the phase stops.**

R5's own trained cell is run over the identical bars as a **reference**, not as
a candidate: it is where the 77 in the band comes from, and a band whose anchor
is never re-measured is a number copied out of a document. Note its Sharpe here
is **not** R9's published +1.3974 and is not meant to be -- R9 scored the 54-cell
grid from that grid's deepest warmup (2,352 bars) and R7c's five cells all warm
2,160, so the reference is re-scored on R7c's own window. The round-trip count,
which is what the band is built on, is 77 either way.

**The closing block is not a declared measurement and settles nothing.** It
exists because the first question a reader should ask of a kill switch that
fired is whether the churn is the hypothesis or a coding error. It is neither
the rate diagnostic R7b already settled (M27) nor a threshold anyone declared:
it is the coverage of the two predicates that were swapped, which is what makes
the turnover explicable rather than merely observed.
"""

from __future__ import annotations

import shutil
import time

import numpy as np
import r7clib as R


def coverage(machine, features, ok) -> dict:
    """How often each side of a machine's lifecycle predicate fires.

    Over the bars the machine can measure at all, so a warmup row is excluded
    rather than counted as a quiet one -- the same set ``StateMachine.run``
    calls ``measurable``.
    """
    energy = features["energy"].to_numpy(dtype="float64")
    strength = features["strength"].to_numpy(dtype="float64")
    if machine.energy_first:
        entering, leaving = energy <= machine.enter_energy, energy > machine.exit_energy
    else:
        entering = strength >= machine.enter_strength
        leaving = strength < machine.exit_strength
    return {
        "entering_share": float(entering[ok].mean()),
        "leaving_share": float(leaving[ok].mean()),
        "measurable_bars": int(ok.sum()),
    }


def main() -> None:
    started = time.time()
    frame = R.R7.load_frame()
    df, _ = frame
    split = R.R7.split_index(df)
    warmup = R.machine_warmup()
    print(f"frame: {len(df)} bars; training half is bars [0, {split}) "
          f"ending {df.index[split]}")
    print(f"every cell warms {warmup} bars, so the training half scores "
          f"{split - warmup} tradeable bars")

    live, refused = R.constructible_cells()
    print(f"\ndeclared grid: {len(R.DECLARED_CELLS)} cells, of which "
          f"{len(live)} are constructible and {len(refused)} refused")
    for enter, exit_ in refused:
        print(f"  REFUSED  {R.label_of(enter, exit_)}: exit_energy must exceed "
              f"enter_energy, which is the dead band the plan asks for")

    by_label = {R.label_of(*cell): cell for cell in live}
    strategies = {label: R.strategy_for(*cell) for label, cell in by_label.items()}
    reference_label = "R5 trained (reference, not a candidate)"
    strategies[reference_label] = R.R7.gate.machine(R.R7.TRAINED)

    root = R.OUT / "selection"
    rows = R.run_over(strategies, frame, root, first_tradeable=warmup, stop=split)

    print(f"\n{R.header()}")
    for label in strategies:
        print(R.line(label, rows[label]))

    scored = {
        label: rows[label] for label in by_label
        if rows[label]["sharpe_tradeable"] is not None
    }
    assert scored, "no candidate cell produced a Sharpe at all"
    winner = max(scored, key=lambda label: scored[label]["sharpe_tradeable"])
    enter_energy, exit_energy = by_label[winner]
    runner_up = sorted(row["sharpe_tradeable"] for row in scored.values())[-2]
    print(f"\nSELECTED  {winner}  Sharpe {scored[winner]['sharpe_tradeable']:+.4f} "
          f"(runner-up {runner_up:+.4f})")

    reference = rows[reference_label]
    print(f"\nR5's trained cell over the identical bars: {reference['trades']} round "
          f"trips, Sharpe {reference['sharpe_tradeable']:+.4f}")
    print(f"  the declared band is {R.KILL_SWITCH_MIN}-{R.KILL_SWITCH_MAX}, a factor "
          f"of three either side of the plan's {R.R5_TRAINING_TRADES}")

    trades = rows[winner]["trades"]
    fired = not (R.KILL_SWITCH_MIN <= trades <= R.KILL_SWITCH_MAX)
    every_cell_fires = all(
        not (R.KILL_SWITCH_MIN <= rows[label]["trades"] <= R.KILL_SWITCH_MAX)
        for label in by_label
    )
    print("\n" + "=" * 78)
    print("THE KILL SWITCH -- tradeability, before any out-of-sample return")
    print(f"  the selected cell trades {trades} round trips on BTC's training half")
    print(f"  {R.mark(not fired)}  "
          + ("OUTSIDE the band: the phase stops here and SOL is NOT spent"
             if fired else "inside the band: the phase continues"))
    if fired:
        print(f"  every constructible cell fires it: "
              f"{every_cell_fires}  (counts "
              f"{sorted(rows[label]['trades'] for label in by_label)})")

    # --- not a declared measurement; see the module docstring -----------------
    features, crowding_measured = R.R7.machine_frame(df.iloc[:split])
    assert crowding_measured, "the training frame lost its funding column"
    ok = np.isfinite(features.to_numpy(dtype="float64")).all(axis=1)
    predicates = {
        "R5 trained": coverage(R.R7.TRAINED, features, ok),
        winner: coverage(R.cell(enter_energy, exit_energy), features, ok),
    }
    print(f"\nwhy, on the {predicates['R5 trained']['measurable_bars']} measurable "
          f"training bars (context, not a verdict):")
    for label, row in predicates.items():
        print(f"  {label:>28}: advancing on {100 * row['entering_share']:5.1f}% of bars, "
              f"failing on {100 * row['leaving_share']:5.1f}%")

    R.write("step1_selection.json", {
        "split_at": str(df.index[split]),
        "frame_bars": len(df),
        "warmup_bars": warmup,
        "tradeable_bars": rows[winner]["tradeable_bars"],
        "declared_cells": [list(cell) for cell in R.DECLARED_CELLS],
        "constructible_cells": [list(cell) for cell in live],
        "refused_cells": [list(cell) for cell in refused],
        "surface": {label: R.slim(row) for label, row in rows.items()},
        "selected": {
            "label": winner, "enter_energy": enter_energy, "exit_energy": exit_energy
        },
        "selected_sharpe_tradeable": scored[winner]["sharpe_tradeable"],
        "runner_up_sharpe_tradeable": runner_up,
        "r5_reference_trades": reference["trades"],
        "kill_switch": {
            "trades": trades,
            "band": [R.KILL_SWITCH_MIN, R.KILL_SWITCH_MAX],
            "fired": fired,
            "every_constructible_cell_fires": every_cell_fires,
        },
        "predicate_coverage_not_a_verdict": predicates,
    })
    shutil.rmtree(root, ignore_errors=True)
    print(f"\nelapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
