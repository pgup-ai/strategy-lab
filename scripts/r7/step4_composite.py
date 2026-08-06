"""Step 4 (plan §3): the composite gate, scored as a binary detector.

``strength >= enter_strength AND |direction| >= direction_floor`` is the actual
entry condition -- the ``advancing`` predicate in ``state/machine.py``, which is
what moves the machine up its lifecycle at all. Scored here at R5's trained
values against the same label, beside each of its two components alone.

**A composite that beats its own best component is the finding that would
justify the state machine's shape**; one that does not says the machine is
carrying two features where one would do.

``strength`` is read in **rank space** (``rolling_percentile`` over 480 bars),
because that is the space ``enter_strength`` is a threshold in --
``strategies/state_machine_core.py``'s ``RANKED_FEATURES``. ``direction`` is
raw, since ranking a signed series destroys its sign. Getting this wrong would
score a threshold that no machine ever applied.

**Declared threshold:** the composite earns its second input if its trend rate
beats **both** single-feature gates by >= 5 pp, in both halves.
"""

from __future__ import annotations

import time

import numpy as np
import r7lib as R


def main() -> None:
    started = time.time()
    df, _ = R.load_frame()
    halves = R.halves_of(df)
    enter = R.TRAINED.enter_strength
    floor = R.TRAINED.direction_floor
    print(f"frame: {len(df)} bars; split {halves.timestamp}")
    print(f"R5's trained values: enter_strength={enter:.4f}, "
          f"direction_floor={floor:.4f} (never swept -- the grid moves the first, "
          f"not the second)")

    frame, crowding_measured = R.machine_frame(df)
    assert crowding_measured
    strength_rank = frame["strength"]
    direction = frame["direction"]
    # Only bars where both inputs are measurable can carry a verdict, and that
    # is exactly the machine's own ``measurable`` predicate.
    defined = (strength_rank.notna() & direction.notna()).to_numpy()
    print(f"both inputs measurable on {int(defined.sum())} of {len(df)} bars "
          f"(from {df.index[int(np.argmax(defined))]})")

    gates = {
        "strength >= 0.80 (rank)": (strength_rank >= enter).to_numpy(),
        "|direction| >= 0.10": (direction.abs() >= floor).to_numpy(),
    }
    gates["composite (both)"] = gates["strength >= 0.80 (rank)"] & gates["|direction| >= 0.10"]

    payload = {}
    for horizon in R.HORIZONS:
        label, cuts = R.trend_label(R.forward_er(df["close"], horizon=horizon), halves)
        print(f"\nH={horizon} -- trend is ER > {cuts[1]:.4f}")
        print(f"{'gate':>26} {'cover':>7} "
              f"{'train base':>11} {'train in':>9} {'train lift':>11} "
              f"{'test base':>10} {'test in':>8} {'test lift':>10}")
        rows = {}
        for name, mask in gates.items():
            table = R.rate_table(R.verdict_of(mask, df.index, defined), label, halves)
            rows[name] = table
            print(f"{name:>26} {table['full']['coverage']:>7.1%} "
                  f"{table['train']['base_rate']:>11.4f} "
                  f"{table['train']['inside_rate']:>9.4f} "
                  f"{table['train']['lift_pp']:>+11.2f} "
                  f"{table['test']['base_rate']:>10.4f} "
                  f"{table['test']['inside_rate']:>8.4f} "
                  f"{table['test']['lift_pp']:>+10.2f}")
        payload[horizon] = rows

    # ---- the declared threshold ----------------------------------------
    print("\n" + "=" * 78)
    print("DECLARED THRESHOLD -- the composite earns its second input if its "
          "trend rate beats")
    print(f"BOTH single-feature gates by >= {R.COMPOSITE_BAR_PP:.0f} pp, in both halves.")
    print(f"\n{'H':>4} {'half':>6} {'composite':>10} {'strength':>9} {'direction':>10} "
          f"{'vs strength':>12} {'vs direction':>13} {'verdict':>8}")
    verdicts = {}
    for horizon in R.HORIZONS:
        rows = payload[horizon]
        cells = []
        for which in ("train", "test"):
            composite = rows["composite (both)"][which]["inside_rate"]
            alone_s = rows["strength >= 0.80 (rank)"][which]["inside_rate"]
            alone_d = rows["|direction| >= 0.10"][which]["inside_rate"]
            gap_s = 100.0 * (composite - alone_s)
            gap_d = 100.0 * (composite - alone_d)
            cells.append((which, composite, alone_s, alone_d, gap_s, gap_d))
        passed = all(
            gap_s >= R.COMPOSITE_BAR_PP and gap_d >= R.COMPOSITE_BAR_PP
            for _, _, _, _, gap_s, gap_d in cells
        )
        verdicts[f"composite H={horizon}"] = passed
        for which, composite, alone_s, alone_d, gap_s, gap_d in cells:
            print(f"{horizon:>4} {which:>6} {composite:>10.4f} {alone_s:>9.4f} "
                  f"{alone_d:>10.4f} {gap_s:>+12.2f} {gap_d:>+13.2f} "
                  f"{R.mark(passed) if which == 'test' else '':>8}")

    cleared = [name for name, ok in verdicts.items() if ok]
    print(f"\nthe composite clears at {len(cleared)}/{len(verdicts)} horizons: "
          f"{cleared or 'none'}")

    R.write("step4_composite.json", {
        "verdicts": verdicts,
        "enter_strength": enter,
        "direction_floor": floor,
        "rates": payload,
    })
    print(f"\nelapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
