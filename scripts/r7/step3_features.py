"""Step 3 (plan §2): every registered feature, scored against forward ER.

R4's table with a different target. R4 asked "does this predict *direction*";
R7 asks "does this predict *whether direction is worth trading*". Several
features that failed R4's bar may clear this one -- that is the hypothesis, and
``energy``/``compression`` are the two designed for it.

**The halves are reported two ways, because the pre-registration is not
determinate here.** Its frame block fixes the split at 2023-10-31 (R5's split,
60/40); its §2 calls itself "R4's table with a different target", and R4's
halves are the aligned sample cut at its own **midpoint**. Those are not the
same cut. The declared split is treated as primary -- "Fixed here, before
anything runs" governs -- and R4's own cut is printed beside it so the two
tables are comparable and neither is chosen after the fact.

**Effective sample size is n/H, not n.** Forward windows at H = 30 overlap 30
ways, so the ~13,000-row IC at H = 90 rests on ~150 independent observations and
its noise band is ~0.16, not ~0.017. Printed per horizon, because a bar set in
IC units alone would mean three different things across the three horizons.

**Declared threshold:** a feature detects chop if |IC| >= 0.10 against ER at
H = 30, **and** both half-sample ICs agree in sign, **and** each is >= 0.05 in
absolute value.
"""

from __future__ import annotations

import time

import numpy as np
import r7lib as R

# The two conditioners the machine reads as trailing ranks rather than as
# levels. Not registry features and NOT candidates under the declared
# threshold -- printed because §3's composite gate is a threshold in this space,
# and a reader comparing the two tables needs both.
RANKED = ("strength", "stability")


def verdict(row30: dict) -> tuple[bool, str]:
    """The plan's threshold, evaluated mechanically on the H=30 row."""
    full = row30["full"]["ic"]
    first, second = row30["train"]["ic"], row30["test"]["ic"]
    reasons = []
    if not abs(full) >= R.IC_BAR:
        reasons.append(f"|IC| {abs(full):.4f} < {R.IC_BAR}")
    if not first * second > 0:
        reasons.append(f"halves disagree in sign ({first:+.4f}/{second:+.4f})")
    if not min(abs(first), abs(second)) >= R.IC_HALF_BAR:
        reasons.append(f"weaker half {min(abs(first), abs(second)):.4f} < {R.IC_HALF_BAR}")
    return not reasons, "; ".join(reasons)


def main() -> None:
    started = time.time()
    df, _ = R.load_frame()
    halves = R.halves_of(df)
    print(f"frame: {len(df)} bars; split {halves.timestamp}")

    columns = R.feature_columns(df)
    for name in RANKED:
        columns[f"{name}_rank"] = R.ranked(columns[name])

    targets = {horizon: R.forward_er(df["close"], horizon=horizon) for horizon in R.HORIZONS}
    print("\nnoise scale, 2 sd of a null IC given overlapping windows:")
    for horizon in R.HORIZONS:
        n = int(targets[horizon].notna().sum())
        print(f"  H={horizon:>3}: n={n}, effective n≈{n // horizon}, "
              f"2/sqrt(n/H)={2 / np.sqrt(n / horizon):.4f}")

    print("\nSpearman IC vs forward ER -- full (train / test), by the declared "
          "2023-10-31 split")
    print(f"{'feature':>20} {'warmup':>7} " + " ".join(
        f"{f'H={horizon}':>28}" for horizon in R.HORIZONS
    ))
    rows = {}
    for name, values in columns.items():
        measured = values.dropna()
        cells = []
        rows[name] = {}
        for horizon in R.HORIZONS:
            table = R.ic_table(measured, targets[horizon], halves, horizon=horizon)
            rows[name][horizon] = table
            cells.append(f"{table['full']['ic']:+.4f}"
                         f"({table['train']['ic']:+.4f}/{table['test']['ic']:+.4f})")
        warmup = len(df) - len(measured) if name.endswith("_rank") else None
        label = f"{name}{'  (rank)' if name.endswith('_rank') else ''}"
        print(f"{label:>20} {(warmup if warmup is not None else len(df) - len(measured)):>7} "
              + " ".join(f"{cell:>28}" for cell in cells))

    print("\nthe same table under R4's own halves (aligned sample cut at its "
          "midpoint), for comparability")
    print(f"{'feature':>20} " + " ".join(f"{f'H={horizon}':>28}" for horizon in R.HORIZONS))
    r4_rows = {}
    for name, values in columns.items():
        measured = values.dropna()
        cells = []
        r4_rows[name] = {}
        for horizon in R.HORIZONS:
            entry = R.r4_style_ic(measured, targets[horizon], horizon=horizon)
            r4_rows[name][horizon] = entry
            cells.append(f"{entry['ic']:+.4f}"
                         f"({entry['first_half_ic']:+.4f}/{entry['second_half_ic']:+.4f})")
        print(f"{name:>20} " + " ".join(f"{cell:>28}" for cell in cells))

    # ---- the declared threshold ----------------------------------------
    print("\n" + "=" * 78)
    print("DECLARED THRESHOLD -- a feature detects chop if |IC| >= "
          f"{R.IC_BAR} at H=30, both halves")
    print(f"agree in sign, and each is >= {R.IC_HALF_BAR} in absolute value.")
    print(f"\n{'feature':>20} {'IC@30':>8} {'train':>8} {'test':>8} {'verdict':>8}  why not")
    verdicts = {}
    for name in columns:
        passed, why = verdict(rows[name][30])
        candidate = not name.endswith("_rank")
        if candidate:
            verdicts[name] = passed
        table = rows[name][30]
        suffix = "" if candidate else "   (not one of the nine)"
        print(f"{name:>20} {table['full']['ic']:>+8.4f} {table['train']['ic']:>+8.4f} "
              f"{table['test']['ic']:>+8.4f} {R.mark(passed):>8}  {why}{suffix}")

    cleared = [name for name, ok in verdicts.items() if ok]
    print(f"\n{len(cleared)}/{len(verdicts)} of the nine registered features clear: "
          f"{cleared or 'none'}")

    # The largest thing on the page, whatever the threshold says about it.
    best = max(
        ((name, rows[name][horizon], horizon) for name in columns for horizon in R.HORIZONS),
        key=lambda item: abs(item[1]["full"]["ic"]),
    )
    print(f"largest |IC| anywhere in the table: {best[0]} at H={best[2]}, "
          f"{best[1]['full']['ic']:+.4f} "
          f"({best[1]['train']['ic']:+.4f}/{best[1]['test']['ic']:+.4f})")

    R.write("step3_features.json", {
        "verdicts": verdicts,
        "declared_split": rows,
        "r4_halves": r4_rows,
        "ranked_extra": list(RANKED),
    })
    print(f"\nelapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
