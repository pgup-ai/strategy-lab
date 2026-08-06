"""Step 7: every declared threshold, evaluated, and the outcome row it lands in.

Reads the previous steps' JSON and does no measurement of its own. The plan
states its thresholds as numbers precisely so this is mechanical (M23), and the
outcome table was declared before any figure existed -- so which row lands is
read off the verdicts rather than chosen after seeing them.
"""

from __future__ import annotations

import r7lib as R

OUTCOMES = {
    1: ("Something clears AND beats COMPRESSION",
        "the phase has a product: a better chop detector, to be wired and gated "
        "the way R5 was"),
    2: ("Nothing clears but COMPRESSION does",
        "the feature set is exhausted and the incumbent is as good as it gets "
        "from these inputs; the answer is a better ESTIMATOR over the same "
        "features (R8's HMM/Kalman), not a better feature"),
    3: ("Nothing clears, COMPRESSION included",
        "chop is NOT detectable from this feature set at 4h. COMPRESSION is "
        "doing something other than what it is named, and the thesis needs a "
        "different frequency, a different data source (OI, book, "
        "cross-sectional), or both. R8 as written should not be built on these "
        "features"),
    4: ("Everything clears comfortably",
        "suspect the label before celebrating -- audit the t+1 anchoring and "
        "the tercile boundaries for leakage first"),
}


def main() -> None:
    control = R.read("step0_control.json")
    label = R.read("step1_label.json")
    incumbent = R.read("step2_incumbent.json")
    features = R.read("step3_features.json")
    composite = R.read("step4_composite.json")
    extension = R.read("step5_enter_strength.json")
    persistence = R.read("step6_persistence.json")

    print("=" * 78)
    print("R7 -- every declared threshold, evaluated")
    print("=" * 78)

    control_checks = control["checks"]
    label_checks = label["checks"]
    print(f"\nStep 0 control (R4's published tables): "
          f"{sum(control_checks.values())}/{len(control_checks)} pass")
    print(f"Step 1 label audit:                      "
          f"{sum(label_checks.values())}/{len(label_checks)} pass")
    for name, ok in list(control_checks.items()) + list(label_checks.items()):
        if not ok:
            print(f"  FAILED: {name}")

    print("\n--- the five declared thresholds ---")
    blocks = [
        ("A feature detects chop (|IC| >= 0.10 at H=30, halves agree, each >= 0.05)",
         features["verdicts"]),
        ("The incumbent COMPRESSION detects chop (>= 10 pp below base, both halves)",
         incumbent["verdicts"]),
        ("The composite gate earns its second input (>= 5 pp over both, both halves)",
         composite["verdicts"]),
        ("enter_strength 0.80 was a boundary artifact (any of 0.85/0.90/0.95 > +1.3974)",
         extension["verdicts"]),
    ]
    for title, verdicts in blocks:
        cleared = [name for name, ok in verdicts.items() if ok]
        print(f"\n{title}")
        print(f"  {R.mark(bool(cleared))}: {len(cleared)}/{len(verdicts)} clear"
              f"{' -- ' + ', '.join(cleared) if cleared else ''}")

    print("\nA candidate is usable (median run length >= 6 bars)")
    if persistence["declared_population_empty"]:
        print("  N/A: nothing cleared the three thresholds above, so there is no "
              "candidate to")
        print("  test. Measured anyway in step 6.")
    else:
        print(f"  {sum(persistence['usable'].values())}/{len(persistence['usable'])} "
              f"of the cleared candidates are usable")

    any_feature = any(features["verdicts"].values())
    incumbent_clears = any(incumbent["verdicts"].values())
    composite_clears = any(composite["verdicts"].values())
    extension_clears = any(extension["verdicts"].values())

    if any_feature and incumbent_clears and composite_clears:
        row = 4
    elif any_feature:
        row = 1
    elif incumbent_clears:
        row = 2
    else:
        row = 3

    print("\n" + "=" * 78)
    title, reading = OUTCOMES[row]
    print(f"OUTCOME ROW {row}: {title}")
    print(f"  {reading}")
    print("=" * 78)
    print("\nThe extension result is separate from the outcome table -- the plan "
          "declares it as its")
    print("own threshold, not as a row. It reads:")
    print(f"  enter_strength 0.80 is {'a boundary artifact' if extension_clears else 'an INTERIOR optimum'}"
          f" of the wider grid.")
    print(f"  control reproduces R9's +1.3974: {extension['control_reproduces']}")

    # The ETH replication is declared "run after BTC and never used to choose
    # anything", so it is read *after* the row is decided and cannot move it.
    # Step 8 runs last, so on the first pass through the chain there is nothing
    # here yet and the row above still stands on its own.
    eth = R.read_if_present("step8_eth_replication.json")
    eth_clearing = {
        f"eth_{kind}_clearing": [name for name, ok in eth[f"{kind}_verdicts"].items() if ok]
        for kind in ("feature", "incumbent", "composite")
    } if eth else {}

    if eth:
        print("\nETH REPLICATION (declared in the plan; run after BTC, chooses nothing):")
        for kind in ("feature", "incumbent", "composite"):
            print(f"  {kind} clearing: {eth_clearing[f'eth_{kind}_clearing'] or 'none'}")
    else:
        print("\nETH replication not yet run (step8).")
    if eth_clearing.get("eth_feature_clearing") and not any_feature:
        print("  The replication does NOT reproduce BTC's null on the feature "
              "threshold: the")
        print("  same feature clears on ETH, same direction, both halves. BTC's "
              "row stands as")
        print("  the verdict; this is the qualification on it.")

    R.write("step7_outcome.json", {
        "outcome_row": row,
        "any_feature_clears": any_feature,
        "incumbent_clears": incumbent_clears,
        "composite_clears": composite_clears,
        "extension_clears": extension_clears,
        "control_checks_pass": sum(control_checks.values()) == len(control_checks),
        "label_checks_pass": sum(label_checks.values()) == len(label_checks),
        "persistence_population_empty": persistence["declared_population_empty"],
        **eth_clearing,
    })


if __name__ == "__main__":
    main()
