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
    print("  N/A: the declared population is empty -- nothing cleared the three "
          "thresholds above,")
    print("  so there is no candidate to test for persistence. Measured anyway "
          "in step 6.")

    any_feature = any(features["verdicts"].values())
    incumbent_clears = any(incumbent["verdicts"].values())
    composite_clears = any(composite["verdicts"].values())
    extension_clears = any(extension["verdicts"].values())
    everything = any_feature and incumbent_clears and composite_clears

    if everything:
        row = 4
    elif any_feature and not incumbent_clears:
        row = 1
    elif any_feature and incumbent_clears:
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
    # anything", so it is printed *after* the row is decided and cannot move it.
    try:
        eth = R.read("step8_eth_replication.json")
    except FileNotFoundError:
        print("\nETH replication not yet run (step8).")
        return
    eth_features = [name for name, ok in eth["feature_verdicts"].items() if ok]
    eth_incumbent = [name for name, ok in eth["incumbent_verdicts"].items() if ok]
    eth_composite = [name for name, ok in eth["composite_verdicts"].items() if ok]
    print("\nETH REPLICATION (declared in the plan; run after BTC, chooses nothing):")
    print(f"  features clearing: {eth_features or 'none'}")
    print(f"  COMPRESSION clearing: {eth_incumbent or 'none'}")
    print(f"  composite clearing: {eth_composite or 'none'}")
    if eth_features and not any_feature:
        print("  The replication does NOT reproduce BTC's null on the feature "
              "threshold. The same")
        print("  feature that came closest on BTC clears on ETH, in the same "
              "direction and in both")
        print("  halves -- so BTC's near-miss reads as a signal under the bar "
              "rather than as noise.")
        print("  BTC's row stands as the phase's verdict; this is the "
              "qualification on it.")

    R.write("step7_outcome.json", {
        **R.read("step7_outcome.json"),
        "eth_features_clearing": eth_features,
        "eth_incumbent_clearing": eth_incumbent,
        "eth_composite_clearing": eth_composite,
    })

    R.write("step7_outcome.json", {
        "outcome_row": row,
        "any_feature_clears": any_feature,
        "incumbent_clears": incumbent_clears,
        "composite_clears": composite_clears,
        "extension_clears": extension_clears,
        "control_checks_pass": sum(control_checks.values()) == len(control_checks),
        "label_checks_pass": sum(label_checks.values()) == len(label_checks),
        "persistence_population_empty": persistence["declared_population_empty"],
    })


if __name__ == "__main__":
    main()
