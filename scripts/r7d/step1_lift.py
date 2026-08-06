"""Step 1 (plan §1): the first kill switch -- does the lift survive tightening?

The cheap measurement, and it runs before any backtest. `energy <= 0.50` is the
one thing four phases have returned that is positive and replicated: at H=30 it
lifts the trend rate **+3.51 / +5.10** on BTC and **+4.49 / +6.61** on ETH, all
four instrument-halves, averaging +4.93 pp (M27). What nobody has measured is
whether that survives being asked to be as **selective** as the gate it would
replace -- `strength >= 0.80` advances on 21.1% of measurable bars where
`energy <= 0.50` admits about half of them.

So the grid is declared in **coverage** (M29): enter targets 15% / 21% / 30% of
measurable bars, and at each one the `energy` value is derived mechanically on
the frame's own training half by :func:`r7dlib.energy_for_coverage`, from the
coverage target alone. Nothing here is chosen by how it scores.

**Declared, before the data was seen:** at a coverage target the lift is
positive in **all four instrument-halves** and their **mean is >= +3.0 pp**.
**If no target clears, the phase stops and the SOL holdout is not spent.**

Four things are fixed here rather than after the fact.

- **H=30 is the verdict horizon.** H=6 and H=90 are printed as declared context
  and are part of no verdict, the same as in R7 and R7b.
- **The four instrument-halves are equally weighted** in the mean, which is what
  "their mean" can mean when the four are named as a set and the anchor (+4.93)
  is the unweighted average of the four published numbers. The bar is checked on
  the same arithmetic that produced the number it was anchored on.
- **The derivation reads the training half only**, and the derived value is
  applied unchanged to both halves -- the same rule R7 applies to the tercile
  boundaries that define the label. A value re-derived on the test half would be
  a threshold the test half helped set.
- **The control runs first.** `energy <= 0.50` must reproduce R7b's four
  published lifts, at H=30, on both frames, before any tightened number is read.

**The gate this phase matches selectivity against is re-measured rather than
copied**, and it lands on different digits from the ones M29 prints. On BTC's
own 7,150 measurable training bars, `strength >= 0.80` covers **21.76%** and
`energy <= 0.35` covers **37.22%**, against M29's 21.1% and 37.9%. Three things
about that, none of which moves an R7c or R7d verdict.

- The measurement here is the one `require_comparable_windows`' own docstring
  makes: its published 23.8% / 21.6% / 20.9% at `rank_window` 240 / 480 / 960,
  with the energy gate pinned at 37.1%, reproduces here **exactly**, on BTC's
  full measurable frame.
- M29's pair is BTC's training half after all, under a **different denominator**:
  dropping each column's own NaNs rather than requiring every input finite gives
  **21.07% / 37.92%**, which is M29's figures to two decimals. ETH's full
  measurable frame lands at 21.12% / 37.99%, close enough to look like the
  source and isn't. The right denominator is the joint one, since that is the
  set the machine walks -- so the charter carries the correction and this is the
  measurement it carries.
- **The direction and the size of M29's finding are unchanged**: the energy grid
  R7c declared was about 1.7x less selective than the gate it replaced (37.2%
  against 21.8%), and 21% remains the coverage of R5's own gate to within a
  percentage point on both instruments. So it is reported here, beside the
  number it disagrees with, and it gates nothing -- a control the plan did not
  declare must not be able to stop a phase, and a discrepancy found while
  checking one must not be quietly dropped either.
"""

from __future__ import annotations

import time

import numpy as np
import r7dlib as R

R7 = R.R7


def labels_for(df, halves) -> dict:
    """R7's trend label at each horizon, with its training-half tercile cut."""
    labels, cuts = {}, {}
    for horizon in R7.HORIZONS:
        series, boundaries = R7.trend_label(
            R7.forward_er(df["close"], horizon=horizon), halves
        )
        labels[horizon], cuts[horizon] = series, boundaries[1]
    return labels, cuts


def measure(name: str, df, halves) -> dict:
    """Every declared cell on one frame, plus the reference gate it is judged beside."""
    print(f"\n{'=' * 78}\n{name}: {len(df)} bars, split {halves.timestamp}")
    frame, defined = R.machine_inputs(df)
    energy, strength = frame["energy"], frame["strength"]
    labels, cuts = labels_for(df, halves)
    for horizon in R7.HORIZONS:
        print(f"  H={horizon}: trend is ER > {cuts[horizon]:.4f}")

    train = halves.mask(frame.index, "train") & defined
    print(f"  {int(train.sum())} measurable training bars of {int(defined.sum())} "
          f"measurable in the frame")

    # The gate this phase is matching selectivity against, measured rather than
    # copied out of M29 -- the docstring says which denominator each figure uses.
    strength_gate = (strength >= R7.TRAINED.enter_strength).to_numpy()
    strength_coverage = R.coverage_of(strength_gate, train)
    r7c_tightest = R.coverage_of((energy <= 0.35).to_numpy(), train)
    print(f"  `strength >= {R7.TRAINED.enter_strength:.2f}` covers "
          f"{strength_coverage:.2%} of them, and `energy <= 0.35` -- R7c's "
          f"tightest cell -- covers {r7c_tightest:.2%}")

    rows: dict = {}

    # The control: R7b's published row, on this harness's own bars.
    reference_gate = (energy <= R.REFERENCE_CEILING).to_numpy()
    rows[f"energy <= {R.REFERENCE_CEILING:.2f} (R7b, reference)"] = {
        "kind": "reference",
        "energy_value": R.REFERENCE_CEILING,
        "training_coverage": R.coverage_of(reference_gate, train),
        "rates": R.lift_rows(reference_gate, defined, frame.index, labels, halves),
    }

    for target in R.ENTER_COVERAGE_TARGETS:
        cell = R.cell_for(energy, defined, halves, target)
        gate = (energy <= cell["enter"]["energy_value"]).to_numpy()
        rows[R.label_of(target)] = {
            "kind": "declared",
            "target_coverage": target,
            "cell": cell,
            "energy_value": cell["enter"]["energy_value"],
            "training_coverage": R.coverage_of(gate, train),
            "rates": R.lift_rows(gate, defined, frame.index, labels, halves),
        }

    print(f"\n  {'gate':>34} {'energy':>8} {'cover':>7} {'H':>4} {'train rate':>11} "
          f"{'test rate':>10} {'train lift':>11} {'test lift':>10}")
    for label, row in rows.items():
        for horizon in R7.HORIZONS:
            table = row["rates"][horizon]
            print(f"  {label:>34} {row['energy_value']:>8.4f} "
                  f"{row['training_coverage']:>7.1%} {horizon:>4} "
                  f"{table['train']['inside_rate']:>11.4f} "
                  f"{table['test']['inside_rate']:>10.4f} "
                  f"{table['train']['lift_pp']:>+11.2f} "
                  f"{table['test']['lift_pp']:>+10.2f}")

    return {
        "bars": len(df),
        "split_at": str(halves.timestamp),
        "measurable_bars": int(defined.sum()),
        "measurable_training_bars": int(train.sum()),
        "strength_gate_training_coverage": strength_coverage,
        "r7c_tightest_cell_training_coverage": r7c_tightest,
        "trend_boundaries": {str(h): cuts[h] for h in cuts},
        "gates": rows,
    }


def derivation_table(frames: dict) -> None:
    """The derived values beside their realised coverages, so a reader can check.

    This is the table M29 asks for. The same coverage target gives a different
    `energy` value on each instrument, which is the whole point -- a coverage is
    portable and a threshold is not (M18).
    """
    print(f"\n{'-' * 78}\nTHE DERIVATION -- coverage declared, `energy` derived "
          f"(training half, per frame)")
    print(f"\n{'frame':>6} {'target':>8} {'enter':>9} {'realised':>9} "
          f"{'exit target':>12} {'exit':>9} {'realised':>9}")
    for name, payload in frames.items():
        for row in payload["gates"].values():
            if row["kind"] != "declared":
                continue
            enter, exit_ = row["cell"]["enter"], row["cell"]["exit"]
            print(f"{name:>6} {enter['target_coverage']:>8.0%} "
                  f"{enter['energy_value']:>9.4f} {enter['realised_coverage']:>9.2%} "
                  f"{exit_['target_coverage']:>12.0%} {exit_['energy_value']:>9.4f} "
                  f"{exit_['realised_coverage']:>9.2%}")


def control(frames: dict) -> dict:
    """R7b's four published lifts, re-measured on this harness's own bars.

    This is the control that gates the phase: the tightened numbers are only
    comparable with the loose one if the loose one reproduces here. The
    selectivity of the gate being replaced is reported beside it as context --
    see the module docstring for which denominator M29's digits used and why the
    difference decides nothing.
    """
    print(f"\n{'-' * 78}\nCONTROL -- before any tightened number is read")
    reference = f"energy <= {R.REFERENCE_CEILING:.2f} (R7b, reference)"
    measured = {}
    ok = True
    for name, payload in frames.items():
        table = payload["gates"][reference]["rates"][R.HORIZON]
        for which in ("train", "test"):
            value = table[which]["lift_pp"]
            want = R.REFERENCE_LIFTS_PP[(name, which)]
            same = round(value, 2) == want
            ok &= same
            measured[f"{name} {which}"] = value
            print(f"  {name} {which:>5}: measured {value:>+6.2f} pp, "
                  f"published {want:>+6.2f} pp  {R.mark(same)}")
    mean = float(np.mean(list(measured.values())))
    mean_ok = round(mean, 2) == R.REFERENCE_MEAN_PP
    ok &= mean_ok
    print(f"  mean of the four: {mean:+.2f} pp, published "
          f"{R.REFERENCE_MEAN_PP:+.2f} pp  {R.mark(mean_ok)}")

    print("\n  context, and it gates nothing -- the selectivity being matched:")
    for name, payload in frames.items():
        print(f"    {name} `strength >= 0.80` covers "
              f"{payload['strength_gate_training_coverage']:.2%} of its measurable "
              f"training bars, so the declared 21% target is R5's own gate to "
              f"within a point")
    print(f"    M29 prints {R.R5_GATE_COVERAGE:.1%} for BTC's, and "
          f"{R.M29_ENERGY_COVERAGE:.1%} for `energy <= 0.35` against "
          f"{frames['BTC']['r7c_tightest_cell_training_coverage']:.2%} measured "
          f"here; see the module docstring")
    return {
        "reference_lifts_pp": measured,
        "reference_mean_pp": mean,
        "r5_gate_training_coverage": {
            name: payload["strength_gate_training_coverage"]
            for name, payload in frames.items()
        },
        "m29_published_pair": [R.R5_GATE_COVERAGE, R.M29_ENERGY_COVERAGE],
        "m29_pair_reproduces": False,
        "passes": bool(ok),
    }


def verdict(frames: dict) -> dict:
    """The declared bar, at H=30, and the same table at the context horizons."""
    out: dict = {}
    for horizon in R7.HORIZONS:
        for target in R.ENTER_COVERAGE_TARGETS:
            label = R.label_of(target)
            lifts = {
                f"{name} {which}": frames[name]["gates"][label]["rates"][horizon][which][
                    "lift_pp"
                ]
                for name in frames
                for which in ("train", "test")
            }
            values = list(lifts.values())
            out[f"{target:.2f} H={horizon}"] = {
                "target_coverage": target,
                "horizon": horizon,
                "lifts_pp": lifts,
                "all_positive": all(value > 0.0 for value in values),
                "mean_pp": float(np.mean(values)),
                "clears": all(value > 0.0 for value in values)
                and float(np.mean(values)) >= R.LIFT_BAR_PP,
                "declared": horizon == R.HORIZON,
            }
    return out


def report(verdicts: dict) -> bool:
    print(f"\n{'=' * 78}\nTHE FIRST KILL SWITCH -- declared: at a coverage target the "
          f"H={R.HORIZON} lift is")
    print(f"positive in all four instrument-halves AND their mean is "
          f">= {R.LIFT_BAR_PP:+.1f} pp.")
    declared = {key: row for key, row in verdicts.items() if row["declared"]}
    print(f"\n{'target':>8} {'BTC train':>10} {'BTC test':>10} {'ETH train':>10} "
          f"{'ETH test':>10} {'mean':>8} {'4/4 +ve':>8} {'verdict':>8}")
    for row in declared.values():
        lifts = row["lifts_pp"]
        print(f"{row['target_coverage']:>8.0%} "
              f"{lifts['BTC train']:>+10.2f} {lifts['BTC test']:>+10.2f} "
              f"{lifts['ETH train']:>+10.2f} {lifts['ETH test']:>+10.2f} "
              f"{row['mean_pp']:>+8.2f} {str(row['all_positive']):>8} "
              f"{R.mark(row['clears']):>8}")
    cleared = [row["target_coverage"] for row in declared.values() if row["clears"]]
    print(f"\n  {len(cleared)}/{len(declared)} coverage targets clear at "
          f"H={R.HORIZON}: {[f'{t:.0%}' for t in cleared] or 'none'}")
    context = [
        key for key, row in verdicts.items() if not row["declared"] and row["clears"]
    ]
    print(f"  (context, not the verdict: {len(context)} of "
          f"{len(verdicts) - len(declared)} target-horizons clear at H=6 or H=90: "
          f"{context or 'none'})")
    return bool(cleared)


def embargo_check(frames: dict, sources: dict) -> dict:
    """The declared verdict, re-measured with the split-crossing labels dropped.

    Not asked for by the plan, and it changes no declared number: it is R7b's
    check, which exists because ``forward_efficiency_ratio`` reaches
    ``close[t+1+H]``, so the training half's last ``H+1`` rows carry a label
    built partly from test-half prices. R7 measured that on ICs (<= 0.0052, no
    verdict moved) and R7b re-measured it on *rates*, which is what this phase's
    verdict is, so R7's number does not transfer and R7b's does not either --
    the gates are different.

    The **gates themselves do not move**: an ``energy`` value derived from a
    coverage target reads no forward return at all, so only the label changes
    here. The crossing rows are dropped from both the tercile that sets the
    label and the rates computed against it.
    """
    print(f"\n{'-' * 78}\nEMBARGO RE-CHECK -- H={R.HORIZON}, the crossing labels dropped")
    crossing = R.HORIZON + 1
    moved: dict = {}
    for name, (df, halves) in sources.items():
        frame, defined = R.machine_inputs(df)
        er = R7.forward_er(df["close"], horizon=R.HORIZON)
        embargoed = er.copy()
        embargoed.iloc[halves.split - crossing : halves.split] = float("nan")
        dropped = int(er.iloc[halves.split - crossing : halves.split].notna().sum())
        print(f"  {name}: {dropped} of {halves.split} training rows have a forward "
              f"window crossing the split")
        for series, tag in ((er, "as measured"), (embargoed, "embargoed")):
            labels, _ = R7.trend_label(series, halves)
            for label, row in frames[name]["gates"].items():
                if row["kind"] != "declared":
                    continue
                gate = (frame["energy"] <= row["energy_value"]).to_numpy()
                table = R7.rate_table(
                    R7.verdict_of(gate, frame.index, defined), labels, halves
                )
                for which in ("train", "test"):
                    moved.setdefault((name, label, which), {})[tag] = table[which]["lift_pp"]

    worst = max(abs(row["as measured"] - row["embargoed"]) for row in moved.values())
    embargoed_verdicts = {}
    for target in R.ENTER_COVERAGE_TARGETS:
        label = R.label_of(target)
        values = [
            row["embargoed"] for key, row in moved.items() if key[1] == label
        ]
        embargoed_verdicts[f"{target:.2f}"] = {
            "lifts_pp": values,
            "mean_pp": float(np.mean(values)),
            "clears": all(value > 0.0 for value in values)
            and float(np.mean(values)) >= R.LIFT_BAR_PP,
        }
    print(f"  largest lift change across all {len(moved)} instrument-half cells: "
          f"{worst:.2f} pp against a {R.LIFT_BAR_PP:.1f} pp bar")
    for key, row in embargoed_verdicts.items():
        print(f"  embargoed {float(key):.0%}: mean {row['mean_pp']:+.2f} pp  "
              f"{R.mark(row['clears'])}")
    unchanged = all(row["clears"] for row in embargoed_verdicts.values())
    print(f"  {R.mark(unchanged)}  the verdict is unchanged under the embargo")
    return {
        "largest_lift_change_pp": worst,
        "embargoed_verdicts": embargoed_verdicts,
        "verdict_unchanged": unchanged,
    }


def main() -> None:
    started = time.time()

    btc, _ = R7.load_frame()
    eth, _ = R7.load_eth_frame()
    sources = {"BTC": (btc, R7.halves_of(btc)), "ETH": (eth, R7.eth_halves(eth))}
    frames = {
        name: measure(f"{name}/USDT perp 4h", df, halves)
        for name, (df, halves) in sources.items()
    }

    derivation_table(frames)
    controls = control(frames)
    assert controls["passes"], (
        "the harness does not reproduce R7b's published lifts or M29's coverage"
    )

    verdicts = verdict(frames)
    cleared = report(verdicts)
    embargo = embargo_check(frames, sources)

    print("\n" + "=" * 78)
    if cleared:
        print("THE FIRST KILL SWITCH DID NOT FIRE at every target -- the phase "
              "continues to selection.")
    else:
        print("THE FIRST KILL SWITCH FIRED at every declared coverage target.")
        print("Per the pre-registration: the lift was a property of the loose "
              "threshold, not of `energy`.")
        print("Steps 2-6 do not run. THE SOL HOLDOUT IS NOT SPENT.")

    R.write("step1_lift.json", {
        "declared_horizon": R.HORIZON,
        "lift_bar_pp": R.LIFT_BAR_PP,
        "enter_coverage_targets": list(R.ENTER_COVERAGE_TARGETS),
        "exit_coverage_multiple": R.EXIT_COVERAGE_MULTIPLE,
        "controls": controls,
        "frames": frames,
        "verdicts": verdicts,
        "embargo": embargo,
        "any_target_clears": cleared,
    })
    print(f"\nelapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
