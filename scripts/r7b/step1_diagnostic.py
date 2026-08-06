"""Step 1 (plan §1): the diagnostic gate -- and it runs before any P&L.

R7's **own** composite threshold, reused verbatim: the composite must beat
**each** single-feature gate by >= **5 pp** on the trend rate, **in both
halves**, at **H=30**, on 4h bars. R7 measured ``strength AND |direction|`` at
+0.00 pp against ``strength`` alone; this asks the same question of
``strength AND energy``, at each candidate ceiling.

**If the energy gate fails this, the P&L runs are not read.** The hypothesis is
that a second axis adds chop discrimination, and that claim is settled by the
rate metric rather than by a return -- R7 exists precisely because a P&L number
scores the estimator and the policy laid over it as one thing.

Three things are fixed here rather than after the fact.

- **BTC is the declared primary and ETH is R7's replication.** The plan makes
  BTC primary everywhere it distinguishes them ("Selection happens here"; ETH is
  "Confirmation"), and R7 ran §3 on BTC and replicated it on ETH in step 8. Both
  are reported; the verdict is BTC's.
- **H=30 is the declared horizon.** H=6 and H=90 are printed as context and are
  not part of the verdict, which is what the plan says and also what R7's noise
  arithmetic supports -- at H=90 an IC bar sits inside the noise band.
- **``energy_ceiling = 1.00`` is the control**, where the composite *is* the
  strength gate by construction and the gap is exactly 0.00 pp. It is run
  anyway, because a harness that cannot reproduce a gap it knows the answer to
  is not measuring what it claims.

The label, the terciles and the 5 pp bar are ``scripts/r7``'s, imported rather
than restated. A second implementation of the statistic R7's verdict rests on
would make the two phases incomparable, which is the whole reason §1 reuses R7's
threshold instead of declaring one of its own.
"""

from __future__ import annotations

import time

import r7blib as R

R7 = R.R7
HORIZON = 30


def gates_for(df, ceiling: float) -> tuple[dict, object]:
    """The two single-feature gates and their composite, on one frame.

    ``strength`` is read in **rank space** -- ``rolling_percentile`` over 480
    bars, ``strategies/state_machine_core.py``'s ``RANKED_FEATURES`` -- because
    that is the space ``enter_strength`` is a threshold in. ``energy`` is read
    raw, because it is *already* a rolling percentile over that same window;
    ranking it again would score a threshold no machine ever applied, which is
    the mistake R7's step 4 documents for ``direction``.

    A bar carries a verdict only where **the machine could see it** -- every
    input in ``REQUIRED_COLUMNS`` measurable, which is ``StateMachine.run``'s own
    ``measurable``. Scoring on the narrower "both inputs of *this* gate" set
    instead would move the boundary with the gate: ``energy`` costs 503 warmup
    bars and ``direction`` 1920, so a strength-and-energy composite would be
    scored over 1,345 training bars a strength-and-direction one was not, and
    R7's published rates for the gate they share would not reproduce. Measured:
    they do not -- the strength gate reads 0.2642 on that set against R7's
    published 0.2513 -- and ``main`` below asserts the reproduction, so the
    diagnostic is R7's question on R7's bars.
    """
    frame, crowding_measured = R7.machine_frame(df)
    assert crowding_measured, "the frame lost its funding column; M20 is the whole point"
    strength_rank, energy = frame["strength"], frame["energy"]
    defined = frame.notna().all(axis=1).to_numpy()

    strength_gate = (strength_rank >= R7.TRAINED.enter_strength).to_numpy()
    energy_gate = (energy <= ceiling).to_numpy()
    return {
        "strength >= 0.80 (rank)": strength_gate,
        f"energy <= {ceiling:.2f}": energy_gate,
        "composite (both)": strength_gate & energy_gate,
    }, defined


def measure(label: str, df, halves) -> dict:
    print(f"\n{'=' * 78}\n{label}: {len(df)} bars, split {halves.timestamp}")
    labels = {}
    for horizon in R7.HORIZONS:
        series, cuts = R7.trend_label(R7.forward_er(df["close"], horizon=horizon), halves)
        labels[horizon] = series
        print(f"  H={horizon}: trend is ER > {cuts[1]:.4f}")

    payload: dict = {}
    for ceiling in R.ENERGY_CEILINGS:
        gates, defined = gates_for(df, ceiling)
        payload[f"{ceiling:.2f}"] = {}
        print(f"\n  energy_ceiling = {ceiling:.2f}"
              f"{'   (control: the composite IS the strength gate)' if ceiling >= 1.0 else ''}")
        print(f"  {'gate':>26} {'H':>4} {'cover':>7} {'train rate':>11} "
              f"{'test rate':>10} {'train lift':>11} {'test lift':>10}")
        for name, mask in gates.items():
            rows = {}
            for horizon in R7.HORIZONS:
                table = R7.rate_table(
                    R7.verdict_of(mask, df.index, defined), labels[horizon], halves
                )
                rows[horizon] = table
                print(f"  {name:>26} {horizon:>4} {table['full']['coverage']:>7.1%} "
                      f"{table['train']['inside_rate']:>11.4f} "
                      f"{table['test']['inside_rate']:>10.4f} "
                      f"{table['train']['lift_pp']:>+11.2f} "
                      f"{table['test']['lift_pp']:>+10.2f}")
            payload[f"{ceiling:.2f}"][name] = rows
    return payload


def verdicts_for(payload: dict) -> dict:
    """The declared threshold, applied at H=30 and reported at all three."""
    out = {}
    for ceiling in R.ENERGY_CEILINGS:
        key = f"{ceiling:.2f}"
        rows = payload[key]
        composite = "composite (both)"
        singles = [name for name in rows if name != composite]
        for horizon in R7.HORIZONS:
            gaps = {
                name: {
                    which: 100.0 * (
                        rows[composite][horizon][which]["inside_rate"]
                        - rows[name][horizon][which]["inside_rate"]
                    )
                    for which in ("train", "test")
                }
                for name in singles
            }
            out[f"{key} H={horizon}"] = {
                "gaps_pp": gaps,
                "clears": all(
                    gap >= R7.COMPOSITE_BAR_PP
                    for name in gaps
                    for gap in gaps[name].values()
                ),
                "declared": horizon == HORIZON,
            }
    return out


def report(label: str, verdicts: dict) -> bool:
    print(f"\n{'-' * 78}\n{label} -- declared: the composite beats BOTH single-feature "
          f"gates by >= {R7.COMPOSITE_BAR_PP:.0f} pp,")
    print(f"in both halves, at H={HORIZON}.")
    print(f"\n{'cell':>16} {'vs strength':>24} {'vs energy':>24} {'verdict':>9}")
    declared = {k: v for k, v in verdicts.items() if v["declared"]}
    for key, row in declared.items():
        cells = []
        for name, gaps in row["gaps_pp"].items():
            cells.append(f"{gaps['train']:+.2f} / {gaps['test']:+.2f}")
        print(f"{key:>16} {cells[0]:>24} {cells[1]:>24} "
              f"{R.mark(row['clears']):>9}")
    cleared = [key for key, row in declared.items() if row["clears"]]
    print(f"\n  {len(cleared)}/{len(declared)} cells clear at H={HORIZON}: {cleared or 'none'}")
    other = [k for k, v in verdicts.items() if not v["declared"] and v["clears"]]
    print(f"  (context, not the verdict: {len(other)} of "
          f"{len(verdicts) - len(declared)} cells clear at H=6 or H=90: {other or 'none'})")
    return bool(cleared)


# R7 §3's published row for the gate this phase shares with it, on BTC at H=30.
# The control: a harness that cannot reproduce the number it is extending is
# measuring a different question.
R7_STRENGTH_GATE = {"coverage": 0.217, "train": 0.2513, "test": 0.3385}


def control(payload: dict) -> bool:
    row = payload[f"{R.CONTROL_CEILING:.2f}"]["strength >= 0.80 (rank)"][HORIZON]
    ok = (
        round(row["full"]["coverage"], 3) == R7_STRENGTH_GATE["coverage"]
        and round(row["train"]["inside_rate"], 4) == R7_STRENGTH_GATE["train"]
        and round(row["test"]["inside_rate"], 4) == R7_STRENGTH_GATE["test"]
    )
    print("\nCONTROL -- R7 §3's `strength >= 0.80` row reproduces")
    print(f"  measured  {row['full']['coverage']:.1%} / {row['train']['inside_rate']:.4f} "
          f"/ {row['test']['inside_rate']:.4f}")
    print(f"  published {R7_STRENGTH_GATE['coverage']:.1%} / "
          f"{R7_STRENGTH_GATE['train']:.4f} / {R7_STRENGTH_GATE['test']:.4f}")
    print(f"  {R.mark(ok)}")
    return ok


def embargo_check(label: str, df, halves) -> dict:
    """Plan §5: the embargo, re-checked against the code rather than restated.

    ``forward_efficiency_ratio`` reaches ``close[t+1+H]``, so the training half's
    last ``H+1`` rows carry a label built partly from test-half prices -- R7's
    own caveat 6, which measured the effect on its ICs at <= 0.0052 and found no
    verdict moved. R7b's verdict is a *rate*, not an IC, so R7's measurement
    does not transfer and this re-measures it: the crossing rows are dropped
    from both the tercile that sets the label and the rates computed against it,
    and every H=30 gap is recomputed.

    Nothing else about purging changes. The machine fits nothing, has no labels
    and no forward horizon of its own; one parameter chosen by grid search is a
    selection, not a fitted model. Purging and embargo become mandatory at R8,
    which is where a model with a training set first exists.
    """
    print(f"\n{'-' * 78}\nPLAN §5 -- embargo re-check, {label}, H={HORIZON}")
    crossing = HORIZON + 1
    split_position = int(halves.split)
    er = R7.forward_er(df["close"], horizon=HORIZON)
    embargoed = er.copy()
    embargoed.iloc[split_position - crossing : split_position] = float("nan")
    dropped = int(er.iloc[split_position - crossing : split_position].notna().sum())
    print(f"  {dropped} of {split_position} training rows have a forward window "
          f"crossing the split")

    moved = {}
    for series, name in ((er, "as measured"), (embargoed, "embargoed")):
        labels, _ = R7.trend_label(series, halves)
        for ceiling in R.ENERGY_CEILINGS:
            gates, defined = gates_for(df, ceiling)
            rates = {
                gate: R7.rate_table(
                    R7.verdict_of(mask, df.index, defined), labels, halves
                )
                for gate, mask in gates.items()
            }
            composite = rates["composite (both)"]
            for gate in rates:
                if gate == "composite (both)":
                    continue
                for which in ("train", "test"):
                    gap = 100.0 * (
                        composite[which]["inside_rate"] - rates[gate][which]["inside_rate"]
                    )
                    moved.setdefault((ceiling, gate, which), {})[name] = gap

    worst = max(abs(v["as measured"] - v["embargoed"]) for v in moved.values())
    # The declared verdict is per **cell**, not per gap: a ceiling clears only
    # if the composite beats *both* single gates by the bar in *both* halves.
    # Asking whether any single gap clears would be a different, easier test.
    cleared = [
        ceiling
        for ceiling in R.ENERGY_CEILINGS
        if all(
            gaps["embargoed"] >= R7.COMPOSITE_BAR_PP
            for key, gaps in moved.items()
            if key[0] == ceiling
        )
    ]
    print(f"  largest gap change across all {len(moved)} gaps: {worst:.2f} pp "
          f"against a {R7.COMPOSITE_BAR_PP:.0f} pp bar")
    print(f"  embargoed cells clearing: {cleared or 'none'}")
    print(f"  {R.mark(not cleared)}  the verdict is unchanged under the embargo")
    return {
        "dropped_rows": dropped,
        "largest_gap_change_pp": worst,
        "embargoed_cells_clearing": cleared,
        "verdict_unchanged": not cleared,
    }


def main() -> None:
    started = time.time()

    btc, _ = R7.load_frame()
    btc_payload = measure("BTC/USDT perp 4h (declared primary)", btc, R7.halves_of(btc))
    control_ok = control(btc_payload)
    assert control_ok, "the diagnostic harness does not reproduce R7's own gate"
    btc_verdicts = verdicts_for(btc_payload)

    eth, _ = R7.load_eth_frame()
    eth_payload = measure("ETH/USDT perp 4h (R7's replication)", eth, R7.eth_halves(eth))
    eth_verdicts = verdicts_for(eth_payload)

    btc_clears = report("BTC -- THE VERDICT", btc_verdicts)
    eth_clears = report("ETH -- the replication", eth_verdicts)
    embargo = {
        "btc": embargo_check("BTC", btc, R7.halves_of(btc)),
        "eth": embargo_check("ETH", eth, R7.eth_halves(eth)),
    }

    print("\n" + "=" * 78)
    print(f"THE DIAGNOSTIC GATE {'CLEARS' if btc_clears else 'FAILS'} on the declared "
          f"primary frame.")
    if not btc_clears:
        print("Per the pre-registration: the hypothesis is dead and the P&L is NOT read.")
        print("Steps 2-4 do not run.")
    print(f"(ETH, the replication: {'clears' if eth_clears else 'fails'}.)")

    R.write("step1_diagnostic.json", {
        "declared_horizon": HORIZON,
        "bar_pp": R7.COMPOSITE_BAR_PP,
        "control_reproduces_r7": control_ok,
        "embargo": embargo,
        "btc_clears": btc_clears,
        "eth_clears": eth_clears,
        "btc": {"verdicts": btc_verdicts, "rates": btc_payload},
        "eth": {"verdicts": eth_verdicts, "rates": eth_payload},
    })
    print(f"\nelapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
