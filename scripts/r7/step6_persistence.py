"""Step 6 (plan §5): persistence of every chop verdict this phase looked at.

"A detector that flips every other bar is useless at any IC, because a position
cannot follow it -- and turnover is the one thing R5's cost stress has
repeatedly turned on."

**The declared population for this step is empty.** §5 says "for every candidate
that clears §Declared thresholds", and nothing cleared: no feature reached
|IC| 0.10 at H = 30, ``COMPRESSION`` did not reach a 10 pp deficit, and the
composite did not beat its components. So the table below is *not* a threshold
evaluation of declared candidates -- it is the same measurement applied to the
verdicts R7 did examine, reported because the persistence question is exactly
what a better **estimator** over these same features (R8) would inherit, and
because a phase that measures nothing about its own null result leaves the next
one to re-measure it.

Two of the rows are constructions R7 chose rather than ones the plan declared,
and they are marked: the raw ``compression`` feature binarized at its
**training-half** top tercile, and ``energy`` at its bottom tercile. They are
here because ``compression``/``energy`` are the near-miss of §2 and the
question "could a position have followed it" is answerable now and not later.
Their rate metrics are printed beside the state's, so the feature and the state
named after it can be read against each other.

**Declared threshold:** a candidate is usable if its median run length is
>= 6 bars (one day).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import r7lib as R
from strategy_lab.state.machine import MarketState


def tercile_verdict(values: pd.Series, halves: R.Halves, *, top: bool) -> pd.Series:
    """Binarize a feature at its training-half tercile boundary.

    Same rule the label uses, for the same reason: a boundary the test half
    helped set would score a test-half rate against its own answer.
    """
    train = values[halves.mask(values.index, "train")].dropna()
    low, high = (float(x) for x in train.quantile([1 / 3, 2 / 3]))
    defined = values.notna().to_numpy()
    mask = (values > high) if top else (values < low)
    return R.verdict_of(mask.to_numpy(), values.index, defined)


def main() -> None:
    started = time.time()
    df, _ = R.load_frame()
    halves = R.halves_of(df)
    print(f"frame: {len(df)} bars; split {halves.timestamp}")

    verdicts: dict[str, tuple[pd.Series, bool]] = {}

    for config_name, machine in (("trained", R.TRAINED), ("default", R.DEFAULT)):
        warmup = R.strategy_warmup(machine)
        states = R.states_of(df, machine)
        live = np.zeros(len(df), dtype=bool)
        live[warmup:] = True
        for state in (MarketState.COMPRESSION, MarketState.CONFIRMED, MarketState.RIDING):
            verdicts[f"{state.value} ({config_name})"] = (
                R.verdict_of((states == state).to_numpy(), df.index, live),
                True,
            )

    frame, _ = R.machine_frame(df)
    defined = (frame["strength"].notna() & frame["direction"].notna()).to_numpy()
    strength_gate = (frame["strength"] >= R.TRAINED.enter_strength).to_numpy()
    direction_gate = (frame["direction"].abs() >= R.TRAINED.direction_floor).to_numpy()
    verdicts["composite gate (strength & direction)"] = (
        R.verdict_of(strength_gate & direction_gate, df.index, defined), True
    )
    verdicts["strength >= 0.80 (rank) alone"] = (
        R.verdict_of(strength_gate, df.index, defined), True
    )

    # The near-miss of §2, binarized -- and named by the side it actually
    # predicts rather than by the word it carries. IC(compression, ER) is
    # **positive** (+0.0906 at H = 30), so high compression precedes *cleaner*
    # forward moves: the chop side of this feature is its bottom tercile, which
    # is the top tercile of ``energy``. ``compression = 1 - energy`` exactly, so
    # the two rows below are one measurement with two names and agree to the
    # digit; both are printed because the charter reasons about the pair
    # separately and a reader checking one against the other should see them
    # land in the same place.
    columns = R.feature_columns(df)
    verdicts["energy top tercile = the chop side (not declared)"] = (
        tercile_verdict(columns["energy"], halves, top=True), False
    )
    verdicts["compression bottom tercile = same set (not declared)"] = (
        tercile_verdict(columns["compression"], halves, top=False), False
    )
    verdicts["compression top tercile = the trend side (not declared)"] = (
        tercile_verdict(columns["compression"], halves, top=True), False
    )

    print("\nrun-length distribution and lag-1 autocorrelation of each verdict")
    print(f"{'verdict':>40} {'half':>6} {'share':>7} {'runs':>6} {'median':>7} "
          f"{'mean':>7} {'p25':>6} {'p75':>7} {'max':>7} {'AC(1)':>8}")
    payload = {}
    for name, (verdict, _declared) in verdicts.items():
        payload[name] = {}
        for which in ("train", "test"):
            row = R.persistence_row(verdict, halves, which)
            payload[name][which] = row
            print(f"{name if which == 'train' else '':>40} {which:>6} "
                  f"{row['share_of_bars']:>7.1%} {row['runs']:>6} "
                  f"{row['median_run']:>7.1f} {row['mean_run']:>7.1f} "
                  f"{row['p25_run']:>6.1f} {row['p75_run']:>7.1f} "
                  f"{row['max_run']:>7.0f} {row['ac1']:>+8.4f}")

    # The two feature verdicts get the rate metric too, so the feature and the
    # state named after it can be compared directly rather than by eye.
    print("\nthe feature verdicts as chop detectors, for comparison with "
          "COMPRESSION in step 2")
    print("A chop detector wants a NEGATIVE lift: fewer trend bars inside the "
          "verdict than outside.")
    print(f"{'verdict':>48} {'H':>4} {'cover':>7} {'train lift pp':>14} "
          f"{'test lift pp':>13}")
    rates = {}
    for name in ("energy top tercile = the chop side (not declared)",
                 "compression top tercile = the trend side (not declared)"):
        verdict = verdicts[name][0]
        rates[name] = {}
        for horizon in R.HORIZONS:
            label, _ = R.trend_label(R.forward_er(df["close"], horizon=horizon), halves)
            table = R.rate_table(verdict, label, halves)
            rates[name][horizon] = table
            print(f"{name if horizon == R.HORIZONS[0] else '':>48} {horizon:>4} "
                  f"{table['full']['coverage']:>7.1%} "
                  f"{table['train']['lift_pp']:>+14.2f} "
                  f"{table['test']['lift_pp']:>+13.2f}")

    print("\n" + "=" * 78)
    print(f"DECLARED THRESHOLD -- a candidate is usable if its median run length "
          f"is >= {R.RUN_LENGTH_BAR:.0f} bars")
    print("(one day). The declared population is EMPTY: nothing cleared §2, §1 "
          "or §3, so no")
    print("row below is a declared candidate. Evaluated anyway, as context for R8.")
    print(f"\n{'verdict':>40} {'train median':>13} {'test median':>12} "
          f"{'would clear':>12}")
    for name, rows in payload.items():
        train, test = rows["train"]["median_run"], rows["test"]["median_run"]
        would = train >= R.RUN_LENGTH_BAR and test >= R.RUN_LENGTH_BAR
        print(f"{name:>40} {train:>13.1f} {test:>12.1f} {R.mark(would):>12}")

    R.write("step6_persistence.json", {
        "declared_population_empty": True,
        "persistence": payload,
        "feature_verdict_rates": rates,
    })
    print(f"\nelapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
