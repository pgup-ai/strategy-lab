"""Step 2 (plan §1): the incumbent, scored as a chop detector.

``COMPRESSION`` is the machine's answer to "is this chop" and has never been
scored as an answer to that question -- only as one input to a traded result.
Here it is scored directly: the base rate of "trend" (top tercile of forward ER,
boundary from the training half) over all bars, against the rate *inside*
``COMPRESSION``, in both halves, at all three horizons.

``RIDING`` and ``CONFIRMED`` get the same treatment, because if the machine's
states carry chop information the **ordering across states** is where it shows.
All six states are printed; the three the plan names carry the verdict.

Two configurations, both already published, neither re-derived (M22): R5's
trained cell is the primary, and the R4 default is the secondary -- reported
because a verdict that held for only one of the two configurations would be a
statement about a cell rather than about the machine.

A state is read only from the bar the strategy's own ``warmup_bars`` allows, so
this scores the states a backtest would actually have traded.

**Declared threshold:** the incumbent detects chop if the trend rate inside
``COMPRESSION`` is at least 10 pp *below* the base rate, in both halves.
"""

from __future__ import annotations

import time

import pandas as pd
import r7lib as R
from strategy_lab.state.machine import MarketState

STATES = list(MarketState)
HEADLINE = (MarketState.COMPRESSION, MarketState.CONFIRMED, MarketState.RIDING)


def main() -> None:
    started = time.time()
    df, _ = R.load_frame()
    halves = R.halves_of(df)
    print(f"frame: {len(df)} bars; split {halves.timestamp} "
          f"(train {halves.split} / test {len(df) - halves.split})")

    labels = {}
    for horizon in R.HORIZONS:
        label, cuts = R.trend_label(R.forward_er(df["close"], horizon=horizon), halves)
        labels[horizon] = label
        print(f"H={horizon}: trend is ER > {cuts[1]:.4f} (training-half top tercile)")

    payload = {}
    for config_name, machine in (("trained", R.TRAINED), ("default", R.DEFAULT)):
        warmup = R.strategy_warmup(machine)
        states = R.states_of(df, machine)
        # Before warmup the machine has not forgotten where it started, so those
        # bars are not the states any backtest traded. Dropped rather than
        # scored: a rate over bars the strategy could not act on is a number
        # about the harness.
        live = pd.Series(False, index=df.index)
        live.iloc[warmup:] = True
        occupancy = states[live.to_numpy()].value_counts()

        print(f"\n=== {config_name} machine: "
              f"enter={machine.enter_strength:.4f} exit={machine.exit_strength:.4f} "
              f"dwell={machine.min_dwell} cool={machine.cooldown}; warmup {warmup} "
              f"({len(df) - warmup} live bars) ===")
        print("occupancy: " + "  ".join(
            f"{state.value} {int(occupancy.get(state, 0))}" for state in STATES
        ))

        rows = {}
        for horizon in R.HORIZONS:
            label = labels[horizon]
            print(f"\nH={horizon} -- trend rate inside each state, against the base rate")
            print(f"{'state':>13} {'bars':>7} {'cover':>6} "
                  f"{'train base':>11} {'train in':>9} {'train out':>10} {'train lift':>11} "
                  f"{'test base':>10} {'test in':>8} {'test out':>9} {'test lift':>10}")
            for state in STATES:
                verdict = R.verdict_of(
                    (states == state).to_numpy(), df.index, live.to_numpy()
                )
                table = R.rate_table(verdict, label, halves)
                # The rate on the complement, because a verdict covering 84% of
                # bars is arithmetically close to the base rate whatever it
                # knows. The pair (inside, outside) is what discriminates; the
                # lift alone understates a wide verdict and flatters a narrow one.
                outside = R.rate_table(
                    R.verdict_of(
                        ((states != state) & live).to_numpy(), df.index, live.to_numpy()
                    ),
                    label,
                    halves,
                )
                for which in ("full", "train", "test"):
                    table[which]["outside_rate"] = outside[which]["inside_rate"]
                    table[which]["separation_pp"] = 100.0 * (
                        table[which]["inside_rate"] - outside[which]["inside_rate"]
                    )
                rows.setdefault(state.value, {})[horizon] = table
                flag = " <-" if state in HEADLINE else ""
                print(f"{state.value:>13} {table['full']['n_inside']:>7} "
                      f"{table['full']['coverage']:>6.1%} "
                      f"{table['train']['base_rate']:>11.4f} "
                      f"{table['train']['inside_rate']:>9.4f} "
                      f"{table['train']['outside_rate']:>10.4f} "
                      f"{table['train']['lift_pp']:>+11.2f} "
                      f"{table['test']['base_rate']:>10.4f} "
                      f"{table['test']['inside_rate']:>8.4f} "
                      f"{table['test']['outside_rate']:>9.4f} "
                      f"{table['test']['lift_pp']:>+10.2f}{flag}")
        payload[config_name] = {
            "warmup_bars": warmup,
            "occupancy": {state.value: int(occupancy.get(state, 0)) for state in STATES},
            "rates": rows,
        }

    # ---- the declared threshold ----------------------------------------
    print("\n" + "=" * 78)
    print("DECLARED THRESHOLD -- the incumbent COMPRESSION detects chop if its "
          "trend rate is")
    print(f"at least {R.INCUMBENT_BAR_PP:.0f} pp BELOW the base rate, in both halves.")
    print(f"\n{'config':>9} {'H':>4} {'train lift pp':>14} {'test lift pp':>13} "
          f"{'verdict':>9}")
    verdicts = {}
    for config_name in payload:
        for horizon in R.HORIZONS:
            table = payload[config_name]["rates"]["compression"][horizon]
            train_pp, test_pp = table["train"]["lift_pp"], table["test"]["lift_pp"]
            passed = train_pp <= -R.INCUMBENT_BAR_PP and test_pp <= -R.INCUMBENT_BAR_PP
            verdicts[f"COMPRESSION {config_name} H={horizon}"] = passed
            print(f"{config_name:>9} {horizon:>4} {train_pp:>+14.2f} {test_pp:>+13.2f} "
                  f"{R.mark(passed):>9}")

    cleared = [name for name, ok in verdicts.items() if ok]
    print(f"\nCOMPRESSION clears in {len(cleared)}/{len(verdicts)} "
          f"config x horizon cells: {cleared or 'none'}")

    # The ordering across states is the second question, and it is answered
    # whether or not the threshold clears.
    print("\nORDERING -- does trend rate rise along the lifecycle? "
          "(test half, trained machine)")
    for horizon in R.HORIZONS:
        order = [
            (state.value,
             payload["trained"]["rates"][state.value][horizon]["test"]["inside_rate"])
            for state in STATES
        ]
        readable = "  ".join(f"{name}={rate:.3f}" for name, rate in order)
        compression = dict(order)["compression"]
        riding = dict(order)["riding"]
        print(f"  H={horizon}: {readable}")
        print(f"           riding - compression = {100 * (riding - compression):+.2f} pp")

    R.write("step2_incumbent.json", {"verdicts": verdicts, "configs": payload})
    print(f"\nelapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
