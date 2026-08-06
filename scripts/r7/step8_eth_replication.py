"""Step 8: the ETH replication the plan declared before any BTC figure existed.

"ETH/USDT perp 4h is declared now as the replication, same protocol, run after
BTC and never used to choose anything."

Same label, same horizons, same training-half terciles, same declared
thresholds, on charter §9.4's ETH frame -- 14,650 bars, split 2023-10-31, test
half 6,048 bars identical to BTC's.

**Three of the plan's five parts replicate; §4 does not and is not run here.**
The ``enter_strength`` extension is a question about *R9's ridge*, and R9's
+1.3974 bar is BTC's training-half M21 scalar. Re-running the extension against
ETH would either compare an ETH cell to a BTC bar, or invent an ETH bar the plan
never declared -- and inventing a threshold after the fact is exactly what M23
was written about. §5's persistence is measured on the same verdicts as step 6.

**Three machine configurations, all published, none re-derived** (M22): R5's
BTC-trained cell, the R4 default, and §9.4's own ETH-selected cell.

A null that replicates on a second instrument is a much stronger statement than
a null on one. A null that does *not* replicate is a finding about the
instrument, and would be reported as such.
"""

from __future__ import annotations

import time

import numpy as np
import r7lib as R
from strategy_lab.state.machine import MarketState, StateMachine

# Charter §9.4: the cell R5's own 54-cell protocol selected on ETH's training
# half. Quoted, not re-derived.
ETH_SELECTED = StateMachine(
    enter_strength=0.80, exit_strength=1.0 / 3.0, min_dwell=8, cooldown=8
)
CONFIGS = (
    ("BTC-trained", R.TRAINED),
    ("R4 default", R.DEFAULT),
    ("ETH-selected", ETH_SELECTED),
)


def main() -> None:
    started = time.time()
    df, rates = R.load_eth_frame()
    halves = R.eth_halves(df)
    print(f"ETH frame: {len(df)} bars {df.index[0]} -> {df.index[-1]} "
          f"(§9.4 says {R.ETH_BARS})")
    print(f"funding: {len(rates)} settlements; split {halves.timestamp} "
          f"(train {halves.split} / test {len(df) - halves.split})")
    assert len(df) == R.ETH_BARS, f"ETH frame moved: {len(df)} bars"
    assert len(df) - halves.split == 6_048, "the ETH test half is no longer BTC's size"

    labels = {}
    for horizon in R.HORIZONS:
        label, cuts = R.trend_label(R.forward_er(df["close"], horizon=horizon), halves)
        labels[horizon] = label
        rate = R.rate_table(
            R.verdict_of(np.ones(len(df), dtype=bool), df.index, np.ones(len(df), bool)),
            label, halves,
        )
        print(f"H={horizon}: trend is ER > {cuts[1]:.4f}; base rate "
              f"train {rate['train']['base_rate']:.4f} / "
              f"test {rate['test']['base_rate']:.4f}")

    print("\nPART 2 -- Spearman IC vs forward ER, full (train / test)")
    columns = R.feature_columns(df)
    targets = {h: R.forward_er(df["close"], horizon=h) for h in R.HORIZONS}
    print(f"{'feature':>20} " + " ".join(f"{f'H={h}':>28}" for h in R.HORIZONS))
    feature_rows, feature_verdicts = {}, {}
    for name, values in columns.items():
        measured = values.dropna()
        cells, feature_rows[name] = [], {}
        for horizon in R.HORIZONS:
            table = R.ic_table(measured, targets[horizon], halves, horizon=horizon)
            feature_rows[name][horizon] = table
            cells.append(f"{table['full']['ic']:+.4f}"
                         f"({table['train']['ic']:+.4f}/{table['test']['ic']:+.4f})")
        print(f"{name:>20} " + " ".join(f"{cell:>28}" for cell in cells))
        row = feature_rows[name][30]
        full, first, second = (row["full"]["ic"], row["train"]["ic"], row["test"]["ic"])
        feature_verdicts[name] = bool(
            abs(full) >= R.IC_BAR and first * second > 0
            and min(abs(first), abs(second)) >= R.IC_HALF_BAR
        )

    print("\nPART 1 -- COMPRESSION / CONFIRMED / RIDING as chop detectors")
    print(f"{'config':>13} {'state':>12} {'H':>4} {'cover':>7} "
          f"{'train lift pp':>14} {'test lift pp':>13}")
    incumbent_rows, incumbent_verdicts = {}, {}
    verdict_series = {}
    for config_name, machine in CONFIGS:
        warmup = R.strategy_warmup(machine)
        states = R.states_of(df, machine)
        live = np.zeros(len(df), dtype=bool)
        live[warmup:] = True
        incumbent_rows[config_name] = {}
        for state in (MarketState.COMPRESSION, MarketState.CONFIRMED, MarketState.RIDING):
            verdict = R.verdict_of((states == state).to_numpy(), df.index, live)
            verdict_series[f"{state.value} ({config_name})"] = verdict
            incumbent_rows[config_name][state.value] = {}
            for horizon in R.HORIZONS:
                table = R.rate_table(verdict, labels[horizon], halves)
                incumbent_rows[config_name][state.value][horizon] = table
                print(f"{config_name if horizon == R.HORIZONS[0] else '':>13} "
                      f"{state.value if horizon == R.HORIZONS[0] else '':>12} "
                      f"{horizon:>4} {table['full']['coverage']:>7.1%} "
                      f"{table['train']['lift_pp']:>+14.2f} "
                      f"{table['test']['lift_pp']:>+13.2f}")
                if state is MarketState.COMPRESSION:
                    incumbent_verdicts[f"COMPRESSION {config_name} H={horizon}"] = bool(
                        table["train"]["lift_pp"] <= -R.INCUMBENT_BAR_PP
                        and table["test"]["lift_pp"] <= -R.INCUMBENT_BAR_PP
                    )

    print("\nPART 3 -- the composite gate at R5's trained values")
    frame, crowding_measured = R.machine_frame(df)
    assert crowding_measured, "ETH frame lost its funding column"
    defined = (frame["strength"].notna() & frame["direction"].notna()).to_numpy()
    gates = {
        "strength >= 0.80 (rank)": (frame["strength"] >= R.TRAINED.enter_strength).to_numpy(),
        "|direction| >= 0.10": (frame["direction"].abs() >= R.TRAINED.direction_floor).to_numpy(),
    }
    gates["composite (both)"] = gates["strength >= 0.80 (rank)"] & gates["|direction| >= 0.10"]
    print(f"{'gate':>26} {'H':>4} {'cover':>7} {'train in':>9} {'test in':>8} "
          f"{'train lift':>11} {'test lift':>10}")
    composite_rows, composite_verdicts = {}, {}
    for horizon in R.HORIZONS:
        composite_rows[horizon] = {}
        for name, mask in gates.items():
            table = R.rate_table(R.verdict_of(mask, df.index, defined), labels[horizon], halves)
            composite_rows[horizon][name] = table
            print(f"{name:>26} {horizon:>4} {table['full']['coverage']:>7.1%} "
                  f"{table['train']['inside_rate']:>9.4f} "
                  f"{table['test']['inside_rate']:>8.4f} "
                  f"{table['train']['lift_pp']:>+11.2f} "
                  f"{table['test']['lift_pp']:>+10.2f}")
        composite_verdicts[f"composite H={horizon}"] = bool(all(
            100.0 * (composite_rows[horizon]["composite (both)"][which]["inside_rate"]
                     - composite_rows[horizon][alone][which]["inside_rate"])
            >= R.COMPOSITE_BAR_PP
            for which in ("train", "test")
            for alone in ("strength >= 0.80 (rank)", "|direction| >= 0.10")
        ))

    print("\nPART 5 -- persistence of the same verdicts")
    verdict_series["composite gate"] = R.verdict_of(gates["composite (both)"], df.index, defined)
    print(f"{'verdict':>34} {'half':>6} {'share':>7} {'median':>7} {'AC(1)':>8}")
    persistence = {}
    for name, verdict in verdict_series.items():
        persistence[name] = {}
        for which in ("train", "test"):
            row = R.persistence_row(verdict, halves, which)
            persistence[name][which] = row
            print(f"{name if which == 'train' else '':>34} {which:>6} "
                  f"{row['share_of_bars']:>7.1%} {row['median_run']:>7.1f} "
                  f"{row['ac1']:>+8.4f}")

    print("\n" + "=" * 78)
    print("ETH REPLICATION -- the same declared thresholds")
    for title, verdicts in (
        ("A feature detects chop", feature_verdicts),
        ("COMPRESSION detects chop", incumbent_verdicts),
        ("The composite earns its second input", composite_verdicts),
    ):
        cleared = [name for name, ok in verdicts.items() if ok]
        print(f"  {R.mark(bool(cleared)):>4}  {title}: {len(cleared)}/{len(verdicts)} clear"
              f"{' -- ' + ', '.join(cleared) if cleared else ''}")

    best = max(
        ((name, feature_rows[name][h], h) for name in columns for h in R.HORIZONS),
        key=lambda item: abs(item[1]["full"]["ic"]),
    )
    print(f"\nlargest |IC| anywhere in ETH's table: {best[0]} at H={best[2]}, "
          f"{best[1]['full']['ic']:+.4f} "
          f"({best[1]['train']['ic']:+.4f}/{best[1]['test']['ic']:+.4f})")

    R.write("step8_eth_replication.json", {
        "bars": len(df),
        "split_index": halves.split,
        "feature_verdicts": feature_verdicts,
        "incumbent_verdicts": incumbent_verdicts,
        "composite_verdicts": composite_verdicts,
        "features": feature_rows,
        "incumbent": incumbent_rows,
        "composite": composite_rows,
        "persistence": persistence,
    })
    print(f"\nelapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
