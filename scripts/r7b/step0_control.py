"""Step 0 (plan §0): the three controls, before any R7b number is read.

1. **The no-op is a no-op.** ``energy_ceiling = 1.0`` reproduces R5's published
   test-half row on BTC -- +15.45% / +0.896 / 4.67% / 73 trades, funding column
   attached (M20). If the no-op is not a no-op, nothing after it means anything.
2. **Byte-identity.** ``state_machine_v1`` and ``state_machine_v2``'s published
   figures are unchanged against ``main``, the constraint that has held since
   R2. Checked rather than assumed: this script is run **once on ``main``**
   with ``R7B_BASELINE=1`` to write ``baseline_main.json``, then again on the
   branch, and every figure is compared to the last bit rather than to a
   tolerance. A tolerance would hide exactly the drift this is here to catch.
3. The safety suites are run by ``pytest``, not from here -- see the phase
   report. This script covers what pytest cannot: the published *figures*,
   which need Postgres and thirty seconds of engine time.

Both v1 (``from_signals``) and v2 (``from_orders``) are run, because the change
is in the machine both of them hold and a no-op on one path is not evidence
about the other.

Every engine run here is ``tests/test_state_machine_gate.py``'s own
``evaluate``, reached through ``r7lib.gate_run`` -- the same call R7 and R9
made, on the bars R5 measured.
"""

from __future__ import annotations

import os
import shutil
import time

import r7blib as R
from strategy_lab.backtests.costs import window_end
from strategy_lab.backtests.exposure_engine import run_exposure_backtest
from strategy_lab.state.machine import StateMachine
from strategy_lab.strategies.exposure_registry import get_exposure_strategy

BASELINE_FILE = "baseline_main.json"
CAPTURING = os.environ.get("R7B_BASELINE") == "1"

# §9.2's v1 rows and §9.3's v2 rows, to the digits the charter prints. The count
# column is each path's own: v1 reports **round trips** and v2 reports **fills**,
# which is §9.3's own distinction -- v2 trained is 74 round trips over 276 fills,
# and printing 74 beside an order count would read as a 3.7x discrepancy where
# there is none.
PUBLISHED = {
    "v1 trained": {"net_return_pct": 15.45, "sharpe": 0.896, "max_drawdown_pct": 4.67,
                   "count": 73},
    "v1 default": {"net_return_pct": 15.52, "sharpe": 0.746, "max_drawdown_pct": 7.11,
                   "count": 153},
    "v2 trained": {"net_return_pct": 25.89, "sharpe": 0.842, "max_drawdown_pct": 10.85,
                   "count": 276},
    "v2 default": {"net_return_pct": 36.70, "sharpe": 0.913, "max_drawdown_pct": 12.55,
                   "count": 416},
}


def exposure_row(machine: StateMachine, frame, first_tradeable: int) -> dict:
    """One ``state_machine_v2`` run over the test half, at R6's own settings.

    Rebuilt here rather than read off a stored artifact because R6 published a
    table and not a run directory. The statistics are the engine's own curve
    reduced by ``r7lib``'s Sharpe -- the same estimator every other number in
    this phase uses -- so the digits below are comparable with §9.3's to the
    precision §9.3 prints, and comparable with *themselves across the change*
    exactly.
    """
    from dataclasses import replace

    df, rates = frame
    strategy = replace(get_exposure_strategy("state_machine_v2"), machine=machine)
    window = df.iloc[first_tradeable - strategy.warmup_bars :]
    funding = rates[
        (rates.index >= window.index[0]) & (rates.index < window_end(window.index))
    ]
    result = run_exposure_backtest(
        df=window,
        strategy=strategy,
        identity=R.BTC_IDENTITY,
        funding=funding,
    )
    # Whole window including the flat warmup, which is the slice ``run_backtest``
    # reduces for v1 and therefore the one §9.3's v1 column reproduces. The
    # warmup adds no drawdown and no return, so only the Sharpe reads it -- and
    # scoring v2 over a different slice than v1 is precisely the M21 artifact.
    equity = result.equity
    peak = equity.cummax()
    return {
        "net_return_pct": 100.0 * (float(equity.iloc[-1]) / R.CASH - 1.0),
        "sharpe": R.R7.sharpe_of(equity),
        "max_drawdown_pct": 100.0 * float((1.0 - equity / peak).max()),
        "orders": int(result.order_count),
        "fees_paid": float(result.fees_paid),
        "funding_paid": float(result.funding_paid),
        "window_bars": int(len(equity)),
        "tradeable_bars": int(len(equity) - strategy.warmup_bars),
    }


def main() -> None:
    started = time.time()
    frame = R.R7.load_frame()
    df, _ = frame
    split = R.R7.split_index(df)
    has_ceiling = "energy_ceiling" in StateMachine.__dataclass_fields__
    print(f"frame: {len(df)} bars; test half from bar {split} ({df.index[split]})")
    print(f"StateMachine has energy_ceiling: {has_ceiling}"
          f"{'  -- capturing the main baseline' if CAPTURING else ''}")

    root = R.OUT / "control"
    rows: dict[str, dict] = {}

    for label, machine in (("v1 trained", R.R7.TRAINED), ("v1 default", R.R7.DEFAULT)):
        strategy = R.R7.gate.machine(machine)
        row = R.R7.gate_run(
            strategy, frame, root / label.replace(" ", "_"),
            first_tradeable=split, stop=len(df),
        )
        rows[label] = R.slim(row)

    for label, machine in (("v2 trained", R.R7.TRAINED), ("v2 default", R.R7.DEFAULT)):
        rows[label] = exposure_row(machine, frame, split)

    print(f"\n{'run':>12} {'net %':>10} {'sharpe':>9} {'maxDD %':>9} {'count':>7}"
          f"{'   published (charter §9.2 / §9.3)':<34}")
    for label, row in rows.items():
        want = PUBLISHED[label]
        count = row.get("trades", row.get("orders"))
        print(f"{label:>12} {row['net_return_pct']:>+10.4f} {row['sharpe']:>+9.4f} "
              f"{row['max_drawdown_pct']:>9.4f} {count:>7}   "
              f"{want['net_return_pct']:>+8.2f} / {want['sharpe']:>+.3f} / "
              f"{want['max_drawdown_pct']:.2f} / {want['count']}")

    if CAPTURING:
        R.write(BASELINE_FILE, rows)
        print(f"\nbaseline written to {R.OUT / BASELINE_FILE} -- re-run on the "
              f"branch without R7B_BASELINE to compare")
        shutil.rmtree(root, ignore_errors=True)
        print(f"\nelapsed {time.time() - started:.1f}s")
        return

    print("\n" + "=" * 78)
    print("CONTROL 1 -- the no-op is a no-op")
    assert has_ceiling, "this branch has no energy_ceiling; run the change first"
    noop = R.R7.gate_run(
        R.strategy_for(R.CONTROL_CEILING), frame, root / "noop",
        first_tradeable=split, stop=len(df),
    )
    want = R.R5_PUBLISHED
    control_1 = (
        round(noop["net_return_pct"], 2) == want["net_return_pct"]
        and round(noop["sharpe"], 3) == want["sharpe"]
        and round(noop["max_drawdown_pct"], 2) == want["max_drawdown_pct"]
        and noop["trades"] == want["trades"]
    )
    print(f"  energy_ceiling=1.00: {noop['net_return_pct']:+.4f}% / "
          f"{noop['sharpe']:+.4f} / {noop['max_drawdown_pct']:.4f}% / "
          f"{noop['trades']} trades   crowding_measured={noop['crowding_measured']}")
    print(f"  R5 published:        {want['net_return_pct']:+.2f}% / "
          f"{want['sharpe']:+.3f} / {want['max_drawdown_pct']:.2f}% / "
          f"{want['trades']} trades")
    print(f"  {R.mark(control_1)}")
    assert noop["crowding_measured"], "the frame lost its funding column; M20 is the point"

    print("\nCONTROL 2 -- byte-identity against main")
    baseline = R.read_if_present(BASELINE_FILE)
    if baseline is None:
        print(f"  SKIP  no {BASELINE_FILE}; run this on main with R7B_BASELINE=1 first")
        control_2 = None
    else:
        control_2 = True
        for label, row in rows.items():
            was = baseline[label]
            for key in ("net_return_pct", "sharpe", "max_drawdown_pct"):
                same = repr(row[key]) == repr(was[key])
                control_2 &= same
                if not same:
                    print(f"  MOVED {label} {key}: {was[key]!r} -> {row[key]!r}")
            count = "trades" if "trades" in row else "orders"
            same = row[count] == was[count]
            control_2 &= same
            if not same:
                print(f"  MOVED {label} {count}: {was[count]} -> {row[count]}")
            print(f"  {'same' if same else 'MOVED':>5}  {label}: "
                  f"{row['net_return_pct']!r} / {row['sharpe']!r} / "
                  f"{row['max_drawdown_pct']!r} / {row[count]}")
        print(f"  {R.mark(bool(control_2))}  every published v1/v2 figure is "
              f"bit-identical to main's")

    shutil.rmtree(root, ignore_errors=True)
    R.write("step0_control.json", {
        "control_1_noop": control_1,
        "control_2_byte_identity": control_2,
        "noop": R.slim(noop),
        "published_runs": rows,
        "baseline": baseline,
    })
    verdicts = [control_1, control_2]
    print(f"\n{'ALL CONTROLS PASS' if all(verdicts) else 'A CONTROL FAILED -- STOP'}")
    print(f"elapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
