"""Step 4 (plan §4): the two burned frames, then the holdout, once.

Order and labelling are both declared, and both matter.

- **BTC's test half and ETH's full frame are burned** for this hypothesis. R7b
  read `energy`'s lift on both test halves and R7d's own step 1 read it again on
  all four instrument-halves, so a P&L here is a number about bars this
  programme has already looked at. They are reported and they are **not
  evidence**.
- **SOL is the only test that counts**, and it is run **once**: whole frame, no
  split, at the coverage target step 2 selected, with **its own `energy` value
  derived on its own bars**. Coverage is what transfers, never the threshold
  (M18) -- deriving SOL's value from BTC's would carry BTC's volatility
  distribution onto a different instrument, which is the failure M18 records.

**The declared holdout threshold**, from the plan's own table: on SOL's full
frame the energy-first machine beats **both** R5's trained cell run unmodified
**and** `donchian` 40/10 on Sharpe net of funding, **and** survives 3x costs with
a positive net return. Both comparators are run here on the identical bars
rather than quoted, since neither has ever been run on this instrument.

**One thing in the plan is underdetermined and is resolved here rather than
quietly.** The grid section says each frame's `energy` value is derived "on each
frame's own training half"; the frames section gives SOL **no split**, and §4
says "its own `energy` value derived on its own bars". For an unsplit frame
those are different sets. Resolved toward the sentence that names SOL -- the
more specific commitment, and R7c's rule for this shape of ambiguity -- so the
holdout's threshold is a quantile of its whole frame. The value the other
reading gives (a 60% cut, which is the only reading with no bar of the
evaluation window in the derivation) is derived and reported beside it, and if
the two thresholds differ enough to matter the run is repeated at the second one
so the verdict cannot depend on the ambiguity.
"""

from __future__ import annotations

import shutil
import time

import r7dlib as R

R7 = R.R7
BURNED = "burned -- reported, not evidence"


def energy_first_cell(energy, defined, halves, target, *, which):
    """The selected coverage target, derived on one frame by the one rule."""
    return R.cell_for(energy, defined, halves, target, which=which)


def evaluate(name: str, frame, cell, *, stop: int, root, first_tradeable: int | None = None) -> dict:
    """The energy-first machine and both comparators, over identical bars.

    R5's trained cell is run **unmodified** -- it is the incumbent the plan
    names, not a variant of the challenger -- and `donchian` 40/10 is R5's own R0
    baseline cell, tuned on BTC's training half and never re-tuned here (M11 is
    about tuning both sides by one rule; re-deriving it per instrument is what
    M22 forbids and M18 measured the cost of).

    ``first_tradeable`` defaults to the **deepest** warmup in the set, which is
    what lets three strategies with different warmups trade one window: the
    machine warms 2,160 bars and donchian 40, and a shared frame *start* would
    hand donchian 2,120 bars the machine cannot see.
    """
    strategies = {
        "energy-first (R7d)": R.strategy_for(cell),
        "R5 trained, unmodified": R7.gate.machine(R7.TRAINED),
        "donchian 40/10": R7.gate.BASELINE,
    }
    if first_tradeable is None:
        first_tradeable = max(strategy.warmup_bars for strategy in strategies.values())
    rows = R.run_mixed(strategies, frame, root, first_tradeable=first_tradeable, stop=stop)
    print(f"\n{name}: {rows['energy-first (R7d)']['tradeable_bars']} tradeable bars "
          f"from {rows['energy-first (R7d)']['first_tradeable']}\n{R.header()}")
    for label in strategies:
        print(R.line(label, rows[label]))
    return rows


def beats(rows: dict, label: str, other: str) -> bool:
    return rows[label]["sharpe_tradeable"] > rows[other]["sharpe_tradeable"]


def ruined(rows: dict) -> dict:
    """Runs whose equity curve reaches zero, where a percentage Sharpe stops meaning it.

    Not a declared measurement, and it settles nothing -- it is here because a
    reader's first question about a comparator showing a positive Sharpe on a
    net return past -100% should be answered in the harness rather than in
    prose. Sizing is non-compounding (initial cash x ``position_pct``), so a
    short position in an instrument that multiplies can lose more than the book
    holds; once equity is negative, ``pct_change`` flips sign with its own
    denominator and the resulting Sharpe is an artifact rather than a
    risk-adjusted return.
    """
    return {
        label: {
            "min_equity": float(row["equity"].min()),
            "final_equity": float(row["equity"].iloc[-1]),
            "bars_at_or_below_zero": int((row["equity"] <= 0.0).sum()),
            "sharpe_tradeable": row["sharpe_tradeable"],
        }
        for label, row in rows.items()
        if float(row["equity"].min()) <= 0.0
    }


def main() -> None:
    started = time.time()
    selection = R.read("step2_selection.json")
    assert not selection["kill_switch"]["fired"], (
        "step 2's kill switch fired; the phase stops there and SOL is not spent"
    )
    target = selection["selected"]["target_coverage"]
    print(f"selected coverage target: {target:.0%} enter / "
          f"{R.EXIT_COVERAGE_MULTIPLE * target:.0%} exit "
          f"({selection['selected']['label']})")

    payload: dict = {"selected_target_coverage": target, "frames": {}}
    root = R.OUT / "evaluation"

    # --- burned frame 1: BTC's test half --------------------------------------
    btc = R7.load_frame()
    df, _ = btc
    split = R7.split_index(df)
    btc_cell = {
        "enter": {"energy_value": selection["selected"]["enter_energy"]},
        "exit": {"energy_value": selection["selected"]["exit_energy"]},
    }
    print(f"\n{'=' * 78}\nBTC/USDT perp 4h -- TEST HALF ({BURNED})")
    print(f"  enter_energy {btc_cell['enter']['energy_value']:.4f} / exit_energy "
          f"{btc_cell['exit']['energy_value']:.4f}, derived on BTC's training half")
    btc_rows = evaluate(
        "BTC test half", btc, btc_cell,
        first_tradeable=split, stop=len(df), root=root / "btc_test",
    )
    # The two comparators over exactly R5's own test half: this run reproduces
    # §9.2's published pair, so a moved figure here would mean the frame moved
    # rather than the challenger losing.
    print(f"  control: R5's row reproduces at "
          f"{btc_rows['R5 trained, unmodified']['net_return_pct']:+.4f}% / "
          f"{btc_rows['R5 trained, unmodified']['sharpe_whole_frame']:+.4f} / "
          f"{btc_rows['R5 trained, unmodified']['trades']} trades, and donchian at "
          f"{btc_rows['donchian 40/10']['net_return_pct']:+.4f}% / "
          f"{btc_rows['donchian 40/10']['sharpe_whole_frame']:+.4f} / "
          f"{btc_rows['donchian 40/10']['trades']} trades")
    payload["frames"]["BTC test half"] = {
        "label": BURNED,
        "cell": btc_cell,
        "rows": {label: R.slim(row) for label, row in btc_rows.items()},
    }

    # --- burned frame 2: ETH's full frame --------------------------------------
    eth = R7.load_eth_frame()
    eth_df, _ = eth
    eth_frame, eth_defined = R.machine_inputs(eth_df)
    eth_cell = energy_first_cell(
        eth_frame["energy"], eth_defined, R7.eth_halves(eth_df), target, which="train"
    )
    print(f"\n{'=' * 78}\nETH/USDT perp 4h -- FULL FRAME ({BURNED})")
    print(f"  enter_energy {eth_cell['enter']['energy_value']:.4f} "
          f"(covers {eth_cell['enter']['realised_coverage']:.2%}) / exit_energy "
          f"{eth_cell['exit']['energy_value']:.4f} "
          f"(covers {eth_cell['exit']['realised_coverage']:.2%}), derived on ETH's "
          f"own training half")
    eth_rows = evaluate(
        "ETH full frame", eth, eth_cell, stop=len(eth_df), root=root / "eth_full",
    )
    payload["frames"]["ETH full frame"] = {
        "label": BURNED,
        "cell": eth_cell,
        "rows": {label: R.slim(row) for label, row in eth_rows.items()},
    }

    # --- the holdout -----------------------------------------------------------
    sol = R.load_sol_frame()
    sol_df, _ = sol
    sol_frame, sol_defined = R.machine_inputs(sol_df)
    sol_halves = R.first_sixty_percent(sol_df)
    sol_cell = energy_first_cell(
        sol_frame["energy"], sol_defined, sol_halves, target, which="full"
    )
    alternative = energy_first_cell(
        sol_frame["energy"], sol_defined, sol_halves, target, which="train"
    )
    print(f"\n{'=' * 78}\nSOL/USDT perp 4h -- THE HOLDOUT, whole frame, one run")
    print(f"  {len(sol_df)} bars, {sol_df.index[0]} to {sol_df.index[-1]}")
    print(f"  enter_energy {sol_cell['enter']['energy_value']:.4f} "
          f"(covers {sol_cell['enter']['realised_coverage']:.2%}) / exit_energy "
          f"{sol_cell['exit']['energy_value']:.4f} "
          f"(covers {sol_cell['exit']['realised_coverage']:.2%}), derived on SOL's "
          f"own bars")
    print(f"  the other reading of the plan -- a 60% cut -- would give "
          f"{alternative['enter']['energy_value']:.4f} / "
          f"{alternative['exit']['energy_value']:.4f}")
    sol_rows = evaluate(
        "SOL whole frame", sol, sol_cell, stop=len(sol_df), root=root / "sol_full",
    )

    challenger = "energy-first (R7d)"
    stress = sol_rows[challenger]["net_by_stress"][R.COST_STRESS_MULTIPLE]
    broke = ruined(sol_rows)
    for label, row in broke.items():
        print(f"\n  NOTE  {label} reaches zero equity on SOL: min "
              f"{row['min_equity']:+.2f}, final {row['final_equity']:+.2f}, "
              f"{row['bars_at_or_below_zero']} bars at or below zero.")
        print(f"        Its Sharpe of {row['sharpe_tradeable']:+.4f} is an artifact of "
              f"a percentage return taken across a sign change, not a")
        print("        risk-adjusted return -- see `ruined` in this file.")
    verdict = {
        "beats_r5_trained": beats(sol_rows, challenger, "R5 trained, unmodified"),
        "beats_donchian": beats(sol_rows, challenger, "donchian 40/10"),
        "survives_3x_costs": stress > 0.0,
    }
    verdict["clears"] = all(verdict.values())

    print(f"\n{'=' * 78}\nTHE HOLDOUT -- declared: beats BOTH comparators on Sharpe net "
          f"of funding,")
    print(f"AND survives {R.COST_STRESS_MULTIPLE:.0f}x costs with a positive net return.")
    print(f"  Sharpe: energy-first "
          f"{sol_rows[challenger]['sharpe_tradeable']:+.4f}  vs R5 trained "
          f"{sol_rows['R5 trained, unmodified']['sharpe_tradeable']:+.4f}  vs donchian "
          f"{sol_rows['donchian 40/10']['sharpe_tradeable']:+.4f}")
    print(f"  net at {R.COST_STRESS_MULTIPLE:.0f}x costs: {stress:+.4f}%")
    for claim, ok in verdict.items():
        if claim != "clears":
            print(f"  {R.mark(ok)}  {claim}")
    print(f"\n  {R.mark(verdict['clears'])}  the holdout")

    # The ambiguity above only needs a second run if it moves the machine, and
    # that is a fact about the two thresholds rather than a matter of opinion.
    same_cell = (
        round(alternative["enter"]["energy_value"], 6)
        == round(sol_cell["enter"]["energy_value"], 6)
        and round(alternative["exit"]["energy_value"], 6)
        == round(sol_cell["exit"]["energy_value"], 6)
    )
    alternative_rows = None
    if not same_cell:
        print(f"\n{'-' * 78}\nthe other reading of SOL's derivation, run so the verdict "
              f"cannot depend on it")
        alternative_rows = evaluate(
            "SOL whole frame, threshold from its first 60%", sol, alternative,
            stop=len(sol_df), root=root / "sol_alt",
        )
        alt_stress = alternative_rows[challenger]["net_by_stress"][R.COST_STRESS_MULTIPLE]
        alt_verdict = {
            "beats_r5_trained": beats(alternative_rows, challenger, "R5 trained, unmodified"),
            "beats_donchian": beats(alternative_rows, challenger, "donchian 40/10"),
            "survives_3x_costs": alt_stress > 0.0,
        }
        alt_verdict["clears"] = all(alt_verdict.values())
        print(f"  {R.mark(alt_verdict['clears'])}  the holdout under the other reading "
              f"(Sharpe {alternative_rows[challenger]['sharpe_tradeable']:+.4f}, "
              f"net@3x {alt_stress:+.4f}%)")
        payload["sol_alternative_derivation"] = {
            "cell": alternative,
            "rows": {label: R.slim(row) for label, row in alternative_rows.items()},
            "verdict": alt_verdict,
        }

    payload["frames"]["SOL whole frame"] = {
        "label": "the holdout -- the only test that counts",
        "cell": sol_cell,
        "bars": len(sol_df),
        "first_bar": str(sol_df.index[0]),
        "last_bar": str(sol_df.index[-1]),
        "rows": {label: R.slim(row) for label, row in sol_rows.items()},
        "runs_reaching_zero_equity": broke,
        "verdict": verdict,
    }
    R.write("step3_evaluation.json", payload)
    shutil.rmtree(root, ignore_errors=True)
    print(f"\nelapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
