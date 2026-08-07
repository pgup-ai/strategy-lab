"""Steps 5-6 (plan §5 then §6): the deflated Sharpe, and the embargo statement.

**The deflation is over three trials**, which is the whole grid this phase
declared: R9 priced a 54-cell search at DSR 0.70 and the discount scales with
the search, so the cheapest way to buy the discount back was to search less.
Both quantities are R9's, imported from ``scripts/r9/step2_dsr`` rather than
re-implemented -- a second deflation would be a second convention, and the two
phases' numbers would stop being comparable:

- ``E[max SR]`` under the null that all three trials have zero true Sharpe. It
  is not zero, because the maximum of three draws grows with their spread.
- the deflated Sharpe itself: the probability that the winner's true Sharpe is
  above zero **given** that it is the best of three.

**Declared, and stated with its frequency (M23):** observed Sharpe >
``E[max SR]``, and DSR >= **0.95**, both at **4h bar frequency**. R9's own
caveat is why the frequency is named -- the same returns deflate to 0.70 per bar
and 0.77 daily, and choosing after the fact is what M23 forbids.

The scalar is M21's, the one the selection was made on: Sharpe over the
tradeable bars of BTC's training half. The three cells are re-run here rather
than read out of step 2's JSON, because the deflation needs the winner's own
per-bar **return series** and not only its Sharpe -- and the re-run is asserted
to reproduce step 2's surface to the last bit, so it is the same three trials
and not a second measurement of them.

**Step 6 is a statement, not a measurement**, and the statement is checked
against the code rather than restated from R9: the machine fits nothing, has no
labels and no forward horizon of its own, so there is no training set to purge
and no label window to embargo. What R7d *does* have is a forward-looking
**diagnostic** label in step 1, whose split-crossing rows are re-measured there
on R7b's precedent.
"""

from __future__ import annotations

import math
import shutil
import time

import numpy as np
import r7dlib as R
import step2_dsr as R9DSR

R7 = R.R7
DSR_BAR = 0.95
BARS_PER_YEAR = 365 * 24 / 4


def main() -> None:
    started = time.time()
    selection = R.read("step2_selection.json")
    cells = selection["cells"]
    winner_label = selection["selected"]["label"]

    frame = R7.load_frame()
    df, _ = frame
    split = R7.split_index(df)
    warmup = selection["warmup_bars"]

    root = R.OUT / "dsr"
    rows = {}
    for label, cell in cells.items():
        rows[label] = R.run_row(
            R.strategy_for(cell), frame, root / R.slug(label),
            first_tradeable=warmup, stop=split,
        )
        published = selection["surface"][label]["sharpe_tradeable"]
        assert repr(rows[label]["sharpe_tradeable"]) == repr(published), (
            f"{label} re-runs at {rows[label]['sharpe_tradeable']!r}, not step 2's "
            f"{published!r}; these are not the same three trials"
        )
    print(f"the three trials re-run bit-identically to step 2's surface, over "
          f"{rows[winner_label]['tradeable_bars']} tradeable bars")

    winner = rows[winner_label]
    sample = winner["returns"].to_numpy(dtype="float64")
    per_bar = float(np.mean(sample) / np.std(sample, ddof=1))
    factor = (winner["sharpe_tradeable"] / per_bar) ** 2
    print(f"annualization factor implied by the engine's Sharpe: {factor:.3f} "
          f"(4h bars in 365d = {BARS_PER_YEAR:.0f})")
    assert abs(factor - BARS_PER_YEAR) < 1.0, (
        f"the engine is not annualizing at 4h bar frequency: {factor:.3f}"
    )

    root_factor = math.sqrt(factor)
    trials = np.array(
        [rows[label]["sharpe_tradeable"] for label in cells], dtype="float64"
    )
    sigma_annual = float(np.std(trials, ddof=1))
    observed_annual = float(winner["sharpe_tradeable"])
    assert observed_annual == float(np.max(trials)), "the winner is not the best trial"

    sigma = sigma_annual / root_factor
    observed = observed_annual / root_factor
    benchmark = R9DSR.expected_max_sharpe(sigma=sigma, trials=len(trials))
    stats = R9DSR.deflated_sharpe(observed=observed, benchmark=benchmark, returns=sample)

    print(f"\n{'-' * 78}\nTHE DEFLATED SHARPE -- {len(trials)} trials, 4h bar frequency")
    print(f"  trials (annualized): "
          f"{', '.join(f'{value:+.4f}' for value in np.sort(trials)[::-1])}")
    print(f"  sd {sigma_annual:.4f}   observed {observed_annual:+.4f}   "
          f"E[max SR] {benchmark * root_factor:+.4f}")
    print(f"  per bar: observed {observed:+.6f}   E[max SR] {benchmark:+.6f}")
    print(f"  winner returns: skew {stats['skew']:+.4f}  excess kurtosis "
          f"{stats['excess_kurtosis']:+.4f}  T {stats['observations']}")
    print(f"  z {stats['z']:+.4f}   DSR {stats['dsr']:.6f}")

    beats_expected_max = observed_annual > benchmark * root_factor
    clears = beats_expected_max and stats["dsr"] >= DSR_BAR
    print(f"\n  {R.mark(beats_expected_max)}  observed > E[max SR]")
    print(f"  {R.mark(stats['dsr'] >= DSR_BAR)}  DSR >= {DSR_BAR:.2f}")
    print(f"  {R.mark(clears)}  the selection is not luck")

    print(f"\n{'-' * 78}\nPLAN §6 -- purging and embargo: STATED, NOT PERFORMED")
    print("  Unchanged from R9's and R7b's reasoning, re-checked against the code:")
    print("  the state machine fits nothing. It has no training set, no labels and")
    print("  no forward horizon of its own -- `StateMachine.run` is a causal walk over")
    print("  five features, each of which is rolling or expanding by construction and")
    print("  poison-probed by `tests/test_feature_lookahead.py`. One threshold pair")
    print("  chosen by a three-cell grid search is a selection, not a fitted model, so")
    print("  there is nothing to purge and no label window to embargo. Purging becomes")
    print("  mandatory at R8, which is where a model with a training set first exists.")
    print("  What R7d does have is a forward-looking *diagnostic* label in step 1, and")
    print("  its split-crossing rows are re-measured there, on R7b's precedent.")

    R.write("step4_dsr.json", {
        "trials": len(trials),
        "frequency": "4h bar",
        "annualization_factor": factor,
        "dsr_bar": DSR_BAR,
        "trials_annualized": {label: rows[label]["sharpe_tradeable"] for label in cells},
        "trial_sd_annualized": sigma_annual,
        "observed_annualized": observed_annual,
        "expected_max_annualized": benchmark * root_factor,
        "observed_per_bar": observed,
        "expected_max_per_bar": benchmark,
        **stats,
        "observed_beats_expected_max": beats_expected_max,
        "dsr_clears_bar": bool(stats["dsr"] >= DSR_BAR),
        "clears": bool(clears),
        "purging_and_embargo": "stated, not performed -- the machine fits nothing",
    })
    shutil.rmtree(root, ignore_errors=True)
    print(f"\nelapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
