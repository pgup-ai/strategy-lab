"""R9 part 1: the deflated Sharpe ratio of the training-half selection.

Bailey & Lopez de Prado. Two things are computed, and the difference between
them matters:

- ``E[max SR]`` under the null that all 54 trials have zero true Sharpe. It is
  not zero, because the maximum of 54 draws grows with the spread of the draws.
- the deflated Sharpe itself: the probability that the winner's true Sharpe is
  above zero *given* that it is the best of 54.

Both are reported under R5's own scalar (whole-window Sharpe) and under M21's
(tradeable bars only), because they are different rankings of the same 54 runs.
"""

from __future__ import annotations

import json
import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import r9lib as R

OUT = R.OUT
OUT.mkdir(parents=True, exist_ok=True)
EULER = 0.5772156649015329
NORMAL = NormalDist()


def expected_max_sharpe(*, sigma: float, trials: int) -> float:
    """``E[max SR]`` over ``trials`` draws of N(0, sigma^2)."""
    first = NORMAL.inv_cdf(1.0 - 1.0 / trials)
    second = NORMAL.inv_cdf(1.0 - 1.0 / (trials * math.e))
    return sigma * ((1.0 - EULER) * first + EULER * second)


def deflated_sharpe(*, observed: float, benchmark: float, returns: np.ndarray) -> dict:
    """``observed`` and ``benchmark`` in per-observation units, as ``returns`` is."""
    series = pd.Series(returns)
    skew = float(series.skew())
    excess_kurtosis = float(series.kurtosis())
    kurtosis = excess_kurtosis + 3.0
    length = len(series)
    denominator = math.sqrt(
        1.0 - skew * observed + (kurtosis - 1.0) / 4.0 * observed**2
    )
    statistic = (observed - benchmark) * math.sqrt(length - 1) / denominator
    return {
        "skew": skew,
        "excess_kurtosis": excess_kurtosis,
        "kurtosis": kurtosis,
        "observations": length,
        "z": statistic,
        "dsr": NORMAL.cdf(statistic),
    }


def main() -> None:
    rows = json.loads((OUT / "training_surface.json").read_text())
    returns = np.load(OUT / "training_returns.npz")

    # The annualization factor the engine's own Sharpe carries, recovered rather
    # than assumed: a per-bar Sharpe is what the deflation formula needs, and
    # the skew/kurtosis terms are only consistent at that frequency.
    winner = max(rows, key=lambda row: row["sharpe_tradeable"])
    sample = returns[winner["cell"]]
    per_bar = float(np.mean(sample) / np.std(sample, ddof=1))
    factor = (winner["sharpe_tradeable"] / per_bar) ** 2
    print(f"annualization factor implied by the engine's Sharpe: {factor:.3f} "
          f"(4h bars in 365d = {365 * 24 / 4})")
    print(f"winner: {winner['label']}")
    print(f"  tradeable bars {len(sample)}, whole window {winner['window_bars']}, "
          f"warmup {winner['warmup_bars']}")

    root = math.sqrt(factor)
    report = {}
    for rule, label in (
        ("sharpe_whole_frame", "R5's rule (whole window, incl. warmup)"),
        ("sharpe_tradeable", "M21's rule (tradeable bars only)"),
    ):
        trials = np.array([row[rule] for row in rows], dtype="float64")
        best = max(rows, key=lambda row: row[rule])
        assert best["cell"] == winner["cell"], "the two rules disagree about the winner"
        sigma_annual = float(np.std(trials, ddof=1))
        observed_annual = float(best[rule])

        # Per-observation units for the deflation itself.
        if rule == "sharpe_whole_frame":
            # The whole-window curve of the winning cell, warmup zeros included:
            # that is the series R5's own scalar scored.
            sample_rule = np.load(OUT / "winner_whole_window_returns.npy")
        else:
            sample_rule = sample
        observed = observed_annual / root
        sigma = sigma_annual / root
        benchmark = expected_max_sharpe(sigma=sigma, trials=len(trials))
        stats = deflated_sharpe(observed=observed, benchmark=benchmark, returns=sample_rule)

        report[rule] = {
            "label": label,
            "trials": len(trials),
            "observed_annualized": observed_annual,
            "runner_up_annualized": float(np.sort(trials)[-2]),
            "trial_mean_annualized": float(np.mean(trials)),
            "trial_median_annualized": float(np.median(trials)),
            "trial_min_annualized": float(np.min(trials)),
            "trial_max_annualized": float(np.max(trials)),
            "trial_sd_annualized": sigma_annual,
            "positive_trials": int((trials > 0).sum()),
            "expected_max_annualized": benchmark * root,
            "observed_per_bar": observed,
            "trial_sd_per_bar": sigma,
            "expected_max_per_bar": benchmark,
            **stats,
        }
        row = report[rule]
        print(f"\n--- {label}")
        print(f"  54 trials: min {row['trial_min_annualized']:+.4f}  median "
              f"{row['trial_median_annualized']:+.4f}  max {row['trial_max_annualized']:+.4f}  "
              f"sd {row['trial_sd_annualized']:.4f}  positive {row['positive_trials']}/54")
        print(f"  observed (annualized) {row['observed_annualized']:+.4f}   runner-up "
              f"{row['runner_up_annualized']:+.4f}")
        print(f"  E[max SR] under the null (annualized) {row['expected_max_annualized']:+.4f}")
        print(f"  winner returns: skew {row['skew']:+.4f}  excess kurtosis "
              f"{row['excess_kurtosis']:+.4f}  T {row['observations']}")
        print(f"  per-bar: observed {row['observed_per_bar']:+.6f}  E[max] "
              f"{row['expected_max_per_bar']:+.6f}")
        print(f"  z {row['z']:+.4f}   DSR {row['dsr']:.6f}")

    (OUT / "dsr.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
