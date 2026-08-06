"""R9 measurement harness.

Run from this directory, in order -- each step reads the previous one's JSON::

    cd scripts/r9 && for step in step0_control step1_surface step1b_winner_curve \\
        step2_dsr step3_walkforward step4_perturbation step5_dropout \\
        step6_checks step7_wf_analysis; do python "$step.py"; done

``R9_OUT`` chooses where the intermediate JSON and the transient report
directories go; it defaults to a temp directory and must never be ``reports/``.
The protocol these scripts execute is fixed by
``docs/plans/2026-08-06-r9-robustness.md``, which was committed before any of
the numbers existed; nothing here selects, tunes or writes anything.

Reuses ``tests/test_state_machine_gate.py``'s ``evaluate``/``out_of_sample``
rather than writing a third harness, and adds only what R9 needs on top:

- the **M21 scalar** -- Sharpe over the tradeable bars alone, recomputed from the
  run's own ``equity_curve.csv`` with the engine's own estimator, so it is the
  same statistic as the published one over a different set of rows;
- the per-bar return series (for the deflated Sharpe's skew and kurtosis);
- a report directory that is deleted as soon as it has been read, so a few
  hundred runs do not accumulate a few hundred inlined chart bundles.

Nothing here writes to the repo, to ``reports/`` or to Postgres.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for entry in (str(REPO), str(REPO / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import tests.test_state_machine_gate as gate  # noqa: E402

CASH = 10_000.0
FREQ = "4h"

# Every run writes its artifacts here and most of them are deleted again on the
# way out. It is deliberately **not** under ``reports/``: a report directory is
# the reproducibility record of a run someone chose to publish, and none of
# these were.
OUT = Path(os.environ.get("R9_OUT", Path(tempfile.gettempdir()) / "strategy-lab-r9"))


def load_frame():
    """The gate's own module-scoped frame fixture, called outside pytest."""
    return gate.frame.__wrapped__()


def sharpe_of(equity: pd.Series, *, base: float = CASH, freq: str = FREQ) -> float:
    """``backtests.engine._equity_risk``'s Sharpe, over whatever rows are handed in.

    The first bar's return is measured against ``base`` rather than dropped,
    exactly as the engine does, so a slice starting at the first tradeable bar
    is scored on the same estimator as the whole window.
    """
    returns = equity.pct_change()
    returns.iloc[0] = equity.iloc[0] / base - 1.0
    return float(returns.vbt.returns(freq=freq).sharpe_ratio())


def returns_of(equity: pd.Series, *, base: float = CASH) -> pd.Series:
    returns = equity.pct_change()
    returns.iloc[0] = equity.iloc[0] / base - 1.0
    return returns


def run(strategy, frame, root: Path, *, first_tradeable: int, stop: int, keep: bool = False):
    """One ``evaluate`` run, plus the tradeable-bars scalar and its returns."""
    root.mkdir(parents=True, exist_ok=True)
    out = gate.evaluate(strategy, frame, root, first_tradeable=first_tradeable, stop=stop)

    report = next((root / strategy.name).glob("*Z_*"))
    equity = pd.read_csv(report / "equity_curve.csv", index_col=0, parse_dates=True)["equity"]
    config = json.loads((report / "config.json").read_text())
    warmup = int(config["warmup_bars"])

    # Nothing trades through the warmup, so the tradeable slice starts from a
    # book still holding exactly its initial cash. Asserted rather than assumed:
    # it is what makes ``base=CASH`` the right denominator for the slice.
    assert abs(float(equity.iloc[warmup - 1]) - CASH) < 1e-9, (
        f"equity moved during warmup: {equity.iloc[warmup - 1]}"
    )
    # A book that never opens a position has a flat curve and no Sharpe at all:
    # the engine writes null rather than 0.0, and so does this. A zero there
    # would read as "measured and mediocre" rather than "never traded".
    flat = out["sharpe"] is None
    if flat:
        assert equity.nunique() == 1, "null Sharpe on a curve that moved"
    else:
        # The recomputation has to agree with the engine's own published Sharpe
        # on the whole window, or the tradeable-bars number is a different
        # statistic rather than the same one over fewer rows.
        assert abs(sharpe_of(equity) - out["sharpe"]) < 1e-9, (
            f"harness Sharpe {sharpe_of(equity):+.6f} != engine {out['sharpe']:+.6f}"
        )

    out["warmup_bars"] = warmup
    out["window_bars"] = len(equity)
    out["sharpe_whole_frame"] = out["sharpe"]
    out["sharpe_tradeable"] = None if flat else sharpe_of(equity.iloc[warmup:])
    out["returns"] = returns_of(equity.iloc[warmup:])
    out["equity"] = equity
    out["config"] = config
    out["first_tradeable"] = str(out["first_tradeable"])
    if not keep:
        shutil.rmtree(root, ignore_errors=True)
    return out


def slim(row: dict) -> dict:
    """``row`` without the pandas objects, so it can be written as JSON."""
    return {key: value for key, value in row.items() if key not in ("returns", "equity", "config")}


def describe(config) -> str:
    return (
        f"enter={config.enter_strength:.4f} exit={config.exit_strength:.4f} "
        f"dwell={config.min_dwell} cool={config.cooldown}"
    )


def cell_key(config) -> str:
    return f"{config.enter_strength:.6f}/{config.exit_strength:.6f}/{config.min_dwell}/{config.cooldown}"
