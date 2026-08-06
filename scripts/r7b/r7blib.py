"""R7b measurement harness -- the energy gate on the state machine's entry.

Run from this directory, in order -- each step reads the previous one's JSON::

    cd scripts/r7b && for step in step0_control step1_diagnostic; \\
        do python "$step.py"; done

``R7B_OUT`` chooses where the intermediate JSON and the transient report
directories go; it defaults to a temp directory and must never be ``reports/``.
The protocol these scripts execute is fixed by
``docs/plans/2026-08-06-r7b-energy-gate.md``, committed before any of the numbers
existed.

**Two steps, not five, and that is the protocol working rather than an
omission.** The plan's §1 declares that the diagnostic gate runs before any P&L
and that *"if the energy gate fails this, the P&L runs are not read"*. It failed.
So the selection, the three evaluations and the deflated Sharpe were never run,
and the code for them was never written -- a harness carrying steps that
nothing executed would be indistinguishable from one whose steps were run and
whose numbers were left out.

**This phase changes code, which the previous three did not**, so the harness
carries one job R7's and R9's did not have: proving the change inert where it is
supposed to be inert. ``step0_control`` reproduces R5's published row and
compares every published ``state_machine_v1``/``v2`` figure against a baseline
captured from ``main`` before the change existed.

Three things it deliberately does not own.

- **The rate metric** is ``scripts/r7``'s, imported rather than reimplemented:
  ``trend_label``, ``forward_er`` and ``rate_table``, with R7's own tercile rule
  and its 5 pp composite bar. A second implementation of the statistic R7's
  verdict rests on would make the two phases incomparable, which is the whole
  reason §1 reuses R7's threshold verbatim. ``step1`` reproduces R7's published
  ``strength >= 0.80`` row as its control.
- **The frame and the engine run** are ``tests/test_state_machine_gate.py``'s
  fixture and ``evaluate``, reached through ``r7lib`` -- the bars R5, R9 and R7
  measured, with the funding column attached (M20).
- **The machine's cells** are R5's trained cell with one field moved. Nothing
  re-derives ``enter_strength``, ``exit_strength``, ``min_dwell`` or ``cooldown``
  on any frame (M22).

Nothing here writes to the repo, to ``reports/`` or to Postgres.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for entry in (str(REPO), str(REPO / "src"), str(REPO / "scripts" / "r7")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import r7lib as R7  # noqa: E402
from strategy_lab.state.machine import StateMachine  # noqa: E402

OUT = Path(os.environ.get("R7B_OUT", Path(tempfile.gettempdir()) / "strategy-lab-r7b"))
# The docstring above says this must never be ``reports/``; that is a guard
# rather than a comment, because a report directory is the reproducibility
# record of a run someone chose to publish and none of these are.
if OUT.resolve() == REPO.resolve() or REPO.resolve() in OUT.resolve().parents:
    raise SystemExit(f"R7B_OUT must point outside the repository, not {OUT}")

# The declared grid: four values, of which 1.00 is the control that reproduces
# R5 exactly, so the search is three live trials. Written out here rather than
# described in prose because ``step4_dsr`` was to have deflated by this length,
# and a grid stated only in a progress log cannot be checked against the
# deflation that priced it. It stayed four cells; the deflation was never
# reached.
ENERGY_CEILINGS: tuple[float, ...] = (0.50, 0.65, 0.80, 1.00)
CONTROL_CEILING = 1.00

CASH = R7.CASH
BTC_IDENTITY = R7.gate.IDENTITY

# R5's published test-half row, to the digits the charter prints.
R5_PUBLISHED = {
    "net_return_pct": 15.45,
    "sharpe": 0.896,
    "max_drawdown_pct": 4.67,
    "trades": 73,
}


def cell(ceiling: float) -> StateMachine:
    """R5's trained cell with one field moved, and nothing else."""
    return replace(R7.TRAINED, energy_ceiling=ceiling)


def strategy_for(ceiling: float):
    return R7.gate.machine(cell(ceiling))


def write(name: str, payload) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


def read_if_present(name: str):
    """The stored payload, or ``None`` for a step that has not run yet."""
    path = OUT / name
    return json.loads(path.read_text()) if path.exists() else None


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


mark = R7.mark
slim = R7.slim_row


__all__ = [
    "BTC_IDENTITY",
    "CASH",
    "CONTROL_CEILING",
    "ENERGY_CEILINGS",
    "OUT",
    "R5_PUBLISHED",
    "R7",
    "cell",
    "mark",
    "read_if_present",
    "slim",
    "strategy_for",
    "write",
]
