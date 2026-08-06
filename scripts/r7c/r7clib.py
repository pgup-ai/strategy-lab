"""R7c measurement harness -- the energy-first lifecycle, and the holdout it saved.

Run from this directory::

    cd scripts/r7c && python step1_selection.py

**Step 0 is ``scripts/r7b/step0_control.py``, reused rather than rewritten**, and
it is run twice: once on a clean ``main`` worktree with ``R7B_BASELINE=1`` to
capture the baseline, then once here. R7c's no-op control *is* R7b's -- the row
it labels ``v1 trained`` is R5's cell with both energy lifecycles at their
defaults, so reproducing +15.45% / +0.896 / 4.67% / 73 there is exactly the
claim "energy-first mode off reproduces R5's row", and its byte-identity check
covers all four published v1/v2 figures against ``main``. A second copy of that
script would be a second thing to keep in step with ``main``.

``R7C_OUT`` chooses where the intermediate JSON and the transient report
directories go; it defaults to a temp directory and must never be ``reports/``.
The protocol this script executes is fixed by
``docs/plans/2026-08-06-r7c-energy-first-lifecycle.md``, committed before any of
the numbers existed.

**One step, not four, and that is the protocol working rather than an
omission.** The plan puts a **kill switch** on tradeability before any
out-of-sample return is read -- the selected cell must trade 26-231 round trips
on BTC's training half -- and says that if it fires *"the SOL holdout is not
spent and the phase stops"*. It fired: the selected cell trades **256**. So the
overlay control, the three evaluations and the deflated Sharpe were never run,
and the code for them was never written. **SOL was not touched.** A harness
carrying steps that nothing executed would be indistinguishable from one whose
steps were run and whose numbers were left out, which is the standard
``scripts/r7b`` set when its own kill switch fired (M28).

Three things this harness deliberately does not own.

- **The frame, the engine run and the split** are ``tests/test_state_machine_gate
  .py``'s fixture and ``evaluate``, reached through ``r7lib.gate_run`` -- the
  bars R5, R9, R7 and R7b measured, funding column attached (M20).
- **The scalar** is ``r7lib.sharpe_of`` over the tradeable bars (M21), the same
  estimator ``scripts/r9`` and ``scripts/r7`` used. A second Sharpe would make
  the phases incomparable.
- **The machine's cells** are R5's trained cell with the two energy thresholds
  set and nothing else moved. ``min_dwell``, ``cooldown``, ``direction_floor``,
  ``stability_floor``, ``crowding_extreme`` and R5's strength thresholds are not
  re-derived on any frame (M22).

**One thing in the pre-registration is arithmetically impossible and is resolved
here rather than quietly.** The declared grid is
``enter_energy in {0.35, 0.50, 0.65}`` x ``exit_energy in {0.65, 0.80}`` and is
called **6 cells**, while the same section requires the constructor to enforce
``exit_energy > enter_energy`` strictly, as the mirror of ``enter_strength >
exit_strength``. ``(0.65, 0.65)`` satisfies the first and is refused by the
second. The refusal wins -- it is the more specific commitment, it is the one
the plan asks to be *tested*, and admitting the cell would mean shipping a
machine with no dead band on the very axis the phase moves the lifecycle onto.
So the surface has **5 constructible cells and one refusal**, and all six are
reported. Nothing turns on the choice: every constructible cell fires the kill
switch, so the phase stops under either reading.

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

OUT = Path(os.environ.get("R7C_OUT", Path(tempfile.gettempdir()) / "strategy-lab-r7c"))
# The docstring above says this must never be ``reports/``; that is a guard
# rather than a comment, because a report directory is the reproducibility
# record of a run someone chose to publish and none of these are.
if OUT.resolve() == REPO.resolve() or REPO.resolve() in OUT.resolve().parents:
    raise SystemExit(f"R7C_OUT must point outside the repository, not {OUT}")

# --- the declared grid, written out rather than described ---------------------
#
# Written here rather than described in prose because a grid stated only in a
# progress log cannot be checked against the verdict it produced -- and because
# the deflated Sharpe was to have deflated by this length. It stayed six cells;
# the deflation was never reached.
ENTER_ENERGIES: tuple[float, ...] = (0.35, 0.50, 0.65)
EXIT_ENERGIES: tuple[float, ...] = (0.65, 0.80)
DECLARED_CELLS: tuple[tuple[float, float], ...] = tuple(
    (enter, exit_) for enter in ENTER_ENERGIES for exit_ in EXIT_ENERGIES
)

# --- the declared thresholds, as numbers (M23) --------------------------------
#
# The band is a factor of three either side of the 77 round trips R5's trained
# cell turns over on BTC's training half. Below it the P&L is a handful of
# draws; above it the run is being scored on costs rather than on state.
KILL_SWITCH_MIN = 26
KILL_SWITCH_MAX = 231
R5_TRAINING_TRADES = 77
COST_STRESS_MULTIPLE = 3.0


def cell(enter_energy: float, exit_energy: float) -> StateMachine:
    """R5's trained cell with the energy lifecycle switched on, and nothing else.

    Raises for a cell the constructor refuses -- see the module docstring for
    why ``(0.65, 0.65)`` is one of the six and is not one of the five.
    """
    return replace(R7.TRAINED, enter_energy=enter_energy, exit_energy=exit_energy)


def constructible_cells() -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """The declared grid split into what the constructor admits and what it refuses."""
    live: list[tuple[float, float]] = []
    refused: list[tuple[float, float]] = []
    for enter, exit_ in DECLARED_CELLS:
        try:
            cell(enter, exit_)
        except ValueError:
            refused.append((enter, exit_))
        else:
            live.append((enter, exit_))
    return live, refused


def strategy_for(enter_energy: float, exit_energy: float):
    return R7.gate.machine(cell(enter_energy, exit_energy))


def label_of(enter_energy: float, exit_energy: float) -> str:
    return f"enter<={enter_energy:.2f} exit>{exit_energy:.2f}"


def machine_warmup() -> int:
    """The warmup every state-machine strategy in this phase carries.

    The energy thresholds do not enter ``convergence_bars`` -- it is
    ``exhaustion_dwell + cooldown + 3 * min_dwell + 2``, all held at R5's values
    (M22) -- so every cell of the grid and R5's own cell warm the same number of
    bars. Asserted rather than assumed, because a surface whose cells warm
    differently is scoring windows rather than configurations (M21).
    """
    warmups = {R7.gate.machine(R7.TRAINED).warmup_bars} | {
        strategy_for(enter, exit_).warmup_bars for enter, exit_ in constructible_cells()[0]
    }
    assert len(warmups) == 1, f"the grid does not share a warmup: {sorted(warmups)}"
    return warmups.pop()


def run_over(strategies: dict, frame, root: Path, *, first_tradeable: int, stop: int) -> dict:
    """Every strategy over one frame, all trading the same bars.

    Each frame starts ``warmup_bars`` before ``first_tradeable`` so the engine's
    own mask lands on the same timestamp for all of them -- comparing runs over
    different tradeable bars is a defect the charter has corrected once already.
    Here every strategy warms the same bars anyway, and the assertions below are
    what keep that a fact rather than an expectation.
    """
    rows = {}
    for label, strategy in strategies.items():
        row = R7.gate_run(
            strategy, frame, root / slug(label), first_tradeable=first_tradeable, stop=stop
        )
        row["label"] = label
        # Every frame in this phase is a perp with the funding column attached,
        # so a machine reading the neutral crowding fallback here means the
        # column went missing -- which is the whole of M20.
        assert row["crowding_measured"], f"{label} lost its funding column"
        rows[label] = row
    bars = {row["tradeable_bars"] for row in rows.values()}
    assert len(bars) == 1, f"runs covered different tradeable bars: {sorted(bars)}"
    starts = {row["first_tradeable"] for row in rows.values()}
    assert len(starts) == 1, f"runs started on different bars: {sorted(starts)}"
    return rows


def slug(label: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in label)


# --- reporting ----------------------------------------------------------------


def header() -> str:
    return (f"{'run':>39} {'net %':>10} {'sharpe':>9} {'maxDD %':>9} {'trades':>7} "
            f"{'net@3x %':>10}")


def line(label: str, row: dict) -> str:
    sharpe = row["sharpe_tradeable"]
    stress = row["net_by_stress"].get(COST_STRESS_MULTIPLE, float("nan"))
    return (f"{label:>39} {row['net_return_pct']:>+10.4f} "
            f"{'    n/a  ' if sharpe is None else format(sharpe, '>+9.4f')} "
            f"{row['max_drawdown_pct']:>9.4f} {row['trades']:>7} {stress:>+10.4f}")


def write(name: str, payload) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


mark = R7.mark
slim = R7.slim_row


__all__ = [
    "COST_STRESS_MULTIPLE",
    "DECLARED_CELLS",
    "ENTER_ENERGIES",
    "EXIT_ENERGIES",
    "KILL_SWITCH_MAX",
    "KILL_SWITCH_MIN",
    "OUT",
    "R5_TRAINING_TRADES",
    "R7",
    "cell",
    "constructible_cells",
    "header",
    "label_of",
    "line",
    "machine_warmup",
    "mark",
    "run_over",
    "slim",
    "slug",
    "strategy_for",
    "write",
]
