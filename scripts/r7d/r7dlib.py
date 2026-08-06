"""R7d measurement harness -- the lift at matched selectivity.

Run from this directory::

    cd scripts/r7d && python step1_lift.py

**Step 0 is ``scripts/r7b/step0_control.py``, reused rather than rewritten**,
exactly as R7c reused it, and it is run twice: once on a clean ``main`` worktree
with ``R7B_BASELINE=1`` to capture the baseline, then once on this branch. Its
``v1 trained`` row is R5's cell with both energy lifecycles at their defaults, so
reproducing +15.45% / +0.896 / 4.67% / 73 there *is* the claim "energy-first mode
off reproduces R5's row", and its byte-identity check covers all four published
v1/v2 figures against ``main``. A second copy of that script would be a second
thing to keep in step with ``main``.

``R7D_OUT`` chooses where the intermediate JSON goes; it defaults to a temp
directory and must never be ``reports/``. The protocol this harness executes is
fixed by ``docs/plans/2026-08-06-r7d-matched-selectivity.md``, committed at
``94e2969`` before any of the numbers existed.

**The grid is declared in coverage and the feature values are derived from it.**
That is M29 applied, and it is the whole methodological point of the phase.
R7c declared ``enter_energy in {0.35, 0.50, 0.65}`` against a gate declared in
``strength`` rank units, so its tightest cell still admitted 37.9% of bars where
the gate it replaces admits 21.1%, and its turnover kill switch was close to
unreachable by construction. Here the declaration is
:data:`ENTER_COVERAGE_TARGETS` -- **15% / 21% / 30% of measurable bars** -- and
:func:`energy_for_coverage` derives the ``energy`` value hitting each target
**on each frame's own training half**, from the coverage target alone. No value
is ever chosen by how it scores.

**Coverage is the portable quantity, not the threshold** (M18). The ``energy``
value giving 21% coverage on one instrument will differ from another's, and that
is the point: declaring the value would carry BTC's volatility distribution onto
a different instrument, which is a cell transferring rather than a method.

Two things this harness deliberately does not own.

- **The rate metric** is ``scripts/r7``'s, imported rather than reimplemented:
  ``trend_label``, ``forward_er`` and ``rate_table``, with R7's own top-tercile
  rule and its training-half boundaries. R7b's published ``energy <= 0.50`` row
  is reproduced as ``step1``'s control, so the tightened numbers sit on the same
  statistic as the loose one they are compared against.
- **The frames** are ``tests/test_state_machine_gate.py``'s fixture (BTC) and
  ``r7lib.load_eth_frame`` (ETH), the bars R5, R9, R7, R7b and R7c measured,
  funding column attached (M20).

**One step, and that is the protocol working rather than an omission.** The plan
puts a kill switch on the diagnostic before any backtest -- *"if it fails, the
phase stops and SOL is not spent"* -- and R7b's standard is that a harness
carrying steps nothing executed is indistinguishable from one whose numbers were
withheld (M28). What ships here is what ran.

Nothing here writes to the repo, to ``reports/`` or to Postgres.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for entry in (
    str(REPO),
    str(REPO / "src"),
    str(REPO / "scripts" / "r7"),
    str(REPO / "scripts" / "r7c"),
    str(REPO / "scripts" / "r9"),
):
    if entry not in sys.path:
        sys.path.insert(0, entry)

OUT = Path(os.environ.get("R7D_OUT", Path(tempfile.gettempdir()) / "strategy-lab-r7d"))
# The docstring above says this must never be ``reports/``; that is a guard
# rather than a comment, because a report directory is the reproducibility
# record of a run someone chose to publish and none of these are.
if OUT.resolve() == REPO.resolve() or REPO.resolve() in OUT.resolve().parents:
    raise SystemExit(f"R7D_OUT must point outside the repository, not {OUT}")

# R9's own output root, pointed inside this phase's *before* anything imports
# it: ``scripts/r9/step2_dsr`` creates its ``R9_OUT`` at import time, and step 5
# imports that module rather than writing a second deflated Sharpe.
os.environ.setdefault("R9_OUT", str(OUT / "r9-scratch"))

import r7clib as R7C  # noqa: E402
import r7lib as R7  # noqa: E402
import r9lib as R9  # noqa: E402

# R7c's cell machinery, imported rather than rewritten: ``run_over`` (every
# strategy over one frame, asserting they trade the same bars and keep their
# funding column), its printing, and the **tradeability band it declared**, which
# this plan adopts unchanged -- 26-231 round trips, R5's 77 within a factor of
# three. R7c's own ``ENTER_ENERGIES`` grid travels with the import and is not
# read here; the grid this phase declares is the coverage one above.
run_over = R7C.run_over
header = R7C.header
line = R7C.line
slug = R7C.slug
slim = R7C.slim
COST_STRESS_MULTIPLE = R7C.COST_STRESS_MULTIPLE
KILL_SWITCH_MIN = R7C.KILL_SWITCH_MIN
KILL_SWITCH_MAX = R7C.KILL_SWITCH_MAX
R5_TRAINING_TRADES = R7C.R5_TRAINING_TRADES

# The declared grid, in coverage rather than in feature units (M29). Written out
# here rather than described in prose because a grid stated only in
# a progress log cannot be checked against the verdict it produced. Three cells,
# not six: R9 priced a 54-cell search at DSR 0.70 and the discount scales with
# the search, so the exit target is tied to the enter target rather than swept.
ENTER_COVERAGE_TARGETS: tuple[float, ...] = (0.15, 0.21, 0.30)
EXIT_COVERAGE_MULTIPLE = 2.0

# The declared thresholds, as numbers (M23). The verdict horizon first, then the
# two horizons that are declared context and not
# part of any verdict. At H=90 an IC bar would sit inside R7's own noise band;
# the rate metric is not an IC, but the horizons were declared this way in R7
# and R7b and moving them now would make the three phases incomparable.
HORIZON = 30

# Positive in all four instrument-halves, and their mean at least this many
# percentage points. Anchored on what was measured at ``energy <= 0.50``, where
# the four halves average +4.93 pp: the lift may lose about 40% of its size to
# tightening and still pass.
LIFT_BAR_PP = 3.0

# R7b's published single-gate row (§9.7), which ``step1`` reproduces as its
# control before reading any tightened number. A harness that cannot reproduce
# the measurement it is tightening is measuring a different question.
REFERENCE_CEILING = 0.50
REFERENCE_LIFTS_PP = {
    ("BTC", "train"): 3.51,
    ("BTC", "test"): 5.10,
    ("ETH", "train"): 4.49,
    ("ETH", "test"): 6.61,
}
REFERENCE_MEAN_PP = 4.93

# M29's own measurement of the gate this phase matches selectivity against and of
# R7c's tightest cell: 21.1% / 37.9% of BTC's training bars, which is where the
# 21% target comes from. Both are re-measured in ``step1`` rather than copied out
# of a document, and come back 21.76% / 37.22% -- the same bars under the joint
# ``measurable`` predicate the machine walks rather than a per-column dropna.
# M29's finding is unchanged in direction and size; see ``step1_lift``'s docstring.
R5_GATE_COVERAGE = 0.211
M29_ENERGY_COVERAGE = 0.379


# The holdout frame: SOL/USDT perp 4h, whole frame, no split, declared in the
# plan **as a bar
# count** so that a refresh which grew the table would fail here rather than
# quietly evaluate the hypothesis on a different instrument-window than the one
# pre-registered. The window is pinned by timestamp for the same reason
# ``tests/test_state_machine_gate.py`` pins BTC's: ``market_candles``
# accumulates, and a holdout whose right edge moves is not a holdout.
SOL_START = "2020-09-14 04:00:00"
SOL_END = "2026-08-06 08:00:00"
SOL_BARS = 12_914


def load_sol_frame():
    """SOL/USDT perp 4h on the plan's frame, with the funding column attached.

    Built the way ``r7lib.load_eth_frame`` is -- the gate fixture is pinned to
    BTC by construction -- and it runs the same funding coverage guard on the way
    out, so a refresh that outran the stored settlements fails here rather than
    silently charging zero carry across uncovered bars. SOL's first settlement
    precedes its first candle, so unlike BTC there is no permanent leading gap.
    """
    from strategy_lab.backtests.costs import funding_coverage_gaps, window_end
    from strategy_lab.db.candles import load_candles
    from strategy_lab.db.funding import load_funding
    from strategy_lab.features.flow import FUNDING_COLUMN, align_funding_to_bars

    df = load_candles(
        exchange="binance", market_type="perp", symbol="SOL/USDT", timeframe="4h",
        start=SOL_START, end=SOL_END,
    )
    assert len(df) == SOL_BARS, (
        f"the holdout frame is {len(df)} bars, not the pre-registered {SOL_BARS}"
    )
    rates = load_funding(
        exchange="binance", market_type="perp", symbol="SOL/USDT"
    )["funding_rate"]
    covered = rates[(rates.index >= df.index[0]) & (rates.index < window_end(df.index))]
    gaps = funding_coverage_gaps(funding=covered, index=df.index)
    assert not gaps, f"SOL funding coverage gaps: {gaps}"
    return df.assign(**{FUNDING_COLUMN: align_funding_to_bars(df.index, rates)}), rates


def first_sixty_percent(df: pd.DataFrame) -> R7.Halves:
    """A 60/40 cut of a frame that has no declared split.

    The holdout is declared with **no split**, so there is no training half to
    derive its threshold on and the plan's two sentences about it point at
    different sets: the general rule says "each frame's own training half" while
    the sentence naming SOL says "its own bars", which for an unsplit frame is
    all of them. ``step3`` derives on all of them -- resolving toward the more
    specific commitment is R7c's rule for this shape of ambiguity -- and reports
    the value this cut gives beside it, which is R7's rule for it.
    """
    split = int(R7.gate.TRAIN_FRACTION * len(df))
    return R7.Halves(split=split, timestamp=df.index[split])


def machine_inputs(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """The five columns the machine reads, and the bars it can measure at all.

    ``measurable`` is ``StateMachine.run``'s own predicate -- every column of
    ``REQUIRED_COLUMNS`` finite -- and not "both inputs of this gate". R7b's
    caveat 1 settled that: scoring each gate on its own inputs starts 1,345
    training bars earlier, since ``energy`` warms in 503 bars against
    ``direction``'s 1,920, and R7's published rates for the shared gate then do
    not reproduce. This is the set the machine walks and the set every published
    rate in R7, R7b and R7c was computed over.
    """
    frame, crowding_measured = R7.machine_frame(df)
    assert crowding_measured, "the frame lost its funding column; M20 is the whole point"
    return frame, frame.notna().all(axis=1).to_numpy()


def coverage_of(gate: np.ndarray, over: np.ndarray) -> float:
    """What share of the bars selected by ``over`` a boolean gate admits."""
    return float(np.asarray(gate)[over].mean())


def energy_for_coverage(
    energy: pd.Series, defined: np.ndarray, halves: R7.Halves, target: float,
    *, which: str = "train",
) -> dict:
    """The ``energy`` value whose coverage on ``which`` half is ``target``.

    This is the derivation the phase turns on, and it reads the coverage target
    and nothing else -- no forward return, no P&L, no ranking of one value
    against another. It is a quantile of ``energy`` over the bars the machine
    can measure in that half, so the value is a property of the instrument's own
    volatility distribution and the *coverage* is what is held fixed across
    instruments (M18).

    The realised coverage is returned beside the value rather than assumed equal
    to the target: ``features.base.rolling_percentile`` is a rank over a
    480-bar window, so ``energy`` lives on a discrete grid of about 1/480 and a
    quantile lands between its points. Reporting both is what makes the
    derivation checkable by a reader.
    """
    sample = energy[halves.mask(energy.index, which) & defined].dropna()
    value = float(np.quantile(sample.to_numpy(dtype="float64"), target))
    return {
        "target_coverage": float(target),
        "energy_value": value,
        "realised_coverage": float((sample <= value).mean()),
        "derived_on_bars": int(len(sample)),
        "derived_on_half": which,
    }


def cell_for(
    energy: pd.Series, defined: np.ndarray, halves: R7.Halves, target: float,
    *, which: str = "train",
) -> dict:
    """One declared cell: an enter target, and an exit target at twice its coverage.

    Both values are derived by the same rule on the same bars. The exit value
    exceeds the enter value by construction, since coverage is monotone in the
    threshold -- which is the dead band ``StateMachine`` requires, arrived at
    from the coverage declaration rather than asserted on top of it.
    """
    enter = energy_for_coverage(energy, defined, halves, target, which=which)
    exit_ = energy_for_coverage(
        energy, defined, halves, EXIT_COVERAGE_MULTIPLE * target, which=which
    )
    assert exit_["energy_value"] > enter["energy_value"], (
        f"exit_energy {exit_['energy_value']} does not exceed enter_energy "
        f"{enter['energy_value']} at coverage target {target}"
    )
    return {"label": label_of(target), "enter": enter, "exit": exit_}


def label_of(target: float) -> str:
    return f"enter cover {target:.0%} / exit cover {EXIT_COVERAGE_MULTIPLE * target:.0%}"


def lift_rows(
    gate: np.ndarray, defined: np.ndarray, index: pd.Index, labels: dict, halves: R7.Halves
) -> dict:
    """R7's rate table for one gate, at every horizon, keyed by horizon."""
    verdict = R7.verdict_of(gate, index, defined)
    return {horizon: R7.rate_table(verdict, labels[horizon], halves) for horizon in R7.HORIZONS}


def machine_for(cell: dict):
    """R5's trained cell with the two derived energy thresholds set, and nothing else.

    ``min_dwell``, ``cooldown``, ``direction_floor``, ``stability_floor``,
    ``crowding_extreme`` and R5's strength thresholds are not re-derived on any
    frame (M22); the strength pair stops driving the lifecycle the moment the
    energy pair is set, but it is still what ``state.policy`` conditions on.
    """
    from dataclasses import replace

    return replace(
        R7.TRAINED,
        enter_energy=cell["enter"]["energy_value"],
        exit_energy=cell["exit"]["energy_value"],
    )


def strategy_for(cell: dict):
    return R7.gate.machine(machine_for(cell))


def shared_warmup(strategies: dict) -> int:
    """The warmup every strategy in a surface carries, asserted rather than assumed.

    The energy thresholds do not enter ``convergence_bars`` -- it is
    ``exhaustion_dwell + cooldown + 3 * min_dwell + 2``, all held at R5's values
    -- so every cell warms what R5's own cell warms. A surface whose cells warm
    differently is scoring windows rather than configurations (M21).
    """
    warmups = {strategy.warmup_bars for strategy in strategies.values()}
    assert len(warmups) == 1, f"the surface does not share a warmup: {sorted(warmups)}"
    return warmups.pop()


def run_row(strategy, frame, root: Path, *, first_tradeable: int, stop: int) -> dict:
    """``r7lib.gate_run`` for a strategy that may not be a state machine.

    ``gate_run`` ends by reading ``strategy_metadata["crowding_measured"]``,
    which only the state-machine adapters write. ``donchian`` -- R5's own R0
    baseline, and one of the two comparators the holdout threshold names -- has
    no such key and raises there, so the comparator cannot be run through it.

    Everything that could drift is still R7's: the engine call is
    ``tests/test_state_machine_gate.py``'s ``evaluate`` and the scalar is
    ``r7lib.sharpe_of`` over the tradeable bars (M21). What is restated is the
    bookkeeping around them, plus the one change that is the point: the crowding
    flag is read **where it exists** and asserted true -- M20 is what it is for
    -- and recorded as ``None`` for a strategy that reads no funding at all.
    """
    root.mkdir(parents=True, exist_ok=True)
    out = R7.gate.evaluate(
        strategy, frame, root, first_tradeable=first_tradeable, stop=stop
    )

    report = next((root / strategy.name).glob("*Z_*"))
    equity = pd.read_csv(
        report / "equity_curve.csv", index_col=0, parse_dates=True
    )["equity"]
    config = json.loads((report / "config.json").read_text())
    warmup = int(config["warmup_bars"])

    assert abs(float(equity.iloc[warmup - 1]) - R7.CASH) < 1e-9, (
        f"equity moved during warmup: {equity.iloc[warmup - 1]}"
    )
    flat = out["sharpe"] is None
    if flat:
        assert equity.nunique() == 1, "null Sharpe on a curve that moved"
    else:
        assert abs(R7.sharpe_of(equity) - out["sharpe"]) < 1e-9, (
            f"harness Sharpe {R7.sharpe_of(equity):+.6f} != engine {out['sharpe']:+.6f}"
        )

    out["warmup_bars"] = warmup
    out["window_bars"] = len(equity)
    out["sharpe_whole_frame"] = out["sharpe"]
    out["sharpe_tradeable"] = None if flat else R7.sharpe_of(equity.iloc[warmup:])
    out["crowding_measured"] = config["strategy_metadata"].get("crowding_measured")
    out["first_tradeable"] = str(out["first_tradeable"])
    # The per-bar series the deflated Sharpe reads, by ``scripts/r9``'s own
    # definition -- the deflation's skew and kurtosis terms are only consistent
    # with the Sharpe if both are computed off the same returns. ``slim`` drops
    # these two keys on the way to JSON.
    out["equity"] = equity
    out["returns"] = R9.returns_of(equity.iloc[warmup:])
    return out


def run_mixed(strategies: dict, frame, root: Path, *, first_tradeable: int, stop: int) -> dict:
    """``r7clib.run_over`` for a surface that includes a non-machine comparator.

    Same two assertions that make a comparison a comparison -- every strategy
    trades the same bars, starting on the same one -- and the same M20 check
    wherever a strategy is in a position to lose its funding column.
    """
    rows = {}
    for label, strategy in strategies.items():
        row = run_row(
            strategy, frame, root / slug(label), first_tradeable=first_tradeable, stop=stop
        )
        row["label"] = label
        assert row["crowding_measured"] is not False, f"{label} lost its funding column"
        rows[label] = row
    bars = {row["tradeable_bars"] for row in rows.values()}
    assert len(bars) == 1, f"runs covered different tradeable bars: {sorted(bars)}"
    starts = {row["first_tradeable"] for row in rows.values()}
    assert len(starts) == 1, f"runs started on different bars: {sorted(starts)}"
    return rows


def read(name: str):
    return json.loads((OUT / name).read_text())


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


__all__ = [
    "COST_STRESS_MULTIPLE",
    "ENTER_COVERAGE_TARGETS",
    "EXIT_COVERAGE_MULTIPLE",
    "HORIZON",
    "KILL_SWITCH_MAX",
    "KILL_SWITCH_MIN",
    "LIFT_BAR_PP",
    "M29_ENERGY_COVERAGE",
    "OUT",
    "R5_GATE_COVERAGE",
    "R5_TRAINING_TRADES",
    "R7",
    "R9",
    "REFERENCE_CEILING",
    "REFERENCE_LIFTS_PP",
    "REFERENCE_MEAN_PP",
    "SOL_BARS",
    "cell_for",
    "coverage_of",
    "energy_for_coverage",
    "first_sixty_percent",
    "header",
    "label_of",
    "lift_rows",
    "line",
    "load_sol_frame",
    "machine_for",
    "machine_inputs",
    "mark",
    "read",
    "run_mixed",
    "run_over",
    "run_row",
    "shared_warmup",
    "slim",
    "slug",
    "strategy_for",
    "write",
]
