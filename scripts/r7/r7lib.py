"""R7 measurement harness -- chop/trend state diagnosis.

Run from this directory, in order -- each step reads the previous one's JSON::

    cd scripts/r7 && for step in step0_control step1_label step2_incumbent \\
        step3_features step4_composite step5_enter_strength step6_persistence \\
        step7_outcome step8_eth_replication; do python "$step.py"; done

``R7_OUT`` chooses where the intermediate JSON and the transient report
directories go; it defaults to a temp directory and must never be ``reports/``.
The protocol these scripts execute is fixed by
``docs/plans/2026-08-06-r7-chop-diagnosis.md``, committed before any of the
numbers existed; nothing here selects, tunes, or writes anything.

Three things it deliberately does not own.

- **The frame** is ``tests/test_state_machine_gate.py``'s own fixture, so R7
  measures the bars R5 and R9 measured, with the funding column attached (M20).
- **The label** is ``features.diagnostics.forward_efficiency_ratio``, added
  beside ``forward_return`` in the module that already owns the ``t+1``
  anchoring -- so a chop IC and a direction IC are the same statistic against
  different targets, which is the whole point of §"Method" in the plan.
- **The IC** is ``features.diagnostics._horizon_ic``, called rather than
  reimplemented. A second Spearman would be a second convention.

The one statistic R7 adds is the **rate metric**: what fraction of bars inside
some verdict are "trend", against the fraction over all bars. Its terciles come
from the **training half only** and are applied unchanged to both halves -- the
label may look forward, since it is a target rather than a feature, but it must
not look across the split or a test-half rate is scored against a boundary the
test half helped set.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for entry in (str(REPO), str(REPO / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import tests.test_state_machine_gate as gate  # noqa: E402
from strategy_lab.features import diagnostics as diag  # noqa: E402
from strategy_lab.features.base import rolling_percentile  # noqa: E402
from strategy_lab.features.registry import get_feature, list_features  # noqa: E402
from strategy_lab.state.machine import MarketState  # noqa: E402
from strategy_lab.strategies.state_machine_core import (  # noqa: E402
    DEFAULT_FEATURES,
    build_feature_frame,
)

OUT = Path(os.environ.get("R7_OUT", Path(tempfile.gettempdir()) / "strategy-lab-r7"))

# Declared in the pre-registration, before anything ran. 1 bar is dropped from
# R4's set because ER over one bar is 1 by construction.
HORIZONS: tuple[int, ...] = (6, 30, 90)

# R5's trained cell, and the rank window ``state_machine_v1`` feeds the machine
# through. Neither is re-derived here (M22).
TRAINED = gate.TRAINED_MACHINE
DEFAULT = gate.DEFAULT_MACHINE
RANK_WINDOW = 480

# The plan's numbers, so a step can print pass/fail rather than a reader
# deciding after the fact (M23).
IC_BAR = 0.10
IC_HALF_BAR = 0.05
INCUMBENT_BAR_PP = 10.0
COMPOSITE_BAR_PP = 5.0
RUN_LENGTH_BAR = 6.0
ENTER_STRENGTH_BAR = 1.3974
TREND_TERCILE = 2  # top of three


def load_frame():
    """The gate's module-scoped fixture, called outside pytest. Funding attached."""
    return gate.frame.__wrapped__()


# The ETH replication the plan declares -- charter §9.4's frame exactly:
# 14,650 bars from the first stored funding settlement, split on R5's calendar
# date so the test half is 6,048 bars, identical to BTC's.
ETH_START = "2019-11-27 08:00:00"
ETH_END = "2026-08-03 20:00:00"
ETH_BARS = 14_650
ETH_SPLIT = pd.Timestamp("2023-10-31 00:00:00", tz="UTC")


def load_eth_frame():
    """ETH/USDT perp 4h on §9.4's frame, with the funding column attached.

    Built here rather than through the gate fixture because the gate is pinned
    to BTC by construction -- its identity, window and expected bar count are
    module constants and the R5 protocol is what they encode. The funding
    coverage guard runs on the way out, the same one the gate asserts, so a
    refresh that outran the stored settlements fails here rather than silently
    charging zero carry across uncovered bars.
    """
    from strategy_lab.backtests.costs import funding_coverage_gaps, window_end
    from strategy_lab.db.candles import load_candles
    from strategy_lab.db.funding import load_funding
    from strategy_lab.features.flow import FUNDING_COLUMN, align_funding_to_bars

    df = load_candles(
        exchange="binance", market_type="perp", symbol="ETH/USDT", timeframe="4h",
        start=ETH_START, end=ETH_END,
    )
    rates = load_funding(
        exchange="binance", market_type="perp", symbol="ETH/USDT"
    )["funding_rate"]
    covered = rates[(rates.index >= df.index[0]) & (rates.index < window_end(df.index))]
    gaps = funding_coverage_gaps(funding=covered, index=df.index)
    assert not gaps, f"ETH funding coverage gaps: {gaps}"
    return df.assign(**{FUNDING_COLUMN: align_funding_to_bars(df.index, rates)}), rates


def eth_halves(df: pd.DataFrame) -> Halves:
    split = int(np.searchsorted(df.index, ETH_SPLIT))
    assert df.index[split] == ETH_SPLIT, f"ETH split moved to {df.index[split]}"
    return Halves(split=split, timestamp=ETH_SPLIT)


def split_index(df: pd.DataFrame) -> int:
    return int(gate.TRAIN_FRACTION * len(df))


@dataclass(frozen=True)
class Halves:
    """One frame's two halves, by the declared 2023-10-31 split.

    The mask is built from **timestamps**, not positions, because a feature
    measured from its own warmup is a shorter index than the frame and a
    positional mask would silently mis-align the two by the warmup's length.
    """

    split: int
    timestamp: pd.Timestamp

    def mask(self, index: pd.Index, which: str) -> np.ndarray:
        if which == "train":
            return np.asarray(index < self.timestamp)
        if which == "test":
            return np.asarray(index >= self.timestamp)
        if which == "full":
            return np.ones(len(index), dtype=bool)
        raise ValueError(f"unknown half {which!r}")


def halves_of(df: pd.DataFrame) -> Halves:
    split = split_index(df)
    assert df.index[split] == gate.SPLIT_AT, f"split moved to {df.index[split]}"
    return Halves(split=split, timestamp=df.index[split])


def feature_columns(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Every registered feature on this frame, raw -- the R4 diagnostic's space."""
    return {name: get_feature(name).compute(df) for name in list_features()}


def machine_frame(df: pd.DataFrame):
    """The four columns ``state_machine_v1`` actually feeds the machine.

    ``strength`` and ``stability`` arrive as trailing ranks here, not as raw
    features, which is the space ``enter_strength`` is a threshold in.
    """
    return build_feature_frame(
        df, features=DEFAULT_FEATURES, rank_window=RANK_WINDOW
    )


def ranked(values: pd.Series) -> pd.Series:
    return rolling_percentile(values, window=RANK_WINDOW)


def states_of(df: pd.DataFrame, machine) -> pd.Series:
    frame, crowding_measured = machine_frame(df)
    assert crowding_measured, "the frame lost its funding column; M20 is the whole point"
    return machine.run(frame)


def strategy_warmup(machine) -> int:
    from dataclasses import replace

    return replace(gate.get_strategy("state_machine_v1"), machine=machine).warmup_bars




def forward_er(close: pd.Series, *, horizon: int) -> pd.Series:
    return diag.forward_efficiency_ratio(close, horizon=horizon)


def trend_label(
    er: pd.Series, halves: Halves, *, tercile: int = TREND_TERCILE
) -> tuple[pd.Series, tuple[float, float]]:
    """Boolean "this is a trend", plus the training-half boundaries that set it.

    Terciles are equal-count buckets of the **training half's** defined ER, and
    the resulting cut points are applied unchanged to both halves. NaN ER -- the
    warmup-free tail with no complete forward window, and any window price never
    moved over -- stays NaN rather than becoming False, so a bar with no label
    is excluded from a rate rather than counted against it.
    """
    train = er[halves.mask(er.index, "train")].dropna()
    low, high = (float(x) for x in train.quantile([1 / 3, 2 / 3]))
    if tercile != TREND_TERCILE:
        raise ValueError("only the top tercile is the declared trend label")
    label = pd.Series(np.nan, index=er.index, dtype="float64")
    defined = er.notna()
    label[defined] = (er[defined] > high).astype("float64")
    return label, (low, high)




def ic_table(values: pd.Series, target: pd.Series, halves: Halves, *, horizon: int) -> dict:
    """Spearman IC full-sample and in both halves, by the declared split.

    ``features.diagnostics``' own estimator on each subsample, so the number is
    R4's statistic against a different target rather than a second convention.
    """
    rows = {}
    for which in ("full", "train", "test"):
        paired = diag._paired(values, target)
        paired = paired[halves.mask(paired.index, which)]
        rows[which] = {"ic": diag._ic(paired), "n": int(len(paired))}
    return rows


def r4_style_ic(values: pd.Series, target: pd.Series, *, horizon: int) -> dict:
    """R4's own halves -- the aligned sample cut at its midpoint, not at the split.

    Reported beside the declared-split table because §2 of the plan calls itself
    "R4's table with a different target" while the frame block declares
    2023-10-31, and the two rules are not the same cut. Neither is dropped.
    """
    entry = diag._horizon_ic(values, target, horizon=horizon)
    return {
        "ic": entry.ic,
        "first_half_ic": entry.first_half_ic,
        "second_half_ic": entry.second_half_ic,
        "n": entry.observations,
    }


def rate_table(verdict: pd.Series, label: pd.Series, halves: Halves) -> dict:
    """Base rate of "trend" against the rate inside ``verdict``, per half.

    ``verdict`` is boolean-or-NaN over the same index; bars where either the
    verdict or the label is undefined are dropped from both the base rate and
    the conditional rate, so the two are computed over the same bars.
    """
    rows = {}
    for which in ("full", "train", "test"):
        mask = halves.mask(label.index, which)
        paired = pd.DataFrame({"verdict": verdict[mask], "label": label[mask]}).dropna()
        inside = paired[paired["verdict"] > 0.0]
        rows[which] = {
            "base_rate": float(paired["label"].mean()) if len(paired) else float("nan"),
            "inside_rate": float(inside["label"].mean()) if len(inside) else float("nan"),
            "n": int(len(paired)),
            "n_inside": int(len(inside)),
            "coverage": float(len(inside) / len(paired)) if len(paired) else float("nan"),
        }
        rows[which]["lift_pp"] = 100.0 * (rows[which]["inside_rate"] - rows[which]["base_rate"])
    return rows


def run_lengths(verdict: pd.Series) -> np.ndarray:
    """Lengths of the consecutive True runs of a boolean-or-NaN verdict.

    NaN breaks a run rather than extending or ending it silently: an
    unmeasurable bar is not evidence the verdict continued.
    """
    values = verdict.to_numpy(dtype="float64")
    lengths: list[int] = []
    current = 0
    for value in values:
        if value == 1.0:
            current += 1
        else:
            if current:
                lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return np.asarray(lengths, dtype="float64")


def lag1_autocorrelation(verdict: pd.Series) -> float:
    return diag._pearson(verdict, verdict.shift(1))


def persistence_row(verdict: pd.Series, halves: Halves, which: str) -> dict:
    mask = halves.mask(verdict.index, which)
    sliced = verdict[mask]
    lengths = run_lengths(sliced)
    # Over the bars that carry a verdict, never over the whole slice. A warmup
    # bar is NaN here, and counting it as "not in this state" put the trained
    # machine's training-half COMPRESSION share at 63.5% against the 83.3%
    # ``rate_table`` measures on the same bars -- the warmup is a fifth of that
    # half. ``rate_table`` already pairs on defined rows, which is why the two
    # disagreed and why its number is the one the charter quotes.
    defined = sliced.dropna()
    return {
        "runs": int(len(lengths)),
        "median_run": float(np.median(lengths)) if len(lengths) else float("nan"),
        "mean_run": float(lengths.mean()) if len(lengths) else float("nan"),
        "p25_run": float(np.percentile(lengths, 25)) if len(lengths) else float("nan"),
        "p75_run": float(np.percentile(lengths, 75)) if len(lengths) else float("nan"),
        "max_run": float(lengths.max()) if len(lengths) else float("nan"),
        "share_of_bars": float(defined.mean()) if len(defined) else float("nan"),
        "measurable_bars": len(defined),
        "ac1": lag1_autocorrelation(sliced),
    }



CASH = 10_000.0
FREQ = "4h"


def sharpe_of(equity: pd.Series, *, base: float = CASH, freq: str = FREQ) -> float:
    """``backtests.engine._equity_risk``'s Sharpe, over whatever rows are handed in.

    The first bar's return is measured against ``base`` rather than dropped,
    exactly as the engine does, so a slice starting at the first tradeable bar
    is scored on the same estimator as the whole window. Identical to
    ``scripts/r9/r9lib.py``'s, deliberately: M21's scalar has to be one
    statistic across the two phases or R7's extension is not comparable with
    R9's incumbent.
    """
    returns = equity.pct_change()
    returns.iloc[0] = equity.iloc[0] / base - 1.0
    return float(returns.vbt.returns(freq=freq).sharpe_ratio())


def gate_run(strategy, frame, root: Path, *, first_tradeable: int, stop: int) -> dict:
    """One ``evaluate`` run, plus M21's tradeable-bars scalar.

    The report directory is written under ``R7_OUT`` and deleted by the caller.
    It is deliberately **not** under ``reports/``: a report directory is the
    reproducibility record of a run someone chose to publish, and none of these
    were.
    """
    root.mkdir(parents=True, exist_ok=True)
    out = gate.evaluate(strategy, frame, root, first_tradeable=first_tradeable, stop=stop)

    report = next((root / strategy.name).glob("*Z_*"))
    equity = pd.read_csv(report / "equity_curve.csv", index_col=0, parse_dates=True)["equity"]
    config = json.loads((report / "config.json").read_text())
    warmup = int(config["warmup_bars"])

    assert abs(float(equity.iloc[warmup - 1]) - CASH) < 1e-9, (
        f"equity moved during warmup: {equity.iloc[warmup - 1]}"
    )
    flat = out["sharpe"] is None
    if flat:
        assert equity.nunique() == 1, "null Sharpe on a curve that moved"
    else:
        assert abs(sharpe_of(equity) - out["sharpe"]) < 1e-9, (
            f"harness Sharpe {sharpe_of(equity):+.6f} != engine {out['sharpe']:+.6f}"
        )

    out["warmup_bars"] = warmup
    out["window_bars"] = len(equity)
    out["sharpe_whole_frame"] = out["sharpe"]
    out["sharpe_tradeable"] = None if flat else sharpe_of(equity.iloc[warmup:])
    out["crowding_measured"] = config["strategy_metadata"]["crowding_measured"]
    out["first_tradeable"] = str(out["first_tradeable"])
    return out


def slim_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in ("returns", "equity")}


def write(name: str, payload) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


def read(name: str):
    return json.loads((OUT / name).read_text())


def read_if_present(name: str):
    """``read``, or ``None`` for a step that has not run yet."""
    path = OUT / name
    return json.loads(path.read_text()) if path.exists() else None


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, MarketState):
        return value.value
    return str(value)


def verdict_of(mask: pd.Series | np.ndarray, index: pd.Index, defined: np.ndarray) -> pd.Series:
    """A boolean verdict as float 1.0/0.0, NaN wherever it is not measurable."""
    values = np.asarray(mask, dtype="float64")
    return pd.Series(np.where(defined, values, np.nan), index=index, dtype="float64")


def mark(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


__all__ = [
    "DEFAULT",
    "HORIZONS",
    "Halves",
    "OUT",
    "RANK_WINDOW",
    "TRAINED",
    "feature_columns",
    "forward_er",
    "gate",
    "gate_run",
    "halves_of",
    "ic_table",
    "load_frame",
    "machine_frame",
    "mark",
    "persistence_row",
    "r4_style_ic",
    "ranked",
    "rate_table",
    "read",
    "run_lengths",
    "slim_row",
    "split_index",
    "states_of",
    "strategy_warmup",
    "trend_label",
    "verdict_of",
    "write",
]
