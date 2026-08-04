# MDE R4 — State Features v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first state-vector features, each with a univariate diagnostic, so the state machine in R5 is assembled from measured components rather than plausible ones.

**Architecture:** A `StateFeature` protocol mirroring `Strategy` — `name`, `version`, `warmup_bars`, `compute(df) -> pd.Series` — registered so the existing lookahead poison probe covers every feature automatically. A diagnostic harness scores each feature on distribution, persistence, redundancy and forward-return information, rendered as a self-contained report in the house style.

**Tech Stack:** Python 3.11, pandas, numpy, pytest.

**Charter:** [docs/research/2026-08-03-market-dynamics-engine.md](../research/2026-08-03-market-dynamics-engine.md) — phase R4. Gate: *each feature has a univariate diagnostic; no feature dumped in unexamined.*

---

## The trap this phase is built around

Half the charter's candidate inputs are percentiles or z-scores. The obvious implementation reads the whole series:

```python
vol_percentile = realized_vol.rank(pct=True)      # WRONG
```

Measured, poisoning bars after row 120 downward:

```text
full-sample rank  row120:  0.6050 -> 1.0000   CHANGED=True   <-- LOOKAHEAD
rolling  rank     row120:  1.0000 -> 1.0000   CHANGED=False
```

A full-sample rank at bar *t* depends on bars after *t*. This is the exact shape of `_SubtleCheat` in `tests/test_lookahead.py` — full-sample normalization, no `shift(-1)` anywhere — which the poison probe was built to catch and which caught it on 25% of probes where an equivalence sweep caught 1%.

**Every percentile, rank, and z-score in this phase is rolling or expanding. No exceptions.** And the safety net has to actually cover features, which today it does not: the probe iterates `strategies.registry.list_strategies()`, and a feature is not a strategy. Task 5 fixes that, and it is the most important task here.

---

## Scope — nine features over eight dimensions, and what is left out

| Dimension | v1 implementation | Status |
|---|---|---|
| **Direction** | normalized EMA spread, signed −1..1 | build |
| **Strength** | directional efficiency: net move ÷ path length | build |
| **Energy** | realized-volatility percentile, **rolling** | build |
| **Compression** | inverted vol percentile **and its derivative** | build |
| **Persistence** | rolling R² of a linear fit on log price | build |
| **Stability** | residual volatility around that fit | build |
| **Participation** | volume percentile (per-instrument) + `breadth` (cross-sectional, R3) | build |
| **Crowding** | funding z-score, **rolling** | build (crypto only) |
| ~~Confidence~~ | — | **deferred** |

**Confidence is deferred deliberately.** The charter defines it as volume/OI confirmation plus cross-asset agreement — i.e. a composite of the other dimensions rather than a measurement of its own. Building it before the components have diagnostics would bake in weights nobody has evidence for. It belongs after R4, not in it.

**Open-interest inputs are excluded.** Binance serves ~30 days of OI history, so hypothesis C1 is already BLOCKED in the charter. Participation therefore uses volume and breadth, both of which have full history.

---

## Conventions

- `.venv/bin/python -m pytest` and `.venv/bin/ruff check src tests`. Suite is **383 passed** on `main`.
- **Run the full suite before committing, not after.**
- `market_candles` holds 133,620 rows of real research data — **read-only**.
- **Mutation-test every test, and assert the mutation applied.** A `.replace()` whose target string does not exist silently does nothing and reads exactly like a test that cannot fail; a syntax error makes pytest exit non-zero while proving nothing. There is a working harness at `scratchpad/mutate_r3.py` — reuse it.
- **`warmup_bars` is measured, not declared.** `rolling(n)` needs `n`; `ewm(span=n, adjust=False)` needs ~`20n`, because the recursion decays its seed rather than dropping it.

### Feature output convention

Every feature returns a `pd.Series` aligned to the input index:

- **Signed features** (Direction, CompressionRelease) range −1..1. CompressionRelease is a first difference, so it is signed by construction — measured on real BTC 4h it spans −0.77..+0.92.
- **Unsigned features** (everything else) range 0..1.
- Warmup rows are `NaN`, never 0.0 — a zero reads as "measured and neutral", which is a different claim from "not yet measurable".

State the range in each docstring and assert it in each test.

---

## File structure

| File | Responsibility |
|---|---|
| `src/strategy_lab/features/base.py` | `StateFeature` protocol, shared windowed helpers |
| `src/strategy_lab/features/registry.py` | manual registration, mirroring `strategies/registry.py` |
| `src/strategy_lab/features/trend.py` | Direction, Strength, Persistence, Stability |
| `src/strategy_lab/features/volatility.py` | Energy, Compression |
| `src/strategy_lab/features/flow.py` | Participation (volume), Crowding (funding) |
| `src/strategy_lab/features/diagnostics.py` | the univariate report — this is the gate |
| `src/strategy_lab/features/diagnostics_report.py` | self-contained HTML, house style |
| `tests/test_feature_lookahead.py` | poison probe over every registered feature |
| `tests/test_state_features.py` | per-feature behaviour and range |
| `tests/test_feature_diagnostics.py` | the harness itself |

---

## Task 1: `StateFeature` protocol and windowed helpers

**Files:**
- Create: `src/strategy_lab/features/base.py`
- Test: `tests/test_state_features.py`

The helpers exist so no feature hand-rolls a percentile. Every percentile in this phase goes through `rolling_percentile`, which is the single place the causality property has to hold.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_lab.features.base import rolling_percentile, rolling_zscore


def series(values) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="4h", tz="UTC", name="timestamp")
    return pd.Series(values, index=index, dtype="float64")


def test_rolling_percentile_is_causal_where_a_full_sample_rank_is_not():
    """A full-sample rank at bar t moves when bars after t change. Measured:
    row 120 goes 0.605 -> 1.000 under a downward poison. The rolling form does not."""
    clean = series(np.arange(200.0))
    poisoned = clean.copy()
    poisoned.iloc[121:] = -1e6

    assert clean.rank(pct=True).iloc[120] != poisoned.rank(pct=True).iloc[120]
    assert rolling_percentile(clean, window=50).iloc[120] == pytest.approx(
        rolling_percentile(poisoned, window=50).iloc[120]
    )


def test_rolling_percentile_spans_zero_to_one():
    values = rolling_percentile(series(np.random.default_rng(3).normal(size=400)), window=100)
    tail = values.iloc[100:]
    assert tail.min() >= 0.0 and tail.max() <= 1.0


def test_rolling_percentile_leaves_warmup_as_nan_not_zero():
    """NaN says 'not yet measurable'; 0.0 would say 'measured, and it is the minimum'."""
    values = rolling_percentile(series(np.arange(100.0)), window=50)
    assert values.iloc[:49].isna().all()
    assert values.iloc[49:].notna().all()


def test_rolling_zscore_is_causal():
    clean = series(np.arange(300.0))
    poisoned = clean.copy()
    poisoned.iloc[201:] = -1e6
    assert rolling_zscore(clean, window=100).iloc[200] == pytest.approx(
        rolling_zscore(poisoned, window=100).iloc[200]
    )


def test_rolling_zscore_of_a_flat_series_is_zero_not_infinite():
    """Zero variance would divide by zero; the guard must not produce inf."""
    values = rolling_zscore(series(np.full(200, 5.0)), window=50)
    assert np.isfinite(values.iloc[50:]).all()
```

- [ ] **Step 2: Run to verify it fails** — `No module named 'strategy_lab.features.base'`

- [ ] **Step 3: Implement `base.py`**

`StateFeature` as a `Protocol` with `name: str`, `version: str`, `warmup_bars: int`, `compute(df) -> pd.Series`, mirroring `strategies/base.Strategy` so the tooling built for one works for the other.

`rolling_percentile(series, *, window)` — the fraction of the trailing `window` observations at or below the current one, `NaN` until the window fills. `rolling_zscore(series, *, window)` — same windowing, guarding zero variance so a flat series returns 0.0 rather than `inf`.

- [ ] **Step 4: Run tests, mutation-test, commit**

Mutation: change `rolling_percentile` to `series.rank(pct=True)` and confirm the causality test fails. Assert the mutation applied.

```bash
.venv/bin/python -m pytest -q && .venv/bin/ruff check src tests
git add src/strategy_lab/features/base.py tests/test_state_features.py
git commit -m "feat(features): add StateFeature protocol and causal windowed helpers"
```

---

## Task 2: Trend features — Direction, Strength, Persistence, Stability

**Files:**
- Create: `src/strategy_lab/features/trend.py`
- Test: `tests/test_state_features.py`

Four dimensions that all read price, kept in one module because they share the rolling linear fit.

Definitions, so the tests are unambiguous:

- **Direction** = `(EMA_fast − EMA_slow) / ATR`, squashed to −1..1 with `tanh`. Uses `ewm`, so `warmup_bars = 20 × slow_span`.
- **Strength** = directional efficiency = `|P_t − P_{t−n}| / Σ|P_i − P_{i−1}|` over the window. Near 1 = one-way move; near 0 = lots of motion, no net displacement. This is the cleanest trend-versus-chop discriminator in the charter and the one to get right.
- **Persistence** = R² of an ordinary least-squares fit of log price on time over the window.
- **Stability** = `1 − normalized residual volatility` around that same fit.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_state_features.py`:

```python
from strategy_lab.features.trend import Direction, Persistence, Stability, Strength
from tests.conftest import synthetic_ohlcv


def trending(n: int, slope: float = 0.002, noise: float = 0.0003) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    close = 100 * np.exp(np.cumsum(np.full(n, slope) + rng.normal(0, noise, n)))
    index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {"open": close * 0.999, "high": close * 1.002, "low": close * 0.998,
         "close": close, "volume": np.full(n, 500.0)}, index=index)


def choppy(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(12)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {"open": close * 0.999, "high": close * 1.004, "low": close * 0.996,
         "close": close, "volume": np.full(n, 500.0)}, index=index)


def test_direction_is_positive_in_an_uptrend_and_negative_in_a_downtrend():
    feature = Direction()
    up = feature.compute(trending(feature.warmup_bars + 200)).iloc[-1]
    down = feature.compute(trending(feature.warmup_bars + 200, slope=-0.002)).iloc[-1]
    assert up > 0.3 and down < -0.3


def test_direction_stays_inside_minus_one_to_one():
    feature = Direction()
    values = feature.compute(trending(feature.warmup_bars + 500, slope=0.05)).dropna()
    assert values.min() >= -1.0 and values.max() <= 1.0


def test_strength_separates_a_clean_trend_from_chop():
    """The whole point of the dimension: same net motion, different path."""
    feature = Strength()
    n = feature.warmup_bars + 300
    assert feature.compute(trending(n)).iloc[-1] > 0.6
    assert feature.compute(choppy(n)).iloc[-1] < 0.4


def test_persistence_is_high_on_a_straight_line_and_low_on_noise():
    feature = Persistence()
    n = feature.warmup_bars + 300
    assert feature.compute(trending(n, noise=0.0)).iloc[-1] > 0.95
    assert feature.compute(choppy(n)).iloc[-1] < 0.7


def test_stability_falls_when_the_path_gets_ragged():
    feature = Stability()
    n = feature.warmup_bars + 300
    assert feature.compute(trending(n, noise=0.0001)).iloc[-1] > feature.compute(choppy(n)).iloc[-1]


@pytest.mark.parametrize("feature", [Direction(), Strength(), Persistence(), Stability()])
def test_every_trend_feature_leaves_warmup_as_nan(feature):
    values = feature.compute(trending(feature.warmup_bars + 50))
    assert values.iloc[: feature.warmup_bars - 1].isna().all()
    assert values.iloc[-1] == values.iloc[-1]  # not NaN
```

- [ ] **Step 2–4: Run to fail, implement `trend.py`, run to pass.**

- [ ] **Step 5: Mutation-test**

For `Strength`, replace the path-length denominator with the window length — a constant — and confirm the trend-versus-chop test fails. That denominator *is* the feature; without it this is just normalized return.

- [ ] **Step 6: Commit** (suite first)

```bash
git add src/strategy_lab/features/trend.py tests/test_state_features.py
git commit -m "feat(features): add Direction, Strength, Persistence, Stability"
```

---

## Task 3: Volatility features — Energy, Compression

**Files:**
- Create: `src/strategy_lab/features/volatility.py`
- Test: `tests/test_state_features.py`

- **Energy** = rolling percentile of realized volatility. High = the market is moving, regardless of direction.
- **Compression** = `1 − Energy`. Its **derivative** is the interesting part: the charter's point is that compression alone is not a signal, but compression *starting to release* is.

Expose the derivative as `CompressionRelease` — the first difference of Compression, so a caller reads a named feature instead of differencing by hand and getting the sign backwards.

Key test: the charter's own example — `Strength` high with `Energy` low is a slow steady trend; `Strength` low with `Energy` high is violent chop. Those must be distinguishable, or the two dimensions are not carrying separate information.

- [ ] **Steps: failing tests → implement → mutation-test → commit.**

Mutation: make `Energy` use a full-sample rank and confirm the feature lookahead probe (Task 5) catches it. If Task 5 is not yet in place, confirm the causality test in Task 1's helpers catches it.

```bash
git commit -m "feat(features): add Energy and Compression with its release derivative"
```

---

## Task 4: Flow features — Participation, Crowding

**Files:**
- Create: `src/strategy_lab/features/flow.py`
- Test: `tests/test_state_features.py`

- **Participation** = rolling percentile of volume. Cross-sectional participation already exists as `breadth` from R3; this is the per-instrument half.
- **Crowding** = rolling z-score of funding, squashed to 0..1. **Crypto only** — `compute` must raise a clear error when the frame has no `funding_rate` column rather than returning zeros, because a silent zero reads as "not crowded".

Funding arrives on its own schedule (8h against 4h bars, up to 47 ms past the boundary), so the feature takes a funding series and aligns it by **containment**, the same rule `backtests/costs.apply_funding` uses. Reuse that helper rather than writing a second alignment.

- [ ] **Steps: failing tests → implement → mutation-test → commit.**

Test that a missing funding column raises. Mutation: return 0.0 instead of raising, and confirm that test fails.

```bash
git commit -m "feat(features): add Participation and funding-based Crowding"
```

---

## Task 5: Extend the lookahead probe to features — the safety net

**Files:**
- Create: `src/strategy_lab/features/registry.py`
- Create: `tests/test_feature_lookahead.py`
- Test: `tests/test_feature_lookahead.py`

**This is the most important task in the phase.** `tests/test_lookahead.py` iterates `strategies.registry.list_strategies()`, so features are currently uncovered — and half of them are percentiles, the exact construction that leaks the future.

Mirror `strategies/registry.py`: manual registration in two places, so a new feature is covered the moment it is registered.

The probe technique is unchanged and already proven: corrupt every bar after *t*, recompute, assert row *t* is unchanged. `tests/test_lookahead.py` has the poison profiles and the sizing rationale — a 400-bar frame missed a real injected bug on 32.5% of seeds, so use the same `PROBE_SPAN` scale rather than picking a fresh number.

- [ ] **Step 1: Write the test, including a cheat that proves it can fail**

```python
@pytest.mark.parametrize("name", list_features())
def test_registered_features_do_not_look_ahead(name):
    feature = get_feature(name)
    df = synthetic_ohlcv(n=feature.warmup_bars + PROBE_SPAN)
    offenders = poison_probe_feature(feature, df, warm=feature.warmup_bars)
    assert offenders == [], f"{name} used future data at bar indices {offenders}"


@dataclass(frozen=True)
class _FullSamplePercentile:
    """The exact trap this phase is built around: no shift(-1), still non-causal."""
    name: str = "full_sample_percentile"
    version: str = "1.0.0"
    warmup_bars: int = 50

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["close"].pct_change().rolling(20).std().rank(pct=True)


def test_the_feature_probe_detects_a_full_sample_percentile():
    df = synthetic_ohlcv(n=600)
    assert poison_probe_feature(_FullSamplePercentile(), df, warm=50), (
        "the probe cannot see full-sample normalization"
    )
```

- [ ] **Steps: run to fail → implement registry and probe → run to pass.**

If a **registered** feature fails the probe, that is a real bug in the feature. Fix the feature; never loosen the probe.

- [ ] **Commit** (suite first)

```bash
git commit -m "test(features): cover every registered feature with the lookahead probe"
```

---

## Task 6: The univariate diagnostic — the R4 gate

**Files:**
- Create: `src/strategy_lab/features/diagnostics.py`
- Test: `tests/test_feature_diagnostics.py`

The gate is *"each feature has a univariate diagnostic; no feature dumped in unexamined."* This is that diagnostic. The charter's §8 methodology names what it must report:

| Metric | Question it answers |
|---|---|
| coverage (% non-NaN after warmup) | is it computable on real data? |
| distribution (min / median / max / IQR) | is it degenerate — saturated or near-constant? |
| lag-1 autocorrelation | does it persist, or is it noise? |
| **turnover** (mean absolute bar-to-bar change) | what would trading it cost? |
| **forward-return IC** (Spearman, several horizons) | does it carry information at all? |
| split-half stability | does the relationship hold in both halves, or is it one regime? |
| max pairwise correlation with other features | is it redundant? |

Two things to get right, both places research quietly lies to itself:

1. **Forward returns must not overlap the feature's own bar.** Return over `[t+1, t+1+h]`, never `[t, t+h]` — including bar *t* puts the feature's own bar inside its own target.
2. **Split-half stability is what separates a feature from a fitted artifact.** Report IC for each half separately, not just the full-sample number. A feature that works in one half and not the other is a regime, not a signal.

`FeatureDiagnostic` is a frozen dataclass; `diagnose(feature, df, horizons)` returns one.

- [ ] **Steps: failing tests → implement → mutation-test → commit.**

Mutations: compute forward return from bar *t* instead of *t+1* and confirm a test catches the overlap; report only the full-sample IC and confirm the split-half test fails.

```bash
git commit -m "feat(features): add the univariate diagnostic harness"
```

---

## Task 7: Diagnostic report

**Files:**
- Create: `src/strategy_lab/features/diagnostics_report.py`
- Modify: `src/strategy_lab/cli.py`
- Test: `tests/test_feature_diagnostics.py`

Follow `backtests/report.py` and `backtests/sweep_report.py` exactly: module-level colour constants (`_UP = "#26a69a"`, `_DOWN = "#ef5350"`), `_fmt_*` helpers, one `_TEMPLATE` with `__PLACEHOLDER__` substitution, `html.escape` on interpolated values, `json.dumps(..., allow_nan=False)`, and **no external assets**.

One table, one row per feature, columns as in Task 6. Colour IC on the diverging scale so a reader sees at a glance which features carry information and which are noise. Flag redundancy — any pair correlated above 0.9 — since two features that agree that closely are one feature.

Add a `features` CLI command that loads candles, diagnoses every registered feature, and writes the report into a timestamped directory under `reports/`, matching how `backtest` and `sweep` build theirs.

- [ ] **Steps: failing tests → implement → run for real → commit.**

---

## Task 8: Run on real data and record what it says

**Files:**
- Modify: `docs/research/2026-08-03-market-dynamics-engine.md`, `README.md`, `CLAUDE.md`

- [ ] **Step 1: Diagnose every feature on the stored BTC perp 4h series**

```bash
.venv/bin/strategy-lab features --exchange binance --market-type perp \
  --symbol BTC/USDT --timeframe 4h --horizons 1,6,30
```

- [ ] **Step 2: Report the numbers plainly**

For each feature: coverage, IC at each horizon, split-half ICs, turnover, and its strongest correlation with another feature.

**Expect most ICs to be small.** Single features rarely reach |IC| > 0.05 on 4h crypto, and a feature can still earn its place as a *regime filter* while predicting nothing directly — Strength and Energy are conditioners, not predictors. Say which is which rather than ranking everything by IC.

**If a feature has near-zero IC in both halves, no regime-filter story, and high correlation with another feature, say so and recommend cutting it.** The gate is that no feature ships unexamined, not that all eight survive.

- [ ] **Step 3: Record in the charter** — a progress row with the table, and R4's roadmap status.

- [ ] **Step 4: Document** — `README.md` gets the `features` command; `CLAUDE.md` gets the rolling-window rule, which is the non-obvious property a contributor will otherwise violate.

---

## R4 GATE

- [ ] Every registered feature passes the lookahead probe, and the probe is proven able to fail via the full-sample-percentile cheat
- [ ] Every feature has a diagnostic with coverage, IC at ≥2 horizons, split-half ICs, turnover, and max correlation
- [ ] Warmup verified per feature, not declared
- [ ] Redundant pairs (|r| > 0.9) identified and a keep/cut recommendation made for each
- [ ] Full suite green, ruff clean

---

## Self-review notes

**Spec coverage.** Charter §2.1's state vector → Tasks 2–4, eight of nine dimensions, with Confidence deferred and its reason stated. §8's univariate-diagnostic requirement → Task 6. The lookahead risk the phase introduces → Task 5.

**Deliberately out of scope.** The state machine that consumes these is R5. Derivatives of state (§2.2) are only represented by `CompressionRelease`, because differencing a feature with no diagnostic yet compounds an unmeasured thing.

**Type consistency.** `StateFeature(name, version, warmup_bars, compute)` is used identically in Tasks 1–5. `FeatureDiagnostic` / `diagnose(feature, df, horizons)` match between Tasks 6 and 7. `rolling_percentile(series, *, window)` is called the same way in Tasks 1, 3, and 4.

**Known risk.** Task 4's Crowding needs a funding series aligned to bars, and the only existing implementation of that rule lives in `backtests/costs`. If that helper is not importable from `features/` without a circular import, the alignment gets duplicated — and a second copy of the 47 ms containment rule is exactly the drift this plan should avoid. Check the import direction before writing the feature.
