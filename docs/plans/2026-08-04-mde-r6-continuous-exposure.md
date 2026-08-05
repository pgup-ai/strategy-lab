# MDE R6 — Continuous-Exposure Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the engine a second strategy contract that emits a continuously drifting target exposure, so a state machine can taper a position instead of only opening and closing it.

**Architecture:** A `TargetExposure` contract alongside the existing `SignalSet`, executed through `vbt.Portfolio.from_orders(size_type="targetpercent")` rather than `from_signals`. The two contracts coexist: the four original strategies keep the boolean path and their byte-identical results, `state_machine_v1` gains a taper on the new one.

**Tech Stack:** Python 3.11, pandas, vectorbt, pytest.

**Charter:** [docs/research/2026-08-03-market-dynamics-engine.md](../research/2026-08-03-market-dynamics-engine.md) — phase R6. Gate: *second strategy contract; vectorbt replaced for this path.*

---

## Two questions settled by measurement before planning

**Does `from_orders` suffice, or is a custom simulator needed?** It suffices. Measured against the installed vectorbt with `size_type="targetpercent"`:

```
target  : [0.0, 0.3, 0.7, 1.0, 1.0, 1.0, 0.55, 0.55, 0.2, 0.0]
position: [0.0, 30.,  70., 100., 100., 100.,  55.,  55., 20., 0.0]
orders  : 6      (the boolean from_signals path gives 1)
```

Shorts, reversal through zero, and costs work too: target `[0, 1, .5, 0, -.5, -1, -.4, 0]` produces 7 orders, a position that crosses zero into a short, and fees actually charged.

**This materially shrinks the phase.** The design doc's "freeze, migrate, retire" language implied leaving vectorbt entirely; the taper is the thing that could not be expressed, and `from_orders` expresses it. Retiring vectorbt outright is not required to satisfy this gate and is not in scope.

**Do the four original strategies migrate?** No. Their documented results in `STRATEGIES.md` are research of record, and byte-identity has been a hard constraint through every phase since R2. They stay on `SignalSet`. Two contracts coexist — which is what the charter's §3.2 repo decision already anticipated.

---

## The debts this phase settles

Four things have been deferred here, and they are all the same constraint:

1. **R5's state machine can exit but cannot scale.** The charter's per-state taper — Riding 70–100% → Exhaustion 55% → Distribution 20% — has never been expressible. `from_signals` consumes `position_size` only on the bar that opens a position.
2. **R2's "volatility targeting" was really `vol-scaled-entry`** and had to be renamed, because the same constraint made it entry-only.
3. **The design doc's "keep `backtests/engine.py` as legacy, do not extend"** has been violated three times (cost model, funding, cost-stress portfolios) and declined on three PRs with "R6 replaces it".
4. **`signals.side` is a CHECK constraint on four discrete values**, which cannot hold a target that drifts.

---

## The risk that matters most

`tests/test_replay_determinism.py` is the proof that backtest and replay agree, and it is written entirely against the boolean path. **A second contract with no determinism proof is a second contract with no guarantee.**

PR #8 is the cautionary precedent: the state machine's convergence guarantee was broken for weeks while the determinism suite passed, because both its comparisons start at bar 0 and a causal strategy passes those by construction. The suite did not lie — it was answering a narrower question than its name suggested.

**So the determinism proof must cover the continuous path explicitly**, including the primed-runner path added in PR #8. A continuous target is *more* exposed to this than a boolean signal, not less: a boolean signal that is one bar late changes when you enter, while a target that is one bar late changes your size on every bar.

---

## Conventions

- `.venv/bin/python -m pytest` and `.venv/bin/ruff check src tests`. Suite is **603 passed** on `main`.
- **Run the full suite before committing, not after.**
- `market_candles` holds 133,620 rows of real research data — **read-only**.
- **Mutation-test every test and assert the mutation applied.** A `.replace()` whose target does not exist reads exactly like a test that cannot fail; a byte-length-identical mutation restored within the same mtime second leaves a stale `.pyc` CPython reuses. Set `PYTHONDONTWRITEBYTECODE=1`, purge `__pycache__`, and check for a **named** failing test. Harness at `scratchpad/mutate_r4.py`.
- An unquoted space-separated path variable does not word-split in zsh — pytest gets one bogus path and exits non-zero having run nothing. Use `${=VAR}` or literal paths.

---

## Backward compatibility — non-negotiable

The four original strategies must produce **byte-identical** `stats.json`, `trades.csv`, and `equity_curve.csv` on their canonical `STRATEGIES.md` commands. Verify by sha256 and report. `state_machine_v1`'s existing boolean behaviour and its published R5 gate numbers must also not move — the taper is a **new** variant, not a replacement.

---

## File structure

| File | Responsibility |
|---|---|
| `src/strategy_lab/strategies/exposure.py` | `TargetExposure`, `ExposureStrategy` protocol |
| `src/strategy_lab/backtests/exposure_engine.py` | `from_orders` execution path |
| `src/strategy_lab/strategies/state_machine_v2.py` | the state machine with its taper |
| `src/strategy_lab/storage/migrations.py` | target-exposure storage |
| `tests/test_target_exposure.py` | the contract |
| `tests/test_exposure_engine.py` | execution, costs, funding |
| `tests/test_exposure_determinism.py` | the proof, for the new path |

---

## Task 1: The `TargetExposure` contract

**Files:**
- Create: `src/strategy_lab/strategies/exposure.py`
- Test: `tests/test_target_exposure.py`

Mirrors `SignalSet`'s shape so the tooling built for one transfers: a frozen dataclass plus a `Protocol` with `name`, `version`, `warmup_bars`, and one compute method.

```python
@dataclass(frozen=True)
class TargetExposure:
    target: pd.Series          # -1..1, the fraction of risk capital to hold
    metadata: dict = field(default_factory=dict)
```

- [ ] **Step 1: Write the failing test**

```python
def test_target_outside_minus_one_to_one_is_refused():
    """A target above 1 asks for leverage the book does not have; below -1, likewise."""
    with pytest.raises(ValueError, match="target"):
        TargetExposure(target=series([0.0, 1.5]))


def test_a_nan_target_is_refused_rather_than_read_as_flat():
    """NaN means 'not yet measurable'; 0.0 means 'measured, and hold nothing'."""
    with pytest.raises(ValueError, match="NaN"):
        TargetExposure(target=series([0.0, float("nan")]))


def test_warmup_rows_are_expressed_as_a_leading_flat_run_not_NaN():
    exposure = TargetExposure(target=series([0.0, 0.0, 0.5]))
    assert exposure.target.iloc[0] == 0.0


def test_the_protocol_is_satisfied_by_a_minimal_implementation():
    assert isinstance(_ConstantExposure(), ExposureStrategy)
```

Note the deliberate asymmetry with features: a feature's warmup is `NaN` because "not yet measurable" and "measured as zero" are different claims. A *target* has no such ambiguity — before warmup you hold nothing, which is exactly 0.0. Say so in the docstring.

- [ ] **Steps 2–4: run to fail, implement, run to pass, mutation-test, commit.**

---

## Task 2: The `from_orders` execution path

**Files:**
- Create: `src/strategy_lab/backtests/exposure_engine.py`
- Test: `tests/test_exposure_engine.py`

`run_exposure_backtest(*, df, strategy, identity, cost_model, funding, ...)`, mirroring `run_backtest`'s signature so a caller moving between them is not surprised.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_drifting_target_produces_an_order_per_change():
    """The whole point: from_signals would give one order for this."""
    result = run_exposure_backtest(df=flat_frame(10), strategy=_Taper(), ...)
    assert result.order_count == 6


def test_the_position_tracks_the_target():
    """A taper that is not executed is a state machine whose behaviour is ignored."""
    ...
    assert position_fractions == pytest.approx(target.tolist(), abs=0.01)


def test_a_target_crossing_zero_reverses_rather_than_flattening():
    ...


def test_funding_is_charged_on_the_held_fraction_not_a_full_unit():
    """A 20% position pays 20% of the carry -- the taper's whole economic point."""
    ...


def test_costs_are_charged_on_every_resize_not_only_on_entry():
    """Tapering is not free; a model that ignores resize cost flatters it."""
    ...
```

That fourth test is the one that matters most economically. R2 measured BTC perp funding at +11.65%/yr paid by longs; if a tapered position pays full-unit carry, the taper looks worse than it is, and if it pays none, better.

The fifth is its mirror: the taper's cost is turnover, and R5 already showed turnover is what kills an edge at 3× costs.

- [ ] **Steps 2–5: implement, run, mutation-test, commit.**

Mutations: charge funding on a full unit regardless of held fraction (test 4 must fail); charge costs only on sign changes rather than every resize (test 5 must fail).

---

## Task 3: Determinism for the continuous path

**Files:**
- Create: `tests/test_exposure_determinism.py`

**This is the task that keeps the guarantee real**, and it is where PR #8's lesson applies.

Three comparisons, not one:

1. **Whole-history vs streaming**, as the boolean path does.
2. **Primed-runner from mid-history** — the path PR #8 added after discovering both existing comparisons start at bar 0 and therefore cannot see a cold-start divergence.
3. **Target-level equality, not just signal-level.** A boolean path compares *when* you trade; a continuous path must compare *how much*, on every bar. Comparing only entry timestamps would pass a strategy whose sizes are all wrong.

- [ ] **Step 1: Write the failing tests**

```python
def test_streaming_reproduces_the_whole_history_target_on_every_bar():
    """Not just entries -- every bar's target, to float equality."""
    ...
    pd.testing.assert_series_equal(streamed_target, whole_history_target)


def test_a_primed_runner_reproduces_the_target_it_started_late_for():
    ...


def test_the_check_can_fail():
    """A non-causal exposure strategy must break the equality."""
    ...
```

- [ ] **Steps 2–4: implement, run, mutation-test, commit.**

Mutation: shift the target by one bar in the streaming path and confirm test 1 fails. A one-bar shift is the realistic failure and the one a signal-level comparison would miss.

---

## Task 3.5: The rebalance model — added after Tasks 1–3, on evidence

Tasks 1–3 built the path on `size_type="targetpercent"` and documented two hazards
honestly. Both turn out to make this phase's actual deliverable — what the taper is
worth — **unmeasurable**, so they are fixed before Task 4 rather than caveated in
Task 6.

On the stored 15,128-bar BTC/USDT 4h history the path turned 10,000 into **557**,
from **4,996 target changes producing 14,404 fills**. Measured on a 3,000-bar
synthetic whose target changes **41 times**:

```
sizing                          orders     fees   final equity
targetpercent (compounds)         1989      244          9,049
targetvalue @ initial cash        1861      251         11,305
```

Both churn, so **compounding is not the cause — price is.** Holding a constant
*fraction* means selling as price rises to stay at that fraction. Sending a new
target only when it moves ≥ a band from the last one *sent*:

```
band    sent   orders    fees    final
0.00    3000     1861     251   11,305      <- trade on every bar
0.01      42       41     155   10,661      <- 41 orders == 41 decisions
0.25      42       41     155   10,661
```

Two changes follow.

**`targetpercent` → `targetvalue`, anchored to initial cash.** CLAUDE.md states the
repo convention: *entries are sized from initial cash × `position_pct`, never from
current equity*. The continuous path silently broke it. Restoring it makes v1-vs-v2
a comparison of **the taper alone** instead of taper-plus-a-sizing-model-change.

**A `rebalance_threshold`, defaulting to 0.05.** Not a cost optimisation — a
statement about what the strategy is. Between decisions the book holds a fixed
*quantity*, so its fraction of equity **rises with a winner**. Continuous
rebalancing to a constant fraction does the opposite: it trims winners and adds to
losers, a mean-reversion overlay bolted onto a strategy whose thesis is *趋势出现后，
需要一直 trend riding*. Band `0.0` is the honest name for that behaviour, not the
neutral default.

The band is measured against the last target **sent**, not the realized position
fraction — realized fraction depends on fills, which depend on the band, which is a
feedback loop a vectorized path cannot precompute.

This invalidates "the position tracks the target on every bar": under a band it
tracks at **decision bars** and drifts between them by design.

---

## Task 4: `state_machine_v2` — the taper

**Files:**
- Create: `src/strategy_lab/strategies/state_machine_v2.py`
- Modify: `src/strategy_lab/strategies/registry.py`
- Test: `tests/test_state_machine_v2.py`

The same `StateMachine` and policy as v1, emitting a continuous target instead of booleans — so the charter's per-state table finally executes:

| State | Target |
|---|---|
| Compression / Reset | 0.00 |
| Breakout | 0.25 |
| Confirmed | 0.55 |
| Riding | 1.00 |
| Exhaustion | 0.55 |

**v1 stays registered and unchanged.** Its R5 gate numbers are published; this is a sibling, not a replacement, and having both is what makes the taper's contribution measurable.

`warmup_bars` derives from the machine exactly as v1's does.

- [ ] **Steps: failing tests → implement → register → mutation-test → commit.**

**v2 goes in its own registry, not `strategies/registry.py`.** Checked before dispatch: **six** parametrized tests across `test_lookahead.py`, `test_replay_determinism.py` and `test_strategy_metadata.py` iterate `list_strategies()` and every one calls `generate_signals`. Adding an exposure strategy there errors at best and skips silently at worst — and a silent skip is the failure this phase is most likely to ship.

Create `strategies/exposure_registry.py` with `list_exposure_strategies()` / `get_exposure_strategy()`, mirroring how `features/registry.py` is separate from `strategies/registry.py`. The continuous suites (Task 3's determinism, plus a lookahead probe over exposure strategies) iterate *that* registry. Manual registration in two places, same as the other two registries.

The lookahead probe needs an exposure variant: the existing one poisons future bars and compares a `SignalSet`'s boolean fields, so it needs to compare a target series instead. Reuse `tests/test_lookahead.py`'s poison profiles and `PROBE_SPAN` sizing rather than inventing new ones — those lengths are measured, not chosen.

---

## Task 5: Target-exposure storage

**Files:**
- Modify: `src/strategy_lab/storage/migrations.py`, `src/strategy_lab/storage/signals.py`
- Test: `tests/test_exposure_storage.py`

`signals.side` is a CHECK on four discrete values and cannot hold a drifting target. Add a `target_exposure NUMERIC(10,6)` column, nullable, so boolean strategies keep writing exactly what they write today.

Follow the established migration rules: idempotent, a cheap no-op on re-run, no table rewrite and no `AccessExclusiveLock` on a second run. `signals` is append-only via two triggers.

- [ ] **Steps: failing tests → migration → storage → mutation-test → commit.**

---

## Task 6: Measure what the taper is worth

**Files:** `docs/research/2026-08-03-market-dynamics-engine.md`, `README.md`, `CLAUDE.md`

- [ ] **Step 1: Run v1 and v2 over the same R5 test half**

Same split (2023-10-31), same costs, funding applied. Report net return, Sharpe, max drawdown net of funding, trades, and **turnover** for both.

- [ ] **Step 2: Say plainly whether the taper earns its complexity**

The honest possibilities, all worth reporting:

- **It improves risk-adjusted return** — the taper does what the charter claims.
- **It improves drawdown but costs return** — a real trade-off; say which the program wants.
- **It costs more in turnover than it saves in drawdown** — R5 measured the machine surviving 3× costs only after turnover fell 159 → 73 trades. A taper *raises* turnover by construction. This is a plausible outcome and would be a genuine finding.

**Do not tune v2 to beat v1.** The R5 parameters were chosen on the training half; reusing them unchanged is what keeps this comparison honest. If v2 needs different parameters to win, that is a separate experiment on the training half, reported as such.

- [ ] **Step 3: Record in the charter**, update the roadmap, and document both contracts in `CLAUDE.md` — which one to use when, and why both exist.

---

## R6 GATE

- [ ] A continuous target executes as a position that tracks it at decision bars, and drifts between them by design, verified against `from_orders`
- [ ] The continuous path has its own determinism proof, covering primed-runner and target-level equality
- [ ] The four original strategies remain byte-identical
- [ ] `state_machine_v1` unchanged and still registered; v2 is a sibling
- [ ] The taper's contribution measured against v1 on the R5 test half, pass or fail
- [ ] Full suite green, ruff clean

---

## Self-review notes

**Spec coverage.** The gate's "second strategy contract" → Tasks 1–2. "vectorbt replaced for this path" → Task 2 replaces `from_signals` with `from_orders`; retiring vectorbt entirely is explicitly out of scope, with the measurement that justifies it. The charter's §2.4 taper → Task 4.

**Deliberately out of scope.** Migrating the four original strategies (byte-identity is a hard constraint). Retiring vectorbt (`from_orders` expresses what was missing). The meta-model is R7.

**Type consistency.** `TargetExposure(target, metadata)` and `ExposureStrategy` are used identically in Tasks 1–4. `run_exposure_backtest(df=, strategy=, identity=, cost_model=, funding=)` matches between Tasks 2, 3, and 6.

**Known risk.** Task 4 registers v2 into a registry that the *boolean* determinism and lookahead suites iterate. If those suites call `generate_signals` on a strategy that only implements `compute_target`, they will error — or worse, skip silently. Settle that explicitly in Task 4 rather than discovering it in Task 6.
