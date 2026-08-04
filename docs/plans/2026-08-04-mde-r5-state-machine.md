# MDE R5 — Rule-Based State Machine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the R4 features into an explicit state machine whose transitions are gated by measured conditioning, and find out whether it beats the R0 baseline out-of-sample.

**Architecture:** A `StateMachine` consuming a feature frame and emitting a `MarketState` per bar, with hysteresis, minimum dwell time and cooldown built into transitions rather than bolted on. A `Policy` maps state to target risk. A thin `Strategy` adapter exposes the whole thing through the existing `SignalSet` contract so it runs on the current engine unchanged.

**Tech Stack:** Python 3.11, pandas, numpy, pytest.

**Charter:** [docs/research/2026-08-03-market-dynamics-engine.md](../research/2026-08-03-market-dynamics-engine.md) — phase R5. Gate: *beats the R0 baseline out-of-sample, with hysteresis + dwell time + cooldown.*

---

## What R4 measured, and what it means for the design

Re-measured on 13,167 BTC/USDT perp 4h bars, IC of `direction` against the `[t+1, t+31]` return, split by `strength` tercile:

| `strength` tercile | IC@30b | first half / second half | n |
|---|---|---|---|
| low | +0.0022 | −0.0361 / +0.0468 | 4,389 |
| **mid** | **−0.1128** | −0.1201 / −0.1101 | 4,389 |
| high | +0.1314 | +0.1953 / +0.0621 | 4,389 |

Unconditional IC is +0.0385.

Three things follow, and they are the whole design:

1. **The conditioning is non-monotone.** A threshold rule — "trade when strength is high" — is not merely suboptimal, it is the wrong shape. The middle tercile carries a *larger* absolute IC than the unconditional signal, with the opposite sign and both halves agreeing. Middling trend quality is a mean-reversion regime, and it is the most usable thing R4 produced.
2. **The low tercile is noise, not an edge.** Its halves disagree in sign. Do not build a state on it; it is where the machine should be flat.
3. **The high tercile decays**, +0.195 → +0.062 across halves. Whatever the machine earns there, expect the out-of-sample half to be the weaker one. This is the single most likely way the R5 gate fails, and it must not be discovered after the fact.

Also carried from R4: `participation` is cut (decision M9). `strength` and `persistence` correlate at +0.674 and carry roughly one feature's worth between them — use one as the primary conditioner, not both. `crowding` is the only non-price input and is consistently negative at every horizon. `compression_release` was untestable by univariate IC — R5 is where it gets judged, on the bars where it fires.

---

## The constraint that bounds this phase

`Portfolio.from_signals` consumes `position_size` **only on the bar that opens a position** — measured in R2, verified against the installed vectorbt: with `size = [1,1,1,1,5,5,5,5]` and an entry every bar, the result is one order of size 1.0 and a position that never resizes.

So the charter's per-state target risk table (Compression 0–5%, Riding 70–100%, Exhaustion 55%…) **cannot be expressed as continuous rebalancing in this phase.** A state change mid-position will not resize it.

Two consequences, both deliberate:

- **R5 sizes at entry only.** The state at the entry bar picks the size; later states can *exit* but cannot scale. Say so in the docstring where a reader will hit it.
- **The exhaustion → distribution taper is deferred to R6**, where the continuous-exposure contract lands. Building a taper here would produce a state machine whose defining behaviour the engine silently ignores — the failure mode R2 found in "volatility targeting" that was really entry scaling.

Do not work around this by emitting repeated entries. `accumulate=False` ignores them, and turning it on changes semantics for every existing strategy.

---

## Out-of-sample is part of the gate, not a later phase

The gate says *beats the R0 baseline out-of-sample*. Formal walk-forward with purging and embargo is R9, but R5 cannot claim its gate without at least an honest split.

**Rule for this phase:** thresholds and dwell parameters are chosen on the **first 60%** of bars and evaluated on the **last 40%**. The test half is looked at once, at the end. Every number reported is labelled with which half it came from.

If a parameter is adjusted after seeing test-half results, that half is burned and the claim is in-sample — say so rather than quietly re-splitting.

---

## Conventions

- `.venv/bin/python -m pytest` and `.venv/bin/ruff check src tests`. Suite is **494 passed** on `main`.
- **Run the full suite before committing, not after.**
- `market_candles` holds 133,620 rows of real research data — **read-only**.
- **Mutation-test every test and assert the mutation applied.** A `.replace()` whose target does not exist reads exactly like a test that cannot fail; a syntax error exits non-zero while proving nothing — check for a **named** failing test. Harness: `scratchpad/mutate_r4.py`.
- Features are loaded through `features.registry`; funding is attached with `align_funding_to_bars(df.index, funding["funding_rate"])`.

---

## File structure

| File | Responsibility |
|---|---|
| `src/strategy_lab/state/machine.py` | `MarketState` enum, `StateMachine`, transition rules |
| `src/strategy_lab/state/policy.py` | state → target risk, entry-time only |
| `src/strategy_lab/strategies/state_machine_v1.py` | `Strategy` adapter emitting `SignalSet` |
| `tests/test_state_machine.py` | transitions, hysteresis, dwell, cooldown |
| `tests/test_state_policy.py` | sizing, and that it is entry-time |
| `tests/test_state_machine_strategy.py` | the adapter, and registry coverage |

---

## Task 1: `MarketState` and the transition skeleton

**Files:**
- Create: `src/strategy_lab/state/machine.py`, `src/strategy_lab/state/__init__.py`
- Test: `tests/test_state_machine.py`

Six states, from the charter's §2.4 lifecycle. The names matter — they are what a reader reasons about when the machine does something surprising.

```
COMPRESSION → BREAKOUT → CONFIRMED → RIDING → EXHAUSTION → RESET → COMPRESSION
```

`RESET` exists so a failed trend has somewhere to go that is not `COMPRESSION`: cooldown applies in `RESET`, and going straight back to `COMPRESSION` would let the machine re-enter on the next bar of the same chop.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import pandas as pd
import pytest

from strategy_lab.state.machine import MarketState, StateMachine


def frame(**columns) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC", name="timestamp")
    return pd.DataFrame(columns, index=index)


def test_the_machine_starts_flat_rather_than_guessing():
    machine = StateMachine()
    assert machine.state is MarketState.COMPRESSION


def test_a_state_is_produced_for_every_bar():
    machine = StateMachine()
    features = frame(direction=[0.0] * 20, strength=[0.1] * 20, crowding=[0.5] * 20)
    states = machine.run(features)
    assert len(states) == len(features)
    assert states.index.equals(features.index)


def test_an_unknown_feature_column_is_rejected_rather_than_ignored():
    """A typo'd column name must not silently disable a transition rule."""
    machine = StateMachine()
    with pytest.raises(KeyError, match="strength"):
        machine.run(frame(direction=[0.0] * 20, crowding=[0.5] * 20))
```

- [ ] **Step 2: Run to verify it fails** — `No module named 'strategy_lab.state'`

- [ ] **Step 3: Implement `MarketState` and a `StateMachine` that produces a state per bar.** Transitions come in Task 2; this task is the skeleton and the input contract.

- [ ] **Step 4: Run tests, commit** (full suite first)

```bash
git add src/strategy_lab/state tests/test_state_machine.py
git commit -m "feat(state): add MarketState and the state machine skeleton"
```

---

## Task 2: Transitions with hysteresis, dwell, and cooldown

**Files:**
- Modify: `src/strategy_lab/state/machine.py`
- Test: `tests/test_state_machine.py`

The three mechanisms the gate names, and what each prevents:

- **Hysteresis** — entering a state needs a higher bar than leaving it. Without it a feature hovering at a threshold flips the state every bar. Use separate `enter_*` / `exit_*` thresholds; never one constant compared twice.
- **Minimum dwell** — a state must persist for *N* bars before any transition out, except the hard exits below. Prevents a one-bar spike from walking the machine through three states.
- **Cooldown** — after `RESET`, the machine cannot re-enter `BREAKOUT` for *M* bars. Prevents re-entering the same chop repeatedly, which is the single largest source of whipsaw cost in the R0 baselines.

**Two hard exits bypass dwell and cooldown**, because refusing to leave on a real break is worse than churning: a `direction` sign flip while in `RIDING`, and `stability` collapsing below its floor. Everything else respects both.

- [ ] **Step 1: Write the failing tests**

```python
def test_hysteresis_stops_a_hovering_feature_from_flipping_every_bar():
    """A feature oscillating around one threshold must not toggle the state."""
    machine = StateMachine(enter_strength=0.30, exit_strength=0.20, min_dwell=1)
    hovering = [0.29, 0.31, 0.29, 0.31, 0.29, 0.31] * 4
    states = machine.run(frame(direction=[0.8] * 24, strength=hovering, crowding=[0.5] * 24))
    assert states.nunique() <= 2, f"state churned across {states.nunique()} values on a hovering feature"


def test_minimum_dwell_blocks_a_one_bar_spike_from_advancing_the_machine():
    machine = StateMachine(min_dwell=5)
    spike = [0.1] * 10 + [0.9] + [0.1] * 10
    states = machine.run(frame(direction=[0.8] * 21, strength=spike, crowding=[0.5] * 21))
    assert states.iloc[10] == states.iloc[9], "a single bar advanced the state despite dwell"


def test_cooldown_prevents_immediate_re_entry_after_a_reset():
    machine = StateMachine(cooldown=8, min_dwell=1)
    pattern = [0.9] * 6 + [0.05] * 3 + [0.9] * 12       # trend, failure, immediate retry
    states = machine.run(frame(direction=[0.8] * 21, strength=pattern, crowding=[0.5] * 21))
    after_reset = states.iloc[9:17]
    assert (after_reset != MarketState.BREAKOUT).all(), "re-entered during cooldown"


def test_a_direction_flip_while_riding_exits_immediately_despite_dwell():
    """Refusing to leave on a real break is worse than churning."""
    machine = StateMachine(min_dwell=20, cooldown=0)
    direction = [0.8] * 12 + [-0.8] * 8
    states = machine.run(frame(direction=direction, strength=[0.9] * 20, crowding=[0.5] * 20))
    assert states.iloc[12] != MarketState.RIDING


def test_every_transition_taken_is_legal():
    """The lifecycle is a cycle; the machine must not jump COMPRESSION -> EXHAUSTION."""
    machine = StateMachine()
    rng = __import__("numpy").random.default_rng(5)
    n = 2000
    states = machine.run(frame(
        direction=rng.uniform(-1, 1, n), strength=rng.uniform(0, 1, n), crowding=rng.uniform(0, 1, n)))
    for previous, current in zip(states.iloc[:-1], states.iloc[1:]):
        assert current in StateMachine.LEGAL_TRANSITIONS[previous], f"{previous} -> {current}"
```

- [ ] **Steps 2–4: run to fail, implement, run to pass.**

- [ ] **Step 5: Mutation-test**

Collapse `enter_strength` and `exit_strength` to one threshold and confirm the hysteresis test fails. Set `min_dwell` to 1 and confirm the dwell test fails. Skip the cooldown counter and confirm that test fails. Assert each mutation applied.

- [ ] **Step 6: Commit** (suite first)

```bash
git commit -m "feat(state): add hysteresis, minimum dwell, and post-reset cooldown"
```

---

## Task 3: The policy — and the non-monotone conditioning

**Files:**
- Create: `src/strategy_lab/state/policy.py`
- Test: `tests/test_state_policy.py`

This is where R4's measurement becomes behaviour.

**Do not write "trade when strength is high".** The measured shape is:

- high `strength` → follow `direction` (IC +0.131)
- **mid `strength` → fade `direction`** (IC −0.113, both halves agreeing)
- low `strength` → flat (IC +0.002, halves disagree in sign — noise)

A monotone threshold discards the middle tercile, which carries the larger absolute IC of the two live regimes.

`crowding` modulates size rather than direction: it is consistently negative at every horizon, so extreme crowding shrinks the target rather than flipping it.

- [ ] **Step 1: Write the failing tests**

```python
def test_high_strength_follows_direction():
    assert target_risk(state=MarketState.RIDING, direction=+0.8, strength=0.9, crowding=0.5) > 0
    assert target_risk(state=MarketState.RIDING, direction=-0.8, strength=0.9, crowding=0.5) < 0


def test_mid_strength_fades_direction():
    """R4 measured IC -0.113 here, both halves agreeing. A monotone rule throws this away."""
    assert target_risk(state=MarketState.RIDING, direction=+0.8, strength=0.5, crowding=0.5) < 0


def test_low_strength_is_flat_because_its_halves_disagreed():
    assert target_risk(state=MarketState.RIDING, direction=+0.8, strength=0.05, crowding=0.5) == 0.0


def test_extreme_crowding_shrinks_the_target_without_flipping_it():
    calm = target_risk(state=MarketState.RIDING, direction=+0.8, strength=0.9, crowding=0.5)
    crowded = target_risk(state=MarketState.RIDING, direction=+0.8, strength=0.9, crowding=0.98)
    assert 0 < crowded < calm


def test_compression_and_reset_are_flat():
    for state in (MarketState.COMPRESSION, MarketState.RESET):
        assert target_risk(state=state, direction=+0.9, strength=0.9, crowding=0.5) == 0.0
```

- [ ] **Steps 2–4: run to fail, implement, run to pass.**

- [ ] **Step 5: Mutation-test**

Make the mid band follow rather than fade and confirm that test fails. Make low strength follow and confirm it fails. Make crowding flip the sign rather than shrink and confirm the crowding test fails.

- [ ] **Step 6: Commit** (suite first)

```bash
git commit -m "feat(state): map state and conditioning to a target risk"
```

---

## Task 4: The `Strategy` adapter

**Files:**
- Create: `src/strategy_lab/strategies/state_machine_v1.py`
- Modify: `src/strategy_lab/strategies/registry.py`
- Test: `tests/test_state_machine_strategy.py`

Exposes the machine through the existing `SignalSet` contract so it runs on the current engine, the replay path, and both existing safety suites unchanged.

Registering it means `tests/test_lookahead.py` and `tests/test_replay_determinism.py` cover it automatically — and they must pass. A state machine reading its own future is the same bug class as a strategy doing so.

**`warmup_bars` is the maximum over every feature it reads**, not a new number. `direction` alone declares 1920.

`position_size` carries the policy's target, with a docstring stating plainly that the engine applies it **at entry only** — a reader who assumes rebalancing will otherwise mis-read every result.

- [ ] **Step 1: Write the failing tests**

```python
def test_it_is_registered_and_covered_by_the_safety_suites():
    assert "state_machine_v1" in list_strategies()


def test_warmup_is_the_deepest_feature_it_reads():
    strategy = get_strategy("state_machine_v1")
    deepest = max(get_feature(name).warmup_bars for name in strategy.features)
    assert strategy.warmup_bars >= deepest


def test_it_emits_no_signal_inside_warmup():
    strategy = get_strategy("state_machine_v1")
    signals = strategy.generate_signals(real_btc_frame(strategy.warmup_bars + 500))
    assert not signals.long_entries.iloc[: strategy.warmup_bars].any()
    assert not signals.short_entries.iloc[: strategy.warmup_bars].any()


def test_position_size_is_present_and_bounded():
    strategy = get_strategy("state_machine_v1")
    size = strategy.generate_signals(real_btc_frame(strategy.warmup_bars + 500)).position_size
    assert size is not None
    assert size.dropna().between(0.0, 1.0).all()
```

- [ ] **Steps 2–5: implement, register in both places, run the safety suites.**

If `tests/test_lookahead.py` fails for `state_machine_v1`, that is a real lookahead bug in the machine or a feature. Fix it; never loosen the probe.

- [ ] **Step 6: Commit** (suite first)

```bash
git commit -m "feat(strategies): expose the state machine through the SignalSet contract"
```

---

## Task 5: The gate — split-sample evaluation against R0

**Files:**
- Create: `tests/test_state_machine_gate.py` (db-marked)
- Modify: `docs/research/2026-08-03-market-dynamics-engine.md`

**This is the phase.** Everything above is machinery.

- [ ] **Step 1: Choose parameters on the first 60% only**

Thresholds, dwell, and cooldown are selected on bars `[0, 0.6n)`. Record what was tried and how many configurations — the charter's §8 rule about multiple-hypothesis control applies, and "we tried 200 and kept the best" needs saying out loud.

- [ ] **Step 2: Evaluate on the last 40%, once**

Run `state_machine_v1` and the R0 baseline (`donchian`, best cell 40/10) over the same test bars, same costs, funding applied. Report for both: total return, Sharpe, max drawdown net of funding, trade count, and time in market.

- [ ] **Step 3: Report the verdict plainly**

**The gate is beating the R0 baseline out-of-sample.** If it does not, say so — that is a result, and the charter's §8 rule is explicit that a component which does not earn its complexity gets deleted rather than tuned until it passes.

Expect this to be hard. R4 measured the high-`strength` regime decaying +0.195 → +0.062 across halves, so the out-of-sample half is the weaker one by construction. If the machine passes only because of the mid-tercile fade, say that too — it is a more interesting finding than a marginal aggregate win.

- [ ] **Step 4: Record in the charter** — a progress row with both halves' numbers, the configuration count from Step 1, and the roadmap status.

---

## Task 6: Document

**Files:** `README.md`, `CLAUDE.md`

`README.md` gets the strategy and how to run it. `CLAUDE.md` gets the two non-obvious properties: the conditioning is non-monotone and a threshold rule is the wrong shape, and `position_size` is applied at entry only so per-state sizing does not rebalance.

---

## R5 GATE

- [ ] `state_machine_v1` passes `tests/test_lookahead.py` and `tests/test_replay_determinism.py`
- [ ] Hysteresis, minimum dwell, and cooldown each pinned by a test that fails without them
- [ ] Every transition taken is legal; no state jumps the lifecycle
- [ ] Parameters chosen on the first 60%, evaluated once on the last 40%, both reported
- [ ] **Out-of-sample comparison against the R0 baseline reported, pass or fail**
- [ ] Full suite green, ruff clean

---

## Self-review notes

**Spec coverage.** Charter §2.4's lifecycle and target-risk table → Tasks 1–3, with the taper deferred and the reason stated. The gate's three named mechanisms → Task 2. "Beats R0 out-of-sample" → Task 5.

**Deliberately out of scope.** Continuous rebalancing is R6 — `from_signals` cannot express it. Formal walk-forward with purging and embargo is R9; this phase uses one honest split, which is the minimum its own gate requires.

**Type consistency.** `MarketState` and `StateMachine(enter_strength=, exit_strength=, min_dwell=, cooldown=)` are used identically in Tasks 1–4. `target_risk(state=, direction=, strength=, crowding=)` matches between Tasks 3 and 4.

**Known risk.** Task 3's mid-tercile fade is a mean-reversion rule inside a trend-following program. If it carries the result, the honest description of `state_machine_v1` is a hybrid, not a trend follower — and the charter's thesis section should say so rather than letting the name imply otherwise.
