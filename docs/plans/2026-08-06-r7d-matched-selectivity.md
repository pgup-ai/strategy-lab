# R7d — the lift at matched selectivity: pre-registration

**Committed before any R7d figure is computed.** Same rule as R7, R7b, R7c, R9
and the ETH replication: the commit adding this file precedes the commit adding
the numbers, and the two are separate.

**Charter:** [docs/research/2026-08-03-market-dynamics-engine.md](../research/2026-08-03-market-dynamics-engine.md).
Redoes R7c under **M29**, and **SOL is still clean** because R7c's kill switch
fired before spending it (M30).

---

## The question, and why it is worth one more phase

Three phases have returned null on the state machine: R7 (states carry no chop
information, 0/6 and 0/9), R7b (composite gate 0/4 and 0/4), R7c (the
energy-first lifecycle does not trade). Against that, **exactly one measurement
has come back positive and replicated** — `energy ≤ 0.50`'s trend-rate lift at
H=30:

| | BTC train / test | ETH train / test |
|---|---|---|
| `energy ≤ 0.50` lift, pp | +3.51 / +5.10 | +4.49 / +6.61 |

Positive in all four instrument-halves. That is the thread, and R7c failed to
test it properly: its grid was declared in `energy` units against a gate declared
in `strength` rank units, so its *tightest* cell still admitted 37.9% of bars
where `strength ≥ 0.80` admits 21.1%, and the turnover kill switch was
near-unreachable by construction (M29).

**What nobody has measured is whether the lift survives being tightened.** R7
measured it by tercile; R7b measured ceilings at 0.50, 0.65 and 0.80. At the
coverage that would actually match R5's gate — around 21%, an `energy` value near
0.155 — the lift is **unmeasured**. If it vanishes there, the energy-first
hypothesis is dead and no holdout needs spending to learn it.

---

## Fixed here, before anything runs

### The grid is declared in **coverage**, not in feature units

This is M29 applied, and it is the whole methodological point of the phase.

- **Enter coverage targets: {15%, 21%, 30%}** of measurable bars. 21% is R5's
  own gate coverage; the others bracket it.
- **Exit coverage = 2× enter coverage**, giving the dead band hysteresis needs
  without a second axis to search. **Three cells**, not six — R9 priced a 54-cell
  search at DSR 0.70 and the discount scales with the search.
- The `energy` **value** that hits each target is **derived on each frame's own
  training half**, mechanically, from the coverage target alone. It is never
  chosen by how it scores.
- **Coverage is the portable quantity, not the threshold.** The `energy` value
  giving 21% coverage on SOL will differ from BTC's, and *that is the point* —
  declaring the value would carry BTC's volatility distribution onto a different
  instrument, which is the M18 failure (a cell does not transfer; a method does).

### Everything else

Held at R5's trained cell — `min_dwell=4`, `cooldown=4`, `direction_floor=0.10`,
`stability_floor`, `crowding_extreme`. **M22 is in force**: nothing else is
re-derived, on any frame. The energy-first mode already exists from R7c and is
off by default; `require_comparable_windows` already refuses a mismatched
`rank_window`.

**Frames:** BTC/USDT perp 4h on R5's frame and split; ETH/USDT perp 4h on §9.4's;
SOL/USDT perp 4h, 12,914 bars, whole frame, no split. Engine defaults throughout:
10bp/side, `opposite_signal_only`, fixed sizing, `position_pct=0.95`, 10,000
initial, **net of funding**, funding column attached (M20). Every statistic at
**4h bar frequency**; the verdict horizon is **H=30**, with H=6 and H=90 as
declared non-verdict context.

**BTC and ETH remain burned** for this hypothesis — R7b read `energy`'s lift on
both test halves. Their R7d numbers are reported and are **not evidence**. SOL is
the only test that counts.

---

## What R7d measures, in order — and it can stop at either of two switches

### 0. Controls

1. The energy-first mode off reproduces R5's row: +15.45% / +0.896 / 4.67% / 73.
2. **Byte-identity against `main`** for all four published v1/v2 rows, from a
   baseline regenerated on a clean `main` worktree.
3. Both safety suites and both exposure suites pass.

### 1. **First kill switch — does the lift survive tightening?**

The cheap one, and it runs before any backtest. At each enter-coverage target,
measure `energy`'s trend-rate lift at H=30 on **all four instrument-halves** of
BTC and ETH, the same rate metric R7 and R7b used.

**If it fails, the phase stops and SOL is not spent.** Nothing about a P&L can
rescue a gate whose only positive measurement disappears when it is asked to be
as selective as the gate it replaces.

### 2. Selection

Highest Sharpe over **tradeable bars** (M21) across the three cells, on **BTC's
training half only**. Full surface reported.

### 3. **Second kill switch — tradeability**

R7c's, unchanged and still right: the selected cell must trade **26–231** round
trips on BTC's training half, R5's 77 within a factor of three. **If it fires,
SOL is not spent.**

### 4. Evaluation

BTC test half and ETH full frame, both labelled **burned**. Then **SOL, whole
frame, one run**, at the coverage target selected in step 2 with its own
`energy` value derived on its own bars.

### 5. Deflated Sharpe

Three trials, 4h bar frequency, `E[max SR]` and the DSR beside the observed.

### 6. Purging and embargo — stated, not performed

Unchanged reasoning, re-checked: the machine fits nothing and has no labels.

---

## Declared thresholds

| Claim | Clears if |
|---|---|
| Controls (§0) | R5's row reproduces to the published digits; all four v1/v2 rows bit-identical to `main` |
| **First kill switch (§1)** | at the selected coverage target, `energy`'s H=30 lift is **positive in all four instrument-halves** *and* their **mean is ≥ +3.0 pp** |
| **Second kill switch (§3)** | the selected cell trades **26–231** round trips on BTC's training half |
| **The holdout (§4)** | on SOL's full frame the energy-first machine beats **both** R5's trained cell run unmodified **and** `donchian` 40/10 on Sharpe net of funding, **and** survives 3× costs with a positive net return |
| The selection is not luck (§5) | observed Sharpe > `E[max SR]`, **and** DSR ≥ **0.95** |

The +3.0 pp mean is anchored on what was measured at `energy ≤ 0.50`, where the
four halves average **+4.93 pp**. So the lift may lose about 40% of its size to
tightening and still pass; losing more than that, or any sign flip, is the signal
that the effect was a property of the loose threshold rather than of `energy`.

---

## Declared before looking

| Outcome | Reading |
|---|---|
| First switch fails | **the lift was a property of the loose threshold, not of `energy`.** The one positive measurement in four phases does not survive being asked to be selective, the state-machine line is exhausted, and R8 should not be built on this state vector. SOL unspent. |
| First clears, second fires | `energy` discriminates chop at matched selectivity but still cannot be traded as a lifecycle. That separates the estimator from the execution for the third time, and the next question is an execution one. SOL unspent. |
| Both clear, holdout clears | the lifecycle was built on the wrong axis, M29 was the only thing hiding it, and R8 finally has inputs and a shape worth modelling. |
| Both clear, holdout fails | `energy` discriminates chop at matched selectivity, trades, and **still does not earn** out of sample. The program's problem is execution rather than estimation — a different research programme from the one §2 describes. |

The first row is the likely one and the cheapest to reach, which is why it runs
first. **If it fires, this is the last phase of the state-machine line** and the
program's remaining unbuilt value is the live path (R10).

**No R7d result changes a published R5, R6, R7, R7b, R7c or R9 figure.**

---

## Method

`scripts/r7d/`, reusing R7's rate metric, R7b's control harness and R7c's cell
machinery. `market_candles`, `funding_rates` and `signals` are read-only
throughout.

---

## Deliverable

A charter §9.9 section, a §9 progress entry, a §10 decision row, and the R7d/R8
roadmap rows. If either kill switch fires, that is the whole deliverable and the
holdout survives.
