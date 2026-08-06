# R7b — the state vector's missing axis: pre-registration

**Committed before any R7b figure is computed.** Same rule as R7, R9 and the ETH
replication: the commit adding this file precedes the commit adding the numbers,
and the two are separate.

**Charter:** [docs/research/2026-08-03-market-dynamics-engine.md](../research/2026-08-03-market-dynamics-engine.md).
A new phase between R7 and R8, because M25 blocked R8 on its **inputs** rather
than on its model: a better estimator over a state vector missing its chop axis
is a better estimator of the wrong thing.

---

## The question

R7 measured two things that compose into one change.

1. **The entry gate is one feature.** `advancing` is
   `strength ≥ enter_strength AND |direction| ≥ direction_floor`, and the second
   term admits 82% of bars and beats `strength` alone by **+0.00 pp** on the test
   half at every horizon.
2. **`energy` is the only feature carrying chop information**, and the machine
   does not read it — `DEFAULT_FEATURES` is
   `(direction, strength, stability, crowding)`. IC vs forward ER is −0.0906 on
   BTC (missing R7's bar by 0.0094) and **−0.1521 on ETH, clearing**, same sign,
   both halves, both instruments.

[§2.1](../research/2026-08-03-market-dynamics-engine.md#21-the-state-vector) named
this two years of decisions ago, and `features/volatility.py`'s own docstring
repeats it:

```
Strength = 80, Energy = 30   →  slow, steady, mature trend
Strength = 20, Energy = 95   →  violent two-way chop
```

So R7b asks one question: **does gating entries on the second axis as well as the
first do what §2.1 says it should?**

---

## Fixed here, before anything runs

### The change

One predicate, added to `advancing`:

```
advancing = (strength >= enter_strength)
          & (|direction| >= direction_floor)
          & (energy <= energy_ceiling)
```

**`direction_floor` stays.** R7 measured it inert *in the entry gate*, but it has
a second job — the reversal test (`state/machine.py`'s `flipped`) — so removing
it would change when a position flips side, which is not what this phase is
testing. The dead term is being **joined**, not replaced.

`energy` is already a rolling percentile over a 480-bar window, the same window
`state_machine_v1` ranks `strength` and `stability` over, so `energy_ceiling`
lives in the same 0..1 rank space as `enter_strength` and needs no new
transformation. This is why the change is one predicate and not a feature
pipeline.

`energy_ceiling` **defaults to 1.0**, which is a no-op, so `state_machine_v1`
and `v2` keep their published behaviour by construction rather than by
inspection. `energy`'s declared warmup (503) is below `direction`'s (1920), so
no strategy's `warmup_bars` moves.

### The three frames

- **BTC/USDT perp 4h** — R5's frame exactly (15,118 bars from 2019-09-10 08:00,
  split 2023-10-31). **Selection happens here, on the training half only.**
- **ETH/USDT perp 4h** — §9.4's frame exactly. Confirmation.
- **SOL/USDT perp 4h** — **the holdout**, fetched for this phase and touched
  **once**. 12,914 bars, 2020-09-14 04:00 → 2026-08-06 08:00, zero funding
  coverage gaps. **No split**: the whole frame is evaluated in one run, so there
  is no split-choice degree of freedom to spend. Disclosed because it was seen
  while characterising the frame: buy-and-hold is +5,411.88% over the first 60%
  and −60.85% over the last 40%.

**Why a third instrument at all.** R7 measured `energy`'s chop IC on BTC's *test*
half and on ETH. Both are therefore informed for this hypothesis, and a result on
either carries the selection discount R9 spent a phase pricing. SOL is the only
frame where "does the energy gate work" has never been asked.

### Engine settings

Engine defaults throughout, as R5/R6/R7/R9: 10bp/side, `opposite_signal_only`,
fixed sizing, `position_pct=0.95`, 10,000 initial, **everything net of funding**,
funding column attached (M20). Frequency for every statistic below: **4h bars**,
the native frame — declared, because R9's was not (M23).

### What is not re-derived

**Nothing but `energy_ceiling`.** `enter_strength`, `exit_strength`, `min_dwell`
and `cooldown` are held at R5's trained cell. M22 is in force: the published cell
is not re-searched, on any frame, in this phase.

The grid is **four values, deliberately tiny**: `energy_ceiling ∈ {0.50, 0.65,
0.80, 1.00}`, where **1.00 is the control** and reproduces R5 exactly. R9 priced
a 54-cell search at DSR 0.70; three live trials is the smallest search that can
answer the question at all, and its deflation is declared below rather than
computed afterwards.

---

## What R7b measures, in order

### 0. Controls, before any R7b number is read

1. `energy_ceiling = 1.0` reproduces R5's published row on BTC — **+15.45% /
   +0.896 / 4.67% / 73 trades**. If the no-op is not a no-op, nothing after it
   means anything.
2. `state_machine_v1` and `v2`'s published figures are **byte-identical** to
   `main`'s, the constraint that has held since R2.
3. Both safety suites pass for the modified machine: the lookahead poison probe
   and replay determinism. A strategy failing either is not safe to trade.

### 1. The diagnostic gate — and it runs before any P&L

R7's **own** composite threshold, reused verbatim: the composite must beat
**each** single-feature gate by ≥ **5 pp** on the trend rate, **in both halves**,
at H=30. R7 measured `strength AND |direction|` at +0.00 pp against `strength`
alone; this asks the same question of `strength AND energy`.

**If the energy gate fails this, the P&L runs are not read.** The hypothesis is
that a second axis adds chop discrimination — that claim is settled by the rate
metric, not by a return, and R7 exists precisely because a P&L number cannot say
whether the estimator or the policy is wrong.

### 2. Selection, on BTC's training half only

Highest Sharpe over **tradeable bars** (M21), across the four cells. One
selection, one parameter, reported with its full surface.

### 3. Evaluation

| frame | how it is read |
|---|---|
| BTC test half | **informed** — `energy` was chosen knowing its IC here. Reported, labelled, not used as evidence of transfer. |
| ETH full frame | **informed**, same reason. The R7 IC that motivated this was measured on it. |
| **SOL full frame** | **the holdout.** One run. Zero parameters derived from it. |

### 4. Deflated Sharpe on the new selection

Three live trials, at **4h bar frequency**. Report `E[max SR]` under the null and
the DSR beside the observed, as R9 did — and, unlike R9, against the threshold
declared below.

### 5. Purging and embargo — stated, not performed

Unchanged from R9's reasoning and re-checked rather than restated: the machine
fits nothing and has no labels. One parameter selected by grid search is not a
fitted model with a forward horizon. They become mandatory at R8.

---

## Declared thresholds

| Claim | Clears if |
|---|---|
| The no-op control | reproduces +15.45% / +0.896 / 4.67% / 73 to the published digits |
| The diagnostic gate (§1) | composite beats **both** single-feature gates by ≥ **5 pp**, **both halves**, H=30 |
| **The holdout (§3)** | on SOL's full frame the energy-gated machine beats **both** (a) the same cell at `energy_ceiling = 1.0` and (b) `donchian` 40/10, on Sharpe net of funding — **and** survives 3× costs with a positive net return |
| The selection is not luck (§4) | observed Sharpe > `E[max SR]` under the null, **and** DSR ≥ **0.95** |

The DSR threshold is stated as a number because R9's was not, and R9 could not
then cite its own table for a verdict (M23). **0.95 is the conventional bar and
R9's 54-cell search scored 0.70 against it.** A three-trial search that cannot
clear it has no defence left.

---

## Declared before looking

| Outcome | Reading |
|---|---|
| Diagnostic clears **and** holdout clears | §2.1's second axis is real and was missing. Wire it, and R8 has inputs worth modelling. |
| Diagnostic clears, holdout fails | the axis discriminates chop but the machine cannot **trade** the distinction — an execution/policy problem, not a state problem. R8 proceeds on the widened inputs; the policy is what gets re-aimed. |
| Diagnostic fails | **the hypothesis is dead and the P&L is not read.** `energy` carries chop information univariately and adds none to `strength`. |
| Diagnostic fails **and** R7's null already stands | chop is not recoverable from the nine features at 4h by any gating of them. The program needs a different frequency or a different data source — order book, OI, cross-sectional — and **R8 should not be built at all** on this state vector. |

The last row is the one this phase exists to be able to reach. It is declared
here so that reporting it is the protocol working rather than a disappointment to
be softened — and it is the row that would send the program back to
[§2.1](../research/2026-08-03-market-dynamics-engine.md#21-the-state-vector) to
ask what a chop axis would have to be measured *from*.

**No R7b result changes a published R5, R6, R7 or R9 figure.** Where they
disagree, both stand as measured and the disagreement is the finding.

---

## Method

`scripts/r7b/`, reusing R7's harness for the rate metric and the gate's own
`evaluate` for the engine runs, as R9 did. The machine change is in
`state/machine.py` and `strategies/state_machine_core.py`; it is a new predicate
and a new default, not a new strategy — `state_machine_v1`/`v2` keep their names,
their registries and their published numbers.

`market_candles`, `funding_rates` and `signals` are read-only throughout, except
for the one-time SOL backfill that preceded this plan and is already stored.

---

## Deliverable

A charter §9.7 section with the tables, a §9 progress entry, a §10 decision row
for whichever outcome lands, and the R7b/R8 roadmap rows. If the diagnostic
fails, that roadmap update is the substantive one and R8 does not start.
