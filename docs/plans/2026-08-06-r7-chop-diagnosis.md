# R7 — chop/trend state diagnosis: pre-registration

**Committed before any R7 figure is computed.** Same rule as R9 and the ETH
replication: the commit adding this file precedes the commit adding the numbers,
and the two are separate.

**Charter:** [docs/research/2026-08-03-market-dynamics-engine.md](../research/2026-08-03-market-dynamics-engine.md),
phase R7 — a **reused slot**. The logistic sizing meta-model that held it is
dropped under M24, not deferred.

---

## The question

The program exists to answer one thing: **ride trends, sit out chop, and the
hard part is telling which you are in.** Every measurement of that so far has
been a **P&L** number — R5's gate, R6's comparison, ETH's replication, R9's
walk-forward. A P&L number scores the estimator and the policy laid over it as
one quantity and cannot say which of the two is wrong.

`COMPRESSION` is the machine's answer to "is this chop". **It has never been
scored as an answer to that question** — only as one input to a traded result.
R7 scores it directly, against a declared label, alongside every other candidate
the repo already has.

R9 supplies the lead. `enter_strength` is the *only* axis stable across all nine
walk-forward folds, it wins at the **most selective value ever tried**, and it
has sat on the swept range's edge in every study that has touched it (R5, the ETH
replication, R9). The gate it controls **is** the chop filter. Either the grid
stopped short of the real optimum, or 0.80 is an interior optimum of a wider grid
that nobody has drawn. That is one cheap measurement and it is declared below.

---

## Fixed here, before anything runs

- **Frame:** BTC/USDT perp 4h, anchored at the first stored funding settlement
  (2019-09-10 08:00), 15,118 bars — R5's frame exactly. Split **2023-10-31**,
  R5's split exactly. **ETH/USDT perp 4h is declared now as the replication**,
  same protocol, run after BTC and never used to choose anything.
- **Frequency:** every IC, rank and rate below is computed on **4h bars**, the
  native frame. No aggregation to daily. Declared because R9's deflation
  frequency was not, and moved the number (M23).
- **The label — forward efficiency ratio.** For horizon *H*,

  ```
  ER[t, H] = |close[t+1+H] − close[t+1]| / Σ|close[i] − close[i−1]|,  i ∈ (t+1, t+1+H]
  ```

  ER near 1 is a clean directional move; ER near 0 is price travelling a long
  path to nowhere, which is what 震荡 names. **Anchored at `close[t+1]`, not
  `close[t]`** — the charter's standing rule, and here it is load-bearing twice
  over, since every candidate feature is a function of `close[t]` and it appears
  in both the numerator and the path sum.
- **Horizons:** *H* ∈ **{6, 30, 90}** bars (1 day, 5 days, 15 days). R4's 1-bar
  horizon is dropped: ER over one bar is 1 by construction.
- **Terciles for the rate metrics come from the training half only** and are
  applied unchanged to both halves. The label may look *forward* — it is a
  target, not a feature — but it must not look *across the split*, or a test-half
  rate is scored against a boundary the test half helped set.
- **Nothing is fitted.** R7 measures; it does not tune, and it registers no
  strategy. M22 is in force: no cell is re-derived anywhere in this phase.

---

## What R7 measures

### 1. The incumbent, scored as a chop detector

`COMPRESSION` against the label, both halves, at all three horizons: the base
rate of "trend" (top tercile of ER) over all bars, and the rate *inside*
`COMPRESSION`. Same shape R5 used to judge `compression_release`, applied to the
state the whole machine rests on. Report `RIDING` and `CONFIRMED` the same way —
if the machine's states carry chop information, the ordering across states is
where it shows.

### 2. Every registered feature, scored the same way

All nine in `features/registry.py`, Spearman IC against forward ER, **full sample
and both halves**, at all three horizons. This is R4's table with a different
target: R4 asked "does this predict *direction*", R7 asks "does this predict
*whether direction is worth trading*". Several features that failed R4's bar may
clear this one — that is the hypothesis, and `energy`/`compression` are the two
designed for it.

### 3. The composite gate

`strength ≥ enter_strength AND |direction| ≥ direction_floor` — the actual
entry condition — scored as a binary detector at R5's trained values, against the
same label. **A composite that beats its own best component is the finding that
would justify the state machine's shape**; one that does not says the machine is
carrying two features where one would do.

### 4. The `enter_strength` extension — R9's lead

Extend the axis to **{0.85, 0.90, 0.95}**, holding the other three at R5's
trained values, scored on the **training half only** by M21's scalar. This is a
grid extension, **not a re-selection**: the winner is not re-derived and no
published figure moves. Its whole purpose is to say whether R9's ridge is a
boundary artifact.

### 5. Persistence

For every candidate that clears §"Declared thresholds", the run-length
distribution of its chop verdict and its lag-1 autocorrelation. A detector that
flips every other bar is useless at any IC, because a position cannot follow it —
and turnover is the one thing R5's cost stress has repeatedly turned on.

---

## Declared thresholds

Stated as numbers, because R9's outcome table declared a comparison without one
and could not then be cited for a verdict (M23).

| Claim | Clears if |
|---|---|
| A feature detects chop | \|Spearman IC\| **≥ 0.10** against ER at *H* = 30, **and** both half-sample ICs agree in sign, **and** each is ≥ **0.05** in absolute value |
| The incumbent `COMPRESSION` detects chop | trend rate inside it is **≥ 10 pp below** the base rate, **in both halves** |
| The composite gate earns its second input | its trend rate beats **both** single-feature gates by ≥ **5 pp**, both halves |
| A candidate is usable | median run length **≥ 6 bars** (one day) — below that no position can follow it |
| `enter_strength` 0.80 was a boundary artifact | any of {0.85, 0.90, 0.95} beats **+1.3974** (M21 scalar, training half) |

R4's bar was \|IC\| 0.07 and nothing cleared it. **0.10 is deliberately higher**,
not lower: chop is a magnitude question rather than a sign question, so it should
be the easier target, and a bar set below R4's would let R7 declare success on
weaker evidence than the phase that found nothing.

---

## Declared before looking

| Outcome | Reading |
|---|---|
| Something clears **and** beats `COMPRESSION` | the phase has a product: a better chop detector, to be wired and gated the way R5 was |
| Nothing clears but **`COMPRESSION` does** | the feature set is exhausted and the incumbent is as good as it gets from these inputs. The answer is a better **estimator** over the same features — R8's HMM/Kalman — not a better feature |
| **Nothing clears, `COMPRESSION` included** | chop is **not detectable from this feature set at 4h**. `COMPRESSION` is then doing something other than what it is named, and the thesis needs a different frequency, a different data source (OI, book, cross-sectional), or both. R8 as written should not be built on these features |
| Everything clears comfortably | **suspect the label before celebrating.** Audit the `t+1` anchoring and the tercile boundaries for leakage first; an easy win on a question four phases could not answer is evidence of a bug |

The third row is the one that matters. It is declared here so that reporting it
is the protocol working rather than a disappointment to be softened — and it is
the row that would send this program back to its own §2.1 state vector.

The fourth row is new, and it exists because R9's `direction`/`strength` dropout
result looked spectacular until it turned out to be a structural gate rather than
a statistical one.

---

## Method

`features/diagnostics.py` already computes IC against a forward return, in halves,
with rolling-only percentiles and the `t+1` anchoring. **R7 adds a target, not a
harness**: forward ER beside forward return, through the same code path, so a
chop IC and a direction IC are the same statistic against different labels and
can be read side by side.

Nothing here writes a strategy, a registry entry or a report of record.
`market_candles`, `funding_rates` and `signals` are read-only, as in R9.

---

## Deliverable

A charter §9.6 section with the tables, a §9 progress entry, a §10 decision row
for whichever outcome lands, and the R7/R8 roadmap rows updated. If the third row
of the outcome table is what happens, that roadmap update is the substantive one
and R8 does not start.
