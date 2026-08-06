# R9 — walk-forward and robustness: pre-registration

**Committed before any R9 figure is computed.** The ETH replication established
the pattern and the reason: a robustness check whose protocol is written after
its result is not a check. The commit adding this file must precede the commit
adding the numbers, and the two must be separate.

**Charter:** [docs/research/2026-08-03-market-dynamics-engine.md](../research/2026-08-03-market-dynamics-engine.md),
phase R9. Gate as written: *purging, embargo, deflated Sharpe, parameter
perturbation, feature dropout*.

---

## The question, sharper than when the gate was written

R5 selected one cell from **54** on the training half and published an
out-of-sample Sharpe of **+0.896**. No multiple-testing correction has ever been
applied to it.

R5's own defence was that the *untuned* R4-default machine also passed
(+0.746), so the verdict carried no selection discount. **The ETH replication
removed that defence**: on a second asset the untuned machine returns −19.58% at
Sharpe −0.563. So the verdict rests entirely on a search over each asset's own
training half, and the size of that search has never been priced in.

R9 asks whether the search finds signal or luck. It is run before R7 because a
sizing model fitted to numbers that have not survived their own selection is a
model of noise — and ETH already measured no per-state effect distinguishable
from zero (max |t| = 1.77).

---

## Fixed here, before anything runs

- **Frame:** BTC/USDT perp 4h, anchored at the first stored funding settlement
  (2019-09-10 08:00), 15,118 bars. Engine defaults throughout: 10bp/side,
  `opposite_signal_only`, fixed sizing, `position_pct=0.95`, 10,000 initial,
  **everything net of funding**.
- **Grid:** `DECLARED_GRID` in `tests/test_state_machine_gate.py`, unchanged —
  3 × 2 × 3 × 3 = 54 cells. Cited, not restated.
- **Selection scalar:** highest Sharpe, as R5 used, **scored over tradeable bars
  only**. This is M21 taking effect for the first time. R5's published selection
  is **not** rescored: rescoring a selection after its test half has been seen is
  what the protocol exists to prevent. If the two rules disagree about any
  fold's winner, that disagreement is reported rather than resolved.
- **The control runs first**, as R6 and the ETH replication established:
  reproduce R5's published row (+15.45% / +0.896 / 4.67% / 73 trades) before any
  R9 number is read. A harness that cannot reproduce the figure it is auditing is
  not auditing it.

---

## What R9 measures

### 1. Deflated Sharpe on the training selection

Bailey and López de Prado. Under the null that all 54 trials have zero true
Sharpe, the *expected maximum* of 54 draws is not zero — it grows with the
spread of the trial Sharpes. Inputs, all from the training half and all already
published in aggregate: the 54 trial Sharpes, their standard deviation, the
winner's Sharpe, the sample length, and the skewness and kurtosis of the
winner's own return series.

Report `E[max SR]` under the null beside the observed +1.215, and the deflated
Sharpe itself.

**What it can and cannot say, stated now so the result is not over-read.** It
prices the *in-sample* selection: whether picking the best of 54 on the training
half is distinguishable from picking the luckiest. It does **not** deflate the
+0.896, which was a single evaluation of one pre-committed cell on data not used
to select it. A low DSR would not make +0.896 wrong; it would make the
*procedure that chose the cell* uninformative, which is a different and arguably
worse problem — it means the number cannot be reproduced by repeating the method.

### 2. Walk-forward

The structural answer to "one split, one look". Expanding training window,
declared here:

- Start after the deepest warmup in the grid (bar **2,352**).
- Minimum training length **3,000** bars before the first evaluation.
- Test blocks of **1,008** bars (≈ six months at 4h), non-overlapping,
  stepping forward to the end of the frame — roughly **nine** folds.
- On each fold: re-derive the winner from the 54 on that fold's training data by
  the declared scalar, then evaluate it **once** on that fold's test block.

Report every fold's selected cell and out-of-sample Sharpe and net return, and
the distribution across folds. **How often the same cell wins is itself a
result**: a method that selects a different corner of the grid every six months
is not a method, whatever its average.

### 3. Parameter perturbation

R5 flagged that the optimum sits on the **edge** of the swept range in
`enter_strength`, and ETH found the same in three of four axes. Perturb each
axis of the R5 winner one step in each available direction and report how the
training and test figures move. A plateau degrades slowly; a ridge falls off.

### 4. Feature dropout

Replace each of the four features (`direction`, `strength`, `stability`,
`crowding`) with its neutral value in turn, re-run the winner, and report what
each removal costs. R4 argued the strongest features are *conditioners* rather
than predictors; this measures whether the machine actually depends on them.

### 5. Purging and embargo — stated, not performed

The gate names them, and they are **near-inapplicable to this strategy**, which
is the honest finding rather than a reason to skip quietly. Purging and embargo
exist to stop *overlapping labels* leaking across a split. `state_machine_v1`
fits nothing and has no labels: it is a causal rule, proven so by
`tests/test_lookahead.py`. The one boundary effect it does have — the machine's
state carrying across the split — was already measured by R5's frame-start
invariance check: the whole-history and test-window runs agree on **73 of 73**
entry timestamps.

They become mandatory at **R7/R8**, where a fitted model does have labels and a
forward horizon. Recorded there rather than performed here as a ritual.

---

## Declared before looking

| Outcome | Reading |
|---|---|
| DSR clears the null's expected maximum **and** walk-forward is positive in most folds | R5 survives; R7/R8 are worth attempting |
| DSR low, walk-forward positive | the single-split selection was lucky, but the method works — re-aim rather than abandon |
| DSR clears, walk-forward inconsistent | the selection is sound and does not persist across regimes; the program is a regime problem, not a model problem |
| Both fail | **R5's result is a selection artifact.** The program's direction changes, and R7/R8 as written should not be built |

The last row is the point of running this. It is declared here so that reporting
it is the protocol working rather than a disappointment to be softened.

**No parameter is tuned on anything R9 measures, and no R9 result changes a
published R5 or R6 figure.** Where they disagree, both stand as measured and the
disagreement is the finding.

---

## Method

One statistics function over the **net-of-funding equity curve**, applied to
every run, as R6 and the ETH replication both used — and validated the same way,
by reproducing R5's published row before any comparison is read.

Walk-forward is 54 cells × ~9 folds plus the out-of-sample runs. The gate's
existing 54-cell surface takes about 33 s, so this is minutes, not hours;
`tests/test_state_machine_gate.py::evaluate` already returns `tradeable_bars`
and `first_tradeable`, which is what the M21 scalar needs.

`market_candles`, `funding_rates` and `signals` are read-only throughout.

---

## Deliverable

A charter §9 progress entry and a §9.5 section carrying the tables, a §10
decision entry for whatever R9 decides about R7, and a roadmap update. If the
result is the last row of the table above, that update is the substantive one.
