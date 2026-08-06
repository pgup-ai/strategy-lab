# R7c — the energy-first lifecycle: pre-registration

**Committed before any R7c figure is computed.** Same rule as R7, R7b, R9 and the
ETH replication: the commit adding this file precedes the commit adding the
numbers, and the two are separate.

**Charter:** [docs/research/2026-08-03-market-dynamics-engine.md](../research/2026-08-03-market-dynamics-engine.md).
Acts on **M27**, and this phase **spends the SOL holdout** that R7b's kill switch
saved.

---

## The question, and what is already answered

R7b measured the entry-gate question and it is **settled, not re-opened here**.
At H=30, trend-rate lift in pp:

| gate | BTC train / test | ETH train / test |
|---|---|---|
| `strength ≥ 0.80` — what the lifecycle gates on | −7.63 / +3.32 | −1.26 / −2.52 |
| `energy ≤ 0.50` — what it ignores | +3.51 / +5.10 | +4.49 / +6.61 |

`energy` is positive in all four instrument-halves; `strength` is negative in
three, including the half R5 selected it on. **Re-running that as if it were an
open diagnostic would be declaring a threshold whose answer is already known**,
which is M23's failure wearing a new hat. So R7c does not re-ask it.

What is **not** answered is whether a *machine* built the other way round trades:
`strength` drives both `advancing` and `failing`, so the hysteresis, the dwell and
the whole six-state lifecycle sit on the axis that measures negative. R7c inverts
that and asks the only questions R7b left open — **does it produce a book, and
does the book earn out of sample.**

**The direction is counterintuitive and is not a sign error.** The machine will
advance when energy is **low** — quiet markets, not violent ones. That is exactly
what [§2.1](../research/2026-08-03-market-dynamics-engine.md#21-the-state-vector)
predicted (`Strength = 20, Energy = 95 → violent two-way chop`) and what R7b
measured twice. A reader who trips on "breakout on low energy" should read it as
*the trend that is worth riding is the orderly one*, which is the thesis this
program started from.

---

## What this phase costs, stated before it runs

**BTC and ETH are burned for this hypothesis.** R7b read `energy`'s lift on both
instruments' **test** halves, so any threshold near 0.50 is informed by data on
both. Their R7c numbers are **reported and are not evidence**.

**SOL is the only test that counts, and R7c spends it.** One run, whole frame, no
split, zero parameters derived from it. After this there is no clean instrument
left in the store — a further attempt on this hypothesis needs a newly fetched
one, and fetching it *after* seeing R7c's result would not make it clean for a
question R7c already asked. That asymmetry is the reason the kill switch below
sits before the SOL run and not after it.

---

## Fixed here, before anything runs

### The change

`StateMachine` gains an **energy-first mode**. Both sides move together, because
splitting them would put the hysteresis across two features:

```
advancing = (energy <= enter_energy) & (|direction| >= direction_floor)
failing   = ~measurable | (energy > exit_energy)
```

with `exit_energy > enter_energy` — the inequality direction is the mirror of
`enter_strength > exit_strength`, and the constructor must enforce it the same
way. `direction` stays: it decides **side**, which energy cannot. `stability` and
`crowding` keep their present roles. `strength` leaves the **lifecycle** and stays
available to the policy.

The mode is off by default, so `state_machine_v1`/`v2` keep their published
behaviour by construction — the same discipline that made R7b's control 2 pass,
and it is a declared control again below.

### The grid

`enter_energy ∈ {0.35, 0.50, 0.65}` × `exit_energy ∈ {0.65, 0.80}` = **6 cells**.
Everything else is held at R5's trained cell (`min_dwell=4`, `cooldown=4`,
`direction_floor=0.10`, `stability_floor`, `crowding_extreme`). **M22 is in
force**: nothing else is re-derived, on any frame.

### The frames

BTC/USDT perp 4h on R5's frame and split; ETH/USDT perp 4h on §9.4's; SOL/USDT
perp 4h, 12,914 bars, whole frame, no split. Engine defaults throughout: 10bp/side,
`opposite_signal_only`, fixed sizing, `position_pct=0.95`, 10,000 initial,
**net of funding**, funding column attached (M20). Every statistic at **4h bar
frequency** — declared, because R9's was not (M23).

---

## What R7c measures, in order

### 0. Controls, before any R7c number is read

1. The energy-first mode **off** reproduces R5's published row — +15.45% /
   +0.896 / 4.67% / 73.
2. **Byte-identity against `main`** for all four published v1/v2 rows, compared as
   floats against a baseline regenerated from a clean `main` worktree, as R7b did.
3. Both safety suites pass for the modified machine, plus the exposure suites.

### 1. The kill switch — tradeability, before any return is read

R7b's kill switch was a rate diagnostic because the open question was about an
*estimator*. Here the estimator question is settled and the open one is about a
**book**, so the switch is placed where a book can be degenerate:

On BTC's **training half**, the selected cell must trade between **26 and 231**
round trips — R5's 77, within a factor of three either way. Below that the P&L is
a handful of draws; above it the run is being scored on costs rather than on
state. **If the selected cell falls outside that band, the SOL holdout is not
spent and the phase stops.**

### 2. Selection, on BTC's training half only

Highest Sharpe over **tradeable bars** (M21) across the 6 cells. One selection,
reported with its full surface.

### 3. The overlay control — the alternative explanation, run before SOL

"Advance only when energy is low" may be nothing more than **trade in low
volatility**, which is a known effect and not a state machine. So R5's *unmodified*
trained machine is run with entries suppressed on bars where `energy > enter_energy`
at the selected value — a pure overlay, no lifecycle change.

**If the overlay captures the energy-first machine's out-of-sample Sharpe to within
0.05, the lifecycle redesign has added nothing** and R7c's finding is a
volatility filter that any strategy here could wear. That is reported as the
result, and the SOL run is still made — it is the cheaper explanation, not a
failure.

### 4. Evaluation

| frame | how it is read |
|---|---|
| BTC test half | **burned** — informed by R7b. Reported, not evidence. |
| ETH full frame | **burned**, same reason. |
| **SOL full frame** | **the holdout, spent here.** One run, zero parameters from it. |

### 5. Deflated Sharpe

Six trials, at 4h bar frequency. `E[max SR]` and the DSR beside the observed.

### 6. Purging and embargo — stated, not performed

Unchanged reasoning, re-checked against the code: the machine fits nothing and has
no labels, and two parameters chosen from a 6-cell grid is not a fitted model with
a forward horizon. They become mandatory at R8.

---

## Declared thresholds

| Claim | Clears if |
|---|---|
| Controls (§0) | R5's row reproduces to the published digits, and all four v1/v2 rows are bit-identical to `main` |
| **Kill switch (§1)** | the selected cell trades **26–231** round trips on BTC's training half |
| **The holdout (§4)** | on SOL's full frame the energy-first machine beats **both** (a) R5's trained cell run unmodified and (b) `donchian` 40/10, on Sharpe net of funding — **and** survives 3× costs with a positive net return |
| The redesign earns its complexity (§3) | the energy-first machine beats the overlay by ≥ **0.05** of Sharpe on SOL |
| The selection is not luck (§5) | observed Sharpe > `E[max SR]`, **and** DSR ≥ **0.95** |

---

## Declared before looking

| Outcome | Reading |
|---|---|
| Kill switch fails | the inversion does not produce a book. **SOL is not spent**, and M27 stands as a statement about rates that no lifecycle has been able to trade. |
| Holdout clears **and** beats the overlay | the lifecycle was built on the wrong axis and R7c fixes it. R8 finally has inputs and a shape worth modelling. |
| Holdout clears, overlay matches it | **the finding is a volatility filter, not a state machine.** Cheaper, portable to every strategy here, and it makes the six-state lifecycle decoration. Report it as the simpler thing it is. |
| Holdout fails | `energy` discriminates chop on the rate metric and **still does not earn**. Three phases will then have separated *measuring* state from *trading* it, and the program's problem is execution, not estimation — which is a different research programme from the one §2 describes. |

The last row is the expensive one: it spends the holdout and returns nothing
tradeable. It is declared here so that reporting it is the protocol working, and
so the decision to spend SOL is made now — in advance, with the cost visible —
rather than after a number makes it tempting.

**No R7c result changes a published R5, R6, R7, R7b or R9 figure.**

---

## Method

`scripts/r7c/`, reusing the gate's `evaluate` and R7b's control harness. The
machine change is in `state/machine.py` and `strategies/state_machine_core.py`.
`market_candles`, `funding_rates` and `signals` are read-only throughout.

---

## Deliverable

A charter §9.8 section, a §9 progress entry, a §10 decision row, and the R7c/R8
roadmap rows. If the kill switch fires, that is the whole deliverable and the
holdout survives to the next phase.
