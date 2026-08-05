# ETH/USDT replication of the R5/R6 protocol — pre-registration

**This document is committed before any ETH figure is computed.** That is the
entire point of it: a replication whose protocol is written after the result is
not a replication. The commit that adds this file must precede the commit that
adds the numbers, and the two must be separate.

## Why this before R7

R6 left the charter's R7 (meta-model for sizing) aimed at the half that measured
nothing: the earning difference was the *ramp*, not the taper, and **no
per-state effect on the test half is distinguishable from zero** (no |t| above
1.77). Every published MDE figure rests on **one asset and one 60/40 split**.
A second asset adds evidence; a model on the first asset re-examines what is
already there. ETH/USDT perp 4h is already stored — 14,650 bars anchored at its
first funding settlement, 7,325 settlements — so this costs no new code and no
new data.

## Data and split — fixed here

- **ETH/USDT perp 4h**, anchored at the first stored funding settlement
  (2019-11-27 08:00), 14,650 bars to 2026-08-03 20:00.
- **Split 2023-10-31 00:00**, the same calendar date as R5.
  - This yields **6,048 test bars, identical to BTC's**, so the two assets'
    test halves cover the same era with the same sample size and a difference
    between them is the asset rather than the regime.
  - ETH's own 60/40 boundary is 2023-12-01, **31 days later**. Both are declared
    so it is visible that the choice is not doing work; if the result differs
    between them that is itself reportable, but the pre-registered split is
    2023-10-31.
- Each run's frame starts `split − warmup_bars`, so the engine's own mask lands
  on the boundary — the rule R5 used, and the reason the two halves are scored
  on the bars each owns.
- Engine defaults: 10bp/side (`CostModel()` = 5bp fee + 5bp slippage),
  `--exit-mode opposite_signal_only`, fixed sizing, `position_pct=0.95`,
  `cash=10,000`, **everything net of funding**.

## The two questions, which are not the same

**Q1 — transfer.** Does BTC's *trained cell* work on ETH, unchanged?
`StateMachine(enter_strength=0.80, exit_strength=1/3, min_dwell=4, cooldown=4)`,
carried across with **zero** parameters re-derived. This is the harsher test and
the more valuable one: it has no free parameters at all, so it cannot be
flattered.

**Q2 — protocol replication.** Does the *method* replicate? Re-run R5's own
54-cell search on **ETH's training half** — `enter_strength` ∈ {0.55, ⅔, 0.80} ×
`exit_strength` ∈ {0.20, ⅓} × `min_dwell` ∈ {2, 4, 8} × `cooldown` ∈ {4, 8, 16} —
selected by the same declared scalar, highest net-of-funding Sharpe, then
evaluated once on the test half.

Q1 failing while Q2 passes would say the machine works but its parameters are
per-asset. Both failing would say R5 was BTC-specific. Both passing is the only
outcome that strengthens the program, and it is not the expected one.

## The baseline gets the same treatment

`donchian` 40/10 on ETH's own bars, plus its **16-cell surface** on both halves
as the fairness check R5 ran — the baseline family may survive out of sample
where the cell training selects does not, and a reader is owed both.

## Also run, because they are free

- The **untuned R4-default machine**, which carries no selection discount.
- **Cost stress 1×/2×/3×** on every declared run.
- **`state_machine_v2`** (continuous) beside v1, since R6's comparison exists
  only on BTC. Report round trips and fills as separate columns — v2's fills
  include resizes and are not a trade count.
- **Buy-and-hold** for both halves, as the scale everything else is read against.

## Declared test-half looks

Three runs (donchian 40/10, machine default, machine trained), the 16-cell
donchian surface, the 54-cell selection's winner, `state_machine_v2` at both
configs, and the stress rows. **Nothing is adjusted after seeing any of it.**
The surface and the stress rows can only make the machine's claim harder.

## What counts as what — declared before looking

| Outcome | Reading |
|---|---|
| Machine beats donchian 40/10 on OOS Sharpe, at either config | **replicates** |
| Machine positive OOS Sharpe but below the baseline | **partial** — the state machine works, the edge over a channel break does not transfer |
| Machine negative OOS Sharpe | **fails** — R5 was BTC-specific |

R5's BTC numbers, for reference rather than as a target: trained **+0.896 / +15.45% /
4.67% / 73 trades**, default **+0.746 / +15.52% / 7.11% / 153**, donchian 40/10
**+0.072 / −6.64% / 43.86% / 114**.

**No parameter is tuned on ETH's test half, and no ETH result changes anything
about the BTC figures already published.** If ETH disagrees, both stand as
measured and the disagreement is the finding.

## Method

One statistics function over the **net-of-funding equity curve** for every run,
both contracts, as R6 established — and validated the same way, by reproducing
`donchian` 40/10's ETH figures from the engine's own written
`equity_curve.csv` before any comparison is read. A harness that cannot
reproduce a run it just wrote is not measuring the thing it claims to.

## Deliverable

A charter §9 progress entry and a §9.4 section carrying the tables, plus a
roadmap update for R7 reflecting whatever this says. `market_candles` is
read-only throughout.
