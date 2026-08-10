# R11 — a direction feature built for a short horizon

**Status:** run, **failed its own gate on 2026-08-10. `thrust` is not registered and its module is deleted.** The numbers that killed it are at the bottom; everything above the results section is as it was committed, before any of them existed.

## Why

R10i (the span sweep, in the charter) measured what the existing `direction`
feature is for, and the answer was narrower than anyone had written down:

| spans | lookback | BTC IC@90 | ETH IC@90 | SOL IC@90 |
|---|---|---|---|---|
| 6/24 | 1d / 4d | +0.0140 * | +0.0067 * | +0.0014 * |
| 12/48 | 2d / 8d | +0.0223 * | +0.0156 * | +0.0202 |
| **24/96** | **4d / 16d** | **+0.0466** | **+0.0494** * | **+0.0335** |
| 48/192 | 8d / 32d | +0.0431 | +0.0394 * | −0.0101 |

`*` = the two half-samples disagree in sign. Perp 4h, `[t+1, t+1+h]` anchoring,
every cell scored from the deepest cell's warmup.

Two readings follow, and the second is what this phase is about.

1. **24/96 is well chosen for what it does.** It is the largest cell on all
   three instruments at h=90. Shortening its spans degrades it monotonically.
2. **It is a ~2-week feature and only a ~2-week feature.** Measured on BTC/USDT
   spot with the shipped 24/96 across timeframes — 15m −0.0207, 1h −0.0196,
   **4h +0.0464**, 1d −0.0794 at h=30 — only 4h is positive at both horizons
   with both halves agreeing. Rescaling the spans to hold the wall clock
   recovers most of the sign (1h at 96/384: +0.0137 against −0.0379) but not the
   half-sample agreement.

So the machine currently has **no directional input that carries information at
a short horizon**, and the state view offers 15m and 1h rungs that draw a sign
from a feature whose IC is negative there. This phase asks whether one can
exist, and refuses to ship it if the answer is no.

## The hypothesis

An EMA spread is a *smoother*. At short horizons it is dominated by
microstructure noise, and averaging more of it is the only lever it has. R7
already measured the one thing that carries chop information here — `energy`,
IC vs forward efficiency −0.0906 on BTC and −0.1521 on ETH, same sign in both
halves of both — and it is a *dispersion* statistic, not a smoother.

**H1.** A short-horizon directional read should measure *where price sits in its
own recent range* rather than the gap between two averages of it. Range position
is bounded by construction, needs no seed, and its warmup is its window.

The candidate, `thrust`:

```
thrust[t] = 2 * rolling_percentile_of(close[t], window=w) - 1        # -1..1
```

...against a signed confirmation that the move is being paid for, not drifted
into. Exact form is fixed in "What is computed" below and does not move after
this document is committed.

## What is computed

A new `StateFeature` named `thrust`, registered in both places in
`features/registry.py`, so the poison probe and the measurability check enrol it
automatically.

```
window w                    = 24 bars
raw[t]                      = 2 * rolling_percentile(close, w)[t] - 1
paid[t]                     = rolling_percentile(|close[t] - close[t-w]| /
                                                 sum(true_range, w), w)[t]
thrust[t]                   = raw[t] * paid[t]
warmup_bars                 = 2w - 1
```

`rolling_percentile` is `features/base.py`'s, which is the only implementation
in this repo and the reason none of this can be full-sample. `paid` is the same
displacement-over-distance idea `strength` uses, ranked so it is comparable
across eras; multiplying rather than gating keeps the feature continuous, which
is what `state/policy.py` reads.

Warmup is `2w - 1` because a rank over `w` needs `w` values of a statistic that
itself needs `w` bars. **No EWM anywhere**, which is the entire point: at
`w = 24` this is a 24-bar feature with a 47-bar warmup, against `direction`'s
96-bar feature with a 1,920-bar warmup.

## Gate — decided now, in advance

`thrust` is **kept** only if all three hold on **BTC and ETH perp 4h**, scored
from the deepest warmup among everything compared, with both half-sample ICs
reported beside every full-sample one:

- **G1.** `|IC|` against the `[t+1, t+1+6]` forward return is **≥ 0.03** on both
  instruments — h=6 being one day at 4h, the horizon `direction` does not serve
  (+0.0173 BTC spot, +0.0092 BTC perp).
- **G2.** Both half-samples **agree in sign** on both instruments. This is the
  gate `direction` fails at every short setting, and it is not negotiable: a
  feature that works in one half is a regime.
- **G3.** `|corr|` with `direction` over the common window is **< 0.7**. Above
  that it is a cheaper spelling of a feature already registered, and the honest
  outcome is to say so rather than to add a ninth row to the diagnostics table.

**If any of the three fails, `thrust` is not registered** and this document is
updated with the numbers that killed it. A feature that fails its own gate and
ships anyway is how a diagnostics page becomes decoration.

Held out: **SOL perp 4h** is not looked at until G1–G3 have been decided on BTC
and ETH. It is the replication, not part of the fit.

## What is explicitly not in scope

- **No change to `direction`, `state_machine_v1` or `state_machine_v2`.** R5's
  and R6's published figures do not move in this phase. If `thrust` passes, what
  consumes it is a separate, separately pre-registered decision.
- **No change to `_EWM_WARMUP_MULTIPLE`.** The 20× buys provable cold-start
  equality between the vectorized and event paths; the measurement that it could
  be 5× is recorded in the charter and is its own phase.
- **No tuning of `w`.** 24 is fixed here, before any number, because a window
  chosen after seeing the ICs is a window fitted to them. If 24 fails the gate,
  that is the result.

## Oracle

The existing machinery, not a new script:

- `tests/test_feature_lookahead.py` enrols `thrust` on registration and poisons
  every bar after *t* — the direct causality proof.
- The same file checks the feature is measurable at its own declared
  `warmup_bars`, which is what stops the probe comparing NaN to NaN.
- `diagnose_features` produces the ICs, both half-samples and the pairwise
  correlations that G1–G3 are read off, through the `features` CLI command that
  writes a timestamped `reports/` directory.

Nothing here needs a bespoke measurement harness, and that is deliberate: R10h
shipped one with two correctness bugs, and the lesson recorded then was that
measurement code is the thing least likely to be checked by what comes after it.


---

# Result — 2026-08-10

BTC and ETH perp 4h, scored from row 1,920 (the deepest warmup compared, which is
`direction`'s — `thrust`'s own is 47). SOL was never looked at: the gate was
decided before the holdout was due, and it did not get that far.

| | h=6 | h=30 | h=90 | corr vs `direction` |
|---|---|---|---|---|
| **BTC** `thrust` | −0.0117 (−.0283/**+**.0090) | −0.0110 (−.0087/−.0118) | +0.0249 (+.0293/+.0184) | +0.3206 |
| BTC `direction` | +0.0157 (+.0211/+.0091) | +0.0396 (+.0422/+.0344) | +0.0877 (+.0561/+.1193) | — |
| **ETH** `thrust` | −0.0192 (−.0275/−.0093) | −0.0050 (−.0043/−.0055) | +0.0130 (+.0074/+.0187) | +0.3102 |
| ETH `direction` | +0.0017 (+.0114/**−**.0128) | +0.0235 (+.0354/+.0034) | +0.0846 (+.0627/+.0945) | — |

Against the gate committed above:

- **G1 — `|IC@6| >= 0.03` on both. FAILED.** −0.0117 on BTC and −0.0192 on ETH,
  both well under the threshold. `thrust` is not a short-horizon signal; on BTC
  it is *weaker* at h=6 than the feature it was built to complement.
- **G2 — both halves agree in sign on both. FAILED.** BTC h=6 splits
  −0.0283 / +0.0090. By this repo's own rule that is a regime, not a signal.
- **G3 — `|corr|` with `direction` under 0.7. Passed**, +0.32 on both. It is
  genuinely a different statistic. It is just not an informative one.

Two of three fail, so `thrust` is removed. The pre-registration said what
happens in this case and this is it: an unregistered feature is an untested one
here, since the registry is what enrols the poison probe, so the module is
deleted rather than left in the tree.

## What it did show, which is not the same as passing

At **h=30 both halves agree in sign on both instruments** — BTC −0.0110
(−.0087/−.0118), ETH −0.0050 (−.0043/−.0055) — and the sign is **negative**. Sitting
high in your own recent range predicts a slightly *lower* forward return over
the next five days. That is consistent across four instrument-halves, which is
more than several registered features manage, and it is mean reversion rather
than trend.

It is far too small to trade on its own (|IC| ~0.01 against `direction`'s 0.04)
and it is emphatically **not** what this phase set out to find. Recorded as a
lead, not a result: the useful precedent is R4, where `direction`'s unconditional
+0.0385 hid a −0.113 inside one `strength` tercile. Whether this does the same is
a different question, with a different pre-registration, on a phase nobody has
written.

## What this phase actually settles

**The registry still has no directional read that works at a short horizon, and
one obvious candidate has now been measured and rejected rather than assumed.**
The state view's 15m and 1h rungs remain exploratory for the reason documented in
the README, and nothing about `direction`, either state machine, or the EWM
multiple moved.
