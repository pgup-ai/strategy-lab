# R11 — a direction feature built for a short horizon

**Status:** pre-registered. Written before any number below the line exists.

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
