# R12 — the EWM warmup multiple, from 20× to 5×

**Status:** pre-registered. Written before any number below the line exists.

## Why

`features/trend.py` carries `_EWM_WARMUP_MULTIPLE = 20`, so `Direction`'s warmup
is `20 × 96 = 1,920` bars and `state_machine_v1`/`v2` declare **2,192**. The
multiple was chosen from one measurement — a cold start disagrees with the
whole-history value on 299/300 probed bars at 10× and on 0/300 at 20× — and has
been carried as a design constant since.

That measurement is about **bit-exactness**, not about information, and the two
are very far apart here. A span-96 EMA under `ewm(adjust=False)` has
`alpha = 2/97`, so its weight decays geometrically:

| | bars | at 4h |
|---|---|---|
| half the weight | 34 | 5.7 days |
| 90% | 111 | 18.5 days |
| 99% | 222 | 37 days |
| 99.9% | 332 | 55 days |

The bar 1,920 back carries weight **8.9e-20**, four orders of magnitude below
float64 epsilon (2.2e-16). Everything older than 288 bars is 0.25% of the value.
The state does not depend on 1,920 bars; it depends on roughly 200–300.

**What the 20× costs is chart.** Warmup is counted in bars, so 2,192 is 365 days
at 4h and 42 years at 1w, and a dataset shorter than it produces no state at all.

## Why 5× and not lower

`derive_warmup_bars` takes the deepest feature plus the machine's convergence.
`Direction` is the deepest only while `20 × 96` exceeds the ranked features'
`96 + (480 - 1) = 575`:

| multiple | `direction` | deepest | total |
|---|---|---|---|
| 20 | 1,920 | 1,920 | **2,192** |
| 10 | 960 | 960 | 1,232 |
| 6 | 576 | 576 | 848 |
| **5** | **480** | **575** | **847** |
| 4 | 384 | 575 | 847 |
| 3 | 288 | 575 | 847 |

So **5× reaches the floor and 4× / 3× buy nothing** — below 5 the binding
constraint is `strength`/`stability`'s trailing rank, not the EMA. Going lower
only makes `Direction` less converged inside a region the strategy masks anyway
(cold-start error at 480 bars is 5.1e-5; at 288 it is 3.3e-3). 5× is therefore
the most converged setting that reaches the minimum achievable warmup, and that
is why it is the number rather than the smallest one that fits.

## What changes

One constant: `features/trend.py::_EWM_WARMUP_MULTIPLE`, 20 → 5. Everything
downstream is derived — `Direction.warmup_bars`, `state_machine_v1.warmup_bars`,
`state_machine_v2.warmup_bars`, the browser's ladder and ribbon, the board's
refusals.

**`strategies/ema_cross.py` keeps its own `_EWM_WARMUP_MULTIPLE = 20` and is not
touched.** It is a separate module's constant on an R0 baseline whose figures
are a comparison point for everything after it; moving it would restate a
published baseline for no benefit this phase claims.

## Gate — decided now, in advance

All four must hold, or the constant goes back to 20:

- **G1.** Over the bars *both* settings can compute — bar 2,192 onward — the
  state series are **identical** on BTC, ETH and SOL perp 4h. Not "close":
  identical. This is what makes it a warmup change rather than a model change.
- **G2.** The whole safety suite stays green, in particular
  `tests/test_strategy_metadata.py` (which replays the cold start against the
  declared warmup), `tests/test_lookahead.py`, `tests/test_feature_lookahead.py`,
  `tests/test_replay_determinism.py` and `tests/test_exposure_determinism.py`.
- **G3.** The full test suite stays green.
- **G4.** R5's published figures are **re-measured and republished**, not
  assumed. The masked region moves, so which bars trade moves; any figure in
  `STRATEGIES.md` or the charter that changes is updated in the same commit,
  with both the old and the new number visible. A figure that does *not* move is
  stated as unmoved rather than left unchecked.

G4 is the point of the phase, not paperwork. The reason this change has not been
made before is precisely that it restates published results, and restating them
silently would be worse than not making it.

## What is explicitly not in scope

- **No change to any span, window, threshold or dwell.** `fast_span=24`,
  `slow_span=96`, `rank_window=480`, `min_dwell=4` and the rest are untouched.
  R10i measured 24/96 as the best cell on all three instruments; this phase is
  about warmup only, and mixing the two would make G1 impossible to read.
- **No change to `ema_cross`.**
- **No re-tuning of anything to the new figures.** If R5 gets worse, it gets
  worse and is published worse.

## Oracle

G1 is the oracle and it is exact: two configurations, one frame, identical
states over the common range. It cannot be satisfied by a change that alters the
model, which is what separates this from a re-parameterisation.
