# R10e — the event path runs every strategy and every contract

**A design plan, not a pre-registration.** Like R10a, R10b and R10c it ships an
interface and a set of binding acceptance checks rather than a figure. Unlike
them, its checks are **numbers R10d already measured**: this phase is judged by
driving three specific counts to zero.

**Charter:** [docs/research/2026-08-03-market-dynamics-engine.md](../research/2026-08-03-market-dynamics-engine.md),
phase R10. Follows R10d ([§9.13](../research/2026-08-03-market-dynamics-engine.md#913-r10d-the-event-paths-remaining-census-items--b-c-and-d-measured)),
which turned the census into six numbers and produced the ordering this plan takes.

---

## Why this before census item (a)

(a) is the only *total* gap and the only one that moves a published number, so
the obvious order is to close it first. R10d says otherwise, for three reasons.

1. **It touches each file once.** (a) needs `Bar`, `BarBuffer`, the feed protocol
   and the storage schema. (d)'s runner needs `Signal` and `engine/runner.py`.
   And `state_machine_v2` **reads crowding** — so closing (a) first means
   re-threading the funding plumbing through the exposure path the moment it
   arrives. Structurally complete first, then correct once.
2. **Nothing here can move a published number, and (a) can.** R10d proved every
   item in this phase is additive: pass-through modes differ on 0 bars, the
   streamed exposure target is exact at 0 of 400, and size is recomputable at 0
   differing. (a) is the one that turns +16.44% into +15.45%, and it deserves its
   own phase with its own bit-identical controls.
3. **(e) cannot verify (a) until (a) lands.** The determinism suite compares
   crowding-neutral against crowding-neutral on frames carrying no funding, so it
   is structurally incapable of checking the thing (a) fixes. That check has to be
   built in the same phase as the fix, not this one.

---

## What reconnaissance changed about the scope

Two items are smaller than the census implies and one has already been done.

- **(d)'s storage half already exists.** `ExposureSignal` is in
  `storage/signals.py`, `write_signals` accepts `Signal | ExposureSignal`,
  `_to_row` handles it, and `signals.target_exposure` has been a nullable
  `NUMERIC(10,6)` since R6. What is missing is **only the runner**.
- **(c) needs no work at all, and this plan does not do it.** See below.
- **`_extract`'s own docstring names this phase**: *"The runner gains an
  `ExitMode` in Phase 1b, when signals start driving positions rather than being
  recorded."* That is item (b), and it is the largest piece here.

---

## What gets built

### 1. A contract check at construction (M40)

`StrategyRunner` accepts `state_machine_v2` and survives **2,192 bars — 365.3
days at 4h** before raising, because `on_bar` returns before touching the
strategy while the buffer is inside warmup. One `hasattr` beside
`require_warmup_bars`, which is already in `__init__` for exactly this reason.

### 2. An `ExitMode` on the runner — item (b)

The runner takes an `ExitMode` and emits the exits that mode implies, **by
calling the engine's own `_exit_signals` over the buffer** rather than
reimplementing it. That is M36's rule reaching a third path: a cheaper route here
would be a fourth answer free to drift from the backtest, the browser and the
board.

Two things a reader has to carry. `trend_structure` **raises** if
`short_entries.any()` — on a growing buffer that is a claim that can become true
mid-run, where on a whole frame it is decided once. And `setup_invalidation_stop`
is **not** a pass-through: `_stop_kwargs` hands `from_signals` an `sl_stop` that
no signal stream carries, so a runner cannot reproduce that configuration at all
and must say so rather than approximate it.

### 3. An exposure runner — item (d)

A **separate** `ExposureRunner`, not a widened `StrategyRunner`. Same argument as
the third registry: a shared class dispatching on contract is the failure the
registry split exists to prevent, and measured there — an empty
`exposure_registry` silently skipped 4 parametrized tests and exited 0. It emits
`ExposureSignal`, and it mirrors `exposure_engine`'s `_banded`, because a target
reaches the book only once it has moved `rebalance_threshold` from the last one
*submitted*, and a runner that emitted on every bar would describe a different
book from the backtest.

`tests/test_exposure_determinism.py`'s local driver is then **pointed at it**, as
that file's own header asks. That is what makes the new runner proven rather than
merely written — and it is check 2.

---

## Acceptance checks

R10d's measurements are the targets, on R10d's frames (BTC/USDT and ETH/USDT perp
4h, each bounded by its own `funding_span`).

1. **Under each strategy's canonical mode, the runner's exits equal the engine's
   on every bar.** R10d measured the gaps as `trend_following_deepseek_v4`
   **7,331 / 7,012**, `turnaround_v1` and `v2` **984 / 1,035**, and 0 for the
   other six. All of them go to **0**. The six already at 0 are the non-vacuity
   guard: a change that broke them would be caught by the same check.
2. **`state_machine_v2` runs on the event path**, and
   `tests/test_exposure_determinism.py` drives the **real runner** rather than its
   local copy, with all three of its comparisons still passing — whole-history vs
   streaming, a runner primed from mid-history, and target-level equality.
3. **`StrategyRunner` refuses at construction** when handed a contract it cannot
   run. The 2,192-bar deferred crash becomes a refusal at bar zero, asserted by
   the exception rather than by a comment.
4. **Every published figure is bit-identical** against a clean `main` worktree —
   the control R10a and R10c both ran.
5. **The lookahead and determinism suites pass for every registered strategy on
   both contracts**, which is what the three registries exist to enforce.
6. **No new persisted field for size**, and the reason recorded rather than
   assumed.

---

## What this deliberately does not do

- **Not census item (a).** Everything here runs crowding-neutral, exactly as it
  does today, and no published figure moves. (a) is R10f.
- **No size field on `Signal`, and (c) closes as "no work needed".** R10d measured
  it recomputable from a cold buffer with **0 differing over 600 bars**, and the
  research browser already renders it — `api/analysis.py` returns `position_size`
  and `browser/page.py` reads it. M35 says do not persist what can be recomputed.
  Adding it would also give **7 of 9** strategies a field they can only ever leave
  `None`, which is the precise argument `ExposureSignal` exists to make about
  `target_exposure`. The census's "a live chart cannot show how big" was true of
  the *signal stream* and false of the product.
- **No live feed.** `MarketDataFeed` still has one implementation.
- **No new research claim.** No strategy, policy, feature or engine change.

---

## Then

R10f carries funding through the completed path, closing (a) and (e) together —
and the determinism suite becomes able to see the gap it is currently blind to.
After that R10's gate is checkable rather than aspirational.
