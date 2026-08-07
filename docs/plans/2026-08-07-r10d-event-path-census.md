# R10d — the event path's remaining census items: pre-registration

**Committed before any R10d figure is computed.** Same rule as R7, R7b, R7c,
R7d, R9 and the ETH replication: the commit adding this file precedes the commit
adding the numbers, and the two are separate.

**Charter:** [docs/research/2026-08-03-market-dynamics-engine.md](../research/2026-08-03-market-dynamics-engine.md),
phase R10. This is a **measurement** phase, not a build — unlike R10a, R10b and
R10c, which shipped interfaces and declared design plans.

---

## The question

R10's gate is *"research and live code paths produce identical output"*. The
2026-08-05 state-of-play entry enumerated six ways they differ. Two are now
numbers:

| | | status |
|---|---|---|
| (a) | funding/crowding absent from `Bar` | **measured** (R10a): `crowding` differs on 6,048/6,048 bars, and pinning it reproduces every stored state with **0 residual** — so (a) is the *entire* gap for `state_machine_v1` |
| (f) | equity bars are restated, not appended | **measured and closed** (R10c): 333/333 SPY weekly bars moved, `donchian` differed on 3 |

The other four are still prose, and **the gate cannot be evaluated against
prose**. R10d turns (b), (c) and (d) into numbers. (e) is not a claim about
behaviour — it is a statement that the determinism suite cannot see (a), already
recorded in CLAUDE.md — so it is out of scope here and closes when (a) does.

---

## Reconnaissance done before writing this, and the priors it creates

Reading code is not measuring, but it produces expectations, and a phase that
hides its priors can dress a confirmation up as a discovery. So, stated:

1. **`_exit_signals` returns `signals.long_exits, signals.short_exits`
   unchanged for `opposite_signal_only`** (`backtests/engine.py:616`). That is a
   pure pass-through, and it is the **canonical** mode for `trend_rider_v1`,
   `state_machine_v1`, `tsmom`, `donchian`, `ema_cross` and `multi_horizon`.
   **Prior: (b)'s claim that "the signal stream matches no single backtest
   configuration" is overstated** — it should hold for modes that *add* an
   engine-side ingredient and fail for the pass-through. If so the finding is
   that (b) is a per-*mode* gap, not a per-*strategy* one.
2. **`tests/test_exposure_determinism.py` already streams exposure strategies
   correctly**, through a test-local driver its own header calls a deliberate
   mirror of `StrategyRunner`, with *"there is no production exposure runner yet
   … when an exposure runner lands, point `streamed_targets` at it rather than
   keeping two."* **Prior: (d) is smaller than "cannot run at all" implies** —
   the streaming semantics are proven and `signals.target_exposure` already
   exists as a column, so what is absent is a runner and a `Signal` that carries
   a level, not an unsolved problem.
3. **`Signal` has no size field at all** (`core/types.py:158`), and
   `position_size` is withheld by `_extract` along with the trend-failure series.

A prior is not a result. Each is written down so the run can contradict it.

---

## The methodological point, fixed before anything runs

**(a) and (b) confound, and must be measured apart.** On a perp,
`state_machine_v1`'s replayed signals already differ from its backtested ones
*because of crowding* — that is (a), measured, total. Comparing a live replay
stream against a backtest on a perp would therefore attribute (a)'s difference
to (b). So:

- **(b) is measured against the whole-history `SignalSet`, not against a replay
  run.** `tests/test_replay_determinism.py` already proves the replay stream
  equals the last row of `generate_signals` per bar; `_extract` reads exactly
  the four side series. So the replay stream's exits **are**
  `signals.long_exits` / `signals.short_exits`, and the comparison is those
  against `_exit_signals(df, signals, mode, failure_bars)` — one frame, one
  strategy call, no funding difference on either side of the comparison.
- **That shortcut is validated, not assumed** (§8 rule 4). One real `replay`
  run, on one strategy and frame, reconstructed into per-bar boolean series and
  asserted equal to the whole-history series it stands in for. If it is not
  equal, the shortcut is withdrawn and (b) is re-measured the slow way.

**Frames.** BTC/USDT perp 4h over R5's frame and split, bounded by
`db.funding.funding_span` at both ends (a db-marked run on a real perp frame that
does not bound its right edge reddens on an unrelated refresh). ETH/USDT perp 4h
on §9.4's frame as the replication. **No holdout is spent** — R10d measures the
repo's own plumbing, not a market hypothesis, so there is no selection to
discount and SOL stays clean.

**Strategies.** All nine registered `SignalSet` strategies for (b) and (c); both
exposure strategies for (d). Every valid (strategy, mode) pair from
[STRATEGIES.md](../../STRATEGIES.md)'s matrix — a pair the matrix marks `✗
raises` is asserted to raise rather than skipped, because "it raises" is the
matrix's claim and an unrun cell is not evidence for it.

---

## What R10d measures, in order

### (b) Exit modes in replay — per (strategy, mode), how many bars differ

For each valid pair: `n_bars`, bars where the effective long exit differs, bars
where the short exit differs, and the same for **entries** as a control — the
engine passes entries through untouched, so a non-zero entry count means the
comparison itself is wrong and the run stops.

Then one level up, because `accumulate=False` makes signal counts and fill
counts different questions (M36's reason for markers being fills): for the
canonical mode of each strategy, **positions derived from the raw side series vs
the backtest's own `trades.csv`** — entry bars, exit bars, and bars on which the
implied position differs.

**Declared readings.**

1. **Pass-through modes differ on 0 bars.** Then (b) is a per-mode gap, the
   census's wording is corrected, and the runner needs an `ExitMode` only for
   the strategies whose canonical mode adds an ingredient — a much smaller
   Phase 1b than "signals cannot drive positions yet" implies.
2. **A pass-through mode differs on any bar.** Then something worse than an
   exit-mode gap exists between `_extract` and `_exit_signals`, and R10d stops
   and becomes a bug hunt. This is a **kill switch**, not a finding to write up.
3. **The additive modes differ, and by how much is the number that matters.**
   A mode differing on 3 bars of 6,048 and one differing on 2,000 argue for
   different Phase 1b scopes; the count is the deliverable, not the fact.

### (c) Size on the wire — is anything lost, or merely not sent?

For every strategy that emits `position_size`: the distribution over **entry**
bars (the only bars where `from_signals` consumes it — R6), how many entry bars
carry a scale ≠ 1.0, and the spread of those values.

Then the question that decides whether (c) is a gap at all: **is
`position_size` recomputable from stored candles?** It is returned by
`generate_signals(df)` over the buffer, so by construction it should be — and
under M35's rule, what is recomputable is not persisted. The check is whether
any strategy's size reads something the buffer does not carry.

**Declared readings.**

1. **Recomputable and non-trivial** (some entries ≠ 1.0). Then (c) is a
   *reporting* gap: the fix is to emit a size the consumer could have derived,
   which is a `Signal` field and not a schema question.
2. **Recomputable and vacuous** (every entry 1.0 on every strategy). Then (c) is
   not a gap today and becomes one when a sizing strategy ships. Recorded as
   such rather than fixed — building a wire field for a constant is the
   `vol-scaled-entry` mistake.
3. **Not recomputable for some strategy.** Then it is a real data-loss gap and
   ranks with (a).

### (d) `TargetExposure` on the event path — what breaks, and how far from working

The failure is recorded first, exactly: what `StrategyRunner` raises when handed
`state_machine_v2`, and where. Then the delta between the proven test-local
driver and a production runner is enumerated against the sockets that already
exist — `signals.target_exposure` (a nullable `NUMERIC(10,6)`, added in R6),
`Signal`'s lack of a level, `BarReason` (which already works for v2, since it is
found by `feature_frame`/`machine` introspection rather than by contract).

The measurement that makes this a number rather than an opinion: **how many bars
of a streamed `state_machine_v2` run carry a target that differs from its
whole-history target** — re-run through the existing driver on the real frame
rather than on `synthetic_ohlcv`, which is the frame the determinism suite has
never used for it.

**Declared readings.**

1. **0 differing bars on a real frame.** Then (d) is a *missing adapter*, not a
   correctness question, and its scope is a runner class plus a level on
   `Signal` — sized in the write-up.
2. **Non-zero.** Then the exposure path has a real streaming defect the
   synthetic suite cannot see, which is (e)'s shape appearing on a second
   contract, and that outranks the adapter.
3. **`state_machine_v2` reads `crowding`**, so a perp frame runs it into (a) as
   well. Reading 1 or 2 is therefore taken on the **crowding-neutral** pair —
   both sides of the comparison without funding — and the funded case is
   reported as context, never as the verdict. Same rule as (b) above.

---

## Stopping rules

- **(b) reading 2 stops the phase.** A pass-through mode that differs is a bug,
  and finishing a census on top of one would be measuring around it.
- **Any item may close as "not a gap"** and that is a result, not a failure to
  find one. R10a's value was partly in proving four of five features identical.
- **Nothing is built in this phase.** Not the `ExitMode` on the runner, not the
  size field, not the exposure runner. The output is numbers plus a scoped
  proposal, and the build is the phase after, argued from what these say.

---

## What this deliberately does not do

- **It does not close (a).** Carrying funding through `Bar`, `BarBuffer`, the
  feed and the storage schema is a phase, as CLAUDE.md says. R10d is what tells
  us whether closing (a) alone is *sufficient* for R10's gate or merely
  necessary — which is the whole reason to measure before building it.
- **It spends no holdout.** No market hypothesis is under test.
- **It moves no published figure**, and every one is re-verified bit-identical
  against a clean `main` worktree at the end, the same control R10a and R10c ran.

---

## Then

R10's gate becomes evaluable: six census items, all six numbers, and a scoped
build ordered by what actually blocks *"research and live code paths produce
identical output"*.
