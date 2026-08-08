# R10f — funding on the event path, closing census items (a) and (e)

**A design plan, not a pre-registration.** Its checks are binding and one of them
is a **published figure that must move to a specific value** — R10a measured the
gap exactly, so closing it has a right answer rather than a direction.

**Charter:** [docs/research/2026-08-03-market-dynamics-engine.md](../research/2026-08-03-market-dynamics-engine.md),
phase R10. Follows R10e, which made the event path run every strategy and every
contract while leaving all of it crowding-neutral. This is the "correct" half.

---

## The gap, already measured

R10a ran `state_machine_v1` over 6,048 BTC/USDT perp 4h bars on both paths:

| | |
|---|---|
| `crowding` differing | **6,048 / 6,048** |
| `direction`, `energy`, `stability`, `strength` differing | **0** |
| state differing | **488 (8.1%)** |
| research features with crowding pinned to the live constant, vs stored states | **0 residual** |

So (a) is the *entire* live/research gap for this strategy, and its size on the
published number is known too: +16.44% / Sharpe +0.801 crowding-neutral against
the charter's **+15.45% / +0.896** measured (M20).

**And (e) is why no suite catches it.** `tests/test_replay_determinism.py`
compares on `synthetic_ohlcv`, which carries no funding, so it proves
crowding-neutral ≡ crowding-neutral. It is structurally incapable of seeing the
one difference that exists.

---

## The load-bearing design decision

`state_machine_core.build_feature_frame` decides whether crowding is real with

```python
crowding_measured = FUNDING_COLUMN in df.columns
```

**So the buffer must materialize `funding_rate` only when bars actually carry
one.** A column always present and NaN for a spot instrument would report
`crowding_measured=True` and feed the feature garbage — turning a fallback that
is *correct* on spot and equity into a silent wrong answer, which is worse than
the gap being closed. Presence must keep meaning "measured".

Three consequences.

1. **`Bar` gains `funding_rate: Decimal | None`**, defaulting to `None`, beside
   `quote_volume` and `trades` — the existing shape for "some venues have this".
   It is **not** added to `_DECIMAL_FIELDS`' unconditional check, because `None`
   is a legitimate value there where it is not for a price.
2. **`BarBuffer` materializes the column iff a bar carried one**, and refuses a
   stream that mixes the two. A perp whose funding stops arriving mid-run is a
   feed fault, and silently dropping the column changes the strategy rather than
   reporting it.
3. **`ReplayFeed.from_database` attaches funding through
   `backtests.funding_frame.with_funding_column`** — the same function the
   backtest uses, not a reimplementation, so the alignment rule (settlements land
   in the bar whose interval *contains* them, because Binance stamps up to 47 ms
   late) and the coverage guard are shared rather than duplicated. A perp replay
   whose funding cannot cover its candles must **refuse**, exactly as a backtest
   of the same range does.

---

## What gets built

1. `Bar.funding_rate`, and `_row_to_bar` reading it from a frame that has the
   column.
2. `BarBuffer` carrying it into `frame()`, present only when real.
3. `ReplayFeed.from_database` attaching it for a perp, through the engine's own
   function and its coverage guard.
4. **The determinism suites run on a funded frame**, which is what closes (e) —
   and the closure is proven by mutation, not asserted.

---

## Acceptance checks

1. **A replay of a funded perp range emits the same signals as a backtest of the
   same range.** R10a's diff is the instrument: `crowding` differing goes
   **6,048 → 0**, the state **488 → 0**. Measured with R10a's own harness, not a
   new one.
2. **(e) is closed and the closure is non-vacuous.** The determinism suite runs
   on a frame carrying funding, and **pinning crowding to the live constant on
   one side makes it fail**. A suite that passes either way has not closed
   anything.
3. **A spot or equity bar carries no funding and `crowding_measured` stays
   `False`.** The fallback that lets the machine run off-perp is preserved, and
   asserted rather than assumed — this is the check that would catch an
   always-present NaN column.
4. **A perp replay whose funding does not cover its candles refuses**, with the
   same guard and the same message a backtest gives. BTC's permanent 40 h leading
   gap is the case that already exists.
5. **Every published figure bit-identical** against a clean `main` worktree. The
   backtest path must not move — this phase changes what *replay* computes, and
   the backtest is the published record.
6. **The full suite passes**, both contracts, both safety suites.

---

## What this deliberately does not do

- **No schema change.** `signals` and `bar_reasons` already store what they need;
  what was missing is what the strategy *read*, not what was written.
- **No live feed.** `MarketDataFeed` still has one implementation.
- **No new research claim.** The backtest is unchanged and remains the published
  record; this makes the event path agree with it.
- **No change to `build_feature_frame`'s fallback.** It is correct — the machine
  genuinely runs on spot and equity — and check 3 is what keeps it that way.

---

## Then

All six census items are closed. R10's gate — *"research and live code paths
produce identical output"* — becomes a measurement rather than an aspiration, and
paper trading is the next thing that can honestly be attempted.
