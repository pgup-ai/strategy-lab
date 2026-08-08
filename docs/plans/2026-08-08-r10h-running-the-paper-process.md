# R10h — the paper process, and the delayed oracle performed

**A design plan, not a pre-registration.** Its checks are binding, and its
readings are declared here because one of them is "the harness failed".

**Charter:** [docs/research/2026-08-03-market-dynamics-engine.md](../research/2026-08-03-market-dynamics-engine.md),
phase R10 — *"research and live code paths produce identical output"*. R10
shipped `LiveFeed` and R10g shipped the book. This phase runs them.

---

## What R10 left behind

Two facts, both checkable in one command each.

1. **Nothing runs any of it.** `grep "LiveFeed(" src/` outside its own module
   returns nothing, and so does the same grep for `PaperBook`/`ExposureBook`.
   There is no `paper` beside `replay`, `serve` and `browse`. Both are libraries
   with no caller.
2. **The delayed oracle has never met a venue.** R10's framing was that a live
   feed has no oracle at the moment it runs but has a *delayed* one — run it over
   a window, fetch the same range later, replay it, diff — and that this is the
   gate. Every test satisfies it with the scripted `_Venue` double and an
   injected clock. The plan's check 5 promised the one venue-touching test would
   be `db`-marked; `grep pytest.mark.db tests/test_live_feed.py` returns nothing.

So R10's gate was passed against a test double. That is not a criticism of the
tests — offline tests are why the class is trustworthy at all — but a test double
agrees with whatever it was written to agree with, and the whole reason this
phase has a delayed oracle is that agreement with a fixture proves nothing about
a venue.

---

## Preparing to run already found the bug

**A perp poll never gets funding, and with R10's withholding rule the process
stalls on its first poll and emits nothing, forever.**

`funding_coverage_gaps` certifies a cadence rather than assuming 8h, so it needs
the interval **observed twice** — three settlements inside the window. A poll
window is `DEFAULT_LOOKBACK_BARS = 5` bars wide. Measured against the real
database and the real stored funding, on BTC/USDT perp 4h:

| poll window | span | settlements | funding column |
|---|---|---|---|
| 5 bars | 16h | 2 | **absent** |
| 8 bars | 1d 12h | 5 | present |
| 20 bars | 4d 16h | 14 | present |

Five bars is 16 hours against an 8h interval, so the window can hold two
settlements and the guard needs three. At 15m five bars is 75 minutes and holds
none. **There is no timeframe this program trades at which a 5-bar window funds
itself.**

Note what this is not. The data is not missing — every settlement is stored. The
guard is being asked a question the window is too narrow to answer, and its
binary answer is "uncovered", which `required=False` turns into "no column",
which the feed now reads as a coverage stall. Before the withholding rule the
same window produced an unfunded stream instead: `crowding` neutral for the life
of the process, silently, which is M20 on the live path. Both are broken. The
stall is the better failure, and it is why this surfaced now rather than in a
month of quietly wrong paper results.

**This is the same defect the browser already fixed once**, and CLAUDE.md already
states the rule for it: *"The top-up reaches back to the earlier of the candle
lookback and the last stored settlement — five 15m bars is 75 minutes against an
8h interval, so a lookback-sized request steps straight over the gap it exists to
close."* `server._fetch_funding` obeys it. `LiveFeed` re-derived the same
five-bar window and walked into the same hole. A rule stated in prose in one
module is not shared; that is the lesson, and the fix puts the widening where the
funding call is rather than in a second place that has to remember.

---

## What gets built

1. **The poll window is bounded below by its own funding.** `_fetch_recent`
   widens its request on a perp until the window can answer the coverage
   question, then attaches funding as it does today. It lives there, beside the
   `with_funding_column` call it exists to satisfy, rather than in `_since_ms`:
   the feed's lookback keeps meaning "bars to re-read for corrections", the
   widening stays next to the thing that needs it, and `LiveFeed` keeps the
   injection seam that makes every one of its tests offline. Over-returning is
   already anticipated — `_seen` dedups it, and the comment on `_seen` explains
   that pruning was rejected precisely because a venue may return more than it
   was asked for.

   **The reach-back is measured, never assumed.** The settlement interval is
   per-contract and nothing here hardcodes 8h, so the floor comes from the stored
   settlements themselves: reach back far enough to contain the number the guard
   needs, read off `db.funding`.

2. **A `paper` command** — the process. `backfill` → `prime_bars` → `stream` →
   `StrategyRunner` (with its `ExitMode`) → `PaperBook`, persisting signals and
   `bar_reasons` under `Mode.PAPER` through the same three writers `replay` uses.

   - **Bounded by wall clock** (`--for-minutes`), because `stream()` does not
     terminate and an unbounded process cannot be a gate. Wall clock rather than
     a bar count: a stalled feed emits no bars, and a bound that waits for bars
     would hang exactly when something is wrong.
   - **Persisted per bar, not at the end.** A run measured in hours that writes
     once at the end loses everything to the failure it was there to observe.
     `write_signals` is idempotent within a run, so an incremental flush is the
     same record arriving sooner.

3. **A bounded retry around the fetch**, because a live process that dies on one
   transient HTTP error cannot complete a window. Bounded and loud: it warns each
   time and re-raises once the budget is spent, so a persistent outage still ends
   the run rather than being absorbed.

4. **`scripts/r10h/delayed_oracle.py`** — the gate. Reads what the paper run
   stored, fetches the same range, replays it, and diffs bars, signals and
   `bar_reasons` per bar.

---

## Acceptance checks

1. **The process runs against Binance for a real window and emits bars.** Not a
   fixture: `strategy-lab paper` against the venue, with the count of bars,
   signals and reasons reported.
2. **Every live bar equals the stored candle for its interval** — `ts_open_ms`,
   OHLCV and `funding_rate` — once the same range has been fetched and stored.
3. **Signals are identical** between the paper run and a replay of the same
   range: same bars, same sides.
4. **`bar_reasons` are identical** per bar, per feature and for the state. This
   is R10a's diff pointed at a live run instead of a replayed one.
5. **A perp poll carries funding.** `crowding_measured` is true for the paper
   run, and `funding_withheld_polls` is 0 over a window whose settlements are
   stored — the direct check on the finding above.
6. **The full suite passes and no published figure moves.**

---

## The readings, declared before the run

1. **Everything matches.** The gate passes and R10's delayed oracle is performed
   rather than simulated.
2. **Bars differ.** Either the feed's closed-bar rule is wrong or the venue
   revises what it served. Either is the phase's result, and check 2 is what
   distinguishes them: a revision moves OHLCV under a matching timestamp, a
   closed-bar error moves the timestamps.
3. **Signals differ while bars agree.** The two drivers disagree given identical
   input, which `tests/test_replay_determinism.py` says cannot happen — so it
   would be a gap in that suite, found the way R10f's was.
4. **Nothing to compare.** Zero bars, or a window so short the strategy never
   emitted. **This is a failed run, not a pass**, and it is called out here
   because an empty diff prints the same "0 differences" as a perfect one. Check
   1 exists to make the count explicit before any diff is read.

---

## What this deliberately does not do

- **No real money and no order placement.** That is R11, and its gate —
  *"expected vs actual fills"* — needs this phase's "expected" to exist first.
- **No supervision or alerting.** R10 deferred the operational half to whichever
  phase owns the process, and this phase owns *running* it, not keeping it alive
  unattended. Retry is in scope because a window cannot complete without it;
  restart policy, health alerting and a service definition are not, and the
  funding-stall escalation already filed in §12 stays there.
- **No websocket.** Unchanged from R10: a later adapter behind the same protocol.
- **No new research claim.** No strategy, feature, policy or engine change.

---

## Then

R11: the canary at small real size, with expected-vs-actual fills finally
comparable — because "expected" is what this phase produces.
