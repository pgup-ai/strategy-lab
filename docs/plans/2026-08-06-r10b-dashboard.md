# R10b — from one instrument to a board

**A design plan, not a pre-registration.** Like R10a it ships an interface and a
cost, not a figure, so it declares a **design** and **acceptance checks** rather
than thresholds a protocol has to bind in advance.

**Charter:** [docs/research/2026-08-03-market-dynamics-engine.md](../research/2026-08-03-market-dynamics-engine.md),
phase R10, the program's main line since **M31**. This is the "command-and-control
centre" half of the product vision.

---

## What already exists, measured rather than assumed

The single-instrument view is **done**. `browse` serves a dataset selector, a
strategy selector, an exit-mode selector, a price chart whose markers are the
engine's own fills, provenance, and a rendered **"Why this bar"** panel carrying
the state and every feature value. `POST /api/refresh` already tops up candles
**and funding together**, which the coverage guard requires.

So R10b adds **breadth**, not depth. Three measurements shape it.

**1. A warm analysis costs ~300–500 ms per instrument.** Measured on the stored
perp frames: BTC 5,602 ms cold (imports, pool, caches), then ETH 491 ms and SOL
310 ms warm. Sixteen datasets is therefore **~6 s serially**, which is too slow
to do on every poll and fine to do once.

**2. The payload is ~40× larger than a tile needs.** One call returns **15,124
bars and 694 markers**; a grid tile needs a state label, the latest signal, and
enough points for a sparkline.

**3. Some frames refuse without a `start`.** BTC/USDT perp candles begin 40 h
before the venue's first settlement, so a full-frame request raises
`FundingUnavailable` — permanently, and correctly. A board that asks for "every
dataset" hits this on the first instrument unless each frame is bounded by its
own funding span.

---

## What gets built

### 1. A board endpoint that slices, never recomputes

`GET /api/board` returns one row per (dataset, strategy) pair: the current state,
the latest fill, the last feature values, the bar it is as of, and provenance —
plus a short tail of closes for a sparkline.

**It must not compute its own answer.** The charter's standing rule is that the
browser is *not free to disagree* with a backtest, which is why its markers are
fills off the engine's own `from_signals` call rather than raw signals. A board
that derived "current state" by a cheaper route would be a **third** path that
can disagree with the other two. So the board calls the same `build_analysis` and
**slices** the result. Slower and correct beats fast and divergent.

### 2. Bounded per frame, by that frame's own funding span

Each row derives its own `start` from `funding_span`, the same rule
`tests/test_state_machine_gate.py` and R7d's harness already follow. A dataset
whose funding cannot cover its candles reports **why** in its row rather than
failing the whole board — one instrument's permanent leading gap must not blank
the other fifteen.

### 3. Cached on the bar, because that is what changes

The result of a full-history recompute changes only when a new bar closes, so the
cache key is `(identity, strategy, exit_mode, last_bar_ts)`. At 4 h bars an entry
is valid for hours; at 15 m, minutes. **A cache keyed on wall-clock time would
expire while the answer had not changed**, which is the same "recompute is cheap,
so recompute" reasoning R10a used to decide *not* to persist the why-layer —
here it argues for holding the result, not for storing it.

Nothing is written: the cache is in-process and dies with `browse`. **The browser
still persists nothing** — no report directory, no `signals` row, no schema.

### 4. Refresh stays explicit

`POST /api/refresh` already exists and already advances candles and funding
together. The board offers it per row and for all rows; it is **not** on a timer.
A background poll that fetches from a venue on its own schedule is a different
thing from a page the user refreshes, and only the second one is honest about
when it last talked to the exchange — which is exactly what the provenance panel
exists to say.

---

## Acceptance checks

1. A board over **every stored perp dataset** returns a row per dataset, and a
   dataset whose funding cannot cover its candles reports that in its row rather
   than raising.
2. Each row's state and latest fill are **identical** to what
   `/api/analysis` returns for the same (dataset, strategy) — asserted directly,
   because a board that disagrees with the single view is the failure mode this
   design is arranged to prevent.
3. A second board request with no new bar serves from cache and issues **no**
   database query for the cached rows.
4. A new bar invalidates exactly the rows whose instrument received it.
5. `browse` still writes nothing: no `reports/` directory, no `signals` row, no
   `bar_reasons` row, no schema change. Asserted, not assumed.
6. The board's first paint is **under a second** on a warm process, which needs
   the rows to arrive incrementally or in parallel rather than as one 6 s blob.

---

## What this deliberately does not do

- **No live feed.** Poll-and-recompute over stored candles sidesteps every one of
  the six census items by construction, because it *is* the research path. The
  feed is for trading, not for looking.
- **No new research claim.** No strategy, policy, feature or engine change, and
  no published figure moves. That is check 5's real purpose.
- **No cross-asset aggregation** — no portfolio equity, no combined exposure.
  Positions from different instruments do not add up without a sizing model, and
  M31 just closed the line that would have supplied one.
- **No equities yet.** The census's item (f) — the Yahoo fetcher rewrites past
  bars on a dividend, so an append-style update diverges from a fresh fetch —
  makes an equity row's freshness a different problem. Perps first; that item is
  still unmeasured and this is not the phase that measures it.

---

## Then

With a board that tells the truth about several instruments at once, the next
question is which of the remaining census items actually blocks trading — (a) is
measured, (b), (c), (d) and (f) are not.
