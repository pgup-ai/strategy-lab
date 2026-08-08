# R10 — the live feed, and the delayed oracle that checks it

**A design plan, not a pre-registration.** Its checks are binding, and one of
them exists only because a live feed cannot be checked the ordinary way.

**Charter:** [docs/research/2026-08-03-market-dynamics-engine.md](../research/2026-08-03-market-dynamics-engine.md),
phase R10 — *"research and live code paths produce identical output"*. The census
closed that for signals (R10e/R10f) and R10g closed it for positions. What is
left is the feed: `MarketDataFeed` has had exactly one implementation since Phase
1a, and it reads from Postgres.

---

## The problem this phase has that none of the others did

**A live feed has no oracle.** That is why the book was built first — `trades.csv`
is frozen and a book either reproduces it or does not. A websocket, or a poll,
produces bars nothing can be diffed against at the moment they arrive.

It has a **delayed** one, though, and that is the gate:

> Run the feed over a window. Later, fetch the same range historically and replay
> it. The bars must be identical, and so must the signals.

A live bar and a stored bar for the same interval are the same fact recorded
twice. Anything the live path does differently — a partial candle taken as final,
a duplicate after a reconnect, a timestamp off by a millisecond — shows up as a
difference against the record the venue itself will serve tomorrow.

---

## It polls, and that is a decision rather than a shortcut

A websocket is the obvious implementation and it is the wrong one to start with.

- **The protocol does not care.** `MarketDataFeed.stream()` yields `BarEvent`s;
  where they came from is the adapter's problem, which is the whole point of
  having a protocol.
- **The venue REST client already exists and is tested.** A websocket adds a
  dependency, a reconnect state machine and a message-framing layer, all
  untestable without a network.
- **At this program's frequency the difference is not observable.** The traded
  timeframes are 4h and 1w. `browse` already argues the same thing for its own
  refresh: *"for 4h bars, poll-and-recompute is indistinguishable from
  real-time."* A websocket is an optimisation for a bar size nothing here trades.

So `LiveFeed` polls, and the phase that needs sub-minute bars can write the
socket behind the same protocol without touching the runner, the book, or any
strategy.

---

## Two follow-ups stop being cosmetic here

Both are already in the charter's §12, filed when they cost nothing.

1. **`backfill()` yields `Bar`, `prime()` takes a DataFrame** — so they do not
   compose, and a live process cannot warm itself from history. That is fatal
   rather than awkward: `state_machine_v1` emits nothing for its first **2,192
   bars**, which at 4h is a year of waiting before a freshly started process says
   anything.
2. **`Subscription.include_forming` is declared and read nowhere** — confirmed,
   `feeds/base.py` is its only occurrence in `src/` and `tests/`. Harmless while
   the only feed yields closed bars; a poll lands mid-bar routinely.

---

## What gets built

1. **`StrategyRunner.prime_bars(bars)`**, appending without emitting, with the
   existing `prime(DataFrame)` delegating to it. That is what makes
   `backfill() → prime` compose, and it is the smaller primitive: the buffer
   holds `Bar`s and the DataFrame is the adapter.
2. **`LiveFeed`** (`feeds/live.py`), polling the venue on the timeframe's own
   cadence and satisfying `MarketDataFeed` — including the two guarantees the
   protocol states and `isinstance` cannot check: ascending order, and never the
   same `(instrument, timeframe, ts_open_ms, is_closed)` twice.
3. **Forming bars are honoured rather than ignored.** With
   `include_forming=False` the feed yields a bar only once it has closed. With
   it true, a forming bar is yielded as `is_closed=False` and **superseded** by
   the closed bar for the same interval — which `BarBuffer` already handles,
   since a repeated timestamp replaces last-wins.

---

## Acceptance checks

1. **A live window equals a replay of the same window.** Bars first, then
   signals: run `LiveFeed` over a window, store what arrived, and replay the same
   range from Postgres — identical `ts_open_ms`, OHLCV and emitted signals. This
   is the delayed oracle and it is the phase's gate.
2. **The protocol's two unenforceable guarantees hold.** Ascending order and no
   duplicate `(instrument, timeframe, ts_open_ms, is_closed)`, asserted with
   `tests/test_replay_feed.py`'s existing behavioural contract checks rather than
   a second copy of them — the protocol's own docstring says `isinstance` proves
   only that the names exist.
3. **A forming bar is superseded, never duplicated.** The same interval arriving
   forming and then closed leaves **one** bar in the buffer, carrying the closed
   values, and `replaced_duplicates` counts it.
4. **A cold start reaches the state a replay reaches.** A runner primed from
   `backfill()` and then fed live bars emits what a replay of the whole range
   emits — which is `tests/test_replay_determinism.py`'s primed-runner comparison,
   pointed at the live path.
5. **No network in the test suite.** The feed is driven through an injected clock
   and an injected fetch, so every check above runs offline; the one test that
   touches a venue is `db`-marked and skips without one.
6. **Every published figure bit-identical**, and the full suite passes.

---

## What this deliberately does not do

- **No websocket.** See above; it is a later adapter behind the same protocol.
- **No order placement.** Paper only — the book holds a position and nothing
  reaches a venue. R11's canary is where real size appears, and its gate
  (*"expected vs actual fills"*) needs this phase's "expected" first.
- **No scheduler or daemon.** A long-running process is an operational concern,
  and the phase that owns it should own restart, supervision and alerting
  together rather than acquiring them one at a time.
- **No new research claim.** No strategy, feature, policy or engine change.

---

## Then

R11: the canary, at small real size, with expected-vs-actual fills finally
comparable — because both halves now exist and agree offline.
