# The read-only research browser — implementation plan

**Goal:** one local page that renders any registered strategy over any stored
candle set — signals, the state and feature values behind them, and the
provenance needed to trust the numbers — with a strategy switcher, and no
`backtest` re-run in the loop.

**Answers Q6** in the charter: build the browser before R9. The 2026-08-05
state-of-play entry is the assessment this implements.

**Charter:** [docs/research/2026-08-03-market-dynamics-engine.md](../research/2026-08-03-market-dynamics-engine.md).
This is **not** an MDE research phase. It ships no strategy, no measurement and
no published figure; it is the tooling the research uses, and it must not be
able to change a result.

---

## The one design move everything else follows from

**Signals are computed server-side, per request, by calling
`generate_signals(df)` over stored candles — the same whole-history vectorized
call `run_backtest` makes.**

Not by joining to the `signals` table, and not from the event path. That single
choice sidesteps all six live-vs-backtest divergences in the state-of-play entry
*by construction*, because the browser **is** the backtest path:

| Divergence | Why it cannot bite here |
|---|---|
| funding/crowding absent on replay | we attach the column exactly as `backtest` does |
| exit modes do not exist in replay | we render a named `ExitMode`, the engine's own |
| size is engine-side | we have the engine |
| `TargetExposure` cannot run on replay | we call `compute_target` directly |
| determinism suite is blind to funding | not in this path at all |
| equity bars are not append-only | we re-read the frame rather than appending |

The cost is that the browser is **not live in the tick sense** — it is
poll-and-recompute. At 4h bars that is indistinguishable from real-time, and it
is honest about what it is.

---

## Two things settled by measurement before planning

**Recompute is not the bottleneck, and I was wrong about it by two orders of
magnitude.** The state-of-play entry estimated "~seconds for the state machine
on 15k bars" and warned it would be "wrong for anything multi-user". Measured on
the stored 15,118-bar BTC/USDT perp 4h frame:

```text
load_candles(15,118 bars)          272 ms     <- the actual cost
state_machine_v1.generate_signals   19 ms
state_machine_v2.compute_target     18 ms
trend_rider_v1_deepseek_v4_pro       6 ms
turnaround_v1 / v2                 3 / 2 ms
donchian / multi_horizon             1 ms
tsmom / ema_cross / trend_following  0 ms
```

So **cache the frame, not the signals**, and a strategy switcher can recompute
on every toggle without anyone noticing. The multi-user caveat in the assessment
is withdrawn.

**The vendored chart already has every primitive the two contracts need.**
`lightweight-charts-5.2.0.standalone.production.js` (191 KB, already inlined by
`report.py`) exposes `CandlestickSeries`, `LineSeries`, `AreaSeries`,
`HistogramSeries`, `BaselineSeries` and `createSeriesMarkers` under the v5
`chart.addSeries(LWC.XSeries, …)` idiom `report.py` already uses.

**`BaselineSeries` is the answer to the `TargetExposure` rendering problem** — it
draws a signed series against a zero baseline, which is exactly a −1..1 target
(measured: `state_machine_v2` spans the full range). No new frontend dependency,
and the "markers cannot draw a continuous target" gap closes with a series type
that was already in the file.

---

## Scope — what this is not

- **Not a live feed.** No websocket, no `Bar` change, no Phase 1b. Freshness is
  "re-fetch recent candles and recompute", reusing `server.py`'s existing
  fetch-and-upsert logic.
- **Not React.** A single server-rendered page reusing the vendored asset and
  `report.py`'s visual language. React + Vite is Phase 3, when the C2 centre
  needs real component structure; adding node tooling to a Python research repo
  to draw one chart buys nothing today.
- **Not a replacement for `serve`.** The per-run `plot.html` is the
  **reproducibility record** — frozen, dated, byte-identical on re-render. The
  browser is a **view**. Two commands, and the browser must never write into
  `reports/`.
- **Not an order path.** Bound to 127.0.0.1, no trading and no execution.
  "Read-only" here means it writes no *derived* state — no report directory,
  no `signals` row, no schema — and never becomes the record of a run. The
  one exception is deliberate and is Task 2's refresh: it upserts
  `market_candles` through `server.refresh_candles`, the existing fetch path,
  called rather than copied. Fetching newer bars of the same public data is
  not the kind of write this constraint exists to prevent.

---

## Backward compatibility — non-negotiable

- `market_candles` stays **read-only** except through the existing fetch path.
- `signals` is append-only; this feature writes **nothing** to it.
- The four original strategies' `stats.json` / `trades.csv` / `equity_curve.csv`
  stay byte-identical on their canonical `STRATEGIES.md` commands.
- `report.py` and `serve` keep working unchanged. If refactoring is needed to
  share code, the frozen reports must re-render byte-identically.

---

## The dependency decision: FastAPI + Pydantic, adopted

**Decided 2026-08-05.** The reason is **inbound validation**, and it is worth
stating precisely, because the first draft of this plan gave the weaker one.

`server.py` hand-rolls query parsing today: a `required()` closure raising
`ValueError` on empty strings, and `after = int(values[0]) if values else None`
with no bounds check — five parameters. This browser's surface is roughly
double: exchange, market_type, symbol, timeframe, strategy, exit_mode, cost
model, `position_pct`, date range, contract. **Every hand-parsed parameter is a
place where a wrong or missing value silently becomes a default, and that is
exactly M20** — a funding column silently absent changed a published figure, and
it took a second-asset replication to notice. An API that silently defaults
`exit_mode` reproduces that failure precisely: a plausible number computed under
settings nobody chose. FastAPI makes it a 422 naming the field.

**What was wrong in the first draft.** It claimed typed *response* models are
what stop provenance fields being dropped. They are not needed for that: this
repo already enforces required fields with frozen dataclasses and validating
`__post_init__` (`TargetExposure`, `SignalSet`, `Bar`, `Signal`). Pydantic adds
nothing outbound that the house idiom does not already do. **The value is
entirely inbound.**

**The argument deliberately not relied on.** "This is Phase 3's API anyway" is
weak, and it is weak for a reason this plan's own author supplied: the C2 centre
is R10, gated behind R9, on numbers ETH showed to be thin — one scalar by 0.083,
with the untuned machine failing out of sample. If R9 kills them, Phase 3 never
happens. Adopt this because *the browser* needs it, not because a successor
might.

**The tiebreaker.** `http.server` is already marked *"Replace. Synchronous,
fetches from the exchange inside a GET handler"* in the design doc's own
inventory. Adding four endpoints to a component the repo has condemned is debt
this codebase is usually good at refusing.

**Measured footprint**, resolved clean against vectorbt / numba / pandas:

```
runtime (9)  fastapi 0.141.1   starlette 1.4.1   pydantic 2.13.4
             pydantic_core 2.46.4   uvicorn 0.52.1   anyio 4.14.2
             h11 0.16.0   annotated-types 0.8.0   typing-inspection 0.4.2
dev (2)      httpx 0.28.1   httpcore 1.0.9        # fastapi.testclient.TestClient
```

Runtime goes in `dependencies`, the two test-only ones in `optional-dependencies.dev`.

---

## File structure

| File | Responsibility |
|---|---|
| `src/strategy_lab/api/app.py` | FastAPI app, read-only, 127.0.0.1 |
| `src/strategy_lab/api/models.py` | Pydantic response models, including provenance |
| `src/strategy_lab/api/analysis.py` | load frame → attach funding → compute → payload |
| `src/strategy_lab/browser/page.py` | the served page, reusing the vendored asset |
| `tests/test_api_analysis.py` | payload correctness and provenance |
| `tests/test_api_routes.py` | the endpoints |

---

## Task 1: The analysis payload

**The core, and everything visible depends on it being right.**

`analysis.py` takes `(identity, strategy, exit_mode, cost settings)` and returns
candles, the strategy's output, the "why" layer, and provenance. It attaches the
funding column through `_with_funding_column`'s rule — perp only, via
`align_funding_to_bars`, never a reindex.

**Both contracts, dispatched by registry:**

- `strategies/registry.py` → `generate_signals` → entry/exit markers, plus
  `position_size` when the strategy provides one.
- `strategies/exposure_registry.py` → `compute_target` → a level series for
  `BaselineSeries`, plus `rebalance_target` and `position_fraction` when a
  backtest is requested.

**The "why" layer is returned, never persisted.** For a state-machine strategy
the payload carries per-bar state and the four feature values. This is the thing
the dashboard exists for and it is currently discarded by both paths — and
returning it rather than storing it means **no migration, no schema change, and
no new way for the research record to be wrong.**

**Provenance rides in every response, non-optional:** `crowding_measured`,
`exit_mode`, `warmup_bars`, the cost model, `funding_attached`, the frame's
first and last bar, and the strategy's `version`. M20 is the reason: two runs of
one strategy can differ because one had the funding column, and a number shown
without that context will eventually contradict the charter with no way to see
why.

- [ ] Steps: failing tests → implement → mutation-test → commit.

**Test that matters most:** the payload's markers for a boolean strategy match
what `run_backtest` produces on the same frame and exit mode. If the browser can
disagree with the backtest, its entire justification is gone.

---

## Task 2: The API

Read-only endpoints: list datasets, list strategies (both registries, labelled
by contract), and the analysis payload. Plus a refresh endpoint that reuses
`server.py`'s fetch-recent-and-upsert path so "newest bars" does not mean
"restart the process".

Bound to 127.0.0.1. No mutation beyond the existing candle upsert.

- [ ] Steps: failing tests → implement → mutation-test → commit.

---

## Task 3: The page

One page: candlestick chart + markers or baseline depending on contract, a
strategy switcher that re-requests and redraws, a per-bar inspector showing the
state and feature values for the hovered bar, and a provenance strip that is
always visible rather than hidden in a tooltip.

Harvest `report.py`'s formatting helpers and CSS rather than reinventing them;
inline the same vendored asset.

- [ ] Steps: implement → verify against a known backtest by eye and by test → commit.

---

## Task 4: Wire it up

A CLI command beside `serve`, not replacing it. Document in `CLAUDE.md` and
`README.md` what each is for — the frozen record versus the live view — because
the distinction is the thing most likely to erode.

- [ ] Steps: implement → full suite → docs → commit.

---

## GATE

- [ ] The browser's markers match `run_backtest` on the same frame and exit mode
- [ ] Both contracts render — markers for `SignalSet`, a baseline for `TargetExposure`
- [ ] The switcher changes strategy without a re-run and without a page reload
- [ ] Per-bar state and feature values are visible for a state-machine strategy
- [ ] Provenance is visible on every view, `crowding_measured` included
- [ ] The four original strategies remain byte-identical; `reports/` is untouched
- [ ] Full suite green, ruff clean

---

## Self-review notes

**The risk that matters most.** A browser that silently disagrees with a
backtest is worse than no browser: it would launder a wrong number through a
trustworthy-looking chart. That is why Task 1's first test compares against
`run_backtest` rather than asserting the payload's shape, and why provenance is
non-optional rather than a detail panel.

**The second risk is scope.** Every item under "what this is not" is something
that would feel natural to add and would turn a week into a phase. The C2 centre
is R10; this is the tool that makes R7–R9 easier to look at.

**Deliberately unresolved.** Multi-symbol comparison on one chart, saved views,
and anything about orders. They belong to the product, and the product needs R9
to have said the strategies are worth controlling.
