# Real-time crypto trading framework — Phase 0 design

Status: **proposed, awaiting approval**. No implementation has started.
Date: 2026-08-02 · Base commit: `55a7c0f`

---

## 1. What already exists

The repo is a **batch research lab**, not a trading system. Everything in it is
"fetch history → compute signals over the whole array → simulate → write an HTML report".
There is no concept of an event, a clock, a position, an order, or a running process.

What carries forward, essentially unchanged:

| Asset | Verdict |
|---|---|
| Postgres 16 + `market_candles` with `ON CONFLICT` upsert | **Keep.** Idempotent writes already solved. Identity model is right. |
| `(exchange, market_type, symbol, timeframe)` identity | **Keep and extend.** Correct grain; already handles `1w` vs `1wk` as distinct sets. |
| `SignalSet` / `Strategy` protocol + 4 strategies | **Keep, unchanged.** See §2 — they are already causal. |
| `backtests/report.py` Lightweight Charts renderer | **Keep and harvest.** Already written against the **v5** API. |
| `timeframes.py`, `filters/`, `universe/` | Keep. |
| `backtests/engine.py` (vectorbt) | **Keep as legacy**, do not extend. See §9. |
| `server.py` (`http.server` + `/api/candles`) | **Replace** in Phase 3. Synchronous, fetches from the exchange inside a GET handler. |
| `market_data/binance.py` | **Replace.** It is a ccxt REST pager with no rate limiting, no resumability, no websocket. |

Existing stored data (real, in Postgres now): 83,348 × BTC/USDT 15m, 3,060 × 1d,
438 × 1w, plus ~14k equity bars across SPY/QQQ/XLK/XLF/SMH/AAPL/MSFT/NVDA.

Two facts about the existing code that constrain everything below:

1. **`generate_signals(df)` takes the entire history and returns entire Series.** That is
   the opposite shape from an event callback. §2 resolves this without a rewrite.
2. **OHLCV is stored as `Float`** (double precision). Requirement 3 says decimal, never
   float. This is a direct conflict — see Decision D4.

---

## 2. The central design problem, and the evidence that resolves it

Requirement 1 says the strategy must run identically in backtest, replay, and live. The
obvious reading is "rewrite every strategy as `on_bar(event)`". That would throw away all
four strategies and the vectorized style the research workflow depends on.

It is not necessary. **A vectorized strategy is already a streaming strategy if it is
causal** — if row *t* of its output depends only on rows ≤ *t* of its input. Then calling
it on `df[0:t+1]` and taking the last row gives exactly what calling it on the full history
and taking row *t* gives.

I tested this rather than assuming it. All four strategies, 150 probe bars each:

```
strategy                          expanding window (full history)   rolling window (60 bars)
turnaround_v1                     EXACT MATCH                       MISMATCH (140 bad values)
turnaround_v2                     EXACT MATCH                       MISMATCH (152 bad values)
trend_following_deepseek_v4       EXACT MATCH                       exact match
trend_rider_v1_deepseek_v4_pro    EXACT MATCH                       exact match
```

Two conclusions, both load-bearing:

- **All four strategies are causal.** They can run in a live event loop today, with zero
  changes to strategy code. Requirement 1 is satisfiable by adapter, not rewrite.
- **A bounded rolling buffer is NOT safe.** The two EWM-based strategies produce wrong
  values from a 60-bar window, because `ewm(adjust=False)` is recursive from the first
  element — its value at *t* depends on *all* prior bars, not the last *N*. The
  SMA/`rolling`-based strategies are window-safe; the EWM ones are not.

  → **The runner must retain full history from a declared anchor, not a fixed-size ring
  buffer.** Had we picked a rolling buffer by default (the obvious choice), live signals
  would have silently diverged from backtest signals on exactly the two strategies
  currently in use for crypto.

### Cost of that decision, measured on the real 83k-bar dataset

```
one vectorized call over 83,348 bars                     5.9 ms
per-bar call, 10,000-bar buffer      1.15 ms/call  →  full replay ≈  0.2 min
per-bar call, 50,000-bar buffer      2.81 ms/call  →  full replay ≈  2.3 min
per-bar call, 83,348-bar buffer      4.32 ms/call  →  full replay ≈  6.0 min
```

So: **live is free** (4 ms once per bar interval), **replay is affordable on bounded
windows**, and **backtest must use the bulk path** — 5.9 ms versus 6 minutes is a 60,000×
difference, and it grows quadratically (5 years of 1m bars would be hours).

### The resulting shape

```
                       Strategy.generate_signals(df)          ← ONE implementation
                        ▲             ▲              ▲
        one call, full range   one call/bar    one call/bar
                        │             │              │
                  ┌─────┴────┐  ┌─────┴────┐   ┌─────┴────┐
                  │ BACKTEST │  │  REPLAY  │   │   LIVE   │
                  └─────┬────┘  └─────┬────┘   └─────┬────┘
                        │             │              │
                  SignalRouter → RiskEngine → ExecutionClient   ← ONE implementation
                        │             │              │
                   ReplayFeed    ReplayFeed     BinanceFeed / OKXFeed
                   (DB, bulk)    (DB, paced)    (WS + REST backfill)
```

The strategy layer and the risk/execution layer each have exactly one implementation
across all three modes. The only difference is bulk versus incremental signal production,
and that difference is **machine-verified equivalent** by the replay-determinism test
rather than asserted.

I would rather state this precisely than claim "one code path" and quietly mean something
weaker.

### This also gives the lookahead test for free

The same equivalence check *is* the lookahead detector: a strategy that peeks at the future
cannot produce the same value from a truncated frame. I verified the test has teeth by
writing two cheating strategies:

```
strategy          equivalence probe   poison probe    verdict
honest_control      0 bad bars          0 bad bars    PASS (causal)
blatant_cheat     106 bad bars          3 bad bars    FAIL — lookahead caught
subtle_cheat        2 bad bars          2 bad bars    FAIL — lookahead caught
```

`blatant_cheat` uses `close.shift(-1)`. `subtle_cheat` has no shift at all — it normalizes
by the **full-sample mean**, which is the realistic way lookahead actually enters a
research codebase. Both are caught.

The interesting result is the hit *rate*. The poison probe (overwrite every bar after *t*
with garbage, assert row *t* is unchanged) caught the subtle cheat on 2 of 8 probes — 25% —
while the equivalence probe caught it on 2 of 200 bars — 1%. **The cheap O(n/step) test is
the more sensitive one.** So both ship, with different jobs:

- **Poison probe** — primary lookahead gate, runs against every registered strategy, fast.
- **Equivalence probe** — replay-determinism gate, bounded window, proves bulk ≡ incremental.

---

## 3. Architecture

```
src/strategy_lab/
  core/          types.py  clock.py  ids.py          — vocabulary, no I/O, no deps
  feeds/         base.py  binance/  okx/  replay.py  — exchange abstraction
                 ratelimit.py  health.py
  storage/       schema.py  candles.py  signals.py   — Postgres
                 runs.py  quality.py
  engine/        runner.py  context.py  portfolio.py — event → strategy → signal
  risk/          limits.py  sizing.py  killswitch.py — strategies propose, this disposes
  execution/     base.py  simulated.py  live_*.py    — fills
  backtest/      simulator.py  metrics.py  walkforward.py
                 engine_vbt.py                       — legacy, frozen
  ops/           logging.py  metrics.py  alerts.py  clocksync.py  shutdown.py
  api/           app.py  routes/  models.py          — FastAPI, read-only, 127.0.0.1
  strategies/    (unchanged)
ui/                                                   — Phase 3, React+Vite
```

Dependency rule, enforced by an import-linter test: `core` depends on nothing;
`strategies` depends only on `core` + pandas; `feeds`, `storage`, `risk`, `execution` may
depend on `core`; `engine` orchestrates; `api` may not import `strategies` or `engine`
internals. **A strategy module importing `execution` or `feeds` is a test failure** — that
is how "a strategy bug must not bypass the risk layer" is enforced structurally rather than
by discipline.

### Data flow, live

```
WS frame → parse to Decimal → BarEvent(is_closed) ──┬─→ storage (closed bars only)
                                                     └─→ StrategyRunner
                                                            │ append to buffer
                                                            │ if closed and past warmup:
                                                            ↓
                                                     generate_signals(buffer)  → last row
                                                            ↓
                                                     Signal (persisted, append-only)
                                                            ↓
                                                     SignalRouter → OrderProposal
                                                            ↓
                                                     RiskEngine.evaluate() → approve/reduce/reject
                                                            ↓
                                                     ExecutionClient.submit(Order)
                                                            ↓
                                                     Fill → Portfolio → storage
```

Backfill-on-gap, reconnect, and staleness detection live entirely inside the feed layer.
The runner sees a clean, ordered, de-duplicated event stream and nothing else.

---

## 4. Key interfaces

```python
# ---------- core/types.py ----------
@dataclass(frozen=True, slots=True)
class InstrumentId:
    exchange: str        # "binance" | "okx"
    market_type: str     # "spot" | "perp"
    symbol: str          # canonical internal form: "BTC/USDT"

@dataclass(frozen=True, slots=True)
class Bar:
    instrument: InstrumentId
    timeframe: str
    ts_open_ms: int          # UTC epoch ms, exchange time — never local
    ts_close_ms: int
    open: Decimal; high: Decimal; low: Decimal; close: Decimal
    volume: Decimal
    quote_volume: Decimal | None
    trades: int | None
    is_closed: bool          # forming bars are explicitly flagged

@dataclass(frozen=True, slots=True)
class BarEvent:
    bar: Bar
    ts_event_ms: int         # exchange event time
    ts_recv_ms: int | None   # local receive time; None in backtest/replay

@dataclass(frozen=True, slots=True)
class FundingEvent:
    instrument: InstrumentId
    ts_event_ms: int
    funding_rate: Decimal
    next_funding_ms: int

MarketEvent = BarEvent | FundingEvent

class Clock(Protocol):
    def now_ms(self) -> int: ...
# LiveClock wraps time.time(); SimClock is driven by event timestamps.
# Nothing below core may call time.time() directly — enforced by a grep test.


# ---------- feeds/base.py ----------
@dataclass(frozen=True)
class Subscription:
    instrument: InstrumentId
    timeframe: str
    include_forming: bool = False

class MarketDataFeed(Protocol):
    name: str
    async def stream(self, subs: Sequence[Subscription]) -> AsyncIterator[MarketEvent]: ...
    async def backfill(self, sub: Subscription, start_ms: int, end_ms: int
                       ) -> AsyncIterator[Bar]: ...
    async def server_time_ms(self) -> int: ...
    def health(self) -> FeedHealth: ...

# BinanceFeed, OkxFeed, ReplayFeed all satisfy this.
# ReplayFeed reads Postgres and yields the same BarEvent objects — this is the
# injection point that makes requirement 1 true.

@dataclass(frozen=True)
class FeedHealth:
    connected: bool
    last_event_ms: int | None
    lag_ms: int | None          # server_time - last_event
    reconnects: int
    gaps_detected: int
    weight_used: dict[str, int]


# ---------- strategies/base.py (extended, backward compatible) ----------
class Strategy(Protocol):
    name: str
    version: str                # NEW — required; signals are stamped with it
    warmup_bars: int            # NEW — declared, excluded from results
    def generate_signals(self, df: pd.DataFrame) -> SignalSet: ...
# SignalSet is unchanged. The 4 existing strategies need 2 class attributes added
# and no logic changes.
#
# A `window_safe` flag was considered and DROPPED. Rationale: the runner keeps full
# history regardless, so nothing consumes it — and it would have shipped wrong.
# `trend_rider_v1_deepseek_v4_pro` looks rolling-only, and empirically matched exactly
# on a 60-bar window, but `filters/regime.compute_atr` uses ewm(span=14) internally.
# It matched by luck: ATR-14 converges fast enough that the `< 0.10` threshold rarely
# flips on the residual. A hand-declared boolean would have been wrong on 1 of 4
# strategies, silently. If the optimization is ever needed, derive the flag by AST
# inspection or an automated probe — do not ask a human to assert it.


# ---------- engine/runner.py ----------
class StrategyRunner:
    def __init__(self, strategy: Strategy, instrument: InstrumentId, timeframe: str,
                 clock: Clock, sink: SignalSink) -> None: ...
    def prime(self, history: pd.DataFrame) -> None: ...   # warmup, emits nothing
    def on_event(self, event: MarketEvent) -> Sequence[Signal]: ...
# on_event is the single entry point. It ignores non-closed bars unless the strategy
# opts in. It never looks at ts_recv_ms. It cannot reach the network or the DB.


# ---------- risk/limits.py ----------
class Verdict(StrEnum): APPROVE = "approve"; REDUCE = "reduce"; REJECT = "reject"

@dataclass(frozen=True)
class RiskDecision:
    verdict: Verdict
    approved_qty: Decimal
    reason: str
    limit_hit: str | None

class RiskEngine(Protocol):
    def evaluate(self, proposal: OrderProposal, state: PortfolioState) -> RiskDecision: ...
    def on_fill(self, fill: Fill) -> None: ...
    def kill_switch_state(self) -> KillSwitchState: ...
# A Signal is not an Order and has no path to ExecutionClient. Only SignalRouter
# converts, and it must call RiskEngine.evaluate first. Type-enforced, not convention.


# ---------- execution/base.py ----------
class ExecutionClient(Protocol):
    mode: Mode                              # backtest | replay | paper | live
    async def submit(self, order: Order) -> OrderAck: ...
    async def cancel(self, client_order_id: str) -> None: ...
    async def positions(self) -> Sequence[Position]: ...
# SimulatedExecutionClient serves backtest AND paper — identical fill logic, so paper
# results are directly comparable to backtest results.
# LiveExecutionClient is stubbed in Phase 1; the interface is fixed now.
```

Every `Order` carries `client_order_id`, derived deterministically from the signal:
`f"{run_id[:8]}-{strategy_short}-{ts_bar_ms}-{side}"` (≤36 chars for Binance
`newClientOrderId`). A retry after a crash regenerates the *same* id, so the exchange
rejects the duplicate instead of double-filling.

---

## 5. Storage

**Recommendation: stay on Postgres.** Justification against the alternatives you named:

- *Parquet + DuckDB* is better for wide analytical scans, but the live path is
  append-one-row-per-bar plus transactional signal/order writes. Parquet is poor at small
  appends, and you would still need a transactional store for signals/orders/positions —
  so it means running two systems, not one.
- *SQLite* would genuinely work at this scale and is one less container. But you already
  run Postgres, `ON CONFLICT` upsert is already written and tested, `JSONB` is a good fit
  for the signal `features` blob, and concurrent writer (live engine) + reader (dashboard)
  is where SQLite starts needing WAL tuning care.
- *TimescaleDB* buys hypertable partitioning and compression that matter at 10⁸–10⁹ rows.
  Five years of 1m BTC is ~2.6M rows. It would be cargo-culting.

Scale check: 2.6M rows/symbol-year at 1m. Postgres handles this without partitioning.

### Schema changes

```sql
-- market_candles: extend in place (nullable adds are cheap)
ALTER TABLE market_candles
  ALTER COLUMN open  TYPE NUMERIC(38,18),   -- ... high, low, close, volume
  ADD COLUMN ts_close_ms  BIGINT,
  ADD COLUMN quote_volume NUMERIC(38,18),
  ADD COLUMN trades        INTEGER,
  ADD COLUMN is_closed     BOOLEAN NOT NULL DEFAULT TRUE,
  ADD COLUMN ingested_via  TEXT;            -- 'rest_backfill' | 'ws' | 'gap_repair'

CREATE TABLE signals (
  id              BIGSERIAL PRIMARY KEY,
  run_id          UUID        NOT NULL REFERENCES runs(run_id),
  mode            TEXT        NOT NULL CHECK (mode IN ('backtest','replay','paper','live')),
  strategy_id     TEXT        NOT NULL,
  strategy_version TEXT       NOT NULL,
  exchange TEXT NOT NULL, market_type TEXT NOT NULL,
  symbol   TEXT NOT NULL, timeframe   TEXT NOT NULL,
  ts_bar_ms       BIGINT      NOT NULL,     -- open time of the bar that produced it
  ts_emit_ms      BIGINT      NOT NULL,
  bar_is_closed   BOOLEAN     NOT NULL,
  side            TEXT        NOT NULL CHECK (side IN
                    ('enter_long','enter_short','exit_long','exit_short')),
  strength        NUMERIC(10,6),            -- nullable; see Pushback P5
  entry_price     NUMERIC(38,18),
  stop_loss       NUMERIC(38,18),
  take_profit     NUMERIC(38,18),
  reason          TEXT        NOT NULL,
  features        JSONB       NOT NULL DEFAULT '{}',
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (run_id, strategy_id, strategy_version,
          exchange, symbol, timeframe, ts_bar_ms, side)
);
```

Historical and live signals land in **one table**, distinguished only by `mode` and
`run_id`, so diffing a replay against a live session is a self-join on
`(symbol, timeframe, ts_bar_ms, side)`. The `UNIQUE` constraint makes re-running a replay
idempotent.

Append-only is enforced by a `BEFORE UPDATE OR DELETE` trigger that raises, plus a
`REVOKE UPDATE, DELETE` on the application role — not by convention.

Also: `runs`, `orders`, `fills`, `positions_snapshot`, `equity_points`,
`data_quality_findings`.

### Decimal, precisely

Requirement 3 says "never float", and that is right for *money*. But every indicator in
this repo is float64 pandas math, and `Decimal` moving averages would be ~100× slower for
no benefit. The rule I propose:

> **Decimal at the boundaries — exchange parse, storage, order sizing, PnL, risk
> accounting. float64 inside the indicator layer only.**

Both exchanges return prices as JSON **strings** (verified, §13), so `Decimal(str)` parsing
is exact with no float roundtrip. The conversion to float64 happens at exactly one place —
building the strategy's DataFrame — and never converts back.

### Data quality

`storage/quality.py` runs after every backfill and on a live timer: gap detection against
the expected bar grid, duplicate timestamps, zero-volume bars, out-of-order arrivals,
and bars whose `ts_close_ms - ts_open_ms` disagrees with the timeframe. Findings are rows
in `data_quality_findings`, surfaced by `GET /health` and the dashboard — not log lines
that scroll away.

---

## 6. Read API contract

Defined now so Phase 3 is never blocked. FastAPI + Pydantic, so OpenAPI/JSON-schema is
generated from the same models the server validates against.

**Conventions**

- Base `http://127.0.0.1:8787/api/v1`. Binding is `127.0.0.1`, hardcoded, with no host
  parameter on the serve command and a startup assertion that rejects any other bind
  address. Making it configurable is how it eventually gets set to `0.0.0.0`.
- Every timestamp is `*_ms`, integer, UTC epoch milliseconds.
- **Every price and quantity is a JSON string**, not a JSON number — JSON numbers are
  float64 and would silently round-trip away the precision §5 just protected.
- Errors: `{"error":{"code","message","detail"}}`.
- Pagination: `{"data":[...], "page":{"next_cursor": str|null, "has_more": bool}}`.

**Endpoints**

```
GET /instruments            → available (exchange, market_type, symbol, timeframes[],
                              first_ts_ms, last_ts_ms, bar_count)
GET /strategies             → (id, version, warmup_bars, window_safe, params)

GET /candles?exchange&market_type&symbol&timeframe&end_ts_ms&limit
GET /signals?run_id&strategy_id&symbol&timeframe&side&mode&start_ts_ms&end_ts_ms&cursor&limit
GET /signals/{id}

GET /runs?mode&strategy_id&symbol&cursor&limit
GET /runs/{run_id}                → config + metrics + warmup_until_ts_ms
GET /runs/{run_id}/equity?cursor&limit
GET /runs/{run_id}/trades?cursor&limit

GET /health                 → per-feed status, gaps, clock drift, quality findings
GET /positions
GET /risk                   → limits, headroom, kill-switch state

GET /stream?topics=bar,signal,health     (Server-Sent Events)
```

`/candles` paginates **backwards** (`end_ts_ms` + `limit`, returns ascending) because a
chart's history request is always "give me N more bars before where I am". Default
`limit` 1000, max 5000. This directly serves the 50k-bar scroll-back target.

```jsonc
// GET /candles?...&timeframe=15m&limit=2
{
  "instrument": {"exchange":"binance","market_type":"perp","symbol":"BTC/USDT"},
  "timeframe": "15m",
  "bars": [
    {"ts_open_ms":1785723300000,"ts_close_ms":1785724199999,
     "o":"63205.31","h":"63286.00","l":"63100.00","c":"63128.00",
     "v":"96.3039","is_closed":true}
  ],
  "page": {"next_end_ts_ms": 1785723299999, "has_more": true}
}

// GET /signals?...
{
  "data": [{
    "id": 8412, "run_id": "…", "mode": "live",
    "strategy_id": "turnaround_v2", "strategy_version": "1.0.0",
    "symbol": "BTC/USDT", "timeframe": "15m",
    "ts_bar_ms": 1785723300000, "ts_emit_ms": 1785724200140,
    "bar_is_closed": true, "side": "enter_long", "strength": null,
    "entry_price": "63128.00", "stop_loss": "62740.10", "take_profit": null,
    "reason": "2 red then green, close>EMA200, close<EMA20*0.99",
    "features": {"ema200":"62110.4","ema20":"63590.2"}
  }],
  "page": {"next_cursor": "eyJ0cyI6MTc4NTcyMzMwMDAwMH0", "has_more": true}
}
```

**SSE rather than WebSocket** for `/stream`: the dashboard is read-only, so the channel is
unidirectional; browser `EventSource` reconnects automatically with no client code; and it
avoids a second framing/heartbeat implementation. If bidirectional control were ever
needed, it is explicitly out of scope (§10).

---

## 7. Backtest realism

The simulator consumes the same `BarEvent` stream and the same `ExecutionClient` as paper.

Configurable cost model: maker/taker fee (bps, separately), slippage (fixed bps and/or
ATR-proportional), funding for perps (applied at each funding timestamp from stored funding
history, using **each symbol's own interval** — it is not universally 8h on either venue,
see §13), order latency, and a participation cap for partial fills. Defaults recorded in
`runs.config` for reproducibility.

Reported: total & annualized return, max drawdown, Sharpe, Sortino, win rate, profit
factor, avg win/loss, expectancy, trade count, exposure time, and — because the first four
are meaningless without it — bars in sample and warmup bars excluded.

Overfitting: `backtest/walkforward.py` provides anchored and rolling walk-forward splits
and reports in-sample versus out-of-sample metrics side by side. See Pushback P6 on why I
am not proposing an automatic "overfitting score".

---

## 8. Risk, execution safety, secrets, operability

**Risk** — position sizing, max position per instrument, max gross exposure, max daily
loss (UTC day, recomputed from persisted fills on startup so a restart cannot reset it),
max drawdown kill-switch, and a `flatten-and-halt` path. **Kill-switch state is persisted**;
if it trips and the process restarts, it stays tripped until an explicit
`strategy-lab risk reset` — an in-memory-only kill switch that clears on restart is the
classic way this control silently fails.

**Modes** — `backtest` | `replay` | `paper` | `live`. Default `paper`. `live` requires
**two independent switches**: `--mode live` *and* env `STRATEGY_LAB_ALLOW_LIVE=1`, plus an
interactive typed confirmation of the instrument. One switch means a stale shell env or a
copy-pasted command can arm it by itself.

**Secrets** — `.env` only (already gitignored), `.env.example` committed. Two mechanical
guards: a `RedactingFilter` installed on the root logger that masks any value loaded into a
secret settings field, and `tests/test_no_secrets.py` scanning tracked files for key-shaped
patterns. README documents read-only keys, trading enabled only when going live, no
withdrawal permission, IP allowlist. **The entire Phase 1–2 data layer needs no keys at
all** — verified, §13.

**Operability** — structured JSON logs carrying `run_id` / `event_id` / `signal_id` /
`client_order_id` so one grep follows an event to its order. Health metrics per §6.
Telegram alerts at two severities (warn: gap detected, clock drift, reconnect storm;
critical: feed down, risk limit breached, kill-switch fired). Graceful shutdown drains the
event queue, refuses to persist a forming bar as closed, and flushes state.
Clock sync checks local time against exchange server time at startup and every 5 minutes,
warning past 500 ms and alerting past 2 s.

---

## 9. Two backtest engines, temporarily

`backtests/engine.py` (vectorbt) cannot do what §7 requires — `Portfolio.from_signals`
has no path to model funding, latency, or partial fills, and it cannot share a code path
with a websocket. The new event-driven simulator is therefore mandatory, not a preference.

But STRATEGIES.md documents live ETF research running against the vectorbt engine, and
deleting it would invalidate that work mid-flight. Recommendation: **freeze
`engine_vbt.py`, build the new simulator alongside it, migrate crypto first**, and retire
vectorbt once the ETF strategies have been re-validated against the new engine and the
numbers reconciled. Two engines is a smell; two engines with a written expiry is a
migration. (Worth noting: vectorbt's open-source line is effectively in maintenance, with
development having moved to the paid vectorbtpro — so this is a dependency worth exiting
regardless.)

---

## 10. Phase 3 dashboard — what is already de-risked

`backtests/report.py` is **already written against Lightweight Charts v5**, not v4:

```js
priceChart.addSeries(LWC.CandlestickSeries, {...});      // v5 generic factory
priceChart.addSeries(LWC.HistogramSeries, {...}, 1);     // 3rd arg = pane index
priceChart.panes()[1].setHeight(88);                     // v5 pane API
LWC.createSeriesMarkers(candleSeries, P.markers);        // v5 markers primitive
candleSeries.update(candle);                             // incremental tick path
```

So the v4→v5 hazard you flagged is already navigated once in this repo, with a working
reference. Phase 3 harvests these patterns into React components rather than rediscovering
them.

Verified against the npm registry and the published 5.2.0 tarball's own `typings.d.ts`
(stronger evidence than the docs, which contain two stale v4 examples):

- **5.2.0 is the newest published version** — nothing newer exists. And the vendored file
  is **byte-identical to the official artifact** (SHA-256 match against the npm tarball).
  No upgrade action, no supply-chain question.
- **`UTCTimestamp` is SECONDS, not milliseconds.** Our API is canonically ms (§6), so the
  UI divides by 1000 at exactly one adapter boundary. The library's branded `Nominal` type
  makes a raw `number` fail typecheck, which turns this classic bug into a compile error.
- **`series.update()` throws on an out-of-order timestamp** — it does not silently no-op.
  The live tick path needs an explicit guard, since a websocket reconnect can legitimately
  replay an older bar.
- **`series.setMarkers()` no longer exists.** Markers go through
  `createSeriesMarkers(series, markers)`, and you must retain the returned handle — it is
  the only way to update or clear them later.
- **Price-anchored markers are new in v5** (`atPriceTop` / `atPriceMiddle` /
  `atPriceBottom`, which make `price` required). This is a better fit for SL/TP than price
  lines for point-in-time events.
- **Prepending history requires a full `setData()`** — there is no prepend primitive, and
  `update()` explicitly cannot go backwards. The 50k-bar scroll-back target is served by
  `barsInLogicalRange().barsBefore < N` → fetch → concat → `setData`.
- **ESM-only, ES2020.** No CommonJS build ships. TypeScript must use
  `moduleResolution: "bundler" | "node16" | "nodenext"`; the legacy `node` resolver ignores
  the `exports` map and will not resolve the package at all.
- Watermarks are `createTextWatermark(pane, opts)` — the v4 `watermark` chart option is
  silently dead in JS and a type error in TS.

The forming-versus-closed distinction will be rendered as a distinct candle style plus an
explicit "forming" badge, and signals on forming bars (if ever enabled) get a visually
distinct marker shape.

Constraints affirmed: read-only, no control endpoints, `127.0.0.1` hardcoded, UI talks
only to the §6 API and never to Postgres or strategy code.

---

## 11. Decisions — RESOLVED 2026-08-02

| # | Decision | Resolution | Reasoning |
|---|---|---|---|
| **D1** | Spot or perps? | ✅ **USDⓈ-M perpetuals** | Requirement 6 asks for funding-rate modeling, which only exists on perps. Shorts work natively — 3 of 4 existing strategies emit shorts, which spot cannot express. |
| **D2** | Which exchange? | ✅ **Binance only for now**; OKX deferred | Narrower than my recommendation. Accepted — see the conformance mitigation below, which is now mandatory rather than optional. |
| **D3** | Storage engine | ✅ **Postgres** (already running) | §5. |
| **D4** | `market_candles` → NUMERIC? | ✅ **Migrate in place**; Decimal at boundaries, float64 inside indicators | Requirement 3 vs. existing `Float` columns. ~100k rows, seconds to rewrite. |
| **D5** | Bar granularity | ✅ **Subscribe natively at the strategy's timeframe** | Local aggregation from 1m adds a component whose bugs are indistinguishable from strategy bugs, and Binance already marks closed bars per interval. |
| **D6** | Async model | ✅ **asyncio, single process, single event loop** | Everything is I/O-bound; the strategy call is 4 ms. `run_in_executor` is the escape hatch. |
| **D7** | Target OS | ✅ **macOS, OS-agnostic code** | No OS-specific APIs, no `uvloop`, `pathlib` everywhere, no assumptions about the event-loop policy. |
| **D8** | vectorbt | ✅ **Freeze, migrate, retire** (§9) | |
| **D9** | Exchange client library | ✅ **Direct** (`aiohttp` + `websockets`) | §13: ccxt drops the `confirm` closed-bar flag, making "closed bars only" unimplementable on it. ccxt stays for the existing equity fetchers. |
| **D10** | Spec location | ✅ `docs/design/` | |

### Consequence of D2: the conformance suite is now load-bearing

My argument for building OKX early was that *an abstraction validated by one implementation
is not validated*. Deferring OKX doesn't remove that risk, so it has to be paid down another
way. Two mitigations, both cheap now and expensive to retrofit:

1. **`MarketDataFeed` gets a shared conformance test suite** that every implementation must
   pass. Even without OKX there are **two** implementations from day one — `BinanceFeed` and
   `ReplayFeed` — so the contract is exercised by more than one thing. `ReplayFeed` is not a
   test double; it is the backtest/replay data path, which makes it a real second citizen.
2. **The OKX findings in §13 become interface constraints**, encoded now even though no OKX
   code is written. Specifically, `MarketDataFeed` must not assume any of the following,
   because Binance and OKX disagree on every one of them:
   - ascending result ordering (OKX returns newest-first)
   - forward pagination (OKX walks backward via `after`)
   - one endpoint covering all history (OKX splits recent vs. history at 1,440 bars)
   - the venue aligning daily+ bars to UTC (OKX defaults to UTC+8)
   - a uniform volume column (OKX `vol` is contracts for swaps, base for spot)
   - the closed-bar flag living in a comparable place

   Backfill is therefore specified as *"page until the requested range is covered"*, with
   direction and ordering owned by the adapter — not as *"increment `startTime` by
   `limit × bar_ms`"*, which is the Binance-shaped assumption that would need tearing out
   later.

The point of doing this now: it costs nothing today, and it means adding OKX later is
additive rather than a redesign.

---

## 12. What I think is wrong or premature in the brief

**P1 — "Replay produces the same signals as live" is only fully true for closed-bar
strategies.** Live sees a bar tick many times before closing; replay from stored OHLCV sees
only the final values. So the forming-bar opt-in in requirement 2 and the replay-determinism
guarantee in requirement 1 are in direct tension: **a strategy that acts on forming bars
cannot be replayed deterministically from candle storage.** Options: (a) keep v1 strictly
closed-bar and defer the opt-in — my recommendation, it costs nothing today; or (b) record
raw websocket frames to a log and replay *those*, which is a real feature with real storage
cost. Choosing (a) silently and discovering this later is the bad outcome.

**P2 — Partial fills are not modelable from OHLCV.** You correctly put L2/order-book out of
scope, but fill ratio is a function of book depth. Any partial-fill model built on bar data
is an invented parameter that will make backtests look precise while being arbitrary. I
suggest implementing only a **participation cap** ("never fill more than X% of the bar's
volume"), labelling it a heuristic, and deferring anything more.

**P3 — Order latency is unobservable at bar resolution.** At 15m bars, 200 ms of latency
changes nothing. The honest bar-resolution model is "signal at bar close → fill at next bar
open + slippage", which I propose as the default. Keep the latency parameter in the config
for when finer data exists, but don't tune against it — it's currently measuring nothing.

**P4 — Two exchanges *and* full ops *and* a risk layer in Phase 1 is a lot before a single
signal has been proven correct.** See §14 for a resequenced version that gets to a
verifiable end-to-end slice sooner.

**P5 — Required `strength`/`confidence` on signals invites fabricated numbers.** Every
current strategy is a boolean rule with no meaningful confidence. Making the column
non-nullable would mean writing `1.0` everywhere, which is worse than null because it looks
like data. Proposed: nullable, populated only when a strategy genuinely computes it.

**P6 — "Flag overfitting risk" is not mechanizable as a score.** There is no honest
single number. What is real: walk-forward in/out-of-sample comparison, parameter-sensitivity
sweeps (is the peak a plateau or a spike?), and trade count next to every ratio. I propose
building those three and *not* shipping an "overfitting: 0.73" number that would invite
false confidence.

**P7 — Sharpe on weekly ETF strategies with ~30 trades is mostly noise.** Not wrong to
report, but I want trade count and a confidence caveat rendered next to it in the UI so it
isn't read as precision.

**P8 — Telegram remote control (mentioned as a future path) deserves an explicit "not
now".** The moment the bot can act, the bot's auth becomes a trading-critical control. v1
alerting should be strictly outbound.

**P9 — Your Phase 3 gate is good and I want to make it concrete.** "Schemas survive one
full strategy iteration unchanged" is the right instinct; I propose the testable version:
*the `signals` and `trades` schemas take zero changes across adding two new strategies and
one full walk-forward cycle.* Otherwise "one iteration" is arguable after the fact.

---

## 13. Appendix — exchange facts, verified live today

Every line below was executed against the live public APIs from this machine, with **no API
key**, rather than recalled. This matters because several of these contradict the obvious
assumption.

**Public market data requires no credentials.** Confirmed on both venues, REST and WS.

| | Binance | OKX |
|---|---|---|
| REST base | `api.binance.com` (spot) / `fapi.binance.com` (perp) | **`openapi.okx.com`** (docs no longer name `www`); regional hosts `us.okx.com`, `eea.okx.com` are mandatory by account region |
| Klines path | `/api/v3/klines`, `/fapi/v1/klines` | `/api/v5/market/candles` |
| Deep history | same endpoint + `startTime` | **separate** `/api/v5/market/history-candles` — the recent endpoint only reaches back **1,440 bars** |
| Max `limit` | 1000 spot / **1500** perp | **300** both (301 silently clamps) |
| Sort order | **ascending** (oldest first) | **descending** (newest first) |
| Backfill direction | `startTime` walks **forward** | `after` walks **backward** |
| Values | JSON **strings** | JSON **strings** (incl. timestamps) |
| Closed-bar flag | `k.x` boolean (WS) | `confirm` field, index 8: `"0"` forming / `"1"` closed |
| WS candles endpoint | spot `wss://stream.binance.com:9443/stream`; **perp `wss://fstream.binance.com/market/...`** | `wss://ws.okx.com:8443/ws/v5/`**`business`** |
| WS heartbeat | server pings; client pongs (20 s / 60 s spot) | client sends literal text `"ping"`, expects `"pong"`; **30 s** idle disconnect |
| REST rate limit | weight budget, `x-mbx-used-weight-1m` header | 40 req/2 s (candles), 20 req/2 s (history) **per IP**; 429 + body code `50011` |
| Server time | `/api/v3/time` | `/api/v5/public/time` |
| Funding interval | **not fixed — per symbol** | **not fixed** — derive from `nextFundingTime - fundingTime` |

### ⚠️ The finding that matters most: Binance futures WebSocket URLs were re-routed

`wss://fstream.binance.com` is now split into three entry points — `/public` (book/depth),
**`/market` (klines)**, and `/private` (user data). Per Binance's own change notice, the
legacy unrouted URLs *"will remain available until 2026-04-23, after which they will be
permanently decommissioned."* **That date has passed.**

The failure mode is the dangerous kind: the classic
`wss://fstream.binance.com/ws/btcusdt@kline_1m` — which is what essentially every tutorial,
older SDK, and from-memory reconstruction produces — **still connects successfully and then
silently never delivers a kline.** No error, no rejection, just a socket that stays open and
quiet. Correct form:

```
wss://fstream.binance.com/market/ws/btcusdt@kline_1m
wss://fstream.binance.com/market/stream?streams=btcusdt@kline_1m/ethusdt@kline_1m
```

This is precisely the class of error your brief warned about, and it lands directly on D1
(perps). Any library we adopt must be verified to emit the `/market` path. It also argues
for a startup assertion: if no kline arrives within ~2× the expected cadence of a live
subscription, fail loudly rather than sit there looking healthy.

### Funding is not on a fixed schedule

I initially recorded "8h both venues" from observing BTCUSDT. That generalization is wrong.
Binance exposes `GET /fapi/v1/fundingInfo` (weight 0), which today returns 743 symbols with
adjusted settings — including `fundingIntervalHours: 4` for some (e.g. LPTUSDT) alongside
the common 8. **Funding cost must be computed from each symbol's own interval**, derived
from `fundingInfo` / observed `fundingTime` deltas, never hardcoded to 8h. Symbols absent
from `fundingInfo` have no documented default, so derive rather than assume.

### Rate limits are higher and shaped differently than commonly cited

Read live from `exchangeInfo` today, not from prose docs (which tell you to do exactly that):

| | Spot | USDⓈ-M Futures |
|---|---|---|
| `REQUEST_WEIGHT` | **6000 / min** | **2400 / min** |
| klines weight | **flat 2**, independent of `limit` | **scales**: 1 / 2 / 5 / 10 by limit tier |
| klines max `limit` | 1000 | **1500** |
| WS ping cadence | server pings every 20 s, pong due within 60 s | ping every 3 min, pong within 10 min |
| WS msg rate | 5/s | 10/s |

The widely-repeated "1200 weight/min" figure is stale. Both venues cap a connection at
**24 hours** and 1024 streams; plan for a scheduled reconnect rather than treating the
daily disconnect as an incident. `RateLimitBudget` should therefore **fetch limits from
`exchangeInfo` at startup** instead of hardcoding them, since the numbers move.

Further findings that would have produced plausible-but-broken code:

1. **OKX `after` means *older*.** Verified: `after=1785722400000` returned
   `[…1500000, …0600000, …9700000]`, all strictly older. Backfill pages with `after`.
2. **`before` used alone does NOT mean "the bars just after this one" — it returns the
   *latest* data.** My first probe was ambiguous about this (I used a recent pivot, so
   "newest bars" and "bars after the pivot" happened to coincide). A second probe with an
   older pivot disambiguated: `before=2026-04-24` returned 2026-08-02, 08-01, 07-31 — the
   newest bars in the book, not 04-25 onward. The docs confirm: *"The latest data will be
   returned when using `before` individually."* Walking forward requires
   **`after` + `before` together**, which yields a bounded window with both bounds
   exclusive. Using `before` alone to resume a forward scan would silently skip history.
3. **`/market/candles` only reaches back 1,440 bars.** It is a rolling window, not a
   shallow page — paging it with `after` bottoms out at exactly 1,440 rows and then returns
   empty. Deep backfill *must* use `/market/history-candles`. Max `limit` on both is
   **300** (not 100), and `limit=301` silently clamps rather than erroring.
4. **OKX candle channels are NOT on `/ws/v5/public`.** Subscribing there returns
   *"Wrong URL or channel … doesn't exist"* (code `60018`); OKX's own error `64001` says
   the channel *"has been migrated to the '/business' URL."*
5. **The two venues page in opposite directions with opposite sort orders**, so
   "resumable paginated backfill" is genuinely different code per venue. This is the single
   strongest argument for D2 (build both early) — a one-exchange abstraction would have
   baked in Binance's direction as if it were universal.

### OKX defaults to UTC+8 for daily and above

For `bar` ≥ 6H, OKX's *default* alignment is Hong Kong time, not UTC. Verified:

| `bar` | opening time | `bar` | opening time |
|---|---|---|---|
| `1D` | 2026-08-02 **16:00Z** | `1Dutc` | 2026-08-03 **00:00Z** ✅ |
| `1W` | Sun **16:00Z** | `1Wutc` | Mon **00:00Z** ✅ |
| `6H` | 22:00 / 16:00 / 10:00Z | `6Hutc` | 00:00 / 18:00 / 12:00Z ✅ |

**Rule: always append `utc` for any bar ≥ 6H.** Below 6H no suffix exists and intervals
divide the UTC day evenly. Note `1W` → `1Wutc` shifts the *week boundary* (Sunday 16:00Z
vs Monday 00:00Z), not merely an offset — a weekly crypto series built without the suffix
is not comparable to the weekly ETF series already in this repo.

### Volume columns are not comparable across market types

OKX `vol` is **contracts** for swaps but **base currency** for spot — verified live as a
100× difference between `BTC-USDT` and `BTC-USDT-SWAP`. Only **`volCcyQuote`** (quote
currency) is consistently defined across both. The normalizer must map `volCcyQuote` to a
single meaning, or backtests silently compare incomparable volumes across market types.

### ⚠️ This disqualifies ccxt for the live path

The agent read the installed ccxt 4.5.54 source in this repo's own venv and found two
disqualifying behaviors:

1. **ccxt REST and ccxt WebSocket disagree on UTC alignment for the same symbol.**
   `ccxt/okx.py` auto-appends the `utc` suffix for timeframes ≥ 6h; `ccxt/pro/okx.py`
   does not (`name = 'candle' + interval`). Verified live: `fetch_ohlcv('BTC/USDT','1d')`
   returns `2026-08-01 00:00Z` (UTC) while `watch_ohlcv('BTC/USDT','1d')` returns
   `2026-08-02 16:00Z` (UTC+8). **A system backfilling with ccxt REST and streaming with
   ccxt.pro gets misaligned daily bars, silently.** That is exactly the backtest/live
   divergence this whole design exists to prevent.
2. **ccxt normalizes candles to 6 columns and drops `confirm`.** With no closed-bar flag,
   "signals fire on closed bars only" cannot be implemented on top of ccxt at all.

So: **ccxt stays for the existing equity/legacy fetchers, but the new feed layer talks to
the exchanges directly** (`aiohttp` + `websockets`, both already installed). This costs a
few hundred lines of per-venue parsing and buys exact control over alignment, the
closed-bar flag, `volCcyQuote`, rate-limit headers, and the `/market` routing fix. Given
that requirement 3 already demands a custom normalization layer, most of that code is
required regardless — ccxt would only be hiding the fields we specifically need.

Also verified: Binance spot klines return 12 array fields with close-time at index 6;
funding history exposes a `rateType` field (`"Regular"`); `premiumIndex` gives
`lastFundingRate` + `nextFundingTime`; OKX candles return 9 fields
`[ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm]`; OKX supports UTC-aligned daily bars via the
`1Dutc` suffix, which matters for a UTC-only system. Binance kline streams push every
2000 ms (1000 ms for the `1s` interval); futures kline streams push every 250 ms.

### SDK landscape — the obvious choice is deprecated

| Package | Version | Status |
|---|---|---|
| `binance-connector` | 3.13.0 | **DEPRECATED** — its own PyPI page says so |
| `binance-sdk-derivatives-trading-usds-futures` | 16.0.0 (2026-07-30) | current official, for perps |
| `binance-sdk-spot` | 11.0.0 (2026-07-30) | current official, for spot |
| `python-binance` | 1.0.37 | maintained, third-party |
| `python-okx` | 0.4.3 (2026-07-07) | maintained, OKX-published, asyncio WS |
| `ccxt` | 4.5.70 (repo has 4.5.54) | maintained; **`ccxt.pro` is merged in** |
| `ccxtpro` | 1.0.1 (2020) | dead stub — installing it is a mistake |

Binance replaced the monolithic connector with modular per-product SDKs. Both official
SDKs are asyncio-native with a `websocket_streams` module.

On ccxt: **`watch_ohlcv` lives in the base `ccxt` package** via the `ccxt.pro` namespace —
there is no separate install. One trap worth recording: `import ccxt.pro` works, but
`hasattr(ccxt, "pro")` after a bare `import ccxt` returns **False**, because `ccxt.pro` is
a lazy submodule. A capability check written the obvious way gives a false negative.

Local environment: `websockets` 16.0, `aiohttp` 3.13.5, `ccxt` 4.5.54 already installed.
Postgres 16 up and healthy on `:5432`.

---

## 14. Proposed sequencing

Reordered per P4 so that a correct end-to-end slice exists before breadth is added.

**Phase 1a — the spine.** `core/` types + clock, `feeds/base.py`, `ReplayFeed` over
existing Postgres data, `StrategyRunner`, `SimulatedExecutionClient`, signals persisted.
Exit test: **`turnaround_v2` replays 83k stored bars and produces signals byte-identical to
the vectorized path** — requirement 1 proven against real data before any network code.

**Phase 1b — the network.** Binance REST backfill (resumable, rate-limit budgeted) +
websocket live feed, gap detection and repair, reconnect with backoff, clock sync.
Exit test: reconnect/gap test kills the socket mid-stream and asserts the gap is detected
and REST-backfilled to a hole-free series.

**Phase 1c — risk + modes.** Risk layer, mode gating, Telegram alerts. (OKX deferred per
D2; the `MarketDataFeed` conformance suite lands in 1b instead, covering `BinanceFeed` and
`ReplayFeed`.)

**Phase 2 — validation.** Event-driven backtest simulator with the §7 cost model, metrics,
walk-forward, plus the full test suite: lookahead poison probe, replay determinism,
indicator unit tests, risk unit tests, reconnect/gap test, secret-scan test.

**Phase 3 — dashboard.** Gated on P9. React + Vite + Tailwind + `lightweight-charts@^5`,
consuming only the §6 API.

Throughout: README covering setup, key configuration, adding a strategy, adding an exchange.

---

## 15. Open questions

None of these block Phase 1a, which needs no network access and no new data — it replays
the 83,348 BTC/USDT 15m bars already in Postgres. They shape Phase 1b:

1. Which symbols and timeframes go live first? Assumed `BTC/USDT` 15m perp, matching the
   grain of your largest existing dataset.
2. How far back should the perp backfill go? Binance USDⓈ-M data starts ~2019-09.
3. The existing 83k 15m dataset is **spot**, and D1 selects **perp** — different
   instruments with different price series. Assumed: keep the spot set as-is (it stays
   useful as replay fodder and as the `1w`/`1d` research base) and backfill perp alongside
   it under its own identity. They coexist without collision because `market_type` is part
   of the candle identity.
