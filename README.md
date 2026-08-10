# Strategy Lab

Local Python research repo for crypto and stock strategy backtesting.

The current stack is intentionally small:

- Postgres 16 in Docker for reproducible OHLCV storage, plus an append-only
  `runs`/`signals` store for replay and live signal history
- `ccxt` for crypto candles
- `yfinance` for stock candles
- `pandas` and `numpy` for strategy research
- `vectorbt` for fast signal-based backtests and plots
- a small event-driven engine (`core/`, `feeds/`, `engine/`) so the same
  strategy code runs in backtest, in replay, and against the live venue —
  see [Paper Trading Against The Live Venue](#paper-trading-against-the-live-venue)

## What do you want to do?

Every section below is named after a piece of the machinery. This table is named
after the question you arrived with.

| I want to… | run | where it is explained |
|---|---|---|
| **just watch what state the market is in** | `strategy-lab browse` → view `state` | [The state view](#the-state-view-regime-only) |
| **watch live prices, signals and state** | `strategy-lab browse` | [The live pill](#the-live-pill-a-forming-candle-and-a-refresh-when-it-closes) |
| see any strategy on any stored candle set | `strategy-lab browse` | [browse — the live view](#two-ways-to-look-at-a-strategy) |
| run a strategy against the venue, on paper | `strategy-lab paper` | [Paper Trading](#paper-trading-against-the-live-venue) |
| backtest and keep a reproducible record | `strategy-lab backtest` | [Backtest](#backtest) |
| re-read a backtest I already ran | `strategy-lab serve` | [serve — the frozen record](#two-ways-to-look-at-a-strategy) |
| prove the live path matches the backtest | `strategy-lab replay` | [Replay](#replay) |
| get data in before any of the above | `strategy-lab fetch-*` | [Fetch Data](#fetch-data) |

## Layout

```text
strategy-lab/
  data/
    raw/
    processed/
  reports/
  src/strategy_lab/
    market_data/     # vehicle/source-specific fetchers
    db/              # normalized candle schema and read/write helpers
    strategies/      # strategy modules
    backtests/       # vectorbt runner and report writer
    feeds/           # MarketDataFeed protocol and the Postgres replay feed
    engine/          # one runner per contract, bar buffers, market clock
    features/        # cross-sectional reads over a market snapshot
    api/             # the browser's read-only endpoints, incl. the board
    browser/         # the browser's one page: board + instrument view
    cli.py
```

## Setup

```bash
cd /Users/jingbofu/Desktop/repo/strategy-lab
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
docker compose up -d postgres
strategy-lab init-db
```

The default database URL is:

```text
postgresql+psycopg://trader:trader@localhost:5432/strategy_lab
```

Override it with `DATABASE_URL` if needed.

## Fetch Data

Crypto spot:

```bash
strategy-lab fetch-crypto \
  --exchange binance \
  --market-type spot \
  --symbol BTC/USDT \
  --timeframe 15m \
  --since 2024-01-01
```

Crypto perps:

```bash
strategy-lab fetch-crypto \
  --exchange binance \
  --market-type perp \
  --symbol BTC/USDT \
  --timeframe 1h \
  --since 2024-01-01
```

Stocks:

```bash
strategy-lab fetch-stock \
  --symbol AAPL \
  --timeframe 1h \
  --period 2y
```

Weekly candles for the whole configured ETF universe (see `src/strategy_lab/universe/etfs.py`):

```bash
strategy-lab fetch-etf-universe \
  --timeframe 1w \
  --start 2020-01-01
```

## Perp And Funding Data

Binance USD-M perpetuals, fetched over the venue's REST API rather than ccxt
(which does not expose funding or open interest cleanly). Run `strategy-lab
migrate` once first — `funding_rates` and `open_interest` are created by the
migrations, not by `init-db`.

Perp candles land in `market_candles` under `market_type=perp`, through the same
`normalize_candle_frame` + `upsert_candles` path as every other candle source:

```bash
strategy-lab fetch-perp --symbol BTC/USDT --timeframe 4h --since 2019-09-01
strategy-lab fetch-perp --symbol ETH/USDT --timeframe 4h --since 2019-11-01
```

Funding is a settlement cash flow, not a candle field, so it gets its own table:

```bash
strategy-lab fetch-funding --symbol BTC/USDT --since 2019-09-01
strategy-lab fetch-funding --symbol ETH/USDT --since 2019-11-01
```

### Open interest is only ~30 days deep and cannot be backfilled

**Binance serves roughly 30 days of open-interest history and no more.**
Measured 2026-08-03: a `startTime` 40 days back returns
`{"code":-1130,"msg":"parameter 'startTime' is invalid."}`. `fetch-open-interest`
therefore takes no `--since` and refuses an out-of-window request rather than
clamping it, because a silently narrowed range would make a 30-day sample look
like history. OI accumulates forward from the first run — schedule it if you
want a series:

```bash
strategy-lab fetch-open-interest --symbol BTC/USDT --period 4h
```

Any study needing years of open interest is not answerable from this source.

### What the venue actually returns

Three things measured while backfilling to 2019, each of which will bite someone
who assumes otherwise:

- **`markPrice` is an empty string on older funding records.** Binance only
  began populating it on 2023-10-31; every earlier row has `markPrice: ""`,
  stored as NULL. `Decimal("")` raises, so this aborted the first backfill run.
- **Funding times are not exactly on the 8h grid.** They land on the boundary or
  up to 47 ms after it (3,260 of BTC's 7,559 settlements). Match funding to bars
  by flooring or reindexing, never by equality against a generated 8h range.
- **The settlement interval is per-contract**, so nothing in this repo hardcodes
  8h. BTC/USDT and ETH/USDT have used 8h continuously since inception, but that
  is an observed property of those two contracts rather than a rule.

## Backtest

```bash
strategy-lab backtest \
  --exchange binance \
  --market-type spot \
  --symbols BTC/USDT,ETH/USDT,SOL/USDT \
  --timeframe 15m \
  --strategy turnaround_v2
```

For spot-style long-only testing:

```bash
strategy-lab backtest \
  --exchange binance \
  --market-type spot \
  --symbols BTC/USDT \
  --timeframe 1d \
  --strategy turnaround_v2 \
  --no-allow-shorts
```

By default, backtests exit on either the opposite strategy signal or continuation failure:

- long continuation failure: 4 consecutive lower closes
- short continuation failure: 4 consecutive higher closes

Tune that threshold with `--failure-bars`:

```bash
strategy-lab backtest \
  --exchange binance \
  --market-type spot \
  --symbols BTC/USDT \
  --timeframe 15m \
  --strategy turnaround_v2 \
  --failure-bars 4
```

To compare against EMA-based trend failure:

- long trend failure: close falls below the EMA trend filter
- short trend failure: close rises above the EMA trend filter

```bash
strategy-lab backtest \
  --exchange binance \
  --market-type spot \
  --symbols BTC/USDT \
  --timeframe 15m \
  --strategy turnaround_v2 \
  --exit-mode trend_failure
```

To compare against the raw setup invalidation stop:

- long stop: below the low of the three-candle reversal setup
- short stop: above the high of the three-candle reversal setup

```bash
strategy-lab backtest \
  --exchange binance \
  --market-type spot \
  --symbols BTC/USDT \
  --timeframe 15m \
  --strategy turnaround_v2 \
  --exit-mode setup_invalidation_stop
```

To compare against the original behavior:

```bash
strategy-lab backtest \
  --exchange binance \
  --market-type spot \
  --symbols BTC/USDT \
  --timeframe 15m \
  --strategy turnaround_v2 \
  --exit-mode opposite_signal_only
```

For weekly trend-following with long-only ETFs (recommended):

```bash
strategy-lab backtest \
  --exchange yahoo \
  --market-type equity \
  --symbols SPY \
  --timeframe 1w \
  --strategy trend_following_deepseek_v4 \
  --exit-mode trend_structure \
  --cash 100000
```

The `trend_structure` exit mode exits on either:

- long: close falls below the 40-week SMA (trend break)
- long: continuation failure (4 consecutive lower closes by default)

No short exits and no opposite signal exits — the engine rejects `trend_structure` runs
whose strategy emits short entries, so pair it with `--no-allow-shorts` or a long-only
strategy.

For the ATR-sized weekly trend rider (exits are built into the strategy, so use the
pass-through exit mode):

```bash
strategy-lab backtest \
  --exchange yahoo \
  --market-type equity \
  --symbols SPY,QQQ,SMH,XLF,XLK \
  --timeframe 1w \
  --strategy trend_rider_v1_deepseek_v4_pro \
  --exit-mode opposite_signal_only \
  --no-allow-shorts
```

Stock example:

```bash
strategy-lab backtest \
  --exchange yahoo \
  --market-type equity \
  --symbols AAPL,MSFT,NVDA \
  --timeframe 1h \
  --strategy turnaround_v2
```

Each run writes a snapshot under `reports/`:

- `config.json`
- `stats.json`
- `trades.csv`
- `equity_curve.csv`
- `plot.html` — self-contained TradingView-style dark report: candlesticks with
  volume, entry/exit price markers, a crosshair-synced equity pane, and a
  per-trade table (PnL, holding period, click a row to zoom to that trade)

That report directory is the reproducibility boundary for comparing strategy changes.

## Replay

`replay` drives stored candles bar-by-bar through the event-driven engine in
`src/strategy_lab/engine/`, calling the exact same `strategy.generate_signals(df)`
the vectorized `backtest` command calls — but once per closed bar over an
expanding buffer, reading only the last row, instead of once over the whole
range. It is the execution path live trading will use, and
`tests/test_replay_determinism.py` proves the two paths agree exactly, signal
for signal — on **funded** frames, so it can see a crowding difference rather
than comparing crowding-neutral against crowding-neutral.

**Funding is attached automatically on a perp**, through the same function the
backtest uses, so the alignment rule and the coverage guard are shared rather
than duplicated. Coverage is *required* only when the strategy actually reads a
funding-derived feature — otherwise BTC's permanent 40-hour leading gap would
make its own range unreplayable for `donchian`, which reads none. `--no-funding`
opts out; a crowding-reading strategy over an uncovered range is refused with the
same message `backtest` gives.

There is **one runner per strategy contract**, and each refuses the other's
strategies at construction rather than a warmup later: `StrategyRunner` drives
`SignalSet` and takes an optional `--exit-mode`-equivalent, `ExposureRunner`
drives `TargetExposure` and emits a signed level.

`replay` persists to the `runs`/`signals`/`bar_reasons` tables that only
`strategy-lab migrate` creates — `init-db` does not. `bar_reasons` is one row per bar
past warmup for any strategy that can explain itself, carrying the state and the
feature values behind it, which is what only this path can record: a backtest and the
browser recompute those on demand from immutable candles.

Run `migrate` once; on a brand-new database run `init-db` first, since `migrate`
widens `market_candles` to `NUMERIC` and needs the table to already exist. `migrate`
is idempotent and safe to re-run:

```bash
strategy-lab init-db      # fresh database only
strategy-lab migrate      # once — creates runs/signals, widens market_candles
```

```bash
strategy-lab replay \
  --exchange binance \
  --market-type spot \
  --symbol BTC/USDT \
  --timeframe 15m \
  --strategy turnaround_v1 \
  --limit-bars 2000
```

```text
Run 99fe4a0f-9086-4047-ab0e-75fc5d84027d: emitted 916 signals, wrote 916.
```

A run that persists something mints a fresh `run_id`, so replaying the same
range twice stores two independent runs rather than zero rows the second time —
that is the append-only audit trail working as intended, not a bug.
`--no-persist`, and a replay that emits neither a signal nor a reason, write no
run header at all rather than leaving an orphan behind. Some
strategy/window combinations legitimately emit nothing: `turnaround_v2`, the
CLI default, fires only 126 times across the *entire* 83,348-bar stored
BTC/USDT 15m history, so a quiet run by itself is not a sign anything is
broken.

Replay is O(n²) by construction: every bar re-evaluates the strategy over the
whole buffer seen so far, not just the new bar. On that same 83,348-bar
series, `turnaround_v2` produces the same 126 signals from both paths, but
`backtest` takes 0.39 s and `replay` takes roughly 43 minutes to do it
bar-by-bar. Use `--limit-bars` for anything beyond a few thousand bars, and
keep using the vectorized `backtest` command — not `replay` — for
whole-history research; `replay` exists to prove the live path matches
backtest, not to replace it for day-to-day iteration.

See [the Phase 1a design doc](docs/design/2026-08-02-realtime-trading-framework.md)
for the full rationale.

## Paper Trading Against The Live Venue

`replay` drives the event engine from stored candles. `paper` drives the same
runner from `LiveFeed`, which polls the venue, and hands what it decides to
`PaperBook`. Nothing reaches an exchange — the book is a ledger.

```bash
strategy-lab paper \
  --symbol BTC/USDT \
  --timeframe 15m \
  --strategy state_machine_v1 \
  --for-minutes 75 \
  --bars-csv live_bars.csv
```

```text
Funding advanced: 6 settlements stored.
Primed 2196 bars (state_machine_v1 wants 2192); buffer carries funding: True.
Warning: state_machine_v1 sizes per bar, and the event path carries no size onto
a Signal -- this book fills every entry at scale 1.0, so its trades will not
match a backtest of the same range.
Ran 75 min: 5 bars, 0 signals, 5 reasons, 0 closed trades. Withheld polls: 27,
funding top-up failures: 0, bars revised after the fact: 0.
```

That warning is not incidental to the example: `state_machine_v1` returns a
per-bar `position_size` and no `Signal` can carry one, so its paper book sizes
every entry at 1.0. The withheld polls are not incidental either — that run
crossed a settlement boundary, and §9.18 of the charter is where the 27 came
from.

It is bounded by the wall clock, because a live stream does not end and a bound
that waited for bars would hang exactly when the feed had stopped producing
them. Signals and `bar_reasons` are written per bar rather than at the end, so a
run measured in hours survives whatever ends it.

**A perp run advances its own funding**, and this is not optional: stored
settlements move only when a funding fetch runs, so a process that polled only
candles would watch its window grow past its coverage until every poll was
withheld. On the same subject, a perp poll deliberately fetches a wider window
than `--bars-csv` shows corrections for — the coverage guard needs three
settlements to certify a cadence, and five bars at 15m is 75 minutes holding
none.

**It writes no candles.** The point of a paper run is to be checkable against
what the venue serves for the same range *later*, and a process that stored its
own bars as the record would be compared against itself. Fetch the range
afterwards and replay it; `scripts/r10h/delayed_oracle.py` does the comparison.

## Multi-Asset

`ReplayFeed.stream()` merges every subscription into one globally time-ordered
stream, breaking ties on the full candle key so the order is identical on every
run. `MarketClock` groups that stream into `MarketSnapshot`s — the set of bars
sharing one close time — and `MultiAssetRunner` drives one strategy per
instrument off those snapshots, keeping a full-history `BarBuffer` for each.
Instruments listed under `context` are buffered and appear in snapshots without
being traded; an instrument in neither `strategies` nor `context` raises rather
than being dropped.

**Candles, not instruments, are what a snapshot holds.** `CandleId` is the pair
`(instrument, timeframe)`, and both the tie-break and `MarketSnapshot.bars` key
on it, because one symbol subscribed at 4h and 1d ties with *itself* at every day
boundary — an instrument-keyed snapshot resolves that by dropping a bar. Read a
snapshot with `snapshot[BTC.at("4h")]`. `MultiAssetRunner` is single-timeframe by
construction, so its buffers stay keyed by instrument (`runner.buffer(BTC)`) and
a bar at any other timeframe raises rather than being absorbed.

**A timestamp is complete only once an event with a later timestamp arrives.**
Never by looking ahead — that is the whole point. So a cross-sectional signal
for bar *t* is emitted when the first *t+1* event lands, the final timestamp
needs an explicit `flush()`, and a snapshot holds only the instruments that
actually have a bar at that time. Absent is not unchanged: crypto trades
around the clock, equities do not, and instruments list and delist.

```python
subs = [Subscription(InstrumentId("binance", "perp", s), "4h")
        for s in ("BTC/USDT", "ETH/USDT")]
feed, clock = ReplayFeed.from_database(subs), MarketClock()

def score(snapshot):
    if snapshot and len(snapshot) >= 2:      # 477 of these hold BTC alone
        breadth(snapshot)   # strategy_lab.features.cross_sectional

async for event in feed.stream(subs):
    score(clock.on_event(event))
score(clock.flush())    # the last timestamp has no successor to release it
```

`breadth` and `confirms` read one snapshot and nothing else. Both assume a
shared timeframe across the universe — a daily bar meets a 4h bar only at day
boundaries, so a mixed-timeframe universe produces mostly single-instrument
snapshots. `breadth` refuses a universe smaller than `min_instruments`
(default 2) rather than returning a well-formed number that is really just one
instrument's direction.

Measured over the stored BTC + ETH perp 4h series: **15,128 snapshots, of which
477 are partial** — that last figure is the stretch where BTC had listed and ETH
had not (2019-09-08 to 2019-11-27). Mean breadth over the **14,651 complete
cross-sections** is **0.512**; the partial ones are skipped rather than scored,
which is what the guard in the loop above is doing.

## State Features

Nine registered features score one dimension of market state each, all through
`StateFeature` — the same `name` / `version` / `warmup_bars` / `compute(df)`
shape as `Strategy`, so the lookahead poison probe covers them without a second
implementation. `list_features()` and `get_feature(name)` mirror the strategy
registry, and registration is manual in the same two places.

```bash
strategy-lab features --exchange binance --market-type perp --symbol BTC/USDT \
  --timeframe 4h --horizons 1,6,30 --start 2019-09-10T08:00:00
```

That writes a timestamped `reports/` directory holding `features.html` — one row
per feature, self-contained — and `diagnostics.json`, the full record. Each row
carries coverage, distribution, lag-1 autocorrelation, turnover, the Spearman
information coefficient at every horizon, and the strongest correlation with any
other feature.

**Every percentile and z-score is rolling, never full-sample.** A full-sample
rank at bar *t* moves when bars after *t* arrive: measured on a 200-bar ramp
poisoned downward from row 121, `series.rank(pct=True)` at row 120 goes
0.605 → 1.000 while the rolling form does not move at all. That is lookahead with
no `shift(-1)` anywhere to grep for, so `rolling_percentile` and `rolling_zscore`
in `features/base.py` are the only two places it is implemented, and
`tests/test_feature_lookahead.py` poisons the future of every registered feature
on every run.

**The IC is measured against the return over `[t+1, t+1+h]`,** anchored one bar
after the feature's own. `close[t+h] / close[t]` contains no *return* from bar
*t*, which is why it reads as safe, but it does contain bar *t*'s *price* as its
denominator — so any feature reading that print divides a high number into its
own target and is paid for it. Measured on a random walk plus an i.i.d. print
error, with a feature that predicts nothing: IC −0.53 anchored at `close[t]`,
−0.01 anchored at `close[t+1]`.

**Both half-sample ICs are reported, not just the full-sample number.** A feature
that works in one half and not the other is a regime, not a signal, and the
average hides exactly that.

`crowding` needs a `funding_rate` column and **raises without one** rather than
returning a neutral 0.5, which would claim nobody is crowded. The `features`
command attaches funding on a perp automatically; on any other market it skips
`crowding`, says so on stderr, and records the skip in `diagnostics.json` —
a silent skip is how a feature ships unexamined.

Measured on 15,118 BTC/USDT perp 4h bars, no feature reaches |IC| 0.07 at any
horizon, which is normal at this frequency. The interesting numbers are
conditional: `direction`'s IC@30b goes from +0.038 unconditionally to +0.131
inside `strength`'s top tercile. See
[§9.1 of the charter](docs/research/2026-08-03-market-dynamics-engine.md#91-r4-feature-diagnostics--btcusdt-perp-4h)
for the full table and the keep/cut calls.

## Two ways to look at a strategy

They are different things and the difference is the point.

**`serve` — the frozen record.** A backtest wrote a dated directory under
`reports/`; its `plot.html` re-renders byte-identically from the run that froze
it. That is the reproducibility boundary for comparing strategy changes, and
`serve` is how you read one:

```bash
strategy-lab serve --port 8750
```

Reports are static files, but the server adds a delayed live feed. Open one
through it (`http://127.0.0.1:8750/<report-dir>/plot.html`) and a "delayed" pill
appears in the header: the page polls `/api/candles` every 60 seconds, which
re-fetches the latest bars from the upstream source (Yahoo Finance or ccxt),
upserts them into Postgres, and streams them onto the chart — including the
current forming bar. On a perp it fetches the funding settlements over the same
window too, so the candle history cannot outrun the funding history into a
coverage refusal. Click the pill to refresh immediately. The same file opened
directly from disk stays fully static.

**`browse` — the live view.** Any registered strategy over any stored candle set,
recomputed per request and **persisted nowhere**:

```bash
strategy-lab browse --port 8760      # loopback only; a routable host is refused
```

### It opens on the board

The landing view is a grid: **one tile per (candle set, strategy)**, carrying the
latest fill, a sparkline, the bar it is as of — and, for a strategy that has a
state machine, its current state and the feature values behind it. Pick the strategies from the selector, filter by market type,
and click any tile to open the single-instrument chart behind it.

Rows arrive **as each finishes** rather than as one blob — `GET /api/board`
streams newline-delimited JSON — because a warm analysis costs ~300–500 ms per
instrument and sixteen of them serially is ~6 s. First tile lands in ~100 ms.

**A tile never derives its own answer.** It slices the same `build_analysis` call
the chart uses, so a tile cannot quietly disagree with the chart it links to.
Nothing is cached between requests: `POST /api/refresh` rewrites overlapping
recent candles by design, so any stamp cheap enough to cache on is a stamp that
cannot see that.

**Each market type states the staleness that applies to it, and not the other's:**

| | what can go stale | what the tile shows |
|---|---|---|
| perp | the **right edge**, up to one funding cadence — the frame is bounded by its own stored settlements | `as of` against the newest stored bar, orange when they differ |
| equity | the **whole history** — the Yahoo fetcher rescales every past bar on a dividend | `candles written`, orange past 31 days |

That equity caveat is measured, not hypothetical: **333 of 333 stored SPY weekly
bars moved** against a fresh fetch (median 0.257%), and `donchian` differed on 3
of them where two ratio-based strategies differed on none. The board opens on
`perp` so that caveat is something you choose to look at rather than inherit.

Refresh is **explicit** — per tile or all tiles — and never on a *timer*. A
background poll that talks to a venue on its own schedule is a different thing
from a page you refreshed, and only the second is honest about when it last did.

### The state view: regime only

The **`state`** rung of the view selector drops everything about trading and
keeps the regime: candles, volume, the state ribbon, and the five features
behind it — `direction`, `strength`, `stability`, `crowding`, `energy` — with the
same live pill and the same timeframe ladder. No arrows, no trades, no exit
mode, no cost model: not hidden, **absent**, because none of them were computed.

That last part is the reason it exists rather than being a checkbox on the
instrument view. Measured on BTC/USDT spot 4h, 18,842 bars:

| stage | cost |
|---|---|
| load candles | 181 ms |
| `generate_signals` | 25 ms |
| state + all five features | 45 ms |
| `Portfolio.from_signals` | **2,951 ms** |

The half you don't want is 92% of the time. `GET /api/state` skips it, and the
same request that took 3,202 ms comes back in ~540 ms. It is still a *slice* of
`build_analysis` rather than a cheaper route to the same answer — it calls the
same `prepare_frame` and the same feature/machine code, and
`tests/test_api_state.py` pins the two to agree bar for bar.

**It only offers the strategies that have a state** (`state_machine_v1`,
`state_machine_v2`), on the server's own `has_state` flag rather than a list of
names.

#### Not every timeframe can carry a state, and the ladder says which

A state needs **2,192 bars** before its first reading — 1,920 of that is
`direction`'s 20×span-96 EMA, plus 272 for the machine to converge. Warmup is
counted in *bars*, so the same number is 365 days at 4h and **42 years at 1w**.
A rung whose stored set is too short is struck through with the count, rather
than clicking into a refusal:

| | crypto (Binance) | equity (Yahoo) |
|---|---|---|
| **15m** | ✅ | ❌ Yahoo caps 15m at 60 days → 1,040 bars |
| **1h** | ✅ | ⚠️ capped at 730 days → ~1,280 readings |
| **4h** | ✅ | ❌ capped at 730 days → 993 bars |
| **1d** | ✅ | ✅ |
| **1w** | ❌ BTC has 438 weeks of the 2,192 needed | ❌ SPY has 1,750 |

The equity gaps are Yahoo's own limits, named in its errors, and no amount of
fetching moves them. The weekly gap is arithmetic: 2,192 weeks is longer than
most instruments have existed.

#### The ribbon starts at warmup, and that is not a cosmetic choice

The machine answers on **every** bar. Inside warmup its inputs are `NaN`, which
it reads as *failing* — and failing renders as `COMPRESSION`. Measured on
BTC/USDT spot 1d with `state_machine_v1`, 3,060 bars against a 2,192-bar warmup:
the machine reports `compression` on **2,114 of the 2,192 warmup bars**. Drawn
from bar zero, 72% of that chart said "chop" over exactly the range where the
machine knew nothing — the one reading a regime chart must never give. So the
ribbon begins at `warmup_bars`, the legend names the blank stretch
*before warmup*, and the provenance strip carries **State from** with the date
and the count.

### The live pill: a forming candle, and a refresh when it closes

**This is how you watch real-time prices with signals and state on one chart:**

```bash
strategy-lab browse --port 8760
```

Then at `http://127.0.0.1:8760` — pick a strategy in the selector, click any
tile's **open**, and the **live** pill is already on. Try
`BTC/USDT · 15m · binance/spot` with `state_machine_v1` to get a moving candle,
markers, the state panel and the state-change list together.

On a Binance dataset at a timeframe the venue streams, the instrument view
offers that **live** toggle, default on.
It opens the venue's kline websocket and draws the forming candle as it moves.

**The forming bar is drawn and never analysed.** A tick reaching
`build_analysis` would produce a state and a marker that flip when the bar
closes — the one reading the event path refuses everywhere else. So the candle
moves while the state panel, the markers and the transition list stay as of the
last *closed* bar, and hovering the live one says `live bar · not analysed until
it closes` rather than showing the previous bar's state as if it were this one's.

**A closing bar is what triggers a refresh, and that is not a timer.** The
stream sets `x: true` on the one update where the stored candles are provably
behind, so the page refreshes exactly then rather than guessing on a clock. The
rule it keeps is the one above — the page still says when it last spoke to the
venue — and `setInterval` remains banned.

**"live" means data, not a socket.** Measured against Binance from one network:
`fstream.binance.com`, which serves **every perp dataset here**, accepts the
connection and then sends nothing — on both the raw and combined stream forms,
at 1m and 4h — while `stream.binance.com` delivers 6 frames in 12 s on the same
pair. Reporting the pill off `onopen` would have left a green dot over a frozen
chart, so it turns green on the first frame and says `connected · no data` if
none arrives within 15 s. If your perp charts sit there, that is what you are
seeing, and spot streams fine.

Only Binance, and only because the stored candles are Binance: a second venue's
ticks drawn onto this venue's candles is two markets' prices in one line.

Signals are computed by the same whole-history `generate_signals(df)` call
`run_backtest` makes over the same stored candles — never read from the `signals`
table and never from the event path — so the browser cannot disagree with a
backtest. What it draws depends on the contract: a `SignalSet` strategy gets
candlesticks and arrows at every **fill** (not every signal — `from_signals`
ignores a repeated same-direction entry, so signals would mark bars no backtest
traded), and a `TargetExposure` strategy gets a baseline pane carrying the signed
−1…+1 target, which no arrow can express.

**The chart carries the lifecycle, not just the fills.** Under the candles is a
**state ribbon** — one band per bar, coloured by the state the machine was in,
so a regime is visible as a shape rather than one bar at a time. A legend under
the chart says what the arrows mean (**↑ bought** — opening a long *or* closing
a short; **↓ sold** — the reverse) and what each ribbon colour is. Click or focus
a ribbon colour and it explains what that state means and what risk the policy
holds there. Volume is an overlay in the bottom fifth of the price pane rather
than a pane of its own, which used to take half the chart.

**A timeframe ladder** sits directly above the price pane: `1h 4h 1d 1w`, plus
whatever else this instrument already has stored. A rung with data switches to
it; a dashed rung has nothing stored and **fetches that timeframe** over the
frame you are looking at, then switches. It never resamples on the client — a 1h
bar built from four 15m bars is not the venue's 1h bar, and a dataset is keyed on
its timeframe precisely so the two cannot be confused. There is deliberately no
month: `timeframe_to_millis("1M")` raises, because a month is not a fixed width
and warmup, funding windows and the poll cadence are all bar-width arithmetic.

**The state sequence is below the chart** — every change the machine made, newest
first, with how long the previous state held and the bar it changed on. Click a
row to pin that bar and see what the strategy did there. Only
`state_machine_v1`/`v2` have a state at all; the other strategies compute no
feature frame and the section stays hidden rather than claiming there were no
changes. Transitions inside warmup are dimmed — measured on BTC/USDT perp 4h over
2023-01-01 → 2024-06-01, 8 of 50 fall there: the machine walked through them, but
the strategy was not acting yet, so they are not decisions.

**And what the fills made.** A trade table under the chart carries every round
trip — opened, closed, side, quantity, both prices, fees, PnL and return — sliced
from the same `Portfolio` the arrows are drawn from, so it cannot disagree with
them. The header nets **closed trades only**: a position still open is valued
against the last bar, and that mark moves on the next one and on whatever range
you asked for, so it is shown italic and left out of the total. This is what the
strategy would have made over the frame on screen at the cost model in the
provenance strip — non-compounding from initial cash, like every backtest here —
not an account. A `TargetExposure` strategy gets no table at all, because
`build_analysis` runs no book for that contract; an empty one would read as "this
strategy never traded".

Hover or click any bar and the panel below shows the state and feature values
behind it; a strategy with no feature frame says so rather than showing an empty
one. A provenance strip is on screen at all times — `crowding_measured`, the exit
mode, warmup, the cost model, the strategy version and the frame's bounds —
because a perp whose funding column went missing is a different run from one
whose did not, and that is exactly how a published figure moved once without
anyone noticing.

Nothing on this path writes to `reports/` or to `signals`; the only writes it can
make are behind the refresh button, which is `serve`'s existing fetch path called
rather than copied — candles for the identity, plus the funding settlements on a
perp. The response reports both counts, because a refresh that moved three
candles and no settlements is the drift that ends with the coverage guard
refusing the dataset you were just looking at.

## Strategies

- `turnaround_v1` — base three-candle reversal logic, no filters (control)
- `turnaround_v2` — v1 plus EMA200 trend and EMA20 extension filters (crypto intraday)
- `trend_following_deepseek_v4` — weekly long-only ETF trend following; exits delegated
  to the engine's `trend_structure` mode
- `trend_rider_v1_deepseek_v4_pro` — weekly long-only ETF trend following with ATR
  volatility gate, ATR position sizing, and fully internal exits; run with
  `--exit-mode opposite_signal_only`
- `tsmom`, `ema_cross`, `donchian`, `multi_horizon` — the Market Dynamics Engine's R0
  baselines: deliberately unfiltered trend rules that every later model has to beat
- `state_machine_v1` — the MDE R5 strategy: a six-state market lifecycle
  (compression → breakout → confirmed → riding → exhaustion → reset) with hysteresis,
  minimum dwell and a post-reset cooldown, sizing each entry from the state it opened in
- `state_machine_v2` — the MDE R6 strategy: the same machine, policy and features on the
  continuous-exposure contract, holding one signed target level per bar instead of
  entry/exit booleans. Not on the `backtest` CLI — see below

```bash
strategy-lab backtest --exchange binance --market-type perp --symbols BTC/USDT \
  --timeframe 4h --strategy state_machine_v1 --exit-mode opposite_signal_only \
  --start "2019-09-10 08:00:00" --cost-stress 1,2,3
```

`state_machine_v1` needs perp candles *and* stored funding — funding is what `crowding`
reads, and a perp backtest refuses to run without it. The `--start` is the venue's first
funding settlement; earlier bars would be charged zero carry, which reads exactly like
free carry. **`replay` supplies `crowding` too**, since `Bar` carries a funding rate and
`ReplayFeed` attaches it through the engine's own function — so a perp replay and the
backtest above now agree bar for bar, where before they differed on all 6,048 of them.
A **backtest** records which it was as `crowding_measured` in `config.json`; a **replay**
writes no such file and stores the per-bar `crowding` values themselves, in the
`bar_reasons` rows it writes for every bar past warmup.

`state_machine_v1` clears the R0 baseline out of sample on risk-adjusted terms — Sharpe
+0.896 against `donchian` 40/10's +0.072 over the same held-out 6,048 bars — while
returning far less than buy-and-hold. Read
[STRATEGIES.md](STRATEGIES.md#state_machine_v1) before quoting any of that.

**There are two strategy contracts.** `SignalSet` says enter, exit and how big to *start*;
`TargetExposure` (`strategies/exposure.py`) says what to *hold* on every bar, and
`backtests/exposure_engine.py` executes it through
`Portfolio.from_orders(size_type="targetvalue")` — a level rather than an event, so a
position can be resized without being closed, which `from_signals` cannot do (it reads a
size only on the bar that opens a position). Both stay: the four original strategies and
`state_machine_v1` keep `SignalSet` and their published results, while `state_machine_v2`
runs the continuous one and is registered in a third manual registry
(`strategies/exposure_registry.py`), because every boolean test suite calls
`generate_signals`. R6 measured what the second contract bought over R5's held-out half:
**the taper it was built for is worth approximately zero** (+15.30 / −63.05 gross over
75/77 bars), what v2 actually buys is 1.7–1.8× v1's average exposure with return scaling
accordingly and Sharpe flat, and the reason is on v1's side — not one of its entries is
sized for `RIDING`, the row its own policy sizes highest. See
[STRATEGIES.md](STRATEGIES.md#state_machine_v2) and
[the charter §9.3](docs/research/2026-08-03-market-dynamics-engine.md#93-r6-continuous-exposure-comparison--btcusdt-perp-4h).

See [STRATEGIES.md](STRATEGIES.md) for the source of truth: per-strategy logic,
parameters, canonical run commands, the exit-mode compatibility matrix, and known issues.

Indicators and signals are derived at backtest time. The database stores raw candles only.

## Add A New Trading Vehicle

Add a fetcher under `src/strategy_lab/market_data/` that returns a DataFrame with:

```text
timestamp index, open, high, low, close, volume
```

Then write it with `normalize_candle_frame(...)` and identify it with:

```text
exchange, market_type, symbol, timeframe
```

The same strategy and backtest code can then run against the new vehicle.
