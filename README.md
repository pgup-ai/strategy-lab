# Strategy Lab

Local Python research repo for crypto and stock strategy backtesting.

The current stack is intentionally small:

- Postgres 16 in Docker for reproducible OHLCV storage, plus an append-only
  `runs`/`signals` store for replay and (later) live signal history
- `ccxt` for crypto candles
- `yfinance` for stock candles
- `pandas` and `numpy` for strategy research
- `vectorbt` for fast signal-based backtests and plots
- a small event-driven engine (`core/`, `feeds/`, `engine/`) so the same
  strategy code can run in backtest, replay, and (later) live

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
    engine/          # event-driven runners, bar buffers, market clock
    features/        # cross-sectional reads over a market snapshot
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
for signal.

`replay` persists to `runs`/`signals` tables that only `strategy-lab migrate`
creates — `init-db` does not. Run `migrate` once; on a brand-new database run
`init-db` first, since `migrate` widens `market_candles` to `NUMERIC` and needs
the table to already exist. `migrate` is idempotent and safe to re-run:

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

Every invocation mints a fresh `run_id`, so replaying the same range twice
stores two independent runs rather than zero rows the second time — that is
the append-only audit trail working as intended, not a bug. Some
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

async for event in feed.stream(subs):
    snapshot = clock.on_event(event)
    if snapshot:
        breadth(snapshot)   # strategy_lab.features.cross_sectional
```

`breadth` and `confirms` read one snapshot and nothing else. Both assume a
shared timeframe across the universe — a daily bar meets a 4h bar only at day
boundaries, so a mixed-timeframe universe produces mostly single-instrument
snapshots. `breadth` refuses a universe smaller than `min_instruments`
(default 2) rather than returning a well-formed number that is really just one
instrument's direction.

Measured over the stored BTC + ETH perp 4h series: **15,128 snapshots, mean
breadth 0.512, 477 partial universes** — that last figure is exactly the
stretch where BTC had listed and ETH had not (2019-09-08 to 2019-11-27).

## Live Report Serving

Reports are static files, but `strategy-lab serve` adds a delayed live feed:

```bash
strategy-lab serve --port 8750
```

Open a report through the server (`http://127.0.0.1:8750/<report-dir>/plot.html`) and a
"delayed" pill appears in the header: the page polls `/api/candles` every 60 seconds,
which re-fetches the latest bars from the upstream source (Yahoo Finance or ccxt),
upserts them into Postgres, and streams them onto the chart — including the current
forming bar. Click the pill to refresh immediately. The same file opened directly from
disk stays fully static.

## Strategies

- `turnaround_v1` — base three-candle reversal logic, no filters (control)
- `turnaround_v2` — v1 plus EMA200 trend and EMA20 extension filters (crypto intraday)
- `trend_following_deepseek_v4` — weekly long-only ETF trend following; exits delegated
  to the engine's `trend_structure` mode
- `trend_rider_v1_deepseek_v4_pro` — weekly long-only ETF trend following with ATR
  volatility gate, ATR position sizing, and fully internal exits; run with
  `--exit-mode opposite_signal_only`

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
