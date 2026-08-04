# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
source .venv/bin/activate          # Python 3.11 venv (project requires >=3.10,<3.13)
pip install -e ".[dev]"            # editable install; provides the strategy-lab CLI
docker compose up -d postgres      # Postgres 16 — required for fetch/backtest commands
strategy-lab init-db               # create the market_candles schema

pytest -q                          # run all tests (works without install; pythonpath=src)
pytest tests/test_backtest_exits.py::test_sma_break_exits_when_close_crosses_below -q
ruff check src tests               # lint (line-length 100)

strategy-lab strategies            # list registered strategies
strategy-lab data-sets             # list candle sets stored in Postgres
strategy-lab fetch-crypto --symbol BTC/USDT --timeframe 15m --since 2024-01-01
strategy-lab fetch-stock --symbol SPY --timeframe 1w --start 2020-01-01
strategy-lab fetch-etf-universe    # batch-fetch the configured ETF universe (weekly)
strategy-lab backtest --exchange yahoo --market-type equity --symbols SPY \
  --timeframe 1w --strategy trend_following_deepseek_v4 --exit-mode trend_structure
strategy-lab sweep --symbol BTC/USDT --timeframe 15m --strategy donchian \
  --grid '{"entry_span":[48,96,192],"exit_span":[24,48,96]}'
strategy-lab serve                 # serve reports/ with the live candle-refresh API
```

`backtest` only reads candles already stored in Postgres — fetch first. Database URL
comes from `DATABASE_URL` (`.env`, see `.env.example`); defaults to the docker-compose
Postgres.

## Architecture

Data flow: `market_data/` fetchers (ccxt for crypto, yfinance for stocks) →
`db.candles.normalize_candle_frame` → Postgres `market_candles` (raw OHLCV only, no
derived data) → `load_candles` → `strategy.generate_signals(df)` → `SignalSet` →
`backtests.engine.run_backtest` → vectorbt `Portfolio.from_signals` → timestamped
directory under `reports/` (`config.json` there is the full reproducibility record;
`plot.html` is a self-contained TradingView-style report rendered by
`backtests/report.py`, which inlines the vendored Lightweight Charts asset).

A second, event-driven flow shares the same strategy layer: `ReplayFeed`
(`feeds/replay.py`, Postgres-backed, satisfies the `MarketDataFeed` protocol in
`feeds/base.py`) yields `BarEvent`s → `StrategyRunner.on_event`
(`engine/runner.py`) appends each closed bar to a `BarBuffer`
(`engine/context.py`) and calls the same `strategy.generate_signals(df)` once
per bar, reading only the last row → `Signal` → `storage.signals.write_signals`
→ the append-only `signals` table. The `replay` CLI drives this path today; a
live feed will drive it later through the same protocol, with no change to the
runner or to strategy code. See
[the Phase 1a design doc](docs/design/2026-08-02-realtime-trading-framework.md)
for the full rationale.

The event-driven flow also runs many instruments at once: `stream()` k-way
merges every subscription into one time-ordered stream, `MarketClock`
(`engine/market_clock.py`) groups it into `MarketSnapshot`s, and
`MultiAssetRunner` (`engine/multi_runner.py`) holds one full-history
`BarBuffer` per instrument and delegates each traded one to its own
`StrategyRunner`. `features/cross_sectional.py` reads a snapshot.

A third flow scores a strategy across a *grid* rather than at one setting:
`sweep_parameters` (`backtests/sweep.py`) rebuilds the strategy per cell with
`dataclasses.replace`, converts each `SignalSet` to a ±1 position with
`positions_from_signals`, and reduces the surface to a `stability_score` that
ranks a broad plateau above a lone spike. `backtests/sweep_report.py` renders it
as a self-contained heatmap. This path is deliberately vectorbt-free and
**gross of costs** — it answers "is this parameter region stable", not "what
would this have earned".

Key design decisions that span multiple files:

- **Candle identity is `(exchange, market_type, symbol, timeframe)`** with the timeframe
  as a literal string — `1w` and `1wk` are distinct datasets. All indicators are computed
  at backtest time from raw candles. In the event engine that identity is `CandleId`
  (`core/types.py`), and it is what every bar-holding structure keys on: the feed's
  merge tie-break, `MarketSnapshot.bars`, and `ReplayFeed.frames`. `InstrumentId` names
  what is *traded* and is the right key only where a timeframe is fixed by
  construction — `MultiAssetRunner`'s per-instrument buffers, which is why it rejects a
  bar at any other timeframe rather than absorbing it. Keying a snapshot by instrument
  alone silently drops bars: BTC 4h and BTC 1d close at the same instant, and measured
  on a 12-bar 4h + 2-bar 1d feed, 2 of 14 bars never reached a snapshot.
- **Exit ownership is split between strategies and the engine.** Strategies return a
  `SignalSet` of exit ingredients (opposite-signal exits, setup stop levels,
  trend-failure series, optional per-bar `position_size` scale); the engine's `ExitMode`
  decides which ingredients fire. Not every strategy supports every mode — see the
  exit-mode × strategy matrix in [STRATEGIES.md](STRATEGIES.md) before changing exit
  behavior or comparing runs. Two strategies invert the usual pattern:
  `trend_rider_v1_deepseek_v4_pro` bakes all exits into its own signals (run with
  `--exit-mode opposite_signal_only`), and `trend_following_deepseek_v4` emits none
  (run with `--exit-mode trend_structure`).
- **Strategies are frozen dataclasses** satisfying the `Strategy` protocol
  (`name`, `version`, `warmup_bars` + `generate_signals`). Registration is manual in
  `strategies/registry.py` — both `list_strategies()` and `get_strategy()`.
- **Position sizing is non-compounding**: the engine sizes entries from *initial* cash ×
  `position_pct` × the strategy's optional ATR scale, never from current equity.
- The Yahoo fetcher rescales OHLC by adjusted close, so equity data is
  dividend-adjusted; crypto data is raw exchange OHLCV.
- **Vectorized and event-driven execution share one strategy implementation,
  proven rather than assumed.** `backtest` calls `generate_signals(df)` once
  over the whole range; `replay` calls it once per closed bar over an
  expanding buffer and reads only the last row — correct only because every
  strategy is causal (row *t* depends on rows ≤ *t* and nothing later).
  `tests/test_replay_determinism.py` asserts the two paths emit identical
  signals; `tests/test_lookahead.py` poisons every bar after *t* and asserts
  row *t* is unchanged, which is the direct causality proof. **A strategy that
  fails either test is not safe to trade.**
- **A timestamp is complete only once an event with a *later* timestamp
  arrives** — never by looking ahead, which is the same lookahead the two
  suites above exist to prevent, and the only completeness a live feed can
  establish. Three consequences, all deliberate, all easy to violate by
  accident. (1) **Cross-sectional work lags one bar**: `MultiAssetRunner`
  holds bar *t* until the first *t+1* event proves *t* done, so signals for
  *t* are emitted then. Dispatching on arrival instead would emit sooner but
  could only ever see the *t−1* cross-section. (2) **The final timestamp needs
  an explicit `flush()`** — nothing arrives after it, so a caller that drops
  `flush()` silently loses the last snapshot and the last bar of every buffer.
  (3) **A snapshot holds only instruments that have a bar at that time**, and
  `absent` must never be read as `unchanged` — instruments list, delist and
  halt, and crypto trades hours equities do not. `breadth` therefore refuses a
  universe below `min_instruments` instead of dividing by whatever showed up;
  on a mixed 4h/1d universe 10 of 12 snapshots hold a single instrument, where
  that quotient is a well-formed number carrying no cross-sectional content.
- **The bar buffer keeps full history, never a rolling window.**
  `turnaround_v1`/`turnaround_v2` compute `ewm(adjust=False)`, which is
  recursive from the first bar. Measured: a 60-bar window produces wrong
  values on 200 of 200 sampled bars for both. The two SMA-based strategies are
  window-safe and the two EWM-based ones are not, and nothing declares which
  is which, so the buffer keeps everything rather than trusting a
  per-strategy flag.
- **Decimal at the boundaries, float64 inside indicators.** `core` types
  (`Bar`, `Signal`) and the `market_candles`/`signals` columns are
  `Decimal`/`NUMERIC(38,18)`; `load_candles` and `BarBuffer` are the two
  places that convert to float64 for the pandas indicator layer, and prices
  never convert back. Widening a `NUMERIC` column later must cast through
  text (`col::text::numeric`) — Postgres' implicit `float8 → numeric` cast
  silently drops the last two significant digits of a float64 (see
  `storage/migrations.py`). The same cast fires on *every write* that binds a
  Python `float` to a `NUMERIC` column, so `normalize_candle_frame` emits
  `Decimal(str(float(x)))`; binding a bare `float` there quietly re-corrupts
  what the migration fixed, on every re-fetch.
- **Funding and open interest are their own tables, not candle columns.**
  Funding is a cash flow settled on the venue's own schedule and OI is a
  point-in-time snapshot, so `funding_rates` and `open_interest`
  (`db/funding.py`, DDL in `storage/migrations.py`) sit beside `market_candles`
  rather than inside it. Perp *candles* do live in `market_candles` under
  `market_type="perp"`, via the same `normalize_candle_frame` +
  `upsert_candles` path as everything else. Three measured facts that are not
  in any doc: **Binance serves only ~30 days of open interest** (a `startTime`
  40 days back returns `-1130`), so OI can only accumulate forward and any
  historical OI study is unanswerable from this source; **`markPrice` is `""`
  on funding records before 2023-10-31**, and `Decimal("")` raises, so it is
  parsed as NULL; and **funding timestamps are up to 47 ms past the 8h
  boundary**, so match funding to the bar whose interval contains it, never by
  equality against a
  generated 8h range. The settlement interval is per-contract — nothing here
  hardcodes 8h.
- **Values that arrive as exchange strings stay `Decimal` end to end.**
  `db/candles.py` uses `Decimal(str(float(x)))` because its input is float64
  out of pandas; `db/funding.py` binds an incoming `Decimal` unchanged, since
  routing an exact decimal through float64 would discard digits the
  `NUMERIC(38,18)` column can hold. Both rules exist to keep a bare `float` out
  of a `NUMERIC` bind — that is the failure mode, not the specific coercion.
- **`signals` is append-only**, enforced by two triggers — row-level `BEFORE
  UPDATE OR DELETE` and statement-level `BEFORE TRUNCATE` (`TRUNCATE` bypasses
  row-level triggers, so both are required). There is no ordinary SQL path to
  modify or remove a signal row; deliberate cleanup requires disabling the
  triggers first:

  ```sql
  ALTER TABLE signals DISABLE TRIGGER USER;
  DELETE FROM signals WHERE ...;
  ALTER TABLE signals ENABLE TRIGGER USER;
  ```

  Each `replay` invocation also mints its own `run_id` — `write_signals`
  idempotency (the `uq_signals_identity` constraint) is *within* a run, not
  across runs, so replaying the same range twice is expected to add a second
  run's worth of rows, not zero.

## Adding a strategy

1. New module in `src/strategy_lab/strategies/` (frozen dataclass, unique `name`,
   plus `version` — a semver string — and `warmup_bars`, in bars).
   `warmup_bars` is **not** simply the largest declared lookback. It is however
   many bars make a cold start agree with a whole-history backtest, which for a
   `rolling(n)` window is `n` but for `ewm(span=n, adjust=False)` is ~`20n` —
   the recursion decays its seed rather than dropping it, so a span-200 EMA is
   still wrong after 200 bars and only becomes bit-exact around 4000.
   `tests/test_strategy_metadata.py` enforces this by replaying the cold start;
   trust it over the declared spans.
2. Register it in `strategies/registry.py` (two places).
3. Decide exit ownership: which `SignalSet` ingredients it provides and which
   `ExitMode`s are valid for it.
4. Add tests under `tests/`.
5. Add its row and section to [STRATEGIES.md](STRATEGIES.md) — that file is the source
   of truth for strategy logic, canonical run commands, and known issues; keep it in
   sync when strategy or engine exit behavior changes.

Once registered, a strategy is automatically exercised by
`tests/test_lookahead.py` and `tests/test_replay_determinism.py` — both iterate
`strategies.registry.list_strategies()`, so no additional test wiring is
needed. A new strategy that fails either is not a test problem: it means the
strategy reads future data, or the two execution paths genuinely disagree for
it.
