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

Key design decisions that span multiple files:

- **Candle identity is `(exchange, market_type, symbol, timeframe)`** with the timeframe
  as a literal string — `1w` and `1wk` are distinct datasets. All indicators are computed
  at backtest time from raw candles.
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
  `storage/migrations.py`).
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
   plus `version` — a semver string — and `warmup_bars` — its largest lookback,
   in bars).
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
