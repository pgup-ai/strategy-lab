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
  (`name` + `generate_signals`). Registration is manual in
  `strategies/registry.py` — both `list_strategies()` and `get_strategy()`.
- **Position sizing is non-compounding**: the engine sizes entries from *initial* cash ×
  `position_pct` × the strategy's optional ATR scale, never from current equity.
- The Yahoo fetcher rescales OHLC by adjusted close, so equity data is
  dividend-adjusted; crypto data is raw exchange OHLCV.

## Adding a strategy

1. New module in `src/strategy_lab/strategies/` (frozen dataclass, unique `name`).
2. Register it in `strategies/registry.py` (two places).
3. Decide exit ownership: which `SignalSet` ingredients it provides and which
   `ExitMode`s are valid for it.
4. Add tests under `tests/`.
5. Add its row and section to [STRATEGIES.md](STRATEGIES.md) — that file is the source
   of truth for strategy logic, canonical run commands, and known issues; keep it in
   sync when strategy or engine exit behavior changes.
