# Strategy Lab

Local Python research repo for crypto and stock strategy backtesting.

The current stack is intentionally small:

- Postgres 16 in Docker for reproducible OHLCV storage
- `ccxt` for crypto candles
- `yfinance` for stock candles
- `pandas` and `numpy` for strategy research
- `vectorbt` for fast signal-based backtests and plots

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

No short exits and no opposite signal exits.

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
- `plot.html`

That report directory is the reproducibility boundary for comparing strategy changes.

## Strategies

`turnaround_v1` is the base reversal logic:

- long after two red candles followed by a green candle
- short after two green candles followed by a red candle

`turnaround_v2` adds the first-phase filters:

- long only above EMA200 trend
- short only below EMA200 trend
- long only when price is below EMA20 extension threshold
- short only when price is above EMA20 extension threshold
- fees and slippage are applied in the backtest runner
- continuation failure exits are applied by default in the backtest runner

`trend_following_deepseek_v4` is a weekly-focused long-only strategy:

- turnaround entry (2 red + 1 green) within a macro uptrend
- long-only above 40-week SMA trend filter
- entry only when price is below 10-week SMA extension threshold
- no opposite signal exits — exits are handled entirely by the engine
- designed for weekly timeframes with equity ETFs (SPY, QQQ, etc.)

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
