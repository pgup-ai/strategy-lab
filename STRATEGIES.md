# Strategy Tracker

Source of truth for what each strategy does, how it is meant to be run, and what is known
about its behavior. Update this file whenever strategy logic, parameters, or engine exit
behavior changes — the README only carries quick-start commands.

Last reviewed: 2026-08-02 at commit `31334c4`.

## At a glance

| Strategy | Direction | Designed for | Entry core | Regime filters | Exit owner | Sizing | Status* |
|---|---|---|---|---|---|---|---|
| `turnaround_v1` | long + short | any timeframe (control) | 2 red candles → 1 green (mirror for shorts) | none | engine | flat | baseline control |
| `turnaround_v2` | long + short | crypto intraday (15m spot) | same as v1 | EMA200 trend + EMA20 extension | engine | flat | active — crypto |
| `trend_following_deepseek_v4` | long only | weekly equity ETFs | same pattern within uptrend | SMA40 trend + 1.20× max extension | engine (`trend_structure`) | flat | active — weekly ETF, simple variant |
| `trend_rider_v1_deepseek_v4_pro` | long only (short path exists) | weekly equity ETFs | same pattern within uptrend | SMA40 trend + ATR volatility gate | strategy (run pass-through) | ATR-scaled | active — weekly ETF, current focus |

\* Status is inferred from report history — correct these labels as research priorities change.

## How signals flow

Strategies return a `SignalSet` of exit *ingredients* (opposite-signal exits, setup stop
levels, EMA trend-failure series, optional per-bar position scale). The engine's
`ExitMode` ([backtests/engine.py](src/strategy_lab/backtests/engine.py)) decides which
ingredients actually fire. The same strategy can therefore behave very differently per
run — always record the `exit_mode` when comparing results (it is written to each
report's `config.json`).

All entries share the same three-candle turnaround pattern: two adverse candles followed
by one reversal candle, evaluated and filled on the signal bar's close.

---

## turnaround_v1

Baseline reversal logic with no filters — kept as the control for measuring what v2's
filters add.

- **Entry**: long after 2 red candles then a green candle; short after 2 green then a red.
- **Exits provided**: opposite entry signal; setup-invalidation stop levels (distance to
  the 3-bar setup low/high); EMA200 trend-failure series.
- **Params**: `trend_failure_ema_span=200`.
- **Run**: same CLI as v2 with `--strategy turnaround_v1`, any exit mode.

## turnaround_v2

v1 plus the first-phase regime filters. Entries become with-trend pullback reversals:
buy the turnaround only in an uptrend while price is *not* extended above the short EMA.

- **Entry (long)**: v1 pattern AND `close > EMA200` AND `close < EMA20 × 0.99`.
- **Entry (short)**: mirror — `close < EMA200` AND `close > EMA20 × 1.01`.
- **Exits provided**: same set as v1 (opposite signal, setup stop, EMA200 trend failure).
- **Params**: `ema_trend_span=200`, `ema_extension_span=20`, `long_extension=0.99`,
  `short_extension=1.01`.
- **Canonical run** (crypto intraday, engine default `continuation_failure` exits):

  ```bash
  strategy-lab backtest --exchange binance --market-type spot \
    --symbols BTC/USDT,ETH/USDT,SOL/USDT --timeframe 15m --strategy turnaround_v2
  ```

## trend_following_deepseek_v4

Weekly long-only ETF strategy, intentionally minimal: it emits **no exits at all** and
delegates exiting entirely to the engine's `trend_structure` mode.

- **Entry**: turnaround pattern AND `close > SMA40` AND `close < SMA40 × 1.20`
  (skip entries more than 20% above trend).
- **Exits provided**: none — all-False exits, no stop levels, no trend-failure series.
- **Params**: `trend_sma_span=40`, `max_extension=1.20`. `allow_shorts` is hardcoded
  False (the registry does not pass the CLI flag through).
- **Canonical run**:

  ```bash
  strategy-lab backtest --exchange yahoo --market-type equity --symbols SPY \
    --timeframe 1w --strategy trend_following_deepseek_v4 \
    --exit-mode trend_structure --cash 100000
  ```

- **Caveats**: `trend_failure` and `setup_invalidation_stop` modes raise (the strategy
  provides neither ingredient). `opposite_signal_only` never exits — it degenerates to
  buy-and-hold from the first signal.

## trend_rider_v1_deepseek_v4_pro

The "pro" weekly variant and the only strategy using the `filters/` package. Unlike the
others it **bakes every exit into its own signals**, so the engine must be run in
pass-through mode.

- **Entry (long)**: turnaround pattern AND `close > SMA40` AND ATR(14)/close < 0.10
  (volatility gate — skip entries when weekly vol is extreme).
- **Exits (internal, OR-ed into `long_exits`)**:
  1. opposite short setup appears;
  2. continuation failure — 3 consecutive lower closes (`failure_bars=3`, internal);
  3. regime break — close falls below SMA40;
  4. momentum divergence — 26-bar rate of change turns negative.
- **Sizing**: ATR volatility targeting — `scale = 0.06 / (ATR14/close)`, clipped to
  [0.3, 1.0], multiplied into `position_pct`. Position shrinks when vol is high.
- **Params**: `sma_trend_span=40`, `roc_momentum_period=26`, `atr_period=14`,
  `atr_max_ratio=0.10`, `failure_bars=3`, `target_atr_ratio=0.06`,
  `min_position_scale=0.3`, `max_position_scale=1.0`.
- **Canonical run** (matches the latest report configs):

  ```bash
  strategy-lab backtest --exchange yahoo --market-type equity \
    --symbols SPY,QQQ,SMH,XLF,XLK --timeframe 1w \
    --strategy trend_rider_v1_deepseek_v4_pro \
    --exit-mode opposite_signal_only --no-allow-shorts
  ```

- **Caveats**: the engine-default `continuation_failure` mode layers a 4-bar rule on top
  of the internal 3-bar rule (the stricter internal rule always fires first, so the
  engine layer is dead weight — but it muddies comparisons). `trend_structure` mode runs
  with the strategy's published SMA span but replaces the internal exits entirely —
  stick to `opposite_signal_only`.

The ETF universe for weekly runs lives in [universe/etfs.py](src/strategy_lab/universe/etfs.py)
(4 broad + 3 international + 11 sector ETFs); batch-fetch with `strategy-lab fetch-etf-universe`.

---

## Exit mode × strategy matrix

Engine defaults: `exit_mode=continuation_failure`, `failure_bars=4`.

| `exit_mode` | turnaround_v1 / v2 | trend_following_deepseek_v4 | trend_rider_v1 |
|---|---|---|---|
| `continuation_failure` (default) | opposite signal OR N adverse closes | N adverse closes only | ⚠ redundant on top of internal 3-bar — avoid |
| `opposite_signal_only` | opposite signal only | ⚠ never exits | ✅ canonical (pass-through) |
| `trend_failure` | opposite signal OR EMA200 cross | ✗ raises | ✗ raises |
| `setup_invalidation_stop` | opposite signal + stop at setup extreme | ✗ raises | ✗ raises |
| `trend_structure` | long-only: raises if short entries exist; SMA40 via fallback | ✅ canonical: SMA40 break OR N adverse closes | long-only (raises if short entries exist — pass `--no-allow-shorts`); replaces the internal exits |

## Engine behavior worth remembering

- **Costs**: default `fees=0.0005` and `slippage=0.0005` per side, `cash=10_000`,
  `position_pct=0.95`.
- **Sizing is non-compounding**: entry shares = initial cash × `position_pct` × scale ÷
  close. Sizes are anchored to *initial* cash, never to current equity.
- **Data**: Yahoo OHLC is rescaled by adj close, so ETF series are dividend-adjusted
  (total-return-like). Crypto candles are raw exchange OHLCV.
- **Timeframe strings are identity**: candles are keyed by the literal timeframe string —
  `1w` and `1wk` are *separate datasets* (both exist in past reports). Pick one spelling
  per instrument and stick to it.
- **Reports**: every run writes `reports/<UTC>_<exchange>_<market>_<symbol>_<tf>_<strategy>/`
  with `config.json` (full parameter snapshot), `stats.json`, `trades.csv`,
  `equity_curve.csv`, `plot.html`. `config.json` is the reproducibility boundary.

## Known issues (review of 2026-08-02)

1. **RESOLVED 2026-08-02 — shorts were silently disabled engine-wide** between commit
   `31334c4` and this fix: `_compute_entry_sizes` assigned share sizes only on
   long-entry bars, so short entries got size 0 and vectorbt placed no short trades.
   Any `turnaround_v1`/`v2` run with shorts enabled in that window traded long-only —
   rerun those backtests before trusting their stats. Now both entry series are sized;
   pinned by `test_run_backtest_opens_short_trades` in tests/test_backtest_exits.py.
2. **RESOLVED 2026-08-02 — metadata key mismatch**: `trend_rider_v1` now publishes
   `trend_sma_span` (the key the engine's `trend_structure` mode reads) instead of
   `sma_trend_span`; pinned by `test_trend_rider_publishes_trend_sma_span_for_engine`.
3. **RESOLVED 2026-08-02 — typo**: `EFT_UNIVERSE` renamed to `ETF_UNIVERSE` across
   universe/etfs.py, universe/__init__.py, and cli.py.
4. **RESOLVED 2026-08-02 — stale CLI help**: the `--exit-mode` help text now lists
   `trend_structure`.
