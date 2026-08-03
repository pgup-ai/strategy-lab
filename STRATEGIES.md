# Strategy Tracker

Source of truth for what each strategy does, how it is meant to be run, and what is known
about its behavior. Update this file whenever strategy logic, parameters, or engine exit
behavior changes — the README only carries quick-start commands.

Last reviewed: 2026-08-03 at commit `498eed5`.

## At a glance

| Strategy | Direction | Designed for | Entry core | Regime filters | Exit owner | Sizing | Status* |
|---|---|---|---|---|---|---|---|
| `turnaround_v1` | long + short | any timeframe (control) | 2 red candles → 1 green (mirror for shorts) | none | engine | flat | baseline control |
| `turnaround_v2` | long + short | crypto intraday (15m spot) | same as v1 | EMA200 trend + EMA20 extension | engine | flat | active — crypto |
| `trend_following_deepseek_v4` | long only | weekly equity ETFs | same pattern within uptrend | SMA40 trend + 1.20× max extension | engine (`trend_structure`) | flat | active — weekly ETF, simple variant |
| `trend_rider_v1_deepseek_v4_pro` | long only (short path exists) | weekly equity ETFs | same pattern within uptrend | SMA40 trend + ATR volatility gate | strategy (run pass-through) | ATR-scaled | active — weekly ETF, current focus |
| `tsmom` | long + short | any (MDE R0 baseline) | sign of the 96-bar trailing return | none | engine | flat | R0 baseline |
| `ema_cross` | long + short | any (MDE R0 baseline) | EMA48 vs EMA192 | none | engine | flat | R0 baseline |
| `donchian` | long + short | any (MDE R0 baseline) | close breaks the 96-bar channel | none | strategy (48-bar reverse channel) | flat | R0 baseline |
| `multi_horizon` | long + short | any (MDE R0 baseline) | sign of a 24/48/96/192 vol-normalized blend | none | engine | flat | R0 baseline |

\* Status is inferred from report history — correct these labels as research priorities change.

The bottom four are the [Market Dynamics Engine](docs/research/2026-08-03-market-dynamics-engine.md)
R0 baselines: the floor every later state-estimation model has to clear out-of-sample.
They are deliberately unfiltered and unoptimized — a baseline that has been tuned is not
a baseline.

## How signals flow

Strategies return a `SignalSet` of exit *ingredients* (opposite-signal exits, setup stop
levels, EMA trend-failure series, optional per-bar position scale). The engine's
`ExitMode` ([backtests/engine.py](src/strategy_lab/backtests/engine.py)) decides which
ingredients actually fire. The same strategy can therefore behave very differently per
run — always record the `exit_mode` when comparing results (it is written to each
report's `config.json`).

The four original strategies share the same three-candle turnaround pattern: two adverse
candles followed by one reversal candle, evaluated and filled on the signal bar's close.
The R0 baselines do not — they are continuous trend-state rules (or, for `donchian`, a
channel break), which is what makes them a fair floor for a trend program.

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

# MDE R0 baselines

Four unfiltered trend rules that exist to be beaten. None of them provides a setup stop
or a trend-failure series, so `setup_invalidation_stop` and `trend_failure` raise for all
four. Default parameters are quoted in bars, and were chosen to be round numbers on a 15m
grid (96 bars = one day) rather than tuned — the R0 parameter sweep is what tests whether
any of them sits on a stable plateau.

**All four are `warmup_bars`-measured, not `warmup_bars`-declared.** `tsmom`, `donchian`
and `multi_horizon` use `rolling`/`pct_change` only, so warmup is exactly the longest
lookback. `ema_cross` uses `ewm(adjust=False)`, which recurses from bar 0, so its warmup
is 20× the slow span — see the note in [CLAUDE.md](CLAUDE.md) and
`tests/test_strategy_metadata.py`.

## tsmom

The single most-documented trend effect in the literature, and the reference floor.

- **Entry**: long while the `lookback`-bar trailing return is positive, short while it is
  negative. A continuous state, not an event — it re-asserts on every bar.
- **Exits provided**: opposite state only.
- **Params**: `lookback=96`, `warmup_bars=96`.

## ema_cross

- **Entry**: long while EMA(`fast_span`) > EMA(`slow_span`), short while it is below.
- **Exits provided**: opposite state only.
- **Params**: `fast_span=48`, `slow_span=192`, `warmup_bars=3840` (= 20 × `slow_span`).
- **Note**: 3840 is not conservatism. At `warmup_bars=192` the span-192 EMA is wrong by up
  to 2.6e-2 *relative* on 299 of 300 probed bars, and the fast-vs-slow comparison still
  comes out identical on all 300 — so the signal-level cold-start test cannot see it.
  `test_recursive_ema_is_bit_exact_after_the_declared_warmup` is what pins this.

## donchian

Turtle-style channel breakout: enter slowly, leave faster, so a trend can be ridden
rather than round-tripped.

- **Entry**: long when close exceeds the prior `entry_span`-bar high; short when it falls
  below the prior `entry_span`-bar low.
- **Exits provided**: reverse break of the shorter `exit_span` channel — independent of
  the entry state, which makes this the only R0 baseline that still exits when shorts are
  disabled (see the caveat below).
- **Params**: `entry_span=96`, `exit_span=48`, `warmup_bars=96`.
- **Note**: every channel is `.rolling(n).max().shift(1)`. The `shift(1)` is load-bearing
  but *not* a lookahead guard — bar *t*'s own high is known at bar *t*'s close, so the
  poison probe passes without it. What it prevents is degeneracy: `high >= close` within a
  bar, so an unshifted channel makes `close > entry_high` unsatisfiable and the strategy
  never trades at all.

## multi_horizon

Averages volatility-normalized trailing returns across several lookbacks. The point is
not a better number — it is removing the single-lookback choice, which is where most trend
backtests quietly overfit.

- **Entry**: long while the blended score exceeds `entry_threshold`, short while it is
  below its negative.
- **Score**: mean over `lookbacks` of `pct_change(lookback) / (rolling(96).std() *
  sqrt(lookback))` — each horizon becomes a unit-scale t-statistic under a random walk, so
  neither a noisier regime nor a longer horizon can dominate the blend by raw magnitude.
- **Exits provided**: opposite state only.
- **Params**: `lookbacks=(24, 48, 96, 192)`, `entry_threshold=0.0`, `warmup_bars=192`.
- **Note**: `rolling(...).std()` is not bit-reproducible across a cold start (pandas adds
  and removes observations one at a time, and the removals leave rounding residue).
  Measured at warmup 192 the score differs from the whole-history value on 195/300 bars —
  by at most 1.1e-15, against a smallest observed |score| of 5.9e-3. Signals are identical
  on every bar, which is why `warmup_bars` is the longest lookback and not a multiple.

### ⚠ Caveat: `--no-allow-shorts` removes the long exit

`tsmom`, `ema_cross` and `multi_horizon` all wire `long_exits = short_state`. Passing
`--no-allow-shorts` forces `short_state` to all-False, which silently takes the long exit
with it: measured on 5,000 synthetic bars, `opposite_signal_only` drops from 2,889 to
**0** long exits for `tsmom` (3,191 → 0 for `ema_cross`, 2,745 → 0 for `multi_horizon`).
A long-only run in that mode never closes a position. Either keep shorts enabled, or use
`continuation_failure`/`trend_structure`, which supply engine-side exits. `donchian` is
unaffected because its exits come from the reverse channel rather than the entry state.

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

R0 baselines (verified against the engine on 5,000 synthetic bars, 2026-08-03):

| `exit_mode` | tsmom / ema_cross / multi_horizon | donchian |
|---|---|---|
| `continuation_failure` (default) | opposite state OR N adverse closes | channel exit OR N adverse closes |
| `opposite_signal_only` | ✅ canonical *with shorts on*; ⚠ **never exits a long** under `--no-allow-shorts` | ✅ canonical — channel exit, unaffected by `--no-allow-shorts` |
| `trend_failure` | ✗ raises (no trend-failure series) | ✗ raises (no trend-failure series) |
| `setup_invalidation_stop` | ✗ raises (no setup stop) | ✗ raises (no setup stop) |
| `trend_structure` | long-only (raises if short entries exist — pass `--no-allow-shorts`); SMA40 via fallback | same; note it *replaces* the channel exit |

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
