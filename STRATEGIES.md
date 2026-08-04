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
| `state_machine_v1` | long + short | crypto perp 4h (MDE R5) | side of the state machine's target risk | the state machine itself | strategy (target side change) | per-state target, entry only | R5 gate passed out of sample |

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
- **Sizing**: ATR-scaled *entries* — `scale = 0.06 / (ATR14/close)`, clipped to
  [0.3, 1.0], multiplied into `position_pct`. A position is **entered** smaller when vol
  is high; like `--size-mode vol-scaled-entry` it does not resize a position already open
  (`from_signals` fills once per state change), so this is not continuous vol targeting
  however much the `0.06` target reads like one.
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

**And all four *derive* it in `__post_init__` rather than storing a constant**, because
the spans are swept: `sweep_parameters` rebuilds every cell with `dataclasses.replace`,
so a warmup frozen at the default under-warms the larger cells. A `warmup_bars=` passed
to a constructor is overwritten — it is a consequence of the spans, not a free parameter.
The sweep in turn evaluates every cell at the *deepest* cell's warmup, which is the only
value that keeps the surface on one sample while leaving each cell converged.

## tsmom

The single most-documented trend effect in the literature, and the reference floor.

- **Entry**: long while the `lookback`-bar trailing return is positive, short while it is
  negative. A continuous state, not an event — it re-asserts on every bar.
- **Exits provided**: opposite state only.
- **Params**: `lookback=96`, `warmup_bars` = `lookback`.

## ema_cross

- **Entry**: long while EMA(`fast_span`) > EMA(`slow_span`), short while it is below.
- **Exits provided**: opposite state only.
- **Params**: `fast_span=48`, `slow_span=192`, `warmup_bars` = 20 × `slow_span` (3840 at
  the default).
- **Note**: 3840 is not conservatism. At `warmup_bars=192` the span-192 EMA is wrong by up
  to 2.6e-2 *relative* on 299 of 300 probed bars, and the fast-vs-slow comparison still
  comes out identical on all 300 — so the signal-level cold-start test cannot see it.
  `test_recursive_ema_is_bit_exact_after_the_declared_warmup` is what pins this.

## donchian

Turtle-style channel breakout: enter slowly, leave faster, so a trend can be ridden
rather than round-tripped.

- **Entry**: long when close exceeds the prior `entry_span`-bar high; short when it falls
  below the prior `entry_span`-bar low.
- **Exits provided**: reverse break of the shorter `exit_span` channel — a separate
  signal rather than the inverse of the entry state, which is what lets the exit be
  faster than the entry.
- **Params**: `entry_span=96`, `exit_span=48`, `warmup_bars` = `max(entry_span,
  exit_span)` — either channel can be the longer one.
- **Note**: every channel is `.rolling(n).max().shift(1)`. The `shift(1)` is load-bearing
  but *not* a lookahead guard — bar *t*'s own high is known at bar *t*'s close, so the
  poison probe passes without it. What it prevents is degeneracy: `high >= close` within a
  bar, so an unshifted channel makes `close > entry_high` unsatisfiable and the strategy
  never trades at all.
- **Known issue — `exit_span >= entry_span` makes `exit_span` inert, with shorts on.**
  `long_exits` is `close < exit_low` and `short_entries` is `close < entry_low`; a wider
  exit channel means `exit_low <= entry_low`, so every bar that trips the exit also trips
  the reversal, and an opposite entry outranks a same-side exit. Measured on the
  15,128-bar BTC perp 4h frame: **zero** bars where a long exit fires without a short
  entry beside it, and `exit_span` 20/40/80 produce bit-identical positions at
  `entry_span=20`. Only `exit_span < entry_span` — the Turtle configuration this is
  modelled on — is a live parameter. With `--no-allow-shorts` there is no reversal to
  outrank the exit and the channel matters again. Pinned by
  `tests/test_sweep.py::test_donchians_exit_channel_is_inert_once_it_is_no_narrower_than_the_entry`.

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
- **Params**: `lookbacks=(24, 48, 96, 192)`, `entry_threshold=0.0`, `warmup_bars` =
  `max(max(lookbacks), 96)` — the volatility window is a lookback too, and binds whenever
  every horizon is shorter than it.
- **Note**: `rolling(...).std()` is not bit-reproducible across a cold start (pandas adds
  and removes observations one at a time, and the removals leave rounding residue).
  Measured at warmup 192 the score differs from the whole-history value on 195/300 bars —
  by at most 1.1e-15, against a smallest observed |score| of 5.9e-3. Signals are identical
  on every bar, which is why `warmup_bars` is the longest lookback and not a multiple.

### `--no-allow-shorts` gates the short entry only, never the long exit

`tsmom`, `ema_cross` and `multi_horizon` all wire `long_exits = short_state`, so the
trend flip is both "close the long" and "open the short". Only the second is optional:
`allow_shorts=False` zeroes `short_entries` and leaves `long_exits` on the raw flip.
Measured on 5,000 synthetic bars, all four baselines emit **identical** `long_exits` with
shorts on and off (2,889 for `tsmom`, 3,191 `ema_cross`, 275 `donchian`, 2,745
`multi_horizon`), so a long-only run under `opposite_signal_only` is a real long/cash
strategy rather than buy-and-hold.

This was wrong until 2026-08-03: the flip state itself was gated, which took the long
exit with it and dropped all three to **0** long exits under `--no-allow-shorts`.
`donchian` was never affected — its exits come from the reverse channel rather than the
entry state, which is the Turtle asymmetry paying off structurally.
`test_disabling_shorts_does_not_disable_the_long_exit` pins all four.

---

## state_machine_v1

The MDE R5 strategy. It runs the six-state lifecycle in `state/machine.py` over four R4
features and sizes each entry from the state it opened in.

It was designed as a hybrid — follow the top `strength` tercile, *fade* the middle one —
but **out of sample it earns its result as a trend follower**: the follow band produced
+102.4% of its test-half PnL on 94 trades and the fade band −2.4% on 65. The fade is kept
for now because it was a positive contributor in-sample (+14.6%) and one half is not
enough to delete a rule on, but it is not what the gate passed on.

- **Entry**: whenever the policy's signed target risk changes side. High `strength`
  follows `direction`; **mid `strength` fades it**; low `strength` is flat.
- **Why the middle band inverts**: R4 measured the IC of `direction` against the
  `[t+1, t+31]` return on 13,167 BTC/USDT perp 4h bars, by `strength` tercile — low
  +0.0022 (halves disagree in sign), **mid −0.1128** (halves −0.1201 / −0.1101), high
  +0.1314 (halves +0.1953 / +0.0621). Unconditional is +0.0385. A "trade when strength is
  high" threshold discards the band with the larger absolute IC.
- **Inputs**: `direction` raw, `strength` and `stability` as trailing ranks over
  `rank_window=480`, `crowding` raw. The ranks are why the thresholds are tercile
  boundaries: R4's conditioning was measured by tercile, and on the stored history the raw
  boundaries move 0.067/0.156 → 0.059/0.139 between halves.
- **Exits provided**: opposite side only, so run it under `--exit-mode
  opposite_signal_only`. The other modes layer engine exits on top of a machine that
  already owns its exits.
- **Sizing**: `position_size` carries the state's target risk (COMPRESSION/RESET 0,
  BREAKOUT 0.35, CONFIRMED 0.70, RIDING 1.00, EXHAUSTION 0.55), damped by extreme
  `crowding` on the paying side only. **The engine applies it on the entry bar and never
  again**, so a state change mid-position cannot resize — the taper is R6's job.
- **Params**: `rank_window=480`, `machine=StateMachine(enter_strength=2/3,
  exit_strength=1/3, min_dwell=4, cooldown=8, …)`, `warmup_bars=2160`.
- **`warmup_bars` is not the max over its features.** `direction` declares 1920, and at
  exactly 1920 the cold-start replay in `tests/test_strategy_metadata.py` disagrees with
  the whole-history run on 52–156 of 300 probed bars depending on seed, because the
  machine is a recursion on top of the features and has its own cold start. 60 more bars
  takes it to zero on every seed tried; the declared 240 is four times that.
- **Run** (the canonical R5 command — the `--start` is the first stored funding
  settlement, and a perp run refuses to start earlier because Binance settled nothing over
  the contract's first 40 hours):

  ```bash
  strategy-lab backtest --exchange binance --market-type perp --symbols BTC/USDT \
    --timeframe 4h --strategy state_machine_v1 --exit-mode opposite_signal_only \
    --start "2019-09-10 08:00:00" --cost-stress 1,2,3
  ```

- **R5 gate: passes.** Parameters chosen on the first 60% of the 15,118-bar BTC/USDT perp
  4h frame (54 configurations), the last 40% evaluated once. Out of sample it returns
  **+23.29% net of funding at Sharpe +0.938 and 8.24% max drawdown**, against the R0
  baseline `donchian` 40/10 at **−6.64% / +0.072 / 43.86%** over the identical 6,048 bars.
  The untuned R4 default also passes (+19.36% / +0.738 / 11.37%). Three limits belong with
  that: it wins on **risk, not return** (buy-and-hold is +85.78% over the same bars), it
  **loses to the best donchian cell chosen with the test half in hand** (40/40 at +112.57%
  / Sharpe +1.070), and its edge **does not survive 3× costs** (+23.29% → +11.34% →
  −0.61%), which its 3-bar median holding period makes unsurprising. Full tables in
  [the charter §9.2](docs/research/2026-08-03-market-dynamics-engine.md#92-r5-split-sample-gate--btcusdt-perp-4h);
  `tests/test_state_machine_gate.py` re-runs the out-of-sample comparison against the
  stored candles.

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
| `opposite_signal_only` | ✅ canonical — opposite state, unaffected by `--no-allow-shorts` | ✅ canonical — channel exit, unaffected by `--no-allow-shorts` |
| `trend_failure` | ✗ raises (no trend-failure series) | ✗ raises (no trend-failure series) |
| `setup_invalidation_stop` | ✗ raises (no setup stop) | ✗ raises (no setup stop) |
| `trend_structure` | long-only (raises if short entries exist — pass `--no-allow-shorts`); SMA40 via fallback | same; note it *replaces* the channel exit |

`state_machine_v1` owns its exits the way `trend_rider_v1` does, so the same warning
applies — anything layered on top fires *before* the machine decides the trend is over:

| `exit_mode` | `state_machine_v1` |
|---|---|
| `continuation_failure` (default) | ⚠ target side change OR N adverse closes — cuts rides the machine would still hold |
| `opposite_signal_only` | ✅ canonical — target side change only |
| `trend_failure` | ✗ raises (no trend-failure series) |
| `setup_invalidation_stop` | ✗ raises (no setup stop) |
| `trend_structure` | long-only (raises if short entries exist — pass `--no-allow-shorts`); ⚠ replaces the machine's own exits with an SMA40 break |

## Engine behavior worth remembering

- **Costs**: default `fees=0.0005` and `slippage=0.0005` per side, `cash=10_000`,
  `position_pct=0.95`. `--cost-stress 1,2,3` re-prices the same signals at 2× and 3×
  execution frictions and renders the comparison in the report.
- **Funding is charged on perps, and it is first-order.** `backtest` loads stored
  funding automatically for `--market-type perp` and *refuses to run* without it —
  a gross-of-carry perp number reads exactly like a net one, and BTC funding averages
  +11.65%/yr paid by longs. Pass `--no-funding` to opt out on purpose. It is charged as
  a discrete cash flow at each venue settlement, matched to the bar whose interval
  contains it (Binance stamps settlements up to 47 ms late; an equality match drops 43%
  of them), against the notional held *into* that bar. Cost stress never scales it —
  funding is a market rate, so tripling it models a different instrument.
- **A partial funding history is refused too**, not just an empty one: the stored series
  must span the candle window with no gap past 1.5× the contract's own measured cadence.
  A missing settlement is charged as zero and is indistinguishable from "the venue paid
  nothing". This bites on real data — BTC/USDT perp klines start `2019-09-08 16:00` but
  the venue's first funding settlement is `2019-09-10 08:00`, so the canonical 4h run
  needs `--start "2019-09-10 08:00:00"`.
- **A funded run's `stats.json` names the curve behind every path statistic.** Funding
  settles outside `Portfolio.from_signals`, so vectorbt's drawdown and Sharpe describe a
  book that never paid carry. Each path statistic (total return, end value, max drawdown
  and its duration, annualized return and volatility, Sharpe, Sortino, Calmar, Omega) is
  therefore emitted twice — `X (gross of funding)` and `X (net of funding)` — and never
  bare; `Net Return [%]` is the net total return under its established name, and
  `Funding Paid` is the total. Trade statistics are unsplit: funding is carry on the
  book, not a cost attributable to any trade. `equity_curve.csv` is the net curve and
  `funding.csv` is the per-settlement audit trail. When funding does *not* apply,
  `stats.json`, `trades.csv` and `equity_curve.csv` are byte-identical to a pre-costs
  run — but only those three. `costs.json` is always written, `config.json` always
  carries `cost_model` / `cost_stress` / `funding_applied` / `funding_settlements`, and
  the report always renders its Costs section.
- **Sizing is non-compounding**: entry shares = initial cash × `position_pct` × scale ÷
  close. Sizes are anchored to *initial* cash, never to current equity.
- **`--size-mode` chooses where that scale comes from.** `fixed` (default) uses whatever
  the strategy supplied, and is bit-for-bit the pre-sizing behaviour. `vol-scaled-entry`
  sets it to `--vol-target ÷ realized volatility`, clipped to `[0, --max-weight]`, so a
  violent regime is *entered* smaller. `--max-weight` is further clipped to
  `1 ÷ --position-pct` — an entry is sized as `cash × position_pct × weight` and the book
  has no leverage, so anything above that cannot be filled — with a warning naming both
  numbers and `max_weight_effective` in `config.json`. At the default 95% deployment the
  advertised `--max-weight 2.0` is really **1.053**.
- **`vol-scaled-entry` scales the entry and never retargets an open position** — the name
  is deliberate, and it is *not* volatility targeting. `Portfolio.from_signals` defaults to
  `accumulate=False`, so a repeated same-direction entry is ignored while a position is
  open and the per-bar weight series is consumed on exactly one bar per position: the one
  that opens it. A position held from a calm regime into a violent one carries its
  calm-regime notional the whole way, and realized risk is therefore *not* held constant.
  Continuous rebalancing needs order-level control `from_signals` cannot express; it is
  **R6**, where the engine moves to `from_orders` or a custom simulator (charter Q4).
  `tests/test_sizing.py::test_only_the_entry_bar_weight_lands_and_a_later_one_never_resizes`
  pins the real behaviour. The withdrawn `vol-target` spelling is **rejected, not aliased**.
- Two more things to know before reading a
  `vol-scaled-entry` run: the estimator is an EWM (span 96) that decays its seed rather than dropping it, so
  weights need roughly **20× span ≈ 1,900 bars** to converge and a shorter frame quietly
  under-trades its early bars; and volatility is annualized from *calendar* bars per
  timeframe, which is exact for 24/7 crypto and overstates the bar count (so understates
  the weight) on an instrument that closes. Combining it with a strategy that already sets
  `position_size` — only `trend_rider_v1_deepseek_v4_pro` — is **rejected**, not stacked:
  its ATR scale is itself an inverse-vol weight, so multiplying would size on `1/vol²` and
  hold neither target.
- **Data**: Yahoo OHLC is rescaled by adj close, so ETF series are dividend-adjusted
  (total-return-like). Crypto candles are raw exchange OHLCV.
- **Timeframe strings are identity**: candles are keyed by the literal timeframe string —
  `1w` and `1wk` are *separate datasets* (both exist in past reports). Pick one spelling
  per instrument and stick to it.
- **Reports**: every run writes `reports/<UTC>_<exchange>_<market>_<symbol>_<tf>_<strategy>/`
  with `config.json` (full parameter snapshot), `stats.json`, `trades.csv`,
  `equity_curve.csv`, `costs.json`, `plot.html`, plus `funding.csv` when funding
  applies. `config.json` is the reproducibility boundary; `costs.json` is the
  gross → fees → slippage → funding → size effect → net breakdown at every stress level.
  **Gross is a second simulation priced at zero**, not net with the costs added back:
  worse fills buy less size in a cash-constrained book, so the two differ. `size_effect`
  is the P&L the shrunken book never earned — the term that closes the waterfall — and
  gross is identical across stress levels because scaling a zero rate changes nothing.

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
