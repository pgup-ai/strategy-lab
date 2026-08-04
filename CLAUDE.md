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
strategy-lab features --exchange binance --market-type perp --symbol BTC/USDT \
  --timeframe 4h --horizons 1,6,30 --start 2019-09-10T08:00:00
strategy-lab backtest --exchange binance --market-type perp --symbols BTC/USDT \
  --timeframe 4h --strategy state_machine_v1 --exit-mode opposite_signal_only \
  --start "2019-09-10 08:00:00" --cost-stress 1,2,3   # the MDE R5 state machine
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

A fourth flow scores *market state* rather than a strategy. `StateFeature`
(`features/base.py`) has the same shape as `Strategy` — `name`, `version`,
`warmup_bars`, `compute(df) -> pd.Series` — with its own manual registry
(`features/registry.py`) so the poison probe covers it. `diagnose_features`
(`features/diagnostics.py`) reduces each feature to coverage, distribution,
lag-1 autocorrelation, turnover, forward-return IC at several horizons **and in
each half of the sample**, plus the pairwise correlations between features;
`features/diagnostics_report.py` and the `features` CLI command render and store
that. Signed features range −1..1, unsigned ones 0..1, and warmup rows are `NaN`
— a 0.0 there reads as "measured and neutral", a different claim from "not yet
measurable".

`state/` consumes that fourth flow. `StateMachine` (`state/machine.py`) walks a
feature frame through six states — compression → breakout → confirmed → riding →
exhaustion → reset — with hysteresis, minimum dwell and a post-reset cooldown
built into the transitions; `state/policy.py` maps state plus conditioning to a
signed target risk; `strategies/state_machine_v1.py` exposes the pair through the
ordinary `SignalSet` contract, so it runs on the existing engine, the replay
path, and both safety suites unchanged.

Key design decisions that span multiple files:

- **The state machine's conditioning is non-monotone, so a threshold rule is the
  wrong *shape*, not merely a suboptimal setting.** R4 measured `direction`'s IC
  against the `[t+1, t+31]` return by `strength` tercile: low +0.002 (halves
  disagree in sign), **mid −0.113** (halves −0.120 / −0.110), high +0.131.
  `state/policy.py` therefore follows `direction` in the top band, **fades** it
  in the middle one, and stands aside in the bottom — "trade when strength is
  high" would discard the band with the larger absolute IC. Two consequences a
  reader has to carry. The bands are **trailing ranks, not feature units**: a
  tercile is a rank statement and the raw boundaries move 0.067/0.156 →
  0.059/0.139 between halves, so a threshold in feature units is a differently
  sized bucket per era. And **the fade did not survive out of sample** — R5
  measured the follow band at **+100.0%** of test-half PnL on 54 trades and the
  fade at **+0.0%** (+0.30 currency units on +1,567.15 total, 19 trades, 31.6%
  win rate), against +70.8% on 53 trades / +29.2% on 24 in-sample — so the
  machine's *design* is a hybrid while its *measured* result is trend following.
  Do not describe it as either without saying which.
- **`position_size` is consumed on the bar that opens a position and never
  again.** `vbt.Portfolio.from_signals` defaults to `accumulate=False`; measured
  in R2 against the installed vectorbt, `size = [1,1,1,1,5,5,5,5]` with an entry
  every bar yields one order of size 1.0 and a position that never resizes. So
  `state_machine_v1`'s per-state target risk picks the size *at entry* — a later
  state can close the position but cannot scale it, and the charter's
  exhaustion → distribution taper is deferred to R6 where the continuous-exposure
  contract lands. Writing a taper against this engine ships a state machine whose
  defining behaviour is silently ignored, which is exactly what "volatility
  targeting" turned out to be before it was renamed `vol-scaled-entry`.
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
- **Every percentile, rank and z-score in `features/` is rolling or expanding —
  never full-sample.** This is the lookahead that has no `shift(-1)` to grep
  for. Measured on a 200-bar ramp poisoned downward from row 121,
  `series.rank(pct=True)` at row 120 moves 0.605 → 1.000 while the rolling form
  does not move at all: bar *t* changed its mind because of bars that had not
  happened yet. That is the shape of `_SubtleCheat` in `tests/test_lookahead.py`,
  and swapping `Energy`'s rolling percentile for a full-sample one passed the
  entire 412-test suite that preceded `tests/test_feature_lookahead.py`. So
  `rolling_percentile` and `rolling_zscore` (`features/base.py`) are the only two
  implementations, no feature hand-rolls either, and `mask_warmup` draws the
  warmup boundary in one place because `rolling(n)` declines to answer early
  while `ewm(adjust=False)` returns a converging number from bar zero.
- **A feature's forward return is anchored one bar *after* the feature's own.**
  `diagnostics.forward_return` measures `[t+1, t+1+h]`. The obvious
  `close[t+h] / close[t] - 1` contains no *return* from bar *t*, which is why it
  reads as safe, but it does contain bar *t*'s *price* as its denominator — and
  every feature here is a function of that same print, so anything rising with a
  high `close[t]` divides a high number into its own target and is paid for it.
  Measured on a random walk plus an i.i.d. print error, with a feature that
  predicts nothing: IC −0.53 anchored at `close[t]`, −0.01 anchored at
  `close[t+1]`. It is also the only convention that is executable. **Both
  half-sample ICs are reported beside the full-sample one**, because a feature
  that works in one half and not the other is a regime, not a signal, and the
  average is what hides that.
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

## Adding a state feature

Same shape, same two-place manual registration — new module under
`src/strategy_lab/features/`, then both `list_features()` and `get_feature()` in
`features/registry.py`. Registering is what enrols it in
`tests/test_feature_lookahead.py` (the poison probe, plus a check that the
feature is actually measurable at its own declared `warmup_bars` — otherwise the
probe compares NaN to NaN and passes without testing anything) and in the
`features` CLI command. Percentiles and z-scores go through `features/base.py`;
warmup rows are `NaN`, never 0.0. A feature that cannot be computed from a given
frame should **raise** rather than return a neutral value, as `Crowding` does
without funding.
