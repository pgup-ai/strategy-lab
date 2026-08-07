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
strategy-lab serve                 # serve the frozen reports/ with the candle-refresh API
strategy-lab browse                # the research browser: recomputed live, writes nothing
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

A fifth flow *looks at* the others rather than adding one. `api/analysis.py`
loads a stored frame, attaches funding by `backtests/funding_frame`'s rule,
dispatches on whichever registry holds the strategy, and returns candles, fills,
the per-bar state and feature values, and provenance; `api/models.py` refuses
any query parameter it does not recognise, by name; `api/app.py` serves that
plus the one page in `browser/page.py`, which inlines the same vendored chart
`backtests/report.py` does. `browse` runs it, on the loopback interface only.
`api/board.py` widens that to several instruments at once — `GET /api/board`
streams one newline-delimited row per (dataset, strategy) as each finishes,
cached on the last bar rather than on the clock, and **slices `build_analysis`
rather than deriving the same answer a cheaper way** (M36).

Key design decisions that span multiple files:

- **`serve` hosts the record; `browse` hosts a view, and the two must not
  merge.** A backtest writes a dated directory under `reports/` whose
  `plot.html` re-renders byte-identically from the run it froze — that is the
  reproducibility boundary, and `serve` is how you read it. The research browser
  recomputes from stored candles on every request and **persists nothing**: no
  report directory, no `signals` row, no schema. So it can show a strategy that
  was never run, over a range nobody backtested, and can never become the record
  of one that was. Two consequences. The browser is **not free to disagree** with
  a backtest, which is why its markers are *fills* off the engine's own
  `from_signals` call rather than the raw `SignalSet` — `accumulate=False`
  ignores a repeated same-direction entry, so signals would mark bars no backtest
  traded, and `tests/test_api_analysis.py` pins the payload against a real
  `trades.csv`. And **provenance is not a detail panel**: `crowding_measured`,
  the exit mode, warmup, the cost model and the frame's bounds are on the page at
  all times, because M20 was one strategy's number moving on a silently absent
  funding column, and a figure read off a chart without that context will
  eventually contradict the charter with no way to see why. **That rule now binds
  every view, not just the chart**: the board calls the same ~300–500 ms
  `build_analysis` a tile does not need, because a cheaper route would be a third
  answer free to drift from the other two, and the drift would surface as a tile
  quietly contradicting the chart it links to. Widening the board is bounded by
  caching, not by computing less — **parallelism does not help here**, measured
  at 1.10× on four threads, since the work is pandas and vectorbt under the GIL.

- **A perp refresh advances candles and funding together, because the coverage
  guard refuses the pair when they drift.** `refresh_candles` fetches bars up to
  the present; funding settles on the venue's own schedule, so refreshing only
  candles grows the window past the last stored settlement and
  `funding_coverage_gaps` then refuses the whole frame — the browser breaking
  the dataset it is showing, by being used. The guard is right and must not be
  loosened: R2 measured carry on this instrument at roughly the size of
  buy-and-hold, so a run charging zero across uncovered bars reports a gross
  number that reads like a net one. Three consequences. The top-up reaches back
  to **the earlier of the candle lookback and the last stored settlement** —
  five 15m bars is 75 minutes against an 8h interval, so a lookback-sized
  request steps straight over the gap it exists to close. A **small trailing gap
  is structural, not a fault**: the last bar's right edge sits up to one cadence
  past the final settlement, and the guard tolerates exactly that. And a
  **leading** gap can be permanent — BTC/USDT perp candles start 40h before the
  venue's first settlement, so `state_machine_v1` refuses that full frame
  forever, which is a fact about the venue rather than something to fetch. Any
  db-marked test on a real perp frame must therefore bound its own right edge
  with `db.funding.funding_span`, or an unrelated refresh reddens it.

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
  state can close the position but cannot scale it. R6 measured what that costs:
  **not one of `state_machine_v1`'s test-half entries is in `RIDING`**, so the
  largest position it ever opened is 52.3% of capital (trained) and 66.5%
  (default) against the **95%** its own `RIDING` row asks for, on a machine that
  spent 209 / 516 bars there. Entries do reach the `CONFIRMED` and `EXHAUSTION`
  rows; `RIDING` is the one they cannot, because it is only ever reached with a
  position already open — the lifecycle passes `BREAKOUT` and `CONFIRMED` first,
  both already carrying a non-zero target, and an entry needs a change of side.
  The position then freezes at whatever size its entry bar carried, so the row
  the policy sizes highest is the one row that never executes. A taper belongs
  on the continuous contract below, not here: writing one against this engine
  ships a state machine whose defining behaviour is silently ignored, which is
  exactly what "volatility targeting" turned out to be before it was renamed
  `vol-scaled-entry`.
- **Two strategy contracts coexist, and what picks between them is whether a
  strategy needs to resize a position it already holds.** `SignalSet` says
  *enter*, *exit* and *how big to start*. `TargetExposure`
  (`strategies/exposure.py`) carries a **level**: `target[t]` in −1..1 is the
  whole of what the book should hold over bar *t*, whatever it held over *t−1*,
  executed by `run_exposure_backtest` (`backtests/exposure_engine.py`) through
  `Portfolio.from_orders(size_type="targetvalue")` — measured, a 10-bar taper
  issues 6 orders there against `from_signals`' 1. Neither contract is migrating
  to the other: the four original strategies keep `SignalSet` and their
  byte-identical results of record, `state_machine_v1` keeps it so R5's published
  numbers do not move, and `state_machine_v2` is the same machine, policy,
  `STATE_TARGET_RISK` and features on the continuous one — having both is what
  made R6's comparison the collapse and nothing else. Three properties of the
  continuous path a reader has to carry. (1) **Warmup is a leading run of 0.0 and
  NaN is refused**, the exact inverse of the feature convention above, because
  `from_orders` reads NaN as "no order", i.e. *hold whatever you held* — the one
  reading a warmup row must never carry. (2) **Size is a currency value against
  initial cash** (`target × position_pct × cash`); `targetpercent` is a fraction
  of *current* equity and would silently compound, breaking the non-compounding
  rule below on the one path that was supposed to be comparable to the other.
  (3) **A target reaches the book only once it has moved `rebalance_threshold`
  (default 0.05) from the last target *submitted***, so between decisions the
  book holds a fixed quantity and its fraction of equity drifts with price, by
  design. Band 0.0 rebalances every bar, trimming winners and adding to losers —
  a mean-reversion overlay on a trend thesis — and it moves the result *before* a
  fee is charged (20,742.99 against 20,261.47, costless and funding-free), so it
  is a model choice and not a cost optimisation.
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
- **One known exception to that, and it is the data the paths carry rather than
  the code they run: `state_machine_v1` on a perp.** `crowding` reads a
  `funding_rate` column, which `backtest` and `sweep` attach on a perp
  (`cli._with_funding_column`) and the event path structurally cannot —
  `core.types.Bar` has no funding field and `BarBuffer` materializes
  `open/high/low/close/volume` and nothing else. So a replay of a perp range
  runs that feature at `NEUTRAL_CROWDING` and emits different signals from a
  backtest of the same range; **the backtest is the published one**, and the
  charter's R5 figures are its. Measured on BTC/USDT perp 4h over R5's test half,
  trained cell: +16.44% / Sharpe +0.801 crowding-neutral against +15.45% /
  +0.896 measured. The determinism suite does not catch this and is not broken —
  `synthetic_ohlcv` carries no funding, so it compares crowding-neutral against
  crowding-neutral, and a suite weakened to hide the gap would be worse than the
  gap. Closing it means carrying funding through `Bar`, `BarBuffer`, the feed and
  the storage schema, which is a phase and not a patch.
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
  what the migration fixed, on every re-fetch. **The `(38,18)` *scale* is a
  second and independent limit, and no test covers it**: a scale counts decimal
  *places* where a float64 needs up to 17 *significant digits*, so a value whose
  shortest round-trip decimal runs past **18 fractional places** is truncated.
  That is a condition on the digits, not on the magnitude: measured, `0.05`,
  `0.0625` and even `0.0052631578947368418` (18 places, 17 significant) all
  round-trip, while `1/480` (19 places) and `2.0682314349096398e-05` (21) do
  not. It therefore needs a small magnitude *and* many significant digits, which
  is why `bar_reasons` stores its 0..1 feature values as unconstrained `NUMERIC`
  (M34). `market_candles` still carries the scale, so a sub-0.01 price carrying
  full float64 precision would lose digits; the repo has never stored one, which
  is the only reason it has never shown.
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
it. A strategy on the continuous-exposure contract registers somewhere else —
see below.

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

`mask_warmup(values, warmup_bars=..., name=...)` takes the feature's own name,
with no default, so the negative-warmup refusal can say which feature declared
it — a helper that cannot name the offender sends the reader to the wrong file.
It is also the only guard that runs inside `compute`, and it does not reach a
feature that delegates: `Compression` returns `1 - Energy.compute(df)`, so its
own warmup never passes through, and the registry test is what covers it.

## Adding a continuous-exposure strategy

There are **three** manual registries, not two: `strategies/registry.py`,
`features/registry.py`, and `strategies/exposure_registry.py`. A strategy that
returns a `TargetExposure` goes in the third — new module under
`src/strategy_lab/strategies/`, then both `list_exposure_strategies()` and
`get_exposure_strategy()`.

**It is a third registry rather than a third entry in the first one because the
boolean suites cannot run an exposure strategy and would not say so.** Six
parametrized tests across `tests/test_lookahead.py`,
`tests/test_replay_determinism.py` and `tests/test_strategy_metadata.py` iterate
`list_strategies()`, and every one of them calls `generate_signals`, which an
exposure strategy does not have. Measured by mutation: an **empty**
`exposure_registry` silently **skips 4 parametrized tests and exits 0** — the
suite goes green by not running, which is the failure this repo mutation-tests
to prevent. Registering here is what enrols a strategy in
`tests/test_exposure_lookahead.py` (the poison probe, funding poisoned along with
the prices because `crowding` reads it, plus a check that the target actually
*moves* over the probe window — a target that is 0.0 everywhere compares equal to
itself as happily as NaN does), in `tests/test_exposure_determinism.py`
(whole-history vs streaming, a runner primed from mid-history, and target-level
equality on every bar rather than side-level), and in the cold-start warmup check
in `tests/test_strategy_metadata.py`.

Warmup rows are `0.0`, never `NaN` — the inverse of the feature convention, for
the reason in the design bullet above. There is no `ExitMode` on this path: a
target of 0.0 *is* the exit, so there is no exit-mode matrix to fill in.

Then add its row and section to [STRATEGIES.md](STRATEGIES.md), the same as step 5
of "Adding a strategy" — it is the source of truth for both contracts, and its
at-a-glance table carries a `Sizing` column precisely so a continuous strategy is
distinguishable from an entry-only one at a glance. Say which execution path the
strategy runs on, and put `rebalance_threshold` where a boolean strategy's exit
mode would go.
