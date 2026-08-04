# Market Dynamics Engine — Research Program

**Status:** Charter written, not started
**Owner:** Jingbo
**Created:** 2026-08-03
**Last updated:** 2026-08-04

> **How to use this file.** It is the durable record for a multi-month research
> program: the thesis, the decisions, the hypotheses, what has been tried, and what
> was learned. It is *not* an implementation plan — each phase gets its own plan
> under `docs/plans/` when that phase starts, the way Phase 1a did.
>
> Three sections are meant to be edited continuously: **§9 Progress Log**,
> **§10 Decision Log**, and the status column in **§6 Phase Roadmap**. Everything
> else changes only when the program's direction changes — and when it does, record
> *why* in §10 rather than silently rewriting the text.

---

## 1. Thesis

Original framing, preserved verbatim:

> 我的理念是：赚趋势的钱。趋势出现之前，震荡行情可以亏一些钱，但是趋势出现后，需要一直
> trend riding 直到又回到震荡。目前最大的问题是如何判断震荡。

**The reframe that matters:** the question is not "is the market ranging right now?"
Ranging is not a state — it is the *absence* of future directional persistence, and
the future is unknowable. The answerable question is:

> How much evidence is there, right now, that directional moves are persisting?

Nobody reliably predicts the next breakout. What works is letting the market prove
itself and then following, accepting many small losses to be positioned for the few
large moves:

```
range → false breakout → retest → confirmed breakout → scale in → hold → decay → range
```

This produces the classic trend-following return shape: many small losses punctuated
by rare large gains. The optimization target is therefore **not win rate**. It is:

```
                      average winner
    ─────────────────────────────────────────────  ×  (how long we hold real trends)
                      average loser
```

A 35–45% win rate is fine and expected.

### 1.1 The three questions the system must answer

Everything below serves these, in order:

1. **Regime detection** — when is it worth risking a probe?
2. **Trend confirmation** — when is it worth scaling up?
3. **Exit logic** — is this the trend ending, or a normal pullback?

### 1.2 What this is not

Explicitly out of scope, to stop the research space exploding:

- Predicting the next bar
- HFT, market making, arbitrage, options
- Single-stock selection
- End-to-end RL or deep neural networks
- Any model that cannot be explained after it loses money

---

## 2. Target architecture

The end state is not a strategy. It is a **state estimator that strategies consume**:

```
Market Data ──► Feature Layer ──► State Estimator ──► Policy Engine ──► Risk ──► Execution
                                        │
                                        └── strategies are consumers of state,
                                            not consumers of indicators
```

### 2.1 The state vector

Market state is continuous and multi-dimensional, not a `trend`/`range` label:

| Dimension | Question it answers | Candidate inputs |
|---|---|---|
| **Direction** | Which way, and how convincingly? | normalized EMA spread, rolling regression t-stat |
| **Strength** | How strong is the move? | ADX, directional efficiency, breakout distance |
| **Confidence** | Should we believe it? | volume/OI confirmation, funding sanity, cross-asset agreement |
| **Energy** | Is there fuel? | ATR/realized-vol percentile, volume percentile, range expansion |
| **Compression** | Is it coiling? | bandwidth percentile, vol percentile, **and its derivative** |
| **Persistence** | Will it continue? | rolling R², directional efficiency, sign autocorrelation |
| **Stability** | Is it clean or ragged? | slope variance, residual vol, MA-crossing count, wick ratios |
| **Participation** | Is anyone else here? | crypto: OI/taker flow/breadth · ETF: advance-decline, sector breadth |
| **Crowding** | Is it too popular? | crypto: funding/premium z-score · ETF: VIX term structure, skew |

Two states carry very different meaning and must not collapse into one number:

```
Strength = 80, Energy = 30   →  slow, steady, mature trend
Strength = 20, Energy = 95   →  violent two-way chop
```

### 2.2 Derivatives — the corrected version of the original idea

The original instinct was to take higher-order derivatives of *price* to detect
breakouts earlier. **Do not do that.** Price is noisy; each differentiation amplifies
the noise, and by the third order the signal is effectively random. This is a
standard result in signal processing.

The salvageable version: differentiate the **smoothed state variables**, not price.

```
S(t) = [Direction, Strength, Energy, Persistence, Stability, Participation, Crowding]

V(t) = S(t) − S(t−1)          velocity     — is the state changing?
A(t) = V(t) − V(t−1)          acceleration — is the change accelerating?
```

This keeps the "rate of change leads level" advantage without the noise blowup. The
interesting reads are the interaction terms, not any single derivative:

```
Strength ↑ + Energy ↑ + Participation ↑   →  trend forming
Strength ↑ + Stability ↓                  →  trend getting ragged
Price ↑   + OI ↓                          →  short covering, not new money
Strength ↓ + Crowding extreme             →  exhaustion
```

**Third order (jerk) is deferred indefinitely.** In control engineering it is only
used after heavy filtering; here a single tick would double it.

### 2.3 State dynamics

Three properties the estimator must have, which naive per-bar classification lacks:

- **Inertia.** Markets do not change personality every bar. `92% → 15% → 88%` is a
  broken estimator, not a real regime flip. State itself gets filtered.
- **Fuzziness.** Output probabilities, not labels: `trend 65% / breakout 28% / range 7%`.
- **Age.** A 3-bar-old trend and a 120-bar-old trend are different risks even at
  identical strength. Duration is part of the state.

### 2.4 Position sizing is part of the state machine, not bolted on

| State | Target risk |
|---|---|
| Compression | 0–5% |
| Energy building | 15% |
| Potential breakout (probe) | 25% |
| Confirmed trend | 40–70% |
| Trend riding | 70–100% |
| Exhaustion | 55% |
| Distribution | 20% |
| Failed / reset | 0% + cooldown |

---

## 3. Where this collides with the current repo

**This is the single most important section for planning purposes.**

Phase 1a (merged, PR #4) built an event-driven engine where a strategy is:

```python
generate_signals(df) -> SignalSet(long_entries, long_exits, short_entries, short_exits, ...)
```

Boolean entry/exit events. MDE emits a **continuous target exposure** that moves every
bar — there is no "entry", only a target that drifts. These are different vocabularies,
and the mismatch surfaces in three places simultaneously:

| Where | Problem |
|---|---|
| `strategies/base.py` | `SignalSet` has no continuous-exposure field |
| `storage/migrations.py` | `signals.side` CHECK is pinned to 4 discrete values |
| `backtests/engine.py` | `vectorbt.Portfolio.from_signals` is structurally boolean-driven |

**Consequence:** MDE forces the vectorbt "freeze, migrate, retire" decision
(design doc §9) earlier than planned. Continuous rebalancing needs `from_orders` or a
custom simulator. Budget for this — it is the largest single cost in the program.

### 3.1 What survives, what doesn't

| Asset | Under MDE |
|---|---|
| `market_candles`, Decimal discipline, migration guards | **Survives — more valuable** |
| `MarketDataFeed` / `ReplayFeed` | Survives unchanged |
| `BarBuffer` (full history) | Survives; the EWM warmup finding generalizes |
| `StrategyRunner` bar loop | Loop survives, extraction step does not |
| **`tests/test_lookahead.py`** | **Survives — becomes critical** |
| **`tests/test_replay_determinism.py`** | **Survives — becomes critical** |
| `signals` table | Needs a sibling table for target exposure |
| `backtests/engine.py` (vectorbt) | Does not survive |
| The 4 existing strategies | Untouched, keep the old contract |

The two bolded rows are why this stays in one repo. HMM forward probabilities and
Kalman filters are **recursive and stateful** — their value at bar *t* depends on all
prior history. That is exactly the class where a streaming implementation silently
diverges from a batch one. This repo already paid to learn that lesson: the EWM-200
warmup needed 4,000 bars, not 200, and a real exit signal flipped at the wrong value.
HMM convergence-from-seed is the same bug in a new hat.

Likewise lookahead: MDE multiplies the feature count by an order of magnitude
(funding, OI, basis, CVD, breadth, VIX, term structure), and every one is a fresh
chance to leak the future. The poison probe is contract-agnostic — it corrupts bars
after *t* and checks row *t* — so it works on a state estimator unchanged.

### 3.2 Repo decision

**One repo, two strategy contracts.** The existing four strategies keep
`generate_signals -> SignalSet`. MDE gets a second contract when it needs one. Both
run through the same feed, buffer, and runner.

Rejected: a separate research repo. It would duplicate the data layer and both
validation gates — the expensive, hard-won parts — and it puts the verification
harness in a different repo from the code it verifies, which is the arrangement most
likely to let them drift.

**When to revisit:** when live trading needs stability while research churns. The
answer then is a versioned package boundary (`strategy-lab-core` as a dependency),
not a second repo. Phase 1c+ concern.

---

## 4. Blockers before research can start

| # | Blocker | Why it matters | Status |
|---|---|---|---|
| B1 | **No perp data.** The 83,348 BTC/USDT bars are **spot**. | Funding, OI, basis, liquidations only exist on perps — they carry most of hypotheses C1/C2/C5 | **Cleared by R1** — perp 4h backfilled to contract inception. |
| B2 | **Funding and OI have no schema.** | Funding is a settlement cash flow on its own schedule; OI is a point-in-time snapshot. Neither is a candle field. | **Cleared by R1** — `funding_rates` and `open_interest` tables. |
| B3 | **`ReplayFeed` is single-asset in practice.** `stream()` with two subscriptions yields all of A then all of B, not time-interleaved. | Every cross-sectional feature (breadth, RSP-confirms-SPY, portfolio vol targeting, GlobalRiskScore) is blocked on this | **Cleared by R3** — `stream()` is a k-way merge, `MarketClock` groups it into complete snapshots |

**B3 is the one the original 24-week plan under-weights.** Multi-asset time-aligned
iteration is a bigger architectural change than adding a state vector, and roughly
half the proposed features cannot be computed at all without it. Sequence it before
the state vector.

---

## 5. Universes and frequencies

Deliberately small at first. Expanding the universe multiplies survivorship,
listing/delisting, and liquidity-regime problems.

### Crypto (long/short, USDⓈ-M perps)

```
BTC → BTC+ETH → +SOL, BNB, XRP, DOGE → dynamic universe
```

Decision cadence 4h · slow state 1d · execution 1h. **Not** 5m/15m — not because low
frequency earns more, but because it keeps costs, data quality, and microstructure
noise controllable while the framework is unproven.

### US ETFs (long/cash)

Traded first: `SPY QQQ IWM TLT GLD`
State-only features: `DIA`, sectors (`XLK XLF XLE XLV XLI XLP XLY XLU`), `IEF DBC UUP`

Decision cadence 1d · slow state 1w · execution MOC or next open.

**Data caveat:** ETFs need total-return series. Adjusted close is often split-adjusted
only, not dividend-adjusted — do not assume `adj_close` is total return.

---

## 6. Phase roadmap

Each phase gets its own `docs/plans/` implementation plan when it starts. A phase does
not begin until the previous gate passes.

| # | Phase | Gate to pass | Needs | Status |
|---|---|---|---|---|
| **R0** | **Baselines** — TSMOM, EMA cross, Donchian, multi-horizon ensemble | Reproducible baseline on existing data; broad stable parameter regions, not one magic setting | Nothing new | **Done** — gate failed at 15m (turnover), passed at 4h once R1 landed the data: donchian 16/16 positive cells. Five progress-log corrections apply; read them before quoting a figure. |
| R1 | Perp + funding/OI data layer | Funding applied at real settlement times in PnL; no lookahead in availability timing | B1, B2 | **Done** — BTC/ETH perp 4h to 2019, full funding, zero gaps. OI forward-only (C1 blocked). |
| R2 | Cost + portfolio layer | Survives 2× and 3× cost stress | R0, R1 | **Done** — donchian nets +260.2% (1×), +113.6% (2×), +26.6% (3×). Beats the perp, loses to spot. |
| R3 | Multi-asset feed | Time-interleaved multi-symbol streaming; determinism test still passes | B3 | **Done** |
| R4 | State features v1 | Each feature has a univariate diagnostic; no feature dumped in unexamined | R3 | **NEXT** |
| R5 | Rule-based state machine | Beats R0 baseline out-of-sample, with hysteresis + dwell time + cooldown | R4 | — |
| R6 | Continuous-exposure contract | Second strategy contract; vectorbt replaced for this path | R5 | — |
| R7 | Meta-model (logistic) | Improves position sizing, not direction | R6 | — |
| R8 | HMM / Kalman | Beats R5 rules out-of-sample by enough to justify the complexity | R7 | — |
| R9 | Walk-forward + robustness | Purging, embargo, deflated Sharpe, parameter perturbation, feature dropout | R8 | — |
| R10 | Paper trading | Research and live code paths produce identical output | R9 | — |
| R11 | Canary (small size) | Expected vs actual fills, costs, state transitions all match | R10 | — |

**Model progression order is strict.** Rules → logistic meta → Gaussian HMM →
Student-t HMM → hidden semi-Markov → Kalman → ensemble. No neural networks.

---

## 7. Hypothesis register

Each is a falsifiable claim with a planned test. Record the result inline when run.

### Foundational

| ID | Hypothesis | Result |
|---|---|---|
| H1 | Trend persists: past risk-adjusted direction has limited but exploitable predictive power | — |
| H2 | Trend opportunity is state-dependent: the same breakout is worth more after compression / with participation | — |
| H3 | The state model mainly improves **sizing and tail risk**, not per-trade win rate | — |

**H3 matters for evaluation.** Do not judge the regime engine on Sharpe alone — check
whipsaw cost, average holding period, winner/loser ratio, and drawdown.

### Crypto

| ID | Hypothesis | Result |
|---|---|---|
| C1 | OI direction confirms trend quality (price↑/OI↑ ≠ price↑/OI↓) | **BLOCKED — not backtestable.** Binance serves only ~30 days of OI history (`startTime` 40d back → `-1130 parameter 'startTime' is invalid`, measured 2026-08-03 and re-confirmed after R1 landed). Perp klines and funding go back to 2019; OI does not. Options: accumulate forward from today and revisit in ~1 year, buy historical OI, or drop C1. Do **not** substitute a 30-day sample for a historical study. **Forward collection started 2026-08-03** — `strategy-lab fetch-open-interest --symbol BTC/USDT --period 4h`, 186 snapshots covering 2026-07-04 → 2026-08-03. The command takes no `--since` and refuses an out-of-window request rather than clamping it, so the 30-day sample cannot be mistaken for history. Schedule it to keep accumulating; C1 becomes answerable around 2027-08. |
| C2 | Funding is a *confirmation* variable at moderate levels and a *contrarian* one only at extremes | — |
| C3 | Breakouts after volatility compression outperform ordinary breakouts | — |
| C4 | Cross-coin breadth leads individual-coin trend confirmation | — |
| C5 | Long and short need asymmetric parameters (liquidation cascades make downside faster) | — |

### ETF

| ID | Hypothesis | Result |
|---|---|---|
| E1 | Breadth confirmation improves breakout quality | — |
| E2 | Overnight and intraday returns represent different regimes | — |
| E3 | Rates regime changes equity trend persistence | — |
| E4 | Graduated de-risking beats binary exit for long-only | — |

---

## 8. Standing methodology rules

Carried from Phase 1a, where each was learned the hard way.

1. **Mutation-test every test.** Break the implementation deliberately; if the suite
   stays green the test is decorative. Phase 1a shipped four decorative tests and one
   that had silently degraded to asserting `[] == []`.
2. **Measure, don't assume.** The lookahead probe missed real bugs on 32.5% of seeds
   at the length originally specified. The migration SQL corrupted 14,700 rows. Both
   looked correct.
3. **Warmup is measured, not declared.** `ewm(adjust=False)` needed ~20× span. HMM and
   Kalman will need their own measured values.
4. **Non-vacuity guards on every comparison test.** Assert the sample is non-empty, or
   a passing test may be testing nothing.
5. **No random train/test splits.** Time-ordered walk-forward, with purging and embargo
   when labels overlap.
6. **Record failed experiments.** Testing 100 features × 20 windows × 10 thresholds
   guarantees a pretty accident. Track the count; use deflated Sharpe.
7. **Every added layer must justify its complexity.** If the HMM moves Sharpe 1.02 →
   1.05 but is unstable across retraining, delete it.

### Labeling

Use triple-barrier labels (volatility-scaled profit target, stop, time limit) rather
than next-bar direction. For this thesis specifically, consider a **trend-quality**
label — net move divided by path length over the horizon — since the goal is capturing
clean sustained moves, not calling direction.

### Known trap: HMM label switching

State indices are arbitrary and permute across retraining. Map states by their
*statistical signature* (mean return, vol, autocorrelation, duration), never by index.
`signals.strategy_version` already exists as the hook: a retrained model is a new
version, signals are append-only, and diffing versions over the same bars is exactly
the label-stability check.

---

## 9. Progress log

Newest first. One entry per meaningful step — what was done, what was learned, what changed.

| Date | Entry |
|---|---|
| 2026-08-04 | **R3 complete. B3 is cleared: the feed is a k-way merge and cross-sectional features are computable at all.** `ReplayFeed.stream()` merges subscriptions on `(ts_event_ms, candle.key)` — the candle key being `(instrument, timeframe)`, because one symbol subscribed at two timeframes ties with itself — lazily, so a large universe is never materialized, and totally ordered, so the interleaving is identical on every run rather than dependent on subscription order. `MarketClock` groups that stream into `MarketSnapshot`s under one causal rule: **a timestamp is complete only once an event with a *later* timestamp arrives.** Looking ahead to check would be the exact lookahead this repo runs two suites to prevent, and it is not available live, so replay is bound by the same rule. Three deliberate consequences: cross-sectional work **lags one bar** (bar *t* is dispatched when the first event with a *later timestamp* lands — not literally *t+1*, which a sparse or mixed-timeframe feed can skip), the final timestamp needs an explicit **`flush()`**, and a snapshot holds **only instruments that have a bar at that time** — absent is never unchanged. `MultiAssetRunner` defers dispatch to snapshot completion, so a strategy evaluating bar *t* reads the whole cross-section *at t* rather than a stale one at *t−1*; it keeps one full-history `BarBuffer` per instrument and delegates each traded one to a `StrategyRunner` rather than copying its extraction. **Its one-instrument output is identical to `StrategyRunner`'s, signal for signal, on 400 bars of `donchian`**, and `test_replay_determinism.py` passes unchanged. **Measured on the stored BTC + ETH perp 4h: 15,128 snapshots, mean breadth 0.512, 477 partial universes** — 6,219 both-up, 5,876 both-down, 2,556 split. The 477 are exactly the stretch where BTC had listed and ETH had not (2019-09-08 → 2019-11-27), so the partial-universe path is exercised at real scale, not only on fixtures. **Mixed timeframes are the trap this phase nearly shipped:** a 1d close meets a 4h close only at day boundaries, so a 4h+1d universe yields **10 of 12 snapshots holding a single instrument** (measured), and breadth divided by instruments-present returns a well-formed number that is really just that one instrument's direction. `breadth` therefore **refuses** a universe below `min_instruments` (default 2), for the same reason it already refuses an empty snapshot — collapsing "no cross-section" into a number is how a halted or single-listed session reads as a signal. **Two of the plan's own tests could not fail as written; mutation testing found both.** The per-instrument-buffer test asserted only lengths, but a single shared buffer *also* reports 400 — both instruments carry the same timestamps and `BarBuffer` overwrites a repeat — so it now asserts contents. The unknown-instrument test matched `ETH/USDT`, which a bare `dict` KeyError also contains through the `InstrumentId` repr, so deleting the guard entirely still passed; it now matches the canonical `binance:perp:ETH/USDT` key that only the guard formats. **Review follow-up: the identity was still one field short.** `MarketSnapshot` keyed by `InstrumentId` and the merge tie-break sorted on it, so a symbol subscribed at 4h and 1d collided at every day boundary — measured, **2 of 14 bars never reached a snapshot**. `CandleId` now carries `(instrument, timeframe)` through the merge key, the snapshot, and `ReplayFeed.frames`; `MultiAssetRunner` stays instrument-keyed and **rejects** a bar at another timeframe rather than absorbing it. The BTC+ETH 4h figures above are unchanged (both are 4h, so nothing collided) and re-measured identical. 383 tests. |
| 2026-08-04 | **CORRECTION to the vol-scaled-entry figures above — the early weights were cold-start estimates, and dropping them makes the result slightly better.** `ewm().std()` returns a finite number from the second observation, so `volatility_target_weights` handed back a full per-bar series whose leading values were arbitrary rather than measured. Measured on the canonical frame at the production span of 96: the cold-start weight peaks at **1.551** and sits at or above the **1.053** the book can actually fill on **104 of the 1,920** warmup bars, against a converged mean of 0.644 — biased high, because an `adjust=False` variance is seeded from almost nothing and climbs, so early volatility reads low and the position it implies reads large. The practical symptom was that moving `--start` changed the sizing of entries over market the two runs had in common. The estimator now declares its own warmup — **20× span**, the multiple this repo already measured for `ewm(adjust=False)` — weights inside it are zero, the engine masks entries through whichever of the strategy's warmup and the estimator's is deeper, and a frame too short for the estimator is **refused** rather than silently under-traded. `--vol-span` is exposed on the CLI because it now sets the run's warmup. **Re-measured, same command (BTC/USDT perp 4h, `donchian`, `opposite_signal_only`, from 2019-09-10 08:00, 30% target / cap 2.0): `fixed` is unchanged to every published digit (max drawdown 39.56% net / 32.18% gross, Sharpe 0.632 / 0.757); `vol-scaled-entry` moves to max drawdown 21.04% → 21.80% net and 17.85% → 18.66% gross, Sharpe 0.830 → 0.884 net and 0.977 → 1.037 gross.** Net return 160.5% → 153.0%, funding paid 3,358 → 2,947, 96 → 86 trades — the ten dropped trades are exactly the ones the cold estimate would have sized. **The conclusion strengthens rather than weakens.** The published pair also compared runs covering different tradeable bars, since `fixed` masks 96 and `vol-scaled-entry` now masks 1,920; **like-for-like over identical bars (both masked at 1,920, 86 trades each) the pair is 40.25% → 21.80% drawdown and Sharpe 0.719 → 0.884 net of funding** — drawdown −45.8% relative and Sharpe +23.0%, against the −46.8% / +31.3% published, so part of the published Sharpe gain was the fixed run's extra cold-start bars rather than the sizing. 354 tests. |
| 2026-08-04 | **`run_backtest` never masked the warmup it declares. No published figure moves; two of the four original strategies do.** The sweep slices its returns at `warmup_bars` and `StrategyRunner` emits nothing until the buffer is past it, but the engine read the field only to ignore it — so a strategy traded on indicators it declares have not converged. Magnitude on the canonical 15,118-bar perp frame: `donchian` masks 96 bars (**0.64%**) but `ema_cross` masks 3,840 (**25.4%**), a quarter of a run on seed-dependent EMAs. `ema_cross` has never been run through the engine, so nothing published depends on it. Measured on the four original strategies' canonical `STRATEGIES.md` commands: `trend_following_deepseek_v4` (SPY 1w) and `trend_rider_v1_deepseek_v4_pro` (SPY/QQQ/SMH/XLF/XLK 1w) are **byte-identical** — their SMA40 is NaN through the masked prefix, so nothing could have fired there; `turnaround_v2` on BTC 15m moves −27.58% → −27.33% (47 → 46 trades) and `turnaround_v1` −100.03% → −102.02% (588 → 666 trades — it exhausts the book either way, and removing the masked trades only changes when it does, which is why the count rises). **Nothing became untradeable**, so no research-scope decision was needed. `donchian` on the perp is likewise unchanged, for the same reason as the ETF pair: `rolling(96)` is NaN through its own warmup. A frame that is entirely warmup is now refused rather than reported as a flat curve, matching the sweep, and the masked prefix is written to `config.json` as `warmup_bars`. |
| 2026-08-04 | **CORRECTION to the R0 gate's `donchian` surface — the sweep held two books where the engine holds one net position. The gate still passes and the headline cell does not move; a third of the surface was reporting a position the engine would never have held.** `positions_from_signals` summed an independent long book and short book, but `vbt.Portfolio.from_signals` keeps **one net position** and *reverses* on an opposite entry (`upon_opposite_entry` defaults to `ReverseReduce`, which with `accumulate=False` reaches `signals_to_size_nb`'s reverse branch). Where an opposite entry fires before the same-side exit, two books cancel to flat — and then stay wrong, because the stale side keeps its holding until its own exit finally arrives. Measured on 40 random four-channel signal frames of 400 bars, the old model disagreed with a live `from_signals` on **6,410 of 16,000 bars (40.1%)**; the sweep is deliberately vectorbt-free, so agreement is now *measured* against the engine rather than reasoned about. **Corrected gate: `donchian` stability +0.573 → +0.572, still 16/16 cells positive, best cell still 40/10 at SR 0.88 / +663.5% gross / 515 trades — unchanged to every published digit. `tsmom` is unchanged in full (+0.386, 5/5, best `lookback=160` at SR 0.85 / +854.2% / 492 trades)**, because its `long_exits` and `short_entries` are the *same series*, so a reversal and a flatten always coincide and both models agree bar for bar. **The three cells with `exit_span > entry_span` all moved**: (20,40) SR 0.801 → 0.673 and +586.3% → +361.6% gross, drawdown −61.9% → −81.5%, 358 → 279 trades; (20,80) SR 0.711 → 0.673, +350.3% → +361.6%, −59.3% → −81.5%, 278 → 279 trades; (40,80) SR 0.478 → 0.550, +117.0% → +179.4%, −68.6% → −73.8%, 212 → 160 trades. **The finding underneath is a degeneracy in `donchian` itself, not in the sweep: with shorts on, `exit_span` is inert once `exit_span >= entry_span`.** `long_exits` is `close < exit_low` and `short_entries` is `close < entry_low`, so a wider exit channel means every bar that trips the exit also trips the reversal, which outranks it — measured on the 15,128-bar frame, **zero** bars where a long exit fires without a short entry beside it, and `exit_span` 20/40/80 give bit-identical positions at `entry_span=20`. So **5 of the gate's 16 cells are now exact duplicates** of another cell, and the surface's apparent breadth along the `exit_span` axis was partly fictional above the diagonal. Only `exit_span < entry_span` — the Turtle configuration the strategy is modelled on — is a live parameter, and with `--no-allow-shorts` the channel is live again because there is no reversal to outrank it. Recorded in STRATEGIES.md and pinned by test. **A future grid should sweep only the lower triangle**, or the stability score is diluted by duplicates. |
| 2026-08-04 | **CORRECTION to the *name*, not the numbers: what shipped is volatility-scaled **entry** sizing, never volatility targeting. The measured benefit is unchanged and survives in full.** `vbt.Portfolio.from_signals` defaults to `accumulate=False`, so a repeated same-direction entry signal is ignored while a position is open. Measured directly on vectorbt 1.0.0 — flat close, an entry on every bar, `size = [1,1,1,1,5,5,5,5]` — the result is **one order, size 1.0, at the first bar** and an assets path of `[1]×8`. The per-bar weight series is therefore consumed on exactly one bar per position: the one that *opens* it. A position held from a calm regime into a violent one carries its calm-regime notional the whole way, so realized risk is **not** held constant, which is precisely what "targeting" asserts. The mode is renamed `--size-mode vol-scaled-entry` (config key `size_mode`); the old `vol-target` spelling is **rejected, not aliased** — it never shipped outside this branch, and an alias would keep the misleading word reachable. **Re-measured under the corrected name on the same command (BTC/USDT perp 4h, `donchian`, `opposite_signal_only`, from 2019-09-10 08:00, 30% target / cap 2.0): max drawdown 39.56% → 21.04% and Sharpe 0.632 → 0.830 net of funding, 32.18% → 17.85% and 0.757 → 0.977 gross — identical to the entry below, to every published digit.** *(— the vol-scaled half is superseded, see 2026-08-04 above: the early weights were unconverged; corrected to 21.80% / 0.884 net and 18.66% / 1.037 gross. The `fixed` half stands.)* That identity is the point: those figures were always measuring entry-only scaling, so nothing needs withdrawing and only the claim attached to them changes. Funding paid still falls 5,377 → 3,358, net return 177.2% → 160.5%, 96 trades either way. **Declining to *open* a large position in a violent regime is a real effect and is the whole of it** — the remaining benefit is a regime filter on entry, not risk control on the book. True per-bar rebalancing needs order-level control `from_signals` cannot express and is **R6** (charter Q4: `from_orders` or a custom continuous-rebalance simulator); `backtests/sizing.py` says so at the point of use. `trend_rider_v1_deepseek_v4_pro`'s "ATR volatility targeting" is the **same misnomer** for the same reason and is relabelled in STRATEGIES.md. `tests/test_sizing.py::test_only_the_entry_bar_weight_lands_and_a_later_one_never_resizes` pins each fill to the weight at its own entry bar on a frame where the weight moves 7.5× during a holding period, and a second test pins vectorbt's `accumulate=False` default itself so an upgrade cannot silently invalidate the docstring. Superseded figures throughout this log are now marked inline where they appear, rather than only corrected in a later row. `fixed` sizing re-verified **byte-identical** (`stats.json`, `trades.csv`, `equity_curve.csv` sha256, 24/24 files) across all four original strategies. |
| 2026-08-04 | **CORRECTION to the vol-targeting figures below — they were measured on a curve the report does not plot. Vol targeting still helps, and by slightly more on the right one.** `pf.stats()` describes the simulated book, and funding settles *outside* `Portfolio.from_signals`, so `31.25% → 17.65%` drawdown and Sharpe `0.794 → 1.057` were **gross of funding** on an instrument where funding is the dominant cost — 5,377 on 10,000 of initial capital for the fixed run, over half of it. Every path statistic is now emitted per curve (`X (gross of funding)` / `X (net of funding)`), never bare. Re-measured on BTC/USDT perp 4h, `donchian`, `opposite_signal_only`, 30% target: **max drawdown 39.56% → 21.04% and Sharpe 0.632 → 0.830 net of funding**, against 32.18% → 17.85% and 0.757 → 0.977 gross of it. *(— the vol-scaled half superseded, see 2026-08-04 above: 21.80% / 0.884 net, 18.66% / 1.037 gross, and like-for-like the pair is 40.25% → 21.80% / 0.719 → 0.884.)* **Vol targeting's advantage is marginally larger on the net curve, not smaller** — drawdown −46.8% relative net vs −44.5% gross, Sharpe +31.3% vs +29.1% — and it has a second effect invisible gross: the smaller book carries less notional, so **funding paid falls 5,377 → 3,358 (−37.5%)**. Net return 177.2% → 160.5%; Sortino 0.931 → 1.239, Calmar 0.402 → 0.707, annualized volatility 30.9% → 18.8% (the 30% target is met by the fixed run and undershot by the targeted one, which is the cap below biting). **Two measurement changes are folded into the gross figures above and are separated here rather than left implied.** (1) `--max-weight 2.0` was never executable: an entry is sized as `cash × position_pct × weight` with no leverage, so at 95% deployment the ceiling is **1.053**, and the requested weight exceeded it on 14 of 527 entry bars. Capped, the published run becomes 17.63% / Sharpe 1.003 (from 17.65% / 1.057) — the cap costs Sharpe, not drawdown. (2) The window now starts **2019-09-10 08:00** rather than 2019-09-08 16:00, because the venue's own first BTC funding settlement is 2019-09-10 08:00 (confirmed against the live API) while klines start 40 hours earlier — those bars were carrying an implicit zero funding, and the coverage guard now refuses rather than absorbing it. On the untrimmed window the published pair reproduces **exactly** (31.248% / 0.7942), which is what confirms the published numbers were the gross ones. Separately, `gross_return_pct` is now a second zero-cost simulation rather than net plus the costs added back; on this run the two differ by 0.64 on 10,000 (donchian is flat 47% of bars, so its book is rarely cash-constrained), but the identity was an assumption and is now a measurement. `fixed` sizing remains byte-identical (`stats.json`, `trades.csv`, `equity_curve.csv` sha256) across all four original strategies. 328 tests. |
| 2026-08-04 | **CORRECTION to the vol-targeting figures below — they were never an engine run, and the wired flag gives different numbers in the same direction.** `backtests/sizing.py` shipped with no caller: the R2 gate's "sizing is volatility-targeted" was satisfied by a module nothing imported, so `52.3% → 32.1%` drawdown and `0.63 → 0.79` Sharpe came from a direct call, not from anything `backtest` could produce. Sizing is now reachable as `--size-mode vol-target` *(— renamed `vol-scaled-entry`, see 2026-08-04 above)* (with `--vol-target` and `--max-weight`), recorded in `config.json`. Re-measured through the engine on BTC/USDT perp 4h, `donchian`, `opposite_signal_only`, 30% target / cap 2.0: max drawdown **31.2% → 17.7%**, Sharpe **0.79 → 1.06**, total return 239.4% → 218.1% (net of funding and 1× costs, 185.2% → 183.5%) *(— every figure in this sentence superseded, see 2026-08-04 above: these were gross of funding)*, 96 trades either way — sizing changes size, not entries. The published pair reproduces almost exactly (**55.0% → 40.4%**, Sharpe 0.59 → 0.74) when the book is a compounding ±1 position gross of costs, which is what the sweep layer computes and what the original call must have used; the engine's book is **non-compounding** — entries are sized from *initial* cash — so it never grows the position that produces the deeper late drawdown. **Both measurements agree on the direction and rough size of the effect (drawdown −27% to −39% relative, Sharpe +25% to +33%); only the absolute levels are book-dependent, and the engine's are the tradeable ones.** `fixed` remains the default and was verified byte-identical (`stats.json`, `trades.csv`, `equity_curve.csv` sha256) across all four original strategies. |
| 2026-08-03 | **CORRECTION to the R0 gate trade counts — `SweepPoint.trades` counted pre-warmup transitions. Every other figure stands.** `_evaluate` sliced `returns` at the grid's shared warmup but counted position changes over the *whole* frame, so a cell's turnover included trades whose returns were excluded. The count is what a reader multiplies by a cost assumption to sanity-check a cell, so the error ran in the pessimistic direction: cells looked more expensive than the returns being reported. Now sliced at the same warmup. Corrected on the 4h gate: `donchian` best cell 40/10 **519 → 515 trades**; `tsmom` `lookback=20` **1,542 → 1,521**. `tsmom` `lookback=160` stays at **492** — a `pct_change(160)` cell is flat over exactly the bars the shared 160-bar warmup discards, the same reason its return was unchanged by the warmup correction below. **No return, Sharpe, drawdown or stability figure moves** — the defect was confined to the count. Also re-measured on the superseded 15m gate: `donchian` 384/192 holds at 301 (its own span *is* the grid's deepest), `tsmom` `lookback=24` 8,545 → 8,519. The `gross − trades × 10bp` net estimates built on these counts were already withdrawn two entries down; the corrected counts do not revive them. |
| 2026-08-03 | **CORRECTION to the R0 gate sweep figures — the sweep under-warmed its own larger cells. The gate still passes.** `warmup_bars` was a hardcoded constant on all four baselines, and `sweep_parameters` took the *template's* value for every cell — so the gate's `donchian` grid swept `entry_span` up to 160 while warming only 96 bars. Both are fixed: each baseline now derives `warmup_bars` from its configured spans in `__post_init__`, and the sweep evaluates every cell at the **deepest** cell's warmup (one sample for the whole surface, every cell converged). Corrected: `donchian` stability **+0.592 → +0.573** *(— superseded, see 2026-08-04 above: +0.572, after the sweep's position model was corrected to one net book)*, still 16/16 positive, best cell still 40/10 but **SR 0.91 → 0.88** and **gross +751.5% → +663.5%** (519 trades, unchanged *— 519 superseded, see 2026-08-03 trade-count correction above: 515*); `tsmom` **+0.407 → +0.386**, still 5/5, best still `lookback=160` at **SR 0.85 / +854.2% gross — both unchanged**, because a `pct_change(160)` cell is flat over exactly the 64 bars the deeper warmup removes. **For these two the damage was dilution, not corruption**: `rolling(n)` is NaN before *n* observations, so an under-warmed cell takes no position rather than a wrong one — verified, `donchian` 160/80 holds flat across every bar from 96 to 159. **No R2 figure moves**: `run_backtest` never reads `warmup_bars`, so the engine nets (+405.2% execution-only, +260.2% with funding) and the vol-targeting result (52.3% → 32.1% drawdown) stand as published *(— the vol-targeting pair is superseded twice, see 2026-08-04 above; the engine nets stand)*. The strategy where this *would* have corrupted rather than diluted is `ema_cross`, which was never in the gate: `ewm` emits values from bar 0, so an under-warmed cell scores a non-converged EMA silently. At the measured 20× multiple its budget caps `slow_span` at 756 on this 15,128-bar frame, and a grid reaching 800 now raises and names the cell instead. |
| 2026-08-03 | **CORRECTION to the R0 gate entry below — the "beats buy-and-hold" claim was wrong.** The net figures were computed as `gross − trades × 10bp`, subtracting percentage points from a *compounded* return. Costs are paid along the way and compound too. Re-run properly through the engine, donchian 40/10 on BTC perp 4h: gross **+751.5%** *(— compare the 2026-08-03 warmup correction above, which restates the 40/10 sweep gross as +663.5%; which book this figure came from is unresolved)*, net of execution **+405.2%**, net of execution **and funding +260.2%** (2× costs → +113.6%, 3× → +26.6%). Perp buy-and-hold is +199.4% (a perpetual long pays 367.7pp of funding over the window); **spot buy-and-hold is +567.1%**. So the rule beats holding the *same instrument* but **loses to holding spot** — it is not yet a reason to trade a perp. A separate sweep-level check at 20bp reproduces the direction (+444.4% gross → +223.0% net vs +635.3% buy-and-hold over the warmed window); absolute values differ from the engine run because of position-model and warmup differences, but the conclusion is identical. **The flat-subtraction formula was in the plan's sweep `_evaluate` and in the verification command — both are fixed.** Donchian survives funding only because it is flat 47.5% of bars and short 22.1%, and shorts *receive* funding. |
| 2026-08-03 | **R2 complete.** Funding modelled as discrete settlement cash flows; volatility targeting cuts max drawdown **52.3% → 32.1%** and lifts Sharpe 0.63 → 0.79 for −36pp of return *(— superseded twice, see 2026-08-04 above; and it is entry scaling, not targeting)*. All four original strategies verified **byte-identical** (stats, trades, and equity-curve sha256) when funding is absent. A perp backtest now refuses to run without funding unless `--no-funding` is passed explicitly, because a gross-of-carry perp number is indistinguishable from a net one. 311 tests. |
| 2026-08-03 | **R2 complete. The donchian edge survives funding at 1× costs, but the R0 gate's headline net number was wrong and must be withdrawn.** Funding is now charged as a discrete settlement cash flow (`backtests/costs.py`), matched to bars by containment so the 3,260 off-grid BTC settlements are not dropped. **Correction to the R0 entry below:** its "~+699.6% net" was computed as `751.5 − 519 trades × 10bp`, a flat subtraction of percentage points from a compounded total. Costs are paid on the way up and compound with everything else; done per bar the same cell nets **+555.9%** at 10bp round-trip and **+405.2%** at the engine's actual default (10bp/side = 20bp round-trip). The gate's "beats buy-and-hold" claim does not survive that correction even *before* funding — spot buy-and-hold over the warmed window is **+567.1%**. **With funding, donchian 40/10 nets +260.2% (1×), +113.6% (2×), +26.6% (3×)**; funding alone costs 145pp at 1×. It stays positive at every stress level and beats the same-instrument baseline — **a perpetual long pays 367.7pp of funding, turning +567.1% into +199.4%** — but it does not beat simply holding spot. The rule is only affordable because it is flat 47.5% of the time and short 22.1%, and shorts *receive* funding. Volatility targeting (`backtests/sizing.py`, 30% target, cap 2.0) cuts max drawdown **52.3% → 32.1%** and lifts Sharpe 0.63 → 0.79 for −36pp of return *(— superseded twice, see 2026-08-04 above; and it is entry scaling, not targeting)*. **Verdict: the edge is real but thin, and it is not yet a reason to trade a perp instead of holding spot.** |
| 2026-08-03 | **R0 GATE PASSES at 4h.** Re-run on the freshly backfilled BTC/USDT perp 4h series (15,128 bars, 2019-09 → 2026-08, buy-and-hold +514.8%). `donchian` 4×4 grid: stability **+0.592** *(— superseded twice, see 2026-08-03 warmup correction: +0.573, and 2026-08-04 net-position correction: +0.572)*, **16/16 cells positive** — a genuine plateau, not a spike. Best cell 40/10: SR 0.91 *(— superseded: 0.88)*, +751.5% gross *(— superseded: +663.5%)*, 519 trades *(— superseded, see 2026-08-03 trade-count correction: 515)*, **~+699.6% net** of 10bp round-trip, beating buy-and-hold *(— **withdrawn**, see 2026-08-03 below: flat subtraction from a compounded return; the corrected net is +405.2% and it does **not** beat buy-and-hold)*. `tsmom` 1×5: stability +0.407 *(— superseded: +0.386)*, 5/5 positive, best lookback=160 at SR 0.85 / ~+805% net *(— net **withdrawn**, same flat-subtraction defect)*. Contrast with the same gate at 15m: −0.029 stability, 2/16 positive, everything negative net. **The 15m failure was turnover, exactly as diagnosed** — moving to the charter's specified 4h cadence cuts trade count ~16× and the edge appears. Caveat: these numbers are gross of **funding**, measured at +11.65%/yr paid by longs on BTC, which is a first-order drag on a long-biased perp rule. R2 decides whether the edge survives it. Note `tsmom` lookback=20 still loses net (−77% *— **withdrawn**, flat subtraction*) at 1,542 trades *(— superseded, see 2026-08-03 trade-count correction: 1,521)* — turnover remains the binding constraint at the fast end. |
| 2026-08-03 | **R1 complete.** BTC perp 4h 15,128 bars (2019-09-08 →), ETH perp 4h 14,651 bars (2019-11-27 →), BTC funding 7,559 rows, ETH funding 7,325, BTC OI 186 rows (30d forward-only). **Zero gaps** — every consecutive diff exactly 4h. Both start dates are contract inception, not truncated requests. `market_candles` 103,841 → 133,620, purely additive; spot and equity rows unchanged. Three venue facts found by running against the live API: funding timestamps land up to **47 ms after** the 8h boundary (an equality match would drop 43% of BTC settlements), `markPrice` is `""` on all funding before 2023-10-31 (60% of history, stored NULL not zero), and mean funding is **+11.65%/yr BTC / +13.97%/yr ETH paid by longs**. |
| 2026-08-03 | **R1 data layer landed — the 4h series the R0 re-run needs now exists.** Backfilled from Binance USD-M: **BTC/USDT 4h, 15,128 bars, 2019-09-08 16:00 → 2026-08-03 20:00** and **ETH/USDT 4h, 14,651 bars, 2019-11-27 04:00 → 2026-08-03 20:00**, both stored under `market_type=perp`. **Zero gaps in either series** — every consecutive diff is exactly 4h, bar counts match the span exactly, no duplicates, no NaNs. Both start dates are the venue's own inception (asking from 2019-01-01 returns nothing earlier), not a truncated request. Funding: **BTC 7,559 rows 2019-09-10 → 2026-08-03**, **ETH 7,325 rows 2019-11-27 → 2026-08-03**, no missing rates. OI: 186 snapshots, 2026-07-04 → 2026-08-03 — 30 days, as C1 predicts, accumulating forward only. **Mean funding is +11.65%/yr (BTC) and +13.97%/yr (ETH) paid by longs**, which is a first-order drag on any long-biased perp trend rule and belongs in the R0 re-run's cost model, not as an afterthought. Three venue facts measured while backfilling, none documented: `markPrice` is `""` before 2023-10-31 (60% of history; `Decimal("")` raises and aborted the first run — stored NULL); funding timestamps run up to **47 ms past** the 8h boundary, never before it, so funding must be matched to the bar whose interval contains its timestamp, rather than by equality against a generated 8h range; and BTC/ETH have used 8h continuously since inception, but that is an observed property of those two contracts, not a rule — nothing in the repo hardcodes it. |
| 2026-08-03 | **R0 gate run and INCONCLUSIVE — not failed.** On BTC/USDT 15m spot (83,348 bars, 2024-01 → 2026-05) both sweeps are spikes, not plateaus: donchian stability −0.029 (2/16 cells positive, both on the grid boundary), tsmom +0.017 (2/5). Net of 10bp round-trip, everything loses; buy-and-hold returns +81.4% at Sharpe 0.76 and beats every gross cell. **Diagnosis: turnover, not absence of trend.** `tsmom` lookback=24 trades 8,545 times *(— superseded, see 2026-08-03 trade-count correction: 8,519)* in 83,348 bars — one trade per ~10 bars, ~850% of capital in fees. `donchian` 384/192 trades only 301 times and loses ~12% net. §5 of this charter specifies **4h** as the crypto decision cadence and explicitly warns against starting at 15m; the gate was run at 15m only because that is the sole stored series. **The gate is re-run at 4h once R1 lands** — that is running it at the specified frequency, not bypassing it. Secondary note: trend following underperforming buy-and-hold in a sustained one-directional bull is expected (see Hurst/Ooi/Pedersen), not a refutation. |
| 2026-08-03 | **R0 sweep run on 83,348 BTC/USDT 15m bars (2024-01-01 → 2026-05-18). No broad stable region on either baseline.** `donchian` 4×4 grid: stability **−0.029**, 2/16 cells positive, both in the `entry_span=384` corner — an edge-of-grid spike, not a plateau. `tsmom` 1×5 grid: stability **+0.017**, 2/5 positive (`lookback` 24 and 48, also the grid edge), collapsing to −0.92 Sharpe at 96. Both are **gross of costs**: at the engine's own default 10bp/side, the best `tsmom` cell goes from +78.9% to **−100%** (8,545 position changes *— superseded: 8,519*), and the best `donchian` cell from +25.5% to −16.8%. *(— both net figures **withdrawn**, see 2026-08-03 below: they came from the sweep's flat-subtraction `_evaluate`, since fixed.)* Buy-and-hold over the same window returned **+81.4% at Sharpe 0.76**, beating every gross cell. Conclusion: these baselines show **no edge over holding on this data**, so the floor the program is gated on is currently buy-and-hold, not a trend rule. |
| 2026-08-03 | Long-only exit collapse fixed: `tsmom`/`ema_cross`/`multi_horizon` derived `long_exits` from the same series they gated for `short_entries`, so `--no-allow-shorts` emitted 0 long exits (vs 2,889/3,191/2,745 with shorts on) — the long-only ETF half would have been buy-and-hold wearing a strategy's name. |
| 2026-08-03 | R0–R2 implementation plan written ([docs/plans/2026-08-03-mde-r0-r2.md](../plans/2026-08-03-mde-r0-r2.md)). Measured Binance history depth first: klines and funding reach 2019, **OI only ~30 days** → C1 moved to BLOCKED. |
| 2026-08-03 | Charter written. Repo decision: one repo, two contracts. Next step fixed as R0 baselines. |
| 2026-08-03 | Phase 1a merged (PR #4): event engine, replay determinism proven on 83,348 real bars, append-only signal store. |

---

## 10. Decision log

Decisions and their reasoning. Amend with a new row rather than editing history.

| # | Decision | Reasoning | Date |
|---|---|---|---|
| M1 | One repo, two strategy contracts | Data layer + both validation gates are the expensive assets and MDE needs them more than current strategies do | 2026-08-03 |
| M2 | Baselines (R0) before any state work | The whole program is gated on a baseline the state engine must beat; costs one week, needs zero new architecture | 2026-08-03 |
| M3 | Multi-asset feed (R3) before state features (R4) | ~Half the proposed features are cross-sectional and cannot be computed on a sequential feed | 2026-08-03 |
| M4 | Derivatives on smoothed state, never on price | Higher-order price differentiation amplifies noise to randomness | 2026-08-03 |
| M5 | Jerk / 3rd order deferred indefinitely | Unusable without heavy filtering | 2026-08-03 |
| M6 | Rules before ML | A rule state machine is explainable and is the honest baseline for any statistical model | 2026-08-03 |
| M7 | Backfill perp klines + funding; collect OI forward only | Measured, not assumed: Binance caps OI history at ~30 days while klines and funding reach 2019. Better to know C1 is blocked now than to discover it in month 4. | 2026-08-03 |
| M8 | Cost stress scales fees and slippage, never funding | Funding is a market rate, not an execution cost — multiplying it models a different market rather than a worse fill | 2026-08-03 |

### Carried from Phase 0 (design doc §11)

D1 perps · D2 Binance first · D3 Postgres · D4 NUMERIC in place · D5 native timeframe
subscription · D6 asyncio single loop · D7 macOS, OS-agnostic · D8 vectorbt freeze/
migrate/retire · D9 direct exchange clients, not ccxt · D10 specs in `docs/`

---

## 11. Open questions

Answer before the phase that depends on each.

| # | Question | Blocks |
|---|---|---|
| Q1 | Keep the existing 83k spot BTC series, or re-fetch as perp? (Different instruments, different series.) | R1 |
| Q2 | How far back should the perp backfill go? Binance USDⓈ-M starts ~2019-09. | R1 |
| Q3 | Target exposure: new table, or widen `signals.side`? Append-only + per-bar targets grows fast. | R6 |
| Q4 | Replace vectorbt with `from_orders`, or write a custom continuous-rebalance simulator? | R6 |
| Q5 | Does the ETF long-only exit go to cash, or rotate to TLT/GLD? (v1 assumes cash.) | R5 |

---

## 12. Inherited follow-ups from Phase 1a

Not blocking, but they will bite during this program.

| Item | Impact here |
|---|---|
| db-marked tests write to the production research database | Will worsen as research volume grows |
| `backfill()` returns `AsyncIterator[Bar]`, `prime()` takes a DataFrame | Blocks the live-warmup path |
| `Subscription.include_forming` declared but never read | Forming-bar handling is a real MDE concern |
| `uq_signals_identity` omits `market_type` | Spot + perp signals on the same bar collapse to one |
| `ruff` never enforces its own 100-char limit (22 violations) | Cosmetic |
| `trend_rider_v1` ATR sizing saturates on 15m crypto | Works on weekly SPY; not a bug, but a trap if reused |

---

## 13. Reading list

Read in order. The goal is the framework, not any single model.

**1 · Trend foundations** — establish that the edge exists before modeling regimes
- Moskowitz, Ooi & Pedersen — *Time Series Momentum*
- Hurst, Ooi & Pedersen — *A Century of Evidence on Trend-Following Investing*

**2 · Regime switching** — latent state, transition matrices, filtered vs smoothed
probabilities, state duration, label switching
- Hamilton (1989) — *Markov Regime Switching* (the origin of this whole line)
- HMM basics → hierarchical HMM → hidden semi-Markov → heavy-tailed emissions

**3 · State-space and signal processing** — the smoothing/lag tradeoff is the real subject
- Kalman filter, local linear trend model, particle filter
- Savitzky–Golay derivatives, wavelet denoising, change-point detection

**4 · Meta-labeling and validation** — research tooling, not alpha
- López de Prado — triple barrier, meta-labeling, purged CV, embargo, deflated Sharpe

**Institutions on this path** (usually under other names — "regime detection",
"adaptive trend following", "dynamic risk"): MAN AHL (most published, closest fit),
AQR (trend persistence, momentum decay), Winton (non-stationarity), Aspect (adaptive
trend), Two Sigma, Bridgewater (macro state machine).
