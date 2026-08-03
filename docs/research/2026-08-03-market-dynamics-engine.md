# Market Dynamics Engine — Research Program

**Status:** Charter written, not started
**Owner:** Jingbo
**Created:** 2026-08-03
**Last updated:** 2026-08-03

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
| B1 | **No perp data.** The 83,348 BTC/USDT bars are **spot**. | Funding, OI, basis, liquidations only exist on perps — they carry most of hypotheses C1/C2/C5 | Open |
| B2 | **Funding and OI have no schema.** | Funding is a settlement cash flow on its own schedule; OI is a point-in-time snapshot. Neither is a candle field. | Open |
| B3 | **`ReplayFeed` is single-asset in practice.** `stream()` with two subscriptions yields all of A then all of B, not time-interleaved. | Every cross-sectional feature (breadth, RSP-confirms-SPY, portfolio vol targeting, GlobalRiskScore) is blocked on this | Open, documented + test-pinned |

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
| **R0** | **Baselines** — TSMOM, EMA cross, Donchian, multi-horizon ensemble | Reproducible baseline on existing data; broad stable parameter regions, not one magic setting | Nothing new | **NEXT** |
| R1 | Perp + funding/OI data layer | Funding applied at real settlement times in PnL; no lookahead in availability timing | B1, B2 | Blocked |
| R2 | Cost + portfolio layer | Survives 2× and 3× cost stress | R0, R1 | — |
| R3 | Multi-asset feed | Time-interleaved multi-symbol streaming; determinism test still passes | B3 | — |
| R4 | State features v1 | Each feature has a univariate diagnostic; no feature dumped in unexamined | R3 | — |
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
| C1 | OI direction confirms trend quality (price↑/OI↑ ≠ price↑/OI↓) | **BLOCKED — not backtestable.** Binance serves only ~30 days of OI history (`startTime` 40d back → `-1130 parameter 'startTime' is invalid`, measured 2026-08-03). Perp klines and funding go back to 2019; OI does not. Options: accumulate forward from today and revisit in ~1 year, buy historical OI, or drop C1. Do **not** substitute a 30-day sample for a historical study. |
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
| 2026-08-03 | **R0 gate run and INCONCLUSIVE — not failed.** On BTC/USDT 15m spot (83,348 bars, 2024-01 → 2026-05) both sweeps are spikes, not plateaus: donchian stability −0.029 (2/16 cells positive, both on the grid boundary), tsmom +0.017 (2/5). Net of 10bp round-trip, everything loses; buy-and-hold returns +81.4% at Sharpe 0.76 and beats every gross cell. **Diagnosis: turnover, not absence of trend.** `tsmom` lookback=24 trades 8,545 times in 83,348 bars — one trade per ~10 bars, ~850% of capital in fees. `donchian` 384/192 trades only 301 times and loses ~12% net. §5 of this charter specifies **4h** as the crypto decision cadence and explicitly warns against starting at 15m; the gate was run at 15m only because that is the sole stored series. **The gate is re-run at 4h once R1 lands** — that is running it at the specified frequency, not bypassing it. Secondary note: trend following underperforming buy-and-hold in a sustained one-directional bull is expected (see Hurst/Ooi/Pedersen), not a refutation. |
| 2026-08-03 | **R0 sweep run on 83,348 BTC/USDT 15m bars (2024-01-01 → 2026-05-18). No broad stable region on either baseline.** `donchian` 4×4 grid: stability **−0.029**, 2/16 cells positive, both in the `entry_span=384` corner — an edge-of-grid spike, not a plateau. `tsmom` 1×5 grid: stability **+0.017**, 2/5 positive (`lookback` 24 and 48, also the grid edge), collapsing to −0.92 Sharpe at 96. Both are **gross of costs**: at the engine's own default 10bp/side, the best `tsmom` cell goes from +78.9% to **−100%** (8,545 position changes), and the best `donchian` cell from +25.5% to −16.8%. Buy-and-hold over the same window returned **+81.4% at Sharpe 0.76**, beating every gross cell. Conclusion: these baselines show **no edge over holding on this data**, so the floor the program is gated on is currently buy-and-hold, not a trend rule. |
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
