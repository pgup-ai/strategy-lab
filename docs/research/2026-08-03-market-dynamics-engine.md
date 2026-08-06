# Market Dynamics Engine — Research Program

**Status:** R0–R6 complete, plus a second-asset replication of the R5/R6 protocol on ETH
([§9.4](#94-eth-replication-of-the-r5r6-protocol--ethusdt-perp-4h)). R7 (meta-model) is next
on paper and wants re-aiming; R9's robustness work and a read-only research browser are both
live alternatives — see **Q6**
**Owner:** Jingbo
**Created:** 2026-08-03
**Last updated:** 2026-08-05

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
| R4 | State features v1 | Each feature has a univariate diagnostic; no feature dumped in unexamined | R3 | **Done** — nine features, all diagnosed on 15,118 real bars; see the R4 table in §9. Confidence deferred. **Participation recommended for cutting**; every other feature keeps its place, several as conditioners rather than predictors. |
| R5 | Rule-based state machine | Beats R0 baseline out-of-sample, with hysteresis + dwell time + cooldown | R4 | **Done — gate passes**, re-measured after the bounded-exit fix (see the 2026-08-04 correction in §9). OOS Sharpe **+0.896** against the baseline's +0.072, max drawdown **4.67%** against 43.86%; the untuned default also passes at +0.746. 54 configurations on the training half, and `tests/test_state_machine_gate.py` re-derives that training winner rather than hardcoding it, so the gate fails if the surface stops selecting the published cell. That is a regression control, **not** a standing re-selection: research and execution hold the R5 cell fixed, which R9 later measured as the better policy (M22). **Replicated on a second asset — the method only** ([§9.4](#94-eth-replication-of-the-r5r6-protocol--ethusdt-perp-4h)): re-running the same 54-cell search on ETH's training half beats ETH's baseline out of sample (+0.868 against +0.785), but BTC's *cell* transfers only partially (+0.184) and **the untuned default fails outright (−0.563)**, so the "no selection discount" clause here is BTC-specific. **Wins on risk, not return** (+15.5% against buy-and-hold's +85.8%), loses to the test-half oracle donchian cell, but **now survives 3× costs** (+6.4%). The result is the follow band, not the mid-tercile fade. **Audited by R9** ([§9.5](#95-r9-walk-forward-and-robustness--btcusdt-perp-4h)): the published +0.896 is untouched and the cell holds up — it beats re-derivation across nine walk-forward folds — but the **search** that produced it is not distinguishable from luck at conventional confidence (DSR 0.7026), and `enter_strength=0.80` is a ridge whose neighbour dies at 3× costs. Quote the figure with M22 attached. See [§9.2](#92-r5-split-sample-gate--btcusdt-perp-4h). |
| R6 | Continuous-exposure contract | Second strategy contract; vectorbt replaced for this path | R5 | **Done — gate passes.** `TargetExposure` executes through `from_orders`, has its own determinism and lookahead proofs, and the four original strategies stay byte-identical. Retiring vectorbt was not required and is not in scope (Q4). **The taper measures ≈0** (+15.30 / −63.05 gross over 75/77 bars); what v2 buys is 1.7–1.8× v1's average exposure and the return scales with it, while Sharpe is flat. The finding is that `state_machine_v1` has never sized an entry for §2.4's **RIDING** row — see [§9.3](#93-r6-continuous-exposure-comparison--btcusdt-perp-4h). **Replicated on ETH** ([§9.4](#94-eth-replication-of-the-r5r6-protocol--ethusdt-perp-4h)): zero `RIDING` entries at all three configurations there too, so the mechanism is a property of the machine. The continuous contract is worth *more* on ETH than here — it adds Sharpe at every configuration instead of washing out. |
| R7 | **Chop/trend state diagnosis** — *reused slot; was "meta-model (logistic)", dropped under M24* | Chop is diagnosed **directly**, not inferred from a P&L number: every candidate estimator scored against a declared chop label on its own terms, half-sample ICs reported beside the full-sample one, before anything trades it | R9 | **Done — gate answered, and the answer sends the program back to [§2.1](#21-the-state-vector).** Every declared threshold evaluated: `COMPRESSION` **0/6** on BTC and **0/9** on ETH, the nine features **0/9** on BTC, the composite gate **0/3** and structurally so — it *is* the `strength` gate, since `direction_floor` admits 82% of bars. The one exception is `energy`/`compression`, which misses BTC's bar by 0.0094 and **clears on ETH** at −0.1521 with both halves agreeing — **and the machine does not read it**. R9's `enter_strength` lead closed: 0.80 is **interior**, not a boundary artifact (M26). Verdict is the pre-registered third row, with the ETH exception recorded; the constructive form is M25. *Superseded content, from when this slot was the logistic meta-model:* Pre-registered before any figure, as R9 and the ETH replication were. The dropped meta-model's reasoning is preserved verbatim below, because what a phase was and why it stopped being that is part of the record. *Superseded content:* — **wants re-aiming before it starts, and the ETH replication sharpens why**: R6 measured the earning difference in the *ramp* (holding more while a move is establishing), not the taper, and no per-state effect on the test half is distinguishable from zero (no t-stat above 1.77). A sizing model aimed at trimming mature trends would be aimed at the half that measured nothing. **ETH adds the stronger argument.** The untuned R4-default machine — same states, same policy, same `STATE_TARGET_RISK` — goes from **+0.746 on BTC to −0.563 on ETH's test half**, and the re-searched cell agrees with BTC on both strength thresholds while disagreeing on **both timing axes**. What fails to transfer is the configuration that *produces* the states, not the curve laid over them, so a sizing model fitted on BTC's states inherits that failure before it sizes anything. R6's null per-state effects said a taper is not worth modelling; ETH says the thing a sizing model would sit on top of is itself asset-specific. **R9 adds the third and sharpest argument, and it is about R7's *form* rather than its target.** A fitted meta-model is a selection procedure re-run on each new window, and R9 measured exactly that procedure — re-deriving the best of 54 every six months — **losing to holding one pre-committed cell in all five folds where the two disagreed** (+18.39% against +31.38% compounded, M22). R7 as scoped would add a per-period fit on top of a machine whose per-period fitting has been measured and costs money. **No R7 plan is written**, and writing one now means answering M22 first: what R7 re-fits, how often, and what evidence would show that re-fit beats not re-fitting. See [§9.4](#94-eth-replication-of-the-r5r6-protocol--ethusdt-perp-4h) and [§9.5](#95-r9-walk-forward-and-robustness--btcusdt-perp-4h). |
| R8 | HMM / Kalman | Beats R5 rules out-of-sample by enough to justify the complexity | R7 | — **blocked on its inputs, not on its model (M25).** R7 measured the four features this would estimate over carrying **no** chop information on either instrument, while the axis §2.1 defines chop with — `energy` — is not among them. A better estimator of a state vector missing its chop axis is a better estimator of the wrong thing, so **the model progression is unchanged and does not start here**. What unblocks it is an inputs phase: widen the machine's feature set toward the nine dimensions §2.1 declared, `energy` first, and re-run R7's own diagnostic against the same declared bars before any model is fitted. |
| R9 | Walk-forward + robustness | Purging, embargo, deflated Sharpe, parameter perturbation, feature dropout | R8 | **Done — run out of order, before R7/R8 and deliberately so**: a sizing model fitted to numbers that have not survived their own selection is a model of noise, and the ETH replication had already removed R5's no-selection-discount defence (M19). Every gate item is answered; purging and embargo are **stated with their reasoning and not performed**, which is the honest finding for a strategy that fits nothing (they become mandatory at R7/R8). **R5's cell survives out of sample and the search that found it does not** — walk-forward re-derivation is 6/9 folds and +18.39% compounded against the pinned cell's 7/9 and +31.38%, losing in all five folds where the books differ (M22); DSR 0.7026 against `E[max SR]` +0.9676 cleared by the observed +1.2155. `enter_strength = 0.80` is the one axis stable across all nine folds, and it is a **ridge** rather than a plateau — one step down dies at 3× costs. See [§9.5](#95-r9-walk-forward-and-robustness--btcusdt-perp-4h). |
| R10 | Paper trading | Research and live code paths produce identical output | R9 | — **the gate is not close, and the gap is now enumerated**: six ways a live signal differs from a backtested one, in the 2026-08-05 state-of-play entry. Two are structural rather than bugs — the runner withholds exit ingredients until Phase 1b gives it an `ExitMode`, and `TargetExposure` strategies cannot run on the event path at all. The product vision (a command-and-control dashboard) lands here, not earlier. |
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
| H3 | The state model mainly improves **sizing and tail risk**, not per-trade win rate | **Supported, R5** (figures re-measured 2026-08-04 after the bounded-exit fix). Out of sample `state_machine_v1` cuts max drawdown 43.86% → **4.67%** against the R0 baseline and lifts winner/loser ratio 1.63 → **2.25**, while total return stays *below* buy-and-hold (+15.5% vs +85.8%). Win rate moves only 37.7% → **43.8%**. Judging on Sharpe alone would have understated it; judging on return alone would have called it a failure. **Second-asset evidence, ETH ([§9.4](#94-eth-replication-of-the-r5r6-protocol--ethusdt-perp-4h)):** the shape replicates and is more extreme. The ETH-selected machine cuts max drawdown to **5.69%** against donchian 40/10's 31.41% and wins Sharpe **+0.868 against +0.785**, while returning **+18.14% against +84.00%** — a 4.6-fold loss on return, wider than BTC's gap. Win rate and winner/loser ratio were **not** measured on ETH, so only the drawdown-and-return half of H3 has a second asset. |

**H3 matters for evaluation.** Do not judge the regime engine on Sharpe alone — check
whipsaw cost, average holding period, winner/loser ratio, and drawdown.

### Crypto

| ID | Hypothesis | Result |
|---|---|---|
| C1 | OI direction confirms trend quality (price↑/OI↑ ≠ price↑/OI↓) | **BLOCKED — not backtestable.** Binance serves only ~30 days of OI history (`startTime` 40d back → `-1130 parameter 'startTime' is invalid`, measured 2026-08-03 and re-confirmed after R1 landed). Perp klines and funding go back to 2019; OI does not. Options: accumulate forward from today and revisit in ~1 year, buy historical OI, or drop C1. Do **not** substitute a 30-day sample for a historical study. **Forward collection started 2026-08-03** — `strategy-lab fetch-open-interest --symbol BTC/USDT --period 4h`, 186 snapshots covering 2026-07-04 → 2026-08-03. The command takes no `--since` and refuses an out-of-window request rather than clamping it, so the 30-day sample cannot be mistaken for history. Schedule it to keep accumulating; C1 becomes answerable around 2027-08. |
| C2 | Funding is a *confirmation* variable at moderate levels and a *contrarian* one only at extremes | — |
| C3 | Breakouts after volatility compression outperform ordinary breakouts | **Not supported via `compression_release`, R5** (re-measured 2026-08-04 against the fixed machine). The timing lift does not replicate — P(a commitment within 6 bars \| a release fired) is 6.73% against a 4.20% base rate in-sample and **4.95% against 4.66%** out of sample — and the payoff split now *disagrees between halves* rather than agreeing negatively (preceded vs plain: +0.33% vs +0.23% at 6 bars in-sample, +0.37% vs +0.63% out of sample; n ≈ 23–30 either side). A firing does mark a ~12–13% larger *absolute* 6-bar move on both halves — expansion without direction — and that is gone by 30 bars. This tests one feature at one threshold, not the hypothesis in full; a breakout-conditioned version (compression percentile *at the breakout bar*) is untested and would be the fair retry. |
| C4 | Cross-coin breadth leads individual-coin trend confirmation | — |
| C5 | Long and short need asymmetric parameters (liquidation cascades make downside faster) | — |
| C6 | Exhaustion is a worse place to hold risk than riding, so a matured move should be *tapered* rather than held flat or exited — the premise under §2.4's 100% → 55% → 20% rows | **Not supported, R6** (BTC/USDT perp 4h, R5's 6,048-bar test half). Measured as per-bar return **on the notional actually held**, so per-state size cancels: RIDING **+6.50 bp/bar** (sd 95.5) against EXHAUSTION **+5.55** (sd 98.0) on the trained machine, and RIDING +7.34 (sd 94.8) against EXHAUSTION **+7.79** (sd 91.3) on the default — the ordering the taper assumes is small on one configuration and *reversed* on the other. Over the 24 / 32 RIDING → EXHAUSTION episodes (279 / 338 bars, median 12), holding the taper earned +315.80 against v1's +276.61 trained and **+195.89 against +316.26** default — worse, at higher exposure. The state that stands out is **CONFIRMED** (+16.6 / +9.2 bp/bar), which §2.4's table sizes *below* RIDING. **No per-state t-stat exceeds 1.77**, so this refutes the premise as a basis for sizing rather than establishing its inverse. **Second-asset check, ETH ([§9.4](#94-eth-replication-of-the-r5r6-protocol--ethusdt-perp-4h)): the structural half replicates; the return half was not re-run.** On ETH's test half **not one entry is in `RIDING` at any of the three configurations** — 42/8/7, 55/2/25 and 83/21/70 across `BREAKOUT`/`CONFIRMED`/`EXHAUSTION` — while the machine spends 104 / 209 / 466 bars there, and the largest target at entry is 0.70, i.e. **66.5% of capital against the 95%** the `RIDING` row asks for. So the mechanism this row sits on is a property of the machine, not of BTC. The per-bar `RIDING`-vs-`EXHAUSTION` comparison itself was **not** re-measured on ETH, so the "not supported" verdict still rests on one asset. |

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
| 2026-08-06 | **R7 complete — the machine detects chop with four of the state vector's nine dimensions, and the axis §2.1 defines chop *with* is one of the five it dropped.** Pre-registered at `ddca7dc` with thresholds as numbers (M23 in force). The control is the strongest one this program has run: R4's conditional IC table reproduced through `diagnose_features` to four decimals on its published n — +0.0385 / +0.0022 / −0.1128 / +0.1314 — then extended to **all 81 cells of §9.1 plus every bar count, warmup and tercile boundary: 114/114**. The label was audited for leakage **before** any figure was read, not only in the outcome row that calls for it. **Results against the declared bars: `COMPRESSION` 0/6 on BTC and 0/9 on ETH** (best deficit −1.79 pp against −10 pp; inside-vs-outside separation flips sign between halves), **the nine features 0/9 on BTC**, **the composite gate 0/3** — and the last is structural: it beats the `strength` gate alone by **+0.00 pp** on the test half at every horizon, because `direction_floor = 0.10` admits 82% of bars, so **the machine's entry gate is one feature, not two**. **`energy`/`compression` is the exception**: −0.0906 on BTC, missing by 0.0094 (2.03 sd against the bar's 2.24), and **clearing on ETH at −0.1521** with both halves agreeing — same feature, same sign, two instruments. The machine **does not read it**; `DEFAULT_FEATURES` is `(direction, strength, stability, crowding)`, and §2.1's own worked example is `Strength = 20, Energy = 95 → violent two-way chop`. The sign runs against the naming: high `energy` is the chop side, so the *state* `COMPRESSION` and the *feature* `compression` are unrelated. **R9's `enter_strength` lead closes cleanly** — extending to {0.85, 0.90, 0.95} gives +0.5002 / +0.5577 / −0.7497 against +1.3974, so 0.80 is an **interior** optimum and the edge worry raised in R5, ETH and R9 is settled (M26). Verdict: the pre-registered third row on BTC, with the ETH exception recorded. **R8 does not start on the current four inputs** (M25). See [§9.6](#96-r7-choptrend-state-diagnosis--btcusdt-perp-4h-replicated-on-eth). |
| 2026-08-06 | **R9 complete — and the answer is not on the pre-registered table. The cell survives; the procedure that picked it does not.** Pre-registered at `78eaf55` before any figure existed; the control reproduced R5's published row to the digit (+15.454329% / +0.8956 / 4.6689% / 73) before a single R9 number was read. **Deflated Sharpe**, 54 training trials: observed **+1.2155** against `E[max SR]` **+0.9676** under the null — cleared, and cleared by 10 of the 54 cells, every one at `enter_strength=0.80` — but the **DSR itself is 0.7026** (0.7697 on daily returns) against the conventional 0.95. The plan's outcome row admits both readings and declared no threshold, so both stand (M23). **Walk-forward**, nine expanding folds, re-deriving the winner from the 54 each time: **6/9 positive, mean +0.687, compounded +18.39%** against buy-and-hold's +75.17%, **four distinct cells in nine folds**. The declared extra is what inverts it — R5's **pinned** cell over the identical blocks, selecting nothing, goes **7/9, mean +1.125, +31.38% compounded**, and beats re-derivation in **all five** folds where the books differ, without exception (M22). `enter_strength=0.80` wins all nine folds; the three axes that move are the ones perturbation says degrade gracefully. **Perturbation**: `enter_strength` is a **ridge** — one step down lands on the same test net return to two decimals (+15.4489% vs +15.4543%) at 1.8× the drawdown, 2.3× the turnover, and **−8.59% at 3× costs against +6.41%**. `exit_strength` ⅓→0.20 is a no-op out of sample (identical equity, 0.000e+00) while differing in training. **Dropout**: `direction` and `strength` are structural gates — neutralise either and the machine never leaves `COMPRESSION`, **0 trades, null Sharpe**; `stability` is the one component that looks *worse* in sample than out (+1.2404 in / +0.6827 out); `crowding` reproduces M20's +16.44 / +0.801 / 71 by a second route. **Purging and embargo stated, not performed**, with both halves of the argument re-checked and the 73/73 frame-start agreement re-measured. **M21 never disagreed with R5's rule about a winner** on the training half or on any fold. See [§9.5](#95-r9-walk-forward-and-robustness--btcusdt-perp-4h). |
| 2026-08-05 | **State of play against the product vision, and six reasons the live path does not yet equal the research path.** No measurement — a read of the code against the goal of a local, near-real-time dashboard carrying per-bar state and feature values per strategy, switchable, historical and current signals in one view. **Where the repo is.** `backtest` writes a self-contained `plot.html` whose markers are baked in at generation time; `serve` is 144 lines of `http.server` over `reports/` with a single `/candles` endpoint, so a chart live-updates its *candles* while its *signals* stay frozen. The event path (`ReplayFeed` → `StrategyRunner` → the append-only `signals` table) produces signals incrementally but joins to no chart, and no live feed exists — `MarketDataFeed` is the socket built for one. **Two gaps that are not about plumbing.** (1) The signals table records *configuration*, not *reasons*: a replayed signal's `features` JSONB is the strategy's own settings stringified (`allow_shorts`, `rank_window`, `crowding_measured`, …), while the state and the four feature values that explain the signal are computed on every bar and **discarded**. The "why" layer the dashboard exists for is persisted nowhere, by either path. (2) A marker vocabulary cannot draw `TargetExposure`; `state_machine_v2` needs level series (`target`, `rebalance_target`, `position_fraction` — all already on `ExposureBacktestResult`), which is a different primitive, not a variant of an arrow. **The census: six ways a live signal differs from a backtested one**, which is R10's gate stated as a list. (a) **Funding/crowding** — `Bar` has no funding field, so replay runs both state machines crowding-neutral (M20; measured +15.45%/+0.896 against +16.44%/+0.801 on BTC's trained cell). (b) **Exit modes do not exist in replay, by design until Phase 1b** — `StrategyRunner._extract` deliberately emits entry/exit booleans and withholds `trend_failure_*`, setup stops and `position_size`, because which of them fire is an `ExitMode` decision `run_backtest` makes; so for the five strategies that depend on engine-side exits the signal stream matches *no* single backtest configuration. (c) **Size is engine-side** — no signal carries one, so a live chart cannot show how big. (d) **`TargetExposure` strategies cannot run on replay at all** — `StrategyRunner` calls `generate_signals`, which they do not have; the `target_exposure` column is a socket with nothing plugged in. (e) **The determinism suite is blind here** — it compares on `synthetic_ohlcv`, which carries no funding, so it proves crowding-neutral ≡ crowding-neutral and structurally cannot see (a). (f) **Equity bars are not append-only** — the Yahoo fetcher rescales all history by adjusted close, so a dividend rewrites past bars and an append-style live update silently diverges from a fresh fetch. **The recommendation, and the one design move that matters.** A read-only research browser that computes signals **server-side on request with `generate_signals` over stored candles** — the same whole-history vectorized call the backtest makes — sidesteps every one of (a)–(f) at once, because it *is* the backtest path. It needs no live feed, no change to `Bar`, no Phase 1b, and it forces the design of the same API Phase 3 needs. For 4h bars, poll-and-recompute is indistinguishable from real-time. The command-and-control centre proper is R10 by the roadmap's own gate — "research and live code paths produce identical output" is exactly what this census says we lack. Sequencing is Q6. |
| 2026-08-05 | **The ETH replication: the *method* replicates and the *parameters* do not, and the untuned machine — R5's own evidence that its verdict carried no selection discount — fails outright on the second asset at −19.58% / Sharpe −0.563 / 30.56% drawdown / 174 round trips.** Protocol pre-registered in [docs/plans/2026-08-05-eth-replication.md](../plans/2026-08-05-eth-replication.md) and committed at `dd485ec` **before any ETH figure existed**, in its own commit; the two questions, the split, the engine settings, the outcome table and all but one of the extras were fixed there, and nothing was adjusted after seeing a result. Full tables in [§9.4](#94-eth-replication-of-the-r5r6-protocol--ethusdt-perp-4h). **Frame:** ETH/USDT perp 4h, **14,650 bars, 2019-11-27 08:00 → 2026-08-03 20:00**, split **2023-10-31 00:00** — the same calendar date as R5, chosen so the test half is **6,048 bars, identical to BTC's**. Training window bars 2,352–8,601 (6,250 tradeable, every cell started at the deepest of the 54 warmups), test 8,602–14,649 (6,048). Engine defaults throughout: 10bp/side, `opposite_signal_only`, fixed sizing, `position_pct=0.95`, 10,000 initial, everything net of funding. **The control ran first**, as R6 established: R5's published BTC row reproduced exactly (+15.45% / +0.896 / 4.67% / 73 trades) before any ETH number was read, and the four headline ETH rows were then independently re-run through `run_backtest` and reproduced to the digit. **Q1 — transfer: PARTIAL.** BTC's trained cell carried across with **zero** parameters re-derived returns **+3.77% / Sharpe +0.184 / maxDD 14.03% / 82 round trips / 13.8% in market / funding 43.13** on ETH's test half. Positive, so the machine is not BTC-specific — but donchian 40/10 over the same bars returns **+84.00% at Sharpe +0.785**, and Q1 **dies at 3× costs (−7.19%)**. The pre-registered reading of "positive out of sample but below the baseline" is *partial*: the state machine works, the edge over a channel break does not transfer. **Q2 — protocol replication: REPLICATES, thinly.** Re-running R5's own 54-cell search on ETH's training half selects `StateMachine(enter_strength=0.80, exit_strength=⅓, min_dwell=8, cooldown=8)` at training Sharpe **+1.3356** (runner-up +1.3021, the same cell at `exit_strength=0.20`). Evaluated once on the test half: **+18.14% / +0.868 / 5.69% / 57 round trips / 82 fills / 10.6% in market / funding 15.63**, surviving 3× costs at **+10.53%**. It beats donchian 40/10 **by 0.083 of Sharpe while losing on return 4.6-fold** (+18.14% against +84.00%) — a reader weighting return over Sharpe reads that as a loss, and the pre-registered table says replicates. **`enter_strength` and `exit_strength` land on BTC's values; both *timing* axes do not** — `min_dwell` and `cooldown` each move 4 → 8. **The untuned R4-default fails, and it is the sharpest single finding here.** −19.58% / **−0.563** / 30.56% / 174 round trips out of sample, from +1.09% / +0.081 in training. R5's "the untuned machine also passes, so the verdict carries no selection discount" **does not transfer at all**: on ETH the verdict rests entirely on a search over that asset's own training half, so the selection discount applies in full. That removes R5's strongest claim to being unsearched, and it is why this entry gives the default its own weight rather than listing it as a free extra. Recorded as M19. **The baseline is healthier on ETH than on BTC, and 40/10 is not ETH's cell.** The R0 gate's 16-cell donchian surface is **16/16 positive on both halves** (BTC: 16/16 train, 14/16 test), median Sharpe +0.536 train / **+0.567** test, best cell **160/40 at +0.9638** (+145.98%) in training and **160/80 at +1.007** (+121.04%) out of it. Under **M11**'s own rule — the baseline tuned on the same half by the same scalar — ETH's honest baseline is therefore **160/40 at test Sharpe +0.696**, not 40/10. **Q2 still beats it; Q1 still loses.** **The 54-cell surface: 54/54 positive in training, and the optimum sits on an edge in three of four axes.** Marginal means of training Sharpe — `enter_strength` 0.55 → +0.424, ⅔ → +0.310, **0.80 → +0.826**; `exit_strength` 0.20 → +0.466, ⅓ → +0.574; `min_dwell` 2 → +0.489, 4 → +0.436, **8 → +0.635**; `cooldown` 4 → +0.498, 8 → +0.534, 16 → +0.528. **Only `cooldown` is interior.** The **top six cells are exactly the six with `enter_strength=0.80, min_dwell=8`** (+1.18 to +1.34) — a real plateau that happens to be the grid's own corner, so the selected cell reads as "the most selective and slowest setting tried", the same shape R5 flagged and the R0 15m gate was rejected for. And **`enter_strength` is non-monotone on ETH** (⅔ *below* 0.55) where on BTC it rose cleanly (+0.139 / +0.440 / +1.014), so even the one axis R5 called dominant does not have the same shape on a second asset. **R6's mechanism replicates exactly, so it is a property of the machine and not of BTC.** **Zero entries in `RIDING` on ETH's test half at all three configurations** — Q2 42 `BREAKOUT` / 8 `CONFIRMED` / 7 `EXHAUSTION`; Q1 55 / 2 / 25; default 83 / 21 / 70 — while the machine spends **104 / 209 / 466 bars** there. The largest target at entry is **0.70 → 66.5% of capital against the 95%** the `RIDING` row asks for. M16 was measured on one asset; it now holds on two. **The continuous contract is worth much more on ETH than R6 measured on BTC.** On BTC v2 moved Sharpe +0.896 → +0.842 and +0.746 → +0.913 — a wash that disagreed in sign. On ETH it **adds at every configuration** and rescues the default from −0.563 to **+0.061**: Q2 +0.868 → +0.872 (+18.14% → +28.92%), Q1 +0.184 → **+0.505** (+3.77% → +18.74%), default −19.58% → −1.40%. Round trips stay **within one of v1's everywhere** — identical decisions, different sizing — and fills are 2–3×, which are resizes and not trades. **Frame-start invariance: 0 signal disagreements across all 6,048 test-half bars**, continuous targets agreeing to 1.7e-12. **Caveats, none softened.** (1) **The 16-cell donchian grid is reconstructed, not cited** — the R0 4h gate's grid exists nowhere in the repo as a constant, and was recovered from four converging pieces of evidence: the stored 15m sweep's committed grid shape, this charter's "swept `entry_span` up to 160", every cell the charter names being in it, and its "5 of 16 are exact duplicates" reproducing exactly. Strong but circumstantial; pinning it as a constant is a **follow-up, not done**. (2) **The selection scalar is diluted by each cell's own warmup.** R5 scores the whole window including the leading flat run. Every cell of a surface shares one frame start -- the deepest warmup in the grid, here 2020-12-23 08:00 and 6,250 tradeable bars for all 54 machine cells and all 16 donchian cells alike -- but `run_backtest` then masks each cell by **its own** `warmup_bars`, so a deeper cell carries more leading flat bars inside a window of the same length. That mask, not the frame start, is the dilution: ETH's donchian 40/40 and 40/80 are **provably the same book** — identical net return, drawdown and round trips to 6dp, identical positions on all 6,048 tradeable bars — yet score **+0.8327 against +0.8300**, the entire gap being 40 extra flat bars. The 54 machine cells span **2,112–2,352** warmup bars, so the artifact is worth up to **~4% of a cell's Sharpe**. It did **not** change the winner (a tradeable-bars-only Sharpe selects the same cell, +1.5612), but it is a defect in the ranking rule. M21. (3) **Five of the 16 donchian cells fall into two duplicate groups**, the degeneracy the R0 correction already recorded. Deduplicated to 13 distinct books -- `entry=20` with `exit` 20/40/80 is one book and `entry=40` with `exit` 40/80 is another, so five cells collapse to two and three are redundant: **13/13 positive on both halves**, median +0.568 train / +0.583 test. "16/16" overstates breadth. (4) **The two halves are the same calendar, not the same market.** ETH's test half is nearly flat — buy-and-hold **+3.51%**, against +200.51% in its own training half and BTC's +85.78% over the identical dates. The pre-registration's "a difference between them is the asset rather than the regime" is **only half true**: it is the asset *and* the asset's own regime, and this is the caveat that most limits the comparison. (5) **Q2's win is one scalar by 0.083, on a strategy returning a fifth of the baseline.** (6) **Two undeclared extras, neither feeding any selection**: `state_machine_v2` at the ETH-selected cell -- the pre-registration's "`state_machine_v2` at both configs" meant the two v1 configurations declared beside it, the R4 default and the BTC-trained cell, so the third is an extra, and the pre-registration's declared secondary split at ETH's own 60/40 boundary (2023-12-01 08:00, 5,860 test bars, buy-and-hold **−11.16%**) — donchian +86.04 / +0.822, Q1 +4.01 / +0.197, Q2 +18.14 / +0.878, default −14.81 / −0.451. **Every verdict is unchanged, so the split choice is not doing work**, which is exactly what the secondary split was declared for. **A reproducibility defect found while verifying, and it is not small.** The `backtest` CLI **never attaches a `funding_rate` column to the frame** — it passes funding to `run_backtest` for cost accounting only, where the `features` command does `df.assign(**{FUNDING_COLUMN: align_funding_to_bars(...)})`. So `state_machine_v1` and `state_machine_v2` run from the CLI read `crowding` as the neutral **0.5 fallback** and record `crowding_measured=False`. Measured on BTC's trained cell: **without the column +16.44% / +0.801 / 6.08% / 71 trades; with it +15.45% / +0.896 / 4.67% / 73 trades**, the second matching §9.2 to every published digit. So **every published state-machine figure was produced with the column, and the documented canonical command does not reproduce them** — the research is not wrong, the CLI is out of step with it. It also explains §9.3's own entry-count note (71/152 rather than 73/153: that check ran without the column). Recorded as a known defect with a follow-up, **not fixed here** — a branch that changes the engine cannot also be the branch that measures it. M20. **Nothing about the BTC figures moves**, and no ETH parameter was tuned on ETH's test half. 705 tests, ruff clean; this branch is documentation only. |
| 2026-08-04 | **R6 complete — gate passes. The continuous-exposure contract executes, the taper it was built for is worth approximately zero, and the reason is that `state_machine_v1` has never sized an entry for the top row of its own policy table.** Full tables in [§9.3](#93-r6-continuous-exposure-comparison--btcusdt-perp-4h). Setup identical to §9.2 — 15,118 bars from 2019-09-10 08:00, split at 2023-10-31 00:00, the 6,048-bar test half (buy-and-hold +85.78%), 10bp/side, net of funding, `position_pct=0.95`, 10,000 initial — with `state_machine_v2` on the new path at its default rebalance band of **0.05, unchanged**. v2 introduces no parameter of its own: same machine, same policy, same `STATE_TARGET_RISK`, same four features, so v1-vs-v2 is the boolean collapse and nothing else. **The control, run before anything else**, because a second execution path is a second chance to measure the wrong thing: one hand-rolled numpy statistics function applied to *both* paths' net-of-funding equity curves reproduces every published v1 figure — trained **+15.4543% / +0.8956 / 4.6689% / 73 trades / 13.99% time in market / funding 21.72**, default **+15.5161% / +0.7460 / 7.1062% / 153 / 20.83% / 26.25** — and agrees with vectorbt's own estimator to <1e-9, so the two engines are scored by one instrument rather than each by its own. v1's rebuilt book matches the engine's written `equity_curve.csv` to 1.8e-12 on all 8,208 bars, and v1 and v2 derive the same `warmup_bars` (2,160 trained / 2,192 default), verified rather than assumed. **What the continuous path does.** Trained: v1 +15.45% / Sharpe **+0.896** / maxDD **4.67%** / 73 round trips / 120 fills / 45.2× capital traded / funding 21.72, against v2 +25.89% / +0.842 / 10.85% / 74 / 276 / 82.8 / 62.77. Default: v1 +15.52% / +0.746 / **7.11%** / 153 / 211 / 110.3 / 26.25, against v2 +36.70% / **+0.913** / 12.55% / 154 / 416 / 164.1 / 90.22. v2's fills equal its decision bars **exactly** (276/276, 416/416) — one order per decision at the 0.05 band, which is what that band is for. **v2 survives 3× costs in both configurations, including the default that kills v1**: trained +25.89 / +17.61 / **+9.24** against v1's +15.45 / +10.93 / +6.41, default +36.70 / +20.18 / **+3.38** against v1's +15.52 / +4.49 / **−6.54**. It carries 2.3× the fills but only 1.5–1.8× the traded notional, because a resize is an increment, not an open-plus-close. **Where the difference comes from, decomposed by bar.** v2 held **more** than v1 on 523 bars (trained) / 807 (default) for **+1,445.31 / +2,783.64** gross — the ramp. It held **less** on 75 / 77 bars for **+15.30 / −63.05** — the taper. The two paths were identical on 5,449 / 5,163 bars, for 0.00. **The taper is the thing this phase was built to measure, and it measures approximately zero.** **Why, and this is the finding: of v1's size-setting entry bars on the test half, not one is in `RIDING`, in either configuration** — trained 56 `BREAKOUT` (target 0.35) + 15 `EXHAUSTION` (0.55); default 92 `BREAKOUT` + 20 `CONFIRMED` (0.70) + 40 `EXHAUSTION`. The largest position v1 ever opened is **52.3% of capital (trained) and 66.5% (default)**, against the **95%** a `RIDING` target asks for — while the machine spent **209 (trained) / 516 (default) bars in `RIDING`**. The mechanism is structural, not statistical: entries land above the `BREAKOUT` row readily — the `CONFIRMED` and `EXHAUSTION` counts are exactly that — but never on the top one, because `RIDING` is only ever reached with a position already open: the lifecycle passes `BREAKOUT` and `CONFIRMED` on the way, both already carrying a non-zero target, and an entry needs a change of side. `from_signals` reads size at entry only, so the position is frozen wherever it opened and cannot scale when the move is confirmed. **The row §2.4 sizes highest is the one row that has never executed.** Relative to a position frozen at its entry size, the states that follow are far more often a step *up* than a step down — which is why the "taper" measures as a ramp. *(A separate check using a cruder funding alignment counted 71/152 entries rather than 73/153 and 56/92 `BREAKOUT`; the structural claim — zero `RIDING` entries, largest size 0.55 / 0.70 — reproduced exactly.)* **The charter's premise fails directly too.** Per-bar return on the notional *actually held*, so per-state size cancels: trained `RIDING` **+6.50 bp/bar** (sd 95.5) against `EXHAUSTION` +5.55 (sd 98.0); default `RIDING` +7.34 (sd 94.8) against `EXHAUSTION` **+7.79** (sd 91.3). **Exhaustion was not a worse place to hold risk than riding** in this half. The state that stands out is `CONFIRMED` (+16.6 / +9.2 bp/bar), which §2.4 sizes *below* `RIDING`, and **no per-state t-stat exceeds 1.77**. Recorded as C6. **None of the plan's three predicted outcomes happened.** Not "improves risk-adjusted return" — Sharpe is a wash and disagrees in *sign* across configurations (−0.054 trained, +0.167 default). Not "improves drawdown but costs return" — the inverse: v2 raises return and **worsens** drawdown 2.3× / 1.8×. And "costs more in turnover than it saves" is **refuted**, since v2 survives 3× costs where v1's default does not. **The honest headline: the taper is worth approximately zero (+15.30 / −63.05 gross over 75/77 bars). What v2 buys is exposure — 1.7–1.8× v1's average, 47.7% / 53.7% of capital when held against 28.4% / 30.9% — and the return scales with it, which is leverage rather than skill. Sharpe, the only column not contaminated by that, is flat.** So **R6's real finding is not about the taper: it is that `state_machine_v1` has never sized an entry for the top row of its own policy table**, and R5's published +15.45% was earned at roughly a third of the risk that policy asked for. That is a defect in the *boolean contract*, not in the policy — and it is exactly what R2 warned of when it deferred the taper to R6 rather than writing one against `from_signals`. **This is not R6 failing its gate.** The gate asks whether the continuous contract works and whether the taper's contribution is measured, pass or fail; both hold, and the taper measuring zero is a *result*. **Caveats, all of which cut the same way.** **v2 is not risk-matched to v1**, so every column except Sharpe is contaminated by the exposure difference — a 1.7× book returning 1.7× more is arithmetic. n is small: 73 / 153 size-setting bars, two configurations, one asset, one 2.75-year half, and no per-state effect is distinguishable from zero. And a **split-boundary contract difference** flatters v2 slightly: v1's *event* contract cannot re-enter a position already live at the first tradeable bar while v2's *level* contract simply holds it, which accounts for the 74-vs-73 round trip and **+9.07 / +114.23** of gross P&L — excluded, the ramp/taper split is +1,436.23 / +15.30 and +2,669.41 / −63.05 and the conclusion is unchanged. It is an artifact of starting a run mid-history and would not appear in a whole-history run. **§9.2 is not corrected — every R5 figure re-measures to the digit, and this changes how they are *read*, not what they are.** `state_machine_v1`'s published results are what a machine frozen at its entry size did, never once at its `RIDING` target; every later state's target was computed on every bar and discarded. **Shipped with it:** `TargetExposure` and `ExposureStrategy` (`strategies/exposure.py`), `run_exposure_backtest` on `from_orders` (`backtests/exposure_engine.py`), `state_machine_v2` in a **third** manual registry (`strategies/exposure_registry.py` — six parametrized tests iterate the boolean one and every one calls `generate_signals`; a mutation proved an empty exposure registry silently **skips 4 parametrized tests and exits 0**), a nullable `target_exposure` column on the append-only `signals` table, and the continuous path's own determinism, lookahead and cold-start-warmup proofs. 686 tests. |
| 2026-08-04 | **CORRECTION to the entire R5 gate — the state machine had states with no bounded exit, so whole-history and cold-start runs could disagree *permanently*. The gate still passes; every published R5 figure moves, and the machine now survives 3× costs.** **The defect.** The lifecycle is strictly forward and every step up required an `advancing` bar, so a tail that neither advanced nor failed parked the machine wherever it stood. `EXHAUSTION` was reachable only from `RIDING` and nothing routed out of it except a hard failure; `BREAKOUT` and `CONFIRMED` parked the same way, and a tail that was strong *and* unstable cycled COMPRESSION → BREAKOUT → RESET at a phase that depended on where the run began. Measured: on a persistent tail of `direction=0.8, strength=0.5, stability=0.9, crowding=0.5`, a cold run stayed in `COMPRESSION` (target risk 0.0) while a warm one sat in `EXHAUSTION` (**−0.55**, a live short) forever — and **14 of 48 sampled constant tails had more than one attractor**. This is the repo's central backtest-equals-live guarantee failing, and **no `warmup_bars` can fix it**: the disagreement is unbounded in the length of the tail. **Scope in practice, stated plainly.** On the stored 15,118-bar history the worst cold-start lag the *old* machine actually produced was **47 bars** (default) and 32 (trained), far inside its 2,160-bar warmup — the longest `EXHAUSTION` episode in that history is 170 bars. So the published numbers were not corrupted by this; the guarantee was. **The fix, three rules, each also the behaviour the lifecycle wanted.** (1) A `BREAKOUT` or `CONFIRMED` that goes `min_dwell` consecutive bars without advancing drops to `RESET` — the mirror of the step-up rule, and it had already lost every bar of progress toward its next step. (2) `RIDING` ends on any non-`advancing` bar, which adds the case that parked it: a lean decayed below `direction_floor`. (3) `EXHAUSTION` runs out after a new `exhaustion_dwell` (default **12** = three `min_dwell` periods, so a move gets as long to end as it took to establish) — the one transition on a plain timer, because a decayed move has no condition left to wait on. Plus a `COMPRESSION → BREAKOUT` gate that now refuses `unstable` or `crowded` bars: a move may only start under conditions that would let it continue. **The invariant is now proved, not assumed** — `StateMachine.convergence_bars` = `exhaustion_dwell + cooldown + 3 × min_dwell + 2`, and from any starting configuration a constant tail reaches the same state inside it (worst observed 31 of a 34-bar bound on the default machine, 77 of 80 on a deliberately slow one). `warmup_bars` is derived from it rather than pinned, so `StateMachine(min_dwell=1000)` — reachable through `sweep_parameters`' `dataclasses.replace` — no longer declares the default's warmup. **Re-measured, whole protocol re-run: 54 cells scored on the training half, winner taken by the declared scalar, test half evaluated once.** The training winner moves to `StateMachine(enter_strength=0.80, exit_strength=⅓, min_dwell=4, cooldown=4)` at Sharpe **+1.215** (was 0.80/⅓/2/16 at +0.769); `enter_strength` is still the one dominant axis (marginal mean Sharpe 0.55 / ⅔ / 0.80 → +0.139 / +0.440 / **+1.014**) and the optimum still sits on the edge of the swept range. **50 of 54 cells are now positive in training, against a machine that used to sit in `EXHAUSTION` 13.3% of all bars.** Out of sample the trained machine lands at **Sharpe +0.896 / net +15.45% / max drawdown 4.67% / 73 trades / 14.0% time in market** (was +0.938 / +23.29% / 8.24% / 159 / 24.7%) and the untuned default at **+0.746 / +15.52% / 7.11% / 153 trades** (was +0.738 / +19.36% / 11.37% / 212). The baseline is untouched by any of this and re-measures **identically** to every published digit — −6.64% / +0.072 / 43.86% / 114 trades — which is the control that says the harness did not move under the machine. **What changed in the conclusions.** (a) **The machine now survives 3× costs**: +15.45% / +10.93% / **+6.41%** at 1×/2×/3×, where the old one died at −0.61%. That is the direct consequence of turnover halving — 159 → 73 trades, median hold 3 → **7 bars** — because a state that could not be left was being left by a *reversal* instead. The untuned default still does not survive 3× (−6.54%), so surviving it is a property of the trained cell, not of the strategy. (b) **The fade band is now exactly zero out of sample rather than slightly negative**: follow 54 trades and **+100.0%** of PnL, fade 19 trades and **+0.30 currency units on +1,567.15** total. In-sample the fade is +29.2% of PnL on 24 trades, up from +14.6%. The R5 reading is unchanged and sharper — a stable IC is not a tradeable one, and this machine is a trend follower whose fade band earns nothing. (c) The out-of-sample Sharpe is now *below* the in-sample one (+0.896 vs +1.215), which removes the one shape that looked like a leak. Frame-start invariance re-checked: the whole-history run and the test-window run agree on **73 of 73** entry timestamps. (d) H3 lands harder: winner/loser ratio **2.25** against donchian's 1.63, win rate 43.8% against 37.7%, drawdown a *ninth* rather than a fifth. **Also corrected by re-measurement.** The training half now starts at bar **2,352** rather than 2,160 — the 54 cells no longer share a warmup, so the surface starts every cell at the deepest one in the grid — giving 6,718 training bars (was 6,910) and a training buy-and-hold of **+222.48%** (was +230.90%). Donchian 40/10 on that window re-measures at +179.18% / **+1.462** / 17.89% / 110 trades. The 16-cell donchian surface: train **16/16** positive, median +0.853, best 40/10; test **14/16**, median +0.438, best **40/40 at +1.070** (+112.57%) — the test half is unchanged, the train half moved only with its window. `compression_release` re-judged against the fixed machine and the verdict is unchanged but for a different reason: the timing lift still fails to replicate (6.73% vs a 4.20% base rate in-sample, **4.95% vs 4.66%** out of sample), and the payoff split no longer agrees *negatively* across halves — it simply disagrees (+0.33% vs +0.23% at 6 bars in-sample, +0.37% vs +0.63% out of sample). The absolute-move marker holds at ×1.130 / ×1.119 at 6 bars and is gone by 30 (×1.032 / ×1.009). Still measured, still wired into nothing. **Test coverage added, because the defect was invisible to every suite that existed.** `tests/test_state_machine.py` now enumerates constant tails against prefixes that reach all six states and asserts convergence inside the declared bound, keeps the reported reading as its own named regression, and pins each of the three new exits; `tests/test_replay_determinism.py` gains the **primed** drive path — `StrategyRunner.prime` handed exactly `warmup_bars` bars from mid-history, then streaming — which is the only one of the three where *where you started* can show at all, since its two existing comparisons both begin at bar 0 and a causal strategy passes them by construction; `tests/test_state_machine_gate.py` now **re-derives its own 54-cell selection** on the training half (~33 s, `pytest.mark.db`) instead of trusting a pinned cell that test-half knowledge would have satisfied just as well, asserts exact trade counts rather than "more than 50", and asserts the untuned machine clears the baseline too. 599 tests. |
| 2026-08-04 | ***Every figure in this entry is superseded — see the bounded-exit correction at the top of this log. The verdict stands and the protocol is unchanged; the machine's numbers all move and it now survives 3× costs. The donchian baseline figures below re-measure identically.*** **R5 GATE PASSES. `state_machine_v1` beats the R0 baseline out of sample — and so does the *untuned* configuration, which is the version of the claim that carries no selection discount.** Full tables in [§9.2](#92-r5-split-sample-gate--btcusdt-perp-4h). Protocol, fixed before anything ran: parameters chosen on the **first 60%** of the canonical 15,118-bar BTC/USDT perp 4h frame and the **last 40%** looked at once, at the end. **54 configurations were tried on the training half** — `enter_strength` ∈ {0.55, ⅔, 0.80} × `exit_strength` ∈ {0.20, ⅓} × `min_dwell` ∈ {2, 4, 8} × `cooldown` ∈ {4, 8, 16} — selected by one scalar declared in advance, highest net-of-funding Sharpe. **The baseline got the same treatment on the same bars**: the R0 gate's own 16-cell donchian surface, whose best training cell is 40/10 at Sharpe +1.433 — the cell this charter already names, arrived at independently. Both sides' frames start at `split − warmup_bars`, so the engine's mask lands on the boundary and both trade the **identical 6,048 test bars** (2023-10-31 00:00 → 2026-08-03 20:00, buy-and-hold +85.78%). All figures net of funding at 10bp/side, `opposite_signal_only`, fixed sizing. **Out of sample: `state_machine_v1` (trained) Sharpe +0.938 / net +23.29% / max drawdown 8.24% / 159 trades / 24.7% time in market, against donchian 40/10 at +0.072 / −6.64% / 43.86% / 114 / 51.2%.** The untuned R4-default machine — thresholds set by R4's terciles before any backtest existed, one hypothesis, zero search — lands at **+0.738 / +19.36% / 11.37%**, so the verdict does not depend on the 54-cell search. **Three things must be said with it.** (1) **The win is on risk, not on return.** +23.29% is well under buy-and-hold's +85.78% and under 6 of the 16 donchian cells; the machine wins Sharpe and drawdown while standing aside three-quarters of the time. That is H3's prediction landing exactly — winner/loser ratio 2.02 against donchian's 1.63, win rate 42.8% against 37.7%, max drawdown a fifth of it. (2) **The baseline as a *family* is not dead out of sample — the cell training selects is.** 14 of 16 donchian cells stay positive on the test half at a median Sharpe of +0.438, and the best of them, 40/40, returns +112.57% at Sharpe +1.070 — beating the machine. That cell is only knowable with the test half in hand, so it is not a legitimate comparator, but a reader deserves to know the machine loses to the *oracle* baseline while beating the *honest* one. Donchian's collapse is concentrated: +30.73% in 2024, **−32.55% in 2025**, against the machine's +9.29% / −2.70%. (3) **It does not survive 3× costs.** Net +23.29% / +11.34% / **−0.61%** at 1×/2×/3×, against donchian's −6.64% / −28.03% / −46.23% — better at every level, but its own edge is gone by 3×, and its median holding period is **3 bars** against donchian's 23, so it is the more fee-sensitive of the two. **The result does *not* rest on the mid-tercile fade — the opposite.** Splitting the machine's out-of-sample trades by the band their entry bar sat in: the **follow** band (top `strength` tercile) is 94 trades and **+102.4%** of total PnL, the **fade** band is 65 trades and **−2.4%**. In-sample the fade was +14.6%. So R4's most interesting finding — a mid-tercile IC of −0.113 with *both halves agreeing* — did not convert into out-of-sample P&L, while the high tercile that R4 flagged as *decaying* (+0.195 → +0.062) is what carried the result. **A stable IC is not a tradeable one**, and that is the most transferable thing R5 measured. `state_machine_v1` is honestly described as a trend follower whose fade band currently earns nothing; deleting the fade is an R6 candidate, not an R5 edit. **Scrutiny applied before publishing, because an out-of-sample Sharpe *above* the in-sample one (+0.938 vs +0.769) is the shape of a leak.** The whole-history run and the test-window run agree on **159 of 159 entry timestamps**, so the machine's state at the split is not an artifact of where its frame began; `tests/test_lookahead.py`, `tests/test_replay_determinism.py` and `tests/test_feature_lookahead.py` all pass over it. The explanation that fits: the training half is a strong-trend regime where standing aside is expensive (donchian +179%), the test half is chop where it is cheap. **`compression_release` judged and NOT wired — "no signal", as a result.** R4 could not test it (median exactly 0.000, an event detector under a univariate IC). Measured here on the bars where it fires, threshold = the training half's own 90th percentile (0.0375), applied as an absolute number to both halves: ~700 firings per half; P(the machine commits to a move within 6 bars \| a release fired) is **6.74% against a 4.78% base rate in-sample and 5.71% against 5.26% out of sample** — the lift does not replicate. Worse for it, the commitments a release *did* precede paid **less** than the ones it did not, on both halves and at both horizons (train −0.36% vs +0.93% at 6 bars, test −0.39% vs +0.33%; n ≈ 26 either side). The one thing it does mark, consistently: the absolute 6-bar move after a firing is ~12–13% larger than average on both halves (2.69% vs 2.37%, then 1.94% vs 1.73%), and that gap is gone by 30 bars. It is a short-horizon volatility marker with no directional content and no gate value — which is a real answer, and evidence *against* C3 as stated. **Test-half looks, stated plainly:** three declared runs (donchian 40/10, machine default, machine trained), plus the 16-cell donchian surface as a fairness check on the baseline and 2×/3× stress rows on the same three. The surface and the stress rows can only make the machine's claim harder, never easier. Nothing was adjusted after seeing any of it. 577 tests. |
| 2026-08-04 | **R4 complete. Nine state features, each with a univariate diagnostic on 15,118 real bars — and one of them is recommended for cutting.** Full table in [§9.1](#91-r4-feature-diagnostics--btcusdt-perp-4h) below; the gate was that no feature ships unexamined, not that all nine survive. Two findings that change how the rest of the program should be read. **First: `participation` is the cut.** Its raw IC@30b is +0.026 with both halves agreeing, which passes a naive screen — but strip out the linear part of `energy` in rank space and **only 26% of it survives (+0.0068, first half +0.001)**, while `energy` minus `participation` keeps 90%. Its forward information is `energy`'s. It also carries the highest turnover of the nine by 4x (0.212 against 0.050 for the next), and its gating effect is the weakest measured. That relationship sits at r = +0.363 — nowhere near the \|r\| > 0.9 redundancy flag — so **a correlation threshold does not detect this class of redundancy at all**, and R5 must not rely on one. **Second: the strongest features here are conditioners, not predictors.** No single feature reaches \|IC\| 0.07 at any horizon, which is normal for 4h crypto, but `direction`'s IC@30b goes from +0.038 unconditionally to **+0.131 in the top `strength` tercile (+0.172 / +0.081 by half)** and to **−0.113 in the middle one (−0.118 / −0.110)** — non-monotone, and stable in both halves either way. `stability` gates it monotonically: +0.114 / +0.055 / −0.079 from low to high tercile. `strength`, `stability`, `persistence` and `energy` therefore earn their place by saying *when another feature works*, and ranking the nine by IC would have thrown away the useful half of the state vector. Third, smaller: `direction` and `crowding` correlate at +0.363 yet point opposite ways at forward returns, and removing each from the other **raises** both ICs (to 135% and 140% at 30 bars) — they are complements, not duplicates. `energy` × `compression` = −1.000 is one measurement deliberately exposed with both signs and is not a redundancy finding. Verification: every published number was re-derived by an independent script — SciPy's `spearmanr`, a hand-indexed forward return, halves sliced by position — with **0 disagreements**. |
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

### 9.1 R4 feature diagnostics — BTC/USDT perp 4h

```bash
strategy-lab features --exchange binance --market-type perp --symbol BTC/USDT \
  --timeframe 4h --horizons 1,6,30 --start 2019-09-10T08:00:00
```

**15,118 bars, 2019-09-10 08:00 → 2026-08-03 20:00.** The start is the first stored
funding settlement, not the first candle: candles begin 2019-09-08 16:00 but Binance
settled nothing on the contract for the first 40 hours, and a stretch with no stored
funding would read as *flat carry* rather than *unknown carry* — the same silent zero
`Crowding` refuses. Ten bars is a cheap price for not inventing one.

IC is Spearman against the return over `[t+1, t+1+h]`, anchored one bar **after** the
feature's own so the feature's own print is not the denominator of its own target.
Each cell is `full-sample (first half / second half)`; **bold marks \|IC\| ≥ 0.04**,
which is emphasis, not significance — nothing here reaches 0.07.

| Feature | warmup | cov | bars | min | med | max | IQR | AC(1) | turnover | IC@1b | IC@6b | IC@30b | max \|r\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| direction | 1920 | 100% | 13,198 | −0.848 | 0.032 | 0.828 | 0.503 | 0.999 | 0.0120 | +0.008 (+0.012/+0.003) | +0.015 (+0.019/+0.011) | +0.039 (+0.039/+0.035) | +0.363 crowding |
| strength | 96 | 100% | 15,022 | 0.000 | 0.099 | 0.624 | 0.134 | 0.976 | 0.0152 | +0.015 (+0.015/+0.015) | **+0.048** (+0.044/+0.054) | **+0.057** (+0.054/+0.061) | +0.674 persistence |
| persistence | 96 | 100% | 15,022 | 0.000 | 0.420 | 0.953 | 0.537 | 0.998 | 0.0141 | +0.008 (+0.017/−0.002) | **+0.040** (+0.055/+0.023) | **+0.061** (+0.084/+0.031) | +0.674 strength |
| stability | 96 | 100% | 15,022 | 0.580 | 0.844 | 0.941 | 0.058 | 0.997 | 0.0024 | −0.006 (−0.002/−0.010) | −0.009 (+0.003/−0.024) | **−0.062** (−0.032/−0.095) | −0.315 energy |
| energy | 503 | 100% | 14,615 | 0.002 | 0.485 | 1.000 | 0.556 | 0.975 | 0.0303 | +0.010 (+0.013/+0.007) | +0.015 (+0.019/+0.011) | **+0.059** (+0.072/+0.047) | −1.000 compression |
| compression | 503 | 100% | 14,615 | 0.000 | 0.515 | 0.998 | 0.556 | 0.975 | 0.0303 | −0.010 (−0.013/−0.007) | −0.015 (−0.019/−0.011) | **−0.059** (−0.072/−0.047) | −1.000 energy |
| compression_release | 504 | 100% | 14,614 | −0.773 | 0.000 | 0.919 | 0.017 | 0.019 | 0.0500 | +0.015 (+0.010/+0.021) | +0.005 (+0.005/+0.004) | −0.014 (−0.014/−0.013) | +0.245 participation |
| participation | 479 | 100% | 14,639 | 0.002 | 0.504 | 1.000 | 0.523 | 0.572 | **0.2117** | +0.012 (+0.013/+0.008) | +0.016 (+0.017/+0.011) | +0.026 (+0.024/+0.024) | −0.363 compression |
| crowding | 184 | 100% | 14,934 | 0.000 | 0.510 | 1.000 | 0.625 | 0.954 | 0.0421 | −0.013 (−0.008/−0.017) | −0.026 (−0.021/−0.032) | −0.021 (−0.004/−0.042) | +0.363 direction |

**Keep, and why.**

| Feature | Verdict | Reason |
|---|---|---|
| `strength` | **keep — conditioner first** | The only feature whose halves agree at all three horizons *and* whose gating is strong: `direction` IC@30b +0.131 in its top tercile against +0.038 unconditional. Non-monotone (mid tercile −0.113), which is itself a finding — "more strength is better" is wrong; "extreme strength is different" is right. |
| `stability` | **keep — conditioner only** | Direct IC is the largest on the page at 30 bars (−0.062) but the halves are −0.032/−0.095, so the level is not tradeable on its own. As a gate it is the cleanest: `direction` IC@30b +0.114 / +0.055 / −0.079 across terciles, monotone, both halves stable at the extremes. Narrowest distribution of the nine (IQR 0.058) — it discriminates within a tight band, which is worth knowing before anyone thresholds it. |
| `direction` | **keep — predictor** | Halves agree at 6 and 30 bars, and it is what the conditioners condition. Lowest turnover but the longest warmup by far (1,920 bars, ~11 months at 4h) — the binding constraint on any short backtest window. |
| `persistence` | **keep, watch** | IC@30b +0.061 is the second largest, but the halves are +0.084/+0.031 — decaying, not stable. Correlated +0.674 with `strength`, and once each is stripped of the other **neither residual has agreeing halves**, so the pair carries roughly one feature's worth of stable information between them, not two. R5 should not treat them as independent inputs. |
| `energy` / `compression` | **keep both** | r = −1.000 by construction, deliberately: the charter reasons about "is there fuel" and "is it coiling" separately and a state machine written against one reads backwards half the time. One dimension, two signs, not a duplicate. Halves agree at 6 and 30 bars. Retains 90% of its IC once `participation` is removed. |
| `compression_release` | **keep — untested by this method** | Tiny ICs, but sign-stable across halves at all three horizons and the most independent feature of the nine (max \|r\| 0.245). Its median is exactly 0.000 with IQR 0.017: it is an *event* detector, and a univariate IC averaged over every bar is the wrong instrument for it. Judge it in R5 on the bars where it fires. |
| `crowding` | **keep — the only non-price input** | Small and consistently negative at every horizon (crowded longs underperform), and a complement to `direction` rather than a duplicate: they correlate +0.363 yet removing each from the other raises both ICs. Also the only feature that would be lost on equities, which is worth knowing before the state machine assumes it. |
| `participation` | **CUT** | Retains **26%** of its IC@30b once `energy` is removed (+0.0068, first half +0.001), while `energy` keeps 90% of its own — the information is `energy`'s. Highest turnover of the nine by 4x (0.212). Weakest gate measured. Its correlation with `energy` is only +0.363, so **no correlation threshold would have caught this** — the cross-sectional `breadth` from R3 remains the participation input worth having. |

### 9.2 R5 split-sample gate — BTC/USDT perp 4h

**Re-measured 2026-08-04** after the state machine gained bounded exits; the figures this
section carried before that are superseded in full and the correction entry at the top of
§9 explains why. The baseline re-measures identically, which is the control.

15,118 bars from 2019-09-10 08:00 (the first stored funding settlement), split 60/40 at
**2023-10-31 00:00**. Both halves are scored on the bars each owns, and each run's frame
starts `warmup_bars` earlier so the engine's own mask lands on the boundary — the machine
warms ~2,200 bars against donchian's 40, and a shared frame start would have handed the
baseline ~2,150 bars the machine cannot see. The 54 cells no longer share a warmup either,
since each derives its own from `StateMachine.convergence_bars`, so the training half
starts every run at the **deepest warmup in the grid (bar 2,352)** — the same rule
`sweep_parameters` applies across a surface. Everything below is **net of funding**, at the
engine defaults (10bp/side, `--exit-mode opposite_signal_only`, fixed sizing, 95%
deployment, 10,000 initial).

Training half: 6,718 tradeable bars, 2020-10-06 08:00 → 2023-10-30 20:00, buy-and-hold
**+222.48%**. Test half: 6,048 bars, 2023-10-31 00:00 → 2026-08-03 20:00, buy-and-hold
**+85.78%**.

| Run | half | net return | Sharpe | max DD | trades | time in market | funding |
|---|---|---|---|---|---|---|---|
| `donchian` 40/10 (R0 baseline) | train | +179.18% | **+1.462** | 17.89% | 110 | 52.7% | 2,368 |
| `state_machine_v1` (R4 default) | train | +8.75% | +0.279 | 18.93% | 192 | 23.3% | 148 |
| `state_machine_v1` (trained, best of 54) | train | +30.52% | +1.215 | 5.25% | 77 | 14.4% | 119 |
| `donchian` 40/10 (R0 baseline) | **test** | −6.64% | +0.072 | 43.86% | 114 | 51.2% | 438 |
| `state_machine_v1` (R4 default) | **test** | +15.52% | +0.746 | 7.11% | 153 | 20.8% | 26 |
| `state_machine_v1` (trained) | **test** | **+15.45%** | **+0.896** | **4.67%** | 73 | 14.0% | 22 |

The machine's small funding bill is a balanced book, not an error: 40 long / 33 short out of
sample, paying 64.5 and receiving 42.8. Donchian pays 843.2 and receives 405.5 on a book
that is in the market nearly four times as often.

**Read these figures as a machine that never sized for `RIDING`.** Nothing here is
corrected — R6 re-measured every number in this table to the digit — but it also measured
that **not one of the test-half entries is in `RIDING`**, in either configuration, so the
largest position this run ever opened is 52.3% of capital (trained) and 66.5% (default)
against the 95% a `RIDING` target asks for, while the machine spent 209 / 516 bars in
`RIDING`. Entries reach §2.4's `CONFIRMED` and `EXHAUSTION` rows; `RIDING` is only ever
entered with a position already open, and `from_signals` reads size at entry, so the
position is frozen wherever it opened. The +15.45% below was earned at roughly a third of
the risk the policy asked for. See
[§9.3](#93-r6-continuous-exposure-comparison--btcusdt-perp-4h).

Note the out-of-sample Sharpe is now *below* the in-sample one (+0.896 against +1.215),
which is the ordinary direction. The previous pass had it the other way round and that had
to be explained; it no longer has to be.

**Selected configuration** (training half, highest net-of-funding Sharpe of 54):
`StateMachine(enter_strength=0.80, exit_strength=1/3, min_dwell=4, cooldown=4)` at +1.215,
and `tests/test_state_machine_gate.py` re-derives that selection from the declared grid
rather than trusting this line. Marginal means of training Sharpe show one dominant axis —
`enter_strength` 0.55 / ⅔ / 0.80 → +0.139 / +0.440 / **+1.014** — and the optimum still
sits on the **edge of the swept range**, which is the shape the R0 15m gate was rejected
for. The search wants a machine that trades even less, and the limit of that is not
trading. Read the selected cell as "the most selective setting tried", not as an interior
optimum. **50 of the 54 cells are positive in training** (the four that are not are all at
`enter_strength=0.55`), so the surface is a plateau in the axis that matters rather than a
spike.

**The R0 baseline's own surface, both halves** — a fairness check, since the R0 gate passed
on a plateau rather than on one cell:

| donchian surface | positive Sharpe | median Sharpe | best cell |
|---|---|---|---|
| train (bars 2352–9069) | 16/16 | +0.853 | 40/10 at +1.462 |
| test (bars 9070–15117) | 14/16 | +0.438 | **40/40 at +1.070** (net +112.57%) |

So the family survives out of sample and the *specific cell training selects* does not. The
machine beats the median cell and the training-selected cell; it loses to the test-half
oracle cell on both Sharpe and return.

**Cost stress, test half** (fees and slippage only — funding is a market rate, decision M8):

| Run | 1× | 2× | 3× |
|---|---|---|---|
| `donchian` 40/10 | −6.64% | −28.03% | −46.23% |
| `state_machine_v1` (default) | +15.52% | +4.49% | −6.54% |
| `state_machine_v1` (trained) | +15.45% | +10.93% | **+6.41%** |

The trained machine now keeps its edge at 3×, where the pre-fix version lost 0.61%. The
mechanism is turnover: 159 → 73 trades and a median hold of 3 → 7 bars, because a state
that could not be left was previously being left by a *reversal*. The untuned default still
dies at 3×, so this is a property of the trained cell rather than of the strategy.

**Where the PnL comes from** — every trade classified by the `strength` band its *entry*
bar sat in, which is what the policy reads:

| band | half | trades | PnL | share of total | win rate |
|---|---|---|---|---|---|
| follow (top tercile) | train | 53 | +2,243.50 | +70.8% | 49.1% |
| fade (mid tercile) | train | 24 | +927.43 | +29.2% | 45.8% |
| follow (top tercile) | **test** | 54 | **+1,566.86** | **+100.0%** | 48.1% |
| fade (mid tercile) | **test** | 19 | +0.30 | **+0.0%** | 31.6% |

The fade band is now *exactly* nothing out of sample rather than slightly negative, on a
31.6% win rate. The reading is unchanged and sharper: R4's mid-tercile IC was the stable
one and it did not convert.

**`compression_release`, judged on the bars where it fires.** Threshold is the training
half's own 90th percentile (0.0375; its median is exactly 0.000), then applied as an
absolute number to both halves. A "commitment" is the machine stepping COMPRESSION →
BREAKOUT; a release "precedes" one if it fired in the 6 bars before it.

| Measurement | train | test |
|---|---|---|
| firings / commitments | 683 / 47 | 666 / 47 |
| P(commit within 6b \| release) vs base rate | 6.73% vs 4.20% | 4.95% vs 4.66% |
| signed forward return, commitments **preceded** vs not, 6b | +0.33% vs +0.23% | +0.37% vs +0.63% |
| signed forward return, preceded vs not, 30b | +0.27% vs +3.16% | +2.14% vs +1.41% |
| \|6-bar move\| after a firing vs all bars | 2.72% vs 2.40% | 1.94% vs 1.73% |
| \|30-bar move\| after a firing vs all bars | 5.90% vs 5.71% | 3.99% vs 3.96% |

The only sign-stable, replicating effect is the fourth row: a firing marks a ~12–13% larger
absolute move over the next 6 bars, gone by 30. No directional content, and the timing lift
does not survive the split — 6.73% against a 4.20% base rate in-sample collapses to 4.95%
against 4.66%. The second and third rows are now *inconsistent* across halves rather than
consistently negative, which is a weaker claim than the pre-fix pass made and points the
same way: nothing here to gate on. Not wired.

*(Re-measured 2026-08-04 against the fixed machine. The commitment counts change because
`COMPRESSION → BREAKOUT` now refuses an unstable or crowded bar; the firing counts are a
property of the feature and are unaffected except by the moved training-window start.)*

### 9.3 R6 continuous-exposure comparison — BTC/USDT perp 4h

`state_machine_v1` on `from_signals` against `state_machine_v2` on
`from_orders(size_type="targetvalue")`, over **the same bars §9.2 evaluated**: the 6,048-bar
test half from 2023-10-31 00:00, buy-and-hold +85.78%, 10bp/side, net of funding,
`position_pct=0.95`, 10,000 initial, v2 at its default rebalance band of 0.05. v2 has **no
parameter of its own** — same machine, same policy, same `STATE_TARGET_RISK`, same four
features, same derived warmup (2,160 trained / 2,192 default, verified equal rather than
assumed) — so the difference between the two columns is the boolean collapse and nothing
else.

Every figure below comes from **one** hand-rolled statistics function applied to both
paths' net-of-funding equity curves, not from each engine's own reporting. It reproduces
every published v1 number (trained +15.4543% / +0.8956 / 4.6689% / 73 trades / 13.99% /
funding 21.72; default +15.5161% / +0.7460 / 7.1062% / 153 / 20.83% / 26.25) and agrees
with vectorbt's own estimator to <1e-9; v1's rebuilt book matches the engine's written
`equity_curve.csv` to 1.8e-12 on all 8,208 bars.

| config | path | net % | Sharpe | maxDD % | round trips | fills | turnover (× capital) | funding | time in mkt % | avg expo when held % |
|---|---|---|---|---|---|---|---|---|---|---|
| trained | v1 | +15.45 | **+0.896** | **4.67** | 73 | 120 | 45.2 | 21.72 | 14.0 | 28.4 |
| trained | v2 | +25.89 | +0.842 | 10.85 | 74 | 276 | 82.8 | 62.77 | 14.3 | 47.7 |
| default | v1 | +15.52 | +0.746 | **7.11** | 153 | 211 | 110.3 | 26.25 | 20.8 | 30.9 |
| default | v2 | +36.70 | **+0.913** | 12.55 | 154 | 416 | 164.1 | 90.22 | 21.6 | 53.7 |

v2's fills equal its decision bars **exactly** — 276/276 and 416/416 — one order per
decision at the 0.05 band, which is what the band is for. A fill count on this path is a
count of decisions *at the band the run used*, and is comparable to a boolean path's trade
count only through that number.

**Cost stress** (fees and slippage only — funding is a market rate, decision M8):

| Run | 1× | 2× | 3× |
|---|---|---|---|
| trained v1 | +15.45% | +10.93% | +6.41% |
| trained v2 | +25.89% | +17.61% | **+9.24%** |
| default v1 | +15.52% | +4.49% | **−6.54%** |
| default v2 | +36.70% | +20.18% | **+3.38%** |

v2 survives 3× in both configurations, including the default that kills v1 — the opposite
of the plan's third predicted outcome, which reasoned that a taper raises turnover by
construction and that turnover is what kills an edge at 3×. It carries 2.3× the fills but
only 1.5–1.8× the traded notional, because a resize is an increment, not an
open-plus-close.

**Where the difference comes from**, every bar classified by whether v2 held more, less, or
exactly what v1 held (gross P&L, trained / default):

| | bars | gross P&L |
|---|---|---|
| v2 held **more** — the ramp | 523 / 807 | **+1,445.31** / **+2,783.64** |
| v2 held **less** — the taper | 75 / 77 | **+15.30** / **−63.05** |
| identical | 5,449 / 5,163 | 0.00 |

**The taper — the thing this phase was built to measure — is worth approximately zero.**
What v2 buys is exposure, and the return scales with it.

**Why: v1 never sizes an entry for `RIDING`.** Of v1's size-setting entry bars, **not one
is in `RIDING`**, in either configuration — trained 56 `BREAKOUT` (target 0.35) + 15
`EXHAUSTION` (0.55); default 92 `BREAKOUT` + 20 `CONFIRMED` (0.70) + 40 `EXHAUSTION`. So
the largest position v1 ever opened is **52.3% of capital (trained), 66.5% (default)**
against the **95%** a `RIDING` target asks for, while the machine spent **209 / 516 bars in
`RIDING`**. Entries land above the `BREAKOUT` row readily — the `CONFIRMED` and
`EXHAUSTION` counts are exactly that — but never on the top one: `RIDING` is only ever
reached with a position already open, since the lifecycle passes `BREAKOUT` and `CONFIRMED`
on the way and both already carry a non-zero target, while an entry needs a change of side.
`from_signals` reads size at entry only, so the position is frozen wherever it opened and
cannot scale when the move is confirmed. **The row §2.4 sizes highest is the one row that
has never executed.** Relative to a position frozen at its entry size, the states that
follow are far more often a step *up* than a step down — which is why the "taper" measures
as a ramp. *(A separate check using a cruder funding alignment counted 71/152 entries rather
than 73/153 and 56/92 `BREAKOUT`; the structural claim — zero `RIDING` entries, largest
size 0.55 / 0.70 — reproduced exactly.)* *([§9.4](#94-eth-replication-of-the-r5r6-protocol--ethusdt-perp-4h)
names what that "cruder alignment" was: the `backtest` CLI attaches no `funding_rate`
column at all, so a run through it reads `crowding` as the neutral 0.5 fallback. 71/152 is
that run; 73/153 is the published one.)*

**Per state, trained** (sums to v1's total exactly):

| state | bars | v1 P&L | v2 P&L | v1 avg exposure | v2 avg exposure |
|---|---|---|---|---|---|
| COMPRESSION | 4,989 | −127.00 | −127.00 | 0% | 0% |
| BREAKOUT | 235 | +212.72 | +188.82 | 27.2% | 27.4% |
| CONFIRMED | 137 | +751.75 | +1,319.13 | 27.6% | 55.3% |
| RIDING | 216 | +431.35 | +897.74 | 25.7% | **76.6%** |
| EXHAUSTION | 279 | +276.61 | +315.80 | 29.8% | 38.6% |

v1 sits at 25–30% in *every* state, which is the frozen-at-entry size showing up as a flat
row rather than as a table.

**RIDING → EXHAUSTION episodes** — the taper's own home ground: 24 episodes / 279 bars
(trained), 32 / 338 (default), median 12 bars. Trained v1 +276.61 against v2 +315.80;
default v1 +316.26 against **v2 +195.89** — worse, at higher exposure.

**The charter's premise, measured directly.** Per-bar return on the notional *actually
held*, so per-state size cancels:

| state | trained bp/bar (sd) | default bp/bar (sd) |
|---|---|---|
| CONFIRMED | **+16.6** | **+9.2** |
| RIDING | +6.50 (95.5) | +7.34 (94.8) |
| EXHAUSTION | +5.55 (98.0) | **+7.79** (91.3) |

**Exhaustion was not a worse place to hold risk than riding** in this half — small on one
configuration, reversed on the other. The state that stands out is `CONFIRMED`, which §2.4
sizes *below* `RIDING`. **No per-state t-stat exceeds 1.77.** Recorded as C6.

**Verdict, and what it is not.** None of the plan's three predicted outcomes happened.
Sharpe is a wash and disagrees in sign across configurations (−0.054 trained, +0.167
default), so "improves risk-adjusted return" does not hold; v2 raises return and *worsens*
drawdown 2.3× / 1.8×, which is the inverse of "improves drawdown but costs return"; and
"costs more in turnover than it saves" is refuted outright by the 3× column. R6's real
finding is not about the taper at all — it is that `state_machine_v1` has never sized an
entry for the top row of its own policy table, and R5's published +15.45% was earned at
roughly a third of the risk that policy asked for. That is a defect in the boolean
contract, not in the policy, and it is what R2 warned of when it deferred the taper to R6
rather than writing one against `from_signals`. **The gate is passed either way**: it asks
whether the continuous contract works and whether the taper's contribution is measured,
pass or fail. Both hold, and the taper measuring zero is a result.

**Caveats, all cutting the same way.** **v2 is not risk-matched to v1**, so every column
above except Sharpe is contaminated by the exposure difference — a 1.7× book returning 1.7×
more is arithmetic, not evidence. n is small: 73 / 153 size-setting bars, two
configurations, one asset, one 2.75-year half, and no per-state effect is distinguishable
from zero. And a **split-boundary contract difference**
flatters v2 slightly: v1's *event* contract cannot re-enter a position already live at the
first tradeable bar while v2's *level* contract simply holds it, which accounts for the
74-vs-73 round trip and **+9.07 / +114.23** of gross P&L. Excluded, the ramp/taper split is
+1,436.23 / +15.30 and +2,669.41 / −63.05 — conclusion unchanged. It is an artifact of
starting a run mid-history and would not appear in a whole-history run.

### 9.4 ETH replication of the R5/R6 protocol — ETH/USDT perp 4h

**Pre-registered before any figure existed.** The protocol is
[docs/plans/2026-08-05-eth-replication.md](../plans/2026-08-05-eth-replication.md),
committed at `dd485ec` in its own commit, one commit before the one carrying these
numbers. The two questions, the split, the engine settings, the free extras and the
outcome table were all fixed there; nothing below was adjusted after seeing a result.

**Frame.** ETH/USDT perp 4h, **14,650 bars, 2019-11-27 08:00 → 2026-08-03 20:00**,
anchored at the first stored funding settlement. Split **2023-10-31 00:00**, the same
calendar date as R5, chosen so the test half is **6,048 bars — identical to BTC's**.
Training window is bars **2,352–8,601** (6,250 tradeable; every cell starts at the deepest
of the 54 warmups, the rule R5 used), test **8,602–14,649** (6,048 bars). Engine defaults
throughout: 10bp/side, `--exit-mode opposite_signal_only`, fixed sizing,
`position_pct=0.95`, 10,000 initial, **everything net of funding**. Buy-and-hold is
**+200.51%** in training and **+3.51%** out of it.

**The control ran first**, as R6 established: R5's published BTC row reproduced exactly
(+15.45% / +0.896 / 4.67% / 73 trades) before any ETH figure was read — that is what fixed
the convention the ETH runs were then measured under — and the four headline ETH rows were
independently re-run through `run_backtest` and reproduced to the digit.

**Verdicts, against the pre-registered table:**

| Question | Verdict | Why |
|---|---|---|
| **Q1 — transfer.** BTC's trained cell on ETH, zero parameters re-derived | **PARTIAL** | +3.77% / **+0.184** out of sample. Positive, so the machine is not BTC-specific — but donchian 40/10 returns +0.785 over the same bars and Q1 dies at 3× costs (−7.19%). The pre-registered reading of "positive out of sample but below the baseline" is *partial*: the state machine works, the edge over a channel break does not transfer. |
| **Q2 — protocol replication.** R5's 54-cell search re-run on ETH's training half | **REPLICATES, thinly** | +18.14% / **+0.868** against the baseline's +0.785, surviving 3× costs at +10.53%. It wins by **0.083 of Sharpe while losing on return 4.6-fold** (+18.14% against +84.00%). |
| **The untuned R4 default**, R5's own no-selection-discount evidence | **FAILS** | **−19.58% / −0.563 / 30.56% / 174 round trips**, from +1.09% / +0.081 in training. |

Q2's selected cell is `StateMachine(enter_strength=0.80, exit_strength=⅓, min_dwell=8,
cooldown=8)` at training Sharpe **+1.3356**; the runner-up is the same cell at
`exit_strength=0.20`, +1.3021. **`enter_strength` and `exit_strength` land on BTC's
values; both *timing* axes do not** — `min_dwell` and `cooldown` each move 4 → 8.

**Both halves, net of funding, 10bp/side.** `round trips` is a trade count on either
contract; `fills` on the continuous path counts resizes as well and is not a trade count.

| Run | half | net % | Sharpe | maxDD % | round trips | fills | in mkt % | funding |
|---|---|---|---|---|---|---|---|---|
| donchian 40/10 | train | +58.62 | +0.601 | 38.85 | 115 | 224 | 52.3 | 2628.56 |
| v1 R4-default | train | +1.09 | +0.081 | 14.40 | 170 | 235 | 21.9 | 278.28 |
| v1 BTC-trained (Q1) | train | +19.78 | +0.669 | 8.11 | 74 | 117 | 14.1 | 244.72 |
| v1 ETH-selected (Q2) | train | +35.80 | +1.336 | 5.26 | 44 | 67 | 11.0 | 206.06 |
| v2 R4-default | train | +22.99 | +0.394 | 17.91 | 170 | 432 | 21.9 | 467.34 |
| v2 BTC-trained | train | +32.52 | +0.643 | 12.05 | 74 | 262 | 14.1 | 390.39 |
| v2 ETH-selected | train | +44.73 | +1.016 | 11.19 | 44 | 161 | 11.0 | 303.93 |
| **buy and hold** | train | **+200.51** | | | | | | |
| donchian 40/10 | **test** | **+84.00** | **+0.785** | 31.41 | 102 | 192 | 52.6 | 751.93 |
| v1 R4-default | **test** | −19.58 | −0.563 | 30.56 | 174 | 236 | 22.1 | 53.29 |
| v1 BTC-trained (Q1) | **test** | +3.77 | +0.184 | 14.03 | 82 | 128 | 13.8 | 43.13 |
| v1 ETH-selected (Q2) | **test** | +18.14 | **+0.868** | **5.69** | 57 | 82 | 10.6 | 15.63 |
| v2 R4-default | **test** | −1.40 | +0.061 | 35.77 | 175 | 474 | 22.5 | 83.21 |
| v2 BTC-trained | **test** | +18.74 | +0.505 | 15.15 | 83 | 277 | 14.0 | 59.02 |
| v2 ETH-selected | **test** | +28.92 | +0.872 | 9.96 | 58 | 178 | 10.7 | 29.54 |
| **buy and hold** | **test** | **+3.51** | | | | | | |

**Cost stress, test half** (fees and slippage only — funding is a market rate, M8):

| Run | 1× | 2× | 3× |
|---|---|---|---|
| donchian 40/10 | +84.00% | +63.96% | **+44.01%** |
| v1 R4-default | −19.58% | −33.02% | −46.61% |
| v1 BTC-trained (Q1) | +3.77% | −1.71% | **−7.19%** |
| v1 ETH-selected (Q2) | +18.14% | +14.33% | **+10.53%** |
| v2 R4-default | −1.40% | −21.72% | −41.12% |
| v2 BTC-trained | +18.74% | +9.88% | +1.09% |
| v2 ETH-selected | +28.92% | +23.06% | +17.20% |

**The R0 baseline's own surface, both halves** — the fairness check R5 ran, and ETH's
baseline is the healthier of the two assets':

| donchian surface | positive | median Sharpe | best cell |
|---|---|---|---|
| train | **16/16** | +0.536 | 160/40 at +0.9638 (+145.98%) |
| test | **16/16** | +0.567 | 160/80 at **+1.007** (+121.04%) |

BTC's was 16/16 train and 14/16 test. **`donchian` 40/10 is not ETH's training-selected
cell — 160/40 is**, so under **M11**'s own rule (baseline tuned on the same half by the
same scalar) ETH's honest baseline is 160/40 at test Sharpe **+0.696**. Q2 still beats it;
Q1 still loses. Two things cut against reading "16/16" as breadth. **Five of the 16 cells
duplicate another** — the `exit_span >= entry_span` degeneracy the R0 correction already
recorded — and deduplicated to 13 distinct books the surface is **13/13 positive on both
halves**, median +0.568 train / +0.583 test. And the *scalar* separating neighbouring
cells is partly an artifact; see the warmup-dilution caveat below.

**The 54-cell surface.** **54/54 positive in training.** Marginal means of training
Sharpe:

| axis | values | | |
|---|---|---|---|
| `enter_strength` | 0.55 → +0.424 | ⅔ → +0.310 | **0.80 → +0.826** |
| `exit_strength` | 0.20 → +0.466 | ⅓ → +0.574 | |
| `min_dwell` | 2 → +0.489 | 4 → +0.436 | **8 → +0.635** |
| `cooldown` | 4 → +0.498 | 8 → +0.534 | 16 → +0.528 |

**The optimum sits on an edge in three of the four axes; only `cooldown` is interior.**
The top six cells are exactly the six with `enter_strength=0.80, min_dwell=8` (+1.18 to
+1.34), so this is a real plateau that happens to be the grid's own corner, and the
selected cell reads as "the most selective and slowest setting tried" — the same shape R5
flagged on BTC and the R0 15m gate was rejected for. Note also that **`enter_strength` is
non-monotone on ETH** (⅔ *below* 0.55) where on BTC it rose cleanly (+0.139 / +0.440 /
+1.014): the one axis R5 called dominant does not have the same shape on a second asset.

**R6's mechanism, on a second asset.** **Not one entry is in `RIDING` on ETH's test half,
at any of the three configurations:**

| config | `BREAKOUT` | `CONFIRMED` | `RIDING` | `EXHAUSTION` | bars spent in `RIDING` |
|---|---|---|---|---|---|
| v1 ETH-selected (Q2) | 42 | 8 | **0** | 7 | 104 |
| v1 BTC-trained (Q1) | 55 | 2 | **0** | 25 | 209 |
| v1 R4-default | 83 | 21 | **0** | 70 | 466 |

The largest target at entry is **0.70 → 66.5% of capital, against the 95%** the `RIDING`
row of §2.4 asks for. **So R6's finding is a property of the machine, not of BTC** (M16).

**The continuous contract is worth much more on ETH than R6 measured on BTC.** There, v2
moved test Sharpe +0.896 → +0.842 and +0.746 → +0.913 — a wash that disagreed in sign.
Here it **adds at every configuration** and rescues the default from a losing run:

| config | v1 test | v2 test |
|---|---|---|
| ETH-selected (Q2) | +18.14% / +0.868 | +28.92% / **+0.872** |
| BTC-trained (Q1) | +3.77% / +0.184 | +18.74% / **+0.505** |
| R4-default | −19.58% / −0.563 | −1.40% / **+0.061** |

Round trips stay **within one of v1's everywhere** — same machine, same policy, same
decisions, different sizing — while fills are 2–3× and are resizes, not trades. v2 is not
risk-matched to v1 here any more than it was in R6, so every column except Sharpe carries
the exposure difference with it.

**Frame-start invariance: 0 signal disagreements across all 6,048 test-half bars**, with
the continuous targets agreeing to 1.7e-12. Where a run's frame began does not show.

**Caveats.**

1. **The 16-cell donchian grid is reconstructed, not cited.** The R0 4h gate's grid exists
   nowhere in the repo as a constant. It was recovered from four converging pieces of
   evidence — the stored 15m sweep's committed grid shape, this charter's "swept
   `entry_span` up to 160", every cell the charter names being in it, and its "5 of 16 are
   exact duplicates" reproducing exactly. Strong, but circumstantial. *(— closed, see
   follow-up 1 below: the reconstruction is now pinned as `R0_DONCHIAN_GRID`. It is what
   was reconstructed here, so this remains how the grid was recovered.)*
2. **The selection scalar is diluted by each cell's own warmup.** R5 scores the whole
   window including the leading flat run, and each cell is masked by its own `warmup_bars` within the surface's common frame.
   Measured here: ETH's donchian 40/40 and 40/80 are **provably the same book** — identical
   net return, drawdown and round trips to 6dp, identical positions on all 6,048 tradeable
   bars — and yet they score **+0.8327 against +0.8300**, the entire gap being 40 extra
   flat bars. The 54 machine cells span **2,112–2,352** warmup bars, so the artifact is
   worth up to **~4% of a cell's Sharpe**: enough to reorder neighbours, and it silently
   rewards the shortest warmup. It did **not** change the winner — a tradeable-bars-only
   Sharpe selects the same cell, at +1.5612 — but it is a defect in the ranking rule. M21.
3. **Five of the 16 donchian cells fall into two duplicate groups**, so "16/16" overstates breadth;
   13/13 on 13 distinct books is the honest form.
4. **The two halves are the same calendar, not the same market.** ETH's test half is
   nearly flat — buy-and-hold **+3.51%**, against **+200.51%** in its own training half and
   BTC's +85.78% over the identical dates. The pre-registration's "a difference between
   them is the asset rather than the regime" is **only half true**: it is the asset *and*
   the asset's own regime. This is the caveat that most limits the comparison.
5. **Q2's win is one scalar by 0.083, on a strategy returning a fifth of the baseline.** A
   reader weighting return over Sharpe reads it as a loss.
6. **Two undeclared extras, neither feeding any selection**: `state_machine_v2` at the
   ETH-selected cell, and the pre-registration's declared **secondary split** at ETH's own
   60/40 boundary — 2023-12-01 08:00, 5,860 test bars, buy-and-hold **−11.16%**: donchian
   +86.04 / +0.822, Q1 +4.01 / +0.197, Q2 +18.14 / +0.878, default −14.81 / −0.451.
   **Every verdict is unchanged, so the split choice is not doing work** — which is what
   the secondary split was declared for.

**A reproducibility defect found while verifying.** The `backtest` CLI **never attaches a
`funding_rate` column to the frame**: it passes funding to `run_backtest` for cost
accounting only, where the `features` command does
`df.assign(**{FUNDING_COLUMN: align_funding_to_bars(...)})`. So `state_machine_v1` and
`state_machine_v2` run from the CLI read `crowding` as the neutral **0.5 fallback** and
record `crowding_measured=False`. Measured on BTC's trained cell:

| BTC trained cell | net % | Sharpe | maxDD % | trades |
|---|---|---|---|---|
| **without** the funding column (what the CLI does) | +16.44 | +0.801 | 6.08 | 71 |
| **with** it (what §9.2 publishes) | **+15.45** | **+0.896** | **4.67** | **73** |

**Every published state-machine figure was produced with the column, and the documented
canonical command does not reproduce them.** The research is not wrong; the CLI is out of
step with it. It also explains §9.3's own entry-count note — 71/152 rather than 73/153 —
which was that check run without the column. Recorded as a known defect (M20), **not fixed
here**: a branch that changes the engine cannot also be the branch that measures it.

**Follow-ups, recorded and not done** — this branch changes no code. Two have since been
closed by later ones and are marked rather than deleted, because what a replication left
open, and what eventually closed it, is part of the record:

1. Pin the R0 4h donchian grid as a constant, so a surface is cited rather than
   reconstructed. — **Done** on `research/r9-robustness`: `R0_DONCHIAN_GRID` in
   `tests/test_sweep.py`, with caveat 3 above *derived* from it rather than stated — the
   16 cells are built, grouped by the position each would have held, and have to collapse
   to the 13 books and the two duplicate groups measured here. The R5 gate's baseline cell
   is checked against the same constant.
2. Attach `funding_rate` in the `backtest` CLI the way `features` does, so the canonical
   command reproduces the published state-machine figures. — **Done**, PR #13 (`0bea797`),
   merged. That closes M20's *defect*; M20's reading — that the published figures are the
   correct ones and the CLI was what disagreed with them — stands unchanged.
3. Score a selection scalar over tradeable bars only (M21). — **In force for R9's own
   selections** ([§9.5](#95-r9-walk-forward-and-robustness--btcusdt-perp-4h)), where it
   ranked all 54 cells on the training half and on every walk-forward fold, and **never
   once disagreed with R5's rule about a winner**. Still **not wired into**
   `sweep_parameters` or the gate test — R9 computes it in its own harness from the
   engine's own `equity_curve.csv`. That wiring stays open.

### 9.5 R9 walk-forward and robustness — BTC/USDT perp 4h

**Pre-registered before any figure existed.** The protocol is
[docs/plans/2026-08-06-r9-robustness.md](../plans/2026-08-06-r9-robustness.md), committed
at `78eaf55` in its own commit, one commit before the scripts and two before this section.
The frame, the grid, the selection scalar, the walk-forward geometry and the outcome table
were fixed there; nothing below was adjusted after seeing a result. `scripts/r9/` is the
harness — it reuses the gate's own `evaluate`, selects nothing, and writes nothing outside
a temp root. The whole pipeline reproduces byte-identically from the committed copies.

**The control ran first.** R5's published row reproduced to the digit — **+15.454329% /
+0.8956 / 4.6689% / 73 trades** against the published +15.45% / +0.896 / 4.67% / 73 — with
the funding column attached, M20's distinction and the difference between auditing the
published figure and auditing a different one. Two further controls: the 54-cell training
surface re-selects `enter_strength=0.80, exit_strength=⅓, min_dwell=4, cooldown=4` at
**+1.2155** (+30.52%, 5.25% DD, 77 trades), and **50/54** cells are positive in training.

**The finding, and the pre-registered table has no row for it: the cell survives; the
procedure that picked it does not.**

**1 · Deflated Sharpe on the training selection.** Bailey and López de Prado, over the 54
training-half trials. Reported under both scalars because they are two rankings of the same
54 runs; the deflation runs in per-bar units, since the skew and kurtosis terms are only
consistent at the frequency the Sharpe is measured at.

| | R5's rule (whole window) | M21's rule (tradeable bars) |
|---|---|---|
| trial min / median / max | −0.3704 / +0.4607 / **+1.2155** | −0.4281 / +0.5295 / **+1.3974** |
| trial sd | 0.4197 | 0.4842 |
| runner-up | +1.1882 | +1.3734 |
| **`E[max SR]` under the null** | **+0.9676** | **+1.1164** |
| winner's returns: skew / excess kurtosis / T | +5.55 / +142.7 / 8,878 | +4.81 / +107.2 / 6,718 |
| z | +0.5318 | +0.5244 |
| **DSR** | **0.7026** | **0.6999** |

**Both readings, because the pre-registration's condition admits both and declared no
threshold.** The observed **+1.2155 clears `E[max SR]` +0.9676**, and so do **10 of the 54
cells — every one of them at `enter_strength=0.80`**, so the winner is not a lone spike.
But the **DSR itself is 0.70** against the conventional 0.95: at conventional confidence
the best-of-54 in-sample Sharpe is *not* distinguishable from the luckiest of 54 zero-Sharpe
trials. The first reads as outcome row 1, the second as row 2. Both stand.

It is also **frequency-sensitive, because the winner's curve is 84.9% exact zeros** — the
machine is out of the market most bars, which is what drives the excess kurtosis to +142.7.
Aggregating the same returns to daily (1,120 days, excess kurtosis +29.27) lifts the DSR to
**0.7697**, still short of 0.95. The frequency was **not fixed by the plan** and was chosen
after the fact; it moves the number without moving the reading, and is why M23 requires a
declared frequency next time.

**What it prices, restated:** the *in-sample selection*. It does **not** deflate the +0.896,
which was one evaluation of a pre-committed cell on data not used to select it.

**2 · Walk-forward.** Geometry exactly as declared — expanding training from bar 2,352,
minimum 3,000 training bars, non-overlapping 1,008-bar test blocks, winner re-derived from
the 54 on each fold's training data and evaluated **once** on its test block. **Nine** full
folds fit; **694 bars (2026-04-10 → 2026-08-03) fall past the last block and are not
evaluated.** Sharpe is the tradeable-bars scalar throughout this table.

| fold | train bars | test block | selected cell | OOS Sharpe | net % | trades | B&H % |
|---|---|---|---|---|---|---|---|
| 1 | 3,000 | 2022-02-18 → 08-05 | 0.80 / 0.20 / 4 / 4 | +1.5043 | +5.13 | 13 | −42.3 |
| 2 | 4,008 | 2022-08-05 → 2023-01-20 | 0.80 / 0.20 / 4 / 4 | −1.5271 | −3.90 | 17 | −10.4 |
| 3 | 5,016 | 2023-01-20 → 07-07 | 0.80 / ⅓ / 2 / 16 | +1.1516 | +1.71 | 8 | +43.1 |
| 4 | 6,024 | 2023-07-07 → 12-22 | 0.80 / ⅓ / 2 / 16 | +1.8900 | +4.55 | 11 | +44.9 |
| 5 | 7,032 | 2023-12-22 → 2024-06-07 | **0.80 / ⅓ / 4 / 4** (R5's) | +1.2727 | +3.44 | 11 | +62.5 |
| 6 | 8,040 | 2024-06-07 → 11-22 | **0.80 / ⅓ / 4 / 4** (R5's) | +0.9157 | +2.52 | 21 | +38.3 |
| 7 | 9,048 | 2024-11-22 → 2025-05-09 | 0.80 / ⅓ / 4 / 8 | −0.2218 | −0.69 | 18 | +4.7 |
| 8 | 10,056 | 2025-05-09 → 10-24 | 0.80 / ⅓ / 4 / 8 | −1.4502 | −2.31 | 10 | +7.9 |
| 9 | 11,064 | 2025-10-24 → 2026-04-10 | 0.80 / ⅓ / 4 / 8 | +2.6500 | +7.12 | 6 | −35.6 |

**6/9 positive**, mean **+0.6872**, median +1.1516, sd 1.4524, t = 1.42; compounded
**+18.39%** against buy-and-hold's **+75.17%** over the same nine blocks; the chained
9,072-bar OOS curve scores **+0.7634**.

**How often the same cell wins — the question the plan said was itself a result: four
distinct cells in nine folds, and R5's own cell wins two of them.** `enter_strength = 0.80`
wins **all nine**; the other three axes move. Two qualifications, cutting opposite ways.
In folds 1 and 2 the top two cells score **exactly equally** — a gap of 0.00e+00 in float64,
because `exit_strength` is inert over those windows and they are the same book — so `max()`
broke the tie by grid iteration order;
the gate's own "the top two tie, so the selection is arbitrary" assertion would **fail**
there, and in fold 2 the tie-broken pair then **diverged out of sample by 1.9 pp of net**.
Counting *books* rather than cells, the choice equals R5's pinned cell on folds 1, 5, 6 and
8 — fold 8 picks a **different cell** that produces the **same book**, agreeing to 4.0e-14
of Sharpe and 8.9e-14 of net return, so the genuinely different set is the five below.

**The declared extra, and it is what inverts the reading. R5's pinned cell, re-derived
nowhere, evaluated on the same nine blocks:**

| | folds positive | mean Sharpe | compounded net | chained Sharpe |
|---|---|---|---|---|
| re-derive every fold (walk-forward) | 6/9 | +0.6872 | +18.39% | +0.7634 |
| **R5's pinned cell, selecting nothing** | **7/9** | **+1.1246** | **+31.38%** | **+1.2228** |

In **all five** folds where re-derivation picked a genuinely different book, re-derivation
was **worse** out of sample — fold 2 −0.72 Sharpe / −1.91 pp, fold 3 −0.34 / −0.87, fold 4
−1.24 / −2.46, fold 7 −0.86 / −2.26, fold 9 −0.79 / −3.28. Not one exception. Note that
folds 1–3 test inside R5's own training half and fold 4 straddles the split, so only folds
5–9 sit on bars R5 never selected on; every fold still trains on strictly earlier bars.

**3 · Parameter perturbation.** The winner is on an **edge in three of four axes**
(`enter` top of 3, `exit` top of 2, `cooldown` bottom of 3); only `min_dwell` is interior,
so there are five in-grid neighbours, not eight. Sharpe here is the whole-window scalar, to
stay comparable with [§9.2](#92-r5-split-sample-gate--btcusdt-perp-4h)'s published row.

| move | train Sharpe | train net | test Sharpe | test net | test DD | test trades | 3× costs |
|---|---|---|---|---|---|---|---|
| **the R5 winner** | **+1.2155** | +30.52% | **+0.8956** | +15.45% | 4.67% | 73 | **+6.41%** |
| `enter` 0.80 → ⅔ | +0.4689 | +15.20% | +0.7078 | +15.45% | 8.59% | 167 | **−8.59%** |
| `exit` ⅓ → 0.20 | +1.1439 | +28.61% | +0.8956 | +15.45% | 4.67% | 73 | +6.41% |
| `dwell` 4 → 2 | +1.0179 | +28.94% | +0.7081 | +12.98% | 7.57% | 93 | +0.84% |
| `dwell` 4 → 8 | +0.9334 | +27.48% | +0.5163 | +8.05% | 5.56% | 54 | +0.49% |
| `cooldown` 4 → 8 | +1.1609 | +29.76% | +0.8501 | +14.02% | 4.64% | 67 | +5.61% |

**`enter_strength` is a ridge, not a plateau.** One step down costs 0.75 of training Sharpe
and lands on **the same test-half net return to two decimals** (+15.4489% against
+15.4543%) — at **1.8× the drawdown, 2.3× the turnover, and −8.59% at 3× costs against
+6.41%**. A reader looking only at net return would call the two settings equivalent; every
other column says otherwise. The other three axes degrade gracefully. `exit_strength`
⅓ → 0.20 is a **no-op out of sample** — identical equity on all 8,208 bars, max absolute
difference 0.000e+00 — while differing in training, which is the same inertness the fold-1
and fold-2 ties showed.

**4 · Feature dropout.** Each feature held at its neutral value in the space the machine
reads it (`direction` 0.0; `strength`/`stability` 0.5 as trailing ranks; `crowding` 0.5 =
`NEUTRAL_CROWDING`), warmup rows left NaN. Validated both ways before any row was read: the
wrapper dropping nothing reproduces the control exactly, and the `crowding` row equals the
unmodified strategy over a frame carrying no `funding_rate` column at all.

| dropped | train Sharpe | train net | test Sharpe | test net | test DD | test trades | 3× |
|---|---|---|---|---|---|---|---|
| none | +1.2155 | +30.52% | +0.8956 | +15.45% | 4.67% | 73 | +6.41% |
| **`direction`** | — | 0.00% | — | 0.00% | — | **0** | 0.00% |
| **`strength`** | — | 0.00% | — | 0.00% | — | **0** | 0.00% |
| `stability` | **+1.2404** | +35.46% | +0.6827 | +14.32% | 6.69% | 83 | +3.91% |
| `crowding` | +1.0229 | +32.18% | +0.8012 | +16.44% | 6.08% | 71 | +5.95% |

- `direction` and `strength` are **structural gates, not statistical inputs**: `advancing`
  requires `strength ≥ enter_strength` *and* `|direction| ≥ direction_floor`, so
  neutralising either leaves the machine in `COMPRESSION` forever. Sharpe is **null, not
  zero** — there is no cost to report, only a dependency.
- `stability` is **the one component whose contribution has the opposite sign in each half**:
  dropping it *raises* training Sharpe (+1.2155 → +1.2404) and *lowers* test Sharpe
  (+0.8956 → +0.6827). So the hard exit on a stability collapse looks like dead weight on
  the half the cell was chosen on, and is worth 0.21 of Sharpe on the half it was not —
  the reverse of the usual in-sample-flatters shape, and a reason not to prune a component
  on training evidence.
- `crowding` costs 0.094 of test Sharpe while *adding* 1.0 pp of net return, reproducing
  M20's **+16.44 / +0.801 / 6.08% / 71** to the digit by a second route.

**5 · Purging and embargo — stated, not performed**, as the plan declared, with both halves
of the argument re-checked against the code rather than restated. `state_machine_v1` is in
`list_strategies()`, so both safety suites cover it; it fits nothing and has no labels —
`StateMachine` is a fixed recursion and `state/policy.py` a constant table plus two rank
bands — so there is nothing to purge. The one boundary effect, machine state carrying
across a split, was re-measured: the whole-history and test-window runs agree on **73 of 73**
entry timestamps after 2023-10-31. They become mandatory at R7/R8, where a fitted model
does have labels and a forward horizon.

**M21 never disagreed with R5's rule about a winner** — not on the training half, not on any
of the nine folds. On the training half the dilution factor spans 0.8606–0.8723 (warmups
2,112–2,352) and 4 of 54 cells move rank, maximum displacement 1; per fold, 4–8 cells move,
maximum displacement 2, top-5 overlap 5/5 in eight folds and 4/5 in fold 1. R5's published
selection was not rescored.

**Caveats.**

1. **N = 54 is a lower bound on trials, and the 54 are correlated.** The program searched
   more than this grid over its life — the pre-correction machine, the feature set, exit
   ownership — so the DSR is an **upper** bound on confidence. Cutting the other way, 10 of
   the 54 clear `E[max SR]` and all sit at one `enter_strength`, so treating them as 54
   independent draws overstates the breadth and pushes `E[max SR]` **up**. Both biases are
   present and neither is resolvable from this experiment.
2. **Nine folds is nine observations** (t = 1.42 walk-forward, 2.12 pinned), and one fold's
   selection was decided by an arbitrary tie-break.
3. **Expanding windows only.** A rolling-window walk-forward is a different experiment; it
   was neither declared nor run.
4. **The `direction` and `strength` dropout rows are degenerate by construction** and
   should not be read as "these features are worth infinite Sharpe".
5. **The pre-registration's outcome table was underdetermined**, which is a defect in the
   protocol rather than in the result — see M23. It declared a comparison ("DSR clears the
   null's expected maximum") without a threshold, and the two readings of that phrase land
   in different rows.

### 9.6 R7 chop/trend state diagnosis — BTC/USDT perp 4h, replicated on ETH

**Pre-registered before any figure existed**, at `ddca7dc`, one commit before the scripts.
The frame, the label, the anchoring, the horizons, **the thresholds as numbers** and the
outcome table were fixed there. `scripts/r7/` is the harness; it selects nothing.

**The control ran first, and it is a stronger one than R9's.** R4's conditional IC table
reproduced through `diagnose_features` — the production path, not a reimplementation — to
four decimals on its published n of 13,167: unconditional **+0.0385**, low **+0.0022**, mid
**−0.1128**, high **+0.1314**, halves included. Extended unasked to **all 81 IC cells of
§9.1** plus every bar count, warmup and raw tercile boundary: **114/114 match**.

**The verdict is the pre-registered third row on BTC — nothing clears, `COMPRESSION`
included — but the honest one-line summary is narrower and more useful: the machine's
states carry no chop information on either instrument, and the one feature that does is a
feature the machine does not read.**

**The label, audited before any figure was read** rather than only in the fourth-outcome
case, because an audit conditional on liking the answer is not an audit. Poison probe:
replacing every bar ≤ *t* leaves `ER[t]` bit-identical at all three horizons; replacing
`close[t+1]` moves it. The probe is shown to work by positive control — a null feature
reading the print error scores IC **−0.395 … −0.453** anchored at `close[t]` against
+0.005 … −0.031 at `close[t+1]`. **But the anchor buys far less on ER than on a return,
which the plan did not anticipate**: ER is a *ratio*, so bar *t*'s print error inflates
numerator and path sum together and largely cancels — no feature's ER IC moves by more than
**0.0046** between anchors. The rule stays mandatory; its value here is small. ER is also
confirmed **scale-free** (median moves 0.0048 across a 25× range of σ on three independent
walks; 0.0015 if one path is held fixed and only σ scaled), which is what makes
the `energy` result below a market fact rather than a units artifact.

**1 · The incumbent.** Lift = trend rate inside the state minus the base rate; the bar was
≤ −10 pp in **both** halves.

| config | H=6 train / test | H=30 train / test | H=90 train / test |
|---|---|---|---|
| R5-trained | +0.44 / +0.11 | +1.35 / −0.86 | +0.33 / −0.72 |
| R4-default | −0.28 / +0.36 | +0.61 / −1.79 | −0.00 / −0.47 |

**0/6.** The best deficit anywhere is **−1.79 pp against a −10 pp bar**, and the sign is
positive as often as not. `COMPRESSION` covers **82.9%** of live bars, which pins its lift
near the base rate by arithmetic — so the complement was measured too rather than left as
an objection, and it does not rescue the result: inside-vs-outside separation at H=30 is
**+8.08 pp in training and −4.89 pp in test**, disagreeing in sign. Nor does the lifecycle
ordering replicate: `RIDING` at H=30 is −8.10 train / +9.28 test, and at H=90 it is
−4.80 / −9.14 — the *ride* state predicting **less** forward efficiency in both halves.

**2 · The nine features against forward ER, H = 30.** Forward windows overlap *H* ways, so
2 sd of a null IC is **0.0399 / 0.0892 / 0.1548** at H = 6 / 30 / 90 — the declared 0.10 bar
at H=30 is ≈2.24 sd, which is a real bar; at H=90 it would sit *inside* the noise band.

| feature | IC@30 (train / test) | verdict |
|---|---|---|
| `direction` | +0.0243 (+0.0541 / −0.0095) | fail — halves disagree |
| `strength` | −0.0465 (−0.0916 / +0.0167) | fail — halves disagree |
| `persistence` | −0.0069 (−0.0256 / +0.0249) | fail — halves disagree |
| `stability` | +0.0552 (+0.0696 / +0.0305) | fail — weaker half 0.031 |
| **`energy`** | **−0.0906 (−0.0800 / −0.1033)** | **fail on \|IC\| alone, by 0.0094** |
| **`compression`** | **+0.0906 (+0.0800 / +0.1033)** | **fail on \|IC\| alone, by 0.0094** |
| `compression_release` | +0.0106 (+0.0110 / +0.0098) | fail |
| `participation` | −0.0225 (−0.0356 / −0.0063) | fail |
| `crowding` | −0.0111 (+0.0017 / −0.0334) | fail — halves disagree |

**0/9.** `energy`/`compression` are one measurement under two names (`r = −1.000` by
construction) and are the only row that passes the sign test **and** the both-halves test,
missing the headline bar by 0.0094 — **2.03 sd against the bar's 2.24**. Note the sign runs
**opposite to the name's connotation**: IC(`compression`, ER) is *positive*, so low
volatility precedes *cleaner* forward moves and the chop side of that axis is **high
`energy`**.

**3 · The composite gate.** `strength ≥ 0.80` (rank space) `AND |direction| ≥ 0.10`:

| gate | coverage | H=30 trend rate, train / test |
|---|---|---|
| `strength ≥ 0.80` | 21.7% | 0.2513 / 0.3385 |
| `\|direction\| ≥ 0.10` | 82.2% | 0.3152 / 0.3107 |
| composite | 21.6% | 0.2508 / 0.3385 |

**0/3, and the reason is structural rather than statistical: the composite *is* the strength
gate.** It beats the strength gate alone by **+0.00 pp on the test half at all three
horizons**. `direction_floor = 0.10` admits 82% of bars and removes essentially nothing
`strength` has not already removed — so **the gate the whole machine rests on is one
feature, not two**. (`direction_floor` was never swept by R5, so "R5's trained value" for it
is the dataclass default; that is a fact about the grid, not a choice made here.)

**4 · The `enter_strength` extension — R9's lead, and it closes.** Control first: the
trained cell re-scores at M21 **+1.3974** and whole-frame **+1.2155**, R9's published pair
exactly.

| cell | M21 Sharpe | net % | maxDD % | trades | 3× costs |
|---|---|---|---|---|---|
| **0.80 (R5's)** | **+1.3974** | +30.52 | 5.25 | 77 | +20.39 |
| 0.85 | +0.5002 | +10.42 | 7.11 | 58 | +3.56 |
| 0.90 | +0.5577 | +9.55 | 5.30 | 41 | +4.74 |
| 0.95 | −0.7497 | −9.30 | 10.86 | 19 | −11.32 |

**0/3 beat the incumbent — `enter_strength = 0.80` is an interior optimum of the wider
grid, not a boundary artifact.** This closes a question flagged separately in R5, in the ETH
replication and in R9 (M26). Nothing was pinned or re-derived; R5's published cell is
unchanged whichever way this came out (M22).

**5 · Persistence.** The declared population is **empty** — nothing cleared §1–§3 — so this
is context for whatever follows R8, explicitly not a threshold evaluation. `COMPRESSION`
runs 46/63 bars median with AC(1) +0.95; the composite gate runs **4 bars** and would fail
the ≥6 usability bar it was never subjected to. The `energy` top tercile — the chop side —
runs 19/17 bars and is **sign-consistent at every horizon in both halves** (−3.74 / −4.65 /
−2.01 train, −1.24 / −5.43 / −7.58 test): far short of 10 pp, never the wrong sign.

**The ETH replication, declared in the plan and run after BTC.** Same label, horizons,
terciles and thresholds, on §9.4's frame. **The null does not fully replicate, and the
exception is the same feature**: `energy`/`compression` **clear on ETH** at |IC| **0.1521**
at H=30 (halves −0.1164 / −0.2002) and −0.1314 at H=90. `COMPRESSION` still clears **0/9**
across three published configurations, and the composite **0/3**. §4 was **not** run on ETH
— its threshold is BTC's training-half scalar, and inventing an ETH bar after the fact is
exactly M23's failure.

**Why this is a lead rather than a dead end, and it is [§2.1](#21-the-state-vector)'s own
words.** The declared state vector has **nine** dimensions. `state_machine_v1` reads
**four** — `DEFAULT_FEATURES` and `REQUIRED_COLUMNS` are both
`(direction, strength, stability, crowding)` — and `energy`, `compression`,
`compression_release`, `participation` and `persistence` are simply absent from it. R7's one
signal-carrying feature is in the dropped five. Worse, or better: §2.1's own worked example
**defines chop in terms of the missing axis** —

```
Strength = 20, Energy = 95   →  violent two-way chop
```

— and R7 measured exactly that relationship (high `energy` → low forward efficiency) on two
instruments. The state named `COMPRESSION` and the feature named `compression` share a word
and nothing else. **So the finding is not that §2.1's state vector is wrong; it is that the
machine implements four ninths of it and dropped the axis §2.1 uses to define the thing the
program exists to detect.**

**Caveats.**

1. **`energy`'s BTC result is a near-miss under a bar set before the data was seen, and it
   stays a miss.** It is reported as failing, and the ETH pass is what makes it a lead
   rather than a rounding argument. A bar moved after the fact would be M23 again.
2. **The plan's "both halves" was underdetermined** — the frame block fixes 2023-10-31 while
   §2 calls itself "R4's table with a different target", and R4 cuts at its own aligned
   midpoint. Both were computed; **the verdict is identical either way**, and the declared
   split is primary.
3. **The thresholds are in IC units and the forward windows overlap.** At H=30 the 0.10 bar
   lands at ≈2.24 sd, which is defensible, but that was luck rather than design — the next
   pre-registration declares the noise convention beside the number.
4. **§5 had no instruction for an empty population** and §1 did not say which machine
   configuration to score. Both published configurations were used; verdict identical.
5. **Four cells of §9.1 are double-rounded** — the 3-decimal table was rounded from a
   4-decimal intermediate, so `direction@30` prints +0.039 for a measured +0.03848,
   `energy@1`/`compression@1` print ±0.010 for ±0.00949, and `participation@6` prints +0.016
   for +0.01548. Transcription, not statistic; noted, not corrected, since the published
   table is what was published.
6. **The training half's last `H+1` labels read test-half closes, and no embargo was
   declared or applied.** `ER[t]` reaches `close[t+1+H]`, so 7 / 31 / 91 of the 7,150
   training rows at H = 6 / 30 / 90 have a forward window crossing the split. Measured by
   dropping exactly those rows and recomputing: the largest train-half IC change across
   the machine's features is **0.0052 at H=30 and 0.0078 at H=90**, and **not one verdict
   moves** at any horizon — the effect is an order of magnitude below the 0.10 bar and well
   inside H=30's own 0.0892 noise band. Left in place rather than fixed, because
   `forward_return` has the identical structure and §9.1's published half-sample ICs carry
   it too: embargoing R7 alone would make its ICs incomparable with the table it was
   validated against. A declared embargo belongs in the next pre-registration, with M23's
   other omissions.

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
| M9 | Redundancy is judged on marginal information, not on a correlation threshold | Measured in R4: `participation` retains 26% of its IC once `energy` is stripped out, at a correlation of only +0.363. A \|r\| > 0.9 flag catches an exact duplicate and nothing else, so R5 screens candidate inputs by what each still says once the others are known. | 2026-08-04 |
| M10 | A feature earns its place as a conditioner or as a predictor, and the two are judged differently | Measured in R4: no feature reaches \|IC\| 0.07 alone, but `direction`'s IC@30b triples inside `strength`'s top tercile. Ranking the state vector by univariate IC would have cut the half of it that works. | 2026-08-04 |
| M11 | A baseline is tuned on the same half, by the same rule, as the thing being compared to it | R5: donchian 40/10 is both the cell this charter already named *and* the best of the R0 gate's 16 cells on the training half — so the comparison is like-for-like by construction rather than by luck. The alternative, holding the baseline at a full-sample choice while the challenger is trained on 60%, hands the baseline the test half. | 2026-08-04 |
| M12 | A stable IC is not a tradeable one, and P&L is what settles it | R5: R4's mid-`strength` tercile had the larger absolute IC (−0.113) with *both halves agreeing*, and the high tercile was the one flagged as decaying. Out of sample the follow band produced **+100.0%** of the machine's PnL and the fade band **+0.0%** (+0.30 on +1,567.15, at a 31.6% win rate). Cross-sectional rank correlation and a traded book disagreed about which regime was real; the book is what the program is gated on. | 2026-08-04 |
| M13 | `compression_release` stays measured and unwired | R5 judged it on the bars where it fires, which R4 could not: the timing lift (6.73% vs a 4.20% base rate) does not replicate out of sample (**4.95% vs 4.66%**), and the payoff split disagrees between halves rather than agreeing either way. It marks a short-lived volatility expansion with no direction. Kept in `features/registry.py` because it is measured and cheap, wired into nothing, per §8.7. | 2026-08-04 |
| M14 | The boolean and continuous contracts **coexist**; neither is migrated to the other, and vectorbt is not retired | `from_orders(size_type="targetvalue")` expresses the level a `SignalSet` cannot, measured before planning (a 10-bar taper gives 6 orders where `from_signals` gives 1), so the gate's "vectorbt replaced for this path" is satisfied by a second execution path rather than by leaving vectorbt (Q4). The four original strategies stay on `SignalSet`: their results are research of record and byte-identity has been a hard constraint since R2. `state_machine_v1` stays registered and unchanged for the same reason — having both is what made the taper's contribution measurable at all, and R6's finding came out of the comparison, not out of either run alone. This is M1's "two strategy contracts" arriving. | 2026-08-04 |
| M15 | The continuous path sizes from **initial cash** and trades only past a **rebalance band** | `targetpercent` is a fraction of *current* equity, which compounds and silently broke CLAUDE.md's non-compounding sizing rule the moment this path existed. Measured on a 3,000-bar synthetic whose target changes 41 times: `targetpercent` 1,989 orders / 9,049 final against `targetvalue` at initial cash 1,861 / 11,305 — both churn, so **compounding is not the cause, price is**: holding a constant fraction means selling into a rise. Hence the band, and it is a statement about what the strategy *is* rather than a cost optimisation — band 0.0 rebalances every bar, trimming winners and adding to losers, a mean-reversion overlay on a thesis of riding trends, and it moves the result **before a single fee is charged** (20,742.99 at the 0.05 default against 20,261.47 at 0.0, costless and funding-free). The band is measured against the last target *submitted*, never the realized fraction — that would define the band in terms of its own output. | 2026-08-04 |
| M16 | The taper measures zero; the *ramp* is what earns, and §2.4's `RIDING` row has never executed | R6 on R5's test half. v2 held more than v1 on 523 / 807 bars for **+1,445.31 / +2,783.64** gross and less on 75 / 77 bars for **+15.30 / −63.05**. The cause is structural: not one of v1's size-setting entry bars is in `RIDING` in either configuration, so the largest position it ever opened is 52.3% / 66.5% of capital against the 95% a `RIDING` target asks for, on a machine that spent 209 / 516 bars there. Entries reach the `CONFIRMED` and `EXHAUSTION` rows; `RIDING` is only ever reached with a position already open — the lifecycle passes `BREAKOUT` and `CONFIRMED` first and an entry needs a change of side — and `from_signals` reads size at entry, so the position freezes wherever it opened and the "taper" measures as a ramp. Sizing research after this (R7) aims at the ramp; the taper is measured and not worth a rule. | 2026-08-04 |
| M17 | Continuous-exposure strategies get their **own** registry — a third one | Checked before writing it, then mutation-tested: **six** parametrized tests across `test_lookahead.py`, `test_replay_determinism.py` and `test_strategy_metadata.py` iterate `strategies.registry.list_strategies()` and every one calls `generate_signals`, which an exposure strategy does not have. Listing one there errors at best and is skipped by a `getattr` guard at worst. Measured the failure directly: an **empty** `exposure_registry` silently **skips 4 parametrized tests and exits 0** — a gate that passes by not running, which is §8.1's failure in its most expensive form. `strategies/exposure_registry.py` is manual in two places like the other two registries, and the continuous determinism, lookahead and warmup suites iterate it. | 2026-08-04 |
| M18 | A **method** transfers across assets; a **cell** does not, so a new instrument gets the protocol re-run rather than the parameters copied | The ETH replication ([§9.4](#94-eth-replication-of-the-r5r6-protocol--ethusdt-perp-4h)) asked both questions separately, on purpose. Q1 — BTC's trained cell carried across with **zero** parameters re-derived — is **partial**: +3.77% / Sharpe **+0.184** out of sample against donchian 40/10's +0.785, dying at 3× costs (−7.19%). Q2 — R5's own 54-cell search re-run on ETH's training half — **replicates**: +18.14% / **+0.868**, beating that baseline by 0.083 of Sharpe and surviving 3× at +10.53%. The selected cell agrees with BTC on `enter_strength` and `exit_strength` and disagrees on **both timing axes** (4/4 → 8/8). A published cell is therefore an artifact of the asset it was searched on, and copying one across is not a transfer test this program passes. | 2026-08-05 |
| M19 | An **untuned** configuration is not evidence that a result is unsearched | R5 leaned on the R4-default machine passing out of sample (+15.52% / +0.746) as proof its verdict carried no selection discount. On ETH the same untuned default returns **−19.58% at Sharpe −0.563** with a 30.56% drawdown over 174 round trips, from +1.09% / +0.081 in training. The default's thresholds come from **BTC's** R4 terciles, so it was never unsearched — it is searched on the first asset and merely not re-searched on the second. R5's claim stands for BTC and is **withdrawn as a general one**: a verdict that needs its own asset's training half carries the full selection discount. | 2026-08-05 |
| M20 | The `backtest` CLI does not reproduce the published state-machine figures, and the **figures** are what is correct | The CLI never attaches a `funding_rate` column to the frame — it passes funding to `run_backtest` for cost accounting only, where the `features` command does `df.assign(**{FUNDING_COLUMN: align_funding_to_bars(...)})`. `state_machine_v1`/`v2` therefore read `crowding` as the neutral 0.5 fallback and record `crowding_measured=False`. Measured on BTC's trained cell: without the column **+16.44% / +0.801 / 6.08% / 71 trades**, with it **+15.45% / +0.896 / 4.67% / 73** — the second reproducing §9.2 to every published digit, so every published figure was produced with the column and the documented canonical command was not. It also explains §9.3's 71/152-vs-73/153 entry-count note. Recorded as a defect with a follow-up rather than fixed inside a replication: a branch that changes the engine cannot also be the branch that measures it. | 2026-08-05 |
| M21 | A selection scalar is scored over the bars a cell can **trade**, not over the frame it was handed | Measured on ETH: donchian 40/40 and 40/80 are provably the same book — identical net return, drawdown and round trips to 6dp, identical positions on all 6,048 tradeable bars — yet score **+0.8327 against +0.8300** under R5's rule, the whole gap being 40 extra leading flat bars, since each cell is masked by its own `warmup_bars` within the surface's common frame. The 54 machine cells span 2,112–2,352 warmup bars, so the artifact is worth up to **~4% of a cell's Sharpe** — enough to reorder neighbours, and it silently rewards the shortest warmup. It did **not** change this selection (a tradeable-bars-only Sharpe picks the same cell, +1.5612), and R5's published selection is **not** rescored: rescoring a selection after its test half has been seen is the thing the protocol exists to prevent. So this binds the **next** selection and restates no past one; wiring it into `sweep_parameters` and the gate test is the open follow-up below, not something already in force. | 2026-08-05 |

| M22 | The **cell** survives out of sample; the **procedure that selected it** does not, so periodic re-selection is a cost rather than a control | R9's walk-forward ([§9.5](#95-r9-walk-forward-and-robustness--btcusdt-perp-4h)), nine expanding folds, winner re-derived from the same 54 on each fold's own training bars and evaluated once. Re-deriving: **6/9 folds positive, mean Sharpe +0.687, compounded +18.39%**. R5's pinned cell over the identical blocks, selecting nothing: **7/9, +1.125, +31.38%**, chained +1.2228 against +0.7634 — and in **all five** folds where re-derivation chose a different book it was worse out of sample, with no exception. The grid is not uniformly unstable: `enter_strength = 0.80` wins all nine folds, and the three axes that do move (`exit_strength`, `min_dwell`, `cooldown`) are the ones whose perturbations degrade gracefully, so re-selection is spending real money to chase differences the test half says are noise. The standing rule this sets: **a cell, once pre-committed and evaluated out of sample, is held — not re-fitted on a schedule.** The DSR is consistent with this rather than against it (0.70 at bar frequency, 0.77 daily, against the conventional 0.95): what is undiagnosable from luck is the *search*, and the answer is to stop searching, not to search more often. | 2026-08-06 |
| M23 | A pre-registration declares a **threshold**, not a comparison — and R9's did not | R9's outcome table says "DSR clears the null's expected maximum", which reads two ways with no way to choose between them after the fact: *observed > `E[max SR]`* is **true** (+1.2155 > +0.9676, and 10 of 54 cells clear it) and lands in row 1; *the DSR clearing a confidence level* is **false** at 0.95 (0.7026) and lands in row 2. Both are reported and neither is picked, because picking one after seeing the numbers is exactly what pre-registration exists to prevent — but the cost is that R9 cannot cite its own table for a verdict, and the verdict it does deliver (M22) has no row at all. The same gap let the deflation **frequency** be chosen after the fact: 0.7026 per bar against 0.7697 daily, on the same returns. Every future pre-registration states the statistic, the threshold, **and** the frequency it is computed at; a declared comparison without a number is not a declaration. | 2026-08-06 |

| M24 | **R7's slot is reused for the question the program was started to answer: telling chop from trend, measured directly rather than through a traded result** | The logistic sizing meta-model is dropped, not deferred. It has returned null three times from three independent directions — R6 found no per-state effect on the test half distinguishable from zero (no \|t\| above 1.77), the ETH replication found the states it would sit on are asset-specific (M18, M19), and R9 found that re-fitting anything on a schedule loses to holding a pre-committed cell (M22). A fourth attempt at the same question would be searching for a result three measurements say is not there. What has **never** been measured directly is chop detection itself: `COMPRESSION` is the machine's answer to it, and every assessment of that answer so far has been a **P&L** number, which scores the estimator and the policy laid over it as one thing and cannot say which is wrong. R4's shape is the right one — a diagnostic per candidate, judged on its own terms before anything trades it — and R9 supplies the first concrete lead, since `enter_strength` is the only axis stable across all nine folds, wins at the **most selective value ever tried**, and sits on the grid's edge in every study that has swept it (R5, ETH, R9). R8's model progression is unchanged and still follows. | 2026-08-06 |

| M25 | **The state machine detects chop with four of §2.1's nine dimensions, and the axis §2.1 defines chop *with* is one of the five it dropped. R8 does not start on the current four.** | R7 ([§9.6](#96-r7-choptrend-state-diagnosis--btcusdt-perp-4h-replicated-on-eth)), against thresholds declared before the data was seen. `COMPRESSION` clears **0/6** as a chop detector on BTC and **0/9** on ETH — best deficit −1.79 pp against a −10 pp bar, and the inside-vs-outside separation flips sign between halves (+8.08 train / −4.89 test). The nine registered features clear **0/9** on BTC. The composite entry gate clears **0/3**, and structurally so: it beats the `strength` gate alone by **+0.00 pp** on the test half at every horizon, because `direction_floor = 0.10` admits 82% of bars — **the gate the machine rests on is one feature wearing two names**. What does carry chop information is `energy`/`compression`: −0.0906 on BTC (halves −0.0800 / −0.1033, missing the 0.10 bar by 0.0094, 2.03 sd against 2.24) and **clearing on ETH at −0.1521** (−0.1164 / −0.2002), same sign, both halves, both instruments. `state_machine_v1` **does not read it** — `DEFAULT_FEATURES` and `REQUIRED_COLUMNS` are `(direction, strength, stability, crowding)` — and §2.1's own worked example defines chop as `Strength = 20, Energy = 95`, which is the relationship R7 measured. Note the sign runs against the naming: high `energy` is the chop side, so the state called `COMPRESSION` and the feature called `compression` are unrelated. **So the conclusion is not that §2.1 is wrong but that the machine implements four ninths of it**, and a better *estimator* over the same four inputs (R8 as written) would be a better estimator of a state vector missing its chop axis. Fix the inputs before the model. | 2026-08-06 |
| M26 | `enter_strength = 0.80` is an **interior optimum** of the wider grid, not an artifact of where the sweep stopped — closed, and not to be re-raised | Flagged as an open worry three separate times: R5 noted the optimum sat on the swept range's edge, the ETH replication found the same on three of four axes, and R9 found it the only axis stable across all nine walk-forward folds while winning at the most selective value tried. R7 extended the axis to {0.85, 0.90, 0.95} on the training half under M21's scalar, with the control re-scoring the trained cell at R9's published +1.3974 exactly: **0/3 beat it**, and by wide margins — +0.5002, +0.5577, **−0.7497**. Explicitly a grid **extension** and not a re-selection: nothing was pinned, nothing re-derived, and R5's published cell was unchanged whichever way it came out (M22). | 2026-08-06 |

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
| Q3 | Target exposure: new table, or widen `signals.side`? Append-only + per-bar targets grows fast. **Answered, R6: neither — a nullable `target_exposure NUMERIC(10,6)` column beside the `side` that implies it.** `side` stays a CHECK on four events; a level is not an event, and a boolean strategy keeps writing exactly the row it wrote before, with NULL there. Growth is bounded by the rebalance band, since only a decision bar is worth storing. | R6 |
| Q4 | Replace vectorbt with `from_orders`, or write a custom continuous-rebalance simulator? **Answered, R6: `from_orders`, and vectorbt is not retired.** Measured before planning — `size_type="targetvalue"` turns a 10-bar taper into 6 orders where `from_signals` gives 1, and handles shorts, reversal through zero, and costs — so the missing expressiveness was the only thing missing. D8's "freeze, migrate, retire" is therefore satisfied as *two paths*, not as a departure (M14). | R6 |
| Q5 | Does the ETF long-only exit go to cash, or rotate to TLT/GLD? (v1 assumes cash.) | R5 |
| Q7 | **R9 measured re-selection losing to holding a pre-committed cell (M22). What does R7 become?** Three readings were on the table: (a) drop R7 and go to R10, the live path; (b) re-aim R7 at the ramp with a rule fitted once and held; (c) attack the thesis the program exists for — telling chop from trend. **Answered 2026-08-06: (c).** The sizing question has now returned null three times and the chop question has never been asked directly — every measurement of it so far has been *through* a traded result, which confounds the estimator with the policy laid over it. R7's slot is reused rather than renumbered; see M24 and the roadmap. | R7 |
| Q6 | Does the read-only research browser get built before R9, or does R9 come first? **Answered 2026-08-05: the browser first**, planned in [docs/plans/2026-08-05-research-browser.md](../plans/2026-08-05-research-browser.md). The loop is felt as friction, and the browser is tooling rather than a research claim — it ships no strategy and no published figure, so it cannot move a number R9 will test. Two measurements taken before planning shrank it further: recompute is **19 ms** for the heaviest strategy against 272 ms to load the frame, so the state-of-play entry's "~seconds… wrong for anything multi-user" is **withdrawn**; and the vendored v5 chart already carries `BaselineSeries`, which draws a signed −1..1 target, so the "markers cannot draw a continuous target" gap closes with a series type that was already in the file. | R7, R10 |

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
