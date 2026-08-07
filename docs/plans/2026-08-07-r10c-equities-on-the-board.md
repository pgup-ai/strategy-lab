# R10c — equities on the board, and what a tile can honestly claim about them

**A design plan, not a pre-registration**, like R10a and R10b: it ships an
interface and a measurement, not a figure. Its **acceptance checks are binding**
and its design comes from reconnaissance done before writing it.

**Charter:** [docs/research/2026-08-03-market-dynamics-engine.md](../research/2026-08-03-market-dynamics-engine.md),
phase R10. R10b shipped the board **perps-only** because census item **(f)** was
unmeasured. This measures it and then widens.

---

## Census item (f), measured

The census said the Yahoo fetcher "rescales all history by adjusted close, so a
dividend rewrites past bars". Confirmed in the code — `adj_factor = adj_close /
close` is applied to every OHLC column — and now quantified against the stored
SPY weekly series:

| | |
|---|---|
| overlapping bars, stored vs a fresh fetch | **333** |
| bars whose close changed | **333 (100%)** |
| median relative move | **0.257%** |
| largest | **0.405%** |
| oldest bar that moved | **2020-01-01** |

**Every stored bar back to 2020 moved.** So an equity history is not a growing
record; it is a **restated** one, and two figures computed from it at different
times are computed from different data for the same dates.

**And it reaches the signals — rarely, and not uniformly.** Run over both
versions of the same 333 bars:

| strategy | entries, stored vs fresh | bars differing |
|---|---|---|
| `trend_following_deepseek_v4` | 21 / 21 | **0** |
| `tsmom` | 209 / 209 | **0** |
| **`donchian`** | **39 / 38** | **3** |

The two that survive intact are ratio-based, and a *uniform* rescale cancels in
a ratio. `donchian` compares a close against a channel's own high and low, and
**the factor is not uniform** — each bar's adjustment reflects the dividends
paid after it, so bars are rescaled by slightly different amounts and a marginal
break can flip. That is the finding: the rewrite is small, real, and
strategy-dependent, and no reader can tell from a tile which kind they are
looking at.

---

## What that means for a tile, and it is not "add equities to the filter"

A perp tile and an equity tile go stale for **different reasons**, and each must
say its own:

| | perp | equity |
|---|---|---|
| what bounds the frame | its own stored `funding_span` | nothing — it runs to the newest bar |
| what can be stale | the **right edge**, up to one funding cadence | the **whole history**, on any dividend |
| what the tile must state | `as of` vs the newest stored bar | when these candles were last **written** |

`market_candles` already carries `updated_at`, maintained on every upsert
(`db/candles.py:159`) — measured, SPY 1w reads `2026-08-03 02:20:07`. So the
answer is available without a schema change, which is the same shape as R10a's
finding: persist only what cannot be recomputed, and read the rest.

---

## What gets built

1. **`stored_datasets` and the board row carry `last_written`**, from
   `max(updated_at)` per dataset. One aggregate, alongside the enumeration that
   already runs.
2. **The frame bound becomes market-type-shaped.** A perp keeps its
   `funding_span` right edge. An equity has no funding, so it runs to its newest
   bar — and the `funding_span` probe is not called for one.
3. **The tile states the risk that applies to it.** A perp continues to show
   `as of` against the newest stored bar. An equity shows `candles written`, and
   flags it when the history is old enough that a dividend has plausibly landed
   since.
4. **The market filter offers equities**, and `DEFAULT_MARKET_TYPE` stays
   `perp` — opening on a market whose tiles carry a restatement caveat is a
   choice a reader should make, not inherit.

---

## Acceptance checks

1. A board over the stored equity datasets returns a row per dataset, and a
   frame shorter than its strategy's warmup reports that in its own row — the
   same rule R10b established, on datasets that will actually hit it (`XLF`,
   `XLK`, `QQQ` and `SMH` hold 333 weekly bars against `ema_cross`'s warmup).
2. Each equity row's state and latest fill are **identical** to `/api/analysis`
   for the same pair, over that row's own window — R10b's check 2, unchanged,
   because M36 binds every view and a second market type is still one view.
3. **No `funding_span` query is issued for an equity dataset.** Asserted by
   statement, not by absence of an error: an equity has no funding and asking is
   how a coverage guard gets invented for a market that has none.
4. An equity tile states `candles written`, and a perp tile still states its
   `as of` lag. Neither shows the other's, because neither risk applies.
5. `browse` still writes nothing on the read path.
6. Every published perp figure and every R10b behaviour is unchanged — the
   board is being widened, not rebuilt.

---

## What this deliberately does not do

- **No change to the fetcher.** The adjustment is *correct*: a dividend-adjusted
  series is the right input for a strategy that compounds. The problem is not
  that history is restated, it is that a tile can imply it was not.
- **No attempt to detect a dividend.** `updated_at` says when the data was
  written, which is what a reader needs; deciding whether a *specific*
  restatement happened means storing a second copy to diff against, and R10a's
  rule says not to store what can be re-fetched.
- **No equity in `state_machine_v1`'s registry path on the board by default.**
  It reads `crowding`, which needs funding; on an equity it falls back to
  neutral and records `crowding_measured=False`, which is exactly the M20
  condition. It stays selectable, and its tile says so, as every tile already
  does.

---

## Then

Census items (b), (c) and (d) remain — all three about the event path rather
than the view — and (a) is measured but not closed. After this the board covers
the scope the product vision opens with: crypto and stock/ETF.
