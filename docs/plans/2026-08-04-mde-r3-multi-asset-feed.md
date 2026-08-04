# MDE R3 — Multi-Asset Feed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the feed and engine process many instruments in one time-ordered stream, so cross-sectional features (breadth, confirmation, portfolio sizing) become computable at all.

**Architecture:** `ReplayFeed.stream()` becomes a k-way merge over subscriptions ordered by event time with a deterministic tie-break. A new `MarketClock` groups the merged stream into `MarketSnapshot`s — the set of bars sharing one event time — and emits a snapshot only once a *later* event arrives, so completeness is established causally rather than by looking ahead. `MultiAssetRunner` holds one `BarBuffer` per instrument and drives per-instrument strategies from snapshots.

**Tech Stack:** Python 3.11, pandas, asyncio, pytest.

**Charter:** [docs/research/2026-08-03-market-dynamics-engine.md](../research/2026-08-03-market-dynamics-engine.md) — this is phase R3.

---

## Why this phase exists

Measured on `main` today:

```
stream([BTC, ETH])  ->  BTC BTC BTC BTC ETH ETH ETH ETH
globally time-ordered? False
```

`stream()` loops subscriptions sequentially. Every cross-sectional feature in the charter's state vector — crypto breadth across coins, RSP-confirms-SPY, sector breadth, portfolio volatility targeting, `GlobalRiskScore` — needs bars from several instruments *at the same timestamp*. None of them can be computed on a sequential feed, which makes this the widest blocker in the program. It is also a larger architectural change than the state vector itself, which the original research plan did not flag.

---

## The one design decision that shapes everything

**How does a cross-sectional feature know that timestamp *t* is complete?**

In replay you could look ahead — and that is exactly the lookahead this repo spends two test suites preventing. The causal rule:

> A timestamp is complete when an event with a **later** timestamp arrives.

Consequences, all deliberate:

1. **Cross-sectional signals lag one bar.** The snapshot for *t* is emitted when the first *t+1* event appears. This is not a limitation to fix later — it is what a live feed can actually know, so replay must behave the same way or the two paths diverge.
2. **The final timestamp needs an explicit flush.** Nothing arrives after it, so `MarketClock` exposes `flush()` for end-of-stream. Forgetting it silently drops the last snapshot.
3. **A snapshot holds only instruments that have a bar at that timestamp.** Crypto trades 24/7, ETFs do not, and instruments list and delist. A partial universe is normal; "absent" must never be read as "unchanged".

**Rejected alternative:** emit at *t* with whatever has arrived. Non-deterministic live (it depends on network arrival order) and therefore unreproducible in replay. Rejected on the same grounds as any other lookahead shortcut.

---

## Conventions

- `.venv/bin/python -m pytest` and `.venv/bin/ruff check src tests`. Suite is **355 passed** on `main`.
- Timestamps are UTC epoch **milliseconds** as `int`; prices are `Decimal` at boundaries, float64 only in the pandas layer.
- **`market_candles` holds 133,620 rows of real research data — read-only.**
- **Run the suite before committing, not after.** A broken commit reached origin earlier in this project because the order was reversed.
- **Mutation-test every test, and assert the mutation applied.** Three separate no-op mutations have produced false "passes" in this project — a `.replace()` whose target string did not exist reads exactly like a test that cannot fail. Print or assert the mutated line before trusting the result.

---

## Backward compatibility — non-negotiable

Single-instrument behaviour must not move. Specifically:

- `tests/test_replay_determinism.py` must keep passing unchanged — it is the proof that backtest and replay agree, and the whole phase rests on it.
- `StrategyRunner` keeps its current constructor and semantics. `MultiAssetRunner` is **new**, not a replacement.
- `stream()` with one subscription must yield exactly what it yields today.

---

## File structure

| File | Responsibility |
|---|---|
| `src/strategy_lab/core/types.py` | add `MarketSnapshot` to the vocabulary |
| `src/strategy_lab/feeds/replay.py` | k-way merge in `stream()` |
| `src/strategy_lab/engine/market_clock.py` | group merged events into complete snapshots |
| `src/strategy_lab/engine/multi_runner.py` | per-instrument buffers, snapshot-driven dispatch |
| `src/strategy_lab/features/cross_sectional.py` | breadth and confirmation over a snapshot |
| `tests/test_multi_asset_feed.py` | merge ordering, ties, partial universes |
| `tests/test_market_clock.py` | completeness rule, flush, out-of-order |
| `tests/test_multi_runner.py` | dispatch, isolation, warmup per instrument |
| `tests/test_cross_sectional.py` | breadth maths, partial universes |

---

## Task 1: `MarketSnapshot`

**Files:**
- Modify: `src/strategy_lab/core/types.py`
- Test: `tests/test_market_clock.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_market_clock.py`:

```python
from __future__ import annotations

from decimal import Decimal

import pytest

from strategy_lab.core.types import Bar, InstrumentId, MarketSnapshot

BTC = InstrumentId("binance", "perp", "BTC/USDT")
ETH = InstrumentId("binance", "perp", "ETH/USDT")


def bar(instrument: InstrumentId, ts_open_ms: int, close: str = "100") -> Bar:
    return Bar(
        instrument=instrument,
        timeframe="4h",
        ts_open_ms=ts_open_ms,
        ts_close_ms=ts_open_ms + 14_400_000 - 1,
        open=Decimal(close),
        high=Decimal(close),
        low=Decimal(close),
        close=Decimal(close),
        volume=Decimal("1"),
        is_closed=True,
    )


def test_snapshot_exposes_its_bars_by_instrument():
    snapshot = MarketSnapshot(ts_event_ms=1_000, bars={BTC: bar(BTC, 0), ETH: bar(ETH, 0)})
    assert snapshot[BTC].instrument == BTC
    assert set(snapshot.instruments) == {BTC, ETH}
    assert len(snapshot) == 2


def test_snapshot_reports_a_missing_instrument_rather_than_inventing_one():
    """Absent must never read as unchanged -- instruments list, delist, and halt."""
    snapshot = MarketSnapshot(ts_event_ms=1_000, bars={BTC: bar(BTC, 0)})
    assert ETH not in snapshot
    assert snapshot.get(ETH) is None
    with pytest.raises(KeyError):
        snapshot[ETH]


def test_snapshot_is_frozen():
    snapshot = MarketSnapshot(ts_event_ms=1_000, bars={BTC: bar(BTC, 0)})
    with pytest.raises(AttributeError):
        snapshot.ts_event_ms = 2_000
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_market_clock.py -q`
Expected: FAIL — `ImportError: cannot import name 'MarketSnapshot'`

- [ ] **Step 3: Implement**

Append to `src/strategy_lab/core/types.py`:

```python
@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Every bar that closed at one event time.

    A snapshot is the unit a cross-sectional feature consumes: breadth over five
    coins is meaningless unless all five bars describe the same instant. It holds
    only instruments that actually have a bar at ``ts_event_ms`` -- crypto trades
    around the clock and equities do not, so a partial universe is the normal
    case and ``absent`` must never be read as ``unchanged``.
    """

    ts_event_ms: int
    bars: dict[InstrumentId, Bar]

    def __getitem__(self, instrument: InstrumentId) -> Bar:
        return self.bars[instrument]

    def __contains__(self, instrument: object) -> bool:
        return instrument in self.bars

    def __len__(self) -> int:
        return len(self.bars)

    def get(self, instrument: InstrumentId) -> Bar | None:
        return self.bars.get(instrument)

    @property
    def instruments(self) -> tuple[InstrumentId, ...]:
        return tuple(self.bars)
```

Add `MarketSnapshot` to the `core/__init__.py` re-export list and its `__all__`.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_market_clock.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit** (run the full suite first)

```bash
.venv/bin/python -m pytest -q && .venv/bin/ruff check src tests
git add src/strategy_lab/core tests/test_market_clock.py
git commit -m "feat(core): add MarketSnapshot for cross-sectional reads"
```

---

## Task 2: k-way merge in `ReplayFeed.stream()`

**Files:**
- Modify: `src/strategy_lab/feeds/replay.py`
- Test: `tests/test_multi_asset_feed.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_multi_asset_feed.py`:

```python
from __future__ import annotations

import asyncio

import pandas as pd

from strategy_lab.core.types import InstrumentId
from strategy_lab.feeds.base import Subscription
from strategy_lab.feeds.replay import ReplayFeed
from tests.conftest import synthetic_ohlcv

BTC = InstrumentId("binance", "perp", "BTC/USDT")
ETH = InstrumentId("binance", "perp", "ETH/USDT")
SOL = InstrumentId("binance", "perp", "SOL/USDT")


def drain(feed: ReplayFeed, subs) -> list:
    async def _run():
        return [event async for event in feed.stream(subs)]

    return asyncio.run(_run())


def test_two_instruments_interleave_by_time():
    frames = {(BTC, "4h"): synthetic_ohlcv(n=4, freq="4h"),
              (ETH, "4h"): synthetic_ohlcv(n=4, freq="4h")}
    events = drain(ReplayFeed(frames=frames), [Subscription(BTC, "4h"), Subscription(ETH, "4h")])

    times = [e.ts_event_ms for e in events]
    assert times == sorted(times), "merged stream must be globally time-ordered"
    assert len(events) == 8


def test_ties_break_deterministically_on_instrument_key():
    """Same bar time across instruments must order identically on every run."""
    frames = {(BTC, "4h"): synthetic_ohlcv(n=3, freq="4h"),
              (ETH, "4h"): synthetic_ohlcv(n=3, freq="4h"),
              (SOL, "4h"): synthetic_ohlcv(n=3, freq="4h")}
    subs = [Subscription(SOL, "4h"), Subscription(BTC, "4h"), Subscription(ETH, "4h")]

    first = [e.bar.instrument.key for e in drain(ReplayFeed(frames=frames), subs)]
    second = [e.bar.instrument.key for e in drain(ReplayFeed(frames=frames), subs)]
    assert first == second

    at_first_time = first[:3]
    assert at_first_time == sorted(at_first_time), "ties order by instrument key"


def test_a_single_subscription_is_unchanged():
    """The one-instrument path is what test_replay_determinism.py rests on."""
    df = synthetic_ohlcv(n=20, freq="4h")
    events = drain(ReplayFeed(frames={(BTC, "4h"): df}), [Subscription(BTC, "4h")])
    assert [e.bar.ts_open_ms for e in events] == [
        int(ts.value // 1_000_000) for ts in df.index
    ]


def test_instruments_with_different_histories_still_merge():
    """ETH lists later than BTC; the merge must not assume equal lengths."""
    btc = synthetic_ohlcv(n=6, freq="4h")
    eth = synthetic_ohlcv(n=6, freq="4h").iloc[3:]
    frames = {(BTC, "4h"): btc, (ETH, "4h"): eth}
    events = drain(ReplayFeed(frames=frames), [Subscription(BTC, "4h"), Subscription(ETH, "4h")])

    times = [e.ts_event_ms for e in events]
    assert times == sorted(times)
    assert len(events) == 9
    assert events[0].bar.instrument == BTC


def test_mixed_timeframes_merge_on_event_time():
    """A 4h and a 1d subscription interleave by close time, not by bar index."""
    frames = {(BTC, "4h"): synthetic_ohlcv(n=12, freq="4h"),
              (ETH, "1d"): synthetic_ohlcv(n=2, freq="1D")}
    events = drain(ReplayFeed(frames=frames), [Subscription(BTC, "4h"), Subscription(ETH, "1d")])
    times = [e.ts_event_ms for e in events]
    assert times == sorted(times)
    assert len(events) == 14
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_multi_asset_feed.py -q`
Expected: FAIL — `test_two_instruments_interleave_by_time` asserts `times == sorted(times)` and gets all of BTC then all of ETH.

- [ ] **Step 3: Implement the merge**

Replace `ReplayFeed.stream` in `src/strategy_lab/feeds/replay.py`:

```python
    async def stream(self, subs: Sequence[Subscription]) -> AsyncIterator[BarEvent]:
        """Yield every subscription's bars as one globally time-ordered stream.

        Ties -- several instruments closing the same bar -- are broken on the
        instrument key so the order is identical on every run. Without a total
        order the replay/live determinism proof does not hold for more than one
        instrument.
        """
        merged = heapq.merge(
            *(self._events_for(sub) for sub in subs),
            key=lambda event: (event.ts_event_ms, event.bar.instrument.key),
        )
        for event in merged:
            self._last_event_ms = event.ts_event_ms
            yield event

    def _events_for(self, sub: Subscription) -> Iterator[BarEvent]:
        df = self.frames.get((sub.instrument, sub.timeframe))
        if df is None or df.empty:
            return
        bar_ms = timeframe_to_millis(sub.timeframe)
        for timestamp, row in _ordered(df).iterrows():
            bar = _row_to_bar(timestamp, row, sub.instrument, sub.timeframe, bar_ms)
            yield BarEvent(bar=bar, ts_event_ms=bar.ts_close_ms, ts_recv_ms=None)
```

Add `import heapq` and `from collections.abc import Iterator` at the top.

`heapq.merge` is the right tool: it is lazy, so a 15,000-bar universe is not materialized, and it is stable, so equal keys keep source order — which is why the explicit instrument key in the sort key is what makes ties deterministic rather than dependent on subscription order.

Remove the now-obsolete "Known limitations" note about sequential subscriptions from the `ReplayFeed` docstring, and the test that pins the old non-interleaved behaviour in `tests/test_replay_feed.py` — it deliberately asserted the wrong ordering so a future fix would trip visibly. This is that fix.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_multi_asset_feed.py tests/test_replay_feed.py tests/test_replay_determinism.py -q`
Expected: all pass. If the determinism test fails, stop — the single-instrument path moved and that is the guarantee the phase rests on.

- [ ] **Step 5: Mutation-test**

Remove the `event.bar.instrument.key` component from the merge key and confirm the tie-break test fails. **Assert the mutation applied before trusting the result** — print the mutated line. Record actual output. Revert.

- [ ] **Step 6: Commit** (full suite first)

```bash
.venv/bin/python -m pytest -q && .venv/bin/ruff check src tests
git add src/strategy_lab/feeds/replay.py tests/test_multi_asset_feed.py tests/test_replay_feed.py
git commit -m "feat(feeds): merge subscriptions into one time-ordered stream"
```

---

## Task 3: `MarketClock` — completeness without lookahead

**Files:**
- Create: `src/strategy_lab/engine/market_clock.py`
- Test: `tests/test_market_clock.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_market_clock.py`:

```python
from strategy_lab.core.types import BarEvent
from strategy_lab.engine.market_clock import MarketClock


def event(instrument: InstrumentId, ts_open_ms: int) -> BarEvent:
    b = bar(instrument, ts_open_ms)
    return BarEvent(bar=b, ts_event_ms=b.ts_close_ms, ts_recv_ms=None)


def test_a_timestamp_is_complete_only_once_a_later_event_arrives():
    """Completeness is established causally: t is done when t+1 shows up."""
    clock = MarketClock()
    assert clock.on_event(event(BTC, 0)) is None
    assert clock.on_event(event(ETH, 0)) is None

    snapshot = clock.on_event(event(BTC, 14_400_000))
    assert snapshot is not None
    assert set(snapshot.instruments) == {BTC, ETH}
    assert snapshot.ts_event_ms == 14_400_000 - 1


def test_flush_releases_the_final_timestamp():
    """Nothing arrives after the last bar, so it needs an explicit flush."""
    clock = MarketClock()
    clock.on_event(event(BTC, 0))
    clock.on_event(event(ETH, 0))

    snapshot = clock.flush()
    assert snapshot is not None and len(snapshot) == 2
    assert clock.flush() is None, "flushing twice must not replay the snapshot"


def test_a_partial_universe_is_emitted_as_is():
    """ETH is halted; the snapshot reports BTC only rather than stalling."""
    clock = MarketClock()
    clock.on_event(event(BTC, 0))
    snapshot = clock.on_event(event(BTC, 14_400_000))
    assert set(snapshot.instruments) == {BTC}


def test_an_out_of_order_event_is_rejected_not_silently_reordered():
    clock = MarketClock()
    clock.on_event(event(BTC, 14_400_000))
    with pytest.raises(ValueError, match="out of order"):
        clock.on_event(event(ETH, 0))


def test_a_duplicate_instrument_at_one_timestamp_keeps_the_last():
    """A reconnect can redeliver a bar; last wins, matching the feed's dedup."""
    clock = MarketClock()
    clock.on_event(event(BTC, 0))
    clock.on_event(BarEvent(bar=bar(BTC, 0, close="999"), ts_event_ms=14_399_999, ts_recv_ms=None))
    snapshot = clock.flush()
    assert snapshot[BTC].close == Decimal("999")
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_market_clock.py -q`
Expected: FAIL — `No module named 'strategy_lab.engine.market_clock'`

- [ ] **Step 3: Implement**

Create `src/strategy_lab/engine/market_clock.py`:

```python
from __future__ import annotations

from strategy_lab.core.types import BarEvent, InstrumentId, MarketSnapshot


class MarketClock:
    """Groups a time-ordered event stream into complete per-timestamp snapshots.

    A timestamp is complete when an event with a **later** timestamp arrives --
    never by looking ahead, which is the same rule a live feed is bound by. The
    consequence is deliberate: a cross-sectional signal for bar *t* is available
    at *t+1*, in replay exactly as in live, so the two paths cannot diverge.

    The final timestamp has no successor, so it is released by ``flush()``.
    """

    def __init__(self) -> None:
        self._ts_event_ms: int | None = None
        self._bars: dict[InstrumentId, object] = {}

    def on_event(self, event: BarEvent) -> MarketSnapshot | None:
        if self._ts_event_ms is not None and event.ts_event_ms < self._ts_event_ms:
            raise ValueError(
                f"event at {event.ts_event_ms} arrived out of order after "
                f"{self._ts_event_ms}; the feed must yield a total time order"
            )

        completed = None
        if self._ts_event_ms is not None and event.ts_event_ms > self._ts_event_ms:
            completed = self._take()

        self._ts_event_ms = event.ts_event_ms
        self._bars[event.bar.instrument] = event.bar
        return completed

    def flush(self) -> MarketSnapshot | None:
        return self._take()

    def _take(self) -> MarketSnapshot | None:
        if self._ts_event_ms is None or not self._bars:
            return None
        snapshot = MarketSnapshot(ts_event_ms=self._ts_event_ms, bars=dict(self._bars))
        self._bars.clear()
        return snapshot
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_market_clock.py -q`
Expected: `8 passed`

- [ ] **Step 5: Mutation-test**

Change `event.ts_event_ms > self._ts_event_ms` to `>=` and confirm a test fails (it would emit a snapshot per event rather than per timestamp). Then delete the `flush()` body's `_take()` call and confirm the flush test fails. **Assert each mutation applied.** Record actual output. Revert.

- [ ] **Step 6: Commit** (full suite first)

```bash
.venv/bin/python -m pytest -q && .venv/bin/ruff check src tests
git add src/strategy_lab/engine/market_clock.py tests/test_market_clock.py
git commit -m "feat(engine): group the merged stream into complete market snapshots"
```

---

## Task 4: `MultiAssetRunner`

**Files:**
- Create: `src/strategy_lab/engine/multi_runner.py`
- Modify: `src/strategy_lab/engine/__init__.py`
- Test: `tests/test_multi_runner.py`

Holds one `BarBuffer` per instrument so per-instrument strategies keep their full history — the same full-history rule `StrategyRunner` follows, because `ewm(adjust=False)` depends on every prior bar.

- [ ] **Step 1: Write the failing test**

Create `tests/test_multi_runner.py`:

```python
from __future__ import annotations

import asyncio

import pytest

from strategy_lab.core.clock import SimClock
from strategy_lab.core.types import InstrumentId, Side
from strategy_lab.engine.multi_runner import MultiAssetRunner
from strategy_lab.feeds.base import Subscription
from strategy_lab.feeds.replay import ReplayFeed
from strategy_lab.strategies.registry import get_strategy
from tests.conftest import synthetic_ohlcv

BTC = InstrumentId("binance", "perp", "BTC/USDT")
ETH = InstrumentId("binance", "perp", "ETH/USDT")


def run(runner: MultiAssetRunner, feed: ReplayFeed, subs) -> list:
    async def _run():
        collected = []
        async for event in feed.stream(subs):
            collected.extend(runner.on_event(event))
        collected.extend(runner.flush())
        return collected

    return asyncio.run(_run())


def two_instrument_feed(n: int = 400):
    return ReplayFeed(frames={(BTC, "4h"): synthetic_ohlcv(n=n, freq="4h", seed=1),
                              (ETH, "4h"): synthetic_ohlcv(n=n, freq="4h", seed=2)})


def test_each_instrument_gets_its_own_buffer():
    strategy = get_strategy("donchian")
    runner = MultiAssetRunner(
        strategies={BTC: strategy, ETH: strategy}, timeframe="4h", clock=SimClock()
    )
    feed = two_instrument_feed()
    run(runner, feed, [Subscription(BTC, "4h"), Subscription(ETH, "4h")])

    assert len(runner.buffer(BTC)) == 400
    assert len(runner.buffer(ETH)) == 400


def test_signals_are_attributed_to_the_right_instrument():
    strategy = get_strategy("donchian")
    runner = MultiAssetRunner(
        strategies={BTC: strategy, ETH: strategy}, timeframe="4h", clock=SimClock()
    )
    signals = run(runner, two_instrument_feed(), [Subscription(BTC, "4h"), Subscription(ETH, "4h")])

    assert signals, "expected signals from a 400-bar donchian run"
    assert {s.instrument for s in signals} <= {BTC, ETH}
    for signal in signals:
        assert signal.side in set(Side)


def test_one_instrument_matches_the_single_asset_runner():
    """A one-instrument MultiAssetRunner must not diverge from StrategyRunner."""
    from strategy_lab.engine.runner import StrategyRunner

    df = synthetic_ohlcv(n=400, freq="4h", seed=1)
    strategy = get_strategy("donchian")

    multi = MultiAssetRunner(strategies={BTC: strategy}, timeframe="4h", clock=SimClock())
    multi_signals = run(multi, ReplayFeed(frames={(BTC, "4h"): df}), [Subscription(BTC, "4h")])

    single = StrategyRunner(strategy=strategy, instrument=BTC, timeframe="4h", clock=SimClock())
    single_signals = []
    for event in ReplayFeed(frames={(BTC, "4h"): df})._events_for(Subscription(BTC, "4h")):
        single_signals.extend(single.on_event(event))

    assert [(s.ts_bar_ms, s.side) for s in multi_signals] == [
        (s.ts_bar_ms, s.side) for s in single_signals
    ]


def test_an_instrument_without_a_strategy_is_buffered_but_never_traded():
    """Context-only instruments feed cross-sectional features without trading."""
    runner = MultiAssetRunner(
        strategies={BTC: get_strategy("donchian")}, timeframe="4h", clock=SimClock(),
        context={ETH},
    )
    signals = run(runner, two_instrument_feed(), [Subscription(BTC, "4h"), Subscription(ETH, "4h")])

    assert len(runner.buffer(ETH)) == 400
    assert all(s.instrument == BTC for s in signals)


def test_an_unknown_instrument_is_rejected_rather_than_silently_dropped():
    runner = MultiAssetRunner(
        strategies={BTC: get_strategy("donchian")}, timeframe="4h", clock=SimClock()
    )
    feed = two_instrument_feed(n=5)
    with pytest.raises(KeyError, match="ETH/USDT"):
        run(runner, feed, [Subscription(BTC, "4h"), Subscription(ETH, "4h")])
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_multi_runner.py -q`
Expected: FAIL — `No module named 'strategy_lab.engine.multi_runner'`

- [ ] **Step 3: Implement**

Create `src/strategy_lab/engine/multi_runner.py` with:

```python
class MultiAssetRunner:
    def __init__(
        self,
        *,
        strategies: dict[InstrumentId, Strategy],
        timeframe: str,
        clock: Clock,
        context: set[InstrumentId] | None = None,
        allow_forming_bars: bool = False,
    ) -> None: ...

    def buffer(self, instrument: InstrumentId) -> BarBuffer: ...
    def snapshot(self) -> MarketSnapshot | None: ...
    def on_event(self, event: BarEvent) -> Sequence[Signal]: ...
    def flush(self) -> Sequence[Signal]: ...
```

Behaviour the tests above pin:

- One `BarBuffer` per instrument, full history, created for both traded and `context` instruments.
- An instrument that is in neither `strategies` nor `context` raises `KeyError` naming the symbol — silently dropping a subscription is how a universe quietly shrinks.
- Each traded instrument delegates to the **same** extraction logic `StrategyRunner` uses, so the two cannot drift. Reuse `StrategyRunner` internally (one per traded instrument) rather than copying `_extract`.
- `on_event` feeds an internal `MarketClock`; `snapshot()` exposes the most recently completed one for cross-sectional features.
- `flush()` releases the final timestamp and returns any signals from it.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_multi_runner.py -q`
Expected: `5 passed`

- [ ] **Step 5: Mutation-test**

Share one `BarBuffer` across instruments and confirm the per-buffer test fails. Drop the unknown-instrument `KeyError` and confirm that test fails. **Assert each mutation applied.** Record actual output. Revert.

- [ ] **Step 6: Commit** (full suite first)

```bash
.venv/bin/python -m pytest -q && .venv/bin/ruff check src tests
git add src/strategy_lab/engine tests/test_multi_runner.py
git commit -m "feat(engine): add MultiAssetRunner with per-instrument buffers"
```

---

## Task 5: Cross-sectional features

**Files:**
- Create: `src/strategy_lab/features/__init__.py`
- Create: `src/strategy_lab/features/cross_sectional.py`
- Test: `tests/test_cross_sectional.py`

The payoff: the first features that were impossible before this phase. Keep to the two the charter names first — breadth and confirmation — and resist adding more. The state vector is R4.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cross_sectional.py`:

```python
from __future__ import annotations

from decimal import Decimal

import pytest

from strategy_lab.core.types import Bar, InstrumentId, MarketSnapshot
from strategy_lab.features.cross_sectional import breadth, confirms

BTC = InstrumentId("binance", "perp", "BTC/USDT")
ETH = InstrumentId("binance", "perp", "ETH/USDT")
SOL = InstrumentId("binance", "perp", "SOL/USDT")


def bar(instrument, open_: str, close: str) -> Bar:
    return Bar(
        instrument=instrument, timeframe="4h", ts_open_ms=0, ts_close_ms=14_399_999,
        open=Decimal(open_), high=Decimal(max(open_, close, key=float)),
        low=Decimal(min(open_, close, key=float)), close=Decimal(close),
        volume=Decimal("1"), is_closed=True,
    )


def snapshot(**bars) -> MarketSnapshot:
    mapping = {BTC: bars.get("btc"), ETH: bars.get("eth"), SOL: bars.get("sol")}
    return MarketSnapshot(ts_event_ms=14_399_999,
                          bars={k: v for k, v in mapping.items() if v is not None})


def test_breadth_is_the_fraction_advancing():
    snap = snapshot(btc=bar(BTC, "100", "110"), eth=bar(ETH, "100", "105"),
                    sol=bar(SOL, "100", "90"))
    assert breadth(snap) == pytest.approx(2 / 3)


def test_breadth_over_a_partial_universe_uses_only_present_instruments():
    """A halted instrument must not count as flat -- it is absent, not unchanged."""
    snap = snapshot(btc=bar(BTC, "100", "110"), eth=bar(ETH, "100", "90"))
    assert breadth(snap) == pytest.approx(0.5)


def test_breadth_of_an_empty_snapshot_is_undefined_rather_than_zero():
    with pytest.raises(ValueError, match="no instruments"):
        breadth(MarketSnapshot(ts_event_ms=0, bars={}))


def test_confirms_requires_the_leader_and_a_quorum_of_followers():
    snap = snapshot(btc=bar(BTC, "100", "110"), eth=bar(ETH, "100", "105"),
                    sol=bar(SOL, "100", "90"))
    assert confirms(snap, leader=BTC, quorum=0.5) is True
    assert confirms(snap, leader=BTC, quorum=0.9) is False


def test_confirms_is_false_when_the_leader_is_absent():
    """No leader bar means no confirmation claim -- not a default True."""
    snap = snapshot(eth=bar(ETH, "100", "105"))
    assert confirms(snap, leader=BTC, quorum=0.5) is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cross_sectional.py -q`
Expected: FAIL — `No module named 'strategy_lab.features'`

- [ ] **Step 3: Implement**

Create `src/strategy_lab/features/cross_sectional.py` with `breadth(snapshot) -> float` and `confirms(snapshot, *, leader, quorum) -> bool`.

Both read only the snapshot — no history, no lookahead. `breadth` raises on an empty snapshot rather than returning 0.0, because "nothing advanced" and "nothing traded" are different facts and collapsing them is how a halted session reads as a bear signal.

- [ ] **Step 4: Run tests, mutation-test, commit** (full suite before committing)

Mutations: make `breadth` count absent instruments as flat (the partial-universe test must fail); make `confirms` return `True` when the leader is missing (that test must fail). Assert each applied.

```bash
git add src/strategy_lab/features tests/test_cross_sectional.py
git commit -m "feat(features): add breadth and confirmation over a market snapshot"
```

---

## Task 6: Prove it on real data, and document

**Files:**
- Modify: `README.md`, `CLAUDE.md`
- Modify: `docs/research/2026-08-03-market-dynamics-engine.md`
- Test: `tests/test_multi_asset_feed.py`

- [ ] **Step 1: Add the real-data test**

Append to `tests/test_multi_asset_feed.py` a `@pytest.mark.db` test that loads stored BTC and ETH perp 4h via `ReplayFeed.from_database`, streams both, and asserts the merged stream is globally time-ordered and contains both instruments. Skip cleanly if either series is missing.

- [ ] **Step 2: Measure breadth on the real universe**

Run and record the output — this is the phase's demonstration that the blocker is gone:

```bash
.venv/bin/python -c "
import asyncio
from strategy_lab.core.types import InstrumentId
from strategy_lab.engine.market_clock import MarketClock
from strategy_lab.features.cross_sectional import breadth
from strategy_lab.feeds.base import Subscription
from strategy_lab.feeds.replay import ReplayFeed

subs = [Subscription(InstrumentId('binance','perp',s), '4h') for s in ('BTC/USDT','ETH/USDT')]
feed = ReplayFeed.from_database(subs)
clock = MarketClock()

async def go():
    values, partial = [], 0
    async for event in feed.stream(subs):
        snap = clock.on_event(event)
        if snap:
            values.append(breadth(snap))
            partial += len(snap) < 2
    snap = clock.flush()
    if snap: values.append(breadth(snap))
    return values, partial

vals, partial = asyncio.run(go())
print(f'snapshots={len(vals):,}  mean breadth={sum(vals)/len(vals):.3f}  partial universes={partial:,}')
"
```

Report the numbers. A large partial count is expected and informative — BTC lists 2019-09, ETH 2019-11, so roughly the first 500 4h bars have one instrument only.

- [ ] **Step 3: Document**

`README.md` gets a short multi-asset section. `CLAUDE.md` gets the completeness rule and its one-bar consequence in the architecture notes — that is the non-obvious property a future contributor will otherwise re-derive or violate. The charter's progress log gets an R3 row and the roadmap status flips to done.

- [ ] **Step 4: Commit** (full suite first)

```bash
.venv/bin/python -m pytest -q && .venv/bin/ruff check src tests
git add -A && git commit -m "docs: document the multi-asset feed and its completeness rule"
```

---

## R3 GATE

- [ ] A merged stream over ≥2 instruments is globally time-ordered, with deterministic ties
- [ ] `tests/test_replay_determinism.py` passes unchanged — single-instrument behaviour did not move
- [ ] One-instrument `MultiAssetRunner` output is identical to `StrategyRunner`
- [ ] Breadth computed over real stored BTC+ETH, with the partial-universe count reported
- [ ] Full suite green, ruff clean

---

## Self-review notes

**Spec coverage.** Charter blocker B3 → Tasks 2–4. Cross-sectional features named in the charter's state vector (Participation/breadth) → Task 5. The completeness rule, which the charter does not address at all, is the design decision section plus Task 3.

**Deliberately out of scope.** The full state vector is R4. Live multi-asset feeds are Phase 1b. Portfolio-level vol targeting needs R4's state, so it is not here despite being cross-sectional — Task 5 stops at the two features the charter names first.

**Type consistency.** `MarketSnapshot(ts_event_ms, bars)` is constructed identically in Tasks 1, 3, and 5. `MarketClock.on_event → MarketSnapshot | None` and `flush() → MarketSnapshot | None` match between Tasks 3 and 4. `MultiAssetRunner(strategies=, timeframe=, clock=, context=)` is used identically in Task 4's tests and its signature.

**Known risk.** Task 4's `test_one_instrument_matches_the_single_asset_runner` calls `feed._events_for(...)`, a private helper added in Task 2. That is deliberate — it isolates the runner comparison from the merge — but it couples a test to a private name. If Task 2 renames it, Task 4 breaks.
