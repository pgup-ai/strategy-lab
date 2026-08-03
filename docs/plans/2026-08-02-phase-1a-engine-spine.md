# Phase 1a — Engine Spine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replay stored candles through an event-driven engine and produce signals provably identical to the existing vectorized path, persisted to Postgres — with zero network code.

**Architecture:** A `MarketDataFeed` yields `BarEvent`s; a `StrategyRunner` accumulates them into a pandas buffer and calls the *unmodified* `Strategy.generate_signals(df)` on each closed bar, reading the last row. Because the four existing strategies are causal (verified in the design doc §2), the last row of an expanding-window call equals the whole-history result — so backtest, replay, and live share one strategy implementation. `ReplayFeed` reads Postgres; `BinanceFeed` (Phase 1b) will implement the same protocol.

**Tech Stack:** Python 3.11, pandas, SQLAlchemy 2 + psycopg 3, Postgres 16, pytest, Typer, `decimal.Decimal` at boundaries.

**Design doc:** [docs/design/2026-08-02-realtime-trading-framework.md](../design/2026-08-02-realtime-trading-framework.md)

---

## Conventions for every task

- Run tests with the repo venv: `.venv/bin/python -m pytest` (or `source .venv/bin/activate` first).
- `pythonpath = ["src"]` is already set in `pyproject.toml`, so tests import `strategy_lab.*` without installing.
- Line length is 100 (`ruff check src tests` must pass before each commit).
- **All timestamps are UTC epoch milliseconds (`int`).** Never naive datetimes, never local time.
- **Prices and quantities are `Decimal` in `core`/`storage`; `float64` only inside the pandas indicator layer.** Convert with `Decimal(str(x))`, never `Decimal(float)`.
- Tests must not require network. Tests requiring Postgres are marked `@pytest.mark.db` and skip cleanly when it is unreachable.

### The tests in this plan are a floor, not a ceiling

Every test below was written alongside the implementation it checks, which is exactly the
condition under which tests document code instead of defending it. Task 1 proved the point:
the seven tests as written passed unchanged against three separate mutations of the
implementation, including one that deleted validation on four of five fields.

So for each task, before reporting done, **mutate your own implementation and confirm the
suite fails.** Pick the two or three changes that would most plausibly slip through review —
an inverted comparison, a dropped element from a loop's collection, a branch that returns a
constant — apply each, run the suite, and confirm red. Revert. If a mutant survives, the
test is decorative: strengthen it before reporting.

Strengthening tests beyond the literal text below is expected and welcome. Weakening an
assertion to make a test pass is never acceptable — if implementation and test disagree,
one of them has a bug, and you must say which.

---

## File structure

| File | Responsibility |
|---|---|
| `src/strategy_lab/core/__init__.py` | Re-export the vocabulary types |
| `src/strategy_lab/core/types.py` | `InstrumentId`, `Bar`, `BarEvent`, `Side`, `Mode`, `Signal` — no I/O, no deps beyond stdlib |
| `src/strategy_lab/core/clock.py` | `Clock` protocol, `LiveClock`, `SimClock` |
| `src/strategy_lab/feeds/__init__.py` | Re-export feed API |
| `src/strategy_lab/feeds/base.py` | `Subscription`, `FeedHealth`, `MarketDataFeed` protocol |
| `src/strategy_lab/feeds/replay.py` | `ReplayFeed` — Postgres-backed, satisfies `MarketDataFeed` |
| `src/strategy_lab/engine/__init__.py` | Re-export engine API |
| `src/strategy_lab/engine/context.py` | `BarBuffer` — accumulates bars, materializes the strategy DataFrame |
| `src/strategy_lab/engine/runner.py` | `StrategyRunner` — the single event entry point |
| `src/strategy_lab/storage/__init__.py` | Re-export storage API |
| `src/strategy_lab/storage/schema.py` | `runs` + `signals` tables |
| `src/strategy_lab/storage/migrations.py` | Idempotent DDL for the NUMERIC migration and new tables |
| `src/strategy_lab/storage/signals.py` | Append-only signal writes and reads |
| `tests/test_core_types.py` | Type invariants |
| `tests/test_clock.py` | Clock behavior |
| `tests/test_strategy_metadata.py` | Every registered strategy declares `version`/`warmup_bars` |
| `tests/test_lookahead.py` | **Poison probe** — the general lookahead gate |
| `tests/test_bar_buffer.py` | Buffer semantics |
| `tests/test_runner.py` | Runner emits the right signals, respects warmup and closed-bar gating |
| `tests/test_replay_feed.py` | Feed ordering and filtering |
| `tests/test_replay_determinism.py` | **Vectorized ≡ streaming** — the crown-jewel test |
| `tests/test_signals_storage.py` | Idempotency and append-only enforcement (db-marked) |
| `tests/conftest.py` | Shared synthetic OHLCV fixture + `db` marker skip logic |

Existing files modified: the four strategy modules (2 attributes each), `strategies/base.py` (protocol), `cli.py` (2 commands), `pyproject.toml` (pytest marker).

---

## Task 1: Core vocabulary types

**Files:**
- Create: `src/strategy_lab/core/__init__.py`
- Create: `src/strategy_lab/core/types.py`
- Test: `tests/test_core_types.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_core_types.py`:

```python
from __future__ import annotations

from decimal import Decimal

import pytest

from strategy_lab.core.types import Bar, BarEvent, InstrumentId, Mode, Side


def make_bar(**overrides) -> Bar:
    defaults = dict(
        instrument=InstrumentId("binance", "perp", "BTC/USDT"),
        timeframe="15m",
        ts_open_ms=1_785_723_300_000,
        ts_close_ms=1_785_724_199_999,
        open=Decimal("63205.31"),
        high=Decimal("63286.00"),
        low=Decimal("63100.00"),
        close=Decimal("63128.00"),
        volume=Decimal("96.3039"),
        is_closed=True,
    )
    defaults.update(overrides)
    return Bar(**defaults)


def test_instrument_id_is_hashable_and_renders_a_stable_key():
    instrument = InstrumentId("binance", "perp", "BTC/USDT")
    assert {instrument: 1}[instrument] == 1
    assert instrument.key == "binance:perp:BTC/USDT"


def test_bar_rejects_non_decimal_prices():
    with pytest.raises(TypeError, match="open must be Decimal"):
        make_bar(open=63205.31)


def test_bar_rejects_close_time_before_open_time():
    with pytest.raises(ValueError, match="ts_close_ms must be after ts_open_ms"):
        make_bar(ts_close_ms=1_785_723_299_999)


def test_bar_rejects_high_below_low():
    with pytest.raises(ValueError, match="high must be >= low"):
        make_bar(high=Decimal("1"), low=Decimal("2"))


def test_bar_is_frozen():
    bar = make_bar()
    with pytest.raises(AttributeError):
        bar.close = Decimal("1")


def test_bar_event_exposes_bar_timestamp():
    event = BarEvent(bar=make_bar(), ts_event_ms=1_785_724_200_140, ts_recv_ms=None)
    assert event.ts_event_ms == 1_785_724_200_140
    assert event.instrument == InstrumentId("binance", "perp", "BTC/USDT")


def test_side_and_mode_are_string_enums():
    assert Side.ENTER_LONG == "enter_long"
    assert Mode.REPLAY == "replay"
    assert Side.ENTER_LONG.opposite_exit == Side.EXIT_LONG
    assert Side.ENTER_SHORT.opposite_exit == Side.EXIT_SHORT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_core_types.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'strategy_lab.core'`

- [ ] **Step 3: Write the implementation**

Create `src/strategy_lab/core/types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

_PRICE_FIELDS = ("open", "high", "low", "close", "volume")


class Side(StrEnum):
    ENTER_LONG = "enter_long"
    EXIT_LONG = "exit_long"
    ENTER_SHORT = "enter_short"
    EXIT_SHORT = "exit_short"

    @property
    def opposite_exit(self) -> Side:
        if self is Side.ENTER_LONG:
            return Side.EXIT_LONG
        if self is Side.ENTER_SHORT:
            return Side.EXIT_SHORT
        return self


class Mode(StrEnum):
    BACKTEST = "backtest"
    REPLAY = "replay"
    PAPER = "paper"
    LIVE = "live"


@dataclass(frozen=True, slots=True)
class InstrumentId:
    exchange: str
    market_type: str
    symbol: str

    @property
    def key(self) -> str:
        return f"{self.exchange}:{self.market_type}:{self.symbol}"


@dataclass(frozen=True, slots=True)
class Bar:
    instrument: InstrumentId
    timeframe: str
    ts_open_ms: int
    ts_close_ms: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    is_closed: bool
    quote_volume: Decimal | None = None
    trades: int | None = None

    def __post_init__(self) -> None:
        for field_name in _PRICE_FIELDS:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                raise TypeError(f"{field_name} must be Decimal, got {type(value).__name__}")
        if self.ts_close_ms <= self.ts_open_ms:
            raise ValueError("ts_close_ms must be after ts_open_ms")
        if self.high < self.low:
            raise ValueError("high must be >= low")


@dataclass(frozen=True, slots=True)
class BarEvent:
    bar: Bar
    ts_event_ms: int
    ts_recv_ms: int | None = None

    @property
    def instrument(self) -> InstrumentId:
        return self.bar.instrument

    @property
    def is_closed(self) -> bool:
        return self.bar.is_closed


@dataclass(frozen=True, slots=True)
class Signal:
    instrument: InstrumentId
    timeframe: str
    strategy_id: str
    strategy_version: str
    ts_bar_ms: int
    ts_emit_ms: int
    side: Side
    bar_is_closed: bool
    reason: str
    entry_price: Decimal | None = None
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    strength: Decimal | None = None
    features: dict | None = None
```

Create `src/strategy_lab/core/__init__.py`:

```python
from __future__ import annotations

from strategy_lab.core.types import Bar, BarEvent, InstrumentId, Mode, Side, Signal

__all__ = ["Bar", "BarEvent", "InstrumentId", "Mode", "Side", "Signal"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_core_types.py -q && .venv/bin/ruff check src tests`
Expected: `7 passed` and ruff reporting `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/strategy_lab/core tests/test_core_types.py
git commit -m "feat(core): add event and signal vocabulary types"
```

---

## Task 2: Clock abstraction

**Files:**
- Create: `src/strategy_lab/core/clock.py`
- Test: `tests/test_clock.py`

The engine must never call `time.time()` directly — otherwise a replay would stamp signals with wall-clock time and diverge from the run it is reproducing.

- [ ] **Step 1: Write the failing test**

Create `tests/test_clock.py`:

```python
from __future__ import annotations

from strategy_lab.core.clock import LiveClock, SimClock


def test_sim_clock_starts_at_zero_and_advances_only_when_told():
    clock = SimClock()
    assert clock.now_ms() == 0
    clock.advance_to(1_785_724_200_000)
    assert clock.now_ms() == 1_785_724_200_000
    clock.advance_to(1_785_724_200_000)
    assert clock.now_ms() == 1_785_724_200_000


def test_sim_clock_never_goes_backwards():
    clock = SimClock(start_ms=1_000)
    clock.advance_to(500)
    assert clock.now_ms() == 1_000


def test_live_clock_returns_millisecond_epoch():
    now = LiveClock().now_ms()
    assert isinstance(now, int)
    # Sanity band: after 2020-01-01 and before 2100-01-01, in ms.
    assert 1_577_836_800_000 < now < 4_102_444_800_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_clock.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'strategy_lab.core.clock'`

- [ ] **Step 3: Write the implementation**

Create `src/strategy_lab/core/clock.py`:

```python
from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now_ms(self) -> int:
        ...


class LiveClock:
    """Wall-clock time. The only place in the package permitted to read the system clock."""

    def now_ms(self) -> int:
        return int(time.time() * 1000)


class SimClock:
    """Deterministic clock driven by event timestamps. Monotonic by construction."""

    def __init__(self, start_ms: int = 0) -> None:
        self._now_ms = start_ms

    def now_ms(self) -> int:
        return self._now_ms

    def advance_to(self, ts_ms: int) -> None:
        if ts_ms > self._now_ms:
            self._now_ms = ts_ms
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_clock.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/strategy_lab/core/clock.py tests/test_clock.py
git commit -m "feat(core): add Clock protocol with live and simulated implementations"
```

---

## Task 3: Declare strategy version and warmup

**Files:**
- Modify: `src/strategy_lab/strategies/base.py:23-27` (the `Strategy` protocol)
- Modify: `src/strategy_lab/strategies/turnaround_v1.py:15-19`
- Modify: `src/strategy_lab/strategies/turnaround_v2.py:15-22`
- Modify: `src/strategy_lab/strategies/trend_following_deepseek_v4.py:13-18`
- Modify: `src/strategy_lab/strategies/trend_rider_v1_deepseek_v4_pro.py:17-28`
- Create: `tests/conftest.py`
- Test: `tests/test_strategy_metadata.py`

`warmup_bars` is the largest lookback each strategy uses. Values are derived from the declared parameters: `turnaround_v1` EMA 200; `turnaround_v2` EMA 200 (extension EMA is 20); `trend_following_deepseek_v4` SMA 40; `trend_rider_v1_deepseek_v4_pro` SMA 40 (ROC 26, ATR 14 are smaller).

- [ ] **Step 1: Write the shared fixture**

Create `tests/conftest.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def synthetic_ohlcv(n: int = 400, seed: int = 7, freq: str = "15min") -> pd.DataFrame:
    """Deterministic random-walk OHLCV with valid high/low ordering."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.003, n)))
    index = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(100, 1000, n),
        },
        index=index,
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return synthetic_ohlcv()
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_strategy_metadata.py`:

```python
from __future__ import annotations

import re

import pytest

from strategy_lab.strategies.registry import get_strategy, list_strategies

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


@pytest.mark.parametrize("name", list_strategies())
def test_every_strategy_declares_a_semver_version(name):
    strategy = get_strategy(name)
    assert SEMVER.match(strategy.version), f"{name} version {strategy.version!r} is not semver"


@pytest.mark.parametrize("name", list_strategies())
def test_every_strategy_declares_a_positive_warmup(name):
    strategy = get_strategy(name)
    assert isinstance(strategy.warmup_bars, int)
    assert strategy.warmup_bars > 0


@pytest.mark.parametrize("name", list_strategies())
def test_warmup_covers_the_largest_declared_lookback(name):
    """warmup_bars must be >= every span/period parameter the strategy declares."""
    strategy = get_strategy(name)
    spans = [
        value
        for field, value in vars(strategy).items()
        if isinstance(value, int) and (field.endswith("_span") or field.endswith("_period"))
    ]
    assert spans, f"{name} declares no span/period parameters to check against"
    assert strategy.warmup_bars >= max(spans)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_strategy_metadata.py -q`
Expected: FAIL — `AttributeError: 'TurnaroundV1' object has no attribute 'version'`

- [ ] **Step 4: Extend the protocol**

In `src/strategy_lab/strategies/base.py`, replace the `Strategy` protocol:

```python
class Strategy(Protocol):
    name: str
    version: str
    warmup_bars: int

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        ...
```

- [ ] **Step 5: Add the attributes to each strategy**

In `turnaround_v1.py`, inside `class TurnaroundV1`, directly under `name`:

```python
    version: str = "1.0.0"
    warmup_bars: int = 200
```

In `turnaround_v2.py`, inside `class TurnaroundV2`, directly under `name`:

```python
    version: str = "1.0.0"
    warmup_bars: int = 200
```

In `trend_following_deepseek_v4.py`, inside `class TrendFollowingDeepseekV4`, directly under `name`:

```python
    version: str = "1.0.0"
    warmup_bars: int = 40
```

In `trend_rider_v1_deepseek_v4_pro.py`, inside `class TrendRiderV1DeepseekV4Pro`, directly under `name`:

```python
    version: str = "1.0.0"
    warmup_bars: int = 40
```

Note: these are frozen dataclasses, so each new attribute needs a default and must be declared before any field without one. All existing fields already have defaults, so appending directly under `name` is safe.

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_strategy_metadata.py -q`
Expected: `12 passed`

- [ ] **Step 7: Verify no existing tests regressed**

Run: `.venv/bin/python -m pytest -q`
Expected: all previously-passing tests still pass (the strategy dataclasses gained fields with defaults, so construction is unchanged).

- [ ] **Step 8: Commit**

```bash
git add src/strategy_lab/strategies tests/conftest.py tests/test_strategy_metadata.py
git commit -m "feat(strategies): declare version and warmup_bars on the Strategy protocol"
```

---

## Task 4: Lookahead poison probe

**Files:**
- Test: `tests/test_lookahead.py`

This is the primary lookahead gate. It overwrites every bar after index `t` with violent-but-legal values and asserts row `t` is unchanged. Per the design doc §2 this catches subtle lookahead (full-sample normalization) far more efficiently than an equivalence sweep.

> **Corrections applied during implementation — the code below is the starting point, not the shipped version.**
>
> 1. **`n=400` is too short.** Measured across 40 seeds, a 400-bar frame misses a genuine
>    injected lookahead bug in `turnaround_v1` on **32.5%** of seeds; it passed on seed 7
>    only by luck, firing at a single probe point. Halving `step` does **not** help (still
>    27.5% miss) — detection depends on covering varied price regimes, not sampling
>    density. At **`n=1200` the miss rate is 0/40** for ~0.2s. Use 1200.
> 2. **A second poison profile helps, but not for the reason first assumed — and the
>    obvious version of it does nothing.** The flat poison sets `open == close`, so
>    candle-direction predicates collapse to one value across the corrupted tail, and all
>    four strategies are built on that predicate family. But the alarming early readings
>    (`long_entries` alone catching 0/10 for `turnaround_v2`) were an artifact of the
>    10-probe-point budget, **not** of the flat poison: once fix 1 raised the frame to
>    1200 bars, flat alone caught a purely candle-direction lookahead cheat on **40/40
>    seeds**. The gate was never walkable. Flat's weakness is statistical — it halves
>    detection events — not structural.
>
>    Critically, **strict alternation does not fix it**: measured against this repo's own
>    three-candle setup, flat scored 6 offenders and strict R,G,R,G scored **6** — because
>    an alternating tail never contains "two red then a green", so the conjunction stays
>    False either way. A de Bruijn sequence over all 8 triples also scored 6. A fixed run
>    pattern scored 44 on the long setup and 5 on the mirrored short setup, i.e. it
>    overfits to whichever pattern its leading bars happen to spell.
>
>    The actual variable is **phase**. The poisoned tail always starts at bar `t+1`, so any
>    fixed pattern hands every probe point the same leading candles, and only those first
>    few bars matter for short-horizon lookahead. The shipped profile therefore draws
>    directions from a PRNG **seeded by the probe index `t`**, so phase varies across probe
>    points — deterministic across runs, unbiased across patterns, and beats flat on 40/40
>    seeds. Do not "simplify" it to a fixed or modular-rotating pattern; modular rotation
>    aliases against `step` (`t % 16` with `step=20` cycles just two phases).
> 3. **`position_size` is inert on synthetic data** — the ATR scale never leaves its
>    `max_scale=1.0` clip, so the field has zero baseline variance. It is not a reliable
>    tripwire here. Noted, not fixed.
> 4. **`pythonpath = ["src"]` breaks `from tests.conftest import ...` under the bare
>    `pytest` entry point** (`python -m pytest` masks it by putting CWD on `sys.path`).
>    CLAUDE.md documents `pytest -q` as the command, so this matters. Fixed with
>    `pythonpath = ["src", "."]`.

- [ ] **Step 1: Write the test, including the cheats that prove it has teeth**

Create `tests/test_lookahead.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from strategy_lab.strategies.base import SignalSet, validate_ohlcv
from strategy_lab.strategies.registry import get_strategy, list_strategies
from tests.conftest import synthetic_ohlcv

SIGNAL_FIELDS = (
    "long_entries",
    "long_exits",
    "short_entries",
    "short_exits",
    "setup_stop_loss",
    "trend_failure_long_exits",
    "trend_failure_short_exits",
    "position_size",
)


def poison_probe(strategy, df: pd.DataFrame, *, warm: int, step: int = 20) -> list[int]:
    """Return the bar indices whose signals changed when the FUTURE was corrupted.

    A causal strategy cannot see past bar t, so replacing bars t+1.. with garbage
    must leave row t byte-identical. Any returned index is lookahead.
    """
    baseline = strategy.generate_signals(df)
    offenders: list[int] = []
    for t in range(warm, len(df) - 1, step):
        poisoned = df.copy()
        tail = poisoned.index[t + 1 :]
        poisoned.loc[tail, ["open", "high", "low", "close"]] = [1e6, 1.1e6, 0.9e6, 1e6]
        poisoned.loc[tail, "volume"] = 1e9
        probed = strategy.generate_signals(poisoned)
        for field in SIGNAL_FIELDS:
            want = getattr(baseline, field, None)
            got = getattr(probed, field, None)
            if want is None or got is None:
                continue
            if not _same(want.iloc[t], got.iloc[t]):
                offenders.append(t)
                break
    return offenders


def _same(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    return bool(a == b)


@pytest.mark.parametrize("name", list_strategies())
def test_registered_strategies_do_not_look_ahead(name):
    strategy = get_strategy(name)
    df = synthetic_ohlcv(n=400)
    offenders = poison_probe(strategy, df, warm=strategy.warmup_bars)
    assert offenders == [], f"{name} used future data at bar indices {offenders}"


# --- The probe must be able to fail. These two strategies prove it. ---


@dataclass(frozen=True)
class _BlatantCheat:
    """Enters when the NEXT bar closes higher — textbook shift(-1) lookahead."""

    name: str = "blatant_cheat"
    version: str = "1.0.0"
    warmup_bars: int = 10

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        validate_ohlcv(df)
        longs = (df["close"].shift(-1) > df["close"]).fillna(False)
        flat = pd.Series(False, index=df.index)
        return SignalSet(longs, flat, flat, flat)


@dataclass(frozen=True)
class _SubtleCheat:
    """No shift(-1) anywhere — but normalizes by the FULL-SAMPLE mean."""

    name: str = "subtle_cheat"
    version: str = "1.0.0"
    warmup_bars: int = 10

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        validate_ohlcv(df)
        z = (df["close"] - df["close"].mean()) / df["close"].std()
        longs = (z < -1.0).fillna(False)
        flat = pd.Series(False, index=df.index)
        return SignalSet(longs, flat, flat, flat)


@pytest.mark.parametrize("cheat", [_BlatantCheat(), _SubtleCheat()])
def test_probe_detects_lookahead(cheat):
    df = synthetic_ohlcv(n=400)
    offenders = poison_probe(cheat, df, warm=cheat.warmup_bars, step=10)
    assert offenders, f"{cheat.name} smuggled future data past the probe"
```

- [ ] **Step 2: Run the test**

Run: `.venv/bin/python -m pytest tests/test_lookahead.py -q`
Expected: `6 passed` — the four registered strategies are clean, and both cheats are caught.

If a registered strategy fails here, **stop and report it**: that is a real lookahead bug in existing research code, not a test problem.

- [ ] **Step 3: Commit**

```bash
git add tests/test_lookahead.py
git commit -m "test: add lookahead poison probe covering all registered strategies"
```

---

## Task 5: Migrate candle storage to NUMERIC

**Files:**
- Create: `src/strategy_lab/storage/__init__.py`
- Create: `src/strategy_lab/storage/migrations.py`
- Modify: `src/strategy_lab/db/candles.py:26-51` (column types)
- Modify: `pyproject.toml` (register the `db` pytest marker)
- Test: `tests/test_migrations.py`

Per decision D4: `Float` → `NUMERIC(38,18)`, plus the columns the live feed needs.

> ## ⚠️ The SQL written below in Step 4 SILENTLY CORRUPTS DATA. Do not run it.
>
> `ALTER COLUMN x TYPE NUMERIC(38,18)` uses Postgres' implicit `float8 → numeric` cast,
> which formats with `%.15g` (`DBL_DIG` = 15 significant digits). A float64 needs **17**
> to round-trip, so the last two digits are discarded. Verified directly:
>
> ```
> 187.6199951171875::float8::numeric        -> 187.619995117188   ✗ lossy
> 187.6199951171875::float8::text::numeric  -> 187.6199951171875  ✓ exact
> ```
>
> Running the literal Step 4 statements against this database altered **~14,700 rows** of
> Yahoo equity data (max relative drift 4.96e-15) while leaving crypto untouched — crypto
> prices are short decimals that fit in 15 digits, whereas dividend-adjusted equity closes
> like `87.84837341308594` do not. Signal *counts* survived, but bit-exact reproducibility
> of the documented ETF research did not. The rows were restored from a pre-migration
> `pg_dump`.
>
> **The correct form casts through text**, because `float8::text` emits the shortest
> round-trip representation:
>
> ```sql
> ALTER TABLE market_candles ALTER COLUMN close TYPE NUMERIC(38,18)
>   USING close::text::numeric
> ```
>
> Post-migration verification, all 103,841 rows: `close::float8::text::numeric = close`
> with **zero** mismatches on both exchanges.
>
> **Take a `pg_dump` before running any type-changing migration.** That backup is the only
> reason this was recoverable — the discarded digits cannot be reconstructed from the
> corrupted values.
>
> Two related findings from the same task:
>
> - **`pd.read_sql` defaults to `coerce_float=True`**, so it was already converting Decimal
>   to float64 and the explicit `astype` loop in `load_candles` was dead code — the
>   "documented Decimal boundary" was really a pandas default that could change under us.
>   `load_candles` now passes `coerce_float=False` so the explicit coercion is load-bearing,
>   with a regression test asserting `float64` dtype.
> - **`USING` forces a full table rewrite under an ACCESS EXCLUSIVE lock on every run.**
>   Harmless at 103k rows (sub-second), but it makes `migrate` O(table) forever. The type
>   changes are now guarded so they only fire when the column is not already `numeric`.

- [ ] **Step 1: Register the `db` marker and skip logic**

In `pyproject.toml`, extend the pytest section:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
markers = [
  "db: requires a reachable Postgres instance",
]
```

Append to `tests/conftest.py`:

```python
def _postgres_reachable() -> bool:
    import socket
    from urllib.parse import urlparse

    from strategy_lab.config import settings

    parsed = urlparse(settings.database_url.replace("postgresql+psycopg", "postgresql"))
    sock = socket.socket()
    sock.settimeout(1.5)
    try:
        sock.connect((parsed.hostname or "localhost", parsed.port or 5432))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def pytest_collection_modifyitems(config, items):
    if _postgres_reachable():
        return
    skip_db = pytest.mark.skip(reason="Postgres not reachable; start it with docker compose up -d postgres")
    for item in items:
        if "db" in item.keywords:
            item.add_marker(skip_db)
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_migrations.py`:

```python
from __future__ import annotations

import pytest
from sqlalchemy import inspect, text

from strategy_lab.db.candles import get_engine
from strategy_lab.storage.migrations import run_migrations

pytestmark = pytest.mark.db


def test_migrations_are_idempotent():
    run_migrations()
    run_migrations()  # second run must not raise


def test_candle_price_columns_are_numeric():
    run_migrations()
    columns = {c["name"]: c["type"] for c in inspect(get_engine()).get_columns("market_candles")}
    for name in ("open", "high", "low", "close", "volume"):
        assert "NUMERIC" in str(columns[name]).upper(), f"{name} is {columns[name]}, expected NUMERIC"


def test_candle_table_gains_live_feed_columns():
    run_migrations()
    columns = {c["name"] for c in inspect(get_engine()).get_columns("market_candles")}
    assert {"ts_close_ms", "quote_volume", "trades", "is_closed", "ingested_via"} <= columns


def test_existing_candles_survive_the_migration():
    run_migrations()
    with get_engine().connect() as conn:
        total = conn.execute(text("SELECT count(*) FROM market_candles")).scalar_one()
    assert total > 0, "migration must preserve existing candle rows"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_migrations.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'strategy_lab.storage'`

- [ ] **Step 4: Write the migration module**

Create `src/strategy_lab/storage/migrations.py`:

```python
from __future__ import annotations

from sqlalchemy import text

from strategy_lab.db.candles import get_engine

# Every statement must be safe to run repeatedly.
MIGRATIONS: tuple[str, ...] = (
    "ALTER TABLE market_candles ALTER COLUMN open   TYPE NUMERIC(38,18)",
    "ALTER TABLE market_candles ALTER COLUMN high   TYPE NUMERIC(38,18)",
    "ALTER TABLE market_candles ALTER COLUMN low    TYPE NUMERIC(38,18)",
    "ALTER TABLE market_candles ALTER COLUMN close  TYPE NUMERIC(38,18)",
    "ALTER TABLE market_candles ALTER COLUMN volume TYPE NUMERIC(38,18)",
    "ALTER TABLE market_candles ADD COLUMN IF NOT EXISTS ts_close_ms  BIGINT",
    "ALTER TABLE market_candles ADD COLUMN IF NOT EXISTS quote_volume NUMERIC(38,18)",
    "ALTER TABLE market_candles ADD COLUMN IF NOT EXISTS trades       INTEGER",
    "ALTER TABLE market_candles ADD COLUMN IF NOT EXISTS is_closed    BOOLEAN NOT NULL DEFAULT TRUE",
    "ALTER TABLE market_candles ADD COLUMN IF NOT EXISTS ingested_via TEXT",
)


def run_migrations(database_url: str | None = None) -> int:
    """Apply idempotent schema upgrades. Returns the number of statements executed."""
    engine = get_engine(database_url)
    with engine.begin() as conn:
        for statement in MIGRATIONS:
            conn.execute(text(statement))
    return len(MIGRATIONS)
```

Create `src/strategy_lab/storage/__init__.py`:

```python
from __future__ import annotations

from strategy_lab.storage.migrations import run_migrations

__all__ = ["run_migrations"]
```

- [ ] **Step 5: Align the SQLAlchemy table definition**

`ALTER TABLE` changes Postgres, but `metadata.create_all` on a fresh database must produce the same types. In `src/strategy_lab/db/candles.py`, change the import line and the five price columns.

Replace the `Float` import with `Numeric`:

```python
from sqlalchemy import (
    Column,
    DateTime,
    Index,
    MetaData,
    Numeric,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
```

Then replace the five column definitions inside `candles_table`:

```python
    Column("open", Numeric(38, 18), nullable=False),
    Column("high", Numeric(38, 18), nullable=False),
    Column("low", Numeric(38, 18), nullable=False),
    Column("close", Numeric(38, 18), nullable=False),
    Column("volume", Numeric(38, 18), nullable=False),
```

`load_candles` uses `pd.read_sql`, which returns `NUMERIC` as `Decimal` objects. The strategy layer needs float64, so coerce there. In `load_candles`, replace the final two lines:

```python
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        df[column] = df[column].astype("float64")
    return df.set_index("timestamp")
```

This is the single documented boundary where Decimal becomes float64 (design doc §5).

- [ ] **Step 6: Add the CLI command**

In `src/strategy_lab/cli.py`, add after the `init_db_command` function:

```python
@app.command("migrate")
def migrate_command() -> None:
    """Apply idempotent schema upgrades to an existing database."""
    from strategy_lab.storage.migrations import run_migrations

    count = run_migrations()
    typer.echo(f"Applied {count} migration statements.")
```

- [ ] **Step 7: Run the migration and the tests**

```bash
.venv/bin/strategy-lab migrate
```
Expected: `Applied 10 migration statements.`

Run: `.venv/bin/python -m pytest tests/test_migrations.py tests/test_candle_normalization.py -q`
Expected: `4 passed` for migrations plus the existing normalization tests still green.

- [ ] **Step 8: Verify existing data is intact and still loads as float**

```bash
.venv/bin/python -c "
from strategy_lab.db import load_candles
df = load_candles(exchange='binance', market_type='spot', symbol='BTC/USDT', timeframe='15m')
print(len(df), df['close'].dtype)
"
```
Expected: `83348 float64`

- [ ] **Step 9: Commit**

```bash
git add src/strategy_lab/storage src/strategy_lab/db/candles.py src/strategy_lab/cli.py pyproject.toml tests/conftest.py tests/test_migrations.py
git commit -m "feat(storage): migrate candle prices to NUMERIC and add live-feed columns"
```

---

## Task 6: Runs and signals schema

**Files:**
- Create: `src/strategy_lab/storage/schema.py`
- Modify: `src/strategy_lab/storage/migrations.py` (append table DDL)
- Test: `tests/test_signals_schema.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_signals_schema.py`:

```python
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from strategy_lab.db.candles import get_engine
from strategy_lab.storage.migrations import run_migrations

pytestmark = pytest.mark.db


def _insert_run(conn) -> uuid.UUID:
    run_id = uuid.uuid4()
    conn.execute(
        text(
            "INSERT INTO runs (run_id, mode, strategy_id, strategy_version, config) "
            "VALUES (:run_id, 'replay', 'turnaround_v2', '1.0.0', '{}'::jsonb)"
        ),
        {"run_id": run_id},
    )
    return run_id


def _insert_signal(conn, run_id, *, ts_bar_ms: int = 1_785_723_300_000) -> None:
    conn.execute(
        text(
            "INSERT INTO signals (run_id, mode, strategy_id, strategy_version, exchange, "
            "market_type, symbol, timeframe, ts_bar_ms, ts_emit_ms, bar_is_closed, side, reason) "
            "VALUES (:run_id, 'replay', 'turnaround_v2', '1.0.0', 'binance', 'perp', "
            "'BTC/USDT', '15m', :ts, :ts, TRUE, 'enter_long', 'test')"
        ),
        {"run_id": run_id, "ts": ts_bar_ms},
    )


def test_signals_reject_an_unknown_side():
    run_migrations()
    with get_engine().begin() as conn:
        run_id = _insert_run(conn)
        with pytest.raises(Exception, match="signals_side_check|violates check constraint"):
            conn.execute(
                text(
                    "INSERT INTO signals (run_id, mode, strategy_id, strategy_version, exchange, "
                    "market_type, symbol, timeframe, ts_bar_ms, ts_emit_ms, bar_is_closed, side, "
                    "reason) VALUES (:run_id, 'replay', 's', '1.0.0', 'binance', 'perp', 'B', "
                    "'15m', 1, 1, TRUE, 'sideways', 'x')"
                ),
                {"run_id": run_id},
            )


def test_duplicate_signals_are_rejected_by_the_unique_constraint():
    run_migrations()
    with get_engine().begin() as conn:
        run_id = _insert_run(conn)
        _insert_signal(conn, run_id)
        with pytest.raises(Exception, match="uq_signals_identity|duplicate key"):
            _insert_signal(conn, run_id)


def test_signals_are_append_only():
    run_migrations()
    engine = get_engine()
    with engine.begin() as conn:
        run_id = _insert_run(conn)
        _insert_signal(conn, run_id, ts_bar_ms=1_785_723_400_000)

    with pytest.raises(Exception, match="append-only"):
        with engine.begin() as conn:
            conn.execute(text("UPDATE signals SET reason = 'tampered' WHERE run_id = :r"),
                         {"r": run_id})

    with pytest.raises(Exception, match="append-only"):
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM signals WHERE run_id = :r"), {"r": run_id})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_signals_schema.py -q`
Expected: FAIL — `relation "runs" does not exist`

- [ ] **Step 3: Append the table DDL**

In `src/strategy_lab/storage/migrations.py`, leave the existing `MIGRATIONS` tuple exactly
as it is. Add a second tuple below it and concatenate — this avoids editing the closing
paren of a long literal, which is where transcription errors happen.

```python
SIGNAL_MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS runs (
      run_id           UUID PRIMARY KEY,
      mode             TEXT NOT NULL CHECK (mode IN ('backtest','replay','paper','live')),
      strategy_id      TEXT NOT NULL,
      strategy_version TEXT NOT NULL,
      config           JSONB NOT NULL DEFAULT '{}',
      started_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
      finished_at      TIMESTAMPTZ,
      warmup_until_ts_ms BIGINT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS signals (
      id               BIGSERIAL PRIMARY KEY,
      run_id           UUID NOT NULL REFERENCES runs(run_id),
      mode             TEXT NOT NULL CHECK (mode IN ('backtest','replay','paper','live')),
      strategy_id      TEXT NOT NULL,
      strategy_version TEXT NOT NULL,
      exchange         TEXT NOT NULL,
      market_type      TEXT NOT NULL,
      symbol           TEXT NOT NULL,
      timeframe        TEXT NOT NULL,
      ts_bar_ms        BIGINT NOT NULL,
      ts_emit_ms       BIGINT NOT NULL,
      bar_is_closed    BOOLEAN NOT NULL,
      side             TEXT NOT NULL CONSTRAINT signals_side_check
                       CHECK (side IN ('enter_long','exit_long','enter_short','exit_short')),
      strength         NUMERIC(10,6),
      entry_price      NUMERIC(38,18),
      stop_loss        NUMERIC(38,18),
      take_profit      NUMERIC(38,18),
      reason           TEXT NOT NULL,
      features         JSONB NOT NULL DEFAULT '{}',
      created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
      CONSTRAINT uq_signals_identity UNIQUE
        (run_id, strategy_id, strategy_version, exchange, symbol, timeframe, ts_bar_ms, side)
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_signals_lookup ON signals (symbol, timeframe, ts_bar_ms)",
    "CREATE INDEX IF NOT EXISTS ix_signals_run ON signals (run_id, ts_bar_ms)",
    """
    CREATE OR REPLACE FUNCTION signals_reject_mutation() RETURNS trigger AS $$
    BEGIN
      RAISE EXCEPTION 'signals is append-only; % is not permitted', TG_OP;
    END;
    $$ LANGUAGE plpgsql
    """,
    "DROP TRIGGER IF EXISTS trg_signals_append_only ON signals",
    """
    CREATE TRIGGER trg_signals_append_only
      BEFORE UPDATE OR DELETE ON signals
      FOR EACH ROW EXECUTE FUNCTION signals_reject_mutation()
    """,
)
```

Then change `run_migrations` to iterate over both tuples:

```python
def run_migrations(database_url: str | None = None) -> int:
    """Apply idempotent schema upgrades. Returns the number of statements executed."""
    statements = MIGRATIONS + SIGNAL_MIGRATIONS
    engine = get_engine(database_url)
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
    return len(statements)
```

The migrate command's output count changes from 10 to 17 accordingly.

- [ ] **Step 4: Write the SQLAlchemy table objects**

Create `src/strategy_lab/storage/schema.py`:

```python
from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Index,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

metadata = MetaData()

runs_table = Table(
    "runs",
    metadata,
    Column("run_id", UUID(as_uuid=True), primary_key=True),
    Column("mode", Text, nullable=False),
    Column("strategy_id", Text, nullable=False),
    Column("strategy_version", Text, nullable=False),
    Column("config", JSONB, nullable=False, server_default="{}"),
    Column("started_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("finished_at", DateTime(timezone=True)),
    Column("warmup_until_ts_ms", BigInteger),
)

signals_table = Table(
    "signals",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("run_id", UUID(as_uuid=True), nullable=False),
    Column("mode", Text, nullable=False),
    Column("strategy_id", Text, nullable=False),
    Column("strategy_version", Text, nullable=False),
    Column("exchange", Text, nullable=False),
    Column("market_type", Text, nullable=False),
    Column("symbol", Text, nullable=False),
    Column("timeframe", String(16), nullable=False),
    Column("ts_bar_ms", BigInteger, nullable=False),
    Column("ts_emit_ms", BigInteger, nullable=False),
    Column("bar_is_closed", Boolean, nullable=False),
    Column("side", Text, nullable=False),
    Column("strength", Numeric(10, 6)),
    Column("entry_price", Numeric(38, 18)),
    Column("stop_loss", Numeric(38, 18)),
    Column("take_profit", Numeric(38, 18)),
    Column("reason", Text, nullable=False),
    Column("features", JSONB, nullable=False, server_default="{}"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint(
        "run_id",
        "strategy_id",
        "strategy_version",
        "exchange",
        "symbol",
        "timeframe",
        "ts_bar_ms",
        "side",
        name="uq_signals_identity",
    ),
    Index("ix_signals_lookup", "symbol", "timeframe", "ts_bar_ms"),
)
```

- [ ] **Step 5: Apply and test**

```bash
.venv/bin/strategy-lab migrate
```
Run: `.venv/bin/python -m pytest tests/test_signals_schema.py -q`
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add src/strategy_lab/storage tests/test_signals_schema.py
git commit -m "feat(storage): add append-only runs and signals tables"
```

---

## Task 7: Signal persistence

**Files:**
- Create: `src/strategy_lab/storage/signals.py`
- Test: `tests/test_signals_storage.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_signals_storage.py`:

```python
from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

from strategy_lab.core.types import InstrumentId, Mode, Side, Signal
from strategy_lab.storage.migrations import run_migrations
from strategy_lab.storage.signals import create_run, load_signals, write_signals

pytestmark = pytest.mark.db

INSTRUMENT = InstrumentId("binance", "perp", "BTC/USDT")


def make_signal(ts_bar_ms: int, side: Side = Side.ENTER_LONG) -> Signal:
    return Signal(
        instrument=INSTRUMENT,
        timeframe="15m",
        strategy_id="turnaround_v2",
        strategy_version="1.0.0",
        ts_bar_ms=ts_bar_ms,
        ts_emit_ms=ts_bar_ms + 900_000,
        side=side,
        bar_is_closed=True,
        reason="2 red then green",
        entry_price=Decimal("63128.00"),
        stop_loss=Decimal("62740.10"),
        features={"ema200": "62110.4"},
    )


@pytest.fixture
def run_id():
    run_migrations()
    return create_run(
        run_id=uuid.uuid4(),
        mode=Mode.REPLAY,
        strategy_id="turnaround_v2",
        strategy_version="1.0.0",
        config={"source": "test"},
    )


def test_write_then_load_round_trips_a_signal(run_id):
    write_signals(run_id, Mode.REPLAY, [make_signal(1_785_723_300_000)])
    loaded = load_signals(run_id=run_id)
    assert len(loaded) == 1
    assert loaded[0].side is Side.ENTER_LONG
    assert loaded[0].entry_price == Decimal("63128.00")
    assert loaded[0].ts_bar_ms == 1_785_723_300_000


def test_rewriting_the_same_signals_is_idempotent(run_id):
    signals = [make_signal(1_785_723_300_000), make_signal(1_785_724_200_000)]
    assert write_signals(run_id, Mode.REPLAY, signals) == 2
    assert write_signals(run_id, Mode.REPLAY, signals) == 0
    assert len(load_signals(run_id=run_id)) == 2


def test_opposite_sides_on_the_same_bar_both_persist(run_id):
    """turnaround_v2 exits long and enters short on one bar — both must survive."""
    ts = 1_785_723_300_000
    written = write_signals(
        run_id,
        Mode.REPLAY,
        [make_signal(ts, Side.EXIT_LONG), make_signal(ts, Side.ENTER_SHORT)],
    )
    assert written == 2


def test_signals_load_in_bar_order(run_id):
    write_signals(
        run_id,
        Mode.REPLAY,
        [make_signal(1_785_724_200_000), make_signal(1_785_723_300_000)],
    )
    loaded = load_signals(run_id=run_id)
    assert [s.ts_bar_ms for s in loaded] == [1_785_723_300_000, 1_785_724_200_000]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_signals_storage.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'strategy_lab.storage.signals'`

- [ ] **Step 3: Write the implementation**

Create `src/strategy_lab/storage/signals.py`:

```python
from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from strategy_lab.core.types import InstrumentId, Mode, Side, Signal
from strategy_lab.db.candles import get_engine
from strategy_lab.storage.schema import runs_table, signals_table


def create_run(
    *,
    run_id: uuid.UUID,
    mode: Mode,
    strategy_id: str,
    strategy_version: str,
    config: dict,
    warmup_until_ts_ms: int | None = None,
    database_url: str | None = None,
) -> uuid.UUID:
    with get_engine(database_url).begin() as conn:
        conn.execute(
            insert(runs_table).values(
                run_id=run_id,
                mode=str(mode),
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                config=config,
                warmup_until_ts_ms=warmup_until_ts_ms,
            )
        )
    return run_id


def write_signals(
    run_id: uuid.UUID,
    mode: Mode,
    signals: Iterable[Signal],
    *,
    database_url: str | None = None,
) -> int:
    """Insert signals, skipping duplicates. Returns the count actually inserted."""
    rows = [_to_row(run_id, mode, signal) for signal in signals]
    if not rows:
        return 0

    statement = (
        insert(signals_table)
        .values(rows)
        .on_conflict_do_nothing(constraint="uq_signals_identity")
        .returning(signals_table.c.id)
    )
    with get_engine(database_url).begin() as conn:
        return len(conn.execute(statement).fetchall())


def load_signals(
    *,
    run_id: uuid.UUID | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    database_url: str | None = None,
) -> list[Signal]:
    query = select(signals_table).order_by(signals_table.c.ts_bar_ms, signals_table.c.id)
    if run_id is not None:
        query = query.where(signals_table.c.run_id == run_id)
    if symbol is not None:
        query = query.where(signals_table.c.symbol == symbol)
    if timeframe is not None:
        query = query.where(signals_table.c.timeframe == timeframe)

    with get_engine(database_url).connect() as conn:
        return [_from_row(row) for row in conn.execute(query).mappings()]


def _to_row(run_id: uuid.UUID, mode: Mode, signal: Signal) -> dict:
    return {
        "run_id": run_id,
        "mode": str(mode),
        "strategy_id": signal.strategy_id,
        "strategy_version": signal.strategy_version,
        "exchange": signal.instrument.exchange,
        "market_type": signal.instrument.market_type,
        "symbol": signal.instrument.symbol,
        "timeframe": signal.timeframe,
        "ts_bar_ms": signal.ts_bar_ms,
        "ts_emit_ms": signal.ts_emit_ms,
        "bar_is_closed": signal.bar_is_closed,
        "side": str(signal.side),
        "strength": signal.strength,
        "entry_price": signal.entry_price,
        "stop_loss": signal.stop_loss,
        "take_profit": signal.take_profit,
        "reason": signal.reason,
        "features": signal.features or {},
    }


def _from_row(row) -> Signal:
    return Signal(
        instrument=InstrumentId(row["exchange"], row["market_type"], row["symbol"]),
        timeframe=row["timeframe"],
        strategy_id=row["strategy_id"],
        strategy_version=row["strategy_version"],
        ts_bar_ms=row["ts_bar_ms"],
        ts_emit_ms=row["ts_emit_ms"],
        side=Side(row["side"]),
        bar_is_closed=row["bar_is_closed"],
        reason=row["reason"],
        entry_price=_as_decimal(row["entry_price"]),
        stop_loss=_as_decimal(row["stop_loss"]),
        take_profit=_as_decimal(row["take_profit"]),
        strength=_as_decimal(row["strength"]),
        features=row["features"],
    )


def _as_decimal(value) -> Decimal | None:
    return None if value is None else Decimal(str(value))


__all__: Sequence[str] = ["create_run", "write_signals", "load_signals"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_signals_storage.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/strategy_lab/storage/signals.py tests/test_signals_storage.py
git commit -m "feat(storage): add idempotent signal persistence"
```

---

## Task 8: Feed protocol

**Files:**
- Create: `src/strategy_lab/feeds/__init__.py`
- Create: `src/strategy_lab/feeds/base.py`
- Test: covered by Task 9 (a Protocol has no behavior of its own to test)

Per the design doc §11, this interface must not assume ascending order, forward pagination, or a single history endpoint — Binance and OKX disagree on all three. `backfill` is therefore specified as "cover this range", with direction owned by the adapter.

- [ ] **Step 1: Write the implementation**

Create `src/strategy_lab/feeds/base.py`:

```python
from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from strategy_lab.core.types import Bar, BarEvent, InstrumentId


@dataclass(frozen=True, slots=True)
class Subscription:
    instrument: InstrumentId
    timeframe: str
    include_forming: bool = False


@dataclass(frozen=True, slots=True)
class FeedHealth:
    connected: bool
    last_event_ms: int | None = None
    lag_ms: int | None = None
    reconnects: int = 0
    gaps_detected: int = 0
    weight_used: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class MarketDataFeed(Protocol):
    """Contract shared by live exchange feeds and the Postgres replay feed.

    Implementations MUST yield bars in ascending ts_open_ms order and MUST NOT
    yield the same (instrument, timeframe, ts_open_ms, is_closed) twice. Venue
    quirks — descending REST responses, backward pagination, split recent/history
    endpoints — are the adapter's problem, not the caller's.
    """

    name: str

    def stream(self, subs: Sequence[Subscription]) -> AsyncIterator[BarEvent]:
        """Yield events until exhausted (replay) or cancelled (live)."""
        ...

    def backfill(self, sub: Subscription, start_ms: int, end_ms: int) -> AsyncIterator[Bar]:
        """Yield every closed bar covering [start_ms, end_ms], ascending."""
        ...

    async def server_time_ms(self) -> int:
        ...

    def health(self) -> FeedHealth:
        ...
```

Create `src/strategy_lab/feeds/__init__.py`:

```python
from __future__ import annotations

from strategy_lab.feeds.base import FeedHealth, MarketDataFeed, Subscription

__all__ = ["FeedHealth", "MarketDataFeed", "Subscription"]
```

- [ ] **Step 2: Verify it imports and lints**

Run: `.venv/bin/python -c "from strategy_lab.feeds import MarketDataFeed; print(MarketDataFeed)" && .venv/bin/ruff check src`
Expected: the protocol prints, ruff passes.

- [ ] **Step 3: Commit**

```bash
git add src/strategy_lab/feeds
git commit -m "feat(feeds): define the MarketDataFeed protocol"
```

---

## Task 9: Replay feed

**Files:**
- Create: `src/strategy_lab/feeds/replay.py`
- Test: `tests/test_replay_feed.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_replay_feed.py`:

```python
from __future__ import annotations

import asyncio
from decimal import Decimal

from strategy_lab.core.types import InstrumentId
from strategy_lab.feeds.base import MarketDataFeed, Subscription
from strategy_lab.feeds.replay import ReplayFeed
from tests.conftest import synthetic_ohlcv

INSTRUMENT = InstrumentId("binance", "perp", "BTC/USDT")
SUB = Subscription(INSTRUMENT, "15m")


def collect(feed, subs):
    async def _run():
        return [event async for event in feed.stream(subs)]

    return asyncio.run(_run())


def test_replay_feed_satisfies_the_protocol():
    assert isinstance(ReplayFeed(frames={}), MarketDataFeed)


def test_replay_feed_yields_every_bar_in_ascending_order():
    df = synthetic_ohlcv(n=50)
    events = collect(ReplayFeed(frames={(INSTRUMENT, "15m"): df}), [SUB])
    assert len(events) == 50
    timestamps = [event.bar.ts_open_ms for event in events]
    assert timestamps == sorted(timestamps)


def test_replay_bars_are_closed_and_decimal():
    df = synthetic_ohlcv(n=5)
    events = collect(ReplayFeed(frames={(INSTRUMENT, "15m"): df}), [SUB])
    bar = events[0].bar
    assert bar.is_closed is True
    assert isinstance(bar.close, Decimal)
    assert bar.instrument == INSTRUMENT
    assert bar.timeframe == "15m"


def test_replay_bar_close_time_is_derived_from_the_timeframe():
    df = synthetic_ohlcv(n=3)
    events = collect(ReplayFeed(frames={(INSTRUMENT, "15m"): df}), [SUB])
    bar = events[0].bar
    assert bar.ts_close_ms - bar.ts_open_ms == 15 * 60 * 1000 - 1


def test_replay_event_has_no_receive_time():
    """ts_recv_ms is a live-only concept; a replay must not invent one."""
    df = synthetic_ohlcv(n=3)
    events = collect(ReplayFeed(frames={(INSTRUMENT, "15m"): df}), [SUB])
    assert all(event.ts_recv_ms is None for event in events)


def test_unknown_subscription_yields_nothing():
    other = Subscription(InstrumentId("binance", "spot", "ETH/USDT"), "1h")
    events = collect(ReplayFeed(frames={(INSTRUMENT, "15m"): synthetic_ohlcv(n=5)}), [other])
    assert events == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_replay_feed.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'strategy_lab.feeds.replay'`

- [ ] **Step 3: Write the implementation**

Create `src/strategy_lab/feeds/replay.py`:

```python
from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

import pandas as pd

from strategy_lab.core.types import Bar, BarEvent, InstrumentId
from strategy_lab.feeds.base import FeedHealth, Subscription
from strategy_lab.timeframes import timeframe_to_millis

FrameKey = tuple[InstrumentId, str]


@dataclass
class ReplayFeed:
    """Replays stored candles as BarEvents. Satisfies the same protocol as a live feed.

    This is the injection point that makes backtest, replay, and live share one
    strategy code path: the runner cannot tell this apart from a websocket.
    """

    frames: dict[FrameKey, pd.DataFrame] = field(default_factory=dict)
    name: str = "replay"
    _last_event_ms: int | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_database(
        cls,
        subscriptions: Sequence[Subscription],
        *,
        start: str | None = None,
        end: str | None = None,
        limit_bars: int | None = None,
        database_url: str | None = None,
    ) -> ReplayFeed:
        from strategy_lab.db import load_candles

        frames: dict[FrameKey, pd.DataFrame] = {}
        for sub in subscriptions:
            df = load_candles(
                exchange=sub.instrument.exchange,
                market_type=sub.instrument.market_type,
                symbol=sub.instrument.symbol,
                timeframe=sub.timeframe,
                start=start,
                end=end,
                database_url=database_url,
            )
            if limit_bars is not None:
                df = df.tail(limit_bars)
            frames[(sub.instrument, sub.timeframe)] = df
        return cls(frames=frames)

    async def stream(self, subs: Sequence[Subscription]) -> AsyncIterator[BarEvent]:
        for sub in subs:
            df = self.frames.get((sub.instrument, sub.timeframe))
            if df is None or df.empty:
                continue
            bar_ms = timeframe_to_millis(sub.timeframe)
            for timestamp, row in df.sort_index().iterrows():
                bar = _row_to_bar(timestamp, row, sub.instrument, sub.timeframe, bar_ms)
                self._last_event_ms = bar.ts_close_ms
                yield BarEvent(bar=bar, ts_event_ms=bar.ts_close_ms, ts_recv_ms=None)

    async def backfill(self, sub: Subscription, start_ms: int, end_ms: int) -> AsyncIterator[Bar]:
        df = self.frames.get((sub.instrument, sub.timeframe))
        if df is None or df.empty:
            return
        bar_ms = timeframe_to_millis(sub.timeframe)
        for timestamp, row in df.sort_index().iterrows():
            ts_open_ms = int(timestamp.timestamp() * 1000)
            if start_ms <= ts_open_ms <= end_ms:
                yield _row_to_bar(timestamp, row, sub.instrument, sub.timeframe, bar_ms)

    async def server_time_ms(self) -> int:
        return self._last_event_ms or 0

    def health(self) -> FeedHealth:
        return FeedHealth(connected=True, last_event_ms=self._last_event_ms)


def _row_to_bar(
    timestamp: pd.Timestamp,
    row: pd.Series,
    instrument: InstrumentId,
    timeframe: str,
    bar_ms: int,
) -> Bar:
    ts_open_ms = int(timestamp.timestamp() * 1000)
    return Bar(
        instrument=instrument,
        timeframe=timeframe,
        ts_open_ms=ts_open_ms,
        ts_close_ms=ts_open_ms + bar_ms - 1,
        open=_decimal(row["open"]),
        high=_decimal(row["high"]),
        low=_decimal(row["low"]),
        close=_decimal(row["close"]),
        volume=_decimal(row["volume"]),
        is_closed=True,
    )


def _decimal(value) -> Decimal:
    return Decimal(str(value))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_replay_feed.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/strategy_lab/feeds/replay.py tests/test_replay_feed.py
git commit -m "feat(feeds): add Postgres-backed replay feed"
```

---

## Task 10: Bar buffer

**Files:**
- Create: `src/strategy_lab/engine/__init__.py`
- Create: `src/strategy_lab/engine/context.py`
- Test: `tests/test_bar_buffer.py`

The buffer keeps **full history**, not a rolling window. Design doc §2 measured that a 60-bar rolling window produces wrong values for the two EWM-based strategies, because `ewm(adjust=False)` depends on every prior bar.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bar_buffer.py`:

```python
from __future__ import annotations

import pandas as pd

from strategy_lab.core.types import InstrumentId
from strategy_lab.engine.context import BarBuffer
from strategy_lab.feeds.replay import _row_to_bar
from tests.conftest import synthetic_ohlcv

INSTRUMENT = InstrumentId("binance", "perp", "BTC/USDT")
BAR_MS = 15 * 60 * 1000


def bars_from(df: pd.DataFrame):
    return [_row_to_bar(ts, row, INSTRUMENT, "15m", BAR_MS) for ts, row in df.iterrows()]


def test_buffer_starts_empty():
    assert len(BarBuffer()) == 0


def test_buffer_frame_matches_the_source_dataframe_exactly():
    df = synthetic_ohlcv(n=30)
    buffer = BarBuffer()
    for bar in bars_from(df):
        buffer.append(bar)

    frame = buffer.frame()
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    pd.testing.assert_frame_equal(frame, df[frame.columns], check_freq=False)


def test_buffer_index_is_utc_and_named_timestamp():
    df = synthetic_ohlcv(n=5)
    buffer = BarBuffer()
    for bar in bars_from(df):
        buffer.append(bar)

    frame = buffer.frame()
    assert frame.index.name == "timestamp"
    assert str(frame.index.tz) == "UTC"


def test_appending_the_same_bar_twice_replaces_rather_than_duplicates():
    """A websocket reconnect can resend the last closed bar."""
    df = synthetic_ohlcv(n=3)
    bars = bars_from(df)
    buffer = BarBuffer()
    for bar in bars:
        buffer.append(bar)
    buffer.append(bars[-1])

    assert len(buffer) == 3


def test_out_of_order_bar_is_rejected():
    df = synthetic_ohlcv(n=3)
    bars = bars_from(df)
    buffer = BarBuffer()
    buffer.append(bars[2])
    buffer.append(bars[0])
    assert len(buffer) == 1


def test_frame_is_cached_until_the_next_append():
    df = synthetic_ohlcv(n=4)
    bars = bars_from(df)
    buffer = BarBuffer()
    for bar in bars[:3]:
        buffer.append(bar)

    first = buffer.frame()
    assert buffer.frame() is first
    buffer.append(bars[3])
    assert buffer.frame() is not first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bar_buffer.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'strategy_lab.engine'`

- [ ] **Step 3: Write the implementation**

Create `src/strategy_lab/engine/context.py`:

```python
from __future__ import annotations

import pandas as pd

from strategy_lab.core.types import Bar

_COLUMNS = ("open", "high", "low", "close", "volume")


class BarBuffer:
    """Accumulates closed bars and materializes the DataFrame strategies consume.

    Retains FULL history rather than a rolling window: strategies using
    ewm(adjust=False) depend on every prior bar, so a bounded window silently
    changes their output (design doc §2).

    Values are stored as float64 — this is the documented Decimal -> float
    boundary. Money never crosses back the other way.
    """

    def __init__(self) -> None:
        self._timestamps: list[pd.Timestamp] = []
        self._columns: dict[str, list[float]] = {name: [] for name in _COLUMNS}
        self._frame: pd.DataFrame | None = None

    def __len__(self) -> int:
        return len(self._timestamps)

    def append(self, bar: Bar) -> None:
        timestamp = pd.Timestamp(bar.ts_open_ms, unit="ms", tz="UTC")

        if self._timestamps:
            last = self._timestamps[-1]
            if timestamp < last:
                return  # stale replay after a reconnect
            if timestamp == last:
                self._write(-1, bar)
                self._frame = None
                return

        self._timestamps.append(timestamp)
        for name in _COLUMNS:
            self._columns[name].append(float(getattr(bar, name)))
        self._frame = None

    def frame(self) -> pd.DataFrame:
        if self._frame is None:
            index = pd.DatetimeIndex(self._timestamps, name="timestamp")
            self._frame = pd.DataFrame(
                {name: self._columns[name] for name in _COLUMNS}, index=index
            )
        return self._frame

    def _write(self, position: int, bar: Bar) -> None:
        for name in _COLUMNS:
            self._columns[name][position] = float(getattr(bar, name))
```

Create `src/strategy_lab/engine/__init__.py`:

```python
from __future__ import annotations

from strategy_lab.engine.context import BarBuffer

__all__ = ["BarBuffer"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_bar_buffer.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/strategy_lab/engine tests/test_bar_buffer.py
git commit -m "feat(engine): add full-history bar buffer"
```

---

## Task 11: Strategy runner

**Files:**
- Create: `src/strategy_lab/engine/runner.py`
- Modify: `src/strategy_lab/engine/__init__.py`
- Test: `tests/test_runner.py`

The runner is the single event entry point. It calls the **unmodified** `generate_signals(df)` and reads the last row.

`setup_stop_loss` is a *fraction* of price, not a price — `setup_invalidation_stop_loss` returns `(close - setup_low) / close`. The runner converts it to an absolute price: `close × (1 - fraction)` for longs, `close × (1 + fraction)` for shorts.

- [ ] **Step 1: Write the failing test**

Create `tests/test_runner.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from strategy_lab.core.clock import SimClock
from strategy_lab.core.types import InstrumentId, Side
from strategy_lab.engine.runner import StrategyRunner
from strategy_lab.feeds.replay import _row_to_bar
from strategy_lab.strategies.base import SignalSet
from strategy_lab.strategies.registry import get_strategy
from tests.conftest import synthetic_ohlcv

INSTRUMENT = InstrumentId("binance", "perp", "BTC/USDT")
BAR_MS = 15 * 60 * 1000


def bars_from(df):
    return [_row_to_bar(ts, row, INSTRUMENT, "15m", BAR_MS) for ts, row in df.iterrows()]


def make_runner(strategy):
    return StrategyRunner(
        strategy=strategy,
        instrument=INSTRUMENT,
        timeframe="15m",
        clock=SimClock(),
    )


@dataclass(frozen=True)
class _AlwaysLong:
    name: str = "always_long"
    version: str = "1.0.0"
    warmup_bars: int = 3

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        true_series = pd.Series(True, index=df.index)
        flat = pd.Series(False, index=df.index)
        return SignalSet(true_series, flat, flat, flat)


def test_runner_suppresses_signals_during_warmup():
    runner = make_runner(_AlwaysLong())
    emitted = []
    for bar in bars_from(synthetic_ohlcv(n=6)):
        emitted.extend(runner.on_bar(bar))

    # warmup_bars=3 -> bars 1,2,3 suppressed; 4,5,6 emit.
    assert len(emitted) == 3


def test_runner_ignores_forming_bars_by_default():
    from dataclasses import replace

    runner = make_runner(_AlwaysLong())
    bars = bars_from(synthetic_ohlcv(n=6))
    emitted = []
    for bar in bars:
        emitted.extend(runner.on_bar(replace(bar, is_closed=False)))

    assert emitted == []


def test_runner_stamps_strategy_identity_and_bar_time():
    runner = make_runner(_AlwaysLong())
    bars = bars_from(synthetic_ohlcv(n=5))
    emitted = []
    for bar in bars:
        emitted.extend(runner.on_bar(bar))

    signal = emitted[0]
    assert signal.strategy_id == "always_long"
    assert signal.strategy_version == "1.0.0"
    assert signal.side is Side.ENTER_LONG
    assert signal.bar_is_closed is True
    assert signal.ts_bar_ms == bars[3].ts_open_ms
    assert signal.entry_price == bars[3].close


def test_runner_converts_stop_fraction_to_an_absolute_price():
    """turnaround_v1 reports the stop as a fraction of price; signals carry a price."""
    runner = make_runner(get_strategy("turnaround_v1"))
    emitted = []
    for bar in bars_from(synthetic_ohlcv(n=260)):
        emitted.extend(runner.on_bar(bar))

    entries = [s for s in emitted if s.side is Side.ENTER_LONG and s.stop_loss is not None]
    assert entries, "expected at least one long entry with a stop"
    for signal in entries:
        assert isinstance(signal.stop_loss, Decimal)
        assert Decimal("0") < signal.stop_loss < signal.entry_price


def test_runner_emits_both_sides_when_a_bar_exits_long_and_enters_short():
    """turnaround_v1 wires long_exits = short_entries, so one bar can do both."""
    runner = make_runner(get_strategy("turnaround_v1"))
    per_bar: dict[int, set[Side]] = {}
    for bar in bars_from(synthetic_ohlcv(n=300)):
        for signal in runner.on_bar(bar):
            per_bar.setdefault(signal.ts_bar_ms, set()).add(signal.side)

    assert any(
        {Side.EXIT_LONG, Side.ENTER_SHORT} <= sides for sides in per_bar.values()
    ), "expected a bar emitting both exit_long and enter_short"


def test_runner_advances_the_clock_from_event_time():
    clock = SimClock()
    runner = StrategyRunner(
        strategy=_AlwaysLong(), instrument=INSTRUMENT, timeframe="15m", clock=clock
    )
    bars = bars_from(synthetic_ohlcv(n=5))
    for bar in bars:
        runner.on_bar(bar)

    assert clock.now_ms() == bars[-1].ts_close_ms
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_runner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'strategy_lab.engine.runner'`

- [ ] **Step 3: Write the implementation**

Create `src/strategy_lab/engine/runner.py`:

```python
from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

import pandas as pd

from strategy_lab.core.clock import Clock
from strategy_lab.core.types import Bar, BarEvent, InstrumentId, Side, Signal
from strategy_lab.engine.context import BarBuffer
from strategy_lab.strategies.base import SignalSet, Strategy
from strategy_lab.timeframes import timeframe_to_millis

_ENTRY_SIDES = {Side.ENTER_LONG, Side.ENTER_SHORT}

# SignalSet field -> emitted side.
_SIDE_BY_FIELD: tuple[tuple[str, Side], ...] = (
    ("long_entries", Side.ENTER_LONG),
    ("long_exits", Side.EXIT_LONG),
    ("short_entries", Side.ENTER_SHORT),
    ("short_exits", Side.EXIT_SHORT),
)


class StrategyRunner:
    """Turns a stream of bars into Signals using an unmodified vectorized strategy.

    On every closed bar it calls strategy.generate_signals(full_buffer) and reads
    the LAST row. Because the strategy is causal, that value equals what a
    whole-history backtest would produce for the same bar — which is what makes
    backtest, replay, and live one code path. tests/test_replay_determinism.py
    enforces that equality.
    """

    def __init__(
        self,
        *,
        strategy: Strategy,
        instrument: InstrumentId,
        timeframe: str,
        clock: Clock,
        allow_forming_bars: bool = False,
    ) -> None:
        self.strategy = strategy
        self.instrument = instrument
        self.timeframe = timeframe
        self.clock = clock
        self.allow_forming_bars = allow_forming_bars
        self.buffer = BarBuffer()

    def prime(self, history: pd.DataFrame) -> None:
        """Load warmup history without emitting signals."""
        bar_ms = timeframe_to_millis(self.timeframe)
        for timestamp, row in history.sort_index().iterrows():
            ts_open_ms = int(timestamp.timestamp() * 1000)
            self.buffer.append(
                Bar(
                    instrument=self.instrument,
                    timeframe=self.timeframe,
                    ts_open_ms=ts_open_ms,
                    ts_close_ms=ts_open_ms + bar_ms - 1,
                    open=Decimal(str(row["open"])),
                    high=Decimal(str(row["high"])),
                    low=Decimal(str(row["low"])),
                    close=Decimal(str(row["close"])),
                    volume=Decimal(str(row["volume"])),
                    is_closed=True,
                )
            )

    def on_event(self, event: BarEvent) -> Sequence[Signal]:
        return self.on_bar(event.bar)

    def on_bar(self, bar: Bar) -> Sequence[Signal]:
        if not bar.is_closed and not self.allow_forming_bars:
            return ()

        if hasattr(self.clock, "advance_to"):
            self.clock.advance_to(bar.ts_close_ms)

        self.buffer.append(bar)
        if len(self.buffer) <= self.strategy.warmup_bars:
            return ()

        signal_set = self.strategy.generate_signals(self.buffer.frame())
        return self._extract(signal_set, bar)

    def _extract(self, signal_set: SignalSet, bar: Bar) -> Sequence[Signal]:
        stop_fraction = _last_float(signal_set.setup_stop_loss)
        emitted: list[Signal] = []

        for field_name, side in _SIDE_BY_FIELD:
            series = getattr(signal_set, field_name, None)
            if series is None or not bool(series.iloc[-1]):
                continue
            emitted.append(
                Signal(
                    instrument=self.instrument,
                    timeframe=self.timeframe,
                    strategy_id=self.strategy.name,
                    strategy_version=self.strategy.version,
                    ts_bar_ms=bar.ts_open_ms,
                    ts_emit_ms=self.clock.now_ms(),
                    side=side,
                    bar_is_closed=bar.is_closed,
                    reason=f"{self.strategy.name}:{field_name}",
                    entry_price=bar.close,
                    stop_loss=_stop_price(bar.close, stop_fraction, side),
                    strength=None,
                    features=_features(signal_set),
                )
            )
        return tuple(emitted)


def _last_float(series: pd.Series | None) -> float | None:
    if series is None or series.empty:
        return None
    value = series.iloc[-1]
    return None if pd.isna(value) else float(value)


def _stop_price(close: Decimal, fraction: float | None, side: Side) -> Decimal | None:
    """setup_stop_loss is a fraction of price; signals carry an absolute price."""
    if fraction is None or side not in _ENTRY_SIDES:
        return None
    offset = Decimal(str(fraction))
    if side is Side.ENTER_LONG:
        return close * (Decimal(1) - offset)
    return close * (Decimal(1) + offset)


def _features(signal_set: SignalSet) -> dict:
    return {key: str(value) for key, value in (signal_set.metadata or {}).items()}
```

Update `src/strategy_lab/engine/__init__.py`:

```python
from __future__ import annotations

from strategy_lab.engine.context import BarBuffer
from strategy_lab.engine.runner import StrategyRunner

__all__ = ["BarBuffer", "StrategyRunner"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_runner.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/strategy_lab/engine tests/test_runner.py
git commit -m "feat(engine): add StrategyRunner emitting signals from closed bars"
```

---

## Task 12: Replay determinism — the exit criterion

**Files:**
- Test: `tests/test_replay_determinism.py`

This is the test Phase 1a exists to pass. It proves the streaming path reproduces the vectorized path exactly, for every registered strategy, which is requirement 1 of the brief.

- [ ] **Step 1: Write the test**

Create `tests/test_replay_determinism.py`:

```python
from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from strategy_lab.core.clock import SimClock
from strategy_lab.core.types import InstrumentId, Side
from strategy_lab.engine.runner import StrategyRunner
from strategy_lab.feeds.base import Subscription
from strategy_lab.feeds.replay import ReplayFeed
from strategy_lab.strategies.registry import get_strategy, list_strategies
from tests.conftest import synthetic_ohlcv

INSTRUMENT = InstrumentId("binance", "perp", "BTC/USDT")
SUB = Subscription(INSTRUMENT, "15m")

_SIDE_BY_FIELD = (
    ("long_entries", Side.ENTER_LONG),
    ("long_exits", Side.EXIT_LONG),
    ("short_entries", Side.ENTER_SHORT),
    ("short_exits", Side.EXIT_SHORT),
)


def vectorized_signals(strategy, df: pd.DataFrame) -> list[tuple[int, Side]]:
    """What a whole-history backtest would produce, skipping warmup."""
    signal_set = strategy.generate_signals(df)
    out: list[tuple[int, Side]] = []
    for position, timestamp in enumerate(df.index):
        if position < strategy.warmup_bars:
            continue
        ts_ms = int(timestamp.timestamp() * 1000)
        for field_name, side in _SIDE_BY_FIELD:
            series = getattr(signal_set, field_name, None)
            if series is not None and bool(series.iloc[position]):
                out.append((ts_ms, side))
    return out


def streamed_signals(strategy, df: pd.DataFrame) -> list[tuple[int, Side]]:
    """What the event-driven runner produces from the same bars."""
    feed = ReplayFeed(frames={(INSTRUMENT, "15m"): df})
    runner = StrategyRunner(
        strategy=strategy, instrument=INSTRUMENT, timeframe="15m", clock=SimClock()
    )

    async def _run():
        collected: list[tuple[int, Side]] = []
        async for event in feed.stream([SUB]):
            for signal in runner.on_event(event):
                collected.append((signal.ts_bar_ms, signal.side))
        return collected

    return asyncio.run(_run())


@pytest.mark.parametrize("name", list_strategies())
def test_streaming_reproduces_vectorized_signals_exactly(name):
    strategy = get_strategy(name)
    df = synthetic_ohlcv(n=600)

    expected = vectorized_signals(strategy, df)
    actual = streamed_signals(strategy, df)

    assert actual == expected, (
        f"{name}: streaming and vectorized paths diverged. "
        f"{len(expected)} expected vs {len(actual)} actual signals."
    )


@pytest.mark.parametrize("name", list_strategies())
def test_replay_is_repeatable(name):
    """Same input, same signals — twice."""
    strategy = get_strategy(name)
    df = synthetic_ohlcv(n=400)
    assert streamed_signals(strategy, df) == streamed_signals(strategy, df)


def test_the_determinism_check_can_fail():
    """A non-causal strategy must break the equality — otherwise this proves nothing."""
    from dataclasses import dataclass

    from strategy_lab.strategies.base import SignalSet, validate_ohlcv

    @dataclass(frozen=True)
    class _Cheat:
        name: str = "cheat"
        version: str = "1.0.0"
        warmup_bars: int = 5

        def generate_signals(self, df: pd.DataFrame) -> SignalSet:
            validate_ohlcv(df)
            longs = (df["close"].shift(-1) > df["close"]).fillna(False)
            flat = pd.Series(False, index=df.index)
            return SignalSet(longs, flat, flat, flat)

    strategy = _Cheat()
    df = synthetic_ohlcv(n=200)
    assert streamed_signals(strategy, df) != vectorized_signals(strategy, df)


@pytest.mark.db
def test_streaming_matches_vectorized_on_real_stored_candles():
    """The real thing: BTC/USDT 15m candles already in Postgres."""
    strategy = get_strategy("turnaround_v2")
    feed = ReplayFeed.from_database(
        [Subscription(InstrumentId("binance", "spot", "BTC/USDT"), "15m")],
        limit_bars=3_000,
    )
    df = feed.frames[(InstrumentId("binance", "spot", "BTC/USDT"), "15m")]
    if df.empty:
        pytest.skip("no stored BTC/USDT 15m candles; run fetch-crypto first")

    instrument = InstrumentId("binance", "spot", "BTC/USDT")
    runner = StrategyRunner(
        strategy=strategy, instrument=instrument, timeframe="15m", clock=SimClock()
    )

    async def _run():
        collected = []
        async for event in feed.stream([Subscription(instrument, "15m")]):
            for signal in runner.on_event(event):
                collected.append((signal.ts_bar_ms, signal.side))
        return collected

    assert asyncio.run(_run()) == vectorized_signals(strategy, df)
```

- [ ] **Step 2: Run the test**

Run: `.venv/bin/python -m pytest tests/test_replay_determinism.py -q`
Expected: `10 passed` (9 without Postgres, with the db-marked one skipped).

This takes roughly 30–60 seconds: it is O(n²) by construction, which is exactly why the backtest path keeps a bulk fast path (design doc §2).

**Deliberate deviation from the design doc, flagged rather than silent:** §14 states the Phase 1a exit test as "replays 83k stored bars." The real-data test here bounds it to **3,000 bars**. At 83k the measured cost is ~6 minutes of strategy calls plus buffer rebuilds — too slow for a suite that should run on every change. 3,000 bars exercises the identical code path in seconds and still spans ~31 days of 15m candles, well past `turnaround_v2`'s 200-bar warmup. Run the full 83k sweep manually before declaring the phase done:

```bash
.venv/bin/python -m pytest tests/test_replay_determinism.py -q -k real --override-ini=addopts= 
```
after temporarily raising `limit_bars` to `None`. Record the result in the phase exit checklist.

If a strategy fails here, do **not** loosen the assertion. A mismatch means the streaming and vectorized paths genuinely disagree, which is the bug class this whole phase exists to prevent.

- [ ] **Step 3: Commit**

```bash
git add tests/test_replay_determinism.py
git commit -m "test: prove streaming replay reproduces vectorized signals exactly"
```

---

## Task 13: Replay CLI command

**Files:**
- Modify: `src/strategy_lab/cli.py` (add the `replay` command)
- Test: `tests/test_replay_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_replay_cli.py`:

```python
from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from strategy_lab.cli import app
from strategy_lab.core.types import InstrumentId
from tests.conftest import synthetic_ohlcv

runner = CliRunner()


def test_replay_reports_signal_counts(monkeypatch):
    df = synthetic_ohlcv(n=400)
    instrument = InstrumentId("binance", "perp", "BTC/USDT")

    from strategy_lab.feeds import replay as replay_module

    def fake_from_database(subscriptions, **kwargs):
        return replay_module.ReplayFeed(frames={(instrument, "15m"): df})

    monkeypatch.setattr(replay_module.ReplayFeed, "from_database", fake_from_database)

    written = {}

    def fake_create_run(**kwargs):
        return kwargs["run_id"]

    def fake_write_signals(run_id, mode, signals, **kwargs):
        rows = list(signals)
        written["count"] = written.get("count", 0) + len(rows)
        return len(rows)

    import strategy_lab.cli as cli_module

    monkeypatch.setattr(cli_module, "_create_run", fake_create_run, raising=False)
    monkeypatch.setattr(cli_module, "_write_signals", fake_write_signals, raising=False)

    result = runner.invoke(
        app,
        [
            "replay",
            "--exchange", "binance",
            "--market-type", "perp",
            "--symbol", "BTC/USDT",
            "--timeframe", "15m",
            "--strategy", "turnaround_v2",
            "--no-persist",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "signals" in result.output.lower()


def test_replay_rejects_an_unknown_strategy():
    result = runner.invoke(app, ["replay", "--strategy", "does_not_exist"])
    assert result.exit_code != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_replay_cli.py -q`
Expected: FAIL — `No such command 'replay'`

- [ ] **Step 3: Add the command**

In `src/strategy_lab/cli.py`, add these imports near the top:

```python
import asyncio
import uuid
```

Then add the command after `backtest`:

```python
@app.command("replay")
def replay_command(
    exchange: str = typer.Option("binance", help="Data exchange/source."),
    market_type: str = typer.Option("perp", help="spot, perp, or equity."),
    symbol: str = typer.Option("BTC/USDT", help="Symbol to replay."),
    timeframe: str = typer.Option("15m", help="Candle timeframe."),
    strategy_name: str = typer.Option("turnaround_v2", "--strategy", help="Strategy name."),
    start: str | None = typer.Option(None, help="Optional replay start time."),
    end: str | None = typer.Option(None, help="Optional replay end time."),
    limit_bars: int | None = typer.Option(None, help="Replay only the last N bars."),
    persist: bool = typer.Option(True, help="Write signals to Postgres."),
) -> None:
    """Replay stored candles bar-by-bar through the event engine."""
    from strategy_lab.core.clock import SimClock
    from strategy_lab.core.types import InstrumentId, Mode
    from strategy_lab.engine.runner import StrategyRunner
    from strategy_lab.feeds.base import Subscription
    from strategy_lab.feeds.replay import ReplayFeed

    strategy = get_strategy(strategy_name)
    instrument = InstrumentId(exchange, market_type, symbol)
    subscription = Subscription(instrument, timeframe)

    feed = ReplayFeed.from_database(
        [subscription], start=start, end=end, limit_bars=limit_bars
    )
    runner = StrategyRunner(
        strategy=strategy, instrument=instrument, timeframe=timeframe, clock=SimClock()
    )

    async def _run() -> list:
        collected = []
        async for event in feed.stream([subscription]):
            collected.extend(runner.on_event(event))
        return collected

    signals = asyncio.run(_run())

    if persist and signals:
        run_id = _create_run(
            run_id=uuid.uuid4(),
            mode=Mode.REPLAY,
            strategy_id=strategy.name,
            strategy_version=strategy.version,
            config={
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "start": start,
                "end": end,
                "limit_bars": limit_bars,
                "warmup_bars": strategy.warmup_bars,
            },
        )
        written = _write_signals(run_id, Mode.REPLAY, signals)
        typer.echo(f"Run {run_id}: emitted {len(signals)} signals, wrote {written}.")
        return

    typer.echo(f"Emitted {len(signals)} signals over {len(runner.buffer)} bars (not persisted).")


def _create_run(**kwargs):
    from strategy_lab.storage.signals import create_run

    return create_run(**kwargs)


def _write_signals(run_id, mode, signals):
    from strategy_lab.storage.signals import write_signals

    return write_signals(run_id, mode, signals)
```

The `_create_run` / `_write_signals` indirection exists so tests can substitute storage without a database.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_replay_cli.py -q`
Expected: `2 passed`

- [ ] **Step 5: Run it against the real stored data**

```bash
.venv/bin/strategy-lab replay --exchange binance --market-type spot --symbol BTC/USDT --timeframe 15m --strategy turnaround_v2 --limit-bars 2000
```
Expected: a line like `Run <uuid>: emitted N signals, wrote N.` with N > 0.

Verify persistence and idempotency by running the same command twice, then:

```bash
.venv/bin/python -c "
from sqlalchemy import text
from strategy_lab.db.candles import get_engine
with get_engine().connect() as c:
    print(c.execute(text('SELECT mode, count(*) FROM signals GROUP BY mode')).fetchall())
"
```
Expected: a `('replay', N)` row. Each invocation creates a new `run_id`, so counts accumulate per run — the unique constraint prevents duplicates *within* a run.

- [ ] **Step 6: Full suite and lint**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check src tests`
Expected: all tests pass, ruff clean.

- [ ] **Step 7: Commit**

```bash
git add src/strategy_lab/cli.py tests/test_replay_cli.py
git commit -m "feat(cli): add replay command driving the event engine"
```

---

## Task 14: Document the phase

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the replay workflow to the README**

Under the existing command examples, add:

````markdown
## Replay

Replay stored candles bar-by-bar through the event engine — the same code path that
will run live. Signals are persisted to the `signals` table.

```bash
strategy-lab migrate    # once, to upgrade an existing database
strategy-lab replay --exchange binance --market-type spot --symbol BTC/USDT \
  --timeframe 15m --strategy turnaround_v2 --limit-bars 2000
```

Replay is O(n²) by construction (each bar re-runs the strategy over the full buffer),
so use `--limit-bars` for large ranges. The vectorized `backtest` command remains the
fast path for whole-history runs.
````

- [ ] **Step 2: Update CLAUDE.md architecture notes**

In the "Architecture" section, after the existing data-flow paragraph, add:

````markdown
Two execution paths share one strategy implementation:

- **Vectorized** (`backtests/engine.py`, `backtest` CLI) — one `generate_signals(df)` call
  over the whole range. Fast; the research path.
- **Event-driven** (`engine/runner.py`, `replay` CLI) — one call per closed bar over an
  expanding buffer, reading the last row. This is the path live trading will use.

`tests/test_replay_determinism.py` asserts the two produce identical signals, and
`tests/test_lookahead.py` poisons future bars to prove strategies are causal. A strategy
that fails either is not safe to trade. The runner keeps **full** history rather than a
rolling window because `ewm(adjust=False)` depends on every prior bar.
````

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document the replay path and determinism guarantees"
```

---

## Phase exit criteria

Phase 1a is complete when all of these hold:

- [ ] `.venv/bin/python -m pytest -q` passes with Postgres running (no skips)
- [ ] `.venv/bin/ruff check src tests` is clean
- [ ] `tests/test_replay_determinism.py::test_streaming_matches_vectorized_on_real_stored_candles` passes against real stored BTC/USDT candles
- [ ] The same test passes once over the **full 83,348-bar** stored series (run manually, `limit_bars=None`) — this is the design doc's stated Phase 1a exit criterion
- [ ] All four registered strategies pass the lookahead poison probe
- [ ] `strategy-lab replay` writes signals to Postgres, and re-running the same range within one run inserts zero duplicates
- [ ] No network calls exist anywhere in `core/`, `engine/`, or `feeds/replay.py`

## Explicitly out of scope for Phase 1a

Deferred to later phases, per the design doc §14 — do not build these here:

- Binance REST/WebSocket feeds, reconnect, gap repair, rate limiting (Phase 1b)
- Risk layer, position sizing, kill switch, mode gating, Telegram alerts (Phase 1c)
- Fill simulation, fees, funding, metrics, walk-forward (Phase 2)
- The FastAPI read API and React dashboard (Phase 3)
