"""The paper command's own wiring, exercised without a venue or a database.

``LiveFeed`` is driven by a scripted fetch and a sleep that does not sleep, the
funding top-up is patched, and the storage indirections record instead of insert
-- so these cover what the command owns: priming from ``backfill``, the cursor
handoff to the poll, the book, the incremental flush, and the bar log.

The parts underneath have their own suites. This is the seam between them, and
before R10h nothing exercised it at all because nothing constructed either half.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

import pandas as pd
import pytest
from typer.testing import CliRunner

from strategy_lab import cli
from strategy_lab.core.clock import SimClock
from strategy_lab.core.types import Mode
from strategy_lab.feeds.live import LiveFeed
from strategy_lab.strategies.base import SignalSet
from tests.conftest import synthetic_ohlcv_with_funding

runner = CliRunner()
TIMEFRAME = "4h"


@dataclass(frozen=True)
class _EveryBar:
    name: str = "every_bar"
    version: str = "9.9.9"
    warmup_bars: int = 3

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        flat = pd.Series(False, index=df.index)
        return SignalSet(pd.Series(True, index=df.index), flat, flat, flat)


@dataclass(frozen=True)
class _Sized:
    """Carries a per-bar scale, which no `Signal` can express."""

    name: str = "sized"
    version: str = "1.0.0"
    warmup_bars: int = 3

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        flat = pd.Series(False, index=df.index)
        return SignalSet(
            pd.Series(True, index=df.index),
            flat,
            flat,
            flat,
            position_size=pd.Series(0.5, index=df.index),
        )


@pytest.fixture
def wired(monkeypatch):
    """The command with its venue, funding top-up and storage replaced."""
    frame = synthetic_ohlcv_with_funding(n=40, freq=TIMEFRAME)
    primed_through = 25
    revealed = {"n": primed_through + 1}
    # The clock walks the fixture rather than the wall: `backfill` bounds itself
    # by the clock, so a clock in 2026 over bars from 2024 filters every one of
    # them out and primes nothing.
    clock = SimClock(int(frame.index[primed_through].value // 10**6))

    def fetch(identity, since_ms, until_ms=None):
        window = frame[frame.index >= pd.Timestamp(since_ms, unit="ms", tz="UTC")]
        if until_ms is not None:  # backfill, which is bounded on both sides
            return window[window.index <= pd.Timestamp(until_ms, unit="ms", tz="UTC")]
        # A poll: one more bar has closed since the last one, as it would live.
        window = window[window.index <= frame.index[revealed["n"] - 1]]
        if revealed["n"] < len(frame):
            revealed["n"] += 1
            clock.advance_to(int(frame.index[revealed["n"] - 1].value // 10**6))
        return window

    async def _noop(_seconds):
        # `asyncio.sleep(0)` rather than a bare return: a coroutine that never
        # suspends never hands control back to the loop, so the command's
        # `wait_for` deadline cannot fire and the run spins forever. Production
        # sleeps for real; only the fixture has to remember to yield.
        await asyncio.sleep(0)

    monkeypatch.setattr(cli, "_advance_funding", lambda identity: 7)
    monkeypatch.setattr(
        cli,
        "_live_feed",
        lambda **kwargs: LiveFeed(**{**kwargs, "fetch": fetch, "sleep": _noop, "clock": clock}),
    )

    written = {"runs": [], "signals": [], "reasons": []}
    monkeypatch.setattr(cli, "_create_run", lambda **kw: written["runs"].append(kw) or kw["run_id"])
    monkeypatch.setattr(
        cli,
        "_write_signals",
        lambda run_id, mode, signals: written["signals"].append((mode, list(signals)))
        or len(signals),
    )
    monkeypatch.setattr(
        cli,
        "_write_bar_reasons",
        lambda run_id, mode, reasons: written["reasons"].append((mode, list(reasons)))
        or len(reasons),
    )
    return written


def _invoke(*args):
    result = runner.invoke(
        cli.app,
        ["paper", "--timeframe", TIMEFRAME, "--for-minutes", "0.01", *args],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    return result.output


def test_a_paper_run_primes_polls_and_records(monkeypatch, wired):
    """The seam the phase exists to close: `backfill` warms the runner, the poll
    feeds it, and what it decided reaches storage under `paper`."""
    monkeypatch.setattr(cli, "get_strategy", lambda name: _EveryBar())

    output = _invoke("--strategy", "every_bar")

    # warmup + lookback, less the one still forming at the clock's instant.
    assert "Primed 8 bars" in output
    assert wired["runs"], "a paper run that happened left no header"
    assert wired["runs"][0]["mode"] is Mode.PAPER
    assert wired["signals"], "the run emitted nothing to storage"
    assert all(mode is Mode.PAPER for mode, _ in wired["signals"])


def test_nothing_is_written_twice_across_flushes(monkeypatch, wired):
    """Signals are flushed per bar so an hours-long run survives what ends it.
    The claim that makes that safe is that a flush writes only the new tail."""
    monkeypatch.setattr(cli, "get_strategy", lambda name: _EveryBar())

    _invoke("--strategy", "every_bar")

    seen = [(s.ts_bar_ms, s.side) for _, batch in wired["signals"] for s in batch]
    assert seen, "nothing was written, so the incremental claim is untested"
    assert len(seen) == len(set(seen)), f"a flush rewrote what an earlier one had: {seen}"


def test_persist_off_writes_nothing_at_all(monkeypatch, wired):
    monkeypatch.setattr(cli, "get_strategy", lambda name: _EveryBar())

    output = _invoke("--strategy", "every_bar", "--no-persist")

    assert "Not persisted." in output
    assert not wired["runs"] and not wired["signals"] and not wired["reasons"]


def test_a_per_bar_scale_is_declared_rather_than_silently_dropped(monkeypatch, wired):
    """`StrategyRunner` emits `strength=None` and no `Signal` carries a size, so
    the book fills every entry at scale 1.0. For a strategy that sizes per bar
    that is a real divergence from its backtest -- the one property R10g's book
    was built to have -- and it is worth a sentence rather than a different
    number nobody was told about."""
    monkeypatch.setattr(cli, "get_strategy", lambda name: _Sized())

    assert "fills every entry at scale 1.0" in _invoke("--strategy", "sized")


def test_a_strategy_without_a_per_bar_scale_says_nothing(monkeypatch, wired):
    """The bound on the above: a warning on every run is a warning nobody reads."""
    monkeypatch.setattr(cli, "get_strategy", lambda name: _EveryBar())

    assert "scale 1.0" not in _invoke("--strategy", "every_bar")


def test_the_bar_log_records_what_the_live_path_received(monkeypatch, wired, tmp_path):
    """Without it the delayed oracle cannot tell a venue revision from a
    closed-bar error, because both surface only as a derived difference."""
    monkeypatch.setattr(cli, "get_strategy", lambda name: _EveryBar())
    path = tmp_path / "bars.csv"

    _invoke("--strategy", "every_bar", "--bars-csv", str(path))

    logged = pd.read_csv(path)
    assert not logged.empty, "the run logged no bars"
    assert list(logged.columns) == [
        "ts_open_ms", "open", "high", "low", "close", "volume", "funding_rate", "is_closed",
    ]
    assert logged["funding_rate"].notna().all(), "a funded perp bar logged no rate"


def test_the_bar_log_is_absent_unless_asked_for(monkeypatch, wired, tmp_path):
    """A live process writes no candles by default: the oracle compares against a
    later independent fetch, and a run that stored its own bars as the record
    would be compared against itself."""
    monkeypatch.setattr(cli, "get_strategy", lambda name: _EveryBar())

    _invoke("--strategy", "every_bar")

    assert not list(tmp_path.iterdir())


def test_advance_funding_asks_nothing_of_a_market_that_settles_nothing():
    """The same rule `board_window` follows: a query that can only return `None`
    is how a coverage guard gets invented for a market that has none."""
    from strategy_lab.market_data.base import MarketDataIdentity

    spot = MarketDataIdentity(
        exchange="binance", market_type="spot", symbol="BTC/USDT", timeframe=TIMEFRAME
    )
    assert cli._advance_funding(spot) is None


def test_a_run_header_carries_what_would_be_needed_to_repeat_it(monkeypatch, wired):
    """A paper run cannot be re-run — its window is gone — so the header is the
    only place the shape of it survives."""
    monkeypatch.setattr(cli, "get_strategy", lambda name: _EveryBar())

    _invoke("--strategy", "every_bar", "--cash", "5000")

    config = wired["runs"][0]["config"]
    for key in ("exchange", "market_type", "symbol", "timeframe", "warmup_bars",
                "primed_bars", "cash", "position_pct", "started_ms"):
        assert key in config, f"the header cannot say what {key} was"
    assert config["cash"] == 5000
    assert isinstance(wired["runs"][0]["run_id"], uuid.UUID)


def test_funding_is_advanced_during_a_stall_not_only_between_them(monkeypatch, wired):
    """The bug the first two real runs found. The top-up ran inside the consumer
    loop, and a withheld poll yields no event -- so the fetch that would end a
    stall could only happen while there was no stall. Measured: coverage lapsed
    at 00:00, one cadence past the 16:00 settlement, and both runs stalled for
    every remaining poll (27 and 26) and lost the bar that closed there.
    """
    monkeypatch.setattr(cli, "get_strategy", lambda name: _EveryBar())
    monkeypatch.setattr(cli, "FUNDING_CHECK_SECONDS", 0.001)

    # A feed that only ever withholds: no event reaches the consumer, which is
    # the condition under which the old code could not fetch at all.
    stalling = {"n": 0}

    class _Stalled:
        lookback_bars = 5
        clock = SimClock(0)

        @property
        def funding_withheld_polls(self):
            stalling["n"] += 1
            return stalling["n"]

        def resume_after(self, *_):
            pass

        async def backfill(self, *_a, **_kw):
            return
            yield  # pragma: no cover - makes this an async generator

        async def stream(self, _subs):
            while True:
                await asyncio.sleep(0)
            yield  # unreachable, and what makes this an async generator

    monkeypatch.setattr(cli, "_live_feed", lambda **_: _Stalled())
    advanced = []
    monkeypatch.setattr(cli, "_advance_funding", lambda identity: advanced.append(1) or 1)

    _invoke("--strategy", "every_bar", "--no-persist")

    assert len(advanced) > 1, (
        "funding was fetched once at startup and never again, so a stall that "
        "began after startup could never end"
    )
