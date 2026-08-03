"""The replay command, exercised without a database.

``ReplayFeed.from_database`` is patched to serve an in-memory frame and the two
storage indirections in ``cli`` are patched to record instead of insert, so these
tests cover the wiring the command owns -- identity, feed arguments, the runner
loop, and what reaches storage -- rather than Postgres.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import pandas as pd
import pytest
from typer.testing import CliRunner

from strategy_lab import cli
from strategy_lab.core.types import Mode, Side
from strategy_lab.feeds.replay import ReplayFeed
from strategy_lab.strategies.base import SignalSet
from tests.conftest import synthetic_ohlcv

runner = CliRunner()


@dataclass(frozen=True)
class _EveryBar:
    """Emits one long entry per bar past warmup, so counts are exactly predictable."""

    name: str = "every_bar"
    version: str = "9.9.9"
    warmup_bars: int = 3

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        flat = pd.Series(False, index=df.index)
        return SignalSet(pd.Series(True, index=df.index), flat, flat, flat)


@dataclass(frozen=True)
class _Silent:
    name: str = "silent"
    version: str = "0.0.1"
    warmup_bars: int = 3

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        flat = pd.Series(False, index=df.index)
        return SignalSet(flat, flat, flat, flat)


@pytest.fixture
def feed_calls(monkeypatch):
    """Serve 10 synthetic bars for whatever subscription the command builds."""
    calls: list[dict] = []
    df = synthetic_ohlcv(n=10)

    def fake_from_database(cls, subscriptions, **kwargs):
        subs = list(subscriptions)
        calls.append({"subscriptions": subs, "kwargs": kwargs})
        return cls(frames={(sub.instrument, sub.timeframe): df for sub in subs})

    monkeypatch.setattr(ReplayFeed, "from_database", classmethod(fake_from_database))
    return calls


@pytest.fixture
def storage_calls(monkeypatch):
    calls: dict[str, list] = {"runs": [], "writes": []}

    def fake_create_run(**kwargs):
        calls["runs"].append(kwargs)
        return kwargs["run_id"]

    def fake_write_signals(run_id, mode, signals):
        signals = list(signals)
        calls["writes"].append({"run_id": run_id, "mode": mode, "signals": signals})
        return len(signals)

    monkeypatch.setattr(cli, "_create_run", fake_create_run)
    monkeypatch.setattr(cli, "_write_signals", fake_write_signals)
    return calls


def use_strategy(monkeypatch, strategy):
    monkeypatch.setattr(cli, "get_strategy", lambda name, **kwargs: strategy)


def test_replay_reports_the_signal_count_without_persisting(monkeypatch, feed_calls):
    use_strategy(monkeypatch, _EveryBar())

    result = runner.invoke(cli.app, ["replay", "--no-persist"])

    # 10 bars, warmup 3 -> bars 4..10 emit, one signal each.
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "Emitted 7 signals over 10 bars (not persisted)."


def test_replay_builds_the_subscription_from_the_options(monkeypatch, feed_calls):
    use_strategy(monkeypatch, _EveryBar())

    result = runner.invoke(
        cli.app,
        [
            "replay",
            "--exchange", "binance",
            "--market-type", "spot",
            "--symbol", "ETH/USDT",
            "--timeframe", "1h",
            "--start", "2024-01-01",
            "--end", "2024-02-01",
            "--limit-bars", "500",
            "--no-persist",
        ],
    )

    assert result.exit_code == 0, result.output
    [call] = feed_calls
    [subscription] = call["subscriptions"]
    assert subscription.instrument.exchange == "binance"
    assert subscription.instrument.market_type == "spot"
    assert subscription.instrument.symbol == "ETH/USDT"
    assert subscription.timeframe == "1h"
    assert call["kwargs"] == {"start": "2024-01-01", "end": "2024-02-01", "limit_bars": 500}


def test_replay_persists_the_run_header_and_every_signal(
    monkeypatch, feed_calls, storage_calls
):
    use_strategy(monkeypatch, _EveryBar())

    result = runner.invoke(cli.app, ["replay", "--symbol", "ETH/USDT", "--limit-bars", "10"])

    assert result.exit_code == 0, result.output
    [run] = storage_calls["runs"]
    [write] = storage_calls["writes"]

    assert isinstance(run["run_id"], uuid.UUID)
    assert run["mode"] is Mode.REPLAY
    assert run["strategy_id"] == "every_bar"
    assert run["strategy_version"] == "9.9.9"
    assert run["config"] == {
        "exchange": "binance",
        "market_type": "perp",
        "symbol": "ETH/USDT",
        "timeframe": "15m",
        "start": None,
        "end": None,
        "limit_bars": 10,
        "warmup_bars": 3,
    }

    # The signals reaching storage are the ones the engine emitted, not a count.
    assert write["run_id"] == run["run_id"]
    assert write["mode"] is Mode.REPLAY
    assert len(write["signals"]) == 7
    assert {signal.side for signal in write["signals"]} == {Side.ENTER_LONG}
    assert {signal.instrument.symbol for signal in write["signals"]} == {"ETH/USDT"}
    assert {signal.timeframe for signal in write["signals"]} == {"15m"}
    assert f"Run {run['run_id']}: emitted 7 signals, wrote 7." in result.output


def test_replay_writes_nothing_when_no_signals_fire(monkeypatch, feed_calls, storage_calls):
    """An empty replay must not leave an orphan run header behind."""
    use_strategy(monkeypatch, _Silent())

    result = runner.invoke(cli.app, ["replay"])

    assert result.exit_code == 0, result.output
    assert storage_calls["runs"] == []
    assert storage_calls["writes"] == []
    assert result.output.strip() == "Emitted 0 signals over 10 bars (not persisted)."


def test_unknown_strategy_exits_non_zero(feed_calls):
    result = runner.invoke(cli.app, ["replay", "--strategy", "does_not_exist", "--no-persist"])

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "does_not_exist" in str(result.exception)
