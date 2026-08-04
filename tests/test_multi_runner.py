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
    # Length alone cannot distinguish two buffers from one shared buffer: both
    # instruments carry the same 400 timestamps and BarBuffer overwrites a repeated
    # one, so a shared buffer would also report 400. Identity is in the contents.
    assert runner.buffer(BTC).frame()["close"].tolist() == (
        synthetic_ohlcv(n=400, freq="4h", seed=1)["close"].tolist()
    )
    assert runner.buffer(ETH).frame()["close"].tolist() == (
        synthetic_ohlcv(n=400, freq="4h", seed=2)["close"].tolist()
    )


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
    # Matching the canonical key rather than the bare symbol is what separates the
    # deliberate guard from an incidental dict miss further down: only the guard
    # formats InstrumentId.key, a raw KeyError carries the dataclass repr instead.
    with pytest.raises(KeyError, match="binance:perp:ETH/USDT"):
        run(runner, feed, [Subscription(BTC, "4h"), Subscription(ETH, "4h")])
