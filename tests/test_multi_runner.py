from __future__ import annotations

import asyncio

import pytest

from strategy_lab.core.clock import SimClock
from strategy_lab.core.types import InstrumentId
from strategy_lab.engine.multi_runner import MultiAssetRunner
from strategy_lab.engine.runner import StrategyRunner
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
    return ReplayFeed(frames={BTC.at("4h"): synthetic_ohlcv(n=n, freq="4h", seed=1),
                              ETH.at("4h"): synthetic_ohlcv(n=n, freq="4h", seed=2)})


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


def test_one_instrument_matches_the_single_asset_runner():
    """A one-instrument MultiAssetRunner must not diverge from StrategyRunner."""
    df = synthetic_ohlcv(n=400, freq="4h", seed=1)
    strategy = get_strategy("donchian")

    multi = MultiAssetRunner(strategies={BTC: strategy}, timeframe="4h", clock=SimClock())
    multi_signals = run(multi, ReplayFeed(frames={BTC.at("4h"): df}), [Subscription(BTC, "4h")])

    single = StrategyRunner(strategy=strategy, instrument=BTC, timeframe="4h", clock=SimClock())
    single_signals = []
    for event in ReplayFeed(frames={BTC.at("4h"): df})._events_for(Subscription(BTC, "4h")):
        single_signals.extend(single.on_event(event))

    assert multi_signals, "two empty lists would match trivially"
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
    assert signals and all(s.instrument == BTC for s in signals)


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


def test_a_bar_at_another_timeframe_is_rejected_rather_than_absorbed():
    """Dispatch keys on instrument, so a second timeframe would overwrite the first.

    The runner holds one buffer per instrument and one configured timeframe; a 1d
    bar arriving at a 4h runner has nowhere correct to go, and appending it to the
    4h buffer would corrupt the series the signal is computed from.
    """
    runner = MultiAssetRunner(
        strategies={BTC: get_strategy("donchian")}, timeframe="4h", clock=SimClock()
    )
    feed = ReplayFeed(frames={BTC.at("1d"): synthetic_ohlcv(n=5, freq="1D")})

    with pytest.raises(ValueError, match="configured for '4h'"):
        run(runner, feed, [Subscription(BTC, "1d")])


def test_both_instruments_produce_signals_in_a_two_strategy_run():
    """The phase's headline claim: one stream, two instruments, both traded.

    Attribution is structurally guaranteed by delegating to a StrategyRunner per
    instrument, but nothing else asserts the multi-asset run actually trades more
    than one -- a dispatch that quietly routed every bar to the first strategy
    would leave every other test in this file green.
    """
    strategy = get_strategy("donchian")
    runner = MultiAssetRunner(
        strategies={BTC: strategy, ETH: strategy}, timeframe="4h", clock=SimClock()
    )
    signals = run(runner, two_instrument_feed(), [Subscription(BTC, "4h"), Subscription(ETH, "4h")])

    traded = {signal.instrument for signal in signals}
    assert traded == {BTC, ETH}, f"expected both instruments to trade, got {traded}"
