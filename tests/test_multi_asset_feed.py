from __future__ import annotations

import asyncio

import pytest

from strategy_lab.core.types import InstrumentId
from strategy_lab.engine.market_clock import MarketClock
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


def test_ties_break_deterministically_on_instrument_key():
    """Subscribed SOL, BTC, ETH: with no key in the sort, ties keep that arrival
    order and the interleaving depends on how the caller happened to list the subs.
    """
    frames = {BTC.at("4h"): synthetic_ohlcv(n=3, freq="4h"),
              ETH.at("4h"): synthetic_ohlcv(n=3, freq="4h"),
              SOL.at("4h"): synthetic_ohlcv(n=3, freq="4h")}
    subs = [Subscription(SOL, "4h"), Subscription(BTC, "4h"), Subscription(ETH, "4h")]

    keys = [e.bar.instrument.key for e in drain(ReplayFeed(frames=frames), subs)]
    assert keys == [BTC.key, ETH.key, SOL.key] * 3


def test_one_symbol_at_two_timeframes_ties_with_itself_and_stays_ordered():
    """The tie the instrument key cannot break: BTC 4h and BTC 1d close together.

    ``heapq.merge`` is stable, so an instrument-only key leaves those two in
    whatever order the caller happened to list the subscriptions -- deterministic
    per call, but a different stream for the same data. Subscribing both ways round
    is what distinguishes a real total order from that accident.
    """
    frames = {BTC.at("4h"): synthetic_ohlcv(n=12, freq="4h"),
              BTC.at("1d"): synthetic_ohlcv(n=2, freq="1D")}
    four_hour, daily = Subscription(BTC, "4h"), Subscription(BTC, "1d")

    forward = [e.bar.candle.key for e in drain(ReplayFeed(frames=frames), [four_hour, daily])]
    reversed_ = [e.bar.candle.key for e in drain(ReplayFeed(frames=frames), [daily, four_hour])]

    assert forward == reversed_, "tie order followed subscription order, not the key"
    # At a shared close the 1d bar sorts first ("1d" < "4h"); asserting the position
    # pins that the timeframe is in the key rather than merely that both arrive.
    assert forward[5:7] == [BTC.at("1d").key, BTC.at("4h").key]


def test_a_symbol_at_two_timeframes_loses_no_bars_in_snapshots():
    """Every streamed bar must reach a snapshot; the snapshot key is what decides.

    Keyed by instrument, the 4h and 1d bars sharing a close time collapse onto one
    entry and the day boundary silently drops a bar -- measured at 2 lost of 14
    before the candle key landed.
    """
    frames = {BTC.at("4h"): synthetic_ohlcv(n=12, freq="4h"),
              BTC.at("1d"): synthetic_ohlcv(n=2, freq="1D")}
    subs = [Subscription(BTC, "4h"), Subscription(BTC, "1d")]

    clock = MarketClock()
    events = drain(ReplayFeed(frames=frames), subs)
    survived = sum(
        len(snapshot)
        for snapshot in (clock.on_event(event) for event in events)
        if snapshot is not None
    )
    final = clock.flush()
    survived += 0 if final is None else len(final)

    assert len(events) == 14, "fixture changed; the loss count below is calibrated to it"
    assert survived == 14


def test_a_single_subscription_is_unchanged():
    """The one-instrument path is what test_replay_determinism.py rests on."""
    df = synthetic_ohlcv(n=20, freq="4h")
    events = drain(ReplayFeed(frames={BTC.at("4h"): df}), [Subscription(BTC, "4h")])
    assert [e.bar.ts_open_ms for e in events] == [
        int(ts.value // 1_000_000) for ts in df.index
    ]


def test_instruments_with_different_histories_still_merge():
    """ETH lists later than BTC; the merge must not assume equal lengths."""
    btc = synthetic_ohlcv(n=6, freq="4h")
    eth = synthetic_ohlcv(n=6, freq="4h").iloc[3:]
    frames = {BTC.at("4h"): btc, ETH.at("4h"): eth}
    events = drain(ReplayFeed(frames=frames), [Subscription(BTC, "4h"), Subscription(ETH, "4h")])

    times = [e.ts_event_ms for e in events]
    assert times == sorted(times)
    assert len(events) == 9
    assert events[0].bar.instrument == BTC


def test_mixed_timeframes_merge_on_event_time():
    """A 4h and a 1d subscription interleave by close time, not by bar index."""
    frames = {BTC.at("4h"): synthetic_ohlcv(n=12, freq="4h"),
              ETH.at("1d"): synthetic_ohlcv(n=2, freq="1D")}
    events = drain(ReplayFeed(frames=frames), [Subscription(BTC, "4h"), Subscription(ETH, "1d")])
    times = [e.ts_event_ms for e in events]
    assert times == sorted(times)
    assert len(events) == 14


@pytest.mark.db
def test_stored_btc_and_eth_perps_merge_into_one_ordered_stream():
    """Read-only, on the real research data: the blocker this phase removes.

    Their histories start two months apart, so this also exercises the unequal-length
    path at the scale it actually occurs rather than on a six-bar fixture.
    """
    subs = [Subscription(BTC, "4h"), Subscription(ETH, "4h")]
    feed = ReplayFeed.from_database(subs)
    stored = {sub.instrument: feed.frames[sub.candle] for sub in subs}
    if any(df.empty for df in stored.values()):
        pytest.skip("binance perp BTC/USDT and ETH/USDT 4h are not both stored")

    events = drain(feed, subs)

    times = [e.ts_event_ms for e in events]
    assert times == sorted(times)
    assert {e.bar.instrument for e in events} == {BTC, ETH}
    assert len(events) == sum(len(df) for df in stored.values())
