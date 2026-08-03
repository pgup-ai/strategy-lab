from __future__ import annotations

import asyncio
import inspect
from decimal import Decimal

import pandas as pd
import pytest

from strategy_lab.core.types import BarEvent, InstrumentId
from strategy_lab.db import list_candle_sets
from strategy_lab.feeds.base import FeedHealth, MarketDataFeed, Subscription
from strategy_lab.feeds.replay import ReplayFeed
from tests.conftest import synthetic_ohlcv

INSTRUMENT = InstrumentId("binance", "perp", "BTC/USDT")
SUB = Subscription(INSTRUMENT, "15m")


def collect(feed, subs):
    async def _run():
        return [event async for event in feed.stream(subs)]

    return asyncio.run(_run())


def drain_backfill(feed, sub, start_ms, end_ms):
    async def _run():
        return [bar async for bar in feed.backfill(sub, start_ms, end_ms)]

    return asyncio.run(_run())


# --------------------------------------------------------------------------------------
# Reusable conformance checks. BinanceFeed should import these rather than settle for
# isinstance(), which proves only that the member NAMES exist.
# --------------------------------------------------------------------------------------


def protocol_members(protocol: type) -> set[str]:
    """Names a Protocol requires. Python 3.11 has no public ``__protocol_attrs__``."""
    members: set[str] = set()
    for klass in protocol.__mro__:
        if klass.__module__ in ("typing", "builtins"):
            continue
        members |= set(getattr(klass, "__annotations__", {}))
        members |= {n for n, v in vars(klass).items() if callable(v) and not n.startswith("_")}
    return members


def assert_signatures_match_protocol(implementation: type, protocol: type) -> None:
    """Check that each protocol method is implemented with the parameters it declares.

    ``isinstance`` against a ``runtime_checkable`` Protocol is a ``hasattr`` sweep; a
    ``backfill(self)`` that ignores the range still passes it. This closes that gap.
    Extra parameters are allowed only if they have defaults.
    """
    for member in sorted(protocol_members(protocol)):
        assert hasattr(implementation, member), f"{implementation.__name__} is missing {member!r}"
        declared = getattr(protocol, member, None)
        if not callable(declared):  # annotation-only members such as ``name``
            continue
        expected = list(inspect.signature(declared).parameters.values())[1:]  # drop self
        actual = list(inspect.signature(getattr(implementation, member)).parameters.values())[1:]
        assert [p.name for p in actual[: len(expected)]] == [p.name for p in expected], (
            f"{implementation.__name__}.{member} does not accept the protocol's parameters"
        )
        assert all(
            p.default is not inspect.Parameter.empty for p in actual[len(expected) :]
        ), f"{implementation.__name__}.{member} adds required parameters the protocol does not"


def assert_feed_contract(feed, subs) -> list[BarEvent]:
    """Behaviour every MarketDataFeed owes its caller, live or replay.

    Ascending order is checked per ``(instrument, timeframe)``: a live feed multiplexing
    a 15m and a 1h stream is not globally ascending by ts_open_ms, and should not have to
    be. Cross-subscription ordering is covered separately.
    """
    assert isinstance(feed, MarketDataFeed)
    assert_signatures_match_protocol(type(feed), MarketDataFeed)
    assert isinstance(feed.name, str) and feed.name

    events = collect(feed, subs)
    seen: set[tuple] = set()
    latest: dict[tuple, int] = {}
    for event in events:
        assert isinstance(event, BarEvent)
        bar = event.bar
        identity = (bar.instrument, bar.timeframe, bar.ts_open_ms, bar.is_closed)
        assert identity not in seen, f"feed yielded {identity} twice"
        seen.add(identity)
        stream = (bar.instrument, bar.timeframe)
        assert bar.ts_open_ms > latest.get(stream, -1), f"{stream} went backwards"
        latest[stream] = bar.ts_open_ms
        for price in (bar.open, bar.high, bar.low, bar.close, bar.volume):
            assert isinstance(price, Decimal)
        assert event.ts_event_ms >= bar.ts_close_ms

    assert isinstance(feed.health(), FeedHealth)
    assert isinstance(asyncio.run(feed.server_time_ms()), int)
    return events


# --------------------------------------------------------------------------------------
# Protocol conformance
# --------------------------------------------------------------------------------------


def test_replay_feed_satisfies_the_protocol():
    assert isinstance(ReplayFeed(frames={}), MarketDataFeed)


def test_isinstance_alone_does_not_prove_conformance():
    """Pins how weak the previous test is, so nobody mistakes it for a real check."""

    class Impostor:
        name = "impostor"

        def stream(self):  # wrong signature: no subs
            ...

        def backfill(self):  # wrong signature: no range
            ...

        async def server_time_ms(self):
            ...

        def health(self):
            ...

    assert isinstance(Impostor(), MarketDataFeed)  # passes, and means almost nothing
    try:
        assert_signatures_match_protocol(Impostor, MarketDataFeed)
    except AssertionError:
        pass
    else:  # pragma: no cover - only reached if the signature check regresses
        raise AssertionError("signature check failed to reject a wrong-signature feed")


def test_replay_feed_satisfies_the_behavioural_feed_contract():
    feed = ReplayFeed(frames={(INSTRUMENT, "15m"): synthetic_ohlcv(n=50)})
    assert len(assert_feed_contract(feed, [SUB])) == 50


# --------------------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------------------


def test_replay_feed_yields_every_bar_in_ascending_order():
    df = synthetic_ohlcv(n=50)
    events = collect(ReplayFeed(frames={(INSTRUMENT, "15m"): df}), [SUB])
    assert len(events) == 50
    timestamps = [event.bar.ts_open_ms for event in events]
    assert timestamps == sorted(timestamps)


def test_replay_feed_sorts_an_out_of_order_frame():
    """A caller-supplied frame is not trusted to be sorted; the ordering guarantee is ours."""
    df = synthetic_ohlcv(n=50)
    shuffled = df.iloc[[7, 0, 49, 23, *range(1, 7), *range(8, 23), *range(24, 49)]]
    assert list(shuffled.index) != sorted(shuffled.index)
    events = collect(ReplayFeed(frames={(INSTRUMENT, "15m"): shuffled}), [SUB])
    timestamps = [event.bar.ts_open_ms for event in events]
    assert len(timestamps) == 50
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


def test_replay_bar_open_time_is_exact_epoch_millis():
    index = pd.date_range("2026-07-01", periods=200, freq="15min", tz="UTC", name="timestamp")
    df = synthetic_ohlcv(n=200).set_index(index)
    events = collect(ReplayFeed(frames={(INSTRUMENT, "15m"): df}), [SUB])
    assert [event.bar.ts_open_ms for event in events] == [ts.value // 1_000_000 for ts in index]


def test_sub_second_open_times_are_not_rounded_down():
    """int(ts.timestamp() * 1000) lands 1 ms low on these; ts_open_ms is a bar identity.

    Whole-second opens (every stored candle) are exact either way, so this uses the
    ms-offset timestamps that actually separate the two conversions.
    """
    offsets_ms = [57, 297, 349, 453, 517, 617, 674, 762, 819, 917]
    base = pd.Timestamp("2004-04-11 10:02:21", tz="UTC")
    index = pd.DatetimeIndex(
        [base + pd.Timedelta(milliseconds=offset) for offset in offsets_ms],
        name="timestamp",
    )
    df = synthetic_ohlcv(n=len(offsets_ms)).set_index(index)
    events = collect(ReplayFeed(frames={(INSTRUMENT, "15m"): df}), [SUB])
    assert [event.bar.ts_open_ms for event in events] == [ts.value // 1_000_000 for ts in index]


def test_replay_event_has_no_receive_time():
    """ts_recv_ms is a live-only concept; a replay must not invent one."""
    df = synthetic_ohlcv(n=3)
    events = collect(ReplayFeed(frames={(INSTRUMENT, "15m"): df}), [SUB])
    assert all(event.ts_recv_ms is None for event in events)


def test_unknown_subscription_yields_nothing():
    other = Subscription(InstrumentId("binance", "spot", "ETH/USDT"), "1h")
    events = collect(ReplayFeed(frames={(INSTRUMENT, "15m"): synthetic_ohlcv(n=5)}), [other])
    assert events == []


def test_stream_does_no_work_until_it_is_iterated():
    """stream() is an async generator: calling it must not touch the frames."""
    feed = ReplayFeed(frames={(INSTRUMENT, "15m"): synthetic_ohlcv(n=5)})
    generator = feed.stream([SUB])
    assert feed.health().last_event_ms is None
    asyncio.run(generator.aclose())
    assert feed.health().last_event_ms is None


def test_server_time_ms_is_zero_before_any_event_and_tracks_the_last_bar():
    feed = ReplayFeed(frames={(INSTRUMENT, "15m"): synthetic_ohlcv(n=5)})
    assert asyncio.run(feed.server_time_ms()) == 0
    events = collect(feed, [SUB])
    assert asyncio.run(feed.server_time_ms()) == events[-1].bar.ts_close_ms


# --------------------------------------------------------------------------------------
# Backfill
# --------------------------------------------------------------------------------------


def test_backfill_yields_only_bars_inside_the_requested_range():
    df = synthetic_ohlcv(n=10)
    feed = ReplayFeed(frames={(INSTRUMENT, "15m"): df})
    all_ms = [ts.value // 1_000_000 for ts in df.index]
    bars = drain_backfill(feed, SUB, all_ms[2], all_ms[5])
    assert [bar.ts_open_ms for bar in bars] == all_ms[2:6]
    assert all(bar.is_closed for bar in bars)


def test_backfill_on_an_unknown_subscription_stops_cleanly():
    """The bare `return` inside an async generator must end iteration, not raise."""
    other = Subscription(InstrumentId("binance", "spot", "ETH/USDT"), "1h")
    feed = ReplayFeed(frames={(INSTRUMENT, "15m"): synthetic_ohlcv(n=5)})
    assert drain_backfill(feed, other, 0, 2**63 - 1) == []
    assert drain_backfill(ReplayFeed(frames={}), SUB, 0, 2**63 - 1) == []


# --------------------------------------------------------------------------------------
# Duplicate identities
# --------------------------------------------------------------------------------------


def test_duplicate_timestamps_are_collapsed_last_wins():
    """The protocol forbids yielding one bar identity twice, so a redelivered bar
    replaces the earlier copy rather than being emitted alongside it. This is the
    websocket-reconnect case the clause exists for; the corrected copy arrives last.

    The frame is deliberately >16 rows. numpy falls back to insertion sort (which is
    stable) below that, so a 4-row version of this test passes even with an unstable
    sort and proves nothing. At 17 rows a default sort_index() keeps the STALE row.
    """
    bars = 20
    df = synthetic_ohlcv(n=bars)
    corrected = df.iloc[[1]].copy()
    corrected.loc[:, ["open", "high", "low", "close", "volume"]] = [110.0, 115.0, 105.0, 112.0, 9.0]
    events = collect(ReplayFeed(frames={(INSTRUMENT, "15m"): pd.concat([df, corrected])}), [SUB])

    timestamps = [event.bar.ts_open_ms for event in events]
    assert timestamps == sorted(timestamps)
    assert len(timestamps) == bars
    assert len(set(timestamps)) == bars

    replaced = events[1].bar
    assert replaced.ts_open_ms == df.index[1].value // 1_000_000
    assert (replaced.close, replaced.high, replaced.volume) == (
        Decimal("112.0"),
        Decimal("115.0"),
        Decimal("9.0"),
    )


def test_duplicate_timestamps_are_collapsed_in_backfill_too():
    df = synthetic_ohlcv(n=3)
    feed = ReplayFeed(frames={(INSTRUMENT, "15m"): pd.concat([df, df.iloc[[1]]])})
    bars = drain_backfill(feed, SUB, 0, 2**63 - 1)
    assert [bar.ts_open_ms for bar in bars] == [ts.value // 1_000_000 for ts in df.index]


# --------------------------------------------------------------------------------------
# Postgres entry point
# --------------------------------------------------------------------------------------


@pytest.mark.db
def test_from_database_replays_a_stored_candle_set():
    """Read-only: picks the smallest set present rather than assuming a symbol exists."""
    sets = list_candle_sets()
    usable = sets[sets["candles"] >= 2].sort_values("candles") if not sets.empty else sets
    if usable.empty:
        pytest.skip("no candle sets stored")
    meta = usable.iloc[0]
    sub = Subscription(
        InstrumentId(meta["exchange"], meta["market_type"], meta["symbol"]),
        meta["timeframe"],
    )

    feed = ReplayFeed.from_database([sub])
    events = assert_feed_contract(feed, [sub])

    assert len(events) == int(meta["candles"])
    assert events[0].bar.ts_open_ms == meta["first_timestamp"].value // 1_000_000
    assert events[-1].bar.ts_open_ms == meta["last_timestamp"].value // 1_000_000

    tail = ReplayFeed.from_database([sub], limit_bars=2)
    assert [event.bar.ts_open_ms for event in collect(tail, [sub])] == [
        event.bar.ts_open_ms for event in events[-2:]
    ]


# --------------------------------------------------------------------------------------
# Known limitations, pinned so they cannot rot into silent surprises
# --------------------------------------------------------------------------------------


def test_multiple_subscriptions_are_replayed_sequentially_not_interleaved():
    """KNOWN LIMITATION: replay drains sub A fully, then sub B.

    A live feed multiplexes both by time. Phase 1a is single-symbol, so this is
    documented rather than fixed - but a multi-symbol replay would see every bar of
    the first instrument before the first bar of the second, which is chronologically
    wrong. Fix this alongside the live feed, not before.
    """
    other_instrument = InstrumentId("binance", "spot", "ETH/USDT")
    other_sub = Subscription(other_instrument, "15m")
    feed = ReplayFeed(
        frames={
            (INSTRUMENT, "15m"): synthetic_ohlcv(n=5),
            (other_instrument, "15m"): synthetic_ohlcv(n=5, seed=11),
        }
    )
    events = collect(feed, [SUB, other_sub])
    instruments = [event.bar.instrument for event in events]
    assert instruments == [INSTRUMENT] * 5 + [other_instrument] * 5
    timestamps = [event.bar.ts_open_ms for event in events]
    assert timestamps != sorted(timestamps), "expected the sequential-drain ordering bug"


