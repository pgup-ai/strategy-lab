"""Which live stream belongs to an identity, and — mostly — that there is none.

Venue knowledge is wrong in ways a chart still renders: a URL that connects to
the wrong market draws real prices for the wrong instrument, and nothing about
the picture says so. So it lives in Python where it can be pinned.
"""

from __future__ import annotations

import pytest

from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.market_data.streams import stream_url


def _id(exchange: str, market_type: str, symbol: str, timeframe: str) -> MarketDataIdentity:
    return MarketDataIdentity(
        exchange=exchange, market_type=market_type, symbol=symbol, timeframe=timeframe
    )


def test_spot_and_perp_are_different_services_not_different_paths():
    """Binance serves USD-M futures from its own host. A spot URL subscribed for
    a perp connects and streams the *spot* price, which on a chart of perp
    candles is a real price for the wrong market."""
    spot = stream_url(_id("binance", "spot", "BTC/USDT", "15m"))
    perp = stream_url(_id("binance", "perp", "BTC/USDT", "15m"))

    assert spot == "wss://stream.binance.com:9443/ws/btcusdt@kline_15m"
    assert perp == "wss://fstream.binance.com/ws/btcusdt@kline_15m"


@pytest.mark.parametrize(
    "identity",
    [
        _id("yahoo", "equity", "SPY", "1w"),          # publishes no stream at all
        _id("binance", "spot", "BTC/USDT", "1wk"),    # the Yahoo spelling of a week
        _id("okx", "perp", "BTC/USDT", "4h"),         # no candles stored from it
    ],
    ids=["equity", "yahoo-timeframe", "unfetched-venue"],
)
def test_there_is_no_stream_for_most_things(identity):
    """Absence is the common answer and it is not an error: the page shows no
    live control rather than a control that cannot connect.

    `1wk` is the sharp one. It is a real dataset under a different name, and a
    "week is a week" mapping would subscribe it to `1w` — a stream whose bars
    are a different series from the ones on screen.
    """
    assert stream_url(identity) is None


def test_the_symbol_is_the_venue_s_spelling_not_this_repo_s():
    assert stream_url(_id("binance", "perp", "ETH/USDT", "4h")).endswith("/ethusdt@kline_4h")


def test_a_settle_suffix_is_not_part_of_the_stream_name():
    """`BTC/USDT:USDT` is ccxt's spelling of a perp and `IdentityQuery` accepts
    it. Left in, the URL still *connects* — it just subscribes to nothing, which
    is the silent half of a wrong stream."""
    assert stream_url(_id("binance", "perp", "BTC/USDT:USDT", "15m")) == (
        "wss://fstream.binance.com/ws/btcusdt@kline_15m"
    )


def test_the_interval_set_is_what_this_repo_can_hold_not_what_binance_publishes():
    """Binance publishes `1M`; `timeframe_to_millis` raises on it because a month
    is not a fixed width and every bar calculation here is width arithmetic. A
    stream for a dataset that cannot exist is a URL nobody can use."""
    from strategy_lab.timeframes import timeframe_to_millis

    assert stream_url(_id("binance", "spot", "BTC/USDT", "1M")) is None
    with pytest.raises(ValueError):
        timeframe_to_millis("1M")
    # And one this repo *can* hold is offered, published by the venue and
    # measurable here.
    assert timeframe_to_millis("1s") == 1000
    assert stream_url(_id("binance", "spot", "BTC/USDT", "1s")) is not None
