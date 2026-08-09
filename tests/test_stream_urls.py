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
