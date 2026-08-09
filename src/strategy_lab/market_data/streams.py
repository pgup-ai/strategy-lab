"""Where a live candle stream lives for one identity, or nothing.

The browser draws a forming bar from a venue's websocket. *Which* URL that is
belongs here rather than in the page script, for the same reason
``CONTRACT_PRIMITIVES`` does: it is venue knowledge, it is wrong in ways a chart
still renders, and Python is where this repo can test it.

**Absence is the common answer and it is not an error.** Yahoo publishes no
stream, ``1wk`` is a Yahoo timeframe, and a venue this repo has never fetched
from has no business being subscribed to. Every one of those returns ``None``,
and the page simply shows no live control.

**Only Binance, and only because the candles are Binance.** A stream from a
second venue drawn onto a series stored from this one would be two markets'
prices in one line -- close enough to look right and wrong in exactly the way
nobody checks. A venue earns a stream here once its candles are stored.
"""

from __future__ import annotations

from strategy_lab.market_data.base import MarketDataIdentity

# Spot and USD-M futures are separate services, not separate paths on one host.
_HOSTS = {
    ("binance", "spot"): "wss://stream.binance.com:9443/ws",
    ("binance", "perp"): "wss://fstream.binance.com/ws",
}

# Binance's kline intervals *that this repo can also hold as a dataset*, which
# is not all of them. `1M` is published and excluded: a month is not a fixed
# width, `timeframe_to_millis` raises on it, and every bar calculation here is
# width arithmetic. `1wk` is excluded for the opposite reason -- it is the Yahoo
# spelling of a week, and a timeframe is an identity here rather than a
# duration, so `1w` and `1wk` are different datasets.
_INTERVALS = frozenset(
    {"1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"}
)


def stream_url(identity: MarketDataIdentity) -> str | None:
    """The kline websocket for this identity, or ``None`` if there is not one."""
    host = _HOSTS.get((identity.exchange, identity.market_type))
    if host is None or identity.timeframe not in _INTERVALS:
        return None
    # `BTC/USDT:USDT` is ccxt's spelling of a perp, and the settle suffix is not
    # part of the venue's stream name. Left in, the URL still connects -- it just
    # subscribes to nothing, which is the silent half of a wrong stream.
    base = identity.symbol.split(":", 1)[0]
    symbol = base.replace("/", "").replace("-", "").lower()
    return f"{host}/{symbol}@kline_{identity.timeframe}"


__all__ = ["stream_url"]
