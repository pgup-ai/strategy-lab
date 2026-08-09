from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from strategy_lab.db import load_candles, upsert_candles
from strategy_lab.db.candles import normalize_candle_frame
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.timeframes import timeframe_to_millis

REFRESH_LOOKBACK_BARS = 5


def parse_identity(query: dict[str, list[str]]) -> tuple[MarketDataIdentity, int | None]:
    def required(name: str) -> str:
        values = query.get(name, [])
        if not values or not values[0].strip():
            raise ValueError(f"missing query parameter {name!r}")
        return values[0].strip()

    identity = MarketDataIdentity(
        exchange=required("exchange"),
        market_type=required("market_type"),
        symbol=required("symbol"),
        timeframe=required("timeframe"),
    )
    after_values = query.get("after", [])
    after = int(after_values[0]) if after_values else None
    return identity, after


def build_candles_payload(df: pd.DataFrame) -> dict:
    volumes = df["volume"].astype(float).fillna(0.0)
    bars = [
        {
            "time": int(ts.timestamp()),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(vol),
        }
        for (ts, row), vol in zip(df.iterrows(), volumes)
    ]
    return {"bars": bars}


def refresh_candles(
    identity: MarketDataIdentity, after: int | None, since: datetime | None = None
) -> dict:
    """Fetch the newest bars, store them, and hand back the tail from storage.

    On a perp the settlements move with them. Candles reach the present on every
    refresh and funding used to stay where it was, so the candle window grew past
    the last stored settlement until ``funding_coverage_gaps`` refused it -- the
    tool breaking the dataset a caller was looking at by the act of looking at it
    again. Both counts are reported because "3 candles and 0 settlements" is the
    drift itself, and it is invisible in a payload that only carries bars.
    """
    bar_ms = timeframe_to_millis(identity.timeframe)
    # `since` is for a timeframe that has no stored bars at all: the five-bar
    # lookback tops up a series that exists, and would leave a brand new one
    # holding five candles and a warmup error.
    funding_start = datetime.now(UTC) - timedelta(milliseconds=bar_ms * REFRESH_LOOKBACK_BARS)
    lookback_start = since or funding_start

    # Both fetches before either write. Storing candles first and then fetching
    # funding meant a venue outage on the funding call left the bars committed
    # and the settlements behind them -- the exact drift the invariant above
    # exists to prevent, reported as a 502 rather than avoided by it. Failing
    # before anything is written leaves the pair where it was.
    fetched = _fetch_recent(identity, lookback_start)
    # Funding keeps its own five-bar catch-up even when the candles reach back
    # years. It is keyed `(exchange, market_type, symbol)` with no timeframe, so
    # a new *timeframe* adds nothing to it -- and `_fetch_funding` starts at the
    # earlier of its argument and the last stored settlement, so handing it the
    # candle `since` re-pages the whole ~7,700-row history to write duplicates.
    pending_funding = _fetch_funding(identity, funding_start)

    # Settlements before bars, which makes the remaining failure a harmless one.
    # The two upserts cannot share a transaction without threading a connection
    # through `db.candles` and `db.funding`, so one of them can still fail after
    # the other committed -- but only one of the two orders leaves a state the
    # coverage guard refuses. Measured on a 60-bar window: funding running five
    # days *ahead* of the candles yields 0 gaps, because settlements outside the
    # window are simply not counted; funding two days *behind* yields the
    # refusal. So a failed candle write leaves funding ahead and harmless, where
    # a failed funding write used to leave candles ahead and the dataset unusable.
    settlements = None
    if pending_funding is not None:
        # Imported at the call site, as `_fetch_funding` imports its own client:
        # binding it at module scope moves the name out from under the tests that
        # patch it, and the first thing that happens then is a live insert.
        from strategy_lab.db.funding import upsert_funding

        settlements = upsert_funding(pending_funding)

    # The venue always returns the bar in progress -- measured, a 15m fetch at
    # 23:29:48 ends with the bar that opened 23:15 -- and storing it puts a
    # partial candle in `market_candles`, where every consumer treats a row as
    # final. It is then *restated* on the next refresh, which is the in-place
    # rewrite of history the equity caveat warns about, and `build_analysis`
    # meanwhile computes a state and markers for a bar that has not finished.
    # It stays in the payload for `serve`, whose chart draws it as forming on
    # purpose; it just never reaches storage.
    fetched, forming = _split_forming(fetched, bar_ms)

    candles = 0
    if not fetched.empty:
        source = "yahoo" if _is_equity(identity) else identity.exchange
        candles = upsert_candles(
            normalize_candle_frame(
                fetched,
                exchange=identity.exchange,
                market_type=identity.market_type,
                symbol=identity.symbol,
                timeframe=identity.timeframe,
                source=source,
            )
        )

    if after is not None:
        start = datetime.fromtimestamp(after, UTC).isoformat()
    else:
        start = lookback_start.isoformat()
    df = load_candles(
        exchange=identity.exchange,
        market_type=identity.market_type,
        symbol=identity.symbol,
        timeframe=identity.timeframe,
        start=start,
    )
    if forming is not None:
        df = pd.concat([df, forming])
    return {
        **build_candles_payload(df),
        "candles_upserted": candles,
        "funding_upserted": settlements,
    }


def _split_forming(frame: pd.DataFrame, bar_ms: int) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """The bars whose interval has closed, and the one still open if there is one.

    Only the final row can be forming: the venue serves ascending bars up to now,
    so everything before the last one closed when the next began.
    """
    if frame.empty:
        return frame, None
    ends = frame.index[-1] + pd.Timedelta(milliseconds=bar_ms)
    if ends <= pd.Timestamp.now(tz="UTC"):
        return frame, None
    return frame.iloc[:-1], frame.iloc[-1:]


def _fetch_funding(identity: MarketDataIdentity, lookback_start: datetime):
    """Settlements to store for a perp, unwritten. ``None`` on anything else.

    Fetching and writing are separate so the caller can get both networks out of
    the way before it commits either result, and the caller writes settlements
    before bars so that the one failure it cannot make atomic leaves funding
    ahead of the candles rather than behind them -- see ``refresh_candles``.

    ``None`` rather than 0: nothing was sought for a spot pair or an equity, and
    a zero there would read as a contract that settled nothing.

    The fetch starts at the earlier of the candle lookback and the last stored
    settlement. The lookback alone is not enough -- five 15m bars is 75 minutes
    against an 8h settlement interval, so most refreshes would step clean over
    the gap they exist to close -- and starting at the last settlement alone
    would re-fetch nothing when funding is already ahead of the bars.

    Funding is keyed ``(exchange, market_type, symbol)`` with no timeframe, so
    there is exactly one series per contract to top up however many candle
    timeframes are stored beside it. Failures propagate: a refresh that swallowed
    a venue error would report the drift closed while leaving it open, and both
    callers already turn an exception here into a 502.

    An unsupported venue returns ``None`` -- sought nothing, wrote nothing --
    rather than raising. Candles route by venue through ccxt and this client
    does not, so fetching for a non-Binance perp would file Binance settlements
    under that venue's name; not fetching prevents that without taking the
    candle refresh down with it, which is what raising here did.
    """
    if identity.market_type != "perp":
        return None

    from strategy_lab.db.funding import funding_span
    from strategy_lab.market_data.binance_futures import (
        SUPPORTED_PERP_EXCHANGES,
        BinanceFuturesClient,
    )

    if identity.exchange not in SUPPORTED_PERP_EXCHANGES:
        # Nothing was sought, which is the same answer a spot pair gets, so the
        # count is None rather than an error. Refusing here regressed a refresh
        # that used to work: perp candles reach storage for any ccxt venue
        # through `fetch-crypto --market-type perp`, and only `fetch-funding`
        # and `fetch-perp` are Binance-only. A stored OKX perp displayed fine
        # and then 502'd on every refresh click -- in `serve`'s live-update
        # button as much as the browser's. Mis-filing is still prevented, by not
        # fetching rather than by refusing the candles beside it.
        return None

    span = funding_span(
        exchange=identity.exchange,
        market_type=identity.market_type,
        symbol=identity.symbol,
    )
    since = lookback_start if span is None else min(lookback_start, span[1].to_pydatetime())
    client = BinanceFuturesClient(exchange=identity.exchange, market_type=identity.market_type)
    return client.fetch_funding(identity.symbol, since=since.isoformat())


def _is_equity(identity: MarketDataIdentity) -> bool:
    return identity.exchange == "yahoo" or identity.market_type == "equity"


def _fetch_recent(identity: MarketDataIdentity, start: datetime) -> pd.DataFrame:
    if _is_equity(identity):
        from strategy_lab.market_data.yahoo import YahooFinanceClient

        return YahooFinanceClient().fetch_ohlcv(
            identity.symbol,
            identity.timeframe,
            start=start.strftime("%Y-%m-%d"),
        )

    from strategy_lab.market_data.binance import CryptoOhlcvClient

    client = CryptoOhlcvClient(
        exchange_id=identity.exchange, market_type=identity.market_type
    )
    return client.fetch_ohlcv(identity.symbol, identity.timeframe, since=start.isoformat())


class ReportRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/candles":
            self._handle_candles(parse_qs(parsed.query))
            return
        super().do_GET()

    def _handle_candles(self, query: dict[str, list[str]]) -> None:
        try:
            identity, after = parse_identity(query)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        try:
            payload = refresh_candles(identity, after)
        except Exception as exc:  # network/db failures become a JSON error
            self._send_json({"error": f"refresh failed: {exc}"}, status=502)
            return
        self._send_json(payload)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def run_server(*, report_root: Path, host: str, port: int) -> None:
    handler = partial(ReportRequestHandler, directory=str(report_root))
    with ThreadingHTTPServer((host, port), handler) as httpd:
        httpd.serve_forever()
