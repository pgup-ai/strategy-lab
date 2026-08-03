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


def refresh_candles(identity: MarketDataIdentity, after: int | None) -> dict:
    bar_ms = timeframe_to_millis(identity.timeframe)
    lookback_start = datetime.now(UTC) - timedelta(
        milliseconds=bar_ms * REFRESH_LOOKBACK_BARS
    )

    fetched = _fetch_recent(identity, lookback_start)
    if not fetched.empty:
        source = "yahoo" if _is_equity(identity) else identity.exchange
        upsert_candles(
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
    return build_candles_payload(df)


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
