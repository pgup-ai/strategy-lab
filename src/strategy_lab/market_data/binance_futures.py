"""Binance USD-M futures REST client: perp klines, funding rates, open interest.

ccxt does not expose funding history or open interest cleanly, so this talks to
``https://fapi.binance.com`` directly.

**Measured against the live API on 2026-08-03 -- these are not read from docs:**
``/fapi/v1/klines`` serves from 2019-09-09, ``/fapi/v1/fundingRate`` from
2019-09-10, and ``/futures/data/openInterestHist`` only about 30 days.

The open-interest limit is hard: a ``startTime`` 40 days back returns
``{"code":-1130,"msg":"parameter 'startTime' is invalid."}``. Open interest can
therefore only be *accumulated forward*, never backfilled, which is why
hypothesis C1 in the program charter is marked BLOCKED. ``fetch_open_interest``
raises rather than clamping an out-of-window request, so a 30-day sample can
never be quietly passed off as history.

Klines come back as a float64 frame because they feed ``normalize_candle_frame``
and the shared ``market_candles`` path. Funding and open interest are parsed
straight from the venue's decimal strings into ``Decimal`` and stay that way
into NUMERIC storage -- no float is ever created for them to lose digits in.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pandas as pd

BASE_URL = "https://fapi.binance.com"
SOURCE = "binance-futures"

# This client is hardwired to `fapi.binance.com`; ``exchange`` is only the label
# written into the stored identity, so any other value files Binance data under
# another venue's name. Every caller that takes a venue from outside -- the CLI's
# perp commands, the browser's refresh -- checks against this first.
SUPPORTED_PERP_EXCHANGES = ("binance",)

# Page sizes the endpoints actually accept, measured 2026-08-03.
KLINE_PAGE_LIMIT = 1500
FUNDING_PAGE_LIMIT = 1000
OPEN_INTEREST_PAGE_LIMIT = 500

# Measured, not documented: `startTime` 40 days back returns -1130, 30 days
# works. Open interest accumulates forward from whenever collection starts.
OPEN_INTEREST_HISTORY_DAYS = 30

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")

# Binance klines are positional arrays:
# [openTime, o, h, l, c, volume, closeTime, ...].
_KLINE_OPEN_TIME = 0
_KLINE_OHLCV = slice(1, 6)
_KLINE_CLOSE_TIME = 6


class BinanceFuturesError(RuntimeError):
    """A request failed and retrying it would not help, or retries ran out."""


def to_venue_symbol(symbol: str) -> str:
    """``BTC/USDT`` -> ``BTCUSDT``. Idempotent, so already-venue symbols pass through."""
    return symbol.replace("/", "").upper()


def parse_klines(raw: list[list], *, now_ms: int | None = None) -> pd.DataFrame:
    """Positional kline arrays -> a UTC-indexed float64 OHLCV frame.

    float64 rather than ``Decimal`` on purpose: this frame is handed to
    ``normalize_candle_frame``, which converts back via ``Decimal(str(float(x)))``
    for the NUMERIC bind. That round-trip is exact for every float64 value, and
    it keeps perp candles on the identical hardened path as spot and equity
    candles rather than inventing a second one.

    The still-forming kline is dropped. Without an ``endTime`` the venue returns
    the open candle as the last row, whose OHLCV is provisional and will change;
    stored through the same upsert as finished bars it is indistinguishable from
    one, and a sweep run before it closes trades on values that no longer exist
    afterwards. ``now_ms`` is injectable so this is testable without a clock.
    """
    now = time.time() * 1000 if now_ms is None else now_ms
    raw = [row for row in raw if int(row[_KLINE_CLOSE_TIME]) < now]
    if not raw:
        # Typed like the populated path: a bare DataFrame(columns=...) would be
        # object dtype on a tz-naive index, breaking callers only on empty ranges.
        return pd.DataFrame(
            {name: pd.Series(dtype="float64") for name in OHLCV_COLUMNS},
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )

    frame = pd.DataFrame(
        [row[_KLINE_OHLCV] for row in raw], columns=list(OHLCV_COLUMNS), dtype="float64"
    )
    index = pd.to_datetime(
        [int(row[_KLINE_OPEN_TIME]) for row in raw], unit="ms", utc=True
    ).rename("timestamp")
    return frame.set_index(index)


def parse_funding(
    raw: list[dict], *, exchange: str, market_type: str, symbol: str
) -> list[dict]:
    """Funding payloads -> rows ready for ``db.funding.upsert_funding``.

    ``fundingTime`` is stored exactly as the venue reported it. Settlement
    intervals are per-contract and have changed on live symbols, so nothing here
    rounds to, validates against, or assumes an 8h schedule.
    """
    return [
        {
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "funding_time_ms": int(row["fundingTime"]),
            "funding_rate": _decimal(row["fundingRate"]),
            "mark_price": _decimal(row.get("markPrice")),
            "source": SOURCE,
        }
        for row in raw
    ]


def parse_open_interest(
    raw: list[dict], *, exchange: str, market_type: str, symbol: str
) -> list[dict]:
    """Open-interest snapshots -> rows ready for ``db.funding.upsert_open_interest``."""
    return [
        {
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "ts_ms": int(row["timestamp"]),
            "open_interest": _decimal(row["sumOpenInterest"]),
            "open_interest_usd": _decimal(row.get("sumOpenInterestValue")),
            "source": SOURCE,
        }
        for row in raw
    ]


def _decimal(value) -> Decimal | None:
    """Binance sends decimal *strings*; ``Decimal`` of one is exact.

    ``str()`` first so a payload that ever returns a JSON number still avoids
    ``Decimal(float)``, which would expand the binary value to its full 55-digit
    form and lose the tail to the column's scale.

    An empty string is a *missing* value, not a zero: the oldest BTCUSDT funding
    records carry ``markPrice: ""``, and ``Decimal("")`` raises
    ``InvalidOperation``, which aborted the 2019 backfill on its first page.
    Reading it as 0 would be worse -- a mark price of zero is a real number that
    would quietly corrupt anything computed from it.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Decimal(text)


@dataclass
class BinanceFuturesClient:
    """Paginating REST client. Inject ``session``/``sleep`` to drive it offline."""

    base_url: str = BASE_URL
    exchange: str = "binance"
    market_type: str = "perp"
    session: Any = None
    timeout: float = 30.0
    max_attempts: int = 5
    backoff_seconds: float = 1.0
    sleep: Callable[[float], None] = field(default=time.sleep)

    def __post_init__(self) -> None:
        if self.session is None:
            import requests

            self.session = requests.Session()

    def fetch_klines(
        self,
        symbol: str,
        timeframe: str,
        *,
        since: str | None = None,
        until: str | None = None,
    ) -> pd.DataFrame:
        """Perp OHLCV, paginated forward until the range is covered."""
        raw = self._paginate(
            path="/fapi/v1/klines",
            params={"symbol": to_venue_symbol(symbol), "interval": timeframe},
            limit=KLINE_PAGE_LIMIT,
            cursor_of=lambda row: int(row[_KLINE_OPEN_TIME]),
            since=since,
            until=until,
        )
        return parse_klines(raw)

    def fetch_funding(
        self,
        symbol: str,
        *,
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict]:
        """Full funding history for one contract, paginated forward."""
        raw = self._paginate(
            path="/fapi/v1/fundingRate",
            params={"symbol": to_venue_symbol(symbol)},
            limit=FUNDING_PAGE_LIMIT,
            cursor_of=lambda row: int(row["fundingTime"]),
            since=since,
            until=until,
        )
        return parse_funding(
            raw, exchange=self.exchange, market_type=self.market_type, symbol=symbol
        )

    def fetch_open_interest(
        self,
        symbol: str,
        *,
        period: str = "4h",
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict]:
        """Open-interest snapshots over the venue's whole ~30-day window.

        Paginated *backward*, because the caller has no ``since`` to page forward
        from: without a ``startTime`` the endpoint answers with its most recent
        page and there is no cursor to advance, so one request yields at most
        ``OPEN_INTEREST_PAGE_LIMIT`` (500) snapshots. That is short of the window
        at every period finer than 2h -- 30 days is 720 rows at ``1h``, 2,880 at
        ``15m``, 8,640 at ``5m`` -- and the shortfall is invisible in what gets
        stored, which is the "truncation read as history" failure this module's
        warnings exist to prevent. Walking ``endTime`` back past the oldest row
        seen collects the full window at any period.
        """
        horizon = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=OPEN_INTEREST_HISTORY_DAYS)
        floor_ms = int(horizon.timestamp() * 1000)
        if since is not None:
            since_ms = _timestamp_ms(since)
            if since_ms < floor_ms:
                raise ValueError(
                    f"Binance serves only ~{OPEN_INTEREST_HISTORY_DAYS} days of open-interest "
                    f"history (measured 2026-08-03: a startTime 40 days back returns -1130). "
                    f"{since} is outside that window. Open interest can only be accumulated "
                    f"forward by polling; it cannot be backfilled."
                )
            floor_ms = since_ms

        raw = self._paginate_back(
            path="/futures/data/openInterestHist",
            params={"symbol": to_venue_symbol(symbol), "period": period},
            limit=OPEN_INTEREST_PAGE_LIMIT,
            cursor_of=lambda row: int(row["timestamp"]),
            floor_ms=floor_ms,
            until=until,
        )
        return parse_open_interest(
            raw, exchange=self.exchange, market_type=self.market_type, symbol=symbol
        )

    def _paginate(
        self,
        *,
        path: str,
        params: dict,
        limit: int,
        cursor_of: Callable[[Any], int],
        since: str | None,
        until: str | None,
    ) -> list:
        """Page forward by ``startTime``, one millisecond past the last row seen.

        Advancing by the cursor rather than by a fixed step is what makes this
        safe for funding, whose settlement interval is per-contract and has
        changed on live symbols. A page shorter than ``limit`` means the server
        has no more rows in range, which is the only clean stop condition --
        never a fixed page count.
        """
        since_ms = _timestamp_ms(since) if since else None
        until_ms = _timestamp_ms(until) if until else None

        collected: list = []
        cursor = since_ms
        while True:
            page_params = {**params, "limit": limit}
            if cursor is not None:
                page_params["startTime"] = cursor
            if until_ms is not None:
                page_params["endTime"] = until_ms

            page = self._get(path, page_params)
            if not page:
                break

            for row in page:
                if until_ms is not None and cursor_of(row) > until_ms:
                    continue
                collected.append(row)

            # Without a start time there is no cursor to advance: the venue
            # returned its most recent page and a second request would repeat it.
            if cursor is None or len(page) < limit:
                break

            next_cursor = cursor_of(page[-1]) + 1
            if next_cursor <= cursor:
                break  # non-advancing cursor; stop rather than loop forever
            if until_ms is not None and next_cursor > until_ms:
                break
            cursor = next_cursor

        return collected

    def _paginate_back(
        self,
        *,
        path: str,
        params: dict,
        limit: int,
        cursor_of: Callable[[Any], int],
        floor_ms: int,
        until: str | None,
    ) -> list:
        """Page backward by ``endTime``, one millisecond before the oldest row seen.

        Forward pagination needs a ``startTime`` to advance from; this walks the
        other end of the window instead, which is what a poll with no start time
        has. ``floor_ms`` -- the caller's start, or the venue's ~30-day horizon --
        is the stop condition alongside a short page, and rows below it are
        dropped rather than kept, since the venue's own answer near the boundary
        is not one it will serve consistently.

        Rows are keyed by timestamp, so an overlapping page cannot store a
        snapshot twice, and returned oldest-first to match the forward path.
        """
        collected: dict[int, Any] = {}
        cursor = _timestamp_ms(until) if until else None
        while True:
            page_params = {**params, "limit": limit}
            if cursor is not None:
                page_params["endTime"] = cursor

            page = self._get(path, page_params)
            if not page:
                break

            oldest = min(cursor_of(row) for row in page)
            for row in page:
                if cursor_of(row) >= floor_ms:
                    collected[cursor_of(row)] = row

            if oldest < floor_ms or len(page) < limit:
                break

            next_cursor = oldest - 1
            if cursor is not None and next_cursor >= cursor:
                break  # non-advancing cursor; stop rather than loop forever
            cursor = next_cursor

        return [collected[key] for key in sorted(collected)]

    def _get(self, path: str, params: dict) -> Any:
        """GET with backoff on 429 and 5xx.

        Exhausted retries raise. A backfill that trips a rate limit and returns
        what it has so far leaves a hole indistinguishable from a market that
        did not trade, which would then be read as a flat regime forever.
        4xx other than 429 is a permanent answer -- retrying -1130 just means
        waiting longer for the same rejection.
        """
        url = f"{self.base_url}{path}"
        for attempt in range(1, self.max_attempts + 1):
            response = self.session.get(url, params=params, timeout=self.timeout)
            status = response.status_code
            if status == 200:
                return response.json()

            retryable = status == 429 or status >= 500
            if not retryable or attempt == self.max_attempts:
                raise BinanceFuturesError(
                    f"{path} failed with HTTP {status} after {attempt} attempt(s): "
                    f"{_body(response)}"
                )
            self.sleep(_retry_delay(response, attempt, self.backoff_seconds))

        raise AssertionError("unreachable")  # pragma: no cover


def _retry_delay(response, attempt: int, backoff: float) -> float:
    header = (getattr(response, "headers", None) or {}).get("Retry-After")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    return backoff * (2 ** (attempt - 1))


def _body(response) -> str:
    try:
        return str(response.json())
    except Exception:  # noqa: BLE001 - a non-JSON error body is still worth showing
        return str(getattr(response, "text", ""))[:500]


def _timestamp_ms(value: str) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return int(timestamp.tz_convert("UTC").timestamp() * 1000)


__all__ = [
    "BASE_URL",
    "FUNDING_PAGE_LIMIT",
    "KLINE_PAGE_LIMIT",
    "OPEN_INTEREST_HISTORY_DAYS",
    "OPEN_INTEREST_PAGE_LIMIT",
    "SOURCE",
    "SUPPORTED_PERP_EXCHANGES",
    "BinanceFuturesClient",
    "BinanceFuturesError",
    "parse_funding",
    "parse_klines",
    "parse_open_interest",
    "to_venue_symbol",
]
