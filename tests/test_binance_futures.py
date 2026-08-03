"""The Binance USD-M futures REST client, exercised entirely offline.

Every test parses a recorded payload or drives a fake session, so the suite is
deterministic and runs without connectivity. The pagination and retry tests are
the ones that matter: a backfill that stops early or swallows a rate limit
leaves a hole that is indistinguishable from a market that did not trade.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from strategy_lab.market_data.binance_futures import (
    FUNDING_PAGE_LIMIT,
    KLINE_PAGE_LIMIT,
    OPEN_INTEREST_HISTORY_DAYS,
    OPEN_INTEREST_PAGE_LIMIT,
    BinanceFuturesClient,
    BinanceFuturesError,
    parse_funding,
    parse_klines,
    parse_open_interest,
)

RAW_KLINE = [[1785772800000, "63659.50", "64058.80", "63626.20", "63836.90", "20815.435",
              1785787199999, "1329198752.90", 498729, "10915.422", "697118356.54", "0"]]
RAW_FUNDING = [{"symbol": "BTCUSDT", "fundingTime": 1785744000000,
                "fundingRate": "0.00006364", "markPrice": "62591.00000000"}]
RAW_OI = [{"symbol": "BTCUSDT", "sumOpenInterest": "108899.06700000",
           "sumOpenInterestValue": "6933046705.78951700", "timestamp": 1785772800000}]

# Recorded from the live endpoint on 2026-08-03, startTime=1567296000000: every
# one of the earliest 1000 BTCUSDT records carries markPrice="" rather than a
# number, and older rows also carry a rateType field.
RAW_FUNDING_EARLY = [{"symbol": "BTCUSDT", "fundingTime": 1568102400000,
                      "fundingRate": "0.00010000", "markPrice": "", "rateType": "Regular"}]

IDENTITY = dict(exchange="binance", market_type="perp", symbol="BTC/USDT")
FOUR_HOURS_MS = 4 * 60 * 60 * 1000


class FakeResponse:
    def __init__(self, payload, status_code: int = 200, headers: dict | None = None):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = str(payload)

    def json(self):
        return self.payload


class FakeSession:
    """Serves canned responses in order, recording every request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {})})
        if not self.responses:
            return FakeResponse([])
        return self.responses.pop(0)


def _client(*responses, **kwargs):
    slept: list[float] = []
    session = FakeSession(*responses)
    client = BinanceFuturesClient(
        session=session, sleep=slept.append, backoff_seconds=1.0, **kwargs
    )
    return client, session, slept


def _kline_page(count: int, start_ms: int = 1785772800000) -> list[list]:
    return [
        [start_ms + i * FOUR_HOURS_MS, "1.0", "2.0", "0.5", "1.5", "10.0",
         start_ms + (i + 1) * FOUR_HOURS_MS - 1, "15.0", 10, "5.0", "7.5", "0"]
        for i in range(count)
    ]


# --- parsers ---------------------------------------------------------------


def test_klines_parse_to_a_utc_indexed_ohlcv_frame():
    df = parse_klines(RAW_KLINE)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index.tz) == "UTC"
    assert df["close"].iloc[0] == pytest.approx(63836.90)


def test_the_open_interest_history_limit_is_declared():
    """Binance serves only ~30 days of OI history; C1 is not backtestable without it."""
    assert OPEN_INTEREST_HISTORY_DAYS <= 30


def test_parsers_emit_decimal_so_no_float_ever_enters_the_storage_path():
    """``Decimal`` is the point: these values go straight into NUMERIC(38,18).

    A float here would be bound as float8 and mangled by Postgres' "%.15g"
    implicit cast, and ``==`` against a Decimal would still pass, so the type
    itself has to be asserted rather than the value.
    """
    funding = parse_funding(RAW_FUNDING, **IDENTITY)
    interest = parse_open_interest(RAW_OI, **IDENTITY)

    assert isinstance(funding[0]["funding_rate"], Decimal)
    assert isinstance(funding[0]["mark_price"], Decimal)
    assert isinstance(interest[0]["open_interest"], Decimal)
    assert isinstance(interest[0]["open_interest_usd"], Decimal)


def test_an_absent_mark_price_is_null_rather_than_a_crash():
    """Binance sends markPrice="" on the oldest funding records, not a number.

    Found by running the real 2019 backfill: ``Decimal("")`` raises
    ``InvalidOperation``, so this aborted the whole history fetch at the first
    page. ``mark_price`` is nullable precisely because the venue does not always
    have one; the rate itself is what must survive.
    """
    rows = parse_funding(RAW_FUNDING_EARLY, **IDENTITY)

    assert rows[0]["mark_price"] is None
    assert rows[0]["funding_rate"] == Decimal("0.00010000")
    assert rows[0]["funding_time_ms"] == 1568102400000


def test_parsed_rows_carry_exactly_the_storage_columns():
    """Nothing joins the parser to the table, so the key set is that seam.

    ``upsert_funding`` binds whatever keys it is handed, and the storage tests
    build their own rows rather than parsing one, so a key the parser emits that
    ``funding_rates`` has no column for would crash only on a real fetch. The
    venue's own extra fields (older rows carry ``rateType``) must stay out.
    """
    rows = parse_funding(RAW_FUNDING_EARLY, **IDENTITY)
    assert set(rows[0]) == {
        "exchange", "market_type", "symbol",
        "funding_time_ms", "funding_rate", "mark_price", "source",
    }


def test_an_empty_kline_payload_keeps_the_populated_shape():
    df = parse_klines([])
    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index.tz) == "UTC"
    assert df["close"].dtype == "float64"


# --- pagination ------------------------------------------------------------


def test_klines_paginate_forward_past_the_last_returned_bar():
    """Never assume a fixed page count -- page until a short page ends it.

    Broken pagination fetched 1,500 rows where 15,128 existed, which reads as a
    market with a short history rather than as a bug.
    """
    first = _kline_page(KLINE_PAGE_LIMIT)
    second = _kline_page(3, start_ms=1785772800000 + KLINE_PAGE_LIMIT * FOUR_HOURS_MS)
    client, session, _ = _client(FakeResponse(first), FakeResponse(second))

    df = client.fetch_klines("BTC/USDT", "4h", since="2026-08-01")

    assert len(df) == KLINE_PAGE_LIMIT + 3
    assert len(session.calls) == 2
    assert session.calls[0]["params"]["symbol"] == "BTCUSDT"
    assert session.calls[0]["params"]["interval"] == "4h"
    assert session.calls[0]["params"]["limit"] == KLINE_PAGE_LIMIT
    # Advance past the last bar, not by a page-sized guess.
    assert session.calls[1]["params"]["startTime"] == first[-1][0] + 1


def test_klines_stop_at_until():
    start = 1785772800000
    client, session, _ = _client(FakeResponse(_kline_page(KLINE_PAGE_LIMIT, start_ms=start)))
    until = pd.Timestamp(start + 2 * FOUR_HOURS_MS, unit="ms", tz="UTC")

    df = client.fetch_klines("BTC/USDT", "4h", since="2026-08-01", until=str(until))

    assert len(df) == 3
    assert df.index.max() == until


def test_funding_pagination_does_not_assume_an_eight_hour_interval():
    """Settlement intervals are per-contract and have changed on live symbols.

    Advancing by a hardcoded 8h would skip rows on a 4h contract and re-request
    the same page forever on a 24h one, so the cursor is the last timestamp
    returned plus one millisecond.
    """
    # Spacing cycles 4h / 8h / 24h, so no single interval describes the page.
    gaps = (4 * 3_600_000, 8 * 3_600_000, 24 * 3_600_000)
    times: list[int] = [1785744000000]
    while len(times) < FUNDING_PAGE_LIMIT:
        times.append(times[-1] + gaps[len(times) % len(gaps)])
    page = [
        {"symbol": "BTCUSDT", "fundingTime": ts, "fundingRate": "0.0001", "markPrice": "60000.0"}
        for ts in times
    ]
    client, session, _ = _client(FakeResponse(page), FakeResponse([]))

    rows = client.fetch_funding("BTC/USDT", since="2026-08-01")

    assert len(rows) == FUNDING_PAGE_LIMIT
    assert session.calls[1]["params"]["startTime"] == page[-1]["fundingTime"] + 1
    assert rows[0]["symbol"] == "BTC/USDT"


def test_funding_requests_the_documented_page_size():
    client, session, _ = _client(FakeResponse(RAW_FUNDING))

    client.fetch_funding("BTC/USDT", since="2026-08-01")

    assert session.calls[0]["params"]["limit"] == FUNDING_PAGE_LIMIT
    assert session.calls[0]["params"]["symbol"] == "BTCUSDT"
    assert "fundingRate" in session.calls[0]["url"]


def test_open_interest_requests_the_period_and_page_size():
    client, session, _ = _client(FakeResponse(RAW_OI))

    rows = client.fetch_open_interest("BTC/USDT", period="4h")

    assert rows[0]["open_interest"] == Decimal("108899.067")
    assert session.calls[0]["params"]["period"] == "4h"
    assert session.calls[0]["params"]["limit"] == OPEN_INTEREST_PAGE_LIMIT
    assert "openInterestHist" in session.calls[0]["url"]


def test_open_interest_refuses_a_start_outside_the_measured_window():
    """Measured 2026-08-03: a startTime 40 days back returns -1130.

    Silently clamping would make a 30-day sample look like a backfill, which is
    exactly the substitution the charter marks hypothesis C1 BLOCKED over.
    """
    client, session, _ = _client(FakeResponse(RAW_OI))
    long_ago = str(pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=90))

    with pytest.raises(ValueError, match="30 days"):
        client.fetch_open_interest("BTC/USDT", period="4h", since=long_ago)

    assert session.calls == []


# --- retries ---------------------------------------------------------------


def test_a_rate_limit_is_retried_rather_than_truncating_the_series():
    client, session, slept = _client(
        FakeResponse({"code": -1003}, status_code=429),
        FakeResponse(_kline_page(2)),
    )

    df = client.fetch_klines("BTC/USDT", "4h", since="2026-08-01")

    assert len(df) == 2
    assert len(session.calls) == 2
    assert slept == [1.0]


def test_a_server_error_is_retried():
    client, session, slept = _client(
        FakeResponse("upstream boom", status_code=502),
        FakeResponse(_kline_page(1)),
    )

    assert len(client.fetch_klines("BTC/USDT", "4h", since="2026-08-01")) == 1
    assert len(session.calls) == 2


def test_retry_after_is_respected_when_the_venue_sends_one():
    client, _, slept = _client(
        FakeResponse({"code": -1003}, status_code=429, headers={"Retry-After": "7"}),
        FakeResponse(_kline_page(1)),
    )

    client.fetch_klines("BTC/USDT", "4h", since="2026-08-01")

    assert slept == [7.0]


def test_backoff_grows_between_attempts():
    client, _, slept = _client(
        FakeResponse({}, status_code=429),
        FakeResponse({}, status_code=429),
        FakeResponse({}, status_code=429),
        FakeResponse(_kline_page(1)),
    )

    client.fetch_klines("BTC/USDT", "4h", since="2026-08-01")

    assert slept == [1.0, 2.0, 4.0]


def test_a_persistent_rate_limit_raises_instead_of_returning_a_short_series():
    """Giving up quietly leaves a hole that looks like missing market data."""
    client, _, _ = _client(*[FakeResponse({}, status_code=429) for _ in range(4)], max_attempts=4)

    with pytest.raises(BinanceFuturesError, match="429"):
        client.fetch_klines("BTC/USDT", "4h", since="2026-08-01")


def test_a_client_error_is_not_retried():
    """-1130 on an out-of-window startTime is a permanent answer, not congestion."""
    client, session, _ = _client(
        FakeResponse({"code": -1130, "msg": "parameter 'startTime' is invalid."}, status_code=400),
    )

    with pytest.raises(BinanceFuturesError, match="-1130"):
        client.fetch_funding("BTC/USDT", since="2026-08-01")

    assert len(session.calls) == 1
