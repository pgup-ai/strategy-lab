"""The three perp/funding/OI fetch commands, exercised without network or database.

The client and the three storage calls are patched, so what these cover is the
wiring the commands own: the identity the rows are stored under, the window
handed to the client, what reaches storage, and whether a failed fetch is loud.
"""

from __future__ import annotations

import re
from decimal import Decimal

import pandas as pd
import pytest
from typer.testing import CliRunner

from strategy_lab import cli
from strategy_lab.market_data.binance_futures import BinanceFuturesError

runner = CliRunner()

KLINES = pd.DataFrame(
    {
        "open": [63659.5, 63836.9],
        "high": [64058.8, 64100.0],
        "low": [63626.2, 63700.0],
        "close": [63836.9, 63900.1],
        "volume": [20815.435, 19000.5],
    },
    index=pd.DatetimeIndex(
        ["2024-01-01T00:00:00Z", "2024-01-01T04:00:00Z"], name="timestamp"
    ),
)

FUNDING_ROWS = [
    {"exchange": "binance", "market_type": "perp", "symbol": "BTC/USDT",
     "funding_time_ms": 1704067200000, "funding_rate": Decimal("0.00037409"),
     "mark_price": Decimal("42000.5"), "source": "binance-futures"},
]

OI_ROWS = [
    {"exchange": "binance", "market_type": "perp", "symbol": "BTC/USDT",
     "ts_ms": 1785772800000, "open_interest": Decimal("108899.067"),
     "open_interest_usd": Decimal("6933046705.789517"), "source": "binance-futures"},
]


class FakeClient:
    def __init__(self, *, klines=None, funding=None, open_interest=None, error=None):
        self._klines = KLINES if klines is None else klines
        self._funding = FUNDING_ROWS if funding is None else funding
        self._open_interest = OI_ROWS if open_interest is None else open_interest
        self._error = error
        self.calls: list[dict] = []

    def _record(self, name, **kwargs):
        self.calls.append({"method": name, **kwargs})
        if self._error is not None:
            raise self._error

    def fetch_klines(self, symbol, timeframe, *, since=None, until=None):
        self._record("fetch_klines", symbol=symbol, timeframe=timeframe,
                     since=since, until=until)
        return self._klines

    def fetch_funding(self, symbol, *, since=None, until=None):
        self._record("fetch_funding", symbol=symbol, since=since, until=until)
        return self._funding

    def fetch_open_interest(self, symbol, *, period="4h", since=None, until=None):
        self._record("fetch_open_interest", symbol=symbol, period=period,
                     since=since, until=until)
        return self._open_interest


def _message(result) -> str:
    """Typer boxes and wraps error text; flatten it so a phrase can be matched."""
    return " ".join(re.sub(r"[│╭╮╰╯─]", " ", result.output).split())


@pytest.fixture
def wiring(monkeypatch):
    state: dict = {"client": FakeClient(), "candles": [], "funding": [], "open_interest": []}

    monkeypatch.setattr(cli, "_futures_client", lambda **kwargs: state["client"])
    monkeypatch.setattr(
        cli, "upsert_candles", lambda rows: (state["candles"].append(list(rows)), len(rows))[1]
    )
    monkeypatch.setattr(
        cli, "_upsert_funding", lambda rows: (state["funding"].append(list(rows)), len(rows))[1]
    )
    monkeypatch.setattr(
        cli,
        "_upsert_open_interest",
        lambda rows: (state["open_interest"].append(list(rows)), len(rows))[1],
    )
    return state


# --- fetch-perp ------------------------------------------------------------


def test_perp_candles_are_stored_under_the_perp_market_type(wiring):
    """Candle identity is (exchange, market_type, symbol, timeframe).

    Storing perp klines as ``spot`` would silently merge two different markets
    into one series -- the perp trades at a basis to spot, so the joined series
    would have invented jumps at every boundary.
    """
    result = runner.invoke(cli.app, ["fetch-perp", "--symbol", "ETH/USDT", "--timeframe", "4h"])

    assert result.exit_code == 0, result.output
    [stored] = wiring["candles"]
    assert {row["market_type"] for row in stored} == {"perp"}
    assert {row["symbol"] for row in stored} == {"ETH/USDT"}
    assert {row["timeframe"] for row in stored} == {"4h"}
    assert {row["exchange"] for row in stored} == {"binance"}


def test_perp_candles_reach_storage_as_decimal(wiring):
    """The shared ``normalize_candle_frame`` path, not a second one.

    A bare float bound to NUMERIC is cast via "%.15g" and loses its last
    significant digits, which is how ~14,700 equity rows were corrupted here
    before.
    """
    runner.invoke(cli.app, ["fetch-perp", "--symbol", "BTC/USDT", "--timeframe", "4h"])

    [stored] = wiring["candles"]
    for column in ("open", "high", "low", "close", "volume"):
        assert all(isinstance(row[column], Decimal) for row in stored)


def test_fetch_perp_passes_the_requested_window_to_the_client(wiring):
    result = runner.invoke(
        cli.app,
        ["fetch-perp", "--symbol", "BTC/USDT", "--timeframe", "4h",
         "--since", "2019-09-01", "--until", "2020-01-01"],
    )

    assert result.exit_code == 0, result.output
    [call] = wiring["client"].calls
    assert call == {"method": "fetch_klines", "symbol": "BTC/USDT", "timeframe": "4h",
                    "since": "2019-09-01", "until": "2020-01-01"}


def test_fetch_perp_dry_run_writes_nothing(wiring):
    result = runner.invoke(cli.app, ["fetch-perp", "--symbol", "BTC/USDT", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert wiring["candles"] == []
    assert "Dry run" in result.output


def test_fetch_perp_reports_the_venue_error_verbatim(wiring):
    """A venue failure must not be relabelled as "no data".

    Swallowing the error and falling through to the empty-result path still
    exits non-zero, so exit code alone cannot tell the two apart -- and a rate
    limit or an auth failure reported as "returned no rows" is exactly the hole
    that gets mistaken for a market that did not trade.
    """
    wiring["client"]._error = BinanceFuturesError(
        "/fapi/v1/klines failed with HTTP 400: {'code': -1121, 'msg': 'Invalid symbol.'}"
    )

    result = runner.invoke(cli.app, ["fetch-perp", "--symbol", "NOPE/USDT"])

    assert result.exit_code != 0
    assert "-1121" in _message(result)
    assert "Invalid symbol." in _message(result)
    assert "returned no rows" not in _message(result)
    assert wiring["candles"] == []


def test_fetch_perp_exits_non_zero_when_the_venue_returns_nothing(wiring):
    """An empty backfill is a failure to notice, not a quiet success.

    Reporting "stored 0" and exiting 0 is how a hole gets mistaken for a market
    that did not trade.
    """
    wiring["client"]._klines = KLINES.iloc[:0]

    result = runner.invoke(cli.app, ["fetch-perp", "--symbol", "BTC/USDT"])

    assert result.exit_code != 0
    assert "returned no rows" in _message(result)
    assert wiring["candles"] == []


# --- fetch-funding ---------------------------------------------------------


def test_fetch_funding_stores_rows_and_reports_the_count(wiring):
    result = runner.invoke(cli.app, ["fetch-funding", "--symbol", "BTC/USDT"])

    assert result.exit_code == 0, result.output
    [stored] = wiring["funding"]
    assert stored == FUNDING_ROWS
    assert "1" in result.output


def test_fetch_funding_passes_the_window_to_the_client(wiring):
    runner.invoke(
        cli.app, ["fetch-funding", "--symbol", "ETH/USDT", "--since", "2019-11-01"]
    )

    [call] = wiring["client"].calls
    assert call["method"] == "fetch_funding"
    assert call["symbol"] == "ETH/USDT"
    assert call["since"] == "2019-11-01"


def test_fetch_funding_dry_run_writes_nothing(wiring):
    result = runner.invoke(cli.app, ["fetch-funding", "--symbol", "BTC/USDT", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert wiring["funding"] == []


def test_fetch_funding_reports_the_venue_error_verbatim(wiring):
    wiring["client"]._error = BinanceFuturesError("HTTP 400 {'code': -1121}")

    result = runner.invoke(cli.app, ["fetch-funding", "--symbol", "NOPE/USDT"])

    assert result.exit_code != 0
    assert "-1121" in _message(result)
    assert "returned no rows" not in _message(result)
    assert wiring["funding"] == []


# --- fetch-open-interest ---------------------------------------------------


def test_fetch_open_interest_stores_rows(wiring):
    result = runner.invoke(cli.app, ["fetch-open-interest", "--symbol", "BTC/USDT"])

    assert result.exit_code == 0, result.output
    [stored] = wiring["open_interest"]
    assert stored == OI_ROWS


def test_fetch_open_interest_warns_that_only_thirty_days_exist(wiring):
    """The limit has to be visible where someone runs the command.

    Anyone reading a stored OI series without this warning would take a 30-day
    sample for history -- the exact substitution that makes hypothesis C1 look
    testable when it is not.
    """
    result = runner.invoke(cli.app, ["fetch-open-interest", "--symbol", "BTC/USDT"])

    assert result.exit_code == 0, result.output
    assert "30 days" in result.output
    assert "cannot be backfilled" in result.output


def test_fetch_open_interest_passes_the_period(wiring):
    runner.invoke(cli.app, ["fetch-open-interest", "--symbol", "BTC/USDT", "--period", "1h"])

    [call] = wiring["client"].calls
    assert call["method"] == "fetch_open_interest"
    assert call["period"] == "1h"
