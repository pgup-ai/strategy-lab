"""The candle-refresh path shared by ``serve`` and the research browser.

What most of the funding tests below defend is a *symmetry*: a refresh fetches
bars up to the present, and on a perp it has to move the settlements with them.
It did not, so every refresh pushed the candle window further past the last
stored settlement until ``funding_coverage_gaps`` refused the frame -- the tool
breaking the dataset a caller was looking at by the act of looking at it again.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab import server
from strategy_lab.server import build_candles_payload, parse_identity, refresh_candles

_PERP = MarketDataIdentity(
    exchange="binance", market_type="perp", symbol="BTC/USDT", timeframe="4h"
)
_SPOT = MarketDataIdentity(
    exchange="binance", market_type="spot", symbol="BTC/USDT", timeframe="4h"
)


def _frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-03", periods=3, freq="7D", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0, 102.0, 104.0],
            "high": [103.0, 105.0, 107.0],
            "low": [99.0, 101.0, 103.0],
            "close": [102.0, 104.0, 106.0],
            "volume": [1_000.0, float("nan"), 1_200.0],
        },
        index=index,
    )


def test_build_candles_payload_shape() -> None:
    payload = build_candles_payload(_frame())

    assert set(payload) == {"bars"}
    assert len(payload["bars"]) == 3
    first = payload["bars"][0]
    assert first == {
        "time": int(pd.Timestamp("2024-01-03", tz="UTC").timestamp()),
        "open": 100.0,
        "high": 103.0,
        "low": 99.0,
        "close": 102.0,
        "volume": 1_000.0,
    }
    assert payload["bars"][1]["volume"] == 0.0


def test_build_candles_payload_empty_frame() -> None:
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).set_index(
        pd.DatetimeIndex([], name="timestamp")
    )

    assert build_candles_payload(empty) == {"bars": []}


def test_parse_identity_returns_identity_and_after() -> None:
    identity, after = parse_identity(
        {
            "exchange": ["yahoo"],
            "market_type": ["equity"],
            "symbol": ["SPY"],
            "timeframe": ["1w"],
            "after": ["1700000000"],
        }
    )

    assert identity.exchange == "yahoo"
    assert identity.symbol == "SPY"
    assert after == 1_700_000_000


def test_parse_identity_rejects_missing_parameter() -> None:
    with pytest.raises(ValueError, match="symbol"):
        parse_identity(
            {"exchange": ["yahoo"], "market_type": ["equity"], "timeframe": ["1w"]}
        )


# --------------------------------------------------------------------------
# A refresh keeps candles and funding in step.
# --------------------------------------------------------------------------


class _Client:
    """``BinanceFuturesClient`` with the network taken out, recording its call."""

    def __init__(self, calls: list[dict], rows: list[dict]):
        self._calls = calls
        self._rows = rows

    def __call__(self, **kwargs):
        self._calls.append({"init": kwargs})
        return self

    def fetch_funding(self, symbol, *, since=None, until=None):
        self._calls.append({"symbol": symbol, "since": since, "until": until})
        return self._rows


@pytest.fixture
def refresh(monkeypatch):
    """Drive ``refresh_candles`` without a venue or a database.

    Returns a callable taking the stored candles a fetch finds, the last stored
    settlement, and the settlements the venue answers with; hands back the
    payload and the calls the funding client saw.
    """

    def _run(*, fetched=None, last_settlement=None, rows=None, identity=_PERP):
        calls: list[dict] = []
        client = _Client(calls, [] if rows is None else rows)
        stored = _frame() if fetched is None else fetched

        monkeypatch.setattr("strategy_lab.server._fetch_recent", lambda *_: stored)
        monkeypatch.setattr("strategy_lab.server.upsert_candles", len)
        monkeypatch.setattr("strategy_lab.server.load_candles", lambda **_: stored)
        monkeypatch.setattr(
            "strategy_lab.db.funding.funding_span",
            lambda **_: None if last_settlement is None else (last_settlement, last_settlement),
        )
        monkeypatch.setattr("strategy_lab.db.funding.upsert_funding", len)
        monkeypatch.setattr(
            "strategy_lab.market_data.binance_futures.BinanceFuturesClient", client
        )
        return refresh_candles(identity, None), calls

    return _run


def test_a_perp_refresh_stores_settlements_alongside_the_bars(refresh):
    payload, calls = refresh(rows=[{"funding_time_ms": 1}, {"funding_time_ms": 2}])

    assert payload["candles_upserted"] == 3
    assert payload["funding_upserted"] == 2
    assert calls[-1]["symbol"] == "BTC/USDT"


def test_a_refresh_that_moved_only_the_bars_is_visible_as_such(refresh):
    """The drift itself, reported rather than left to be discovered later by a
    strategy that suddenly refuses the dataset."""
    payload, _ = refresh(rows=[])

    assert payload["candles_upserted"] == 3
    assert payload["funding_upserted"] == 0


def test_the_funding_fetch_reaches_back_to_the_last_stored_settlement(refresh):
    """A refresh window narrower than the settlement interval -- five 15m bars
    against 8h funding -- would otherwise step clean over the gap it exists to
    close, and the drift would persist one refresh at a time."""
    lagging = pd.Timestamp("2020-01-01", tz="UTC")

    _, calls = refresh(last_settlement=lagging)

    assert pd.Timestamp(calls[-1]["since"]) == lagging


def test_funding_already_ahead_of_the_bars_still_refetches_the_candle_window(refresh):
    """Settlements are revised on the venue, and the upsert is last-write-wins, so
    the window the bars cover is re-asked for rather than assumed final."""
    ahead = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)

    _, calls = refresh(last_settlement=ahead)

    since = pd.Timestamp(calls[-1]["since"])
    assert since < pd.Timestamp.now(tz="UTC")
    assert since > pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)


def test_a_contract_with_no_stored_funding_tops_up_the_window_rather_than_the_history(
    refresh,
):
    """A refresh button is not a backfill. The whole history is ``fetch-funding``'s
    job, and quietly writing seven years of settlements from a poll would be a
    surprise the caller never asked for."""
    _, calls = refresh(last_settlement=None)

    since = pd.Timestamp(calls[-1]["since"])
    assert since > pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)


def test_nothing_but_a_perp_seeks_settlements_at_all(refresh):
    """``None`` rather than 0: no settlements were sought for a spot pair, which
    is a different claim from a contract that settled none."""
    payload, calls = refresh(identity=_SPOT)

    assert payload["funding_upserted"] is None
    assert calls == []


def test_a_venue_failure_on_the_funding_top_up_is_not_swallowed(monkeypatch):
    """A refresh that reported success while leaving funding behind would put the
    caller back exactly where this change started, one poll later."""

    class _Broken(_Client):
        def fetch_funding(self, symbol, *, since=None, until=None):
            raise RuntimeError("binance said no")

    monkeypatch.setattr("strategy_lab.server._fetch_recent", lambda *_: _frame())
    monkeypatch.setattr("strategy_lab.server.upsert_candles", len)
    monkeypatch.setattr("strategy_lab.server.load_candles", lambda **_: _frame())
    monkeypatch.setattr("strategy_lab.db.funding.funding_span", lambda **_: None)
    monkeypatch.setattr(
        "strategy_lab.market_data.binance_futures.BinanceFuturesClient",
        _Broken([], []),
    )

    with pytest.raises(RuntimeError, match="binance said no"):
        refresh_candles(_PERP, None)


def test_a_perp_on_another_venue_files_no_funding_and_still_refreshes(refresh):
    """Candles route by venue through ccxt and the funding client does not, so a
    top-up here would store Binance settlements under the other venue's name.

    Not fetching is what prevents that. Raising also prevented it and took the
    candle refresh down with it: perp candles reach storage for any ccxt venue
    through ``fetch-crypto --market-type perp``, so a legitimately stored bybit
    perp displayed fine and then 502'd on every refresh click -- in ``serve``'s
    live-update button as much as the browser's.
    """
    payload, _ = refresh(identity=replace(_PERP, exchange="bybit"))

    assert payload["candles_upserted"] >= 0
    assert payload["funding_upserted"] is None, "settlements were sought on a venue with no client"


def test_the_lookback_is_the_timeframe_rather_than_a_fixed_span(refresh):
    """Five bars of whatever is being refreshed: a 1w dataset reaches back weeks
    and a 15m one reaches back an hour, and funding follows the same window."""
    weekly = replace(_PERP, timeframe="1w")

    _, calls = refresh(identity=weekly)

    since = datetime.fromisoformat(calls[-1]["since"])
    assert since < datetime.now(UTC) - timedelta(days=28)


def test_a_funding_outage_leaves_the_candles_where_they_were(monkeypatch):
    """The invariant is that the pair moves together, not that a 502 reports it.

    Candles were upserted before funding was even fetched, so a venue outage on
    the funding call committed the bars and left the settlements behind -- the
    exact drift ``CLAUDE.md`` says a refresh exists to prevent, arrived at
    through the code that prevents it. Both fetches now precede both writes, so
    the failure lands before anything is stored.
    """
    written: list[str] = []
    monkeypatch.setattr(
        server, "_fetch_recent", lambda identity, start: _frame()
    )
    monkeypatch.setattr(
        server, "upsert_candles", lambda frame: written.append("candles") or 1
    )
    monkeypatch.setattr(
        "strategy_lab.db.funding.upsert_funding",
        lambda frame: written.append("funding") or 1,
    )

    def outage(identity, lookback_start):
        raise RuntimeError("funding endpoint is down")

    monkeypatch.setattr(server, "_fetch_funding", outage)

    with pytest.raises(RuntimeError, match="funding endpoint is down"):
        server.refresh_candles(_PERP, after=None)

    assert written == [], f"a failed funding fetch still wrote {written}"
