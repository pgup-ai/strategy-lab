"""The browser's endpoints, and the validation that is the reason they exist.

``server.py`` hand-parses its query string and a wrong value there becomes a
default nobody chose. This suite's centre of gravity is therefore the 422s: an
unknown ``exit_mode``, an unparseable timeframe, a misspelled parameter name and
an out-of-range rate all have to be refused **by name**, because M20 was a
setting that went missing quietly and moved a published figure.

Postgres is never reached: ``list_candle_sets`` and the analysis payload are
patched, so what is under test is the surface rather than the database.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from strategy_lab.api import app as app_module
from strategy_lab.api.analysis import DatasetUnavailable
from strategy_lab.backtests.funding_frame import FundingUnavailable
from tests.conftest import synthetic_ohlcv

_PERP = {
    "exchange": "binance",
    "market_type": "perp",
    "symbol": "BTC/USDT",
    "timeframe": "4h",
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        "strategy_lab.api.analysis.load_candles",
        lambda **kwargs: synthetic_ohlcv(n=600, freq="4h"),
    )
    monkeypatch.setattr(
        "strategy_lab.backtests.funding_frame.funding_rates",
        lambda identity, frame, **_: None,
    )
    return TestClient(app_module.create_app())


def _analysis(client, **overrides):
    params = {**_PERP, "strategy": "donchian", **overrides}
    return client.get("/api/analysis", params=params)


# --------------------------------------------------------------------------
# Inbound validation -- the reason the dependency was adopted.
# --------------------------------------------------------------------------


def test_an_unknown_exit_mode_is_refused_by_name(client):
    response = _analysis(client, exit_mode="hope")

    assert response.status_code == 422
    assert "exit_mode" in _fields(response)


def test_an_unparseable_timeframe_is_refused_by_name(client):
    response = _analysis(client, timeframe="4hh")

    assert response.status_code == 422
    assert "timeframe" in _fields(response)


def test_an_unknown_market_type_is_refused_rather_than_queried(client):
    """Storage keys candles on exactly three, so a fourth returns an empty frame
    that reads as "this strategy did nothing" rather than as a typo."""
    response = _analysis(client, market_type="futures")

    assert response.status_code == 422
    assert "market_type" in _fields(response)


def test_an_unregistered_strategy_is_refused_by_name(client):
    response = _analysis(client, strategy="no_such_strategy")

    assert response.status_code == 422
    assert "strategy" in _fields(response)


def test_a_misspelled_parameter_is_refused_rather_than_ignored(client):
    """The M20 shape in miniature: ``--funding`` typed as ``fundng`` would
    otherwise run with funding on while the caller believed it off."""
    response = _analysis(client, fundng="false")

    assert response.status_code == 422
    assert "fundng" in _fields(response)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("position_pct", 1.5),
        ("position_pct", 0.0),
        ("fees", -0.1),
        ("slippage", 2.0),
        ("cash", 0.0),
    ],
)
def test_a_cost_setting_outside_its_range_is_refused_by_name(client, field, value):
    response = _analysis(client, **{field: value})

    assert response.status_code == 422
    assert field in _fields(response)


def test_a_start_that_is_not_a_timestamp_is_refused_by_name(client):
    response = _analysis(client, start="last tuesday")

    assert response.status_code == 422
    assert "start" in _fields(response)


def test_an_exit_mode_on_the_continuous_contract_is_refused_at_the_boundary(client):
    """Cross-field, so the model checks the registry rather than the spelling."""
    response = _analysis(client, strategy="state_machine_v2", exit_mode="trend_failure")

    assert response.status_code == 422
    assert "no exit mode" in response.text


def _board(client, **overrides):
    return client.get("/api/board", params={"strategies": "donchian", **overrides})


def test_an_unregistered_strategy_on_the_board_is_refused_by_name(client):
    """At the boundary, before a single row is streamed.

    A stream that has already started cannot be turned into a 422, so a
    misspelled name would arrive as a row-shaped hole in the middle of a board
    that otherwise looks complete.
    """
    response = _board(client, strategies="donchian,no_such_strategy")

    assert response.status_code == 422
    assert "strategies" in _fields(response)


def test_the_same_strategy_twice_on_the_board_is_refused_rather_than_deduplicated(client):
    """Two identical tiles per instrument, the second recomputed to say the same."""
    response = _board(client, strategies="donchian,donchian")

    assert response.status_code == 422
    assert "duplicate strategy" in response.text


def test_a_misspelled_board_parameter_is_refused_rather_than_ignored(client):
    response = _board(client, market="perp")

    assert response.status_code == 422
    assert "market" in _fields(response)


def test_an_exit_mode_is_refused_when_any_board_strategy_has_no_exit_mode(client):
    """One mode covers every row, so one continuous strategy refuses it.

    Applying it to the boolean rows and dropping it for the continuous ones
    would label half a board with a setting that changed nothing there.
    """
    response = _board(client, strategies="donchian,state_machine_v2", exit_mode="trend_failure")

    assert response.status_code == 422
    assert "no exit mode" in response.text


def test_a_sparkline_tail_outside_its_range_is_refused_by_name(client):
    response = _board(client, spark_bars=5_000)

    assert response.status_code == 422
    assert "spark_bars" in _fields(response)


def test_a_negative_refresh_cursor_is_refused_by_name(client):
    response = client.post("/api/refresh", params={**_PERP, "after": -1})

    assert response.status_code == 422
    assert "after" in _fields(response)


def _fields(response) -> set[str]:
    return {str(part) for error in response.json()["detail"] for part in error["loc"]}


# --------------------------------------------------------------------------
# The payloads.
# --------------------------------------------------------------------------


def test_the_analysis_response_carries_bars_markers_and_provenance(client):
    response = _analysis(client, exit_mode="continuation_failure")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["bars"]) == 600
    assert payload["markers"], "donchian trades on this frame"
    assert {marker["kind"] for marker in payload["markers"]} <= {"entry", "exit"}
    assert payload["provenance"]["exit_mode"] == "continuation_failure"
    assert payload["provenance"]["strategy"] == "donchian"
    assert payload["provenance"]["contract"] == "signal_set"
    assert payload["provenance"]["cost_model"]["fee"] == 0.0005


def test_every_provenance_field_survives_the_response_model(client):
    """A response model silently drops what it has not declared, so a provenance
    field added to the payload and forgotten here would vanish from the wire.
    ``extra="forbid"`` is what turns that into an error; this pins the list."""
    from dataclasses import fields

    from strategy_lab.api.analysis import Provenance

    response = _analysis(client)

    assert set(response.json()["provenance"]) == {f.name for f in fields(Provenance)}


def test_the_continuous_contract_answers_with_a_target_and_no_markers(monkeypatch):
    """Past the warmup, because ``state_machine_v2`` declares 2,192 bars.

    This ran on the 600-bar fixture and passed, which it could only do while the
    browser returned the strategy's raw target. The engine refuses a frame
    shorter than a warmup rather than drawing a flat line over it, and now so
    does this path -- see the test below.
    """
    bars = 2_600
    monkeypatch.setattr(
        "strategy_lab.api.analysis.load_candles",
        lambda **kwargs: synthetic_ohlcv(n=bars, freq="4h"),
    )
    monkeypatch.setattr(
        "strategy_lab.backtests.funding_frame.funding_rates",
        lambda identity, frame, **_: None,
    )
    client = TestClient(app_module.create_app())

    response = _analysis(client, strategy="state_machine_v2")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["markers"] == []
    assert len(payload["target"]) == bars
    assert payload["provenance"]["contract"] == "target_exposure"
    assert payload["provenance"]["exit_mode"] is None


def test_a_frame_shorter_than_the_warmup_is_refused_on_both_contracts(client):
    """The boolean path already refused; the continuous one drew a flat line.

    A target that is 0.0 on every bar because the frame never reached the
    strategy's warmup looks exactly like a strategy that chose to hold nothing.
    Both contracts now say which it is, in the engine's own words.
    """
    for strategy in ("state_machine_v1", "state_machine_v2"):
        response = _analysis(client, strategy=strategy)
        assert response.status_code == 400, f"{strategy}: {response.text}"
        assert "warmup bars but the frame has" in response.json()["detail"]


def test_the_strategy_list_labels_both_registries_by_contract(client):
    from strategy_lab.strategies.exposure_registry import list_exposure_strategies
    from strategy_lab.strategies.registry import list_strategies

    listed = client.get("/api/strategies").json()
    by_name = {entry["name"]: entry for entry in listed}

    assert set(by_name) == set(list_strategies()) | set(list_exposure_strategies())
    assert by_name["state_machine_v2"]["contract"] == "target_exposure"
    assert by_name["donchian"]["contract"] == "signal_set"


def test_the_dataset_list_is_storages_own_four_part_identity(client, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "list_candle_sets",
        lambda: pd.DataFrame(
            [
                {
                    "exchange": "binance",
                    "market_type": "perp",
                    "symbol": "BTC/USDT",
                    "timeframe": "4h",
                    "candles": 15_128,
                    "first_timestamp": pd.Timestamp("2019-09-08", tz="UTC"),
                    "last_timestamp": pd.Timestamp("2026-08-03", tz="UTC"),
                }
            ]
        ),
    )

    [row] = client.get("/api/datasets").json()

    assert row["symbol"] == "BTC/USDT"
    assert row["timeframe"] == "4h"
    assert row["candles"] == 15_128


# --------------------------------------------------------------------------
# What the browser refuses to do.
# --------------------------------------------------------------------------


def test_a_missing_dataset_is_a_404_rather_than_an_empty_chart(client, monkeypatch):
    def _absent(**kwargs):
        raise DatasetUnavailable("No candles stored for binance/perp/DOGE/USDT/4h")

    monkeypatch.setattr("strategy_lab.api.analysis.load_candles", _absent)

    response = _analysis(client, symbol="DOGE/USDT")

    assert response.status_code == 404
    assert "No candles stored" in response.json()["detail"]


def test_unfetched_funding_is_a_409_naming_the_fetch_command(client, monkeypatch):
    """Only a strategy that reads a funding-derived feature gets this far, and
    for that strategy a neutral fallback is a different strategy."""

    def _missing(identity, frame, **kwargs):
        raise FundingUnavailable("No stored funding ... strategy-lab fetch-funding")

    monkeypatch.setattr("strategy_lab.backtests.funding_frame.funding_rates", _missing)

    response = _analysis(client, strategy="state_machine_v1")

    assert response.status_code == 409
    assert "fetch-funding" in response.json()["detail"]


def test_the_refresh_endpoint_reuses_the_existing_fetch_and_upsert_path(
    client, monkeypatch
):
    """Called, not copied: a second upsert rule is a second way for a stored
    candle to differ from the one the rest of the lab wrote."""
    seen = {}

    def _refresh(identity, after, since):
        seen["identity"] = identity
        seen["after"] = after
        return {"bars": [{"time": 1, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
                          "volume": 3.0}],
                "candles_upserted": 3, "funding_upserted": 0}

    monkeypatch.setattr("strategy_lab.server.refresh_candles", _refresh)

    response = client.post("/api/refresh", params={**_PERP, "after": 1_700_000_000})

    assert response.status_code == 200, response.text
    assert seen["identity"].symbol == "BTC/USDT"
    assert seen["after"] == 1_700_000_000
    assert response.json()["bars"][0]["close"] == 1.5


def test_what_a_refresh_wrote_reaches_the_caller_and_not_only_the_database(
    client, monkeypatch
):
    """Three candles and no settlements is the drift that made the browser refuse
    the dataset it had just been showing. It is a fact about the write, so it
    belongs in the response rather than in a server log nobody is reading."""
    monkeypatch.setattr(
        "strategy_lab.server.refresh_candles",
        lambda identity, after, since: {
            "bars": [], "candles_upserted": 3, "funding_upserted": 0
        },
    )

    body = client.post("/api/refresh", params=_PERP).json()

    assert body["candles_upserted"] == 3
    assert body["funding_upserted"] == 0


def test_a_refresh_that_sought_no_settlements_says_so_rather_than_reporting_none(
    client, monkeypatch
):
    """``null`` is "not a perp, nothing was sought"; ``0`` is "asked, and the
    contract settled nothing". Collapsing them would make an equity refresh
    indistinguishable from the drift above."""
    monkeypatch.setattr(
        "strategy_lab.server.refresh_candles",
        lambda identity, after, since: {
            "bars": [], "candles_upserted": 1, "funding_upserted": None
        },
    )

    body = client.post("/api/refresh", params={**_PERP, "market_type": "spot"}).json()

    assert body["funding_upserted"] is None


def test_a_refresh_that_reported_no_counts_is_an_error_rather_than_a_default(
    client, monkeypatch
):
    """Same reason ``_Strict`` forbids extras. A default here would let a refresh
    that stopped reporting what it wrote serialize as "0 candles, no funding
    sought" -- a plausible answer nobody computed, which is the failure the whole
    provenance habit exists to prevent."""
    monkeypatch.setattr(
        "strategy_lab.server.refresh_candles", lambda identity, after, since: {"bars": []}
    )

    with pytest.raises(Exception, match="candles_upserted"):
        client.post("/api/refresh", params=_PERP)


def test_a_venue_failure_on_refresh_is_a_502_not_a_traceback(client, monkeypatch):
    def _boom(identity, after, since):
        raise RuntimeError("binance said no")

    monkeypatch.setattr("strategy_lab.server.refresh_candles", _boom)

    response = client.post("/api/refresh", params=_PERP)

    assert response.status_code == 502
    assert "binance said no" in response.json()["detail"]


def test_refreshing_is_a_post_because_it_writes(client):
    """Every other endpoint is a GET that cannot change stored data, and the one
    that can should not be reachable by following a link."""
    assert client.get("/api/refresh", params=_PERP).status_code == 405


@pytest.fixture
def served(monkeypatch):
    """``uvicorn.run`` recorded rather than run.

    Without this the guard below cannot be tested at all: a host that gets past
    it starts a real server and blocks the suite forever, so a broken guard
    would hang rather than fail.
    """
    calls = []
    monkeypatch.setattr(
        "uvicorn.run", lambda app, **kwargs: calls.append(kwargs), raising=True
    )
    return calls


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", ""])
def test_binding_off_the_loopback_interface_is_refused(host, served):
    """Unauthenticated, and it can make the process fetch from an exchange."""
    with pytest.raises(ValueError, match="refusing to bind"):
        app_module.run_api(host=host)

    assert served == [], "nothing may be bound once the host has been refused"


def test_the_loopback_host_and_the_chosen_port_reach_the_server(served):
    app_module.run_api(host="127.0.0.1", port=9999)

    assert served == [{"host": "127.0.0.1", "port": 9999}]


def test_no_handler_blocks_the_event_loop():
    """Every handler here blocks, so none of them may be ``async``.

    FastAPI runs an ``async def`` handler on the event loop and a plain ``def``
    in a threadpool. These read Postgres, inline a 191 KB asset from disk, run a
    whole-history ``from_signals``, and call the venue synchronously -- measured,
    a 1 ms ``/api/strategies`` took **3,640 ms** when it landed inside an
    in-flight analysis, and 35 ms once the handlers were synchronous. The page
    polls, so the stall lands on the thing being drawn.
    """
    import inspect

    from fastapi.routing import APIRoute

    from strategy_lab.api.app import create_app

    blocking = [
        route.endpoint
        for route in create_app().routes
        if isinstance(route, APIRoute) or route.path == "/"
    ]
    assert blocking, "no routes found; the check would pass vacuously"
    offenders = [fn.__name__ for fn in blocking if inspect.iscoroutinefunction(fn)]
    assert offenders == [], f"async handlers run on the event loop: {offenders}"


def test_every_endpoint_that_slices_a_frame_reads_a_bound_the_same_way():
    """The bound below is only useful if it is the *same* bound everywhere.

    `/api/state` shipped with bare `start`/`end` strings and none of the
    validation beside them, so the page sent one `<input type="date">` value to
    both endpoints and got two different frames — the state view ending the
    evening before the instrument view, on a path whose entire claim is that it
    cannot disagree with `build_analysis`. Asserted against the base rather than
    against a list of models, so a fourth endpoint inherits or is caught here.
    """
    from strategy_lab.api.models import AnalysisQuery, BoundedQuery, StateQuery

    identity = dict(exchange="binance", market_type="perp", symbol="BTC/USDT",
                    timeframe="4h", strategy="state_machine_v1")
    analysis = AnalysisQuery(**identity, end="2023-10-31", start="2023-01-01")
    state = StateQuery(**identity, end="2023-10-31", start="2023-01-01")

    assert (state.start, state.end) == (analysis.start, analysis.end)
    assert state.end == "2023-10-31 23:59:59"
    assert issubclass(StateQuery, BoundedQuery) and issubclass(AnalysisQuery, BoundedQuery)


def test_a_date_only_end_covers_the_whole_day_it_names(client):
    """``<input type="date">`` sends ``YYYY-MM-DD`` and storage filters ``<=``.

    Midnight is the first instant of the named day, so a 4h frame kept one bar
    of it and dropped the other five, while ``start`` included the whole of its
    first day. A user picked a day and the chart ended the evening before, which
    reads as stale data rather than as a boundary.
    """
    from strategy_lab.api.models import AnalysisQuery

    named = AnalysisQuery(
        exchange="binance", market_type="perp", symbol="BTC/USDT",
        timeframe="4h", strategy="donchian", end="2023-10-31",
    )
    assert named.end == "2023-10-31 23:59:59"

    # An explicit instant is left alone: only a bare date is a day.
    exact = AnalysisQuery(
        exchange="binance", market_type="perp", symbol="BTC/USDT",
        timeframe="4h", strategy="donchian", end="2023-10-31 00:00:00",
    )
    assert exact.end == "2023-10-31 00:00:00"


def test_a_new_timeframe_is_backfilled_from_the_caller_s_own_left_edge(client, monkeypatch):
    """The five-bar top-up is for a series that exists. Applied to a timeframe
    with nothing stored it leaves five candles and a warmup error, so the ladder
    passes the frame it is looking at and gets history to match."""
    seen = {}

    def _refresh(identity, after, since):
        seen["since"] = since
        return {"bars": [], "candles_upserted": 900, "funding_upserted": None}

    monkeypatch.setattr("strategy_lab.server.refresh_candles", _refresh)

    response = client.post(
        "/api/refresh",
        params={
            "exchange": "binance", "market_type": "spot", "symbol": "BTC/USDT",
            "timeframe": "1h", "since": "2024-01-01T00:00:00Z",
        },
    )

    assert response.status_code == 200
    assert seen["since"] is not None
    assert seen["since"].year == 2024


def test_a_plain_refresh_still_asks_for_no_particular_start(client, monkeypatch):
    """`since` is the new-timeframe path only. A top-up that passed one would
    re-fetch the whole history on every click of the refresh button."""
    seen = {}
    monkeypatch.setattr(
        "strategy_lab.server.refresh_candles",
        lambda identity, after, since: seen.update(since=since)
        or {"bars": [], "candles_upserted": 3, "funding_upserted": 1},
    )

    client.post(
        "/api/refresh",
        params={
            "exchange": "binance", "market_type": "perp", "symbol": "BTC/USDT",
            "timeframe": "4h",
        },
    )

    assert seen["since"] is None


def test_an_unparseable_since_is_a_client_error_not_a_traceback(client):
    """It was parsed with `pd.Timestamp` inside the route but outside its
    `try`, so a bad value escaped as a 500."""
    response = client.post(
        "/api/refresh",
        params={
            "exchange": "binance", "market_type": "spot", "symbol": "BTC/USDT",
            "timeframe": "1h", "since": "not-a-date",
        },
    )

    assert response.status_code == 422


def test_a_bare_date_is_midnight_utc_rather_than_a_422(client, monkeypatch):
    """`AnalysisQuery` already takes date-only bounds, so refusing one here would
    make the same date mean two things depending on the endpoint. Appending a
    zone to it alone yields `2024-01-01+00:00`, which is not a datetime."""
    seen = {}
    monkeypatch.setattr(
        "strategy_lab.server.refresh_candles",
        lambda identity, after, since: seen.update(since=since)
        or {"bars": [], "candles_upserted": 0, "funding_upserted": None},
    )

    response = client.post(
        "/api/refresh",
        params={
            "exchange": "binance", "market_type": "spot", "symbol": "BTC/USDT",
            "timeframe": "1h", "since": "2024-01-01",
        },
    )

    assert response.status_code == 200, response.text
    assert seen["since"].isoformat() == "2024-01-01T00:00:00+00:00"


def test_a_since_without_a_zone_is_read_as_utc(client, monkeypatch):
    """Everything downstream compares against aware datetimes, so a naive one
    raises deep in the fetch rather than at the boundary that accepted it."""
    seen = {}
    monkeypatch.setattr(
        "strategy_lab.server.refresh_candles",
        lambda identity, after, since: seen.update(since=since)
        or {"bars": [], "candles_upserted": 0, "funding_upserted": None},
    )

    response = client.post(
        "/api/refresh",
        params={
            "exchange": "binance", "market_type": "spot", "symbol": "BTC/USDT",
            "timeframe": "1h", "since": "2024-01-01T00:00:00",
        },
    )

    assert response.status_code == 200
    assert seen["since"].tzinfo is not None
