"""The board, and the one claim it rests on: it computes no answer of its own.

A board that disagreed with the single-instrument view would be a **third**
path -- after the backtest and the browser -- free to say a different thing
about the same bars, which is the failure the whole design is arranged to
prevent. So the centre of gravity here is
``test_a_rows_state_and_latest_fill_are_the_single_views_own``: every answered
row is compared field by field against ``/api/analysis`` over the window that
row's own provenance says it used, on real stored frames.

The rest of the file is the four properties the board needs in order to be
usable at all and one that it must never acquire:

* a refusal is **data about one instrument**, not a blank board;
* a second look with no new bar reads **no candle**;
* a new bar invalidates **only** the rows that received it;
* rows **arrive as they finish**, because the cost is serial and measured;
* and ``browse`` still writes **nothing** -- no report directory, no ``signals``
  row, no ``bar_reasons`` row, no schema change.

The equity section at the end adds what a second market type needs: a frame's
bound is chosen **by its market type**, and an instrument that settles nothing is
never asked about settlements -- asserted by statement, at the call and again at
the SQL, because that failure is silent rather than loud.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, func, select
from sqlalchemy.engine import Engine

from strategy_lab.api import app as app_module
from strategy_lab.api.board import (
    MAX_SPARK_BARS,
    BoardStamp,
    BoardWindow,
    DatasetRef,
    funding_window,
    stored_datasets,
    stream_board,
)
from strategy_lab.backtests.funding_frame import FundingUnavailable
from strategy_lab.db import list_candle_sets, load_candles
from strategy_lab.db.candles import get_engine
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.storage.schema import bar_reasons_table, signals_table
from tests.conftest import synthetic_ohlcv

# The strategy the board is worth looking at with: the boolean contract's only
# state machine, so its rows carry a state, a feature frame *and* fills, where a
# marker-only strategy fills in three fields of a tile and a continuous one
# leaves the fill empty.
_MACHINE = "state_machine_v1"

# A stamp for the tests that drive ``_compute`` directly. Its value is never
# read there -- the window is handed to a stubbed ``build_analysis`` -- so it
# stands for "whatever the enumeration said" rather than for a real dataset.
_ANY_STAMP = BoardStamp(
    dataset_last_bar="2024-01-02", window=BoardWindow(start=None, end=None)
)


def _rows(response) -> list[dict]:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def _key(row: dict) -> tuple[str, ...]:
    identity = row["identity"]
    return (
        identity["exchange"],
        identity["market_type"],
        identity["symbol"],
        identity["timeframe"],
    )


# --------------------------------------------------------------------------
# Check 1 -- a row per dataset, and a refusal that stays inside its own row.
# --------------------------------------------------------------------------


@pytest.mark.db
def test_a_board_over_every_stored_perp_answers_one_row_per_dataset():
    """Including the ones it cannot answer.

    BTC/USDT perp candles begin 40 h before the venue's first settlement, so an
    unbounded request raises ``FundingUnavailable`` permanently and correctly --
    a board that asked for "every dataset" and did not bound each frame by its
    own funding span would fail on the first instrument and show none of the
    others.

    A dataset shorter than the strategy's warmup reports the engine's own
    refusal without touching its neighbours. That used to be asserted against a
    25-bar 1h probe set that happened to be stored; a backfill took it to 57,915
    bars and this failed — the test had been pinned to the data being broken.
    The refused-row *shape* is proven deterministically in
    ``test_a_refused_row_carries_the_reason_and_nothing_else``, and what belongs
    here is that the two groups partition the board.
    """
    client = TestClient(app_module.create_app())
    stored = list_candle_sets()
    expected = {
        (row.exchange, row.market_type, row.symbol, row.timeframe)
        for row in stored[stored["market_type"] == "perp"].itertuples()
    }
    assert expected, "no stored perp datasets; fetch one before running this"

    rows = _rows(client.get("/api/board", params={"strategies": _MACHINE, "market_type": "perp"}))

    assert {_key(row) for row in rows} == expected
    answered = [row for row in rows if row["unavailable"] is None]
    assert answered, "every perp row refused; the funding bound is not being applied"
    for row in answered:
        assert row["state"] is not None
        assert row["as_of"] is not None
        assert row["closes"], "a row with no sparkline tail"
        assert row["provenance"]["crowding_measured"] is True

    refused = [row for row in rows if row["unavailable"] is not None]
    assert len(answered) + len(refused) == len(rows), "a row was neither"


def test_a_refused_row_carries_the_reason_and_nothing_else(monkeypatch):
    """Half an answer beside an explanation is worse than neither: a reader
    cannot tell which fields the refusal invalidated.

    Constructed rather than found. This used to ride on a 25-bar probe set
    sitting in the research database, so deepening that dataset deleted the
    coverage — the assertion that mattered was never about *that* dataset.
    """
    _stub_datasets(monkeypatch, [("BTC/USDT", "4h")])

    def _too_short(*_args, **_kwargs):
        # The engine's own message, raised where the engine raises it: a frame
        # shorter than the declared warmup.
        raise ValueError(
            "state_machine_v1 declares 2192 warmup bars but the frame has 25; "
            "every bar would be masked"
        )

    monkeypatch.setattr("strategy_lab.api.board.build_analysis", _too_short)
    client = TestClient(app_module.create_app())

    [row] = _rows(client.get("/api/board", params={"strategies": _MACHINE}))

    assert "the frame has 25" in row["unavailable"]
    assert row["provenance"] is None
    assert (row["state"], row["features"], row["latest_fill"], row["closes"]) == (
        None,
        None,
        None,
        [],
    )
    # Both facts about the stored candles survive a refusal: neither describes a
    # run that did not happen.
    assert row["dataset_last_bar"] and row["last_written"]


def test_funding_that_cannot_cover_the_candles_is_reported_in_the_row(monkeypatch):
    """The coverage guard's refusal, carried rather than raised.

    Patched rather than found, because every stored perp *is* covered once its
    frame is bounded -- which is the point of the bound. What has to be pinned
    is that when the guard does refuse, the board answers 200 with the reason on
    that instrument's row and its neighbour is **answered**: one contract's gap
    must not blank the other fifteen.
    """
    _stub_datasets(monkeypatch, [("BTC/USDT", "4h"), ("ETH/USDT", "4h")])
    monkeypatch.setattr(
        "strategy_lab.api.analysis.load_candles",
        lambda **kwargs: synthetic_ohlcv(n=600, freq="4h"),
    )

    def _uncovered(identity, frame, **kwargs):
        if identity.symbol == "BTC/USDT":
            raise FundingUnavailable("Stored funding ... does not cover ... fetch-funding")
        return None

    monkeypatch.setattr("strategy_lab.backtests.funding_frame.funding_rates", _uncovered)
    client = TestClient(app_module.create_app())

    rows = _rows(client.get("/api/board", params={"strategies": "donchian"}))

    by_symbol = {row["identity"]["symbol"]: row for row in rows}
    assert "does not cover" in by_symbol["BTC/USDT"]["unavailable"]
    assert "fetch-funding" in by_symbol["BTC/USDT"]["unavailable"]
    assert by_symbol["ETH/USDT"]["unavailable"] is None
    assert by_symbol["ETH/USDT"]["closes"]


@pytest.mark.db
def test_a_perp_frame_is_bounded_by_its_own_funding_span():
    """The bound is stored funding's, not the candle table's.

    Both sides matter and for different reasons. The head can be permanently
    uncovered -- BTC's 40 h -- and the tail moves whenever candles are fetched
    without funding, which is what leaves the guard refusing a frame the board
    was showing a moment ago. ``as_of`` is therefore allowed to sit behind
    ``dataset_last_bar``, and the row says so rather than implying it is current.
    """
    identity = MarketDataIdentity(
        exchange="binance", market_type="perp", symbol="BTC/USDT", timeframe="4h"
    )
    window = funding_window(identity)
    assert window.start is not None and window.end is not None

    client = TestClient(app_module.create_app())
    rows = _rows(client.get("/api/board", params={"strategies": _MACHINE, "market_type": "perp"}))
    [btc] = [row for row in rows if _key(row) == ("binance", "perp", "BTC/USDT", "4h")]

    assert btc["unavailable"] is None, "the flagship perp is exactly the frame the bound exists for"
    assert pd.Timestamp(btc["provenance"]["first_bar"]) >= pd.Timestamp(window.start)
    assert pd.Timestamp(btc["as_of"]) <= pd.Timestamp(window.end)
    for row in rows:
        if row["unavailable"] is None:
            assert pd.Timestamp(row["as_of"]) <= pd.Timestamp(row["dataset_last_bar"])


# --------------------------------------------------------------------------
# Check 2 -- the reason this is a slice and not a second computation.
# --------------------------------------------------------------------------


@pytest.mark.db
def test_a_rows_state_and_latest_fill_are_the_single_views_own():
    """Asserted directly, against ``/api/analysis`` over the row's own window.

    The window comes off the row's provenance rather than out of the board's
    own bounding function, so this compares two answers rather than one answer
    with itself: whatever frame the board says it used, the single view over
    that frame has to agree on the state, the last fill, every last feature
    value and the whole provenance block bar the clock.
    """
    client = TestClient(app_module.create_app())
    rows = _rows(client.get("/api/board", params={"strategies": _MACHINE, "market_type": "perp"}))
    answered = [row for row in rows if row["unavailable"] is None]
    assert answered

    for row in answered:
        provenance = row["provenance"]
        payload = client.get(
            "/api/analysis",
            params={
                **row["identity"],
                "strategy": row["strategy"],
                "start": provenance["first_bar"],
                "end": provenance["last_bar"],
            },
        )
        assert payload.status_code == 200, payload.text
        single = payload.json()

        assert single["why"]["states"][-1] == row["state"]
        assert {
            name: values[-1] for name, values in single["why"]["features"].items()
        } == row["features"]
        assert (single["markers"][-1] if single["markers"] else None) == row["latest_fill"]
        assert single["bars"][-1]["close"] == row["closes"][-1]
        # ``generated_at`` is the one field that must differ: two computations
        # at two instants. Everything else describes the run and cannot.
        assert {k: v for k, v in single["provenance"].items() if k != "generated_at"} == {
            k: v for k, v in provenance.items() if k != "generated_at"
        }


def test_a_row_is_the_last_bar_of_every_series_the_payload_carries(monkeypatch):
    """The slice itself, on a payload whose every bar differs.

    The end-to-end parity test above cannot see an off-by-one in the *state*:
    a state machine dwells, so ``states[-2]`` is usually ``states[-1]`` and the
    comparison passes on two values that happen to be equal. Here every series
    is distinct per bar, so "the last one" is the only slice that answers.
    """
    from strategy_lab.api.analysis import AnalysisPayload, Marker, Provenance, WhyLayer
    from strategy_lab.api.board import _compute

    bars = [
        {"time": 100 + i, "open": 1.0, "high": 2.0, "low": 0.5, "close": 10.0 + i, "volume": 3.0}
        for i in range(4)
    ]
    payload = AnalysisPayload(
        bars=bars,
        markers=[
            Marker(time=101, kind="entry", side="long", price=1.0, size=1.0),
            Marker(time=103, kind="exit", side="long", price=2.0, size=1.0),
        ],
        # A tile shows the latest *fill*, never a round trip -- there is no room
        # for one, and `latest_fill` is the slice it takes.
        trades=[],
        position_size=None,
        target=[0.1, 0.2, 0.3, 0.4],
        why=WhyLayer(
            states=["compression", "breakout", "confirmed", "riding"],
            features={"energy": [0.1, 0.2, 0.3, 0.4]},
        ),
        provenance=Provenance(
            identity={}, strategy=_MACHINE, version="1.0.0", contract="signal_set",
            exit_mode="continuation_failure", failure_bars=4, warmup_bars=1,
            allow_shorts=True, reads_crowding=True, crowding_measured=True,
            funding_attached=True,
            cost_model=None, first_bar="2024-01-01", last_bar="2024-01-02",
            bar_count=4, generated_at="2024-01-02T00:00:00+00:00",
        ),
    )
    monkeypatch.setattr("strategy_lab.api.board.build_analysis", lambda *a, **k: payload)
    dataset = DatasetRef(
        identity=MarketDataIdentity(
            exchange="binance", market_type="perp", symbol="BTC/USDT", timeframe="4h"
        ),
        last_bar="2024-01-02",
        last_written="2024-01-03 09:00:00+00:00",
    )

    row = _compute(dataset, strategy=_MACHINE, stamp=_ANY_STAMP, exit_mode=None)

    assert row.state == "riding"
    assert row.features == {"energy": 0.4}
    assert row.latest_fill == payload.markers[-1]
    assert row.target == 0.4
    assert row.closes == [10.0, 11.0, 12.0, 13.0]
    assert row.as_of == "2024-01-02"
    # Off the enumeration, not off the payload: when the bars were written is a
    # fact about storage, and the analysis has no opinion about it.
    assert row.last_written == "2024-01-03 09:00:00+00:00"


# --------------------------------------------------------------------------
# Check 3 -- every request recomputes, which is the browser's contract.
# --------------------------------------------------------------------------


@pytest.fixture
def statements():
    """Every SQL statement any engine executes, in order.

    Listened for on the ``Engine`` class rather than an instance because
    ``get_engine`` builds a fresh one per call, so an instance-level listener
    would watch a connection pool nothing else uses.
    """
    seen: list[str] = []

    def _record(conn, cursor, statement, parameters, context, executemany):
        seen.append(statement)

    event.listen(Engine, "before_cursor_execute", _record)
    try:
        yield seen
    finally:
        event.remove(Engine, "before_cursor_execute", _record)


@pytest.mark.db
def test_every_board_request_recomputes_from_stored_candles(statements, monkeypatch):
    """The browser's contract, and the board is not exempt from it.

    An earlier version memoised each row against the newest stored candle and
    the funding window. ``POST /api/refresh`` upserts *overlapping* recent
    candles by design, so a corrective refresh that rewrote the last few bars
    without adding one left that stamp unmoved and the tile stale -- while
    ``/api/analysis`` recomputed and disagreed, which is the one failure M36
    exists to prevent. So every request reads the candles again, and the cost is
    paid by streaming rather than by remembering.
    """
    loads: list[str] = []

    def _counted(**kwargs):
        loads.append(kwargs["symbol"])
        return load_candles(**kwargs)

    monkeypatch.setattr("strategy_lab.api.analysis.load_candles", _counted)
    client = TestClient(app_module.create_app())
    params = {"strategies": _MACHINE, "market_type": "perp"}

    first = _rows(client.get("/api/board", params=params))
    assert loads, "the first board computed nothing"
    answered = sorted(loads)

    loads.clear()
    second = _rows(client.get("/api/board", params=params))

    assert sorted(loads) == answered, "the second board did not recompute every row"
    # Same answer, recomputed: only the instant each was generated may differ.
    assert [_without_generated_at(row) for row in second] == [
        _without_generated_at(row) for row in first
    ]


def _without_generated_at(row: dict) -> dict:
    provenance = row.get("provenance")
    if not provenance:
        return row
    return {**row, "provenance": {k: v for k, v in provenance.items() if k != "generated_at"}}


def _stub_datasets(
    monkeypatch,
    pairs: list[tuple[str, str]],
    *,
    market_type: str = "perp",
    last_written: str = "2024-01-01 00:00:00+00:00",
) -> pd.DataFrame:
    """The enumeration query, replaced by a frame a test can move a bar on."""
    sets = pd.DataFrame(
        [
            {
                "exchange": "binance",
                "market_type": market_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": 600,
                "first_timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
                "last_timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
                "last_written": pd.Timestamp(last_written),
            }
            for symbol, timeframe in pairs
        ]
    )
    monkeypatch.setattr("strategy_lab.api.board.list_candle_sets", lambda: sets)
    monkeypatch.setattr("strategy_lab.api.board.funding_span", lambda **kwargs: None)
    return sets


# --------------------------------------------------------------------------
# Check 5 -- what browsing must never do.
# --------------------------------------------------------------------------


@pytest.mark.db
@pytest.mark.parametrize("market_type", ["perp", "equity"])
def test_a_board_request_writes_nothing(statements, tmp_path, monkeypatch, market_type):
    """No report directory, no ``signals`` row, no ``bar_reasons`` row, no DDL.

    Asserted rather than assumed, and asserted three ways because they fail
    differently: a stray ``run_backtest`` would leave a dated directory under
    the working directory, a stray write would land in one of the two
    append-only tables, and a stray migration would issue DDL against a database
    this whole surface treats as read-only.

    Both market types, because widening the board is exactly the kind of change
    that adds a writer: an equity's freshness comes from ``updated_at``, which is
    a column a write maintains, and reading it must not be the thing that moves
    it.
    """
    engine = get_engine()
    with engine.connect() as conn:
        before = {
            "signals": conn.execute(select(func.count()).select_from(signals_table)).scalar(),
            "bar_reasons": conn.execute(
                select(func.count()).select_from(bar_reasons_table)
            ).scalar(),
        }

    client = TestClient(app_module.create_app())
    monkeypatch.chdir(tmp_path)
    statements.clear()
    rows = _rows(
        client.get("/api/board", params={"strategies": _MACHINE, "market_type": market_type})
    )
    client.get("/")

    assert rows
    assert list(tmp_path.iterdir()) == [], "browsing left something on disk"
    with engine.connect() as conn:
        after = {
            "signals": conn.execute(select(func.count()).select_from(signals_table)).scalar(),
            "bar_reasons": conn.execute(
                select(func.count()).select_from(bar_reasons_table)
            ).scalar(),
        }
    assert after == before

    writing = re.compile(r"^\s*(insert|update|delete|create|alter|drop|truncate)\b", re.I)
    offenders = [sql for sql in statements if writing.match(sql)]
    assert offenders == [], f"the board issued a write: {offenders}"


# --------------------------------------------------------------------------
# Check 6 -- why first paint is not the sum of every row.
# --------------------------------------------------------------------------


def test_rows_are_yielded_as_they_finish_rather_than_as_one_blob(monkeypatch):
    """The mechanism behind the first-paint number, asserted rather than timed.

    A row is a whole-history ``build_analysis``, 330-400 ms warm on the stored
    perp frames, and **parallelism does not help** -- four threads over three of
    them measured 1.10x, the work being pandas and vectorbt under the GIL. What
    a caller can have instead is the first row early. A timing assertion here
    would measure the machine; what has to hold is that pulling one row runs one
    analysis, which is what makes the page's incremental reader possible at all.
    """
    computed: list[str] = []

    def _analysis(identity, **kwargs):
        computed.append(identity.symbol)
        raise ValueError("not the point of this test")

    monkeypatch.setattr("strategy_lab.api.board.build_analysis", _analysis)
    datasets = [
        DatasetRef(
            identity=MarketDataIdentity(
                exchange="binance", market_type="perp", symbol=symbol, timeframe="4h"
            ),
            last_bar="2024-01-01 00:00:00+00:00",
            last_written="2024-01-01 00:30:00+00:00",
        )
        for symbol in ("BTC/USDT", "ETH/USDT", "SOL/USDT")
    ]
    monkeypatch.setattr("strategy_lab.api.board.funding_span", lambda **kwargs: None)

    rows = stream_board(datasets, strategies=[_MACHINE])

    assert next(rows).identity["symbol"] == "BTC/USDT"
    assert computed == ["BTC/USDT"], "the whole board was computed before the first row"
    assert next(rows).identity["symbol"] == "ETH/USDT"
    assert computed == ["BTC/USDT", "ETH/USDT"]


# --------------------------------------------------------------------------
# The rest of the surface.
# --------------------------------------------------------------------------


@pytest.mark.db
def test_the_enumeration_is_storages_own_identity_newest_bar_and_newest_write():
    """Both stamps, from the one group-by rather than a query per dataset.

    They answer different questions and only look alike on a venue whose history
    grows: an equity series is restated on a dividend, so every bar can move
    without the newest one changing.
    """
    stored = list_candle_sets()
    perps = stored[stored["market_type"] == "perp"]

    listed = stored_datasets(market_type="perp")

    assert len(listed) == len(perps)
    assert {ref.identity.symbol for ref in listed} == set(perps["symbol"])
    assert all(ref.last_bar for ref in listed)
    assert all(ref.last_written for ref in listed)


def test_every_board_row_field_survives_the_wire_model(monkeypatch):
    """A response model silently drops what it has not declared.

    The board streams, so ``BoardRowModel`` is applied per line rather than by
    FastAPI -- which means a field added to ``BoardRow`` and forgotten there
    would vanish from the wire with nothing raised, exactly the loss
    ``extra="forbid"`` exists to turn into an error. This pins the list.
    """
    from dataclasses import fields

    from strategy_lab.api.board import BoardRow

    _stub_datasets(monkeypatch, [("BTC/USDT", "4h")])
    monkeypatch.setattr(
        "strategy_lab.api.analysis.load_candles",
        lambda **kwargs: synthetic_ohlcv(n=600, freq="4h"),
    )
    monkeypatch.setattr(
        "strategy_lab.backtests.funding_frame.funding_rates", lambda *a, **k: None
    )
    client = TestClient(app_module.create_app())

    [row] = _rows(client.get("/api/board", params={"strategies": "donchian"}))

    assert set(row) == {field.name for field in fields(BoardRow)}


def test_the_continuous_contract_reaches_a_row_as_a_target(monkeypatch):
    """``state_machine_v2`` has no fills to show, so its tile shows the level.

    A board that only knew how to draw a latest fill would render the continuous
    contract as "no fills" -- which is true and useless, the same way a marker
    vocabulary cannot express "62% long".
    """
    _stub_datasets(monkeypatch, [("BTC/USDT", "4h")])
    monkeypatch.setattr(
        "strategy_lab.api.analysis.load_candles",
        lambda **kwargs: synthetic_ohlcv(n=2600, freq="4h"),
    )
    monkeypatch.setattr(
        "strategy_lab.backtests.funding_frame.funding_rates", lambda *a, **k: None
    )
    client = TestClient(app_module.create_app())

    [row] = _rows(client.get("/api/board", params={"strategies": "state_machine_v2"}))

    assert row["contract"] == "target_exposure"
    assert row["latest_fill"] is None
    assert row["target"] is not None
    assert row["state"] is not None


def test_the_sparkline_tail_is_bounded_by_what_was_asked_for(monkeypatch):
    _stub_datasets(monkeypatch, [("BTC/USDT", "4h")])
    monkeypatch.setattr(
        "strategy_lab.api.analysis.load_candles",
        lambda **kwargs: synthetic_ohlcv(n=600, freq="4h"),
    )
    monkeypatch.setattr(
        "strategy_lab.backtests.funding_frame.funding_rates", lambda *a, **k: None
    )
    client = TestClient(app_module.create_app())

    def closes(**overrides):
        params = {"strategies": "donchian", **overrides}
        [row] = _rows(client.get("/api/board", params=params))
        return row["closes"]

    assert len(closes()) == 120
    # A row is built with ``board.MAX_SPARK_BARS`` and trimmed on the
    # way out, so a longer tail asked for second is served in full rather than
    # from a copy already cut to the first request's length.
    assert len(closes(spark_bars=MAX_SPARK_BARS)) == MAX_SPARK_BARS


# --------------------------------------------------------------------------
# R10c -- a second market type, on the same one computation.
# --------------------------------------------------------------------------

# Weekly ETF sets: short enough that ``state_machine_v1`` (2,192 warmup bars)
# refuses them and long enough that ``donchian`` (96) does not, which is what
# makes a board of real answers *and* refusals rather than a stub. A weekly set
# reaching 2,192 bars would be 42 years of history, so this stays true of any
# fetch -- but the *count* in the refusal is read from the frame rather than
# written down here, because a backfill moved these from 333 bars to 345 and the
# literal was the only thing that failed.
_SHORT_WEEKLIES = {("yahoo", "equity", symbol, "1w") for symbol in ("XLF", "XLK", "QQQ", "SMH")}

_EQUITY_STRATEGIES = f"{_MACHINE},donchian"


def _equity_identity() -> MarketDataIdentity:
    return MarketDataIdentity(
        exchange="yahoo", market_type="equity", symbol="SPY", timeframe="1w"
    )


@pytest.mark.db
def test_a_board_over_every_stored_equity_answers_one_row_per_dataset_and_strategy():
    """Check 1, on the datasets that actually hit the warmup rule: a short frame
    is a fact about that instrument's stored bars, not a reason for the board to
    blank, so the refusal rides in its own row while its neighbours answer.
    """
    client = TestClient(app_module.create_app())
    stored = list_candle_sets()
    equities = {
        (row.exchange, row.market_type, row.symbol, row.timeframe)
        for row in stored[stored["market_type"] == "equity"].itertuples()
    }
    assert equities, "no stored equity datasets; fetch one before running this"
    assert _SHORT_WEEKLIES <= equities, "the short weekly sets are not stored"

    rows = _rows(
        client.get(
            "/api/board", params={"strategies": _EQUITY_STRATEGIES, "market_type": "equity"}
        )
    )

    assert len(rows) == len(equities) * 2
    assert {_key(row) for row in rows} == equities
    by_pair = {(_key(row), row["strategy"]): row for row in rows}

    for key in _SHORT_WEEKLIES:
        refused = by_pair[(key, _MACHINE)]
        assert refused["unavailable"] is not None, key
        # Read from the frame, so this asserts the message reports the real
        # length rather than that the database still holds a particular one.
        stored_bars = len(load_candles(
            exchange=key[0], market_type=key[1], symbol=key[2], timeframe=key[3]
        ))
        assert f"but the frame has {stored_bars}" in refused["unavailable"], key
        # The refusal is the whole of the row, and the two facts about the stored
        # bars survive it -- neither describes a run that did not happen.
        assert refused["provenance"] is None
        assert refused["dataset_last_bar"] and refused["last_written"]
        assert by_pair[(key, "donchian")]["unavailable"] is None, key

    answered = [row for row in rows if row["unavailable"] is None]
    assert answered
    for row in answered:
        assert row["as_of"] is not None
        assert row["closes"], "a row with no sparkline tail"
        # No settlement bounds the frame, so it reaches the newest stored bar --
        # the perp's one-bar lag (M37) is structurally absent here.
        assert row["as_of"] == row["dataset_last_bar"]


@pytest.mark.db
def test_an_equity_rows_state_and_latest_fill_are_the_single_views_own():
    """Check 2, unchanged from R10b: M36 binds every view, and a second market
    type is still one view.

    The window comes off the row's own provenance rather than out of the board's
    bounding function, so this compares two answers rather than one answer with
    itself.
    """
    client = TestClient(app_module.create_app())
    rows = _rows(
        client.get(
            "/api/board", params={"strategies": _EQUITY_STRATEGIES, "market_type": "equity"}
        )
    )
    answered = [row for row in rows if row["unavailable"] is None]
    assert answered
    assert any(row["state"] is not None for row in answered), (
        "no answered row carries a state; this compares nothing about the why layer"
    )

    for row in answered:
        provenance = row["provenance"]
        payload = client.get(
            "/api/analysis",
            params={
                **row["identity"],
                "strategy": row["strategy"],
                "start": provenance["first_bar"],
                "end": provenance["last_bar"],
            },
        )
        assert payload.status_code == 200, payload.text
        single = payload.json()

        assert (single["why"]["states"][-1] if single["why"] else None) == row["state"]
        assert (
            None
            if single["why"] is None
            else {name: values[-1] for name, values in single["why"]["features"].items()}
        ) == row["features"]
        assert (single["markers"][-1] if single["markers"] else None) == row["latest_fill"]
        assert single["bars"][-1]["close"] == row["closes"][-1]
        assert {k: v for k, v in single["provenance"].items() if k != "generated_at"} == {
            k: v for k, v in provenance.items() if k != "generated_at"
        }


def test_only_a_perp_is_asked_about_funding_and_asking_otherwise_is_refused(monkeypatch):
    """Check 3, on the function the board calls, with a perp control beside it.

    Asserted by statement rather than by the absence of an error: a
    ``funding_span`` for an equity returns ``None`` and everything carries on
    working, having invented a coverage question for a market with no answer to
    it. Without the perp in the list the empty assertion would also pass on a
    board that never bounded anything.
    """
    with pytest.raises(ValueError, match="settles nothing"):
        funding_window(_equity_identity())

    asked: list[str] = []

    def _record(**kwargs) -> None:
        asked.append(kwargs["symbol"])
        return None

    def _uncomputed(*args, **kwargs):
        raise ValueError("the analysis is not what this test is about")

    monkeypatch.setattr("strategy_lab.api.board.funding_span", _record)
    monkeypatch.setattr("strategy_lab.api.board.build_analysis", _uncomputed)
    datasets = [
        DatasetRef(
            identity=identity,
            last_bar="2024-01-01 00:00:00+00:00",
            last_written="2024-01-01 00:30:00+00:00",
        )
        for identity in (
            _equity_identity(),
            MarketDataIdentity(
                exchange="binance", market_type="spot", symbol="BTC/USDT", timeframe="1d"
            ),
            MarketDataIdentity(
                exchange="binance", market_type="perp", symbol="BTC/USDT", timeframe="4h"
            ),
        )
    ]

    rows = list(stream_board(datasets, strategies=["donchian"]))

    assert len(rows) == 3
    assert asked == ["BTC/USDT"], "a market that settles nothing was asked about settlements"


@pytest.mark.db
def test_no_funding_query_reaches_the_database_for_an_equity_board(statements):
    """Check 3 again, one layer down: the SQL, not the call.

    The function-level assertion above is exact and the substitution below is
    durable -- a future bound that read ``funding_rates`` by some other route
    would satisfy the first and fail this. The perp control is in the same test
    so an assertion that stopped matching cannot pass by matching nothing.
    """
    client = TestClient(app_module.create_app())

    statements.clear()
    equity = _rows(
        client.get("/api/board", params={"strategies": "donchian", "market_type": "equity"})
    )
    funding_reads = [sql for sql in statements if "funding_rates" in sql]

    assert equity
    assert funding_reads == [], f"an equity board queried funding: {funding_reads}"

    statements.clear()
    perp = _rows(
        client.get("/api/board", params={"strategies": "donchian", "market_type": "perp"})
    )
    assert perp
    assert [sql for sql in statements if "funding_rates" in sql], (
        "the perp control issued no funding query; this test would pass on a "
        "board that had stopped bounding anything"
    )


@pytest.mark.db
def test_an_equity_row_carries_when_its_candles_were_last_written():
    """``max(updated_at)``, per dataset, and not a second spelling of the newest
    bar. The re-query is per row rather than in aggregate because what can go
    wrong is the grouping: one set's write time attached to another's identity.
    """
    from sqlalchemy import and_

    from strategy_lab.db.candles import candles_table

    client = TestClient(app_module.create_app())
    rows = _rows(
        client.get("/api/board", params={"strategies": "donchian", "market_type": "equity"})
    )
    assert rows

    engine = get_engine()
    for row in rows:
        identity = row["identity"]
        with engine.connect() as conn:
            written = conn.execute(
                select(func.max(candles_table.c.updated_at)).where(
                    and_(
                        candles_table.c.exchange == identity["exchange"],
                        candles_table.c.market_type == identity["market_type"],
                        candles_table.c.symbol == identity["symbol"],
                        candles_table.c.timeframe == identity["timeframe"],
                    )
                )
            ).scalar()
        assert pd.Timestamp(row["last_written"]) == pd.Timestamp(written), identity
        # A write happens after the bar it writes, never before.
        assert pd.Timestamp(row["last_written"]) >= pd.Timestamp(row["dataset_last_bar"])
    # And on this market they are genuinely different questions: every stored
    # equity set was written after its newest bar closed.
    assert any(row["last_written"] != row["dataset_last_bar"] for row in rows)
