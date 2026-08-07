"""The page, and the one decision on it that can silently be wrong.

That decision is which primitive draws which contract. A ``SignalSet`` is a set
of fills and markers say that exactly; a ``TargetExposure`` is a signed level in
-1..1 that markers cannot express -- measured, ``state_machine_v2`` spans the
full range, so a marker vocabulary would have to round "62% long" to an arrow or
to nothing. Drawn wrong, the chart still looks like a chart, which is why the
mapping lives in Python and is pinned here in both directions.

The rest of the suite exists because "the page serves" is not a test of a page:
it also has to inline the chart it draws with, keep provenance out of a tooltip,
and never grow a way to write into ``reports/``.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from strategy_lab.api.analysis import Contract
from strategy_lab.api.app import create_app
from strategy_lab.browser.page import (
    CONTRACT_PRIMITIVES,
    CONTRACTS_WITH_EXIT_MODE,
    bootstrap_config,
    render_browser_html,
)
from strategy_lab.strategies.exposure_registry import list_exposure_strategies
from strategy_lab.strategies.registry import list_strategies
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
    return TestClient(create_app())


@pytest.fixture(scope="module")
def page() -> str:
    return render_browser_html()


def _script(page: str) -> str:
    """The page's own script, without the 191 KB of vendored chart under it."""
    return page[page.rindex("<script>") :]


def _authored(page: str) -> str:
    """Markup and script this repo wrote -- the vendored library cut out.

    Grepping the whole page for anything means grepping 191 KB of minified
    third-party JavaScript, where every short string occurs somewhere.
    """
    return page[: page.index('<script id="config"')] + _script(page)


# --------------------------------------------------------------------------
# One primitive per contract, and the right one.
# --------------------------------------------------------------------------


def test_every_contract_has_a_primitive_to_be_drawn_with():
    """A contract with no entry would be drawn by whichever branch ran last."""
    assert set(CONTRACT_PRIMITIVES) == {contract.value for contract in Contract}


def test_the_continuous_contract_is_drawn_as_a_level_and_not_as_fills():
    assert CONTRACT_PRIMITIVES[Contract.TARGET_EXPOSURE.value] == "baseline"
    assert CONTRACT_PRIMITIVES[Contract.SIGNAL_SET.value] == "markers"


def test_the_page_draws_a_baseline_series_for_the_level(page):
    """``BaselineSeries`` specifically: it draws a signed series against a zero
    baseline, which is what a -1..1 target is. A line series would draw the same
    numbers with no reading of which side of flat the book is on."""
    script = _script(page)

    assert "LWC.BaselineSeries" in script
    assert "baseValue: { type: 'price', price: 0 }" in script


def test_markers_are_only_ever_set_from_the_markers_primitive(page):
    """The single ``setMarkers`` call is guarded, so a continuous payload cannot
    put arrows on the chart even if one arrived carrying markers."""
    script = _script(page)
    calls = re.findall(r"\.setMarkers\((.*?)\);", script, flags=re.S)

    assert len(calls) == 1, "one place decides, or the guard below proves nothing"
    assert "primitive === 'markers'" in calls[0]
    assert ": []" in calls[0], "the other contract must be cleared, not left stale"


def test_the_level_is_only_ever_set_from_the_baseline_primitive(page):
    script = _script(page)
    [levels] = re.findall(r"var levels = (.*?);", script, flags=re.S)

    assert "primitive === 'baseline'" in levels
    assert "exposureSeries.setData(levels);" in script


@pytest.mark.parametrize(
    ("strategy", "contract"),
    [(name, Contract.SIGNAL_SET.value) for name in list_strategies()]
    + [(name, Contract.TARGET_EXPOSURE.value) for name in list_exposure_strategies()],
)
def test_every_registered_strategy_resolves_to_a_primitive(client, strategy, contract):
    """Registry and page agree for every strategy, not only the two spot-checked
    ones -- a strategy registered later is drawn or the mapping is incomplete."""
    listed = {row["name"]: row for row in client.get("/api/strategies").json()}

    assert listed[strategy]["contract"] == contract
    assert CONTRACT_PRIMITIVES[listed[strategy]["contract"]] in {"markers", "baseline"}


def test_the_level_a_continuous_strategy_answers_with_is_unmarkable(monkeypatch):
    """The other half of the claim, and the measurement behind it: this target
    takes values markers have no word for, so drawing it as fills is not a
    styling choice. ``state_machine_v2`` warms 2,192 bars, so the frame has to
    clear that before there is a level to look at at all."""
    monkeypatch.setattr(
        "strategy_lab.api.analysis.load_candles",
        lambda **kwargs: synthetic_ohlcv(n=3000, freq="4h"),
    )
    monkeypatch.setattr(
        "strategy_lab.backtests.funding_frame.funding_rates",
        lambda identity, frame, **_: None,
    )
    payload = TestClient(create_app()).get(
        "/api/analysis", params={**_PERP, "strategy": "state_machine_v2"}
    ).json()
    levels = {value for value in payload["target"] if value is not None}

    assert CONTRACT_PRIMITIVES[payload["provenance"]["contract"]] == "baseline"
    assert payload["markers"] == []
    assert levels - {0.0, 1.0, -1.0}, "a level markers could not have expressed"
    assert max(abs(value) for value in levels) <= 1.0


# --------------------------------------------------------------------------
# The exit-mode control, against what the API actually accepts.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("contract", sorted(contract.value for contract in Contract))
def test_the_exit_mode_control_offers_exactly_what_the_api_accepts(client, contract):
    """Not asserted against the page's own list: the model is the authority, and
    a control that offered a setting the API refuses would make a strategy
    switch answer 422 instead of drawing."""
    strategy = {
        Contract.SIGNAL_SET.value: "donchian",
        Contract.TARGET_EXPOSURE.value: "state_machine_v2",
    }[contract]

    response = client.get(
        "/api/analysis",
        params={**_PERP, "strategy": strategy, "exit_mode": "opposite_signal_only"},
    )

    assert (response.status_code == 200) == (contract in CONTRACTS_WITH_EXIT_MODE)


def test_the_exit_modes_offered_are_the_engines_own(page):
    """Retyped modes are how a control keeps offering one the engine dropped."""
    from strategy_lab.backtests import ExitMode

    assert bootstrap_config()["exitModes"] == [mode.value for mode in ExitMode]
    assert "trend_structure" in page


# --------------------------------------------------------------------------
# The "why" layer -- the thing this tool exists for.
# --------------------------------------------------------------------------


def test_a_bar_shows_the_state_and_every_feature_behind_it(page):
    """A chart that shows only *that* a signal fired is ``plot.html``, which
    already exists. The state and the feature values are what this adds."""
    script = _script(page)

    assert "chip('State', view.why.states[i], 'state')" in script
    assert "Object.keys(view.why.features).forEach" in script


def test_a_strategy_with_no_feature_frame_says_so_rather_than_showing_nothing(page):
    """An empty panel reads as a strategy with no state; ``donchian`` has none to
    show, which is a different claim and has to be the one made."""
    script = _script(page)

    assert "computes no feature frame" in script
    assert "el('why-note').textContent = view.why" in script, (
        "the message has to be the alternative to the feature list, not a "
        "caption that renders either way"
    )


def test_an_unmeasurable_feature_reads_as_absent_rather_than_neutral(page):
    """``NaN`` crosses the wire as ``null``; drawn as 0.000 it would claim the
    feature was measured and came out neutral, which is a different market."""
    script = _script(page)

    assert "value === null ? '—' : fmt(value, 3)" in script


def test_the_fill_on_a_bar_is_shown_where_it_can_be_read(page):
    """The arrows carry no label -- 692 of them on the stored perp frame overlap
    into a band -- so price and quantity live on the bar's own panel."""
    script = _script(page)

    marker_builder = script[
        script.index("function toMarkers") : script.index("function toLevels")
    ]
    # Reorder the two functions and the slice inverts to empty, which passes the
    # negative assertion below without reading a line of the builder.
    assert marker_builder, "toLevels precedes toMarkers; the slice read nothing"

    assert "(view.fills[bar.time] || []).forEach" in script
    assert "fmt(fill.price, 2) + ' × ' + fmt(fill.size, 4)" in script
    assert "text:" not in marker_builder


# --------------------------------------------------------------------------
# Provenance, permanently.
# --------------------------------------------------------------------------


def test_the_provenance_strip_names_crowding_measured(page):
    assert "crowding_measured" in page


def test_provenance_is_a_strip_rather_than_a_tooltip(page):
    """M20's lesson: a number displayed without the funding context will
    eventually contradict the charter with no way to see why. Hidden behind a
    hover, it is a field somebody has to think to look for."""
    assert '<div class="provenance instrument-only" id="provenance"></div>' in page
    assert "title=" not in _provenance_block(page)


def test_an_unmeasured_crowding_on_a_perp_is_flagged_rather_than_stated(page):
    """A perp whose crowding was not measured is not the funded run the charter
    publishes, and the difference is 16.44% against 15.45% on R5's test half."""
    script = _script(page)

    assert "var crowdingBlind = perp && !prov.crowding_measured;" in script
    assert "crowdingBlind ? 'warn' : ''" in script
    assert "class = 'alert'" in script.replace("className", "class")


def test_every_provenance_field_the_payload_carries_reaches_the_strip(client):
    """A field added to ``Provenance`` and forgotten here would ride the wire and
    never be seen, which is the same silence as not carrying it."""
    from dataclasses import fields

    from strategy_lab.api.analysis import Provenance

    script = _script(render_browser_html())
    rendered = set(re.findall(r"prov\.([a-z_]+)", script))
    # ``identity`` is read field by field rather than as a whole.
    rendered |= {"identity"} if "prov.identity." in script else set()

    assert {field.name for field in fields(Provenance)} <= rendered


def _provenance_block(page: str) -> str:
    start = page.index("function renderProvenance")
    return page[start : page.index("\n  }\n", start)]


# --------------------------------------------------------------------------
# Serving it.
# --------------------------------------------------------------------------


def test_the_page_is_served_as_html_from_the_root(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>research browser · strategy-lab</title>" in response.text


def test_the_page_inlines_the_vendored_chart_rather_than_fetching_it(client):
    """Self-contained for the same reason the frozen report is: a page that
    reaches out for its chart draws nothing the day the CDN moves."""
    from strategy_lab.backtests.report import chart_library_source

    library = chart_library_source()
    body = client.get("/").text

    assert library in body
    assert "src=" not in _authored(body), "no external script or image"
    assert "<link" not in body, "no external stylesheet"


def test_the_page_and_the_report_share_one_stylesheet_shell(client):
    """The frozen record and the live view are the same product; two copies of
    the palette are how they stop looking like it."""
    from strategy_lab.backtests.report import SHELL_CSS

    assert SHELL_CSS in client.get("/").text


def test_the_page_writes_nowhere_but_the_existing_refresh_path(page):
    """Read-only is a property of the page as well as of the API.

    Every request it makes is a GET except the POSTs, and every POST goes to
    ``/api/refresh``, which is ``server.refresh_candles``. The board added two
    more of those -- one per tile and one for every tile -- and no fourth
    endpoint that writes; asserted on the URL rather than on the count, so
    another refresh button is fine and another writer is not.
    """
    script = _script(page)
    posts = re.findall(r"getJSON\('(/[^'?]+)[^;]*?method: '([A-Z]+)'", script, flags=re.S)
    fetched = set(re.findall(r"(?:getJSON|readRows)\('(/[^'?]+)", script))

    assert posts, "no POST found at all; the regex stopped matching"
    assert {url for url, _ in posts} == {"/api/refresh"}
    assert {method for _, method in posts} == {"POST"}
    assert fetched == {
        "/api/analysis",
        "/api/board",
        "/api/refresh",
        "/api/datasets",
        "/api/strategies",
    }


def test_serving_and_analysing_leaves_the_report_tree_alone(client, tmp_path, monkeypatch):
    """``run_backtest`` writes a dated directory; this path deliberately does not
    go through it. A browser that wrote into ``reports/`` would put an
    unreproducible run beside the reproducible ones."""
    monkeypatch.chdir(tmp_path)

    client.get("/")
    assert client.get(
        "/api/analysis", params={**_PERP, "strategy": "donchian"}
    ).status_code == 200

    assert list(tmp_path.iterdir()) == []


def test_the_switcher_re_requests_rather_than_reloading(page):
    """A reload would re-read the 272 ms frame and lose the zoom; the whole
    reason recompute was measured is that it does not need to."""
    script = _script(page)

    assert "location.reload" not in script
    assert (
        "strategySel.addEventListener('change', function () { syncExitEnabled(); reload(); });"
        in script
    )


def test_a_slower_earlier_request_cannot_overwrite_a_later_one(page):
    """Two fast switches in flight at once would otherwise leave one strategy's
    chart under another's provenance strip."""
    script = _script(page)

    assert "var token = ++pending;" in script
    assert "if (token !== pending) return;" in script

def test_the_level_is_revealed_before_it_is_filled(page):
    """A chart sized while its wrapper is ``display:none`` has no width.

    ``linkRange`` is bidirectional, so the degenerate first range such a chart
    reports was mirrored onto the price chart — measured on the stored BTC/USDT
    perp 4h frame, the continuous contract opened showing one bar against a
    600-wide price scale while the candles were fine. Revealing before filling
    is the fix, and the order is the whole of it.
    """
    reveal = page.index("el('exposure-wrap').hidden")
    fill = page.index("exposureSeries.setData(levels)")
    assert reveal < fill, "the exposure pane is filled before it has a width"


def test_the_price_chart_is_the_authority_on_what_range_is_shown(page):
    """Whoever wins the two-way range link decides where the user is looking.

    The price pane carries the candles and survives a strategy switch, so it
    leads; the exposure pane follows it on reveal rather than reporting a range
    of its own into the link.
    """
    assert "priceChart.timeScale().getVisibleLogicalRange()" in page
    assert "exposureChart.timeScale().setVisibleLogicalRange(range)" in page


# --------------------------------------------------------------------------
# The board, and the instrument view it did not replace.
# --------------------------------------------------------------------------


def test_the_board_is_read_as_a_stream_rather_than_as_a_response(page):
    """``response.json()`` here waits for the slowest instrument before drawing
    the fastest. A row is a whole-history recompute and the cost is serial --
    four threads over three stored perp frames measured 1.10x -- so the only
    thing that makes first paint short of the total is reading as it arrives."""
    script = _script(page)

    assert "response.body.getReader()" in script
    assert "new TextDecoder()" in script
    assert "readRows('/api/board?'" in script


def test_both_views_are_reachable_and_only_one_is_shown(page):
    """The board did not replace the instrument view; it is what you open it
    from. Hidden rather than unmounted, so the price chart keeps its size."""
    script = _script(page)

    assert "body.board-view .instrument-only" in page
    assert "body.instrument-view .board-only" in page
    assert "viewSel.addEventListener('change'" in script
    assert "setView(viewSel.value);" in script
    assert "function openInstrument(row)" in script
    assert "setView('instrument');" in script


def test_a_tile_offers_refresh_for_itself_and_for_the_whole_board(page):
    """Explicit, both of them. A background poll that fetched from a venue on
    its own schedule would be a different thing from a page somebody refreshed,
    and only the second is honest about when it last spoke to the exchange."""
    script = _script(page)

    assert "function refreshOne(identity, button)" in script
    assert "function refreshAll()" in script
    assert "setInterval" not in script, "the board must not poll on a timer"
    assert "setTimeout" not in script


def test_a_row_the_board_could_not_answer_says_why_on_its_own_tile(page):
    """One instrument's permanent leading funding gap must not blank the rest,
    so a refusal is drawn as that tile's content rather than as a page error."""
    script = _script(page)

    assert "if (row.unavailable)" in script
    assert "refusal.textContent = row.unavailable.split('\\n')[0];" in script
    assert "refusal.title = row.unavailable;" in script


def test_a_tile_whose_frame_ends_behind_the_newest_candle_says_so(page):
    """A perp frame is bounded by its own stored funding span, so ``as of`` can
    sit a bar behind the newest stored candle. Silently, that is a board that
    looks current and is not; said out loud, it is a fact about settlements."""
    script = _script(page)

    assert "row.as_of === row.dataset_last_bar ? '' : 'lag'" in script
    assert "tileLine('newest stored bar', stamp(row.dataset_last_bar), 'lag')" in script


def test_a_tile_carries_the_provenance_a_figure_cannot_be_read_without(page):
    """M20 in miniature, sixteen times over: a state and a fill on a tile with
    no strategy version, exit mode or crowding flag beside them is exactly the
    number-without-context that moved a published figure."""
    script = _script(page)

    assert "prov.crowding_measured" in script
    assert "prov.exit_mode" in script
    assert "prov.warmup_bars" in script
    assert "prov.version" in script
    assert "prov.generated_at" in script


def test_the_market_filter_offers_storages_own_vocabulary(page):
    """Taken from ``models.MarketType``, so a fourth market type is not a filter
    the API answers 422 to -- and the identity model takes it from there too."""
    from typing import get_args

    from strategy_lab.api.models import IdentityQuery, MarketType
    from strategy_lab.browser.page import DEFAULT_MARKET_TYPE, MARKET_TYPES

    assert MARKET_TYPES == get_args(MarketType)
    assert MARKET_TYPES == get_args(IdentityQuery.model_fields["market_type"].annotation)
    assert bootstrap_config()["marketTypes"] == list(MARKET_TYPES)
    assert DEFAULT_MARKET_TYPE in MARKET_TYPES


def test_a_continuous_strategy_shows_a_level_on_its_tile_rather_than_no_fills(page):
    """Markers have no word for "62% long", so a tile that only knew how to draw
    a latest fill would render the continuous contract as "none"."""
    script = _script(page)

    assert "} else if (row.target !== null) {" in script
    assert "tileLine('target', fmt(row.target, 3))" in script


def test_the_deepest_candle_set_is_selected_rather_than_the_first(page):
    """Storage holds probe sets as small as 25 bars.

    Landing on one opens the tool on a warmup error about a dataset nobody
    chose, which reads as the tool being broken rather than as the frame being
    short.
    """
    assert "row.candles > deepest.candles" in page
    assert "datasetSel.value = deepest.id" in page


def test_one_datasets_refresh_failure_does_not_cancel_the_rest(page):
    """The board's whole claim is that one instrument cannot blank the others.

    ``/api/refresh`` reports a venue or database failure as a 502 by design, so
    a rejection here is the designed path rather than a surprise. Chained
    without a per-identity catch it rejected the whole batch: the datasets after
    it were never fetched, and the reload that would have drawn the ones that
    *did* fetch was skipped too.
    """
    script = _script(page)

    assert "failures.push(identity.symbol" in script
    # The count is reported as "N of M", so a partial batch is not read as a
    # whole one. Matched without the surrounding layout, which is not the claim.
    assert "identities.length - failures.length" in script
    # After the reload, which clears the banner.
    assert "if (failures.length) setError(failures.join(' · '));" in script


def test_a_superseded_board_is_aborted_rather_than_drained(page):
    """Every row is a full recompute and nothing else cancels one.

    The strategy switcher re-requests without a reload, so two switches while a
    board streams left three boards competing for one threadpool -- and the
    instrument view queues behind them. An abort is this function superseding
    itself, so it is not an error.
    """
    script = _script(page)

    assert "function abortBoard()" in script
    assert "fetch(url, { signal: signal })" in script
    assert "if (error && error.name === 'AbortError') return;" in script
    # Before the token, not after: `abortBoard` bumps the same counter, so a
    # token taken first is stale the moment it is taken and the board that was
    # just started draws nothing.
    body = script[script.index("function loadBoard()"):]
    assert body.index("abortBoard();") < body.index("var token = ++boardPending;")


def test_leaving_the_board_stops_it_rather_than_letting_it_finish(page):
    """Switching views does not start another board, so nothing else stops one.

    ``boardPending`` only moves when a board starts, so an in-flight stream's
    own guard stays satisfied after a switch: it would keep appending tiles to a
    hidden host and calling ``setStatus`` over the instrument view, while the
    server finished rows nobody will look at -- a full recompute each, now that
    none are held.
    """
    script = _script(page)

    body = script[script.index("function setView(name)"):]
    body = body[: body.index("function reload()")]
    assert "if (name !== 'board') {" in body
    assert "abortBoard();" in body


def test_opening_a_tile_carries_its_exact_edges_not_a_rounded_day(page):
    """A tile's frame ends at a settlement, which a ``type="date"`` cannot hold.

    A bare ``end`` means *all* of that day to the API, so the chart would read
    bars the tile excluded and disagree with the tile that opened it -- and once
    the lag is larger those bars can outrun the funding the tile was bounded by,
    refusing the frame outright. The dates stay human; the request carries the
    edges.
    """
    script = _script(page)

    assert "exactBounds = {" in script
    assert "start: row.provenance.first_bar," in script
    assert "end: row.provenance.last_bar" in script
    assert "if (exactBounds.start) params.set('start', exactBounds.start);" in script
    # Dropped by every route that changes which instrument is being asked
    # about, asserted by the route rather than by a count: a strategy or
    # exit-mode change deliberately *keeps* them, because a funding span is per
    # contract and the board gives every strategy on one dataset one window.
    for owner in (
        "datasetSel.addEventListener('change'",   # a different instrument
        "viewSel.addEventListener('change'",      # entered without a tile
    ):
        after = script[script.index(owner):]
        assert "exactBounds = null;" in after[: after.index("});")], owner
    # And editing a date, which makes the visible dates the truth again.
    assert "exactBounds = null;\n      load();" in script


def test_refreshing_the_instrument_drops_the_frame_the_tile_pinned(page):
    """A refresh moves the frame's right edge, so the tile's edges stop describing it.

    Kept, the reload would re-request the old ``end`` -- ``load_candles`` filters
    ``timestamp <= end`` -- and draw no new bars while the status reported the
    candles it had just taken. The board, recomputing against the new funding
    window, would then show a later bar than the chart it links to.
    """
    script = _script(page)

    body = script[script.index("el('refresh').addEventListener"):]
    body = body[: body.index("});")]
    assert "exactBounds = null;" in body
    assert body.index("exactBounds = null;") < body.index("return load();")


def test_a_refresh_that_outlives_the_board_view_does_not_restart_it(page):
    """``refreshOne`` and ``refreshAll`` resolve on their own schedule.

    Open a tile while one is in flight and the continuation would start a full
    board -- a whole-history recompute per row -- against a hidden host, then
    retitle the page over the chart the user is looking at. ``setView`` aborts
    the *stream*; it cannot abort a POST that has not resolved yet.
    """
    script = _script(page)

    assert "function reloadBoardIfShown()" in script
    assert "viewSel.value === 'board' ? loadBoard() : Promise.resolve();" in script
    # Both refresh paths go through it rather than calling loadBoard directly.
    for owner in ("function refreshOne(", "function refreshAll("):
        body = script[script.index(owner):]
        body = body[: body.index("\n  function ", 1)]
        assert "reloadBoardIfShown()" in body, owner
        assert "return loadBoard();" not in body, owner
