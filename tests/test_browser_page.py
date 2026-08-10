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


def _style(page: str) -> str:
    return page[page.index("<style>") : page.index("</style>")]


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
    """Arrows reach the chart from one place and only under one condition, so a
    continuous payload cannot put them there even if one arrived carrying
    markers. A view may *clear* them — the state view runs no book and does —
    but nothing may set them from anything else."""
    script = _script(page)
    calls = re.findall(r"\.setMarkers\((.*?)\);", script, flags=re.S)

    assert calls, "no setMarkers call at all; the regex stopped matching"
    setting = [call for call in calls if call.strip() != "[]"]
    assert len(setting) == 1, f"more than one place decides: {setting}"
    assert "primitive === 'markers'" in setting[0]
    assert ": []" in setting[0], "the other contract must be cleared, not left stale"


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


def test_a_pinned_crowding_is_flagged_by_the_strategy_rather_than_the_market(page):
    """A strategy running with ``crowding`` pinned to neutral is not the funded run
    the charter publishes -- 16.44% against 15.45% on R5's test half -- and which
    strategy that is cannot be read off the market type. The gate this replaced,
    ``perp && !crowding_measured``, marked ``donchian`` on a funded perp where
    nothing is wrong and stayed silent on an equity machine running a feature
    short.
    """
    script = _script(page)

    assert "var crowdingBlind = prov.reads_crowding && !prov.crowding_measured;" in script
    assert "crowdingBlind ? 'warn' : ''" in script
    assert "class = 'alert'" in script.replace("className", "class")
    assert "perp && !prov.crowding_measured" not in script


def test_a_board_tile_flags_a_pinned_crowding_on_the_same_predicate(page):
    """Both surfaces or neither: a tile silent about what the strip warns on is
    the board quietly contradicting the chart it links to (M36's failure mode)."""
    block = _within(page, "function tileFeatures(row)")

    assert "prov.reads_crowding && !prov.crowding_measured" in block
    assert "crowding pinned neutral" in block
    # The market type still picks the *wording* -- fetchable on a perp, permanent
    # off it -- which is the distinction the predicate itself must not make.
    assert "market_type === 'perp'" in block


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
    return _within(page, "function renderProvenance")


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
        "/api/datasets",
        "/api/refresh",
        "/api/state",
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
    assert "setInterval" not in script, "nothing may fetch on a clock"


def test_the_only_unclicked_fetch_is_the_one_a_closing_bar_asks_for(page):
    """The rule this page kept was "never on a timer", and the reason was
    honesty about when it last spoke to a venue -- not a ban on timers as such.
    A closing bar is not a clock: it is the venue naming the one moment the
    stored candles are provably behind, and the page still says when it last
    refreshed.

    So `setInterval` stays banned, and the single `setTimeout` here is the
    first-frame watchdog, which changes a label and fetches nothing.
    """
    script = _script(page)

    assert script.count("setTimeout") == 1
    after = script[script.index("live.watchdog = setTimeout") :]
    watchdog = after[: after.index("FIRST_TICK_GRACE_MS")]
    assert watchdog, "the watchdog body was not found"
    for fetching in ("getJSON", "fetch(", "/api/"):
        assert fetching not in watchdog, "the watchdog must not talk to anything"
    assert "refreshInstrument()" in _within(script, "function onTick(message)")


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


def test_each_tile_states_the_staleness_that_applies_to_its_own_market(page):
    """A perp's right edge can lag its newest candle by up to one settlement
    (M37); an equity's cannot lag at all, and what goes stale is its whole
    history. Neither line belongs on the other tile: stating a risk that does not
    exist for a market is the same kind of wrong as stating none.
    """
    script = _script(page)
    freshness = _within(script, "function tileFreshness(row, wrap)")
    restated, funded = freshness.split("return;", 1)

    # The restated branch says when the candles were written, and cannot say
    # anything about a right edge that has nothing to lag behind.
    assert "row.identity.market_type === CFG.restatedMarketType" in restated
    assert "writtenLine(row)" in restated
    assert "dataset_last_bar" not in restated

    # The funded branch is R10b's, unchanged, and carries no write time.
    assert "tileLine('newest stored bar', stamp(row.dataset_last_bar), 'lag')" in funded
    assert "written" not in funded

    assert "'candles written'" in _within(script, "function writtenLine(row)")


def test_the_restated_market_and_when_it_is_flagged_are_decided_in_python(page):
    """Which market has a restated history decides what a tile claims, and a
    hardcoded ``'equity'`` in JavaScript is a contract decision somewhere no test
    looks -- the same reason the chart primitives and the exit modes are tables
    here.
    """
    from strategy_lab.browser.page import RESTATED_MARKET_TYPE, RESTATEMENT_STALE_DAYS

    config = bootstrap_config()
    script = _script(page)

    assert config["restatedMarketType"] == RESTATED_MARKET_TYPE
    assert config["restatementStaleDays"] == RESTATEMENT_STALE_DAYS
    assert "'equity'" not in script, "the market type is retyped in the page script"
    # The direction too, not merely the name: ``<=`` here would flag every
    # freshly written history and nothing else, which is the same control
    # reading exactly backwards.
    assert "age >= CFG.restatementStaleDays" in _within(script, "function writtenLine(row)")


def test_a_write_stamp_is_parsed_rather_than_left_to_the_browser(page):
    """Postgres stamps carry a space where ISO 8601 wants a ``T``.

    ``Date`` parses that by implementation-defined luck, and an unparseable one
    yields ``NaN`` days, which compares false against every threshold and reads
    as fresh -- a stale history silently unflagged, which is the failure this
    line exists to prevent.
    """
    age = _within(_script(page), "function writtenAgeDays(written)")

    assert "written.replace(' ', 'T')" in age
    assert "isNaN(at) ? null" in age


def test_the_default_market_is_the_one_whose_tiles_carry_no_restatement_caveat(page):
    """``perp`` still, now that equities are selectable: opening on a market whose
    every tile carries "this history is restated on a dividend" is a choice a
    reader should make rather than inherit.
    """
    from strategy_lab.browser.page import DEFAULT_MARKET_TYPE, RESTATED_MARKET_TYPE

    assert DEFAULT_MARKET_TYPE != RESTATED_MARKET_TYPE
    assert bootstrap_config()["defaultMarketType"] == DEFAULT_MARKET_TYPE
    assert "marketSel.value = CFG.defaultMarketType;" in _script(page)


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
    # The cost model too, which CLAUDE.md's rule names beside the others and the
    # tile omitted -- and through `costText`, so a tile and the chart read one
    # the same way rather than through two formatters that can drift.
    tile = _within(script, "function tile(row)")
    assert "costText(prov.cost_model)" in tile


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
    # Shown after the reload, which clears the banner -- and through the
    # view-guarded writer, since a board failure says nothing about a chart.
    assert "if (failures.length) boardError(failures.join(' · '));" in script


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


def test_switching_views_stops_whichever_one_is_being_left(page):
    """Neither switch starts a request on the side it leaves, so neither guard
    fires on its own.

    Leaving the board keeps appending tiles to a hidden host and finishing a
    full recompute per dataset. Leaving the instrument view is the mirror: an
    ``/api/analysis`` still in flight resolves into the board, draws a chart
    nobody is looking at and retitles the page with the instrument's symbol.
    """
    script = _script(page)

    body = script[script.index("function setView(name)"):]
    body = body[: body.index("function reload()")]
    assert "abandonInstrument();" in body and "abortBoard();" in body
    # And the venue socket: a stream left open behind the board is a connection
    # nobody is watching, still redrawing a hidden chart on every tick.
    assert "closeSocket();" in body
    # The two guards key on different counters, so each branch may bump the one
    # it is not about to take a token from.
    assert "pending += 1;" in _within(script, "function abandonInstrument()")
    assert "boardPending += 1;" in _within(script, "function abortBoard()")


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
    assert "exactBounds = null;\n      reload();" in script


def test_refreshing_the_instrument_moves_only_the_edge_the_fetch_moved(page):
    """A refresh moves the frame's right edge and nothing else.

    Keeping the old ``end`` draws no new bars -- ``load_candles`` filters
    ``timestamp <= end`` -- while the status reports the candles just taken, and
    the board then shows a later bar than the chart it links to. Clearing *both*
    edges is the other failure: the head bound is what holds the frame off a
    permanent leading funding gap, and the date input still carries the old day,
    so the window widens to that day's 23:59:59 -- no newer bars, and up to a
    full day past the last settlement, which the guard tolerates only to one
    funding cadence.
    """
    script = _script(page)

    body = _within(script, "function refreshInstrument()")
    assert "exactBounds = { start: exactBounds && exactBounds.start, end: null };" in body
    assert "el('end').value = '';" in body
    assert body.index("end: null };") < body.index("return reload();")


def test_a_board_refresh_says_nothing_over_the_instrument_view(page):
    """A refresh chain outlives the view that started it.

    The reload is guarded; the writes before it were not, so each remaining link
    wrote its progress over the chart being read -- and a failure banner too,
    which dims the provenance strip of a view the failure says nothing about.
    """
    script = _script(page)

    # The writers are guarded, not merely named -- a helper that forwarded
    # unconditionally would satisfy the call-site check below and change nothing.
    for writer, wrapped in (("boardStatus", "setStatus"), ("boardError", "setBanner")):
        body = script[script.index("function " + writer + "(text)"):]
        body = body[: body.index("\n  }")]
        assert "if (onBoard()) " + wrapped + "(text)" in body, writer
    # `setBanner`, never `setError`: the latter also dims `#provenance` and
    # rewrites the status with "showing the last run that succeeded", which are
    # claims about the *chart*. A board failure makes neither, and nothing on a
    # board redraw would have undimmed it.
    assert "el('provenance').classList.toggle('stale'" not in _within(script, "function setBanner")
    # Progress is transient and may be dropped; a *failure* is held, because it
    # is a fact about the data rather than about the view. It has to survive a
    # trip to the chart *and* a board redraw the user asked for in between --
    # both of which otherwise wipe the banner, which is the same silence as
    # never showing it.
    held = _within(script, "function boardError(text)")
    assert "heldBoardError = text;" in held

    board = script[script.index("function loadBoard()"):]
    board = board[: board.index("\n  function ", 1)]
    assert "setBanner(heldBoardError || '');" in board
    assert "heldBoardError = null;" not in board, "a redraw does not unfail a fetch"

    # Cleared by the user retrying, which is the one event that makes it stale --
    # and the visible banner with it, or the previous failure stays up while a
    # new fetch is in flight and reads as this attempt's.
    cleared = _within(script, "function clearBoardError()")
    assert "heldBoardError = null;" in cleared
    assert "if (onBoard()) setBanner('');" in cleared
    for owner in ("function refreshOne(", "function refreshAll("):
        body = script[script.index(owner):]
        body = body[: body.index("\n  function ", 1)]
        assert "clearBoardError();" in body, owner
    for owner in ("function refreshOne(", "function refreshAll("):
        body = script[script.index(owner):]
        body = body[: body.index("\n  function ", 1)]
        assert "setStatus(" not in body, owner
        assert "setError(" not in body, owner


def test_a_refresh_that_outlives_the_board_view_does_not_restart_it(page):
    """``refreshOne`` and ``refreshAll`` resolve on their own schedule.

    Open a tile while one is in flight and the continuation would start a full
    board -- a whole-history recompute per row -- against a hidden host, then
    retitle the page over the chart the user is looking at. ``setView`` aborts
    the *stream*; it cannot abort a POST that has not resolved yet.
    """
    script = _script(page)

    body = script[script.index("function reloadBoardIfShown()"):]
    body = body[: body.index("\n  }")]
    assert "onBoard()" in body and "loadBoard()" in body
    assert "Promise.resolve()" in body
    # Both refresh paths go through it rather than calling loadBoard directly.
    for owner in ("function refreshOne(", "function refreshAll("):
        body = script[script.index(owner):]
        body = body[: body.index("\n  function ", 1)]
        assert "reloadBoardIfShown()" in body, owner
        assert "return loadBoard();" not in body, owner


def _within(script: str, marker: str) -> str:
    """One function body, from its declaration to the next one at file scope."""
    body = script[script.index(marker):]
    return body[: body.index("\n  }")]


def test_refresh_all_is_refused_while_the_board_is_still_arriving(page):
    """``boardIdentities`` fills as the rows land, and ``refreshAll`` snapshots it.

    Clicked mid-stream it fetches only the datasets already drawn, then reports
    "N of N" for a partial N -- and the reload after it recomputes the untouched
    ones from unchanged candles, so nothing on screen says a dataset was
    skipped. Disabled for the duration instead: a board you cannot see yet is
    not one you can refresh.
    """
    script = _script(page)

    board = script[script.index("function loadBoard()"):]
    board = board[: board.index("\n  function ", 1)]
    assert "el('refresh-all').disabled = true;" in board
    assert board.index("disabled = true;") < board.index("disabled = false;")

    # Re-enabled on the *success* path only, and after its own token guard, so
    # neither a superseded load nor a truncated one lifts the disable: a stream
    # that ended early leaves `boardIdentities` partial, and "refresh all" over
    # a partial list reports it as the whole batch. Each tile keeps its own
    # button, which is the escape hatch that stays honest.
    done = board[board.index("}, signal).then(function () {"):]
    done = done[: done.index("}).catch(")]
    assert "el('refresh-all').disabled = false;" in done
    assert done.index("if (token !== boardPending) return;") < done.index("disabled = false;")
    caught = board[board.index("}).catch("):]
    assert "disabled = false;" not in caught, "a truncated board must stay unrefreshable"


def test_the_state_sequence_is_derived_from_the_series_the_page_already_has(page):
    """A transition is a `!==` over `why.states`, computed in the page.

    Deriving it server-side would be M36's third answer: the board tile, the
    per-bar chip and the transition list would each have their own route to the
    same fact, free to drift. Here there is one series and two renderings of it.
    """
    script = _script(page)

    assert "shiftsFrom(payload.why.states, payload.bars)" in script
    assert "if (states[i] === states[i - 1]) continue;" in script


def test_a_strategy_with_no_state_machine_shows_no_sequence(page):
    """Eight of nine registered strategies compute no feature frame, so there is
    no state to sequence — an empty list would claim there were no changes."""
    script = _script(page)

    assert "payload.why ? shiftsFrom(payload.why.states, payload.bars) : null" in script
    render = script[script.index("function renderShifts()") : script.index("function pinBar")]
    assert "if (!view.shifts) {\n      section.hidden = true;" in render


def test_a_transition_inside_warmup_is_marked_rather_than_hidden(page):
    """Measured on BTC/USDT perp 4h over 2023-01-01 → 2024-06-01: 8 of 50
    changes fall inside `state_machine_v1`'s 2,192-bar warmup. The machine really
    walked through them, so hiding them would misreport the sequence — but the
    strategy was not acting yet, so showing them plain would read as decisions.
    """
    script = _script(page)

    assert "var early = warmup !== null && shift.index < warmup;" in script
    assert "'shift' + (early ? ' warmup' : '')" in script
    # The row, not a list of its children: enumerating them is what left
    # `.shift-from` undimmed, so a warmup row showed the state it left at full
    # strength and the state it moved into at half.
    assert ".shift.warmup { opacity: 0.55; }" in _style(page)
    # And the reader is told *how many*, not just that some are: a dimmed row
    # with no count is a style nobody can act on.
    assert "if (early) inWarmup += 1;" in script
    assert "el('transitions-note').textContent = inWarmup" in script
    # The note carries the *reason*, which is also the ribbon's: inside warmup
    # the machine's inputs are not measurable, it reads that as failing, and
    # failing renders as compression. Without it a reader has a dimmed row and a
    # ribbon that starts late, and nothing connecting them.
    assert "not measurable yet" in script
    assert "not readings of" in script or "not readings of the market" in script


def test_clicking_a_transition_recentres_only_when_the_bar_is_off_screen(page):
    """Otherwise comparing two rows yanks the chart on every click."""
    script = _script(page)

    assert "if (range && (i < range.from || i > range.to))" in script


def test_a_transition_row_is_a_button_rather_than_a_clickable_div(page):
    """It is an action, and everything else actionable on this page is a button.
    A `div` is unreachable by keyboard and announces nothing, so a row that pins
    a bar could only be used with a mouse."""
    script, style = _script(page), _style(page)
    render = script[script.index("function renderShifts()") : script.index("function pinBar")]

    assert "document.createElement('button')" in render
    assert "row.type = 'button';" in render
    assert "row.setAttribute('aria-pressed', String(selected));" in script
    assert ".shift:focus-visible" in style


def test_a_transition_carries_the_bar_its_dwell_is_measured_from(page):
    """`dwellText` reads `bars[shift.previousIndex]`, so a shift without it
    indexes `undefined` and the whole list fails to render. It is set where the
    shift is built rather than stitched on by the caller — assembling an object
    in two places is what made it read as missing."""
    script = _script(page)
    build = script[script.index("function shiftsFrom") : script.index("function dwellText")]

    assert "previousIndex: previousIndex," in build
    assert "previousIndex = i;" in build
    assert "dwellText(view.bars, shift.previousIndex, shift.index)" in script


def test_a_live_tick_is_drawn_but_never_analysed(page):
    """The one reading the whole event path refuses. A tick reaching
    `build_analysis` produces a state and a marker that flip when the bar
    closes — `include_forming=False` exists for this, and a timestamp is
    complete only once a later one arrives. So the candle moves and the
    why-layer stays where the last closed bar left it."""
    tick = _within(_script(page), "function onTick(message)")

    assert "candleSeries.update(live.bar);" in tick
    assert "build_analysis" not in tick and "/api/analysis" not in tick
    # And the gap that leaves is named rather than papered over: a live bar has
    # no index, and falling through would show the previous bar's state as if it
    # were this one's.
    assert "'live bar · not analysed until it closes'" in _script(page)


def test_a_closing_bar_is_what_triggers_the_refresh_not_a_timer(page):
    """`serve` polls every 60s and guesses. The stream says `x: true` on the
    one update where the stored candles are provably behind, so the refresh
    happens exactly then and the page can say when it last did."""
    script = _script(page)
    tick = _within(script, "function onTick(message)")

    assert "if (!k.x) return;" in tick
    assert "refreshInstrument()" in tick
    assert "setInterval" not in script, "a timer would be the thing this replaces"


def test_the_live_control_is_hidden_where_there_is_no_stream(page):
    """Most datasets have none — Yahoo publishes nothing, and `1wk` is a Yahoo
    timeframe. A control that cannot connect is worse than no control."""
    script = _script(page)

    assert "streams[id] = row.stream || null;" in script
    assert "view.stream = streams[datasetSel.value] || null;" in script
    assert "el('live').hidden = !view.stream;" in script


def test_the_page_composes_no_venue_url_of_its_own(page):
    """Which URL a stream lives at is venue knowledge, and it is wrong in ways a
    chart still renders. It comes from `market_data.streams`, tested in Python,
    and reaches the page as a field on the dataset row."""
    script = _script(page)

    assert "wss://" not in script
    assert "@kline_" not in script


def test_live_means_a_frame_arrived_not_that_a_socket_opened(page):
    """Measured against Binance from one network: `fstream.binance.com` — every
    perp dataset in this repo — accepts the connection and sends nothing, on
    both the raw and combined stream forms at 1m and 4h, while
    `stream.binance.com` delivers 6 frames in 12 s on the same pair.

    So `onopen` is a proxy for working, and reporting off it leaves a green dot
    over a frozen chart. Green comes from the first frame; silence gets said out
    loud.
    """
    socket = _within(_script(page), "function openSocket()")

    assert "socket.onopen" in socket
    # To the handler's own close, not to the next handler's start: slicing
    # between two markers goes empty if they are ever reordered, and this
    # assertion is a negative -- it would pass on nothing.
    after = socket[socket.index("socket.onopen") :]
    onopen = after[: after.index("};")]
    assert onopen, "the onopen body was not found"
    assert "setLive('on', 'live')" not in onopen
    assert "setLive('wait', 'waiting for data');" in socket
    assert "if (!live.ticks) setLive('on', 'live');" in socket
    assert "setLive('off', 'connected · no data');" in socket


def test_volume_is_an_overlay_rather_than_a_pane_of_its_own(page):
    """As a pane it took half the chart: `panes()[1].setHeight(80)` did not hold
    and the separator would not drag, leaving price squeezed into the top half of
    an already wide range."""
    script = _script(page)

    volume = script[script.index("var volumeSeries") : script.index("var stateSeries")]
    assert "priceScaleId: ''" in volume
    assert "scaleMargins: { top: 0.82, bottom: 0 }" in script


def test_the_state_ribbon_is_coloured_from_the_series_the_payload_carries(page):
    """The state was readable one bar at a time by hovering. As a ribbon the
    shape of a regime is visible at a glance, which is what a lifecycle strategy
    is about — and it is the same `why.states` the chip and the change list use,
    so the three cannot disagree."""
    script = _script(page)
    draw = _within(script, "function drawStateRibbon(payload)")

    assert "payload.why ? payload.why.states : null" in draw
    assert "CFG.stateColors[states[warmup + i]]" in draw
    # Constant height: the ribbon says which state, never how much of it.
    assert "value: 1," in draw
    # And nothing to colour means no ribbon and no legend, rather than a blank band.
    assert "stateSeries.setData([]);" in draw


def test_the_ribbon_starts_at_warmup_rather_than_at_bar_zero(page):
    """The lie this view exists to not tell.

    The machine answers on every bar. Inside warmup its inputs are NaN, it reads
    that as *failing*, and failing renders as COMPRESSION — measured on BTC/USDT
    spot 1d with `state_machine_v1`, 3,060 bars against a 2,192-bar warmup, the
    machine reports `compression` on 2,114 of the 2,192 warmup bars. Drawn from
    bar zero, 72% of that chart said "chop" over the range where the machine
    knew nothing, which is the one reading a regime chart must never give.
    """
    draw = _within(_script(page), "function drawStateRibbon(payload)")

    assert "payload.bars.slice(warmup)" in draw, "the ribbon still starts at bar zero"
    assert "var warmup = payload.provenance.warmup_bars || 0;" in draw
    # The slice re-bases the index and `states` is not sliced with it, so the
    # offset has to be added back or every band shows a state from 2,192 bars
    # earlier — which still renders, and still looks like a ribbon.
    assert "states[warmup + i]" in draw
    # And the blank stretch is named, or it reads as a rendering failure.
    assert 'id="warmup-swatch"' in page
    assert "before warmup" in page


def test_the_lifecycle_keeps_its_own_order_and_its_colours_come_from_python(page):
    """Alphabetical would describe a different thing from the one the machine
    walks: compression → breakout → confirmed → riding → exhaustion → reset."""
    from strategy_lab.browser.page import STATE_COLORS, STATE_DESCRIPTIONS
    from strategy_lab.state.machine import MarketState

    assert list(STATE_COLORS) == [state.value for state in MarketState]
    # A colour with no explanation is a band the reader cannot name. The module
    # refuses to import without the pair, so this pins both to the enum rather
    # than to each other -- otherwise adding a state to `MarketState` alone
    # passes.
    assert list(STATE_DESCRIPTIONS) == [state.value for state in MarketState]
    assert "#" not in _within(_script(page), "function drawStateRibbon(payload)")


def test_the_timeframe_rungs_sit_with_the_chart_not_with_the_query(page):
    """The header's fields change *what is asked*; the rungs change *what the
    chart is*, and a reader reaches for them where TradingView puts them."""
    markup = page[: page.index('<script id="config"')]
    charts = markup.index('<div class="charts')

    assert charts < markup.index('id="ladder"') < markup.index('id="price-pane"')
    assert 'id="ladder"' not in markup[: markup.index('<span class="range"')]


def test_the_state_view_offers_only_strategies_that_have_a_state(page):
    """`/api/state` refuses a strategy with no feature frame, and the refusal
    would read as a fault in the dataset. Filtered on the server's own
    `has_state`, not on a list of names kept here — the same reason the contract
    is published rather than inferred from the name."""
    script = _script(page)
    body = _within(script, "function syncStrategyChoices()")

    assert "row.has_state ? 'yes' : 'no'" in script
    assert "stateOnly && options[i].dataset.hasState !== 'yes'" in body
    # Landing on a hidden option is the same refusal by another route.
    assert "strategySel.selectedOptions[0].hidden" in body


def test_a_rung_too_shallow_for_its_strategy_says_so_instead_of_refusing(page):
    """Warmup is counted in *bars*, so one strategy is comfortable at 4h and
    permanently impossible at 1w on the same instrument: 2,192 bars is 365 days
    against 42 years. Measured — BTC/USDT spot 1w holds 438 bars, and no venue
    has the other 1,754."""
    ladder = _within(_script(page), "function renderLadder()")

    assert "depth <= warmup" in ladder
    assert "'shallow'" in ladder
    # Distinct from `absent`, which is one click from being fixed.
    assert ".ladder button.shallow" in _style(page)
    assert "'absent'" in ladder


def test_the_state_view_asks_for_no_parameter_that_implies_execution(page):
    """A cost model on a path that opened no book would describe one that never
    ran, and `/api/state` refuses unknown parameters by name — so an exit mode
    sent from here is a 422, not a quietly ignored field."""
    body = _within(_script(page), "function stateQuery()")

    for absent in ("exit_mode", "fees", "slippage", "cash", "position_pct"):
        assert absent not in body, f"the state view sent {absent}"
    assert "params.set('strategy', strategySel.value);" in body


def test_the_realised_total_counts_closed_trades_only(page):
    """vectorbt values an open trade against the last close, so its PnL moves on
    the next bar and on whatever range was asked for. Summing it into the net
    would report an unrealised number under a realised name — and the one on the
    page is the number most likely to be quoted."""
    body = _within(_script(page), "function renderTrades()")

    assert "trades.filter(function (t) { return t.status === 'closed'; })" in body
    for derived in ("won", "net"):
        assert re.search(rf"var {derived} = closed\.", body), (
            f"{derived} was computed from something other than the closed trades"
        )


def test_the_trade_table_is_absent_where_no_book_ran_rather_than_empty(page):
    """The continuous contract returns no trades because ``build_analysis`` runs
    no portfolio for it — not because the strategy sat out. An empty table under
    that chart would be a claim about the strategy instead of about the path."""
    body = _within(_script(page), "function renderTrades()")

    assert "var costs = view.payload.provenance.cost_model;" in body
    assert "if (!costs) {" in body


def test_no_two_functions_in_the_page_script_share_a_name(page):
    """A function declaration does not shadow an earlier one — it replaces it,
    silently, for every call site in the scope. The whole page is one IIFE, so a
    second `stamp` written for the trade table took over the board's, which then
    called it with an epoch and threw `text.slice is not a function`. Nothing
    warns; the collision is only visible when it runs."""
    names = re.findall(r"^  function (\w+)\(", _script(page), flags=re.MULTILINE)

    duplicated = {name for name in names if names.count(name) > 1}
    assert not duplicated, f"declared more than once in one scope: {sorted(duplicated)}"


def test_the_ladder_offers_no_month(page):
    """Binance publishes `1M` klines and `timeframe_to_millis` raises on it — a
    month is not a fixed width, and warmup, funding windows and the poll cadence
    are all `bar_ms` arithmetic. A rung that fetches candles and then cannot be
    analysed is the live control that cannot connect, again."""
    from strategy_lab.browser.page import TIMEFRAME_LADDERS
    from strategy_lab.timeframes import timeframe_to_millis

    for ladder in TIMEFRAME_LADDERS.values():
        assert "1M" not in ladder
        for timeframe in ladder:
            assert timeframe_to_millis(timeframe) > 0
    with pytest.raises(ValueError):
        timeframe_to_millis("1M")


def test_a_timeframe_switch_never_resamples_on_the_client(page):
    """A 1h bar built from four 15m bars is not the venue's 1h bar, and this repo
    keys a dataset on its timeframe precisely so the two cannot be confused. A
    rung with nothing stored fetches that timeframe instead."""
    switch = _within(_script(page), "function switchTimeframe(timeframe, stored)")

    assert "/api/refresh?" in switch
    assert "params.set('since'" in switch, "a new timeframe needs its own history"
    for resampling in ("resample", "aggregate", "reduce("):
        assert resampling not in switch


def test_the_ladder_carries_the_timeframes_this_instrument_already_has(page):
    """Otherwise leaving 15m for 1h strands 15m in the dropdown, because it is
    not one of the four rungs."""
    render = _within(_script(page), "function renderLadder()")

    assert "if (mine && rungs.indexOf(parts[3]) < 0) rungs.push(parts[3]);" in render
    assert "if (rungs.indexOf(current) < 0) rungs.push(current);" in render
    assert "CFG.timeframeOrder.indexOf(a) - CFG.timeframeOrder.indexOf(b)" in render


def test_a_frame_from_a_replaced_socket_reaches_nothing(page):
    """`closeSocket()` nulls the reference, but a message already in flight is
    still dispatched. Ungated it draws the old market's candle onto the new
    dataset's chart, and a stale `x: true` POSTs a refresh nobody asked for — to
    the app's only write."""
    socket = _within(_script(page), "function openSocket()")

    for handler in ("socket.onopen", "socket.onmessage"):
        after = socket[socket.index(handler) :]
        body = after[: after.index("};")]
        assert "if (live.socket !== socket) return;" in body, handler
    assert "if (live.socket === socket) setLive('off', 'stream error');" in socket


def test_a_bar_close_refresh_does_not_redraw_over_the_board(page):
    """This refresh is not always started by a click, so the view can change
    while it is in flight. `abandonInstrument` invalidates requests already
    running; it cannot stop one begun afterwards."""
    script = _script(page)

    tick = _within(script, "function onTick(message)")
    # Named as the view it must not draw over, rather than the one it may: a bar
    # closing should refresh whichever chart is up, and an allow-list of one
    # silently froze the state view's live pill the moment it was added.
    assert "if (viewSel.value === 'board') return;" in tick
    refresh = _within(script, "function refreshInstrument()")
    assert "var startedIn = viewSel.value;" in refresh
    assert "if (viewSel.value !== startedIn) return null;" in refresh


def test_the_ladder_spells_a_week_the_way_each_venue_does(page):
    """Yahoo's own error names its set — `[1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h,
    4h, 1d, 5d, 1wk, 1mo, 3mo]` — so `4h` is fine there and `1w` returns nothing.
    That is the `1w`/`1wk` split this repo already keys datasets on, and a rung
    that fetches nothing is the live control that cannot connect."""
    from strategy_lab.browser.page import TIMEFRAME_LADDERS

    assert "1w" in TIMEFRAME_LADDERS["binance"] and "1wk" not in TIMEFRAME_LADDERS["binance"]
    assert "1wk" in TIMEFRAME_LADDERS["yahoo"] and "1w" not in TIMEFRAME_LADDERS["yahoo"]
    assert "4h" in TIMEFRAME_LADDERS["yahoo"], "Yahoo does serve 4h — measured"
    assert "CFG.timeframeLadders[id.exchange] || CFG.defaultTimeframeLadder" in _script(page)
