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
    assert '<div class="provenance" id="provenance"></div>' in page
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
    """Read-only is a property of the page as well as of the API: every fetch it
    makes is a GET except the one POST that is ``server.refresh_candles``."""
    posts = re.findall(r"method: '([A-Z]+)'", _script(page))
    fetched = set(re.findall(r"getJSON\('(/[^'?]+)", _script(page)))

    assert posts == ["POST"]
    assert fetched == {"/api/analysis", "/api/refresh", "/api/datasets", "/api/strategies"}


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
    assert "strategySel.addEventListener('change', function () { syncExitEnabled(); load(); });" \
        in script


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


def test_the_deepest_candle_set_is_selected_rather_than_the_first(page):
    """Storage holds probe sets as small as 25 bars.

    Landing on one opens the tool on a warmup error about a dataset nobody
    chose, which reads as the tool being broken rather than as the frame being
    short.
    """
    assert "row.candles > deepest.candles" in page
    assert "datasetSel.value = deepest.id" in page
