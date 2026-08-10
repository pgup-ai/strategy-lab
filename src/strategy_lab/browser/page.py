"""The page: a board of instruments, one chart behind it, and why either says so.

Two views of the same API. The **board** is one tile per (dataset, strategy) --
state, latest fill, feature values, the bar it is as of, provenance and a
sparkline -- and the **instrument** view is the chart, the "why" panel and the
provenance strip for one pair. Neither is a summary of the other: the board is
``/api/board``'s rows drawn as tiles, and a tile's ``open`` control switches to
the instrument view for that exact pair, which re-asks ``/api/analysis``.

Nothing on this page is server-rendered from data: the switcher changes strategy
without a reload, so every figure arrives from ``/api/analysis`` and is drawn
client side. Measured end to end on the largest stored frame -- BTC/USDT perp 4h,
15,114 bars -- a switch is 650-750 ms, of which ~500 ms is server and network,
15 ms is JSON parse, and the rest is handing 15k candles to the chart. On a
438-bar weekly set it is 136 ms. Fluid rather than instant, and there is
deliberately no frame cache: caching a frame beside a refresh endpoint that
mutates it needs invalidation, and a stale frame is a worse bug than half a
second.

**The board is read as a stream, not as a response.** ``/api/board`` emits one
JSON row per line as each finishes, because a row is a whole-history recompute
and four threads over three frames measured 1.10x -- the cost is serial and
cannot be parallelised away. Reading the body with ``getReader()`` and drawing
each line as it lands is what turns a 1.2 s board into a 206 ms first paint; a
``response.json()`` here would wait for the slowest instrument before drawing
the fastest.

**Refresh is a button, per tile and for all of them, and never a timer.** A
background poll that fetches from a venue on its own schedule is a different
thing from a page a user refreshes, and only the second one is honest about
when it last spoke to the exchange -- which is what the provenance block on
every tile exists to say.

What Python contributes is the shell, the vendored chart asset, and the three
things that must not be re-decided in JavaScript:

* **which primitive draws which contract** (``CONTRACT_PRIMITIVES``). A
  ``SignalSet`` is a set of fills, which markers say exactly; a
  ``TargetExposure`` is a signed level in -1..1, which markers cannot express at
  all -- measured, ``state_machine_v2`` spans the full range, and a marker
  vocabulary has no word for "62% long". Hardcoding that in the page script
  would put a contract decision somewhere no test looks, so it is a table here,
  a registry member is refused if it has no entry, and
  ``tests/test_browser_page.py`` pins both directions.
* **the exit modes offered**, taken from the engine's own ``ExitMode`` rather
  than retyped, so a mode added to the engine appears here and a mode removed
  from it disappears.
* **which market type has a restated history, and when that becomes worth
  flagging** (``RESTATED_MARKET_TYPE``, ``RESTATEMENT_STALE_DAYS``). A perp tile
  and an equity tile go stale for different reasons and each states only its
  own: a perp's frame is bounded by its own stored funding, so what can lag is
  its **right edge**; an equity's runs to its newest bar and what can be stale
  is its **whole history**, because a dividend rescales every past bar.
* **the styling**, imported from ``backtests/report.py``: the frozen report and
  the live view are the same product and must not drift into looking like two.

The page is inert without the API behind it. It fetches, it draws, and it POSTs
to the one endpoint that writes -- it has no path of its own to ``reports/``,
``signals`` or ``market_candles``.
"""

from __future__ import annotations

import json
from typing import get_args

from strategy_lab.api.analysis import Contract
from strategy_lab.api.models import MarketType
from strategy_lab.backtests import ExitMode
from strategy_lab.backtests.report import (
    DOWN,
    DOWN_DIM,
    SHELL_CSS,
    UP,
    UP_DIM,
    chart_library_source,
)

# Which chart primitive expresses which contract. See the module docstring for
# why this is a table in Python rather than a branch in the page script.
CONTRACT_PRIMITIVES: dict[str, str] = {
    Contract.SIGNAL_SET.value: "markers",
    Contract.TARGET_EXPOSURE.value: "baseline",
}

_unmapped = {contract.value for contract in Contract} - set(CONTRACT_PRIMITIVES)
if _unmapped:
    # A third contract that reached the page unmapped would be drawn by
    # whichever branch happened to be last, which is how a level gets rendered
    # as a set of fills and nobody notices.
    raise RuntimeError(
        f"no chart primitive for contract(s) {sorted(_unmapped)}: the research "
        f"browser cannot draw a contract it has not been told how to draw"
    )

# ``models.AnalysisQuery`` is the authority: it refuses an exit mode on the
# continuous contract, where a target of 0.0 *is* the exit. This is here so the
# control greys out instead of a strategy switch answering 422, and
# ``tests/test_browser_page.py`` checks it against what the API actually accepts
# rather than against this list.
CONTRACTS_WITH_EXIT_MODE: tuple[str, ...] = (Contract.SIGNAL_SET.value,)

# The market types the board can be filtered to, from storage's own vocabulary
# rather than retyped: ``models.MarketType`` is the authority and a fourth value
# here would be a control whose only outcome is a 422.
MARKET_TYPES: tuple[str, ...] = get_args(MarketType)

# What the board opens on, and it stays ``perp`` now that equities are on it:
# opening on a market whose every tile carries a restatement caveat is a choice
# a reader should make rather than inherit. The filter is on the page rather
# than in the endpoint, so nothing is hidden from a caller who asks for the rest.
DEFAULT_MARKET_TYPE = "perp"

# The market type whose stored history is *restated* rather than extended. The
# Yahoo fetcher rescales every OHLC column by ``adj_close / close``, which is
# correct -- a dividend-adjusted series is the right input for a strategy that
# compounds -- so what a tile owes a reader is not a different number but the
# fact that the number can move backwards in time. Measured on the stored SPY
# weekly series against a fresh fetch: 333 of 333 bars moved, median 0.257%, and
# ``donchian`` differed on 3 of them where two ratio-based strategies differed
# on none.
RESTATED_MARKET_TYPE = "equity"

# How old a stored equity history has to be before "nothing has been restated
# since" stops being a safe assumption. Quarterly is the ordinary distribution
# cadence for a US equity or equity ETF and monthly is the shortest routine one,
# so a month is the point past which at least one ex-dividend date has plausibly
# passed for *something* in the universe. Deliberately not an attempt to detect a
# dividend: that means storing a second copy to diff against, and R10a's rule is
# to persist only what cannot be re-fetched. The flag says "old enough to be
# worth refreshing", never "this was restated".
RESTATEMENT_STALE_DAYS = 31

if RESTATED_MARKET_TYPE not in MARKET_TYPES:
    raise RuntimeError(
        f"the board flags {RESTATED_MARKET_TYPE!r} histories as restatable, which "
        f"storage does not key candles on: {sorted(MARKET_TYPES)}"
    )

if DEFAULT_MARKET_TYPE not in MARKET_TYPES:
    raise RuntimeError(
        f"the board opens on market type {DEFAULT_MARKET_TYPE!r}, which storage "
        f"does not key candles on: {sorted(MARKET_TYPES)}"
    )


# What a chart offers to switch between.
#
# **No month.** Binance publishes `1M` klines, but a month is not a fixed width
# and every bar calculation here assumes one -- `timeframe_to_millis("1M")`
# raises, and warmup, funding windows and the poll cadence are all `bar_ms`
# arithmetic. A rung that fetches candles and then cannot be analysed is the
# same defect as a live control that cannot connect: better absent than broken.
# Per venue, because a week is not spelled the same way twice. Yahoo's own
# error names its set -- `[1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 4h, 1d, 5d, 1wk,
# 1mo, 3mo]` -- so `4h` is fine there and `1w` returns nothing, which is the
# `1w`/`1wk` split this repo already keys datasets on. A rung that fetches
# nothing is the live control that cannot connect, again.
TIMEFRAME_LADDERS: dict[str, tuple[str, ...]] = {
    "binance": ("1h", "4h", "1d", "1w"),
    "yahoo": ("1h", "4h", "1d", "1wk"),
}
DEFAULT_TIMEFRAME_LADDER: tuple[str, ...] = ("1h", "4h", "1d")

# Shortest first, for ordering rungs the ladder did not name. A timeframe this
# instrument already has is worth a rung even when it is not one of the four
# above -- otherwise leaving 15m for 1h strands you in the dropdown.
TIMEFRAME_ORDER: tuple[str, ...] = (
    "1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h",
    "6h", "8h", "12h", "1d", "3d", "1w", "1wk",
)

# The lifecycle in order, with a colour per state. In Python because the order
# is the machine's -- compression -> breakout -> confirmed -> riding ->
# exhaustion -> reset -- and a legend that listed them alphabetically would
# describe a different thing from the one the strategy walks.
STATE_COLORS: dict[str, str] = {
    "compression": "#4a5163",
    "breakout": "#d1a54a",
    "confirmed": "#26a69a",
    "riding": "#1de9b6",
    "exhaustion": "#ef5350",
    "reset": "#7e6bab",
}

# What each state means, for the legend. The percentages are
# ``state.policy.STATE_TARGET_RISK`` -- the whole of what a state *does*, since
# the policy maps state to a target risk and nothing else -- and they are quoted
# rather than imported because the browser shows the machine's own vocabulary,
# not one strategy's tuning of it.
#
# ``riding`` carries R6's finding rather than the row's intent, because that is
# the sentence a reader of this chart most needs: the machine spends real time
# there, the ribbon shows it, and no entry has ever been sized by it.
STATE_DESCRIPTIONS: dict[str, str] = {
    "compression": (
        "Quiet. The entry gate has not opened, so the policy holds nothing (0% risk). "
        "This is the state the thesis is about: chop, waited out."
    ),
    "breakout": (
        "The gate opened — enough directional lean, energy under its ceiling. "
        "First and smallest position (35% risk)."
    ),
    "confirmed": (
        "The move survived the breakout's minimum dwell. Size steps up (70% risk)."
    ),
    "riding": (
        "The trend is running: the row sized highest (100% risk). It is only ever "
        "reached with a position already open, and an entry needs a change of side "
        "— so no entry has ever been sized here, and a position that reaches it "
        "keeps whatever size its entry bar carried."
    ),
    "exhaustion": (
        "The advance stopped. Sized down (55% risk), and served out a dwell before "
        "the machine will reset."
    ),
    "reset": (
        "A finished or failed move, flat (0% risk). The cooldown is served here so "
        "the machine cannot re-enter on the next bar of the same chop."
    ),
}

if set(STATE_DESCRIPTIONS) != set(STATE_COLORS):
    raise RuntimeError(
        "every state the ribbon colours must also be explained: "
        f"{sorted(set(STATE_COLORS) ^ set(STATE_DESCRIPTIONS))}"
    )


def bootstrap_config() -> dict[str, object]:
    """Everything the page script must not decide for itself, as one JSON blob."""
    return {
        "primitives": CONTRACT_PRIMITIVES,
        "exitModeContracts": list(CONTRACTS_WITH_EXIT_MODE),
        "exitModes": [mode.value for mode in ExitMode],
        "marketTypes": list(MARKET_TYPES),
        "defaultMarketType": DEFAULT_MARKET_TYPE,
        "restatedMarketType": RESTATED_MARKET_TYPE,
        "restatementStaleDays": RESTATEMENT_STALE_DAYS,
        "colors": {"up": UP, "down": DOWN, "upDim": UP_DIM, "downDim": DOWN_DIM},
        "stateColors": STATE_COLORS,
        "stateDescriptions": STATE_DESCRIPTIONS,
        # The rungs a chart offers, in order. Not every one is stored for every
        # instrument -- the switcher fetches the missing one rather than hiding
        # it, because "this timeframe does not exist here" is a fact about
        # storage, not about the market.
        "timeframeLadders": {k: list(v) for k, v in TIMEFRAME_LADDERS.items()},
        "defaultTimeframeLadder": list(DEFAULT_TIMEFRAME_LADDER),
        "timeframeOrder": list(TIMEFRAME_ORDER),
    }


def render_browser_html() -> str:
    """The whole page, self-contained apart from the API it talks to."""
    config = json.dumps(bootstrap_config(), separators=(",", ":")).replace("</", "<\\/")
    return (
        _TEMPLATE.replace("__SHELL_CSS__", SHELL_CSS)
        .replace("__CONFIG__", config)
        .replace("__LIB__", chart_library_source())
    )


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>research browser · strategy-lab</title>
<style>
__SHELL_CSS__
  /* The one token the report spells inline (its .warn, its open-status pill) and
     this page needs by name, because provenance flags it in several places. */
  :root { --flag: #ff9800; }
  header { padding-bottom: 12px; }
  .controls { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  select, button, input {
    background: var(--panel); color: var(--ink); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 9px; font: inherit; font-size: 12px;
  }
  select:focus, button:focus, input:focus { outline: 1px solid #4a5163; }
  button { cursor: pointer; }
  button:disabled { color: var(--ink-dim); cursor: default; }
  input::-webkit-calendar-picker-indicator { filter: invert(0.7); }
  .banner {
    color: var(--down); font-size: 11.5px; line-height: 1.5; padding: 9px 20px;
    border-bottom: 1px solid var(--border); background: #241416;
  }
  .banner[hidden] { display: none; }
  .field { display: flex; flex-direction: column; gap: 3px; }
  .field label {
    color: var(--ink-dim); font-size: 9.5px; text-transform: uppercase;
    letter-spacing: 0.7px;
  }
  .provenance {
    display: flex; gap: 8px; flex-wrap: wrap; align-items: stretch;
    padding: 10px 20px; border-bottom: 1px solid var(--border); background: #171a24;
  }
  .provenance .chip { min-width: 0; padding: 5px 10px; }
  .provenance .chip-value { font-size: 12.5px; font-weight: 600; }
  .provenance .chip.warn { border-color: #7a5418; background: #241d10; }
  .provenance .chip.warn .chip-value { color: var(--flag); }
  .provenance.stale { opacity: 0.45; }
  .alert {
    flex-basis: 100%; color: var(--flag); font-size: 11.5px; line-height: 1.5;
    max-width: 96ch;
  }
  .charts { padding: 12px 20px 0; }
  #price-pane { height: 440px; }
  #exposure-pane { height: 150px; margin-top: 10px; }
  #exposure-wrap[hidden] { display: none; }
  .pane {
    position: relative; border: 1px solid var(--border-soft); border-radius: 8px;
    overflow: hidden; background: var(--bg);
  }
  .pane-tag {
    position: absolute; top: 8px; left: 10px; z-index: 3; color: var(--ink-dim);
    font-size: 11px; letter-spacing: 0.4px;
  }
  #legend {
    position: absolute; top: 24px; left: 10px; z-index: 3; font-size: 12px;
    color: var(--ink-dim); display: flex; gap: 10px; flex-wrap: wrap;
  }
  #legend b { color: var(--ink); font-weight: 600; }
  h2 {
    font-size: 13px; color: var(--ink-dim); text-transform: uppercase;
    letter-spacing: 0.8px; margin: 0 0 10px;
  }
  h2 .pin { text-transform: none; letter-spacing: 0; color: var(--ink); }
  section.why { padding: 16px 20px 8px; }
  #transitions { display: flex; flex-direction: column; gap: 2px; max-height: 320px;
    overflow-y: auto; }
  .shift { display: grid; grid-template-columns: 148px 1fr 110px; gap: 10px;
    width: 100%; align-items: baseline; padding: 5px 8px; border-radius: 4px;
    cursor: pointer; border: 1px solid transparent; background: transparent;
    color: inherit; font: inherit; font-size: 13px; text-align: left; }
  .shift:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
  .shift:hover { background: #232733; }
  .shift.selected { border-color: #4a5163; background: #232733; }
  .shift-when { color: var(--ink-dim); font-variant-numeric: tabular-nums; }
  .shift-to { font-weight: 600; }
  .shift-from { color: var(--ink-dim); }
  .shift-dwell { color: var(--ink-dim); text-align: right;
    font-variant-numeric: tabular-nums; }
  /* Dimmed rather than hidden: the machine really walked through these, but the
     strategy was not acting yet, so they are not decisions. */
  .shift.warmup { opacity: 0.55; }
  .shift.warmup .shift-to { font-weight: 400; }
  /* Above the price pane rather than in the header: the rungs change what the
     chart is, where the header's fields change what is being asked. */
  .chart-bar { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
  .chart-bar-right { margin-left: auto; display: inline-flex; gap: 6px;
    align-items: center; }
  .ladder { display: inline-flex; gap: 2px; }
  .ladder button { padding: 4px 9px; font-size: 12px; }
  .ladder button.current { border-color: var(--accent); color: var(--ink); }
  .ladder button.absent { color: var(--ink-dim); border-style: dashed; }
  /* Distinct from `absent`: that one is a fetch away, this one may never be.
     Struck through because clicking it cannot help. */
  .ladder button.shallow { color: var(--ink-dim); opacity: 0.5;
    text-decoration: line-through; }
  .legend-strip { display: flex; flex-wrap: wrap; gap: 8px 18px; margin: 6px 20px 0;
    font-size: 12px; color: var(--ink-dim); align-items: center; }
  /* `hidden` is a UA rule that any author `display` defeats. */
  .legend-strip[hidden] { display: none; }
  .legend-strip .mk { font-size: 14px; }
  .legend-strip .mk.up { color: var(--up); }
  .legend-strip .mk.down { color: var(--down); }
  .swatch { display: inline-flex; align-items: center; gap: 4px; margin-left: 8px; }
  .swatch i { width: 9px; height: 9px; border-radius: 2px; display: inline-block;
    flex: none; }
  /* No fill, because the ribbon draws nothing there. */
  .swatch i.hollow { background: transparent; border: 1px dashed #4a5163; }
  button.swatch { padding: 1px 5px 1px 3px; border: 1px solid transparent;
    background: transparent; color: inherit; font: inherit; font-size: 12px;
    cursor: help; border-radius: 4px; }
  button.swatch:hover, button.swatch:focus-visible { border-color: #4a5163;
    background: #232733; color: var(--ink); }
  /* Block, not the strip's flex: the swatch and its sentence are one run of
     text, and as flex items the dash wraps onto a line of its own. */
  .state-explainer { display: block; max-width: 104ch; line-height: 1.6; }
  .state-explainer i { width: 9px; height: 9px; border-radius: 2px;
    display: inline-block; margin-right: 6px; }
  .state-explainer b { color: var(--ink); font-weight: 600; }
  #trades { max-height: 320px; overflow-y: auto; }
  #trades table { border-collapse: collapse; width: 100%; font-size: 13px; }
  #trades th {
    position: sticky; top: 0; background: var(--bg); z-index: 1; text-align: right;
    color: var(--ink-dim); font-weight: 500; font-size: 11px; letter-spacing: 0.4px;
    text-transform: uppercase; padding: 4px 8px; border-bottom: 1px solid var(--border-soft);
  }
  #trades td { padding: 4px 8px; font-variant-numeric: tabular-nums; text-align: right; }
  #trades th:first-child, #trades td:first-child { text-align: left; }
  #trades tbody tr { cursor: pointer; }
  #trades tbody tr:hover { background: #232733; }
  #trades .side-long { color: var(--up); }
  #trades .side-short { color: var(--down); }
  #trades .gain { color: var(--up); }
  #trades .loss { color: var(--down); }
  /* An open trade's PnL is a mark against the last close, not money -- it must
     not read like the closed rows it sits beside. */
  #trades tr.open td { opacity: 0.6; font-style: italic; }
  .live { display: inline-flex; align-items: center; gap: 6px; }
  .live-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--ink-dim); }
  .live.on .live-dot { background: var(--up); }
  .live.wait .live-dot { background: #d1a54a; }
  .why-chips { display: flex; gap: 8px; flex-wrap: wrap; }
  .why-chips .chip { min-width: 84px; }
  .why-chips .chip.state { border-color: #4a5163; background: #232733; }
  .why-chips .chip.absent .chip-value { color: var(--ink-dim); }
  .note { color: var(--ink-dim); font-size: 11.5px; margin: 0; max-width: 82ch; }
  #status { color: var(--ink-dim); font-size: 11.5px; }
  footer { padding: 14px 20px 20px; color: var(--ink-dim); font-size: 11px; }
  /* One page, three views. Whichever is not showing is display:none rather than
     unmounted: the price chart keeps its size and its zoom across a switch. */
  body.board-view .instrument-only, body.instrument-view .board-only { display: none; }
  /* `instrument-only` means "a chart view", which the state view also is. What
     separates them is `analysis-only`: the parts that describe a book. */
  body.state-view .board-only, body.state-view .analysis-only { display: none; }
  #board {
    display: grid; gap: 12px; padding: 14px 20px 6px;
    grid-template-columns: repeat(auto-fill, minmax(330px, 1fr));
  }
  .tile {
    border: 1px solid var(--border-soft); border-radius: 8px; background: var(--panel);
    padding: 11px 13px 12px; display: flex; flex-direction: column; gap: 8px;
  }
  .tile.refused { border-color: #7a5418; }
  .tile-head { display: flex; align-items: baseline; gap: 8px; }
  .tile-symbol { font-size: 14px; font-weight: 700; }
  .tile-where { color: var(--ink-dim); font-size: 11px; }
  .tile-actions { margin-left: auto; display: flex; gap: 6px; }
  .tile-actions button { padding: 3px 8px; font-size: 11px; }
  .state-pill {
    align-self: flex-start; border: 1px solid #4a5163; background: #232733;
    border-radius: 999px; padding: 2px 10px; font-size: 11.5px; letter-spacing: 0.4px;
  }
  .spark { display: block; width: 100%; height: 42px; }
  .tile-line { font-size: 11.5px; color: var(--ink-dim); }
  .tile-line b { color: var(--ink); font-weight: 600; }
  .tile-line.lag { color: var(--flag); }
  .tile-line.refusal { color: var(--flag); line-height: 1.5; }
  .tile-features { display: flex; gap: 5px; flex-wrap: wrap; font-size: 11px; }
  .tile-features span {
    border: 1px solid var(--border-soft); border-radius: 4px; padding: 1px 6px;
    color: var(--ink-dim);
  }
  .tile-features span b { color: var(--ink); font-weight: 600; }
  .tile-features span.warn { border-color: #7a5418; color: var(--flag); }
  .board-note { padding: 4px 20px 0; }
</style>
</head>
<body class="board-view">
<header>
  <h1 id="title">strategy-lab</h1>
  <div class="controls">
    <div class="field">
      <label for="view">View</label>
      <select id="view">
        <option value="board">board</option>
        <option value="instrument">instrument</option>
        <option value="state">state</option>
      </select>
    </div>
    <div class="field board-only">
      <label for="market">Market</label>
      <select id="market"></select>
    </div>
    <div class="field instrument-only">
      <label for="dataset">Candle set</label>
      <select id="dataset"></select>
    </div>
    <div class="field">
      <label for="strategy">Strategy</label>
      <select id="strategy"></select>
    </div>
    <div class="field analysis-only">
      <label for="exit-mode">Exit mode</label>
      <select id="exit-mode"></select>
    </div>
    <div class="field instrument-only">
      <label for="start">From</label>
      <input id="start" type="date">
    </div>
    <div class="field instrument-only">
      <label for="end">To</label>
      <input id="end" type="date">
    </div>
    <div class="field board-only">
      <label for="refresh-all">Candles</label>
      <button id="refresh-all" type="button"
              title="Fetch the newest bars for every dataset on the board, then recompute">
        refresh all
      </button>
    </div>
  </div>
  <span class="range" id="status"></span>
</header>
<p class="banner" id="error" hidden></p>
<div class="board-only">
  <div id="board"></div>
  <p class="note board-note">
    One row per candle set and strategy, each sliced from the same
    <code>/api/analysis</code> computation the instrument view draws — a board
    that answered by a cheaper route would be a third path free to disagree with
    the other two. Perp frames are bounded by their own stored funding span, so
    <b>as of</b> can sit a bar behind the newest stored candle; refresh advances
    candles and settlements together. Equity frames run to their newest bar and
    settle nothing, so they show <b>candles written</b> instead: that history is
    dividend-adjusted, which restates past bars rather than appending to them.
    Nothing here is on a timer.
  </p>
</div>
<div class="provenance instrument-only" id="provenance"></div>
<div class="charts instrument-only">
  <div class="chart-bar">
    <span id="ladder" class="ladder" role="group" aria-label="Candle timeframe"></span>
    <span class="chart-bar-right">
      <button id="live" type="button" class="live off" hidden
              title="Draw the venue's forming candle, and recompute when it closes">
        <span class="live-dot"></span><span id="live-text">live</span>
      </button>
      <button id="refresh" type="button" title="Fetch the newest bars and recompute">
        refresh
      </button>
    </span>
  </div>
  <div class="pane" id="price-pane">
    <span class="pane-tag">PRICE · VOLUME</span>
    <div id="legend"></div>
  </div>
  <div id="exposure-wrap" hidden>
    <div class="pane" id="exposure-pane">
      <span class="pane-tag">TARGET EXPOSURE &minus;1 … +1</span>
    </div>
  </div>
</div>
<p class="legend-strip instrument-only" id="chart-legend">
  <span class="analysis-only"><b class="mk up">&uarr;</b> bought &mdash; opening a long or closing a short</span>
  <span class="analysis-only"><b class="mk down">&darr;</b> sold &mdash; opening a short or closing a long</span>
  <span id="state-legend" hidden>state ribbon: <span id="state-swatches"></span>
    <span class="swatch" id="warmup-swatch"><i class="hollow"></i>before warmup</span></span>
</p>
<p class="legend-strip instrument-only state-explainer" id="state-explainer" hidden></p>
<section class="why instrument-only analysis-only" id="trades-section" hidden>
  <h2>Trades <span class="pin" id="trades-summary"></span></h2>
  <div id="trades"></div>
  <p class="note" id="trades-note"></p>
</section>
<section class="why instrument-only">
  <h2>Why this bar <span class="pin" id="why-bar"></span></h2>
  <div class="why-chips" id="why-chips"></div>
  <p class="note" id="why-note"></p>
</section>
<section class="why instrument-only" id="transitions-section" hidden>
  <h2>State changes <span class="pin" id="transitions-count"></span></h2>
  <div id="transitions"></div>
  <p class="note" id="transitions-note"></p>
</section>
<footer>
  strategy-lab research browser · a live view, recomputed per request · the
  frozen record is the per-run report under <code>reports/</code> ·
  charts by TradingView Lightweight Charts&trade;
</footer>
<script id="config" type="application/json">__CONFIG__</script>
<script>__LIB__</script>
<script>
(function () {
  var CFG = JSON.parse(document.getElementById('config').textContent);
  var LWC = LightweightCharts;
  var COLORS = CFG.colors;
  var ENGINE_DEFAULT = '';

  var el = function (id) { return document.getElementById(id); };
  var viewSel = el('view');
  var marketSel = el('market');
  var datasetSel = el('dataset');
  var strategySel = el('strategy');
  var exitSel = el('exit-mode');
  var statusEl = el('status');

  var theme = {
    autoSize: true,
    layout: {
      background: { type: 'solid', color: '#131722' },
      textColor: '#d1d4dc',
      fontSize: 11,
      attributionLogo: false
    },
    grid: { vertLines: { color: '#1e222d' }, horzLines: { color: '#1e222d' } },
    crosshair: {
      mode: 0,
      vertLine: { color: '#787b86', labelBackgroundColor: '#2a2e39' },
      horzLine: { color: '#787b86', labelBackgroundColor: '#2a2e39' }
    },
    rightPriceScale: { borderColor: '#2a2e39' },
    timeScale: { borderColor: '#2a2e39', timeVisible: true, secondsVisible: false }
  };

  var priceChart = LWC.createChart(el('price-pane'), theme);
  var candleSeries = priceChart.addSeries(LWC.CandlestickSeries, {
    upColor: COLORS.up, downColor: COLORS.down, borderVisible: false,
    wickUpColor: COLORS.up, wickDownColor: COLORS.down
  });
  candleSeries.priceScale().applyOptions({ scaleMargins: { top: 0.08, bottom: 0.06 } });
  // An overlay on the price pane rather than a pane of its own. As a pane it
  // took half the chart -- `panes()[1].setHeight(80)` did not hold, and the
  // separator would not drag -- which left the price series squeezed into the
  // top half of an already wide range. Overlaid, price keeps the whole pane and
  // volume lives in the bottom fifth of it, which is what TradingView does.
  var volumeSeries = priceChart.addSeries(LWC.HistogramSeries, {
    priceFormat: { type: 'volume' }, lastValueVisible: false, priceLineVisible: false,
    priceScaleId: ''
  });
  volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });

  // One bar per candle at constant height, coloured by the state the machine was
  // in. The state was already in the payload and readable only one bar at a time
  // by hovering; as a ribbon the shape of a regime is visible at a glance, which
  // is the thing a lifecycle strategy is actually about.
  var stateSeries = priceChart.addSeries(LWC.HistogramSeries, {
    priceFormat: { type: 'volume' }, lastValueVisible: false, priceLineVisible: false,
    priceScaleId: 'state'
  }, 1);
  priceChart.panes()[1].setStretchFactor(12);
  priceChart.panes()[0].setStretchFactor(88);
  var markerLayer = LWC.createSeriesMarkers(candleSeries, []);

  var exposureChart = LWC.createChart(el('exposure-pane'), theme);
  // Pinned to the contract's own bounds rather than autoscaled: a target that
  // never leaves +-0.3 should look small, and an autoscaled pane would draw it
  // exactly like one that saturates.
  var exposureSeries = exposureChart.addSeries(LWC.BaselineSeries, {
    baseValue: { type: 'price', price: 0 },
    topLineColor: COLORS.up, topFillColor1: 'rgba(38, 166, 154, 0.35)',
    topFillColor2: 'rgba(38, 166, 154, 0.04)',
    bottomLineColor: COLORS.down, bottomFillColor1: 'rgba(239, 83, 80, 0.04)',
    bottomFillColor2: 'rgba(239, 83, 80, 0.35)',
    lineWidth: 2,
    autoscaleInfoProvider: function () {
      return { priceRange: { minValue: -1, maxValue: 1 } };
    }
  });

  var syncing = false;
  function linkRange(from, to) {
    from.timeScale().subscribeVisibleLogicalRangeChange(function (range) {
      if (syncing || !range) return;
      syncing = true;
      to.timeScale().setVisibleLogicalRange(range);
      syncing = false;
    });
  }
  linkRange(priceChart, exposureChart);
  linkRange(exposureChart, priceChart);

  var view = {
    bars: [], index: {}, fills: {}, why: null, payload: null,
    pinned: null, dataset: null,
    // What a refresh fetched, tagged with the view that asked. One shared slot
    // untagged meant a tile's result could be read out by the instrument view a
    // user had since switched to, which is a status line describing something
    // that did not happen here.
    refreshed: null
  };

  function claimRefreshed(name) {
    var held = view.refreshed;
    view.refreshed = null;
    return held && held.view === name ? ' · ' + held.text : '';
  }

  function refreshText(result) {
    var parts = [result.candles_upserted + ' candles'];
    // null is "not a perp, nothing sought" and says nothing worth showing; 0 on
    // a perp is the drift itself and has to be said out loud.
    if (result.funding_upserted !== null && result.funding_upserted !== undefined) {
      parts.push(result.funding_upserted + ' settlements');
    }
    return 'refreshed ' + parts.join(' · ');
  }

  function setStatus(text) {
    statusEl.textContent = text;
  }

  function setBanner(message) {
    // The banner alone. Shared by both views, and says nothing about either.
    el('error').textContent = message || '';
    el('error').hidden = !message;
  }

  function setError(message) {
    // A failed request leaves the previous chart up, which is the right thing to
    // do -- and makes the controls describe something other than what is drawn.
    // The provenance strip still tells the truth, so it is dimmed rather than
    // cleared: what you are looking at, and that it is not what you asked for.
    //
    // Both of those are claims about the *chart*, which is why a board failure
    // goes through `setBanner` instead: dimming a provenance strip a board
    // refresh says nothing about left it dimmed until the next instrument load,
    // asserting the chart was stale when nothing about it had changed.
    setBanner(message);
    el('provenance').classList.toggle('stale', Boolean(message));
    if (message) setStatus('showing the last run that succeeded');
  }

  // A bar's own time, to the minute. Bars are UTC everywhere in this lab, so
  // rendering one in the reader's zone would put a chart, a state change and a
  // trade on three different clocks.
  function utcMinute(seconds) {
    return new Date(seconds * 1000).toISOString().slice(0, 16).replace('T', ' ');
  }

  function fmt(value, digits) {
    if (value === null || value === undefined) return '—';
    return Number(value).toLocaleString(undefined, {
      minimumFractionDigits: digits, maximumFractionDigits: digits
    });
  }

  function chip(label, value, cls) {
    var wrap = document.createElement('div');
    wrap.className = 'chip' + (cls ? ' ' + cls : '');
    var name = document.createElement('span');
    name.className = 'chip-label';
    name.textContent = label;
    var text = document.createElement('span');
    text.className = 'chip-value';
    text.textContent = value;
    wrap.appendChild(name);
    wrap.appendChild(text);
    return wrap;
  }

  // ---------------------------------------------------------------- controls

  function option(value, label) {
    var opt = document.createElement('option');
    opt.value = value;
    opt.textContent = label;
    return opt;
  }

  function fillMarkets() {
    // "every market" is a real answer here rather than a missing filter -- the
    // board is one row per stored candle set, and storage holds three kinds.
    marketSel.appendChild(option('', 'every market'));
    CFG.marketTypes.forEach(function (name) {
      marketSel.appendChild(option(name, name));
    });
    marketSel.value = CFG.defaultMarketType;
  }

  function fillExitModes() {
    // "engine default" rather than this page's guess at one: which mode a run
    // used is then read off the provenance strip, where it is a measured fact.
    exitSel.appendChild(option(ENGINE_DEFAULT, 'engine default'));
    CFG.exitModes.forEach(function (mode) {
      exitSel.appendChild(option(mode, mode));
    });
  }

  var streams = {};
  // How many bars each stored set holds, for the ladder's reach check.
  var depths = {};

  function fillDatasets(rows) {
    var deepest = null;
    rows.forEach(function (row) {
      var id = datasetKey(row);
      streams[id] = row.stream || null;
      depths[id] = row.candles;
      datasetSel.appendChild(option(
        id,
        row.symbol + ' · ' + row.timeframe + ' · ' + row.exchange + '/' +
        row.market_type + ' · ' + row.candles.toLocaleString() + ' bars'
      ));
      if (!deepest || row.candles > deepest.candles) deepest = { id: id, candles: row.candles };
    });
    // The deepest set rather than whichever the API listed first. Storage holds
    // small probe sets -- a 25-bar 1h series among them -- and landing on one
    // opens the tool on "declares 4000 warmup bars but the frame has 25", which
    // is a true error about a set nobody chose and reads as the tool being
    // broken. Depth is the only ordering that cannot pick a frame too short for
    // the strategy beside it.
    if (deepest) datasetSel.value = deepest.id;
  }

  function fillStrategies(rows) {
    rows.forEach(function (row) {
      var opt = option(row.name, row.name + '  (' + row.contract + ')');
      opt.dataset.contract = row.contract;
      opt.dataset.hasState = row.has_state ? 'yes' : 'no';
      opt.dataset.warmupBars = String(row.warmup_bars);
      strategySel.appendChild(opt);
    });
  }

  // Which strategies the state view can answer for, on the server's own
  // predicate rather than a list of names kept here.
  function syncStrategyChoices() {
    var stateOnly = viewSel.value === 'state';
    var options = strategySel.options;
    for (var i = 0; i < options.length; i += 1) {
      options[i].hidden = stateOnly && options[i].dataset.hasState !== 'yes';
    }
    // Landing on a hidden option would ask `/api/state` for a strategy it
    // refuses, and the refusal would look like a bug in the dataset.
    if (stateOnly && strategySel.selectedOptions[0] &&
        strategySel.selectedOptions[0].hidden) {
      for (var j = 0; j < options.length; j += 1) {
        if (!options[j].hidden) { strategySel.value = options[j].value; break; }
      }
    }
  }

  function identity() {
    var parts = (datasetSel.value || '').split('|');
    return {
      exchange: parts[0], market_type: parts[1], symbol: parts[2], timeframe: parts[3]
    };
  }

  function exitModeApplies() {
    var chosen = strategySel.selectedOptions[0];
    return !chosen || CFG.exitModeContracts.indexOf(chosen.dataset.contract) >= 0;
  }

  // ------------------------------------------------------------------ drawing

  function drawBars(bars) {
    var candles = bars.map(function (bar) {
      return {
        time: bar.time, open: bar.open, high: bar.high, low: bar.low, close: bar.close
      };
    });
    candleSeries.setData(candles);
    volumeSeries.setData(bars.map(function (bar) {
      return {
        time: bar.time, value: bar.volume,
        color: bar.close >= bar.open ? COLORS.upDim : COLORS.downDim
      };
    }));
  }

  function drawStateRibbon(payload) {
    var states = payload.why ? payload.why.states : null;
    if (!states) {
      stateSeries.setData([]);
      el('state-legend').hidden = true;
      // With it: an explanation of a state left over from the last strategy,
      // sitting under a chart that has no states at all.
      el('state-explainer').hidden = true;
      return;
    }
    // **The ribbon starts at warmup, not at bar zero.** The machine answers on
    // every bar, and inside warmup its inputs are NaN, which it reads as
    // *failing* -- and failing renders as COMPRESSION. Measured on BTC/USDT
    // spot 1d with `state_machine_v1`, 3,060 bars against a 2,192-bar warmup:
    // the machine reports `compression` on 2,114 of those 2,192 bars. Drawn
    // from bar zero, 72% of this chart said "chop" over the range where the
    // machine knew nothing, which is the exact reading it exists to give.
    var warmup = payload.provenance.warmup_bars || 0;
    // Constant height: the ribbon says *which* state, never how much of it.
    stateSeries.setData(payload.bars.slice(warmup).map(function (bar, i) {
      // No fallback colour: `STATE_COLORS` is asserted against `MarketState` in
      // Python, so a missing one means a state was added there and not here --
      // which should look wrong rather than quietly grey.
      // `warmup + i`, because the slice re-based the index and `states` did not.
      return { time: bar.time, value: 1, color: CFG.stateColors[states[warmup + i]] };
    }));
    el('state-legend').hidden = false;
  }

  function toMarkers(markers) {
    return markers.map(function (marker) {
      var entering = marker.kind === 'entry';
      var isLong = marker.side === 'long';
      // Buys point up whichever way the trade is: entering a long and exiting a
      // short are both purchases, and an arrow that meant "entry" instead would
      // draw a short exit as a sale.
      var up = entering === isLong;
      // No label on the arrow. The frozen report can afford one because a run
      // there holds a handful of trades; state_machine_v1 fills 692 times over
      // this frame, and 692 labels overwrite each other into a solid band. The
      // price and quantity are on the bar's own panel instead, where they can
      // be read.
      return {
        time: marker.time,
        position: up ? 'belowBar' : 'aboveBar',
        color: up ? COLORS.up : COLORS.down,
        shape: up ? 'arrowUp' : 'arrowDown'
      };
    });
  }

  function toLevels(target, bars) {
    var points = [];
    (target || []).forEach(function (value, i) {
      if (value === null || !bars[i]) return;
      points.push({ time: bars[i].time, value: value });
    });
    return points;
  }

  function draw(payload) {
    var primitive = CFG.primitives[payload.provenance.contract];
    if (!primitive) {
      setError('no chart primitive for contract ' + payload.provenance.contract);
      return;
    }
    view.payload = payload;
    view.bars = payload.bars;
    view.why = payload.why;
    view.index = {};
    payload.bars.forEach(function (bar, i) { view.index[bar.time] = i; });
    view.fills = {};
    payload.markers.forEach(function (marker) {
      (view.fills[marker.time] = view.fills[marker.time] || []).push(marker);
    });

    drawBars(payload.bars);
    drawStateRibbon(payload);
    // Both assignments are unconditional so neither contract can inherit the
    // other's leftovers: a level is never drawn as a set of fills, and a set of
    // fills is never drawn as a level.
    markerLayer.setMarkers(primitive === 'markers' ? toMarkers(payload.markers) : []);
    var levels = primitive === 'baseline' ? toLevels(payload.target, payload.bars) : [];
    // Reveal before setData, then hand the exposure chart the price chart's
    // range. A chart sized while its wrapper is display:none has no width, so
    // its first visible range is degenerate -- and linkRange is bidirectional,
    // so that degenerate range was being mirrored onto the price chart, leaving
    // the continuous contract showing one bar against a 600-wide price scale.
    // Price is the authority on where we are looking; exposure follows it.
    el('exposure-wrap').hidden = primitive !== 'baseline';
    exposureSeries.setData(levels);
    if (primitive === 'baseline') {
      var range = priceChart.timeScale().getVisibleLogicalRange();
      if (range) exposureChart.timeScale().setVisibleLogicalRange(range);
    }

    renderLadder();
    view.stream = streams[datasetSel.value] || null;
    el('live').hidden = !view.stream;
    openSocket();

    renderProvenance(payload.provenance);
    view.shifts = payload.why ? shiftsFrom(payload.why.states, payload.bars) : null;
    renderShifts();
    renderTrades();
    view.pinned = null;
    renderBar(payload.bars.length - 1);
    document.title = payload.provenance.identity.symbol + ' · ' +
      payload.provenance.strategy + ' · strategy-lab';
    el('title').textContent = payload.provenance.identity.symbol;
    applyBarSpacing(payload);
  }

  function applyBarSpacing(payload) {
    // Smallest gap rather than the first one: an equity session ends on Friday
    // and resumes on Monday, so an hourly set can open with a three-day step and
    // would have its clock hidden as if it were daily.
    var spacing = 86400;
    payload.bars.slice(0, 200).forEach(function (bar, i, sample) {
      if (i) spacing = Math.min(spacing, bar.time - sample[i - 1].time);
    });
    priceChart.applyOptions({ timeScale: { timeVisible: spacing < 86400 } });
    // Zoom survives a strategy switch, which is the point of the switcher; a
    // different candle set is a different x-axis and has to refit.
    var shown = datasetSel.value;
    if (shown !== view.dataset) {
      view.dataset = shown;
      priceChart.timeScale().fitContent();
    }
  }

  // -------------------------------------------------------------- provenance

  function costText(cost) {
    if (!cost) return 'not executed';
    return fmt(cost.fee * 10000, 1) + ' bps fee · ' + fmt(cost.slippage * 10000, 1) +
      ' bps slip · ' + fmt(cost.position_pct * 100, 0) + '% of ' + fmt(cost.cash, 0);
  }

  function renderProvenance(prov) {
    var host = el('provenance');
    host.replaceChildren();
    var perp = prov.identity.market_type === 'perp';
    // Not the market type, which was wrong in both directions: it marked
    // `donchian` on a funded perp, where nothing is wrong, and stayed silent on
    // a state machine running an equity with crowding pinned to 0.5 -- the M20
    // condition itself. A marker that fires when nothing is wrong is how a
    // reader learns to ignore it, so the false alarm is not the lesser half.
    var crowdingBlind = prov.reads_crowding && !prov.crowding_measured;

    // Chips appear when the field does. A state reading has no contract, exit
    // mode, shorts rule or cost model -- not defaulted here, absent, because
    // `build_state` executed nothing that could have one, and a chip reading
    // "none" would describe a book that was never opened.
    host.appendChild(chip('Strategy', prov.strategy + ' v' + prov.version));
    if (prov.contract) host.appendChild(chip('Contract', prov.contract));
    if ('exit_mode' in prov) {
      host.appendChild(chip(
        'Exit mode', prov.exit_mode || 'none · a target of 0.0 is the exit'
      ));
    }
    if (prov.failure_bars) {
      host.appendChild(chip('Failure bars', String(prov.failure_bars)));
    }
    host.appendChild(chip('Warmup', prov.warmup_bars.toLocaleString() + ' bars'));
    // The number to judge this view by: warmup is counted in bars, so a frame
    // can be long enough to answer and still answer over very little of itself.
    if ('measurable_bars' in prov) {
      host.appendChild(chip(
        'State from',
        prov.measurable_from.slice(0, 10) + '  (' +
        prov.measurable_bars.toLocaleString() + ' bars)'
      ));
    }
    if ('allow_shorts' in prov) {
      host.appendChild(chip('Shorts', prov.allow_shorts ? 'allowed' : 'long only'));
    }
    host.appendChild(chip(
      'crowding_measured',
      prov.crowding_measured ? 'yes' : 'no',
      crowdingBlind ? 'warn' : ''
    ));
    host.appendChild(chip(
      'Funding column',
      prov.funding_attached ? 'attached' : 'absent',
      perp && !prov.funding_attached ? 'warn' : ''
    ));
    if ('cost_model' in prov) {
      host.appendChild(chip('Cost model', costText(prov.cost_model)));
    }
    host.appendChild(chip(
      'Frame',
      prov.first_bar.slice(0, 10) + ' → ' + prov.last_bar.slice(0, 10) + '  (' +
      prov.bar_count.toLocaleString() + ' bars)'
    ));
    host.appendChild(chip('Computed', prov.generated_at.slice(0, 19).replace('T', ' ')));

    if (crowdingBlind) {
      var alert = document.createElement('p');
      alert.className = 'alert';
      // The split is the market type because that decides whether the gap can
      // be closed at all. Not `funding_attached`, which used to split it: above
      // this line the column is absent by construction, since `crowding_measured`
      // *is* whether the strategy found it, so that branch is unreachable now.
      alert.textContent = perp
        ? prov.strategy + ' reads crowding and this perp frame carries no funding ' +
          'column, so it ran with crowding pinned to neutral rather than measured. ' +
          'Not comparable with a funded run: measured on BTC/USDT perp 4h over R5\\'s ' +
          'test half, the crowding-neutral variant returned +16.44% against the ' +
          'published +15.45%. Fetch funding for this range.'
        : prov.identity.market_type + ' settles no funding, and ' + prov.strategy +
          ' reads crowding — so it ran on every bar with that feature pinned to ' +
          'neutral, one feature short of the machine the charter measured. This is ' +
          'permanent here rather than a fetch away: on a perp the same frame would ' +
          'be refused instead of falling back.';
      host.appendChild(alert);
    }
  }

  // --------------------------------------------------------------- why layer

  function renderLegend(bar) {
    var legend = el('legend');
    legend.replaceChildren();
    if (!bar) return;
    ['O', 'H', 'L', 'C'].forEach(function (label, i) {
      var value = [bar.open, bar.high, bar.low, bar.close][i];
      var wrap = document.createElement('span');
      wrap.textContent = label + ' ';
      var strong = document.createElement('b');
      strong.textContent = fmt(value, 2);
      wrap.appendChild(strong);
      legend.appendChild(wrap);
    });
    var change = bar.close - bar.open;
    var pct = bar.open ? (change / bar.open) * 100 : 0;
    var delta = document.createElement('span');
    delta.style.color = change >= 0 ? COLORS.up : COLORS.down;
    delta.textContent = (change >= 0 ? '+' : '') + fmt(change, 2) +
      ' (' + (pct >= 0 ? '+' : '') + fmt(pct, 2) + '%)';
    legend.appendChild(delta);
  }

  function renderBar(i) {
    var bar = view.bars[i];
    if (!bar) return;
    renderLegend(bar);
    el('why-bar').textContent = utcMinute(bar.time) + ' UTC' +
      (view.pinned === i ? ' · pinned' : '');

    var host = el('why-chips');
    host.replaceChildren();
    var payload = view.payload;
    if (!payload) return;

    (view.fills[bar.time] || []).forEach(function (fill) {
      host.appendChild(chip(
        fill.side + ' ' + fill.kind,
        fmt(fill.price, 2) + ' × ' + fmt(fill.size, 4),
        fill.side === 'long' ? 'up' : 'down'
      ));
    });
    if (view.why) {
      host.appendChild(chip('State', view.why.states[i], 'state'));
      Object.keys(view.why.features).forEach(function (name) {
        var value = view.why.features[name][i];
        host.appendChild(chip(
          name, value === null ? '—' : fmt(value, 3), value === null ? 'absent' : ''
        ));
      });
    }
    if (payload.target) {
      var level = payload.target[i];
      host.appendChild(chip(
        'Target', level === null ? '—' : fmt(level, 3),
        level === null ? 'absent' : ''
      ));
    }
    if (payload.position_size) {
      var scale = payload.position_size[i];
      host.appendChild(chip(
        'Size scale', scale === null ? '—' : fmt(scale, 3),
        scale === null ? 'absent' : ''
      ));
    }

    el('why-note').textContent = view.why
      ? 'State and features are recomputed from the same frame the strategy saw. ' +
        'A dash is a bar the feature could not yet be measured on — not a zero, ' +
        'which would read as measured and neutral.'
      : payload.provenance.strategy + ' computes no feature frame, so there is no ' +
        'state layer behind these bars. The chart shows what it did; this strategy ' +
        'has nothing further to show about why.';
  }

  // ------------------------------------------------------- the state sequence

  // A `!==` over a series the payload already carries, so it cannot disagree
  // with the per-bar chip or the board tile. Deriving it server-side would be
  // M36's third answer, free to drift from the two that already agree.
  function shiftsFrom(states, bars) {
    var out = [];
    // Bar 0 for the first change: it is the opening state rather than one
    // something moved into, and its dwell is measured from the frame's start.
    var previousIndex = 0;
    for (var i = 1; i < states.length; i += 1) {
      if (states[i] === states[i - 1]) continue;
      out.push({
        index: i,
        previousIndex: previousIndex,
        from: states[i - 1],
        to: states[i],
        time: bars[i].time
      });
      previousIndex = i;
    }
    return out;
  }

  function dwellText(bars, fromIndex, toIndex) {
    var count = toIndex - fromIndex;
    var seconds = bars[toIndex].time - bars[fromIndex].time;
    var hours = seconds / 3600;
    var span = hours < 48 ? fmt(hours, 1) + 'h' : fmt(hours / 24, 1) + 'd';
    return count + (count === 1 ? ' bar · ' : ' bars · ') + span;
  }

  function renderShifts() {
    var section = el('transitions-section');
    var host = el('transitions');
    host.replaceChildren();
    if (!view.shifts) {
      section.hidden = true;
      return;
    }
    section.hidden = false;

    var warmup = view.payload.provenance.warmup_bars;
    var inWarmup = 0;
    // Newest first: the question a monitor answers most often is what it is
    // doing now, not what it did first.
    view.shifts.slice().reverse().forEach(function (shift) {
      var early = warmup !== null && shift.index < warmup;
      if (early) inWarmup += 1;
      var row = document.createElement('button');
      row.type = 'button';
      row.setAttribute('aria-pressed', 'false');
      row.className = 'shift' + (early ? ' warmup' : '');
      row.dataset.index = shift.index;

      var when = document.createElement('span');
      when.className = 'shift-when';
      when.textContent = utcMinute(shift.time);
      var what = document.createElement('span');
      var from = document.createElement('span');
      from.className = 'shift-from';
      from.textContent = shift.from + ' \u2192 ';
      var to = document.createElement('span');
      to.className = 'shift-to';
      to.textContent = shift.to;
      what.appendChild(from);
      what.appendChild(to);
      var dwell = document.createElement('span');
      dwell.className = 'shift-dwell';
      dwell.textContent = dwellText(view.bars, shift.previousIndex, shift.index);

      row.appendChild(when);
      row.appendChild(what);
      row.appendChild(dwell);
      row.addEventListener('click', function () { pinBar(shift.index); });
      host.appendChild(row);
    });

    el('transitions-count').textContent = view.shifts.length +
      (view.shifts.length === 1 ? ' change' : ' changes') + ' · newest first';
    el('transitions-note').textContent = inWarmup
      ? inWarmup + ' of these fall inside the ' + warmup.toLocaleString() +
        '-bar warmup and are dimmed — which is also why the ribbon starts where ' +
        'it does. The machine answers on every bar, and inside warmup its inputs ' +
        'are not measurable yet; it reads that as failing, which renders as ' +
        'compression. So those are transitions of the machine, not readings of ' +
        'the market, and the strategy was not acting on them either.'
      : 'Dwell is how long the previous state held. Click a row to pin that bar.';
  }

  function pinBar(i) {
    view.pinned = i;
    renderBar(i);
    markSelectedShift(i);
    var range = priceChart.timeScale().getVisibleLogicalRange();
    // Only recentre when the bar is off screen: a click on a visible row should
    // not yank the chart out from under someone comparing two of them.
    if (range && (i < range.from || i > range.to)) {
      var half = Math.max(20, (range.to - range.from) / 2);
      priceChart.timeScale().setVisibleLogicalRange({ from: i - half, to: i + half });
    }
  }

  function markSelectedShift(i) {
    Array.prototype.forEach.call(el('transitions').children, function (row) {
      var selected = Number(row.dataset.index) === i;
      row.classList.toggle('selected', selected);
      row.setAttribute('aria-pressed', String(selected));
    });
  }

  // ------------------------------------------------------------------ trades

  // The quote currency, when the symbol names one. `BTC/USDT:USDT` quotes in
  // USDT; `SPY` names no currency at all, and printing a guessed one beside a
  // PnL would be a claim about the account this never touched.
  function quoteUnit(symbol) {
    var parts = String(symbol).split('/');
    return parts.length > 1 ? parts[1].split(':')[0] : '';
  }

  function cell(row, text, cls) {
    var td = document.createElement('td');
    td.textContent = text;
    if (cls) td.className = cls;
    row.appendChild(td);
  }

  function renderTrades() {
    var section = el('trades-section');
    var host = el('trades');
    host.replaceChildren();
    var trades = view.payload.trades;
    var costs = view.payload.provenance.cost_model;
    // Empty is a different statement from absent. The continuous contract runs
    // no book here at all, so it gets no section; a boolean strategy that
    // simply never traded over this range gets one saying so, because "no
    // trades" is the result.
    if (!costs) {
      section.hidden = true;
      return;
    }
    section.hidden = false;

    var unit = quoteUnit(view.payload.provenance.identity.symbol);
    var closed = trades.filter(function (t) { return t.status === 'closed'; });
    var won = closed.filter(function (t) { return t.pnl > 0; }).length;
    var net = closed.reduce(function (sum, t) { return sum + t.pnl; }, 0);
    var open = trades.length - closed.length;

    var table = document.createElement('table');
    var head = document.createElement('tr');
    ['opened', 'closed', 'side', 'qty', 'in', 'out', 'fees', 'pnl', 'return']
      .forEach(function (name) {
        var th = document.createElement('th');
        th.textContent = name;
        head.appendChild(th);
      });
    var thead = document.createElement('thead');
    thead.appendChild(head);
    table.appendChild(thead);

    var body = document.createElement('tbody');
    // Newest first, matching the state changes above it.
    trades.slice().reverse().forEach(function (trade) {
      var row = document.createElement('tr');
      row.className = trade.status;
      cell(row, utcMinute(trade.entry_time));
      cell(row, trade.exit_time === null ? 'still open' : utcMinute(trade.exit_time));
      cell(row, trade.direction, 'side-' + trade.direction);
      cell(row, fmt(trade.size, 6));
      cell(row, fmt(trade.entry_price, 2));
      cell(row, trade.exit_price === null ? '—' : fmt(trade.exit_price, 2));
      cell(row, fmt(trade.fees, 2));
      cell(row, fmt(trade.pnl, 2), trade.pnl >= 0 ? 'gain' : 'loss');
      cell(row, fmt(trade.return_pct * 100, 2) + '%',
           trade.return_pct >= 0 ? 'gain' : 'loss');
      // The exit is the bar that decided the result; an open trade has only its
      // entry to point at.
      var at = view.index[trade.exit_time === null ? trade.entry_time : trade.exit_time];
      row.addEventListener('click', function () { pinBar(at); });
      body.appendChild(row);
    });
    table.appendChild(body);
    host.appendChild(table);

    el('trades-summary').textContent = closed.length
      ? closed.length + ' closed · net ' + fmt(net, 2) + (unit ? ' ' + unit : '') +
        ' · ' + fmt((won / closed.length) * 100, 1) + '% won' +
        (open ? ' · ' + open + ' still open' : '')
      : (trades.length ? trades.length + ' open, none closed yet' : 'none');
    el('trades-note').textContent = trades.length
      ? 'What this strategy would have made over the frame above, at ' +
        fmt(costs.fee * 100, 3) + '% fee and ' + fmt(costs.slippage * 100, 3) +
        '% slippage, sized non-compounding from ' + fmt(costs.cash, 0) +
        ' — the same run the arrows are drawn from, not an account. ' +
        'Net and win rate count closed trades only: an open one is marked ' +
        'against the last bar, and that mark moves. Click a row to pin its bar.'
      : 'No trade over this range. The arrows and this table come from the same ' +
        'run, so an empty table means the chart is empty too.';
  }

  // ---------------------------------------------------------------- live feed

  // The venue's forming candle, drawn but **never analysed**. A tick that
  // reached `build_analysis` would produce a state and a marker that flip when
  // the bar closes, which is the one reading the whole event path refuses --
  // `include_forming=False`, and a timestamp is complete only once a later one
  // arrives. So the candle moves and the why-layer stays where the last closed
  // bar left it, and `renderBar` says so rather than letting the previous bar's
  // state read as this one's.
  var live = { socket: null, on: true, bar: null, ticks: 0, watchdog: null };

  // A socket that opens is not a feed that delivers. Measured against Binance
  // from here: `fstream.binance.com` -- every perp dataset in this repo --
  // accepts the connection and then sends nothing, on both the raw and combined
  // stream forms, while `stream.binance.com` delivers 6 frames in 12s on the
  // same pair. `onopen` is therefore a proxy for working, and reporting "live"
  // off it would leave a green dot over a frozen chart. It says live once a
  // frame has actually arrived, and says so when none does.
  var FIRST_TICK_GRACE_MS = 15000;

  function setLive(cls, text) {
    var button = el('live');
    button.className = 'live ' + cls;
    el('live-text').textContent = text;
  }

  function closeSocket() {
    if (live.watchdog) { clearTimeout(live.watchdog); live.watchdog = null; }
    if (!live.socket) return;
    var socket = live.socket;
    live.socket = null;          // before close(), so onclose knows it was us
    socket.close();
  }

  function openSocket() {
    closeSocket();
    live.bar = null;
    var url = view.stream;
    if (!url || !live.on) {
      setLive('off', live.on ? 'no stream' : 'paused');
      return;
    }
    setLive('wait', 'connecting');
    live.ticks = 0;
    var socket = new WebSocket(url);
    live.socket = socket;
    socket.onopen = function () {
      if (live.socket !== socket) return;
      setLive('wait', 'waiting for data');
      live.watchdog = setTimeout(function () {
        if (live.socket === socket && !live.ticks) setLive('off', 'connected · no data');
      }, FIRST_TICK_GRACE_MS);
    };
    socket.onmessage = function (event) {
      // The guard `onclose` and the watchdog already carry. Without it a frame
      // in flight when the dataset changed draws the old market's candle onto
      // the new chart, and a stale `x: true` POSTs a refresh nobody asked for.
      if (live.socket !== socket) return;
      if (!live.ticks) setLive('on', 'live');
      live.ticks += 1;
      onTick(JSON.parse(event.data));
    };
    socket.onerror = function () {
      if (live.socket === socket) setLive('off', 'stream error');
    };
    socket.onclose = function () {
      // Only report a drop we did not ask for; switching dataset closes it too.
      if (live.socket === socket) { live.socket = null; setLive('off', 'disconnected'); }
    };
  }

  function onTick(message) {
    var k = message.k;
    if (!k) return;
    live.bar = {
      time: Math.floor(k.t / 1000),
      open: Number(k.o), high: Number(k.h), low: Number(k.l), close: Number(k.c)
    };
    candleSeries.update(live.bar);
    if (!k.x) return;
    // The venue says this bar is final, which is the whole reason this is not a
    // timer: the one moment the stored candles are provably behind. Refresh
    // stores it and recomputes, so state and markers catch up to the bar the
    // chart already drew.
    if (viewSel.value === 'board') return;
    setLive('wait', 'bar closed · refreshing');
    refreshInstrument().then(function () {
      if (live.socket) setLive('on', 'live');
    });
  }

  el('live').addEventListener('click', function () {
    live.on = !live.on;
    openSocket();
  });

  priceChart.subscribeCrosshairMove(function (param) {
    if (view.pinned !== null) return;
    var i = param.time !== undefined ? view.index[param.time] : undefined;
    // A live bar has no index: it was drawn after the analysis, on purpose. Say
    // so rather than falling back to the last analysed bar, whose state would
    // then read as this one's.
    if (i === undefined && live.bar && param.time === live.bar.time) {
      el('why-bar').textContent = 'live bar · not analysed until it closes';
      el('why-chips').replaceChildren();
      return;
    }
    renderBar(i === undefined ? view.bars.length - 1 : i);
  });
  priceChart.subscribeClick(function (param) {
    var i = param.time !== undefined ? view.index[param.time] : undefined;
    if (i === undefined) return;
    view.pinned = view.pinned === i ? null : i;
    renderBar(i);
    markSelectedShift(view.pinned === null ? -1 : i);
  });

  // ----------------------------------------------------------------- the board

  function tileLine(label, value, cls) {
    var line = document.createElement('p');
    line.className = 'tile-line' + (cls ? ' ' + cls : '');
    line.appendChild(document.createTextNode(label + ' '));
    var strong = document.createElement('b');
    strong.textContent = value;
    line.appendChild(strong);
    return line;
  }

  function stamp(text) {
    // Bars are UTC everywhere in this lab and the minute is the finest thing a
    // tile has room for.
    return text ? text.slice(0, 16) : '—';
  }

  function datasetKey(identity) {
    // Storage's own four-part identity, in the order the dataset select uses,
    // so a board row and a dataset option are the same string.
    return [identity.exchange, identity.market_type, identity.symbol,
            identity.timeframe].join('|');
  }

  function sparkline(closes) {
    var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'spark');
    svg.setAttribute('viewBox', '0 0 100 30');
    // Stretched to the tile: the shape of the tail is the point, and a tile is
    // not a price scale. The tooltip says how many bars it covers, because the
    // same shape over 120 bars and over 12 is not the same claim.
    svg.setAttribute('preserveAspectRatio', 'none');
    if (closes.length < 2) return svg;
    // An SVG tooltip is a <title> child, not an attribute. It says how many
    // bars are under the line, because the same shape over 120 bars and over 12
    // is not the same claim and the tile has no axis to say which.
    var caption = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    caption.textContent = closes.length + ' closes, ending on the bar above';
    svg.appendChild(caption);
    var lo = Math.min.apply(null, closes);
    var hi = Math.max.apply(null, closes);
    var span = (hi - lo) || 1;
    var points = closes.map(function (value, i) {
      var x = (i / (closes.length - 1)) * 100;
      var y = 29 - ((value - lo) / span) * 28;
      return x.toFixed(2) + ',' + y.toFixed(2);
    }).join(' ');
    var poly = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    poly.setAttribute('points', points);
    poly.setAttribute('fill', 'none');
    poly.setAttribute('stroke-width', '1.2');
    poly.setAttribute('vector-effect', 'non-scaling-stroke');
    poly.setAttribute('stroke',
      closes[closes.length - 1] >= closes[0] ? COLORS.up : COLORS.down);
    svg.appendChild(poly);
    return svg;
  }

  function tileHead(row) {
    var head = document.createElement('div');
    head.className = 'tile-head';
    var symbol = document.createElement('span');
    symbol.className = 'tile-symbol';
    symbol.textContent = row.identity.symbol + ' · ' + row.identity.timeframe;
    var where = document.createElement('span');
    where.className = 'tile-where';
    where.textContent = row.identity.exchange + '/' + row.identity.market_type;
    var actions = document.createElement('div');
    actions.className = 'tile-actions';
    var open = document.createElement('button');
    open.type = 'button';
    open.textContent = 'open';
    open.title = 'draw this pair in the instrument view';
    open.addEventListener('click', function () { openInstrument(row); });
    var refresh = document.createElement('button');
    refresh.type = 'button';
    refresh.textContent = 'refresh';
    refresh.title = 'fetch this dataset\\'s newest bars and settlements, then recompute';
    refresh.addEventListener('click', function () { refreshOne(row.identity, refresh); });
    actions.appendChild(open);
    actions.appendChild(refresh);
    head.appendChild(symbol);
    head.appendChild(where);
    head.appendChild(actions);
    return head;
  }

  function tileFeatures(row) {
    var host = document.createElement('div');
    host.className = 'tile-features';
    Object.keys(row.features || {}).forEach(function (name) {
      var value = row.features[name];
      var chip = document.createElement('span');
      chip.textContent = name + ' ';
      var strong = document.createElement('b');
      // A dash is "not measurable on this bar", never a zero -- which would
      // read as measured and neutral, a different claim about the market.
      strong.textContent = value === null ? '—' : fmt(value, 3);
      chip.appendChild(strong);
      host.appendChild(chip);
    });
    var prov = row.provenance;
    // The strip's predicate, for the strip's reason. `crowding 0.500` is already
    // in the chips above, so what this adds is that the 0.5 was *pinned*.
    if (prov && prov.reads_crowding && !prov.crowding_measured) {
      var blind = document.createElement('span');
      blind.className = 'warn';
      blind.textContent = 'crowding pinned neutral';
      blind.title = row.identity.market_type === 'perp'
        ? prov.strategy + ' reads crowding and this perp frame carries no funding ' +
          'column — not comparable with a funded run, and the charter\\'s R5 figures ' +
          'are the crowding-measured ones.'
        : row.identity.market_type + ' settles no funding, so ' + prov.strategy +
          ' ran one feature short with crowding pinned to 0.5. Permanent here: on ' +
          'a perp this frame would be refused rather than fall back.';
      host.appendChild(blind);
    }
    return host;
  }

  function writtenAgeDays(written) {
    // Postgres stamps read `2026-08-03 02:20:07.225509+00:00`, whose space makes
    // it not ISO 8601 -- Date parses that by implementation-defined luck, so the
    // same swap the tile bounds already do. An unparseable stamp is null rather
    // than NaN days, which would compare false against every threshold and read
    // as fresh.
    if (!written) return null;
    var at = new Date(written.replace(' ', 'T')).getTime();
    return isNaN(at) ? null : (Date.now() - at) / 86400000;
  }

  function writtenLine(row) {
    // Never "restated": nothing here detects a dividend, and claiming one would
    // need a second stored copy to diff against -- which R10a's rule refuses.
    // What this says is how old these bars are as *writes*, flagged once that is
    // long enough for a distribution to have plausibly landed since.
    var age = writtenAgeDays(row.last_written);
    var stale = age !== null && age >= CFG.restatementStaleDays;
    var line = tileLine('candles written', stamp(row.last_written), stale ? 'lag' : '');
    line.title = 'Dividend-adjusted, so a distribution rescales past bars rather ' +
      'than appending one: measured on the stored SPY weekly series, 333 of 333 ' +
      'bars moved against a fresh fetch (median 0.257%) and donchian differed on ' +
      '3 of them. This is the newest write in the set — refresh re-fetches the ' +
      'recent bars.' + (stale
        ? ' Older than ' + CFG.restatementStaleDays + ' days, which is long enough ' +
          'for a distribution to have landed since.'
        : '');
    return line;
  }

  // Each tile states its own staleness and not the other's: a perp's right edge
  // can lag its newest candle by up to one settlement (M37), where an equity's
  // cannot lag at all and what goes stale is the whole history. Carrying both
  // lines on both tiles would state a risk that does not exist for one of them,
  // which is the same kind of wrong as stating neither.
  function tileFreshness(row, wrap) {
    if (row.identity.market_type === CFG.restatedMarketType) {
      wrap.appendChild(tileLine('as of', stamp(row.as_of)));
      wrap.appendChild(writtenLine(row));
      return;
    }
    wrap.appendChild(tileLine('as of', stamp(row.as_of),
      row.as_of === row.dataset_last_bar ? '' : 'lag'));
    if (row.as_of !== row.dataset_last_bar) {
      // Bounded by stored funding rather than by the newest candle, so the lag
      // is a fact about settlements and not a stale page. Said out loud because
      // a board silently a bar behind is exactly the figure-without-context the
      // provenance habit exists to prevent.
      wrap.appendChild(tileLine('newest stored bar', stamp(row.dataset_last_bar), 'lag'));
    }
  }

  function tile(row) {
    var wrap = document.createElement('div');
    wrap.className = 'tile' + (row.unavailable ? ' refused' : '');
    wrap.appendChild(tileHead(row));

    if (row.unavailable) {
      var refusal = document.createElement('p');
      refusal.className = 'tile-line refusal';
      // First line only: the loader's refusals carry a fetch command and a
      // covered span underneath, which belong in a tooltip rather than in a
      // tile that has fifteen siblings.
      refusal.textContent = row.unavailable.split('\\n')[0];
      refusal.title = row.unavailable;
      wrap.appendChild(refusal);
      wrap.appendChild(tileLine('newest stored bar', stamp(row.dataset_last_bar)));
      // No `as of` to compare against here, so the extent line above carries no
      // lag claim -- but the write time is still a fact about the stored bars,
      // and on a restated history it is the one that says whether re-fetching
      // would change what is refusing.
      if (row.identity.market_type === CFG.restatedMarketType) {
        wrap.appendChild(writtenLine(row));
      }
      return wrap;
    }

    var prov = row.provenance;
    if (row.state !== null) {
      var pill = document.createElement('span');
      pill.className = 'state-pill';
      pill.textContent = row.state;
      wrap.appendChild(pill);
    }
    wrap.appendChild(sparkline(row.closes));
    tileFreshness(row, wrap);
    if (row.latest_fill) {
      var fill = row.latest_fill;
      wrap.appendChild(tileLine(
        fill.side + ' ' + fill.kind,
        fmt(fill.price, 2) + ' × ' + fmt(fill.size, 4) + '  ' + utcMinute(fill.time)
      ));
    } else if (row.target !== null) {
      wrap.appendChild(tileLine('target', fmt(row.target, 3)));
    } else {
      wrap.appendChild(tileLine('fills', 'none on this frame'));
    }
    wrap.appendChild(tileFeatures(row));
    if (prov) {
      wrap.appendChild(tileLine(
        prov.strategy + ' v' + prov.version + ' ·',
        (prov.exit_mode || 'target 0.0 is the exit') + ' · ' +
        prov.bar_count.toLocaleString() + ' bars · warmup ' +
        prov.warmup_bars.toLocaleString() + ' · computed ' +
        prov.generated_at.slice(11, 19)
      ));
      // Its own line rather than appended to that one, which is already long.
      // Every tile carries the engine's defaults, and stating it on each is the
      // point: a reader should not have to know the board never varies costs to
      // know what these numbers were charged. `costText` rather than a second
      // formatter, so a tile and the chart read a cost model the same way.
      wrap.appendChild(tileLine('costs', costText(prov.cost_model)));
    }
    return wrap;
  }

  function readRows(url, onRow, signal) {
    // Read as it arrives. A row is a whole-history recompute (330-400 ms warm)
    // and the server emits each one as it finishes, so response.json() here
    // would wait for the slowest instrument before drawing the fastest -- 1.2 s
    // instead of 206 ms to first paint, measured on the four stored perp sets.
    return fetch(url, { signal: signal }).then(function (response) {
      if (!response.ok) {
        return response.json().then(function (body) {
          throw new Error(detail(body) || ('HTTP ' + response.status));
        });
      }
      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var buffer = '';
      function pump() {
        return reader.read().then(function (chunk) {
          buffer += decoder.decode(chunk.value || new Uint8Array(), { stream: !chunk.done });
          var lines = buffer.split('\\n');
          buffer = lines.pop();
          lines.forEach(function (line) { if (line.trim()) onRow(JSON.parse(line)); });
          if (!chunk.done) return pump();
          if (buffer.trim()) onRow(JSON.parse(buffer));
        });
      }
      return pump();
    });
  }

  var boardPending = 0;
  var boardAbort = null;
  // Set only when a tile opens the instrument view; see `query`.
  var exactBounds = null;

  function reloadBoardIfShown() {
    // A refresh started on the board can resolve after the user has left it.
    return onBoard() ? loadBoard() : Promise.resolve();
  }

  function onBoard() {
    return viewSel.value === 'board';
  }

  // A board failure that landed off the board, waiting for the board to return.
  var heldBoardError = null;

  // A refresh chain outlives the view that started it, and its progress and
  // failures belong to that view. Written unguarded they land on the
  // instrument view's status line, and `setError` also dims its provenance
  // strip -- a board failure reported over a chart it says nothing about.
  function boardStatus(text) {
    if (onBoard()) setStatus(text);
  }

  function clearBoardError() {
    // A retry makes the last attempt's banner stale, and leaving it up while a
    // new fetch is in flight misreports which attempt it describes.
    heldBoardError = null;
    if (onBoard()) setBanner('');
  }

  function boardError(text) {
    // Held *and* shown, not one or the other. A refresh failure is a fact about
    // the data rather than about the view, so it has to survive both a trip to
    // the chart and a board redraw the user asked for in between -- a strategy
    // switch mid-refresh would otherwise wipe the banner as its last row lands,
    // which is the same silence as never showing it.
    heldBoardError = text;
    if (onBoard()) setBanner(text);
  }

  function abortBoard() {
    // Stops the rows at the server, not just on screen. Bumping the token as
    // well so anything already decoded and queued bails out too.
    boardPending += 1;
    if (boardAbort) boardAbort.abort();
    boardAbort = null;
  }
  var boardIdentities = [];

  function boardQuery() {
    var params = new URLSearchParams({ strategies: strategySel.value });
    if (marketSel.value) params.set('market_type', marketSel.value);
    if (exitSel.value !== ENGINE_DEFAULT && exitModeApplies()) {
      params.set('exit_mode', exitSel.value);
    }
    return params;
  }

  function loadBoard() {
    if (!strategySel.value) return Promise.resolve();
    // Stop the superseded board at the server, not just on screen. Each row is
    // a full recompute and nothing else cancels it, so switching strategy twice
    // while one streams leaves three boards competing for the same threadpool
    // and starves the instrument view behind them. Before the token, not after:
    // `abortBoard` bumps the same counter, so taking one first would invalidate
    // the board being started.
    abortBoard();
    var token = ++boardPending;
    // `boardIdentities` fills as the rows land, and `refreshAll` snapshots it.
    // Clicked mid-stream it would fetch only the datasets already drawn, then
    // report "N of N" for a partial N -- and the reload after it recomputes the
    // untouched ones from unchanged candles, so nothing on screen says a
    // dataset was skipped.
    el('refresh-all').disabled = true;
    boardAbort = typeof AbortController === 'function' ? new AbortController() : null;
    var signal = boardAbort ? boardAbort.signal : undefined;
    var host = el('board');
    host.replaceChildren();
    boardIdentities = [];
    setStatus('board · computing …');
    var started = performance.now();
    var painted = null;
    var rows = 0;
    return readRows('/api/board?' + boardQuery().toString(), function (row) {
      if (token !== boardPending) return;
      if (painted === null) painted = Math.round(performance.now() - started);
      rows += 1;
      // One entry per dataset, not per row: two strategies on one instrument
      // are two tiles and one candle set, and "refresh all" fetching it twice
      // would be two calls to the venue for the same bars.
      var id = datasetKey(row.identity);
      if (!boardIdentities.some(function (seen) { return datasetKey(seen) === id; })) {
        boardIdentities.push(row.identity);
      }
      host.appendChild(tile(row));
      setStatus(rows + ' rows · first in ' + painted + ' ms');
    }, signal).then(function () {
      if (token !== boardPending) return;
      // Here rather than after the catch below: a truncated stream leaves
      // `boardIdentities` holding only the rows that arrived, and "refresh all"
      // over a partial list reports it as the whole batch. A board that did not
      // finish is not one you can refresh all of -- each tile keeps its own
      // button, which is the honest escape hatch.
      el('refresh-all').disabled = false;
      // After the rows, which is where `refreshAll` puts its own failures too:
      // this clears the banner, so anything shown before it is gone. The held
      // failure is re-shown rather than consumed -- only a fresh refresh clears
      // it, since redrawing the board does not make a failed fetch have
      // succeeded.
      setBanner(heldBoardError || '');
      document.title = 'board · ' + strategySel.value + ' · strategy-lab';
      el('title').textContent = 'board';
      setStatus(rows === 0
        ? 'no candle sets stored for this market'
        : rows + ' rows · first in ' + painted + ' ms · all in ' +
          Math.round(performance.now() - started) + ' ms' +
          claimRefreshed('board'));
    }).catch(function (error) {
      // An abort is this function superseding itself, not a failure.
      if (error && error.name === 'AbortError') return;
      // Whatever arrived stays on screen. A truncated board is more use than a
      // blank one, and the banner says it is truncated. The banner alone: a
      // board that failed to stream says nothing about the chart's provenance.
      if (token === boardPending) setBanner(error.message);
    });
  }

  function refreshOne(identity, button) {
    if (button) button.disabled = true;
    clearBoardError();
    boardStatus('fetching ' + identity.symbol + ' ' + identity.timeframe + ' …');
    return getJSON('/api/refresh?' + new URLSearchParams(identity).toString(),
                   { method: 'POST' })
      .then(function (result) {
        view.refreshed = {
          view: 'board',
          text: identity.symbol + ' ' + identity.timeframe + ' ' + refreshText(result)
        };
        // The whole board recomputes, since nothing is held between requests.
        // One dataset moved; the rest cost what they always cost -- so if the
        // user opened a tile while the POST was in flight, this would start a
        // full board against a hidden host and retitle the page over the chart
        // they are looking at.
        return reloadBoardIfShown();
      })
      .catch(function (error) { boardError(error.message); })
      .then(function () { if (button) button.disabled = false; });
  }

  function refreshAll() {
    var button = el('refresh-all');
    button.disabled = true;
    clearBoardError();
    // One venue call at a time. Firing sixteen at once is rude to the exchange
    // and interleaves the failures, and the whole point of an explicit refresh
    // is that somebody chose this moment to make them.
    var identities = boardIdentities.slice();
    var candles = 0;
    var settlements = 0;
    var failures = [];
    var chain = Promise.resolve();
    identities.forEach(function (identity, i) {
      chain = chain.then(function () {
        boardStatus('fetching ' + identity.symbol + ' ' + identity.timeframe +
                    ' (' + (i + 1) + '/' + identities.length + ') …');
        return getJSON('/api/refresh?' + new URLSearchParams(identity).toString(),
                       { method: 'POST' })
          .then(function (result) {
            candles += result.candles_upserted;
            if (result.funding_upserted) settlements += result.funding_upserted;
          })
          // Per identity, so one venue failure -- which /api/refresh reports as
          // a 502 by design -- does not cancel the datasets after it, and does
          // not skip the reload that would have drawn the ones that did fetch.
          // The board's whole claim is that one instrument cannot blank the
          // others; a refresh that abandons fifteen of them breaks it.
          .catch(function (error) {
            failures.push(identity.symbol + ' ' + identity.timeframe + ': ' +
                          error.message);
          });
      });
    });
    return chain
      .then(function () {
        view.refreshed = { view: 'board', text:
          'refreshed ' + (identities.length - failures.length) + ' of ' +
          identities.length + ' datasets · ' + candles + ' candles · ' +
          settlements + ' settlements' };
        return reloadBoardIfShown();
      })
      // After the reload, which clears the banner: a failure the user cannot
      // see is the same as one that did not happen.
      .then(function () { if (failures.length) boardError(failures.join(' · ')); })
      .catch(function (error) { boardError(error.message); })
      .then(function () { button.disabled = false; });
  }

  function openInstrument(row) {
    var id = datasetKey(row.identity);
    var known = Array.prototype.some.call(datasetSel.options, function (opt) {
      return opt.value === id;
    });
    if (!known) {
      // Raised on the board, before the view switches, so it is the board's.
      setBanner('no candle set ' + id + ' in the dataset list');
      return;
    }
    datasetSel.value = id;
    if (row.provenance) {
      // The board bounds a perp frame by its stored funding span and these
      // inputs speak whole days, so the head is rounded *up* to the next whole
      // UTC day when the span opens mid-day: a day boundary below the first
      // settlement puts unfunded bars back in the window and the coverage guard
      // refuses the frame the tile was just showing. The bars lost are the
      // front of a warmup.
      var first = new Date(row.provenance.first_bar.replace(' ', 'T'));
      if (first.getUTCHours() || first.getUTCMinutes()) {
        first = new Date(first.getTime() + 86400000);
      }
      el('start').value = first.toISOString().slice(0, 10);
      el('end').value = row.provenance.last_bar.slice(0, 10);
      // The inputs show whole days; the request carries the tile's own edges.
      exactBounds = {
        start: row.provenance.first_bar,
        end: row.provenance.last_bar
      };
    }
    setView('instrument');
  }

  // ------------------------------------------------------------------ loading

  function query() {
    var params = new URLSearchParams(identity());
    params.set('strategy', strategySel.value);
    if (exitSel.value !== ENGINE_DEFAULT && exitModeApplies()) {
      params.set('exit_mode', exitSel.value);
    }
    // Bounds are what make a frame whose funding history starts late viewable
    // at all: a strategy that reads funding is refused over the uncovered head
    // rather than run neutral through it.
    ['start', 'end'].forEach(function (bound) {
      if (el(bound).value) params.set(bound, el(bound).value);
    });
    // A tile's frame ends at a settlement, which is a time of day these
    // `type="date"` inputs cannot hold -- and a bare `end` means *all* of that
    // day to the API, so the chart would read bars the tile excluded and
    // disagree with the tile that opened it. Worse when the lag is larger: the
    // extra bars can outrun the funding the tile was bounded by, and the whole
    // frame is refused. Held beside the inputs and cleared the moment the user
    // edits one, since from then on the visible dates are the truth.
    if (exactBounds) {
      if (exactBounds.start) params.set('start', exactBounds.start);
      if (exactBounds.end) params.set('end', exactBounds.end);
    }
    return params;
  }

  function getJSON(url, options) {
    return fetch(url, options).then(function (response) {
      return response.json().then(function (body) {
        if (response.ok) return body;
        throw new Error(detail(body) || ('HTTP ' + response.status));
      });
    });
  }

  function detail(body) {
    if (!body || !body.detail) return '';
    if (typeof body.detail === 'string') return body.detail;
    return body.detail.map(function (item) {
      return (item.loc || []).join('.') + ': ' + item.msg;
    }).join('; ');
  }

  var pending = 0;

  function abandonInstrument() {
    // The mirror of `abortBoard`. `load`'s guard only fires when another
    // `load` bumps `pending`, and entering the board starts no instrument
    // request -- so an `/api/analysis` still in flight would resolve into the
    // board, draw a chart nobody is looking at and retitle the page with the
    // instrument's symbol. Invalidated rather than aborted: it is one request
    // rather than a row per dataset, and `getJSON` carries no signal.
    pending += 1;
  }

  function load() {
    if (!datasetSel.value || !strategySel.value) return Promise.resolve();
    var token = ++pending;
    setStatus('computing …');
    var started = performance.now();
    return getJSON('/api/analysis?' + query().toString())
      .then(function (payload) {
        // A slower earlier request must not overwrite a faster later one: the
        // chart would then show one strategy under another's provenance.
        if (token !== pending) return;
        setError('');
        draw(payload);
        setStatus(Math.round(performance.now() - started) + ' ms · ' +
          payload.bars.length.toLocaleString() + ' bars' +
          claimRefreshed('instrument'));
      })
      .catch(function (error) {
        if (token === pending) setError(error.message);
      });
  }

  function stateQuery() {
    var params = new URLSearchParams(identity());
    params.set('strategy', strategySel.value);
    // No exit mode, no cost model: `/api/state` refuses any parameter it does
    // not recognise, by name, and none of those move a feature or a state.
    ['start', 'end'].forEach(function (bound) {
      if (el(bound).value) params.set(bound, el(bound).value);
    });
    if (exactBounds) {
      if (exactBounds.start) params.set('start', exactBounds.start);
      if (exactBounds.end) params.set('end', exactBounds.end);
    }
    return params;
  }

  function loadState() {
    if (!datasetSel.value || !strategySel.value) return Promise.resolve();
    var token = ++pending;
    setStatus('reading state …');
    var started = performance.now();
    return getJSON('/api/state?' + stateQuery().toString())
      .then(function (payload) {
        if (token !== pending) return;
        setError('');
        drawState(payload);
        setStatus(Math.round(performance.now() - started) + ' ms · ' +
          payload.provenance.measurable_bars.toLocaleString() + ' of ' +
          payload.bars.length.toLocaleString() + ' bars carry a state' +
          claimRefreshed('instrument'));
      })
      .catch(function (error) {
        if (token === pending) setError(error.message);
      });
  }

  function drawState(payload) {
    view.payload = payload;
    view.bars = payload.bars;
    view.why = payload.why;
    view.index = {};
    payload.bars.forEach(function (bar, i) { view.index[bar.time] = i; });
    // Nothing was executed, so there is nothing to mark. Assigned rather than
    // left alone for the reason `draw` assigns both of its layers: a view must
    // not inherit the last one's fills.
    view.fills = {};
    markerLayer.setMarkers([]);
    el('exposure-wrap').hidden = true;

    drawBars(payload.bars);
    drawStateRibbon(payload);
    renderLadder();
    view.stream = streams[datasetSel.value] || null;
    el('live').hidden = !view.stream;
    openSocket();

    renderProvenance(payload.provenance);
    view.shifts = shiftsFrom(payload.why.states, payload.bars);
    renderShifts();
    renderTrades();
    view.pinned = null;
    renderBar(payload.bars.length - 1);
    document.title = payload.provenance.identity.symbol + ' · state · strategy-lab';
    el('title').textContent = payload.provenance.identity.symbol;
    applyBarSpacing(payload);
  }

  function syncExitEnabled() {
    var applies = exitModeApplies();
    exitSel.disabled = !applies;
    exitSel.title = applies ? '' :
      'the continuous-exposure contract has no exit mode: a target of 0.0 is the exit';
  }

  function setView(name) {
    // Hidden rather than unmounted, so the price chart keeps its size and the
    // zoom survives a trip through the board.
    document.body.className = name + '-view';
    viewSel.value = name;
    syncStrategyChoices();
    // Whichever view is being left stops writing into the one being entered.
    // Neither switch starts a request on the side it leaves, so neither guard
    // fires on its own: leaving the board would keep appending tiles to a
    // hidden host and finishing a full recompute per dataset, and leaving the
    // instrument view would draw a chart over the board and retitle the page.
    if (name === 'board') { abandonInstrument(); closeSocket(); } else abortBoard();
    return reload();
  }

  function reload() {
    if (viewSel.value === 'board') return loadBoard();
    return viewSel.value === 'state' ? loadState() : load();
  }

  // `exactBounds` belongs to the tile that opened the view, and only to that
  // dataset: the window is a *funding* span, which is per contract. So a
  // strategy or exit-mode change keeps it -- same instrument, same funded
  // range, and the board itself gives every strategy on a dataset one window --
  // while picking a different dataset, or entering the instrument view without
  // a tile, must drop it or the request carries the previous instrument's edges
  // and asks for a range this one may not have funding for. `setView` assigns
  // `viewSel.value` without firing `change`, so a tile's own bounds survive it.
  viewSel.addEventListener('change', function () {
    exactBounds = null;
    setView(viewSel.value);
  });
  marketSel.addEventListener('change', loadBoard);
  el('refresh-all').addEventListener('click', refreshAll);
  datasetSel.addEventListener('change', function () {
    exactBounds = null;
    reload();
  });
  strategySel.addEventListener('change', function () { syncExitEnabled(); reload(); });
  exitSel.addEventListener('change', reload);
  [el('start'), el('end')].forEach(function (control) {
    control.addEventListener('change', function () {
      // Editing a date makes the visible dates the truth, so the tile's exact
      // edges stop applying -- otherwise a user who moved `From` would still be
      // sent the tile's original bound and see their own change ignored.
      exactBounds = null;
      reload();
    });
  });
  function refreshInstrument() {
    // The view can change while the fetch is in flight, and this one is not
    // always started by a click -- a closing bar starts it too. Reloading then
    // draws a chart over the board and retitles the page, which is what
    // `abandonInstrument` prevents for requests already running but cannot for
    // one begun afterwards.
    var startedIn = viewSel.value;
    var button = el('refresh');
    button.disabled = true;
    setStatus('fetching newest bars …');
    var params = new URLSearchParams(identity());
    var last = view.bars[view.bars.length - 1];
    if (last) params.set('after', String(last.time));
    return getJSON('/api/refresh?' + params.toString(), { method: 'POST' })
      // The counts exist so drift is visible at the moment it opens, not two
      // clicks later as a 409. A perp that moved candles and no settlements is
      // exactly the state the coverage guard will refuse next, and thrown away
      // here it looked identical to a clean refresh.
      .then(function (result) {
        view.refreshed = { view: 'instrument', text: refreshText(result) };
        // The tile's edges described the frame *before* this fetch. Only the
        // right one is stale: kept, the reload re-requests the old `end` and
        // draws no new bars while the status reports the candles it just took,
        // and the board -- recomputing against the new funding window -- then
        // shows a later bar than the chart it links to.
        //
        // The *head* bound stays. It is what holds the frame off a permanent
        // leading funding gap, and dropping it falls back to a date rounded up
        // to the next whole UTC day. Clearing the whole thing was the first fix
        // here and it was wrong twice over: the date input still held the old
        // day, so the window merely widened to that day's 23:59:59 -- no newer
        // bars, and up to a full day past the last settlement, which the
        // coverage guard tolerates only to one funding cadence.
        exactBounds = { start: exactBounds && exactBounds.start, end: null };
        el('end').value = '';
        if (viewSel.value !== startedIn) return null;
        return reload();
      })
      .catch(function (error) { setError(error.message); })
      .then(function () { button.disabled = false; });
  }

  el('refresh').addEventListener('click', refreshInstrument);

  // A rung switches to the stored dataset for that timeframe, and offers to
  // fetch it when there is none. Never a client-side resample: a 1h bar built
  // from four 15m bars is not the venue's 1h bar, and this repo keys a dataset
  // on its timeframe precisely so the two cannot be confused.
  function datasetFor(timeframe) {
    var id = identity();
    return [id.exchange, id.market_type, id.symbol, timeframe].join('|');
  }

  function renderLadder() {
    var host = el('ladder');
    host.replaceChildren();
    var id = identity();
    var current = id.timeframe;
    // The four rungs, plus every timeframe this instrument already has stored,
    // so switching away from one never strands it in the dropdown.
    var rungs = (CFG.timeframeLadders[id.exchange] || CFG.defaultTimeframeLadder).slice();
    Array.prototype.forEach.call(datasetSel.options, function (option) {
      var parts = option.value.split('|');
      var mine = parts[0] === id.exchange && parts[1] === id.market_type &&
        parts[2] === id.symbol;
      if (mine && rungs.indexOf(parts[3]) < 0) rungs.push(parts[3]);
    });
    if (rungs.indexOf(current) < 0) rungs.push(current);
    rungs.sort(function (a, b) {
      return CFG.timeframeOrder.indexOf(a) - CFG.timeframeOrder.indexOf(b);
    });
    rungs.forEach(function (timeframe) {
      var wanted = datasetFor(timeframe);
      var stored = Object.prototype.hasOwnProperty.call(streams, wanted) ||
        Array.prototype.some.call(datasetSel.options, function (o) { return o.value === wanted; });
      // Warmup is counted in *bars*, so the same strategy is comfortable on one
      // rung and permanently impossible on another: 2,192 bars is 365 days at 4h
      // and 42 years at 1w. Better said on the rung than discovered by clicking
      // it and reading a refusal.
      var warmup = Number(
        strategySel.selectedOptions[0] &&
        strategySel.selectedOptions[0].dataset.warmupBars
      );
      var depth = depths[wanted];
      var shallow = stored && warmup && depth !== undefined && depth <= warmup;
      var button = document.createElement('button');
      button.type = 'button';
      button.textContent = timeframe;
      button.className = timeframe === current
        ? 'current'
        : (stored ? (shallow ? 'shallow' : '') : 'absent');
      button.title = !stored
        ? timeframe + ' is not stored for this instrument yet — click to fetch it'
        : shallow
          ? strategySel.value + ' needs ' + warmup.toLocaleString() +
            ' bars before its first state and this set has ' +
            depth.toLocaleString()
          : 'switch to ' + timeframe;
      button.addEventListener('click', function () { switchTimeframe(timeframe, stored); });
      host.appendChild(button);
    });
  }

  function switchTimeframe(timeframe, stored) {
    var wanted = datasetFor(timeframe);
    if (stored) {
      datasetSel.value = wanted;
      exactBounds = null;
      return reload();
    }
    setStatus('fetching ' + timeframe + ' candles …');
    var parts = wanted.split('|');
    var params = new URLSearchParams({
      exchange: parts[0], market_type: parts[1], symbol: parts[2], timeframe: parts[3]
    });
    // From this frame's own left edge, so the new timeframe covers what is on
    // screen rather than the five bars a top-up would take.
    var first = view.bars && view.bars[0];
    if (first) params.set('since', new Date(first.time * 1000).toISOString());
    return getJSON('/api/refresh?' + params.toString(), { method: 'POST' })
      .then(function () { return getJSON('/api/datasets'); })
      .then(function (rows) {
        datasetSel.replaceChildren();
        fillDatasets(rows);
        datasetSel.value = wanted;
        exactBounds = null;
        return reload();
      })
      .catch(function (error) { setError(error.message); });
  }

  Object.keys(CFG.stateColors).forEach(function (name) {
    // A `button`, not a `span`: the explanation has to be reachable without a
    // mouse, and `title` alone is hover-only. Focusing one shows the same text.
    var swatch = document.createElement('button');
    swatch.type = 'button';
    swatch.className = 'swatch';
    swatch.title = CFG.stateDescriptions[name];
    var dot = document.createElement('i');
    dot.style.background = CFG.stateColors[name];
    swatch.appendChild(dot);
    swatch.appendChild(document.createTextNode(name));
    swatch.addEventListener('click', function () { explainState(name); });
    swatch.addEventListener('focus', function () { explainState(name); });
    el('state-swatches').appendChild(swatch);
  });

  function explainState(name) {
    var note = el('state-explainer');
    note.replaceChildren();
    var dot = document.createElement('i');
    dot.style.background = CFG.stateColors[name];
    var label = document.createElement('b');
    label.textContent = name;
    note.appendChild(dot);
    note.appendChild(label);
    note.appendChild(document.createTextNode(' — ' + CFG.stateDescriptions[name]));
    note.hidden = false;
  }

  fillMarkets();
  fillExitModes();
  Promise.all([getJSON('/api/datasets'), getJSON('/api/strategies')])
    .then(function (results) {
      fillDatasets(results[0]);
      fillStrategies(results[1]);
      if (!results[0].length) {
        setError('no candle sets stored — fetch some first');
        return;
      }
      syncExitEnabled();
      // The board first: it is the view that answers "what is everything
      // doing", and the instrument view is what you open from it.
      return setView('board');
    })
    .catch(function (error) { setError(error.message); });
})();
</script>
</body>
</html>
"""

__all__ = [
    "CONTRACT_PRIMITIVES",
    "DEFAULT_MARKET_TYPE",
    "MARKET_TYPES",
    "RESTATED_MARKET_TYPE",
    "RESTATEMENT_STALE_DAYS",
    "bootstrap_config",
    "render_browser_html",
]
