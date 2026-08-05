"""The page: one chart, a strategy switcher, the "why" behind a bar, provenance.

Nothing on this page is server-rendered from data: the switcher changes strategy
without a reload, so every figure arrives from ``/api/analysis`` and is drawn
client side. Measured end to end on the largest stored frame -- BTC/USDT perp 4h,
15,114 bars -- a switch is 650-750 ms, of which ~500 ms is server and network,
15 ms is JSON parse, and the rest is handing 15k candles to the chart. On a
438-bar weekly set it is 136 ms. Fluid rather than instant, and there is
deliberately no frame cache: caching a frame beside a refresh endpoint that
mutates it needs invalidation, and a stale frame is a worse bug than half a
second.

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
* **the styling**, imported from ``backtests/report.py``: the frozen report and
  the live view are the same product and must not drift into looking like two.

The page is inert without the API behind it. It fetches, it draws, and it POSTs
to the one endpoint that writes -- it has no path of its own to ``reports/``,
``signals`` or ``market_candles``.
"""

from __future__ import annotations

import json

from strategy_lab.api.analysis import Contract
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


def bootstrap_config() -> dict[str, object]:
    """Everything the page script must not decide for itself, as one JSON blob."""
    return {
        "primitives": CONTRACT_PRIMITIVES,
        "exitModeContracts": list(CONTRACTS_WITH_EXIT_MODE),
        "exitModes": [mode.value for mode in ExitMode],
        "colors": {"up": UP, "down": DOWN, "upDim": UP_DIM, "downDim": DOWN_DIM},
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
  .why-chips { display: flex; gap: 8px; flex-wrap: wrap; }
  .why-chips .chip { min-width: 84px; }
  .why-chips .chip.state { border-color: #4a5163; background: #232733; }
  .why-chips .chip.absent .chip-value { color: var(--ink-dim); }
  .note { color: var(--ink-dim); font-size: 11.5px; margin: 0; max-width: 82ch; }
  #status { color: var(--ink-dim); font-size: 11.5px; }
  footer { padding: 14px 20px 20px; color: var(--ink-dim); font-size: 11px; }
</style>
</head>
<body>
<header>
  <h1 id="title">strategy-lab</h1>
  <div class="controls">
    <div class="field">
      <label for="dataset">Candle set</label>
      <select id="dataset"></select>
    </div>
    <div class="field">
      <label for="strategy">Strategy</label>
      <select id="strategy"></select>
    </div>
    <div class="field">
      <label for="exit-mode">Exit mode</label>
      <select id="exit-mode"></select>
    </div>
    <div class="field">
      <label for="start">From</label>
      <input id="start" type="date">
    </div>
    <div class="field">
      <label for="end">To</label>
      <input id="end" type="date">
    </div>
    <div class="field">
      <label for="refresh">Candles</label>
      <button id="refresh" type="button" title="Fetch the newest bars and recompute">
        refresh
      </button>
    </div>
  </div>
  <span class="range" id="status"></span>
</header>
<p class="banner" id="error" hidden></p>
<div class="provenance" id="provenance"></div>
<div class="charts">
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
<section class="why">
  <h2>Why this bar <span class="pin" id="why-bar"></span></h2>
  <div class="why-chips" id="why-chips"></div>
  <p class="note" id="why-note"></p>
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
  var volumeSeries = priceChart.addSeries(LWC.HistogramSeries, {
    priceFormat: { type: 'volume' }, lastValueVisible: false, priceLineVisible: false
  }, 1);
  priceChart.panes()[1].setHeight(80);
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
    pinned: null, dataset: null
  };

  function setStatus(text) {
    statusEl.textContent = text;
  }

  function setError(message) {
    // A failed request leaves the previous chart up, which is the right thing to
    // do -- and makes the controls describe something other than what is drawn.
    // The provenance strip still tells the truth, so it is dimmed rather than
    // cleared: what you are looking at, and that it is not what you asked for.
    el('error').textContent = message || '';
    el('error').hidden = !message;
    el('provenance').classList.toggle('stale', Boolean(message));
    if (message) setStatus('showing the last run that succeeded');
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

  function fillExitModes() {
    // "engine default" rather than this page's guess at one: which mode a run
    // used is then read off the provenance strip, where it is a measured fact.
    exitSel.appendChild(option(ENGINE_DEFAULT, 'engine default'));
    CFG.exitModes.forEach(function (mode) {
      exitSel.appendChild(option(mode, mode));
    });
  }

  function fillDatasets(rows) {
    rows.forEach(function (row) {
      var id = [row.exchange, row.market_type, row.symbol, row.timeframe].join('|');
      datasetSel.appendChild(option(
        id,
        row.symbol + ' · ' + row.timeframe + ' · ' + row.exchange + '/' +
        row.market_type + ' · ' + row.candles.toLocaleString() + ' bars'
      ));
    });
  }

  function fillStrategies(rows) {
    rows.forEach(function (row) {
      var opt = option(row.name, row.name + '  (' + row.contract + ')');
      opt.dataset.contract = row.contract;
      strategySel.appendChild(opt);
    });
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
    // Both assignments are unconditional so neither contract can inherit the
    // other's leftovers: a level is never drawn as a set of fills, and a set of
    // fills is never drawn as a level.
    markerLayer.setMarkers(primitive === 'markers' ? toMarkers(payload.markers) : []);
    var levels = primitive === 'baseline' ? toLevels(payload.target, payload.bars) : [];
    exposureSeries.setData(levels);
    el('exposure-wrap').hidden = primitive !== 'baseline';

    renderProvenance(payload.provenance);
    view.pinned = null;
    renderBar(payload.bars.length - 1);
    document.title = payload.provenance.identity.symbol + ' · ' +
      payload.provenance.strategy + ' · strategy-lab';
    el('title').textContent = payload.provenance.identity.symbol;

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
    var crowdingBlind = perp && !prov.crowding_measured;

    host.appendChild(chip('Strategy', prov.strategy + ' v' + prov.version));
    host.appendChild(chip('Contract', prov.contract));
    host.appendChild(chip(
      'Exit mode', prov.exit_mode || 'none · a target of 0.0 is the exit'
    ));
    if (prov.failure_bars !== null) {
      host.appendChild(chip('Failure bars', String(prov.failure_bars)));
    }
    host.appendChild(chip('Warmup', prov.warmup_bars.toLocaleString() + ' bars'));
    host.appendChild(chip('Shorts', prov.allow_shorts ? 'allowed' : 'long only'));
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
    host.appendChild(chip('Cost model', costText(prov.cost_model)));
    host.appendChild(chip(
      'Frame',
      prov.first_bar.slice(0, 10) + ' → ' + prov.last_bar.slice(0, 10) + '  (' +
      prov.bar_count.toLocaleString() + ' bars)'
    ));
    host.appendChild(chip('Computed', prov.generated_at.slice(0, 19).replace('T', ' ')));

    if (crowdingBlind) {
      var alert = document.createElement('p');
      alert.className = 'alert';
      // Both branches say only what is true of *this* strategy. "Crowding-
      // neutral variant" is a claim about a strategy that reads crowding, and
      // said about one that does not it invents a difference.
      alert.textContent = prov.funding_attached
        ? 'Perp, funding attached, and still no crowding behind these bars: ' +
          prov.strategy + ' reads no funding-derived feature, so what is drawn ' +
          'here is what it would do on a venue that never settled.'
        : 'Perp with no funding column on this frame. ' + prov.strategy + ' reads ' +
          'no funding-derived feature, so its own output is unaffected — but ' +
          'nothing here is comparable with a funded run, and the charter\\'s R5 ' +
          'figures are the crowding-measured ones. A strategy that does read ' +
          'funding is refused on this frame rather than falling back to neutral.';
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
    el('why-bar').textContent = new Date(bar.time * 1000).toISOString()
      .slice(0, 16).replace('T', ' ') + ' UTC' + (view.pinned === i ? ' · pinned' : '');

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

  priceChart.subscribeCrosshairMove(function (param) {
    if (view.pinned !== null) return;
    var i = param.time !== undefined ? view.index[param.time] : undefined;
    renderBar(i === undefined ? view.bars.length - 1 : i);
  });
  priceChart.subscribeClick(function (param) {
    var i = param.time !== undefined ? view.index[param.time] : undefined;
    if (i === undefined) return;
    view.pinned = view.pinned === i ? null : i;
    renderBar(i);
  });

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
          payload.bars.length.toLocaleString() + ' bars');
      })
      .catch(function (error) {
        if (token === pending) setError(error.message);
      });
  }

  function syncExitEnabled() {
    var applies = exitModeApplies();
    exitSel.disabled = !applies;
    exitSel.title = applies ? '' :
      'the continuous-exposure contract has no exit mode: a target of 0.0 is the exit';
  }

  datasetSel.addEventListener('change', load);
  strategySel.addEventListener('change', function () { syncExitEnabled(); load(); });
  [exitSel, el('start'), el('end')].forEach(function (control) {
    control.addEventListener('change', load);
  });
  el('refresh').addEventListener('click', function () {
    var button = el('refresh');
    button.disabled = true;
    setStatus('fetching newest bars …');
    var params = new URLSearchParams(identity());
    var last = view.bars[view.bars.length - 1];
    if (last) params.set('after', String(last.time));
    getJSON('/api/refresh?' + params.toString(), { method: 'POST' })
      .then(load)
      .catch(function (error) { setError(error.message); })
      .then(function () { button.disabled = false; });
  });

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
      return load();
    })
    .catch(function (error) { setError(error.message); });
})();
</script>
</body>
</html>
"""

__all__ = ["CONTRACT_PRIMITIVES", "bootstrap_config", "render_browser_html"]
