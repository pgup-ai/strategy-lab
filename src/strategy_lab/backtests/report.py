from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd

_ASSET_DIR = Path(__file__).resolve().parent / "assets"
_LIB_FILENAME = "lightweight-charts-5.2.0.standalone.production.js"

_UP = "#26a69a"
_DOWN = "#ef5350"
_UP_DIM = "rgba(38, 166, 154, 0.45)"
_DOWN_DIM = "rgba(239, 83, 80, 0.45)"

# A funded run stores no bare path statistic -- each names its curve -- so
# exactly one of the first two rows renders: the gross key on a funded run, the
# plain one everywhere else.
_STAT_KEYS = [
    ("Total Return [%] (gross of funding)", "Gross of Funding", "signed_pct"),
    ("Total Return [%]", "Total Return", "signed_pct"),
    ("Net Return [%]", "Net Return", "signed_pct"),
    ("Benchmark Return [%]", "Benchmark", "signed_pct"),
    ("Win Rate [%]", "Win Rate", "pct"),
    ("Max Drawdown [%]", "Max Drawdown", "pct"),
    ("Sharpe Ratio", "Sharpe", "num"),
    ("Profit Factor", "Profit Factor", "num"),
    ("Total Trades", "Trades", "int"),
]

_RISK_ROWS = [
    ("Max Drawdown [%]", "Max Drawdown", "pct"),
    ("Annualized Volatility [%]", "Annualized Volatility", "pct"),
    ("Sharpe Ratio", "Sharpe", "num"),
    ("Sortino Ratio", "Sortino", "num"),
    ("Calmar Ratio", "Calmar", "num"),
]


def _format_duration(delta: pd.Timedelta) -> str:
    parts = delta.components
    if parts.days:
        return f"{parts.days}d {parts.hours}h" if parts.hours else f"{parts.days}d"
    if parts.hours:
        return f"{parts.hours}h {parts.minutes}m" if parts.minutes else f"{parts.hours}h"
    return f"{parts.minutes}m"


def _fmt_price(value: float) -> str:
    return f"{value:,.2f}"


def _fmt_qty(value: float) -> str:
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def _fmt_signed(value: float, suffix: str = "") -> str:
    return f"{value:+,.2f}{suffix}"


def _fmt_stat(value: object, kind: str) -> str:
    if isinstance(value, float) and (value != value):
        return "—"
    if isinstance(value, float) and value in (float("inf"), float("-inf")):
        return "∞"
    if kind == "signed_pct":
        return f"{value:+,.2f}%"
    if kind == "pct":
        return f"{value:,.2f}%"
    if kind == "int":
        return f"{int(value)}"
    if isinstance(value, (int, float)):
        return f"{value:,.2f}"
    return str(value)


def _pnl_class(value: float) -> str:
    return "up" if value >= 0 else "down"


def _fmt_money(value: float) -> str:
    return f"{value:,.2f}"


def _fmt_multiple(value: float) -> str:
    return f"{value:g}x"


def _cost_chip(label: str, text: str, cls: str = "", extra: str = "") -> str:
    return (
        f'<div class="chip{extra}"><span class="chip-label">{escape(label)}</span>'
        f'<span class="chip-value{cls}">{escape(text)}</span></div>'
    )


def _cost_flow(base: dict, funding_applied: bool) -> str:
    """Gross minus each cost equals net, spelled out left to right.

    Funding gets the emphasised chip because it is the cost that decides whether
    a perp result is tradeable, and the one a reader is most likely to skip past
    on the way to the headline return.
    """
    funding_paid = float(base["funding_paid"])
    funding_label = "Funding Received" if funding_paid < 0 else "Funding Paid"
    funding_text = _fmt_money(abs(funding_paid)) if funding_applied else "not modelled"
    funding_cls = "" if not funding_applied else (" down" if funding_paid > 0 else " up")

    parts = [
        _cost_chip(
            "Gross Return",
            _fmt_stat(base["gross_return_pct"], "signed_pct"),
            f" {_pnl_class(float(base['gross_return_pct']))}",
        ),
        '<span class="op">&minus;</span>',
        _cost_chip("Fees", _fmt_money(float(base["fees_paid"]))),
        '<span class="op">&minus;</span>',
        _cost_chip("Slippage", _fmt_money(float(base["slippage_paid"]))),
        '<span class="op">&minus;</span>',
        _cost_chip(funding_label, funding_text, funding_cls, extra=" key"),
        '<span class="op">=</span>',
        _cost_chip(
            "Net Return",
            _fmt_stat(base["net_return_pct"], "signed_pct"),
            f" {_pnl_class(float(base['net_return_pct']))}",
            extra=" key",
        ),
    ]
    return "\n".join(parts)


def _stress_rows(stress: list[dict], funding_applied: bool) -> str:
    rows = []
    for row in stress:
        multiple = float(row["multiple"])
        net = float(row["net_return_pct"])
        funding = _fmt_money(float(row["funding_paid"])) if funding_applied else "—"
        marker = ' class="base"' if multiple == 1.0 else ""
        rows.append(
            f"<tr{marker}>"
            f"<td>{escape(_fmt_multiple(multiple))}</td>"
            f"<td>{escape(_fmt_money(float(row['fees_paid'])))}</td>"
            f"<td>{escape(_fmt_money(float(row['slippage_paid'])))}</td>"
            f"<td>{escape(funding)}</td>"
            f'<td class="{_pnl_class(float(row["gross_return_pct"]))}">'
            f"{escape(_fmt_stat(row['gross_return_pct'], 'signed_pct'))}</td>"
            f'<td class="{_pnl_class(net)}">'
            f"{escape(_fmt_stat(net, 'signed_pct'))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _risk_table(stats: dict) -> str:
    """Every risk statistic funding moves, gross beside net.

    The curve plotted above is the net one while ``pf.stats()`` measures the
    simulated book, which never sees a settlement. Publishing only one of the
    two is how a drawdown gets quoted off a curve nobody is looking at.
    """
    rows = []
    for key, label, kind in _RISK_ROWS:
        gross = stats.get(f"{key} (gross of funding)")
        net = stats.get(f"{key} (net of funding)")
        if gross is None or net is None:
            continue
        rows.append(
            f"<tr><td>{escape(label)}</td>"
            f"<td>{escape(_fmt_stat(gross, kind))}</td>"
            f"<td>{escape(_fmt_stat(net, kind))}</td></tr>"
        )
    if not rows:
        return ""
    return (
        '<div class="table-wrap cost-table">'
        "<table><thead><tr>"
        "<th>Risk</th><th>Gross of funding</th><th>Net of funding</th>"
        "</tr></thead><tbody>"
        f"{''.join(rows)}"
        "</tbody></table></div>"
    )


def _cost_section(costs: dict | None, stats: dict) -> str:
    if not costs or not costs.get("stress"):
        return ""

    stress = costs["stress"]
    funding_applied = bool(costs.get("funding_applied"))
    base = next(row for row in stress if float(row["multiple"]) == 1.0)

    note = (
        "Funding is charged at each venue settlement against the notional held "
        "into that bar. It is a market rate, so cost stress scales fees and "
        "slippage only. The benchmark above is price-only: a perpetual long "
        "pays this same funding for every bar it is held."
        if funding_applied
        else "No funding series was supplied, so these returns are gross of "
        "carry. On a perpetual future that is not a tradeable number."
    )
    table = ""
    if len(stress) > 1:
        table = (
            '<div class="table-wrap cost-table">'
            "<table><thead><tr>"
            "<th>Cost stress</th><th>Fees</th><th>Slippage</th><th>Funding</th>"
            "<th>Gross Return</th><th>Net Return</th>"
            "</tr></thead><tbody>"
            f"{_stress_rows(stress, funding_applied)}"
            "</tbody></table></div>"
        )
    warn = "" if funding_applied else ' <span class="warn">gross of funding</span>'
    return (
        '<section class="costs">'
        f"<h2>Costs{warn}</h2>"
        f'<div class="flow">{_cost_flow(base, funding_applied)}</div>'
        f'<p class="note">{escape(note)}</p>'
        f"{_risk_table(stats)}"
        f"{table}"
        "</section>"
    )


def _build_payload(
    df: pd.DataFrame, trades: pd.DataFrame, equity: pd.Series, identity: dict
) -> dict:
    times = [int(ts.timestamp()) for ts in df.index]
    opens, highs = df["open"].astype(float), df["high"].astype(float)
    lows, closes = df["low"].astype(float), df["close"].astype(float)
    volumes = df["volume"].astype(float).fillna(0.0)

    candles = [
        {"time": t, "open": o, "high": h, "low": lo, "close": c}
        for t, o, h, lo, c in zip(times, opens, highs, lows, closes)
    ]
    volume = [
        {"time": t, "value": v, "color": _UP_DIM if c >= o else _DOWN_DIM}
        for t, v, o, c in zip(times, volumes, opens, closes)
    ]
    equity_points = [
        {"time": int(ts.timestamp()), "value": float(v)} for ts, v in equity.items()
    ]

    markers = []
    zoom_ranges = []
    for order, (_, trade) in enumerate(trades.iterrows()):
        is_long = trade["Direction"] == "Long"
        entry_time = int(pd.Timestamp(trade["Entry Timestamp"]).timestamp())
        entry_side = "B" if is_long else "S"
        markers.append(
            {
                "time": entry_time,
                "position": "belowBar" if is_long else "aboveBar",
                "color": _UP if is_long else _DOWN,
                "shape": "arrowUp" if is_long else "arrowDown",
                "text": f"{entry_side} {_fmt_price(float(trade['Avg Entry Price']))}",
                "seq": order * 2,
            }
        )
        exit_time = entry_time
        if trade["Status"] == "Closed":
            exit_time = int(pd.Timestamp(trade["Exit Timestamp"]).timestamp())
            exit_side = "S" if is_long else "B"
            markers.append(
                {
                    "time": exit_time,
                    "position": "aboveBar" if is_long else "belowBar",
                    "color": _DOWN if is_long else _UP,
                    "shape": "arrowDown" if is_long else "arrowUp",
                    "text": f"{exit_side} {_fmt_price(float(trade['Avg Exit Price']))}",
                    "seq": order * 2 + 1,
                }
            )
        zoom_ranges.append({"from": entry_time, "to": exit_time})

    markers.sort(key=lambda m: (m["time"], m.pop("seq")))

    bar_seconds = 604_800
    if len(times) > 1:
        deltas = pd.Series(times).diff().dropna()
        bar_seconds = int(deltas.median())

    return {
        "candles": candles,
        "volume": volume,
        "equity": equity_points,
        "markers": markers,
        "zoom": zoom_ranges,
        "barSeconds": bar_seconds,
        "intraday": bar_seconds < 86_400,
        "identity": {
            "exchange": str(identity.get("exchange", "")),
            "market_type": str(identity.get("market_type", "")),
            "symbol": str(identity.get("symbol", "")),
            "timeframe": str(identity.get("timeframe", "")),
        },
        "colors": {"upDim": _UP_DIM, "downDim": _DOWN_DIM},
    }


def _trade_rows(trades: pd.DataFrame) -> str:
    if trades.empty:
        return '<tr><td class="empty" colspan="12">No trades in this run</td></tr>'

    rows = []
    cumulative = 0.0
    for order, (_, trade) in enumerate(trades.iterrows(), start=1):
        is_open = trade["Status"] != "Closed"
        direction = str(trade["Direction"])
        entry_ts = pd.Timestamp(trade["Entry Timestamp"])
        exit_ts = pd.Timestamp(trade["Exit Timestamp"])
        pnl = float(trade["PnL"])
        ret = float(trade["Return"])
        cumulative += pnl
        held = _format_duration(exit_ts - entry_ts)
        exit_date = "—" if is_open else escape(f"{exit_ts:%Y-%m-%d}")
        exit_price = "—" if is_open else _fmt_price(float(trade["Avg Exit Price"]))
        status = "Open" if is_open else "Closed"
        rows.append(
            f'<tr data-trade="{order - 1}">'
            f"<td>{order}</td>"
            f'<td><span class="pill {direction.lower()}">{escape(direction)}</span></td>'
            f"<td>{_fmt_qty(float(trade['Size']))}</td>"
            f"<td>{escape(f'{entry_ts:%Y-%m-%d}')}</td>"
            f"<td>{_fmt_price(float(trade['Avg Entry Price']))}</td>"
            f"<td>{exit_date}</td>"
            f"<td>{exit_price}</td>"
            f"<td>{held}</td>"
            f'<td class="{_pnl_class(pnl)}">{_fmt_signed(pnl)}</td>'
            f'<td class="{_pnl_class(ret)}">{_fmt_signed(ret * 100, "%")}</td>'
            f'<td class="{_pnl_class(cumulative)}">{_fmt_signed(cumulative)}</td>'
            f'<td><span class="status {status.lower()}">{status}</span></td>'
            "</tr>"
        )
    return "\n".join(rows)


def _stat_lookup(stats: dict, key: str) -> tuple[str, object] | None:
    """``(label suffix, value)`` for ``key``, following the funded-run rename.

    A funded run splits every path statistic into a gross and a net key, so a
    chip either finds the plain name or finds the net-of-funding variant -- and
    when it is the latter the chip says so, because the drawdown a reader takes
    away must match the curve drawn beside it.
    """
    if key in stats:
        return "", stats[key]
    net = f"{key} (net of funding)"
    if net in stats:
        return " (net)", stats[net]
    return None


def _stat_chips(stats: dict) -> str:
    chips = []
    for key, label, kind in _STAT_KEYS:
        found = _stat_lookup(stats, key)
        if found is None:
            continue
        suffix, value = found
        label += suffix
        cls = ""
        if kind == "signed_pct" and isinstance(value, (int, float)) and value == value:
            cls = f" {_pnl_class(float(value))}"
        extra = " key" if key == "Net Return [%]" else ""
        chips.append(
            f'<div class="chip{extra}"><span class="chip-label">{escape(label)}</span>'
            f'<span class="chip-value{cls}">{escape(_fmt_stat(value, kind))}</span></div>'
        )
    return "\n".join(chips)


def render_report_html(
    *,
    df: pd.DataFrame,
    trades: pd.DataFrame,
    equity: pd.Series,
    config: dict,
    stats: dict,
    costs: dict | None = None,
) -> str:
    identity = config.get("identity", {})
    symbol = str(identity.get("symbol", ""))
    meta_bits = [
        identity.get("exchange", ""),
        identity.get("market_type", ""),
        identity.get("timeframe", ""),
        config.get("strategy", ""),
        f"exit: {config.get('exit_mode', '')}",
    ]
    date_range = f"{df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}"

    payload = json.dumps(
        _build_payload(df, trades, equity, identity),
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")

    lib_source = (_ASSET_DIR / _LIB_FILENAME).read_text(encoding="utf-8")

    page = (
        _TEMPLATE.replace("__SYMBOL__", escape(symbol))
        .replace("__META__", escape(" · ".join(str(b) for b in meta_bits if b)))
        .replace("__RANGE__", escape(date_range))
        .replace("__CHIPS__", _stat_chips(stats))
        .replace("__COSTS__", _cost_section(costs, stats))
        .replace("__ROWS__", _trade_rows(trades))
        .replace("__PAYLOAD__", payload)
    )
    return page.replace("__LIB__", lib_source)


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__SYMBOL__ · strategy-lab</title>
<style>
  :root {
    --bg: #131722; --panel: #1e222d; --border: #2a2e39; --border-soft: #232733;
    --ink: #d1d4dc; --ink-dim: #787b86; --up: #26a69a; --down: #ef5350;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--ink); min-height: 100vh;
    font: 13px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-variant-numeric: tabular-nums;
  }
  header {
    display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;
    padding: 14px 20px 10px; border-bottom: 1px solid var(--border);
  }
  h1 { font-size: 20px; font-weight: 700; letter-spacing: 0.2px; }
  .meta { color: var(--ink-dim); font-size: 12px; }
  .range { margin-left: auto; color: var(--ink-dim); font-size: 12px; }
  .chips { display: flex; gap: 10px; flex-wrap: wrap; padding: 12px 20px; }
  .chip {
    background: var(--panel); border: 1px solid var(--border-soft); border-radius: 6px;
    padding: 7px 12px; display: flex; flex-direction: column; gap: 2px; min-width: 96px;
  }
  .chip-label { color: var(--ink-dim); font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.7px; }
  .chip-value { font-size: 15px; font-weight: 600; }
  .chips .chip.key { border-color: #4a5163; background: #232733; }
  .up { color: var(--up); } .down { color: var(--down); }
  .charts { padding: 0 20px; }
  .pane {
    position: relative; border: 1px solid var(--border-soft); border-radius: 8px;
    overflow: hidden; background: var(--bg);
  }
  #price-pane { height: 460px; }
  #equity-pane { height: 170px; margin-top: 10px; }
  .pane-tag {
    position: absolute; top: 8px; left: 10px; z-index: 3; color: var(--ink-dim);
    font-size: 11px; letter-spacing: 0.4px;
  }
  #legend {
    position: absolute; top: 24px; left: 10px; z-index: 3; font-size: 12px;
    color: var(--ink-dim); display: flex; gap: 10px;
  }
  #legend b { color: var(--ink); font-weight: 600; }
  section.trades { padding: 16px 20px 28px; }
  h2 { font-size: 13px; color: var(--ink-dim); text-transform: uppercase;
    letter-spacing: 0.8px; margin: 0 0 10px; }
  .table-wrap {
    overflow-x: auto; border: 1px solid var(--border-soft); border-radius: 8px;
    background: var(--panel);
  }
  table { border-collapse: collapse; width: 100%; min-width: 900px; }
  th, td { padding: 8px 12px; text-align: right; white-space: nowrap; }
  th {
    color: var(--ink-dim); font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.7px; border-bottom: 1px solid var(--border);
    position: sticky; top: 0; background: var(--panel);
  }
  th:nth-child(-n+2), td:nth-child(-n+2) { text-align: left; }
  td { border-bottom: 1px solid var(--border-soft); font-size: 12.5px; }
  tbody tr { cursor: pointer; }
  tbody tr:hover { background: rgba(41, 98, 255, 0.07); }
  tbody tr.active { background: rgba(41, 98, 255, 0.14); }
  td.empty { text-align: center; color: var(--ink-dim); padding: 22px; }
  .pill { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  .pill.long { color: var(--up); background: rgba(38, 166, 154, 0.12); }
  .pill.short { color: var(--down); background: rgba(239, 83, 80, 0.12); }
  .status { font-size: 11px; color: var(--ink-dim); }
  .status.open { color: #ff9800; }
  section.costs { padding: 18px 20px 0; }
  section.costs .warn {
    color: #ff9800; font-size: 11px; letter-spacing: 0.4px; margin-left: 6px;
  }
  .flow { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  .flow .op { color: var(--ink-dim); font-size: 15px; padding: 0 2px; }
  .flow .chip.key { border-color: #4a5163; background: #232733; }
  .note { color: var(--ink-dim); font-size: 11.5px; margin: 10px 0 0; max-width: 76ch; }
  .cost-table { margin-top: 12px; }
  .cost-table table { min-width: 620px; }
  .cost-table tbody tr { cursor: default; }
  .cost-table tbody tr:hover { background: none; }
  .cost-table tbody tr.base td { color: var(--ink); font-weight: 600; }
  #live {
    display: none; align-items: center; gap: 6px; margin-left: 12px;
    background: var(--panel); border: 1px solid var(--border-soft); border-radius: 999px;
    color: var(--ink-dim); font-size: 11px; padding: 4px 10px; cursor: pointer;
  }
  #live.visible { display: inline-flex; }
  #live .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--ink-dim); }
  #live.on .dot { background: var(--up); animation: pulse 2s ease-in-out infinite; }
  #live.err .dot { background: var(--down); }
  @keyframes pulse { 50% { opacity: 0.35; } }
  footer { padding: 0 20px 20px; color: var(--ink-dim); font-size: 11px; }
</style>
</head>
<body>
<header>
  <h1>__SYMBOL__</h1>
  <span class="meta">__META__</span>
  <button id="live" type="button" title="Refresh now">
    <span class="dot"></span><span id="live-text">delayed feed</span>
  </button>
  <span class="range">__RANGE__</span>
</header>
<div class="chips">
__CHIPS__
</div>
<div class="charts">
  <div class="pane" id="price-pane">
    <span class="pane-tag">PRICE · VOLUME</span>
    <div id="legend"></div>
  </div>
  <div class="pane" id="equity-pane"><span class="pane-tag">EQUITY</span></div>
</div>
__COSTS__
<section class="trades">
  <h2>Trades</h2>
  <div class="table-wrap">
  <table>
    <thead><tr>
      <th>#</th><th>Side</th><th>Size</th><th>Entry</th><th>Entry Px</th>
      <th>Exit</th><th>Exit Px</th><th>Held</th><th>PnL</th><th>Return</th>
      <th>Cum PnL</th><th>Status</th>
    </tr></thead>
    <tbody id="trade-body">
__ROWS__
    </tbody>
  </table>
  </div>
</section>
<footer>strategy-lab backtest report · charts by TradingView Lightweight Charts™</footer>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>__LIB__</script>
<script>
(function () {
  var P = JSON.parse(document.getElementById('payload').textContent);
  var LWC = LightweightCharts;
  var theme = {
    autoSize: true,
    layout: {
      background: { type: 'solid', color: '#131722' },
      textColor: '#d1d4dc',
      fontSize: 11,
      attributionLogo: false
    },
    grid: {
      vertLines: { color: '#1e222d' },
      horzLines: { color: '#1e222d' }
    },
    crosshair: {
      mode: 0,
      vertLine: { color: '#787b86', labelBackgroundColor: '#2a2e39' },
      horzLine: { color: '#787b86', labelBackgroundColor: '#2a2e39' }
    },
    rightPriceScale: { borderColor: '#2a2e39' },
    timeScale: {
      borderColor: '#2a2e39',
      timeVisible: P.intraday,
      secondsVisible: false
    }
  };

  var priceChart = LWC.createChart(document.getElementById('price-pane'), theme);
  var candleSeries = priceChart.addSeries(LWC.CandlestickSeries, {
    upColor: '#26a69a', downColor: '#ef5350', borderVisible: false,
    wickUpColor: '#26a69a', wickDownColor: '#ef5350'
  });
  candleSeries.setData(P.candles);
  candleSeries.priceScale().applyOptions({ scaleMargins: { top: 0.08, bottom: 0.06 } });

  var volumeSeries = priceChart.addSeries(LWC.HistogramSeries, {
    priceFormat: { type: 'volume' }, lastValueVisible: false,
    priceLineVisible: false
  }, 1);
  volumeSeries.setData(P.volume);
  priceChart.panes()[1].setHeight(88);

  LWC.createSeriesMarkers(candleSeries, P.markers);

  var equityChart = LWC.createChart(document.getElementById('equity-pane'), theme);
  var equitySeries = equityChart.addSeries(LWC.AreaSeries, {
    lineColor: '#2962ff', lineWidth: 2,
    topColor: 'rgba(41, 98, 255, 0.26)', bottomColor: 'rgba(41, 98, 255, 0.02)',
    priceFormat: {
      type: 'custom',
      formatter: function (value) {
        if (Math.abs(value) >= 1e6) return (value / 1e6).toFixed(2) + 'M';
        if (Math.abs(value) >= 1e3) return (value / 1e3).toFixed(1) + 'K';
        return value.toFixed(0);
      }
    }
  });
  equitySeries.setData(P.equity);

  var byTimeClose = {};
  P.candles.forEach(function (c) { byTimeClose[c.time] = c; });
  var byTimeEquity = {};
  P.equity.forEach(function (p) { byTimeEquity[p.time] = p.value; });
  var byTimeVolume = {};
  P.volume.forEach(function (v) { byTimeVolume[v.time] = v.value; });

  var legend = document.getElementById('legend');
  function addFigure(label, value) {
    var wrap = document.createElement('span');
    wrap.textContent = label + ' ';
    var strong = document.createElement('b');
    strong.textContent = value;
    wrap.appendChild(strong);
    legend.appendChild(wrap);
  }
  function renderLegend(bar) {
    legend.replaceChildren();
    if (!bar) return;
    addFigure('O', bar.open.toFixed(2));
    addFigure('H', bar.high.toFixed(2));
    addFigure('L', bar.low.toFixed(2));
    addFigure('C', bar.close.toFixed(2));
    var chg = bar.close - bar.open;
    var pct = bar.open ? (chg / bar.open) * 100 : 0;
    var delta = document.createElement('span');
    delta.style.color = chg >= 0 ? '#26a69a' : '#ef5350';
    delta.textContent = (chg >= 0 ? '+' : '') + chg.toFixed(2) +
      ' (' + (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%)';
    legend.appendChild(delta);
    var vol = byTimeVolume[bar.time];
    if (vol !== undefined) addFigure('Vol', Math.round(vol).toLocaleString());
  }
  renderLegend(P.candles[P.candles.length - 1]);

  var syncing = false;
  function linkRange(from, to) {
    from.timeScale().subscribeVisibleLogicalRangeChange(function (range) {
      if (syncing || !range) return;
      syncing = true;
      to.timeScale().setVisibleLogicalRange(range);
      syncing = false;
    });
  }
  linkRange(priceChart, equityChart);
  linkRange(equityChart, priceChart);

  priceChart.subscribeCrosshairMove(function (param) {
    var bar = param.time !== undefined ? byTimeClose[param.time] : null;
    renderLegend(bar || P.candles[P.candles.length - 1]);
    if (param.time !== undefined && byTimeEquity[param.time] !== undefined) {
      equityChart.setCrosshairPosition(byTimeEquity[param.time], param.time, equitySeries);
    } else {
      equityChart.clearCrosshairPosition();
    }
  });
  equityChart.subscribeCrosshairMove(function (param) {
    if (param.time !== undefined && byTimeClose[param.time]) {
      var bar = byTimeClose[param.time];
      renderLegend(bar);
      priceChart.setCrosshairPosition(bar.close, param.time, candleSeries);
    } else {
      priceChart.clearCrosshairPosition();
    }
  });

  var body = document.getElementById('trade-body');
  body.addEventListener('click', function (event) {
    var row = event.target.closest('tr[data-trade]');
    if (!row) return;
    var zoom = P.zoom[Number(row.dataset.trade)];
    if (!zoom) return;
    var pad = P.barSeconds * 6;
    priceChart.timeScale().setVisibleRange({
      from: zoom.from - pad, to: zoom.to + pad
    });
    var active = body.querySelector('tr.active');
    if (active) active.classList.remove('active');
    row.classList.add('active');
  });

  function fitAll() {
    priceChart.timeScale().fitContent();
    equityChart.timeScale().fitContent();
  }
  fitAll();
  requestAnimationFrame(fitAll);

  var livePill = document.getElementById('live');
  var liveText = document.getElementById('live-text');
  var liveEnabled = false;

  function setLive(state, text) {
    livePill.className = 'visible ' + state;
    liveText.textContent = text;
  }

  function applyBars(bars) {
    var last = P.candles[P.candles.length - 1];
    bars.forEach(function (bar) {
      if (last && bar.time < last.time) return;
      var candle = {
        time: bar.time, open: bar.open, high: bar.high,
        low: bar.low, close: bar.close
      };
      candleSeries.update(candle);
      volumeSeries.update({
        time: bar.time, value: bar.volume,
        color: bar.close >= bar.open ? P.colors.upDim : P.colors.downDim
      });
      if (last && bar.time === last.time) {
        P.candles[P.candles.length - 1] = candle;
      } else {
        P.candles.push(candle);
      }
      byTimeClose[bar.time] = candle;
      byTimeVolume[bar.time] = bar.volume;
      last = P.candles[P.candles.length - 1];
    });
    renderLegend(P.candles[P.candles.length - 1]);
  }

  function refresh() {
    var last = P.candles[P.candles.length - 1];
    var params = new URLSearchParams(P.identity);
    if (last) params.set('after', String(last.time));
    return fetch('/api/candles?' + params.toString())
      .then(function (res) {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      })
      .then(function (data) {
        applyBars(data.bars || []);
        setLive('on', 'delayed · ' + new Date().toLocaleTimeString());
      });
  }

  if (location.protocol.indexOf('http') === 0) {
    refresh()
      .then(function () {
        liveEnabled = true;
        setInterval(function () {
          refresh().catch(function () { setLive('err', 'feed error'); });
        }, 60000);
      })
      .catch(function () { /* no API behind this page: stay static */ });
    livePill.addEventListener('click', function () {
      if (!liveEnabled) return;
      refresh().catch(function () { setLive('err', 'feed error'); });
    });
  }
})();
</script>
</body>
</html>
"""
