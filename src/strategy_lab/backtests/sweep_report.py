from __future__ import annotations

import json
import statistics
from html import escape

from strategy_lab.backtests.sweep import SweepPoint, stability_score

_UP = "#26a69a"
_DOWN = "#ef5350"
_UP_RGB = "38, 166, 154"
_DOWN_RGB = "239, 83, 80"

# A cell at the surface's strongest |Sharpe| reads at _ALPHA_MAX; one at zero
# still reads at _ALPHA_MIN so an empty cell is distinguishable from no cell.
_ALPHA_MIN = 0.08
_ALPHA_MAX = 0.92


def _fmt_sharpe(value: float) -> str:
    return f"{value:+.2f}"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:+,.1f}%"


def _fmt_score(value: float) -> str:
    return f"{value:.3f}"


def _fmt_param(value: object) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _cell_color(sharpe: float, scale: float) -> str:
    """Diverging fill: teal above zero, red below, strength by |Sharpe|.

    Scaled against the surface's own extreme rather than an absolute number, so
    the page answers "how does this cell compare to its neighbours" -- a plateau
    fills as one solid block, a spike as a single bright cell in a dark field.
    """
    weight = min(abs(sharpe) / scale, 1.0) if scale > 0 else 0.0
    alpha = _ALPHA_MIN + (_ALPHA_MAX - _ALPHA_MIN) * weight
    rgb = _UP_RGB if sharpe >= 0 else _DOWN_RGB
    return f"rgba({rgb}, {alpha:.3f})"


def _axes(points: list[SweepPoint]) -> tuple[str | None, list[str]]:
    """Split the swept parameters into a row axis and one or more column axes.

    One dimension lays out as a single row -- there are no neighbours in a
    second direction to show. Three or more fold every axis after the first into
    the column label rather than silently dropping one.
    """
    names = list(points[0].params)
    if len(names) <= 1:
        return None, names
    return names[0], names[1:]


def _ordered(values: list) -> list:
    """Grid order, de-duplicated, preserving first appearance.

    Not sorted: ``itertools.product`` already emits the caller's declared order,
    which reads the way the grid was written and works for values that are not
    sortable against each other.
    """
    seen: list = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _grid_table(points: list[SweepPoint], scale: float) -> str:
    row_axis, col_axes = _axes(points)

    def col_key(point: SweepPoint) -> tuple:
        return tuple(point.params[name] for name in col_axes)

    # Values only -- the axis names live once, in the corner cell.
    def col_label(key: tuple) -> str:
        return " · ".join(_fmt_param(v) for v in key)

    columns = _ordered([col_key(p) for p in points])
    rows = _ordered([p.params[row_axis] for p in points]) if row_axis else [None]
    by_position = {
        ((p.params[row_axis] if row_axis else None), col_key(p)): (index, p)
        for index, p in enumerate(points)
    }

    header = "".join(f"<th>{escape(col_label(key))}</th>" for key in columns)
    across = " · ".join(col_axes)
    corner = escape(f"{row_axis} ↓ / {across} →" if row_axis else f"{across} →")

    body = []
    for row in rows:
        label = _fmt_param(row) if row_axis else ""
        cells = [f'<th class="row-label">{escape(label)}</th>']
        for key in columns:
            found = by_position.get((row, key))
            if found is None:
                cells.append('<td class="cell empty"></td>')
                continue
            index, point = found
            cells.append(
                f'<td class="cell" data-point="{index}" '
                f'style="background: {_cell_color(point.sharpe, scale)}">'
                f'<span class="sharpe">{escape(_fmt_sharpe(point.sharpe))}</span>'
                f'<span class="ret">{escape(_fmt_pct(point.total_return))}</span>'
                "</td>"
            )
        body.append("<tr>" + "".join(cells) + "</tr>")

    return (
        f'<table><thead><tr><th class="corner">{corner}</th>{header}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _chips(points: list[SweepPoint]) -> str:
    sharpes = [p.sharpe for p in points]
    best = max(points, key=lambda p: p.sharpe)
    best_params = " · ".join(f"{k}={_fmt_param(v)}" for k, v in best.params.items())
    positive = sum(1 for s in sharpes if s > 0)

    chips = [
        ("Stability", _fmt_score(stability_score(points)), ""),
        ("Best Sharpe", _fmt_sharpe(best.sharpe), _sign_class(best.sharpe)),
        ("Median Sharpe", _fmt_sharpe(statistics.median(sharpes)), _sign_class(
            statistics.median(sharpes)
        )),
        ("Positive Cells", f"{positive}/{len(points)}", ""),
        ("Best Cell", best_params, ""),
    ]
    return "\n".join(
        f'<div class="chip"><span class="chip-label">{escape(label)}</span>'
        f'<span class="chip-value{cls}">{escape(value)}</span></div>'
        for label, value, cls in chips
    )


def _sign_class(value: float) -> str:
    return " up" if value >= 0 else " down"


def render_sweep_html(*, points: list[SweepPoint], config: dict) -> str:
    """Render a parameter sweep as a self-contained stability surface."""
    if not points:
        raise ValueError("Cannot render a sweep with no points")

    identity = config.get("identity") or {}
    strategy = str(config.get("strategy", ""))
    meta_bits = [
        identity.get("exchange", ""),
        identity.get("market_type", ""),
        identity.get("timeframe", ""),
        strategy,
        f"{len(points)} cells",
    ]
    heading = str(identity.get("symbol", "")) or strategy or "parameter sweep"
    date_range = " → ".join(
        str(config[key]) for key in ("data_start", "data_end") if config.get(key)
    )

    scale = max((abs(p.sharpe) for p in points), default=0.0)
    # allow_nan=False is deliberate: NaN is not valid JSON, and a browser that
    # cannot parse the payload drops the hover detail silently rather than
    # complaining. Failing here instead makes a NaN Sharpe loud.
    payload = json.dumps(
        [
            {
                "params": {str(k): v for k, v in p.params.items()},
                "sharpe": p.sharpe,
                "total_return": p.total_return,
                "max_drawdown": p.max_drawdown,
                "trades": p.trades,
            }
            for p in points
        ],
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")

    return (
        _TEMPLATE.replace("__HEADING__", escape(heading))
        .replace("__META__", escape(" · ".join(str(b) for b in meta_bits if b)))
        .replace("__RANGE__", escape(date_range))
        .replace("__CHIPS__", _chips(points))
        .replace("__GRID__", _grid_table(points, scale))
        .replace("__SCALE__", escape(_fmt_sharpe(scale)))
        .replace("__NEG_SCALE__", escape(_fmt_sharpe(-scale)))
        .replace("__UP__", _UP)
        .replace("__DOWN__", _DOWN)
        .replace("__PAYLOAD__", payload)
    )


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__HEADING__ sweep · strategy-lab</title>
<style>
  :root {
    --bg: #131722; --panel: #1e222d; --border: #2a2e39; --border-soft: #232733;
    --ink: #d1d4dc; --ink-dim: #787b86; --up: __UP__; --down: __DOWN__;
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
  .up { color: var(--up); } .down { color: var(--down); }
  section { padding: 4px 20px 20px; }
  h2 { font-size: 13px; color: var(--ink-dim); text-transform: uppercase;
    letter-spacing: 0.8px; margin: 10px 0; }
  .table-wrap {
    overflow-x: auto; border: 1px solid var(--border-soft); border-radius: 8px;
    background: var(--panel);
  }
  table { border-collapse: separate; border-spacing: 3px; width: 100%; }
  th {
    color: var(--ink-dim); font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.7px; padding: 8px 10px; white-space: nowrap; text-align: center;
  }
  th.corner, th.row-label { text-align: right; }
  td.cell {
    border-radius: 5px; padding: 12px 10px; min-width: 92px; text-align: center;
    cursor: default; display: table-cell;
  }
  td.cell.empty { background: var(--bg); }
  .sharpe { display: block; font-size: 15px; font-weight: 700; }
  .ret { display: block; font-size: 11px; color: rgba(209, 212, 220, 0.75); }
  #detail {
    margin-top: 12px; padding: 9px 12px; border: 1px solid var(--border-soft);
    border-radius: 6px; background: var(--panel); color: var(--ink-dim);
    font-size: 12px; min-height: 34px; display: flex; gap: 18px; flex-wrap: wrap;
  }
  #detail b { color: var(--ink); font-weight: 600; }
  .legend { display: flex; align-items: center; gap: 8px; padding: 12px 20px 0;
    color: var(--ink-dim); font-size: 11px; }
  .ramp {
    width: 220px; height: 9px; border-radius: 999px;
    background: linear-gradient(to right,
      rgba(239, 83, 80, 0.92), rgba(239, 83, 80, 0.08),
      rgba(38, 166, 154, 0.08), rgba(38, 166, 154, 0.92));
  }
  footer { padding: 0 20px 22px; color: var(--ink-dim); font-size: 11px; }
</style>
</head>
<body>
<header>
  <h1>__HEADING__</h1>
  <span class="meta">__META__</span>
  <span class="range">__RANGE__</span>
</header>
<div class="chips">
__CHIPS__
</div>
<div class="legend">
  <span>__NEG_SCALE__</span><span class="ramp"></span><span>__SCALE__</span>
  <span>Sharpe · a plateau fills as one block, an overfit as a single bright cell</span>
</div>
<section>
  <h2>Stability surface</h2>
  <div class="table-wrap">
__GRID__
  </div>
  <div id="detail">Hover a cell for its full metrics.</div>
</section>
<footer>
  strategy-lab parameter sweep · a broad stable region beats a lone spike ·
  <strong>gross of fees and slippage</strong> — a high-turnover cell can be the best
  here and a total loss once costs are charged
</footer>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
(function () {
  var P = JSON.parse(document.getElementById('payload').textContent);
  var detail = document.getElementById('detail');

  function figure(label, value) {
    var wrap = document.createElement('span');
    wrap.textContent = label + ' ';
    var strong = document.createElement('b');
    strong.textContent = value;
    wrap.appendChild(strong);
    return wrap;
  }

  function pct(value) {
    return (value >= 0 ? '+' : '') + (value * 100).toFixed(1) + '%';
  }

  document.querySelectorAll('td.cell[data-point]').forEach(function (cell) {
    cell.addEventListener('mouseenter', function () {
      var point = P[Number(cell.dataset.point)];
      if (!point) return;
      detail.replaceChildren();
      Object.keys(point.params).forEach(function (key) {
        detail.appendChild(figure(key, String(point.params[key])));
      });
      detail.appendChild(figure('Sharpe', point.sharpe.toFixed(2)));
      detail.appendChild(figure('Return', pct(point.total_return)));
      detail.appendChild(figure('Max DD', pct(point.max_drawdown)));
      detail.appendChild(figure('Trades', String(point.trades)));
    });
  });
})();
</script>
</body>
</html>
"""
