"""The univariate diagnostic as one page, in the same house style as the sweep.

One row per feature, one column group per horizon, and each IC cell carrying the
full-sample number over both halves -- ``features.diagnostics`` carries the
reasoning for why the halves are the reading.

Self-contained by the same rule as every other report here: no external asset,
no network, so a report opened in two years renders exactly as it did the day it
was written.
"""

from __future__ import annotations

import json
from html import escape

from strategy_lab.features.diagnostics import (
    REDUNDANT_CORRELATION,
    DiagnosticSet,
    FeatureDiagnostic,
    HorizonIC,
    to_record,
)

_UP_RGB = "38, 166, 154"
_DOWN_RGB = "239, 83, 80"

# A single feature on 4h crypto rarely reaches |IC| 0.05, so the colour ramp
# saturates there rather than at 1.0. Scaled to the page's own strongest cell
# instead, one feature with a lucky horizon would wash out every other row.
_IC_SCALE = 0.05
_ALPHA_MIN = 0.06
_ALPHA_MAX = 0.92

_DASH = "—"


def _fmt_ic(value: float) -> str:
    return _DASH if value != value else f"{value:+.3f}"


def _fmt_pct(value: float) -> str:
    return _DASH if value != value else f"{value * 100:.1f}%"


def _fmt_num(value: float) -> str:
    return _DASH if value != value else f"{value:.3f}"


def _fmt_halves(entry: HorizonIC) -> str:
    return f"{_fmt_ic(entry.first_half_ic)} / {_fmt_ic(entry.second_half_ic)}"


def _fmt_partner(partner: str, value: float) -> str:
    return _DASH if not partner or value != value else f"{_fmt_ic(value)} {partner}"


def _ic_color(value: float) -> str:
    """Diverging fill: teal for a positive IC, red for negative, by magnitude."""
    if value != value:
        return "transparent"
    weight = min(abs(value) / _IC_SCALE, 1.0)
    alpha = _ALPHA_MIN + (_ALPHA_MAX - _ALPHA_MIN) * weight
    rgb = _UP_RGB if value >= 0 else _DOWN_RGB
    return f"rgba({rgb}, {alpha:.3f})"


def _ic_cell(entry: HorizonIC) -> str:
    """One horizon for one feature: the full-sample IC over both half-sample ICs.

    Bordered when the halves disagree in sign, which is the whole reason both
    are on the page.
    """
    classes = "cell" if entry.halves_agree else "cell split"
    return (
        f'<td class="{classes}" style="background: {_ic_color(entry.ic)}">'
        f'<span class="ic">{escape(_fmt_ic(entry.ic))}</span>'
        f'<span class="halves">{escape(_fmt_halves(entry))}</span>'
        "</td>"
    )


def _row(diagnostic: FeatureDiagnostic, result: DiagnosticSet) -> str:
    partner, correlation = result.max_correlation(diagnostic.name)
    redundant = correlation == correlation and abs(correlation) >= REDUNDANT_CORRELATION
    cells = [
        f'<th class="row-label">{escape(diagnostic.name)}</th>',
        f'<td class="stat">{escape(_fmt_pct(diagnostic.coverage))}</td>',
        f'<td class="stat">{diagnostic.observations:,}</td>',
        f'<td class="stat">{escape(_fmt_num(diagnostic.minimum))}</td>',
        f'<td class="stat">{escape(_fmt_num(diagnostic.median))}</td>',
        f'<td class="stat">{escape(_fmt_num(diagnostic.maximum))}</td>',
        f'<td class="stat">{escape(_fmt_num(diagnostic.iqr))}</td>',
        f'<td class="stat">{escape(_fmt_num(diagnostic.autocorrelation))}</td>',
        f'<td class="stat">{escape(_fmt_num(diagnostic.turnover))}</td>',
        *[_ic_cell(entry) for entry in diagnostic.ics],
        f'<td class="stat{" flag" if redundant else ""}">'
        f"{escape(_fmt_partner(partner, correlation))}</td>",
    ]
    return "<tr>" + "".join(cells) + "</tr>"


def _table(result: DiagnosticSet) -> str:
    leading = ["feature", "coverage", "bars", "min", "median", "max", "IQR", "AC(1)", "turnover"]
    header = "".join(f"<th>{escape(name)}</th>" for name in leading)
    header += "".join(
        f'<th class="ic-head">IC · {horizon}b</th>' for horizon in result.horizons
    )
    header += "<th>max |r|</th>"
    body = "".join(_row(diagnostic, result) for diagnostic in result.diagnostics)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _redundancy(result: DiagnosticSet) -> str:
    """Pairs past the redundancy threshold, or a line saying there were none.

    Stated either way. A missing section reads as "not checked", which is the
    one thing this page must never imply.
    """
    pairs = result.redundant_pairs()
    if not pairs:
        return (
            f"<p class='note'>No pair reaches |r| {REDUNDANT_CORRELATION:.2f}. "
            "Every feature carries something the others do not.</p>"
        )
    rows = "".join(
        f"<li><b>{escape(first)}</b> · <b>{escape(second)}</b> "
        f"<span class='r'>r = {escape(_fmt_ic(value))}</span></li>"
        for first, second, value in pairs
    )
    return (
        f"<p class='note'>{len(pairs)} pair(s) at or past |r| "
        f"{REDUNDANT_CORRELATION:.2f} — one feature under two names, unless the "
        "duplication is deliberate.</p>"
        f"<ul class='pairs'>{rows}</ul>"
    )


def _chips(result: DiagnosticSet) -> str:
    scored = [
        (diagnostic.name, entry)
        for diagnostic in result.diagnostics
        for entry in diagnostic.ics
        if entry.ic == entry.ic
    ]
    agreeing = sum(
        1
        for diagnostic in result.diagnostics
        if any(entry.halves_agree for entry in diagnostic.ics)
    )
    best = _DASH
    if scored:
        owner, entry = max(scored, key=lambda pair: abs(pair[1].ic))
        best = f"{_fmt_ic(entry.ic)} {owner} @{entry.horizon}b"

    chips = [
        ("Features", str(len(result.diagnostics))),
        ("Horizons", " · ".join(f"{h}b" for h in result.horizons)),
        ("Strongest IC", best),
        ("Halves Agree", f"{agreeing}/{len(result.diagnostics)}"),
        ("Redundant Pairs", str(len(result.redundant_pairs()))),
    ]
    return "\n".join(
        f'<div class="chip"><span class="chip-label">{escape(label)}</span>'
        f'<span class="chip-value">{escape(value)}</span></div>'
        for label, value in chips
    )


def _payload(result: DiagnosticSet) -> str:
    """The numbers behind the page, as JSON -- the same record written to disk.

    ``allow_nan=False`` is an assertion: :func:`to_record` has already nulled
    every unmeasurable statistic, so anything left to reject is a bug.
    """
    return json.dumps(
        to_record(result)["features"],
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")


def render_diagnostics_html(*, result: DiagnosticSet, config: dict) -> str:
    """Render a feature diagnostic set as a self-contained page."""
    if not result.diagnostics:
        raise ValueError("Cannot render a diagnostic report with no features")

    identity = config.get("identity") or {}
    meta_bits = [
        identity.get("exchange", ""),
        identity.get("market_type", ""),
        identity.get("timeframe", ""),
        f"{config['candle_count']:,} bars" if config.get("candle_count") else "",
    ]
    heading = str(identity.get("symbol", "")) or "state features"
    date_range = " → ".join(
        str(config[key]) for key in ("data_start", "data_end") if config.get(key)
    )
    skipped = config.get("skipped") or {}
    skipped_html = "".join(
        f"<li><b>{escape(str(name))}</b> — {escape(str(reason))}</li>"
        for name, reason in skipped.items()
    )

    return (
        _TEMPLATE.replace("__HEADING__", escape(heading))
        .replace("__META__", escape(" · ".join(str(bit) for bit in meta_bits if bit)))
        .replace("__RANGE__", escape(date_range))
        .replace("__CHIPS__", _chips(result))
        .replace("__TABLE__", _table(result))
        .replace("__REDUNDANCY__", _redundancy(result))
        .replace(
            "__SKIPPED__",
            f"<h2>Not examined</h2><ul class='pairs'>{skipped_html}</ul>"
            if skipped_html
            else "",
        )
        .replace("__IC_SCALE__", escape(f"{_IC_SCALE:+.3f}"))
        .replace("__NEG_IC_SCALE__", escape(f"{-_IC_SCALE:+.3f}"))
        .replace("__PAYLOAD__", _payload(result))
    )


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__HEADING__ state features · strategy-lab</title>
<style>
  :root {
    --bg: #131722; --panel: #1e222d; --border: #2a2e39; --border-soft: #232733;
    --ink: #d1d4dc; --ink-dim: #787b86; --down: #ef5350;
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
  section { padding: 4px 20px 20px; }
  h2 { font-size: 13px; color: var(--ink-dim); text-transform: uppercase;
    letter-spacing: 0.8px; margin: 16px 0 10px; }
  .table-wrap {
    overflow-x: auto; border: 1px solid var(--border-soft); border-radius: 8px;
    background: var(--panel);
  }
  table { border-collapse: separate; border-spacing: 3px; width: 100%; }
  th {
    color: var(--ink-dim); font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.7px; padding: 8px 10px; white-space: nowrap; text-align: center;
  }
  th.ic-head { color: var(--ink); }
  th.row-label { text-align: right; color: var(--ink); font-size: 12px;
    text-transform: none; letter-spacing: 0; }
  td.stat {
    text-align: right; padding: 10px; white-space: nowrap; font-size: 12px;
    color: rgba(209, 212, 220, 0.85);
  }
  td.stat.flag { color: var(--down); font-weight: 600; }
  td.cell {
    border-radius: 5px; padding: 9px 10px; min-width: 96px; text-align: center;
    cursor: default;
  }
  td.cell.split { outline: 1px dashed rgba(239, 83, 80, 0.65); outline-offset: -1px; }
  .ic { display: block; font-size: 14px; font-weight: 700; }
  .halves { display: block; font-size: 10px; color: rgba(209, 212, 220, 0.7); }
  .note { color: var(--ink-dim); font-size: 12px; margin: 4px 0 8px; }
  .pairs { list-style: none; display: flex; gap: 10px; flex-wrap: wrap; }
  .pairs li {
    background: var(--panel); border: 1px solid var(--border-soft); border-radius: 6px;
    padding: 7px 12px; font-size: 12px;
  }
  .pairs .r { color: var(--ink-dim); }
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
      rgba(239, 83, 80, 0.92), rgba(239, 83, 80, 0.06),
      rgba(38, 166, 154, 0.06), rgba(38, 166, 154, 0.92));
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
  <span>__NEG_IC_SCALE__</span><span class="ramp"></span><span>__IC_SCALE__</span>
  <span>Spearman IC against the forward return over [t+1, t+1+h] · small numbers
  are the norm · a dashed cell is one whose two halves disagree in sign</span>
</div>
<section>
  <h2>Univariate diagnostics</h2>
  <div class="table-wrap">
__TABLE__
  </div>
  <div id="detail">Hover a feature for what the table has no column for.</div>
  <h2>Redundancy</h2>
__REDUNDANCY__
__SKIPPED__
</section>
<footer>
  strategy-lab feature diagnostics · IC is measured against the return from the
  <strong>next</strong> bar's close, never the feature's own · the two half-sample
  numbers under each IC are the reading; the full-sample one above them is a summary
  that hides a regime
</footer>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
(function () {
  var P = JSON.parse(document.getElementById('payload').textContent);
  var byName = {};
  P.forEach(function (row) { byName[row.name] = row; });
  var detail = document.getElementById('detail');

  function figure(label, value) {
    var wrap = document.createElement('span');
    wrap.textContent = label + ' ';
    var strong = document.createElement('b');
    strong.textContent = value;
    wrap.appendChild(strong);
    return wrap;
  }

  function num(value, digits) {
    if (value === null) return '\\u2014';
    return (value >= 0 ? '+' : '') + value.toFixed(digits);
  }

  document.querySelectorAll('tbody tr').forEach(function (row) {
    var label = row.querySelector('th.row-label');
    if (!label) return;
    row.addEventListener('mouseenter', function () {
      var record = byName[label.textContent];
      if (!record) return;
      detail.replaceChildren();
      detail.appendChild(figure('version', record.version));
      detail.appendChild(figure('warmup', String(record.warmup_bars) + 'b'));
      record.ic.forEach(function (entry) {
        detail.appendChild(figure(
          'IC@' + entry.horizon + 'b',
          num(entry.ic, 4) + ' (' + num(entry.first_half_ic, 3) + ' / ' +
            num(entry.second_half_ic, 3) + ' over ' + entry.observations + ' bars)'
        ));
      });
    });
  });
})();
</script>
</body>
</html>
"""
