from __future__ import annotations

import re

import pytest

from strategy_lab.backtests.sweep import SweepPoint
from strategy_lab.backtests.sweep_report import render_sweep_html


def _points() -> list[SweepPoint]:
    return [
        SweepPoint(
            {"entry_span": e, "exit_span": x},
            total_return=0.1 * e,
            sharpe=0.5,
            max_drawdown=-0.2,
            trades=12,
        )
        for e in (24, 48)
        for x in (12, 24)
    ]


def _cells(html: str) -> list[str]:
    return re.findall(r'<td class="cell"[^>]*>.*?</td>', html, flags=re.S)


def test_report_is_self_contained():
    html = render_sweep_html(points=_points(), config={"strategy": "donchian"})
    assert "<script src=" not in html, "must not reference external assets"
    assert "<link" not in html
    assert "http://" not in html and "https://" not in html
    assert html.lstrip().startswith("<!DOCTYPE html>")


def test_report_renders_every_grid_cell():
    points = _points()
    html = render_sweep_html(points=points, config={"strategy": "donchian"})

    assert len(_cells(html)) == len(points), "one rendered cell per swept combination"
    for span in ("24", "48", "12"):
        assert span in html
    assert "donchian" in html


def test_report_handles_a_one_dimensional_grid():
    points = [
        SweepPoint({"lookback": n}, total_return=0.1, sharpe=0.4, max_drawdown=-0.1, trades=5)
        for n in (24, 48, 96)
    ]
    html = render_sweep_html(points=points, config={"strategy": "tsmom"})

    assert "96" in html
    assert len(_cells(html)) == 3
    # A 1-D grid lays out as a single row: one <tr> of cells, so every rendered
    # cell shares it.
    body_rows = re.findall(r"<tr>.*?</tr>", html, flags=re.S)
    rows_with_cells = [row for row in body_rows if 'class="cell"' in row]
    assert len(rows_with_cells) == 1


def test_cells_are_coloured_by_the_sign_and_size_of_sharpe():
    """Colour is what answers "plateau or spike" before any number is read."""
    points = [
        SweepPoint({"n": 0}, total_return=-0.3, sharpe=-2.0, max_drawdown=-0.4, trades=9),
        SweepPoint({"n": 1}, total_return=0.02, sharpe=0.1, max_drawdown=-0.1, trades=9),
        SweepPoint({"n": 2}, total_return=0.9, sharpe=2.0, max_drawdown=-0.1, trades=9),
    ]
    losing, faint, winning = _cells(render_sweep_html(points=points, config={}))

    def alpha(cell: str) -> float:
        return float(re.search(r"rgba\([\d\s,]+,\s*([\d.]+)\)", cell).group(1))

    assert "239, 83, 80" in losing, "a negative Sharpe must use the down colour"
    assert "38, 166, 154" in winning, "a positive Sharpe must use the up colour"
    assert "38, 166, 154" in faint
    assert alpha(winning) > alpha(faint), "a stronger Sharpe must read as a stronger cell"
    assert alpha(losing) == pytest.approx(alpha(winning)), (
        "the scale must be symmetric, or a spike and a crater look different sizes"
    )


def test_report_declares_that_the_numbers_are_gross_of_costs():
    """Measured on 83,348 BTC 15m bars: the best tsmom cell reads +0.74 Sharpe
    here and -100% once the engine's default 10bp per side is charged, because it
    flips 8,545 times. Showing the first number without saying so is exactly the
    flattering report this page exists to prevent.
    """
    html = render_sweep_html(points=_points(), config={"strategy": "donchian"}).lower()

    assert "gross of fees" in html
    assert "slippage" in html


def test_report_escapes_interpolated_values():
    html = render_sweep_html(
        points=_points(), config={"strategy": "<script>alert(1)</script>"}
    )
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_report_refuses_to_emit_a_nan_payload():
    """A NaN serializes to invalid JSON that fails silently in the browser."""
    points = [
        SweepPoint({"n": n}, total_return=0.1, sharpe=float("nan"), max_drawdown=-0.1, trades=5)
        for n in (1, 2)
    ]
    with pytest.raises(ValueError):
        render_sweep_html(points=points, config={"strategy": "tsmom"})


def test_report_rejects_an_empty_sweep():
    with pytest.raises(ValueError, match="no points"):
        render_sweep_html(points=[], config={"strategy": "tsmom"})
