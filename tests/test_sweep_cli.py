"""The sweep command, exercised without a database.

``load_candles`` is patched to serve an in-memory frame, so these tests cover the
wiring the command owns -- grid parsing, the report directory, both written
artifacts, and what it echoes -- rather than Postgres.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from strategy_lab import cli
from tests.conftest import synthetic_ohlcv

runner = CliRunner()


@pytest.fixture
def candles(monkeypatch):
    df = synthetic_ohlcv(n=900)
    monkeypatch.setattr(cli, "load_candles", lambda **kwargs: df)
    return df


def _invoke(tmp_path, *args):
    return runner.invoke(
        cli.app, ["sweep", "--report-root", str(tmp_path), *args]
    )


def test_sweep_writes_both_artifacts_and_echoes_the_score(candles, tmp_path):
    result = _invoke(
        tmp_path, "--strategy", "donchian", "--grid", '{"entry_span":[24,48],"exit_span":[12,24]}'
    )

    assert result.exit_code == 0, result.output
    [report_dir] = list(tmp_path.iterdir())
    assert (report_dir / "sweep.html").exists()
    assert (report_dir / "points.json").exists()
    assert "donchian_sweep" in report_dir.name

    echoed = " ".join(result.output.split())
    assert "Stability score:" in echoed
    assert "/4 cells with positive Sharpe" in echoed


def test_points_json_is_the_full_reproducibility_record(candles, tmp_path):
    result = _invoke(
        tmp_path, "--strategy", "tsmom", "--grid", '{"lookback":[24,48,96]}',
        "--symbol", "ETH/USDT", "--timeframe", "1h", "--market-type", "perp",
    )

    assert result.exit_code == 0, result.output
    [report_dir] = list(tmp_path.iterdir())
    record = json.loads((report_dir / "points.json").read_text())

    assert record["config"]["grid"] == {"lookback": [24, 48, 96]}
    assert record["config"]["strategy"] == "tsmom"
    assert record["config"]["identity"]["symbol"] == "ETH/USDT"
    assert record["config"]["identity"]["timeframe"] == "1h"
    assert record["config"]["candle_count"] == 900
    assert isinstance(record["stability_score"], float)
    assert [point["params"] for point in record["points"]] == [
        {"lookback": 24},
        {"lookback": 48},
        {"lookback": 96},
    ]
    assert all(
        set(point) == {"params", "total_return", "sharpe", "max_drawdown", "trades"}
        for point in record["points"]
    )


def test_the_written_html_is_the_rendered_surface(candles, tmp_path):
    result = _invoke(tmp_path, "--strategy", "tsmom", "--grid", '{"lookback":[24,48]}')

    assert result.exit_code == 0, result.output
    [report_dir] = list(tmp_path.iterdir())
    html = (report_dir / "sweep.html").read_text()

    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "stability" in html.lower()
    assert "<script src=" not in html
    # The echoed score and the score on the page must be the same number.
    score = json.loads((report_dir / "points.json").read_text())["stability_score"]
    assert f"{score:.3f}" in html
    assert f"Stability score: {score:.3f}" in result.output


@pytest.mark.parametrize(
    "bad_grid",
    ["not json", '["entry_span"]', "{}", '{"entry_span":[]}', '{"entry_span":42}'],
)
def test_a_malformed_grid_exits_non_zero(candles, tmp_path, bad_grid):
    result = _invoke(tmp_path, "--strategy", "donchian", "--grid", bad_grid)

    assert result.exit_code != 0
    assert not list(tmp_path.iterdir()), "a rejected grid must not leave a report behind"


def test_a_parameter_the_strategy_does_not_have_exits_non_zero(candles, tmp_path):
    result = _invoke(tmp_path, "--strategy", "tsmom", "--grid", '{"entry_span":[24]}')

    assert result.exit_code != 0
    assert "does not accept parameter" in result.output
    assert not list(tmp_path.iterdir())
