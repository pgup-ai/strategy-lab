"""The sweep command, exercised without a database.

``load_candles`` is patched to serve an in-memory frame, so these tests cover the
wiring the command owns -- grid parsing, the funding column, the report
directory, both written artifacts, and what it echoes -- rather than Postgres.

The sweep is gross of costs on purpose, so funding is never *charged* here. It
is still attached, because ``crowding`` reads it as an input and a cell scored
without it is scoring a different strategy.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from strategy_lab import cli
from strategy_lab.features.flow import FUNDING_COLUMN
from tests.conftest import synthetic_ohlcv, synthetic_ohlcv_with_funding

runner = CliRunner()

# state_machine_v1 is the only registered strategy that reads funding, and at
# rank_window 480 it warms 2,192 bars.
_MACHINE_BARS = 2400
_MACHINE_GRID = '{"rank_window":[440,480]}'
_PERP = ["--market-type", "perp", "--symbol", "BTC/USDT", "--timeframe", "4h"]


@pytest.fixture
def candles(monkeypatch):
    df = synthetic_ohlcv(n=900)
    monkeypatch.setattr(cli, "load_candles", lambda **kwargs: df)
    return df


@pytest.fixture
def settlements(monkeypatch):
    """Stored funding for the perp path, served without a database."""
    df = synthetic_ohlcv(n=900)
    rates = synthetic_ohlcv_with_funding(n=900)[FUNDING_COLUMN]
    monkeypatch.setattr(cli, "load_candles", lambda **kwargs: df)
    monkeypatch.setattr(cli, "_funding_rates", lambda identity, df: rates[rates != 0.0])
    return df


@pytest.fixture
def machine_candles(monkeypatch):
    """A perp frame past ``state_machine_v1``'s warmup, served with no funding column.

    ``market_candles`` holds raw OHLCV and nothing derived, so a fixture that
    handed the column over would let the command skip attaching it and leave the
    assertions green.
    """
    df = synthetic_ohlcv_with_funding(n=_MACHINE_BARS, freq="4h")
    rates = df[FUNDING_COLUMN]
    monkeypatch.setattr(cli, "load_candles", lambda **kwargs: df.drop(columns=FUNDING_COLUMN))
    monkeypatch.setattr(cli, "_funding_rates", lambda identity, df: rates[rates != 0.0])
    return df


def _invoke(tmp_path, *args):
    return runner.invoke(
        cli.app, ["sweep", "--report-root", str(tmp_path), *args]
    )


def _record(tmp_path) -> dict:
    [report_dir] = list(tmp_path.iterdir())
    return json.loads((report_dir / "points.json").read_text())


def test_points_json_is_the_full_reproducibility_record(settlements, tmp_path):
    result = _invoke(
        tmp_path, "--strategy", "tsmom", "--grid", '{"lookback":[24,48,96]}',
        "--symbol", "ETH/USDT", "--timeframe", "1h", "--market-type", "perp",
    )

    assert result.exit_code == 0, result.output
    [report_dir] = list(tmp_path.iterdir())
    assert "tsmom_sweep" in report_dir.name
    record = json.loads((report_dir / "points.json").read_text())

    assert record["config"]["grid"] == {"lookback": [24, 48, 96]}
    assert record["config"]["strategy"] == "tsmom"
    assert record["config"]["identity"]["symbol"] == "ETH/USDT"
    assert record["config"]["identity"]["timeframe"] == "1h"
    assert record["config"]["candle_count"] == 900
    assert record["config"]["funding_attached"] is True
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

    assert "stability" in html.lower()
    # The score in points.json, the score on the page, and the echoed score must
    # all be the same number.
    score = json.loads((report_dir / "points.json").read_text())["stability_score"]
    assert f"{score:.3f}" in html
    assert f"Stability score: {score:.3f}" in result.output
    assert "/2 cells with positive Sharpe" in " ".join(result.output.split())


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


def test_a_perp_sweep_scores_every_cell_with_crowding_measured(machine_candles, tmp_path):
    """**The regression.** A cell scored on a frame with no ``funding_rate``
    column is scoring a strategy running on three features and a neutral fourth,
    and a ``SweepPoint`` carries no metadata to record that in -- so the surface
    would look like a study of ``state_machine_v1`` while being a study of
    something else. What proves the column arrived is the surface moving.

    Nothing here is a cost difference: this path never charges funding at all.
    """
    assert _invoke(
        tmp_path / "measured", "--strategy", "state_machine_v1", "--grid", _MACHINE_GRID, *_PERP
    ).exit_code == 0
    assert _invoke(
        tmp_path / "neutral", "--strategy", "state_machine_v1", "--grid", _MACHINE_GRID,
        "--no-funding", *_PERP,
    ).exit_code == 0

    measured, neutral = _record(tmp_path / "measured"), _record(tmp_path / "neutral")
    assert measured["config"]["funding_attached"] is True
    assert neutral["config"]["funding_attached"] is False
    assert measured["points"] != neutral["points"]


def test_a_spot_sweep_attaches_nothing_because_there_is_nothing_to_attach(candles, tmp_path):
    result = _invoke(tmp_path, "--strategy", "tsmom", "--grid", '{"lookback":[24]}')

    assert result.exit_code == 0, result.output
    assert _record(tmp_path)["config"]["funding_attached"] is False
