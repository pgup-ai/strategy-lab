"""The features command, exercised without a database.

``load_candles`` is patched to serve an in-memory frame, so these cover the
wiring the command owns -- horizon parsing, funding attachment, which features a
frame can support, the report directory, both artifacts and what it echoes --
rather than Postgres.

The frame is 4h and long enough to clear Direction's 1,920-bar warmup, since a
command that quietly diagnoses eight of nine features is exactly the failure the
R4 gate exists to prevent.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest
from typer.testing import CliRunner

from strategy_lab import cli
from strategy_lab.features.registry import list_features
from tests.conftest import synthetic_ohlcv, synthetic_ohlcv_with_funding

runner = CliRunner()

# Direction needs 1,920 bars of warmup and the diagnostic needs a sample past it.
_BARS = 2600


@pytest.fixture
def candles(monkeypatch):
    df = synthetic_ohlcv(n=_BARS, freq="4h")
    monkeypatch.setattr(cli, "load_candles", lambda **kwargs: df)
    return df


@pytest.fixture
def funded(monkeypatch, candles):
    """The perp path: stored funding, aligned to bars by the real helper."""
    funding = synthetic_ohlcv_with_funding(n=_BARS, freq="4h")["funding_rate"]
    settlements = funding[funding != 0.0]
    monkeypatch.setattr(cli, "_funding_rates", lambda identity, df, **_: settlements)
    return settlements


def _invoke(tmp_path, *args):
    return runner.invoke(cli.app, ["features", "--report-root", str(tmp_path), *args])


def test_every_registered_feature_gets_a_row_on_a_funded_perp_frame(funded, tmp_path):
    """The R4 gate through the CLI: nine registered, nine diagnosed, none skipped."""
    result = _invoke(tmp_path, "--horizons", "1,6")

    assert result.exit_code == 0, result.output
    [report_dir] = list(tmp_path.iterdir())
    record = json.loads((report_dir / "diagnostics.json").read_text())

    assert [f["name"] for f in record["features"]] == list_features()
    assert record["config"]["skipped"] == {}
    assert record["config"]["funding_attached"] is True
    assert record["horizons"] == [1, 6]
    for feature in record["features"]:
        assert [entry["horizon"] for entry in feature["ic"]] == [1, 6]
        assert feature["coverage"] == pytest.approx(1.0)


def test_diagnostics_json_is_the_full_reproducibility_record(funded, tmp_path):
    result = _invoke(
        tmp_path, "--symbol", "ETH/USDT", "--timeframe", "4h", "--horizons", "6"
    )

    assert result.exit_code == 0, result.output
    [report_dir] = list(tmp_path.iterdir())
    assert "features" in report_dir.name
    config = json.loads((report_dir / "diagnostics.json").read_text())["config"]

    assert config["identity"]["symbol"] == "ETH/USDT"
    assert config["identity"]["market_type"] == "perp"
    assert config["candle_count"] == _BARS
    assert config["horizons"] == [6]
    assert config["data_start"] and config["data_end"]


def test_the_written_html_is_the_rendered_table(funded, tmp_path):
    result = _invoke(tmp_path, "--horizons", "6")

    assert result.exit_code == 0, result.output
    [report_dir] = list(tmp_path.iterdir())
    html = (report_dir / "features.html").read_text()
    record = json.loads((report_dir / "diagnostics.json").read_text())

    assert "Univariate diagnostics" in html
    for name in list_features():
        assert f'<th class="row-label">{name}</th>' in html
    # The IC on the page and the IC in the record are the same number.
    energy = next(f for f in record["features"] if f["name"] == "energy")
    assert f"{energy['ic'][0]['ic']:+.3f}" in html


def test_a_frame_with_no_funding_skips_crowding_loudly_and_says_so_in_the_record(
    candles, tmp_path
):
    """Skipping is honest; skipping silently is how a feature ships unexamined."""
    result = _invoke(tmp_path, "--market-type", "spot", "--horizons", "6")

    assert result.exit_code == 0, result.output
    assert "Skipping crowding" in result.output
    assert "funding_rate" in result.output

    [report_dir] = list(tmp_path.iterdir())
    record = json.loads((report_dir / "diagnostics.json").read_text())
    assert "crowding" in record["config"]["skipped"]
    assert [f["name"] for f in record["features"]] == [
        name for name in list_features() if name != "crowding"
    ]
    assert record["config"]["funding_attached"] is False
    # And the page names it, rather than leaving a nine-feature registry looking
    # like an eight-feature one.
    assert "Not examined" in (report_dir / "features.html").read_text()


def test_no_funding_on_a_perp_also_skips_crowding(funded, tmp_path):
    result = _invoke(tmp_path, "--no-funding", "--horizons", "6")

    assert result.exit_code == 0, result.output
    assert "Skipping crowding" in result.output


def test_the_echoed_lines_carry_the_split_halves(funded, tmp_path):
    result = _invoke(tmp_path, "--horizons", "6")

    assert result.exit_code == 0, result.output
    condensed = " ".join(result.output.split())
    assert "IC@6b" in condensed
    # Full sample, then both halves in parentheses.
    assert re.search(r"IC@6b [+-]\d\.\d{4} \([+-]\d\.\d{3}/[+-]\d\.\d{3}\)", condensed)


@pytest.mark.parametrize("bad", ["", "6,0", "six"])
def test_a_malformed_horizon_list_exits_non_zero(funded, tmp_path, bad):
    result = _invoke(tmp_path, "--horizons", bad)

    assert result.exit_code != 0
    assert not list(tmp_path.iterdir()), "a rejected horizon must not leave a report behind"


def test_a_frame_shorter_than_a_feature_s_warmup_exits_non_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli, "load_candles", lambda **kwargs: synthetic_ohlcv(n=120, freq="4h")
    )
    result = _invoke(tmp_path, "--market-type", "spot", "--horizons", "6")

    assert result.exit_code != 0
    assert "warmup" in result.output
    assert not list(tmp_path.iterdir())


def test_no_stored_candles_exits_non_zero_with_a_fetch_hint(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "load_candles", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(cli, "list_candle_sets", lambda: pd.DataFrame())
    result = _invoke(tmp_path, "--symbol", "DOGE/USDT")

    assert result.exit_code != 0
    assert "No candles loaded" in result.output
    assert not list(tmp_path.iterdir())
