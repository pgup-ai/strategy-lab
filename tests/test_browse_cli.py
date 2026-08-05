"""``browse``, and the fact that it did not replace ``serve``.

Two commands for two different things: ``serve`` hosts the dated directory a
backtest froze, which re-renders byte-identically and is the reproducibility
boundary; ``browse`` recomputes per request and persists nothing. The pressure
is always toward one command, so the split is asserted rather than assumed.

``uvicorn.run`` is recorded rather than run -- a test that let it bind would
serve forever and the suite would hang instead of failing.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from strategy_lab import cli
from strategy_lab.api.app import DEFAULT_PORT

runner = CliRunner()


@pytest.fixture
def served(monkeypatch):
    calls = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: calls.append(kwargs))
    return calls


def _commands() -> set[str]:
    return {command.name for command in cli.app.registered_commands}


def test_browse_joins_serve_rather_than_replacing_it():
    assert {"browse", "serve"} <= _commands()


def test_browse_serves_the_api_on_the_loopback_interface_by_default(served):
    result = runner.invoke(cli.app, ["browse"])

    assert result.exit_code == 0, result.output
    assert served == [{"host": "127.0.0.1", "port": DEFAULT_PORT}]
    assert f"http://127.0.0.1:{DEFAULT_PORT}" in result.output


def test_browse_passes_the_chosen_port_through(served):
    assert runner.invoke(cli.app, ["browse", "--port", "9123"]).exit_code == 0

    assert served == [{"host": "127.0.0.1", "port": 9123}]


def test_browse_refuses_a_routable_host_rather_than_warning(served):
    """Unauthenticated, read-only by convention rather than by enforcement, and
    it can make the process fetch from an exchange."""
    result = runner.invoke(cli.app, ["browse", "--host", "0.0.0.0"])

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "refusing to bind" in str(result.exception)
    assert served == []


def test_serve_still_hosts_the_frozen_reports(monkeypatch, tmp_path):
    """The record has to stay reachable; a browser is not a substitute for it."""
    hosted = {}
    monkeypatch.setattr(
        "strategy_lab.server.run_server",
        lambda **kwargs: hosted.update(kwargs),
    )

    result = runner.invoke(cli.app, ["serve", "--report-root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert hosted["report_root"] == tmp_path
    assert hosted["host"] == "127.0.0.1"
