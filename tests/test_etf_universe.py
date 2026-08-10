"""The batch fetch, and the symbol that hid behind it.

`XIU` sat unfetched for the life of this repo: Yahoo answers an unknown ticker
with an *empty frame* rather than an error, `fetch-etf-universe` printed one
"No data returned" line among eighteen, and the command exited 0. Measured on
2026-08-10 — a bare `XIU` returns 0 rows over 2020-01-01 -> now where `XIU.TO`
returns 1,657 daily bars.

The single-symbol fetchers already refuse an empty fetch through
`_raise_empty_fetch`, whose docstring says why: "Reporting 'stored 0' and
exiting 0 is how a hole in a series gets mistaken for a market that did not
trade." The batch one was the only path that did not, and it is the one that
runs over eighteen symbols where nobody reads every line.
"""

from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from strategy_lab.cli import app
from strategy_lab.universe.etfs import ETF_UNIVERSE


def _frame(rows: int = 3) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="1D", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}, index=index
    )


class _Client:
    """Yahoo's own behaviour for an unknown ticker: an empty frame, not a raise."""

    source = "yahoo"

    def __init__(self, empty: set[str]) -> None:
        self.empty = empty
        self.asked: list[str] = []

    def fetch_ohlcv(self, symbol, timeframe, **_):
        self.asked.append(symbol)
        return _frame().iloc[:0] if symbol in self.empty else _frame()


def _run(monkeypatch, empty: set[str], symbols: str):
    client = _Client(empty)
    monkeypatch.setattr(
        "strategy_lab.market_data.yahoo.YahooFinanceClient", lambda *a, **k: client
    )
    monkeypatch.setattr("strategy_lab.cli.upsert_candles", lambda records: len(records))
    monkeypatch.setattr(
        "strategy_lab.cli.normalize_candle_frame", lambda df, **kwargs: list(range(len(df)))
    )
    result = CliRunner().invoke(app, ["fetch-etf-universe", "--symbols", symbols])
    return result, client


def test_a_symbol_the_venue_cannot_resolve_fails_the_batch(monkeypatch):
    """**The gate.** Exiting 0 here is how `XIU` stayed missing."""
    result, _ = _run(monkeypatch, {"XIU"}, "SPY,XIU,QQQ")

    assert result.exit_code != 0, "a symbol that returned nothing still reported success"
    assert "XIU" in result.output


def test_the_other_symbols_are_still_fetched(monkeypatch):
    """The bound on the above: raising on the spot would cost the seventeen
    tickers after the bad one their fetch, which is worse than the bug."""
    result, client = _run(monkeypatch, {"XIU"}, "SPY,XIU,QQQ")

    assert client.asked == ["SPY", "XIU", "QQQ"]
    assert "Upserted" in result.output


def test_a_batch_where_every_symbol_answers_succeeds(monkeypatch):
    """Otherwise the test above passes against a command that always fails."""
    result, _ = _run(monkeypatch, set(), "SPY,QQQ")

    assert result.exit_code == 0, result.output


def test_a_non_us_listing_in_the_universe_carries_its_exchange_suffix():
    """Yahoo resolves `XIU` to nothing and `XIU.TO` to the TSX listing, and the
    difference is invisible at fetch time because both are "no error". Checked
    against the definition's own inception rather than a hardcoded list, so a
    future non-US ETF is covered by the same rule."""
    canadian = [etf for etf in ETF_UNIVERSE if etf.name.startswith("iShares S&P/TSX")]

    assert canadian, "the universe lost its TSX listing; this test now proves nothing"
    for etf in canadian:
        assert etf.symbol.endswith(".TO"), (
            f"{etf.symbol} is a TSX listing without its suffix, which Yahoo "
            f"answers with an empty frame rather than an error"
        )


def test_every_universe_symbol_is_distinct():
    symbols = [etf.symbol for etf in ETF_UNIVERSE]

    assert len(symbols) == len(set(symbols))
