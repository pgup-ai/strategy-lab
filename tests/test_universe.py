from __future__ import annotations

from strategy_lab.universe.etfs import ETF_UNIVERSE, list_etfs


def test_etf_universe_exposes_symbols() -> None:
    symbols = list_etfs()

    assert "SPY" in symbols
    assert len(symbols) == len(ETF_UNIVERSE)
