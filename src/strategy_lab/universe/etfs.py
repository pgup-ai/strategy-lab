from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EtfDefinition:
    symbol: str
    name: str
    sector: str
    inception: str


# Broad market / index ETFs
BROAD_ETFS: list[EtfDefinition] = [
    EtfDefinition(symbol="SPY", name="SPDR S&P 500 ETF", sector="broad", inception="1993-01-29"),
    EtfDefinition(symbol="QQQ", name="Invesco QQQ Trust", sector="broad", inception="1999-03-10"),
    EtfDefinition(symbol="IWM", name="iShares Russell 2000 ETF", sector="broad", inception="2000-05-22"),
    EtfDefinition(symbol="DIA", name="SPDR Dow Jones Industrial Average ETF", sector="broad", inception="1998-01-20"),
]

# International equity ETFs
INTERNATIONAL_ETFS: list[EtfDefinition] = [
    EtfDefinition(symbol="XIU", name="iShares S&P/TSX 60 Index ETF", sector="broad", inception="1999-09-28"),
    EtfDefinition(symbol="EFA", name="iShares MSCI EAFE ETF", sector="broad", inception="2001-08-14"),
    EtfDefinition(symbol="EEM", name="iShares MSCI Emerging Markets ETF", sector="broad", inception="2003-04-07"),
]

# Sector ETFs (US)
SECTOR_ETFS: list[EtfDefinition] = [
    EtfDefinition(symbol="SMH", name="VanEck Semiconductor ETF", sector="technology", inception="2000-05-05"),
    EtfDefinition(symbol="XLK", name="Technology Select Sector SPDR", sector="technology", inception="1998-12-16"),
    EtfDefinition(symbol="XLF", name="Financial Select Sector SPDR", sector="financial", inception="1998-12-16"),
    EtfDefinition(symbol="XLE", name="Energy Select Sector SPDR", sector="energy", inception="1998-12-16"),
    EtfDefinition(symbol="XLV", name="Health Care Select Sector SPDR", sector="healthcare", inception="1998-12-16"),
    EtfDefinition(symbol="XLY", name="Consumer Discretionary Select Sector SPDR", sector="consumer", inception="1998-12-16"),
    EtfDefinition(symbol="XLP", name="Consumer Staples Select Sector SPDR", sector="consumer", inception="1998-12-16"),
    EtfDefinition(symbol="XLI", name="Industrial Select Sector SPDR", sector="industrial", inception="1998-12-16"),
    EtfDefinition(symbol="XLU", name="Utilities Select Sector SPDR", sector="utilities", inception="1998-12-16"),
    EtfDefinition(symbol="XLB", name="Materials Select Sector SPDR", sector="materials", inception="1998-12-16"),
    EtfDefinition(symbol="XLRE", name="Real Estate Select Sector SPDR", sector="realestate", inception="2015-10-08"),
]

ETF_UNIVERSE: list[EtfDefinition] = BROAD_ETFS + INTERNATIONAL_ETFS + SECTOR_ETFS


def list_etfs() -> list[str]:
    return [etf.symbol for etf in ETF_UNIVERSE]
