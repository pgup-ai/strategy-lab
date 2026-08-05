"""The research browser's HTTP surface: read-only, loopback, one page and four
endpoints.

Read-only means read-only. Nothing here writes to ``signals``, nothing writes
into ``reports/``, and the only write of any kind is the candle upsert
``server.py`` already performs on a refresh -- the same fetch-recent-and-upsert
path, called rather than reimplemented, so "newest bars" never means "restart
the process".

New endpoints go here rather than onto ``server.py`` because that module is
already condemned in the design doc's own inventory -- *"Replace. Synchronous,
fetches from the exchange inside a GET handler"* -- and adding four handlers to
a component the repo has decided to retire is debt this codebase is usually good
at refusing. ``serve`` keeps working unchanged: the per-run ``plot.html`` is the
frozen reproducibility record and this is a live view, two different things that
must not become one.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from strategy_lab.api.analysis import (
    DatasetUnavailable,
    build_analysis,
    registered_strategies,
)
from strategy_lab.api.models import (
    AnalysisModel,
    AnalysisQuery,
    DatasetModel,
    RefreshModel,
    RefreshQuery,
    StrategyModel,
)
from strategy_lab.backtests.funding_frame import FundingUnavailable
from strategy_lab.db import list_candle_sets
from strategy_lab.market_data.base import MarketDataIdentity

# Loopback only. The app reads a research database and can make the process
# fetch from an exchange, so binding it to a routable address is a change of
# kind rather than of convenience.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
DEFAULT_PORT = 8760


def create_app() -> FastAPI:
    """The read-only app.

    **No handler here is ``async``, deliberately.** Every one of them blocks:
    the page reads a 191 KB asset off disk, ``datasets`` queries Postgres,
    ``analysis`` loads a frame and runs a whole-history ``from_signals``, and
    ``refresh`` makes a synchronous call to the venue. FastAPI runs an ``async
    def`` handler *on the event loop* and a plain ``def`` in a threadpool, so
    declaring these async stalls every other request for the duration --
    measured, a 1 ms ``/api/strategies`` took **3,640 ms** when it landed inside
    an in-flight analysis, and the page polls. Adding ``async`` back is the
    reflex this comment exists to stop.
    """
    app = FastAPI(
        title="strategy-lab research browser",
        description=(
            "Read-only views of stored candles and registered strategies. Signals "
            "are computed per request by the same vectorized call run_backtest "
            "makes, never read from the signals table."
        ),
        version="0.1.0",
    )

    @app.exception_handler(DatasetUnavailable)
    async def _no_dataset(request: Request, exc: DatasetUnavailable) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(FundingUnavailable)
    async def _no_funding(request: Request, exc: FundingUnavailable) -> JSONResponse:
        # 409 rather than 404: the candles exist, and it is their funding history
        # that cannot support the strategy that was asked for. The message names
        # the fetch command that fixes it.
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(ValueError)
    async def _bad_request(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/", response_class=HTMLResponse)
    def page() -> HTMLResponse:
        """The one page, rendered fresh so an edit to it needs no build step.

        Deliberately uncached: the asset it inlines is 191 KB off local disk,
        and a cached page is a stale page the first time somebody changes one.
        """
        from strategy_lab.browser.page import render_browser_html

        return HTMLResponse(render_browser_html())

    @app.get("/api/datasets", response_model=list[DatasetModel])
    def datasets() -> list[dict[str, Any]]:
        """Every candle set stored, on storage's own four-part identity."""
        return [_dataset_row(row) for _, row in list_candle_sets().iterrows()]

    @app.get("/api/strategies", response_model=list[StrategyModel])
    def strategies() -> list[dict[str, Any]]:
        """Both registries, each entry labelled with the contract it answers on."""
        return [asdict(entry) for entry in registered_strategies()]

    @app.get("/api/analysis", response_model=AnalysisModel)
    def analysis(query: Annotated[AnalysisQuery, Query()]) -> dict[str, Any]:
        """Candles, what the strategy did, why it did it, and under what settings."""
        payload = build_analysis(
            MarketDataIdentity(
                exchange=query.exchange,
                market_type=query.market_type,
                symbol=query.symbol,
                timeframe=query.timeframe,
            ),
            strategy_name=query.strategy,
            exit_mode=query.exit_mode,
            failure_bars=query.failure_bars,
            start=query.start,
            end=query.end,
            fees=query.fees,
            slippage=query.slippage,
            cash=query.cash,
            position_pct=query.position_pct,
            funding=query.funding,
            allow_shorts=query.allow_shorts,
        )
        return asdict(payload)

    @app.post("/api/refresh", response_model=RefreshModel)
    def refresh(query: Annotated[RefreshQuery, Query()]) -> dict[str, Any]:
        """Fetch the last few bars from the venue, upsert them, and return them.

        The one write in this app, and it is ``server.refresh_candles`` -- the
        existing path, called rather than copied, so the browser cannot store a
        candle by a rule the rest of the lab does not use. POST because it has an
        effect on storage; every other endpoint here is a GET that cannot.

        On a perp that path advances funding alongside the candles, and the
        response carries both counts. A refresh that moved only the bars would
        push the candle window past the last stored settlement and leave the
        coverage guard refusing the dataset the caller was just looking at.
        """
        from strategy_lab.server import refresh_candles

        identity = MarketDataIdentity(
            exchange=query.exchange,
            market_type=query.market_type,
            symbol=query.symbol,
            timeframe=query.timeframe,
        )
        try:
            return refresh_candles(identity, query.after)
        except Exception as exc:  # a venue or database failure is not a bad request
            raise HTTPException(status_code=502, detail=f"refresh failed: {exc}") from exc

    return app


def _dataset_row(row: pd.Series) -> dict[str, Any]:
    return {
        "exchange": str(row["exchange"]),
        "market_type": str(row["market_type"]),
        "symbol": str(row["symbol"]),
        "timeframe": str(row["timeframe"]),
        "candles": int(row["candles"]),
        "first_timestamp": str(row["first_timestamp"]),
        "last_timestamp": str(row["last_timestamp"]),
    }


def run_api(*, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
    """Serve the browser API on the loopback interface.

    A non-loopback host is refused rather than warned about. This process holds
    an unauthenticated view of a research database and a button that makes it
    fetch from an exchange; nothing here is hardened for a network, and a
    ``--host 0.0.0.0`` typed once is how that becomes somebody else's problem.
    """
    import uvicorn

    if host not in LOOPBACK_HOSTS:
        raise ValueError(
            f"refusing to bind the research browser to {host!r}: it is "
            f"unauthenticated and read-only by convention rather than by "
            f"enforcement. Bind one of {', '.join(sorted(LOOPBACK_HOSTS))}."
        )
    uvicorn.run(create_app(), host=host, port=port)


__all__ = ["DEFAULT_PORT", "LOOPBACK_HOSTS", "create_app", "run_api"]
