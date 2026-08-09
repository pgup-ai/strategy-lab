"""The research browser's HTTP surface: read-only, loopback, one page and five
endpoints.

Read-only means read-only. Nothing here writes to ``signals`` or
``bar_reasons``, nothing writes into ``reports/``, and the only write of any
kind is the candle upsert ``server.py`` already performs on a refresh -- the
same fetch-recent-and-upsert path, called rather than reimplemented, so "newest
bars" never means "restart the process".

New endpoints go here rather than onto ``server.py`` because that module is
already condemned in the design doc's own inventory -- *"Replace. Synchronous,
fetches from the exchange inside a GET handler"* -- and adding handlers to a
component the repo has decided to retire is debt this codebase is usually good
at refusing. ``serve`` keeps working unchanged: the per-run ``plot.html`` is the
frozen reproducibility record and this is a live view, two different things that
must not become one.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from strategy_lab.api.analysis import (
    DatasetUnavailable,
    build_analysis,
    registered_strategies,
)
from strategy_lab.api.board import stored_datasets, stream_board
from strategy_lab.api.models import (
    AnalysisModel,
    AnalysisQuery,
    BoardQuery,
    BoardRowModel,
    DatasetModel,
    RefreshModel,
    RefreshQuery,
    StrategyModel,
)
from strategy_lab.backtests.funding_frame import FundingUnavailable
from strategy_lab.db import list_candle_sets
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.market_data.streams import stream_url

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

    @app.get("/api/board")
    def board(query: Annotated[BoardQuery, Query()]) -> StreamingResponse:
        """One row per (dataset, strategy), streamed as newline-delimited JSON.

        Streamed rather than assembled because the cost is unavoidable and
        serial: a row is a whole-history ``build_analysis`` -- 330-400 ms warm
        on the stored perp frames -- and four threads over three of them
        measured **1.10x**, the work being pandas and vectorbt under the GIL.
        So the board cannot be made fast; it can be made to arrive. First paint
        is the first row rather than the sum of all of them.

        Still a plain ``def``. The generator below is a synchronous iterator,
        which Starlette drains in the threadpool, so the event loop is no more
        blocked here than by ``analysis`` -- ``create_app``'s docstring carries
        the 3,640 ms that rule was measured on, and this handler holds the
        threadpool for seconds rather than milliseconds.

        Each line is validated through ``BoardRowModel`` on its way out, which
        is where ``response_model`` would have done it had this returned a list.

        The enumeration happens *here* rather than inside the generator: once
        the first byte is written the status code is spent, so a database that
        is unreachable has to fail before the response starts or it arrives as a
        200 that stops after zero rows.
        """
        datasets = stored_datasets(market_type=query.market_type)

        def lines() -> Iterator[str]:
            for row in stream_board(
                datasets,
                strategies=query.names,
                exit_mode=query.exit_mode,
                spark_bars=query.spark_bars,
            ):
                yield BoardRowModel.model_validate(asdict(row)).model_dump_json() + "\n"

        return StreamingResponse(lines(), media_type="application/x-ndjson")

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
            return refresh_candles(identity, query.after, query.since)
        except Exception as exc:  # a venue or database failure is not a bad request
            raise HTTPException(status_code=502, detail=f"refresh failed: {exc}") from exc

    return app


def _dataset_row(row: pd.Series) -> dict[str, Any]:
    identity = MarketDataIdentity(
        exchange=str(row["exchange"]),
        market_type=str(row["market_type"]),
        symbol=str(row["symbol"]),
        timeframe=str(row["timeframe"]),
    )
    return {
        "exchange": identity.exchange,
        "market_type": identity.market_type,
        "symbol": identity.symbol,
        "timeframe": identity.timeframe,
        # `None` for anything without one, which is most of them. The page shows
        # a live control only where there is something to connect to.
        "stream": stream_url(identity),
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
