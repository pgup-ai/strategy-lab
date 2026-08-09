from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import typer

from strategy_lab.api.app import DEFAULT_PORT as BROWSER_PORT
from strategy_lab.backtests import ExitMode, SizeMode, run_backtest
from strategy_lab.backtests.sizing import DEFAULT_VOL_SPAN
from strategy_lab.db import init_db, list_candle_sets, load_candles, upsert_candles
from strategy_lab.db.candles import normalize_candle_frame
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.market_data.binance_futures import (
    OPEN_INTEREST_HISTORY_DAYS,
    SUPPORTED_PERP_EXCHANGES,
)
from strategy_lab.market_data.binance_futures import SOURCE as BINANCE_FUTURES_SOURCE
from strategy_lab.strategies import get_strategy, list_strategies
from strategy_lab.strategies.base import require_warmup_bars
from strategy_lab.universe.etfs import ETF_UNIVERSE


app = typer.Typer(help="Fetch candles, store them locally, and run reproducible backtests.")


@app.command("init-db")
def init_db_command() -> None:
    """Create the local Postgres schema."""
    init_db()
    typer.echo("Initialized database schema.")


@app.command("migrate")
def migrate_command() -> None:
    """Apply idempotent schema upgrades to an existing database."""
    from strategy_lab.storage.migrations import run_migrations

    count = run_migrations()
    typer.echo(f"Applied {count} migration statements.")


@app.command("fetch-crypto")
def fetch_crypto(
    symbol: str = typer.Option("BTC/USDT", help="Exchange symbol, for example BTC/USDT."),
    timeframe: str = typer.Option("15m", help="Candle timeframe supported by the exchange."),
    exchange: str = typer.Option("binance", help="ccxt exchange id."),
    market_type: str = typer.Option("spot", help="spot or perp."),
    since: str | None = typer.Option(None, help="UTC start time, for example 2024-01-01."),
    until: str | None = typer.Option(None, help="UTC end time, for example 2024-12-31."),
    limit: int = typer.Option(1000, help="Exchange page size."),
) -> None:
    """Fetch crypto candles from ccxt and upsert them into Postgres."""
    from strategy_lab.market_data.binance import CryptoOhlcvClient

    client = CryptoOhlcvClient(exchange_id=exchange, market_type=market_type)
    df = client.fetch_ohlcv(symbol, timeframe, since=since, until=until, limit=limit)
    records = normalize_candle_frame(
        df,
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
        source=client.exchange_id,
    )
    count = upsert_candles(records)
    typer.echo(f"Upserted {count} candles for {exchange}/{market_type}/{symbol}/{timeframe}.")


@app.command("fetch-perp")
def fetch_perp(
    symbol: str = typer.Option("BTC/USDT", help="Contract symbol, for example BTC/USDT."),
    timeframe: str = typer.Option("4h", help="Candle timeframe, for example 4h."),
    since: str = typer.Option("2019-09-01", help="UTC start time. History begins 2019-09-09."),
    until: str | None = typer.Option(None, help="UTC end time. Defaults to now."),
    exchange: str = typer.Option(
        "binance", help="Venue id used in the stored identity. Only 'binance' is supported."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Fetch and report, but store nothing."),
) -> None:
    """Backfill Binance USD-M perp candles into `market_candles`.

    Stored under `market_type="perp"`, through the same
    `normalize_candle_frame` + `upsert_candles` path as spot and equity candles,
    so perp prices get the same Decimal binding rather than a second code path
    that would have to relearn it.
    """
    client = _futures_client(exchange=exchange)
    df = _fetch(lambda: client.fetch_klines(symbol, timeframe, since=since, until=until))
    identity = MarketDataIdentity(
        exchange=exchange, market_type="perp", symbol=symbol, timeframe=timeframe
    )
    if df.empty:
        _raise_empty_fetch(f"{exchange}/perp/{symbol}/{timeframe}", since, until)

    records = normalize_candle_frame(
        df,
        exchange=identity.exchange,
        market_type=identity.market_type,
        symbol=identity.symbol,
        timeframe=identity.timeframe,
        source=BINANCE_FUTURES_SOURCE,
    )
    label = f"{exchange}/perp/{symbol}/{timeframe}"
    if dry_run:
        typer.echo(f"Dry run: {len(records)} candles for {label}, nothing written.")
        return

    count = upsert_candles(records)
    typer.echo(
        f"Upserted {count} candles for {label} "
        f"({df.index.min()} -> {df.index.max()})."
    )


@app.command("fetch-funding")
def fetch_funding(
    symbol: str = typer.Option("BTC/USDT", help="Contract symbol, for example BTC/USDT."),
    since: str = typer.Option("2019-09-01", help="UTC start time. History begins 2019-09-10."),
    until: str | None = typer.Option(None, help="UTC end time. Defaults to now."),
    exchange: str = typer.Option(
        "binance", help="Venue id used in the stored identity. Only 'binance' is supported."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Fetch and report, but store nothing."),
) -> None:
    """Backfill perp funding rates into `funding_rates`.

    Settlement times are stored exactly as the venue reports them. The interval
    is per-contract -- 8h for most Binance perps but not all and not always --
    so nothing here assumes or enforces one.
    """
    client = _futures_client(exchange=exchange)
    rows = _fetch(lambda: client.fetch_funding(symbol, since=since, until=until))
    label = f"{exchange}/perp/{symbol}"
    if not rows:
        _raise_empty_fetch(f"funding for {label}", since, until)

    if dry_run:
        typer.echo(f"Dry run: {len(rows)} funding rows for {label}, nothing written.")
        return

    count = _upsert_funding(rows)
    first = _utc(rows[0]["funding_time_ms"])
    last = _utc(rows[-1]["funding_time_ms"])
    typer.echo(f"Upserted {count} funding rows for {label} ({first} -> {last}).")


@app.command("fetch-open-interest")
def fetch_open_interest(
    symbol: str = typer.Option("BTC/USDT", help="Contract symbol, for example BTC/USDT."),
    period: str = typer.Option("4h", help="Snapshot period: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d."),
    exchange: str = typer.Option(
        "binance", help="Venue id used in the stored identity. Only 'binance' is supported."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Fetch and report, but store nothing."),
) -> None:
    """Collect open-interest snapshots into `open_interest`.

    There is deliberately no `--since`: Binance serves only ~30 days of OI
    history. Each run collects that whole window -- paginated backward, so a
    fine `--period` is not silently cut to one 500-row page -- and anything
    older is simply not available from this venue. Run it on a schedule if OI is
    wanted as a series reaching further back than 30 days.
    """
    typer.secho(
        f"Warning: Binance serves only ~{OPEN_INTEREST_HISTORY_DAYS} days of open-interest "
        "history, so this cannot be backfilled -- it accumulates forward from the first run. "
        "Do not read a short OI series as history.",
        fg=typer.colors.YELLOW,
    )
    client = _futures_client(exchange=exchange)
    rows = _fetch(lambda: client.fetch_open_interest(symbol, period=period))
    label = f"{exchange}/perp/{symbol}/{period}"
    if not rows:
        _raise_empty_fetch(f"open interest for {label}", None, None)

    if dry_run:
        typer.echo(f"Dry run: {len(rows)} open-interest rows for {label}, nothing written.")
        return

    count = _upsert_open_interest(rows)
    first = _utc(rows[0]["ts_ms"])
    last = _utc(rows[-1]["ts_ms"])
    typer.echo(f"Upserted {count} open-interest rows for {label} ({first} -> {last}).")


# These four exist so the fetch tests can substitute the venue and storage,
# matching `_create_run`/`_write_signals` below.
def _futures_client(*, exchange: str, **kwargs):
    from strategy_lab.market_data.binance_futures import BinanceFuturesClient

    if exchange not in SUPPORTED_PERP_EXCHANGES:
        raise typer.BadParameter(
            f"{exchange!r} is not supported: this client only speaks to Binance USD-M "
            f"futures, so another --exchange would file Binance data under that venue's "
            f"name. Supported: {', '.join(SUPPORTED_PERP_EXCHANGES)}."
        )
    return BinanceFuturesClient(exchange=exchange, **kwargs)


def _upsert_funding(rows):
    from strategy_lab.db.funding import upsert_funding

    return upsert_funding(rows)


def _upsert_open_interest(rows):
    from strategy_lab.db.funding import upsert_open_interest

    return upsert_open_interest(rows)


def _fetch(call):
    """Turn a venue error into a clean non-zero exit rather than a traceback."""
    from strategy_lab.market_data.binance_futures import BinanceFuturesError

    try:
        return call()
    except (BinanceFuturesError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _raise_empty_fetch(label: str, since: str | None, until: str | None) -> None:
    """An empty fetch exits non-zero on purpose.

    Reporting "stored 0" and exiting 0 is how a hole in a series gets mistaken
    for a market that did not trade.
    """
    window = f" for {since} -> {until or 'now'}" if since else ""
    raise typer.BadParameter(
        f"The venue returned no rows for {label}{window}. "
        "Check the symbol and that the requested window is within the venue's history."
    )


def _utc(ms: int) -> str:
    import pandas as pd

    return str(pd.Timestamp(ms, unit="ms", tz="UTC"))


@app.command("fetch-stock")
def fetch_stock(
    symbol: str = typer.Option("AAPL", help="Ticker symbol."),
    timeframe: str = typer.Option("1h", help="Yahoo interval, for example 1h or 1d."),
    period: str = typer.Option("2y", help="Yahoo period used when start is not supplied."),
    start: str | None = typer.Option(None, help="UTC start date, for example 2024-01-01."),
    end: str | None = typer.Option(None, help="UTC end date, for example 2024-12-31."),
) -> None:
    """Fetch stock candles from Yahoo Finance and upsert them into Postgres."""
    from strategy_lab.market_data.yahoo import YahooFinanceClient

    client = YahooFinanceClient()
    df = client.fetch_ohlcv(symbol, timeframe, period=period, start=start, end=end)
    records = normalize_candle_frame(
        df,
        exchange="yahoo",
        market_type="equity",
        symbol=symbol,
        timeframe=timeframe,
        source=client.source,
    )
    count = upsert_candles(records)
    typer.echo(f"Upserted {count} candles for yahoo/equity/{symbol}/{timeframe}.")


@app.command("fetch-etf-universe")
def fetch_etf_universe(
    timeframe: str = typer.Option("1w", help="Candle timeframe for all ETFs."),
    start: str = typer.Option("2020-01-01", help="UTC start date."),
    end: str | None = typer.Option(None, help="UTC end date. Defaults to today."),
    symbols: str | None = typer.Option(
        None,
        help="Comma-separated symbols to fetch. Defaults to all configured ETFs.",
    ),
) -> None:
    """Fetch weekly candles for the configured ETF universe from Yahoo Finance."""
    from strategy_lab.market_data.yahoo import YahooFinanceClient

    if symbols:
        target = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        target = [etf.symbol for etf in ETF_UNIVERSE]

    client = YahooFinanceClient()
    for symbol in target:
        typer.echo(f"Fetching {symbol} {timeframe} from {start} ...")
        df = client.fetch_ohlcv(symbol, timeframe, start=start, end=end)
        if df.empty:
            typer.echo(f"  No data returned for {symbol}")
            continue
        records = normalize_candle_frame(
            df,
            exchange="yahoo",
            market_type="equity",
            symbol=symbol,
            timeframe=timeframe,
            source=client.source,
        )
        count = upsert_candles(records)
        typer.echo(f"  Upserted {count} candles for yahoo/equity/{symbol}/{timeframe}.")


@app.command("backtest")
def backtest(
    symbols: str = typer.Option(
        "BTC/USDT", "--symbols", "--symbol", help="Comma-separated symbols."
    ),
    strategy_name: str = typer.Option("turnaround_v2", "--strategy", help="Strategy name."),
    exchange: str = typer.Option("binance", help="Data exchange/source."),
    market_type: str = typer.Option("spot", help="spot, perp, or equity."),
    timeframe: str = typer.Option("15m", help="Candle timeframe."),
    start: str | None = typer.Option(None, help="Optional backtest start time."),
    end: str | None = typer.Option(None, help="Optional backtest end time."),
    fees: float = typer.Option(0.0005, help="One-way fee rate."),
    slippage: float = typer.Option(0.0005, help="One-way slippage rate."),
    cash: float = typer.Option(10_000.0, help="Initial cash."),
    allow_shorts: bool = typer.Option(
        True,
        help="Allow short entries. Disable for spot-style long-only testing.",
    ),
    exit_mode: ExitMode = typer.Option(
        ExitMode.CONTINUATION_FAILURE,
        help=(
            "Exit behavior: continuation failure, trend failure, trend structure "
            "(long-only), setup invalidation stop, or opposite signal only."
        ),
    ),
    failure_bars: int = typer.Option(
        4,
        min=1,
        help="Adverse consecutive close count used by continuation_failure.",
    ),
    position_pct: float = typer.Option(
        0.95,
        min=0.01,
        max=1.0,
        help="Fraction of capital deployed per trade.",
    ),
    size_mode: SizeMode = typer.Option(
        SizeMode.FIXED,
        "--size-mode",
        help=(
            "fixed deploys --position-pct on every entry. vol-scaled-entry scales it by "
            "target / realized volatility, so a violent regime is entered smaller. It "
            "scales the ENTRY only: an open position is never resized, because "
            "from_signals fills once per state change, so a position held through a "
            "calm-to-violent shift keeps its calm-regime notional. The estimator is an "
            "EWM that decays its seed instead of dropping it, so weights need roughly "
            "20x --vol-span to converge and no entry is taken before that."
        ),
    ),
    vol_span: int = typer.Option(
        DEFAULT_VOL_SPAN,
        "--vol-span",
        min=1,
        help=(
            "Span of the EWM volatility estimator under --size-mode vol-scaled-entry. "
            "It also sets the run's warmup: entries are masked for 20x this many bars, "
            "and a frame shorter than that is refused rather than sized off an "
            "estimate that has not converged."
        ),
    ),
    vol_target: float = typer.Option(
        0.30,
        "--vol-target",
        min=0.0001,
        help="Annualized volatility each entry is sized for under --size-mode vol-scaled-entry.",
    ),
    max_weight: float = typer.Option(
        2.0,
        "--max-weight",
        min=0.0001,
        help=(
            "Cap on the entry size multiplier, so a calm stretch cannot lever up. "
            "The book has no leverage, so anything above 1 / --position-pct cannot be "
            "filled and is capped, with a warning and the effective value in config.json."
        ),
    ),
    cost_stress: str = typer.Option(
        "1",
        "--cost-stress",
        help="Comma-separated fee/slippage multiples to compare, for example 1,2,3.",
    ),
    funding: bool = typer.Option(
        True,
        "--funding/--no-funding",
        help=(
            "Charge stored perp funding at its settlement times, and attach it to the "
            "frame as the funding_rate column strategies read. Perps only."
        ),
    ),
    report_root: Path = typer.Option(Path("reports"), help="Report output folder."),
) -> None:
    """Run a vectorbt backtest for one or more stored symbols."""
    strategy = get_strategy(strategy_name, allow_shorts=allow_shorts)
    multiples = _parse_cost_stress(cost_stress)
    for symbol in _split_symbols(symbols):
        identity = MarketDataIdentity(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        df = load_candles(
            exchange=identity.exchange,
            market_type=identity.market_type,
            symbol=identity.symbol,
            timeframe=identity.timeframe,
            start=start,
            end=end,
        )
        if df.empty:
            _raise_missing_candles(identity)
        df, rates = _with_funding_column(identity, df, enabled=funding)
        try:
            result = run_backtest(
                df=df,
                strategy=strategy,
                identity=identity,
                fees=fees,
                slippage=slippage,
                cash=cash,
                exit_mode=exit_mode,
                failure_bars=failure_bars,
                position_pct=position_pct,
                report_root=report_root,
                funding=rates,
                cost_stress=multiples,
                size_mode=size_mode,
                vol_target=vol_target,
                max_weight=max_weight,
                vol_span=vol_span,
            )
        except ValueError as exc:
            # Incompatible flag combinations (sizing collisions, exit modes a
            # strategy cannot serve) are user input, not a crash.
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(f"Wrote report for {symbol}: {result.report_dir}")
        _echo_costs(result)


def _parse_cost_stress(raw: str) -> tuple[float, ...]:
    try:
        multiples = tuple(float(part) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise typer.BadParameter(f"--cost-stress must be comma-separated numbers: {exc}") from exc
    if not multiples or any(multiple <= 0 for multiple in multiples):
        raise typer.BadParameter("--cost-stress values must all be > 0, for example 1,2,3")
    return multiples


def _with_funding_column(identity: MarketDataIdentity, df, *, enabled: bool, required: bool = True):
    """``funding_frame.with_funding_column``, loading through this module's wrapper.

    The rule itself -- perp only, ``align_funding_to_bars`` rather than a reindex
    -- lives in ``backtests/funding_frame.py`` so the research browser attaches
    funding by the same one. What is passed in here is only the loader, so a
    missing history still exits as user error rather than as a traceback.
    """
    from strategy_lab.backtests.funding_frame import with_funding_column

    return with_funding_column(
        identity, df, enabled=enabled, required=required, load_rates=_funding_rates
    )


def _funding_rates(identity: MarketDataIdentity, df, *, required: bool = True):
    """Stored funding for a perp, with an unusable history reported as user error.

    An unfetched or partial funding history is something the operator can fix
    with the command the message names, so it exits 2 rather than surfacing the
    library's ``ValueError`` as a traceback.
    """
    from strategy_lab.backtests.funding_frame import FundingUnavailable, funding_rates

    try:
        return funding_rates(identity, df, required=required)
    except FundingUnavailable as exc:
        raise typer.BadParameter(str(exc)) from exc


def _echo_costs(result) -> None:
    breakdown = json.loads(result.costs_path.read_text())
    base = next(row for row in breakdown["stress"] if row["multiple"] == 1.0)
    typer.echo(
        f"  gross {base['gross_return_pct']:+.2f}%  "
        f"fees {base['fees_paid']:,.2f}  slippage {base['slippage_paid']:,.2f}  "
        f"funding {base['funding_paid']:,.2f}  "
        f"size effect {base['size_effect']:,.2f}  "
        f"net {base['net_return_pct']:+.2f}%"
    )
    for row in breakdown["stress"]:
        if row["multiple"] != 1.0:
            typer.echo(f"  {row['multiple']:g}x costs: net {row['net_return_pct']:+.2f}%")


@app.command("sweep")
def sweep_command(
    symbol: str = typer.Option("BTC/USDT", help="Symbol to sweep."),
    strategy_name: str = typer.Option("donchian", "--strategy", help="Strategy name."),
    grid: str = typer.Option(
        ...,
        "--grid",
        help='Parameter grid as JSON, e.g. \'{"entry_span":[48,96],"exit_span":[24,48]}\'.',
    ),
    exchange: str = typer.Option("binance", help="Data exchange/source."),
    market_type: str = typer.Option("spot", help="spot, perp, or equity."),
    timeframe: str = typer.Option("15m", help="Candle timeframe."),
    start: str | None = typer.Option(None, help="Optional sweep start time."),
    end: str | None = typer.Option(None, help="Optional sweep end time."),
    funding: bool = typer.Option(
        True,
        "--funding/--no-funding",
        help=(
            "Attach stored perp funding as the funding_rate column strategies read. "
            "Perps only. An input, not a cost -- the surface stays gross of costs."
        ),
    ),
    report_root: Path = typer.Option(Path("reports"), help="Report output folder."),
) -> None:
    """Score a strategy across a parameter grid and render the stability surface.

    A single tuned parameter proves nothing; the R0 gate is a broad region where
    neighbouring parameters behave similarly. The stability score and the
    heatmap both exist to make a lone spike look like the overfit it is rather
    than like a result.

    Funding is attached to the frame on a perp because ``crowding`` reads it, and
    a cell scored without it is scoring a different strategy. It is *not* charged
    here -- this surface is gross of costs by design, and R2's cost model is what
    the single-run ``backtest`` applies.
    """
    from dataclasses import asdict
    from datetime import UTC, datetime

    from strategy_lab.backtests.engine import build_report_dir
    from strategy_lab.backtests.sweep import stability_score, sweep_parameters
    from strategy_lab.backtests.sweep_report import render_sweep_html

    parsed_grid = _parse_grid(grid)
    identity = MarketDataIdentity(
        exchange=exchange, market_type=market_type, symbol=symbol, timeframe=timeframe
    )
    df = load_candles(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
    )
    if df.empty:
        _raise_missing_candles(identity)
    # Only a strategy that reads ``crowding`` actually needs the column, and this
    # surface never charges funding, so an unfunded perp must not be fatal here the
    # way it is for a backtest: sweeping ``donchian`` over a perp worked before this
    # attachment existed and has to keep working. The strategy declares its own
    # inputs, so ask it rather than hardcoding a list.
    needs_funding = "crowding" in getattr(get_strategy(strategy_name), "features", ())
    df, rates = _with_funding_column(identity, df, enabled=funding, required=needs_funding)
    attached = rates is not None
    if funding and market_type == "perp" and not needs_funding and not attached:
        typer.secho(
            f"No usable stored funding, so {strategy_name} is scored without it. "
            "Harmless here -- it reads no funding-derived feature, and a sweep never "
            "charges carry.",
            fg=typer.colors.YELLOW,
        )

    try:
        points = sweep_parameters(
            df=df, strategy_name=strategy_name, grid=parsed_grid, timeframe=timeframe
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    score = stability_score(points)
    config = {
        "identity": asdict(identity),
        "strategy": strategy_name,
        "grid": parsed_grid,
        "data_start": str(df.index.min()),
        "data_end": str(df.index.max()),
        "candle_count": int(len(df)),
        # What happened, not what was asked for: a perp sweep of a strategy that
        # reads no funding feature runs without the column rather than failing.
        "funding_attached": attached,
        "generated_at": datetime.now(UTC).isoformat(),
    }

    report_dir = build_report_dir(report_root, identity, f"{strategy_name}_sweep")
    report_dir.mkdir(parents=True, exist_ok=False)
    (report_dir / "sweep.html").write_text(
        render_sweep_html(points=points, config=config), encoding="utf-8"
    )
    (report_dir / "points.json").write_text(
        json.dumps(
            {
                "config": config,
                "stability_score": score,
                "points": [asdict(point) for point in points],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    positive = sum(1 for point in points if point.sharpe > 0)
    best = max(points, key=lambda point: point.sharpe)
    typer.echo(f"Wrote sweep for {symbol}: {report_dir}")
    typer.echo(
        f"Stability score: {score:.3f} "
        f"({positive}/{len(points)} cells with positive Sharpe, best {best.sharpe:+.2f} "
        f"at {best.params})"
    )


def _parse_grid(raw: str) -> dict[str, list]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"--grid is not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict) or not parsed:
        raise typer.BadParameter(
            '--grid must be a non-empty JSON object, e.g. \'{"lookback":[24,48,96]}\''
        )
    for name, values in parsed.items():
        if not isinstance(values, list) or not values:
            raise typer.BadParameter(
                f"--grid entry {name!r} must be a non-empty list of values, got {values!r}"
            )
    return parsed


@app.command("features")
def features_command(
    exchange: str = typer.Option("binance", help="Data exchange/source."),
    market_type: str = typer.Option("perp", help="spot, perp, or equity."),
    symbol: str = typer.Option("BTC/USDT", help="Symbol to diagnose."),
    timeframe: str = typer.Option("4h", help="Candle timeframe."),
    horizons: str = typer.Option(
        "1,6,30", help="Comma-separated forward-return horizons, in bars."
    ),
    start: str | None = typer.Option(None, help="Optional start time."),
    end: str | None = typer.Option(None, help="Optional end time."),
    funding: bool = typer.Option(
        True,
        "--funding/--no-funding",
        help="Attach stored perp funding, which Crowding needs. Perps only.",
    ),
    report_root: Path = typer.Option(Path("reports"), help="Report output folder."),
) -> None:
    """Score every registered state feature on one stored series.

    The R4 gate is that no feature ships unexamined, and this is the examination:
    coverage, distribution, lag-1 persistence, turnover, and the Spearman
    information coefficient against the forward return at each horizon -- each
    reported for both halves of the sample as well as the whole, because a
    feature that works in one half and not the other is a regime, not a signal.
    Expect small numbers: one feature rarely reaches |IC| 0.05 on 4h crypto.
    """
    from dataclasses import asdict
    from datetime import UTC, datetime

    from strategy_lab.backtests.engine import build_report_dir
    from strategy_lab.features.diagnostics import diagnose_features, to_record
    from strategy_lab.features.diagnostics_report import render_diagnostics_html
    from strategy_lab.features.registry import list_features

    parsed_horizons = _parse_horizons(horizons)
    identity = MarketDataIdentity(
        exchange=exchange, market_type=market_type, symbol=symbol, timeframe=timeframe
    )
    df = load_candles(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
    )
    if df.empty:
        _raise_missing_candles(identity)

    df, _ = _with_funding_column(identity, df, enabled=funding)

    diagnosable, skipped = _diagnosable_features(list_features(), df)
    if not diagnosable:
        raise typer.BadParameter(
            "No registered feature can be computed on this frame:\n"
            + "\n".join(f"  {name}: {reason}" for name, reason in skipped.items())
        )
    for name, reason in skipped.items():
        typer.secho(f"Skipping {name}: {reason}", fg=typer.colors.YELLOW)

    try:
        result = diagnose_features(diagnosable, df, horizons=parsed_horizons)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    config = {
        "identity": asdict(identity),
        "horizons": list(parsed_horizons),
        "data_start": str(df.index.min()),
        "data_end": str(df.index.max()),
        "candle_count": int(len(df)),
        "funding_attached": bool(funding and market_type == "perp"),
        "skipped": skipped,
        "generated_at": datetime.now(UTC).isoformat(),
    }

    report_dir = build_report_dir(report_root, identity, "features")
    report_dir.mkdir(parents=True, exist_ok=False)
    (report_dir / "features.html").write_text(
        render_diagnostics_html(result=result, config=config), encoding="utf-8"
    )
    (report_dir / "diagnostics.json").write_text(
        json.dumps({"config": config, **to_record(result)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    typer.echo(f"Wrote feature diagnostics for {symbol}: {report_dir}")
    for line in _feature_lines(result):
        typer.echo(line)
    for first, second, value in result.redundant_pairs():
        typer.echo(f"  redundant: {first} · {second} r={value:+.3f}")


def _parse_horizons(raw: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise typer.BadParameter(f"--horizons must be comma-separated integers: {exc}") from exc
    if not parsed or any(horizon < 1 for horizon in parsed):
        raise typer.BadParameter("--horizons values must all be >= 1 bar, for example 1,6,30")
    return parsed


def _diagnosable_features(names, df):
    """Split registered features into those this frame can support and those it cannot.

    Crowding refuses a frame with no funding, so an equity or spot series simply
    cannot carry it. Skipping is the honest outcome; a *silent* skip is how a
    feature ships unexamined, so every skip is echoed and written into the config.
    """
    from strategy_lab.features.registry import get_feature

    diagnosable, skipped = [], {}
    for name in names:
        feature = get_feature(name)
        # Outside the ``try`` on purpose: the ``except`` below files a ValueError
        # under ``skipped``, so a guard raising inside it would report a broken
        # declaration as "this frame cannot carry that feature" -- the silent
        # outcome, reached through the clause that exists to make skips honest.
        # What it refuses: ``head(-5 + 2)`` is 197 rows of a 200-row frame, and a
        # probe that reads nearly the whole frame is no longer a probe.
        require_warmup_bars(name, feature.warmup_bars)
        try:
            feature.compute(df.head(feature.warmup_bars + 2))
        except ValueError as exc:
            skipped[name] = str(exc).splitlines()[0]
            continue
        diagnosable.append(feature)
    return diagnosable, skipped


def _feature_lines(result) -> list[str]:
    """One line per feature: the ICs with their halves, and the closest neighbour."""
    lines = []
    for diagnostic in result.diagnostics:
        partner, correlation = result.max_correlation(diagnostic.name)
        ics = "  ".join(
            f"IC@{entry.horizon}b {entry.ic:+.4f} "
            f"({entry.first_half_ic:+.3f}/{entry.second_half_ic:+.3f})"
            for entry in diagnostic.ics
        )
        neighbour = f"  max|r| {correlation:+.3f} {partner}" if partner else ""
        lines.append(
            f"  {diagnostic.name:<20} cov {diagnostic.coverage:6.1%}  "
            f"turn {diagnostic.turnover:.4f}  {ics}{neighbour}"
        )
    return lines


@app.command("replay")
def replay_command(
    exchange: str = typer.Option("binance", help="Data exchange/source."),
    market_type: str = typer.Option("perp", help="spot, perp, or equity."),
    symbol: str = typer.Option("BTC/USDT", help="Symbol to replay."),
    timeframe: str = typer.Option("15m", help="Candle timeframe."),
    strategy_name: str = typer.Option("turnaround_v2", "--strategy", help="Strategy name."),
    start: str | None = typer.Option(None, help="Optional replay start time."),
    end: str | None = typer.Option(None, help="Optional replay end time."),
    limit_bars: int | None = typer.Option(None, help="Replay only the last N bars."),
    persist: bool = typer.Option(True, help="Write signals to Postgres."),
    funding: bool = typer.Option(
        True, help="Attach stored funding to perp bars so crowding is measured."
    ),
) -> None:
    """Replay stored candles bar-by-bar through the event engine.

    Same strategy object and same runner the live path will use; only the feed
    differs. That is also why this is slow next to ``backtest`` -- the strategy is
    re-evaluated on every bar rather than once over the range -- and
    ``tests/test_replay_determinism.py`` is what says the two agree anyway.

    **Funding reaches this path, so a perp replay matches its backtest.** Bars
    carry ``funding_rate`` and ``BarBuffer`` materializes the column, so
    ``features.flow.Crowding`` reads the same values here that ``backtest``
    attaches -- measured, ``state_machine_v1``'s per-bar features and state now
    agree between the two paths on every bar, where before R10f ``crowding``
    differed on all 6,048 of them.

    ``--funding/--no-funding`` says whether to attach it at all. When it is on,
    coverage is **required only for a strategy that reads a funding-derived
    feature** -- the same question ``sweep`` asks -- so a crowding-reading run
    over an uncovered range is refused with the message ``backtest`` gives, while
    ``donchian`` still replays BTC's permanent 40 h leading gap. ``--no-funding``,
    or an uncovered range for a strategy that does not read it, means no column at
    all: the feature falls back to a neutral 0.5 and the run records
    ``crowding_measured: false`` rather than reporting a number as though it had
    been measured.

    **This path also writes ``bar_reasons``**, one row per bar past warmup for
    any strategy that can explain itself, carrying the state label and the
    feature values the machine actually read. ``backtest``, ``sweep`` and the
    browser write none: they recompute the same values per request from immutable
    candles, and a stored copy of that would be a second research record to drift.
    What only this path can record is what a run saw at the moment it decided --
    including, today, that it decided with ``crowding`` neutral.

    **The run header, the signals and the reasons commit in three transactions,
    not one**, so a failure in the third leaves a run with signals and no
    reasons. That is deliberate rather than overlooked. It is the state every
    run before this phase is already in, so it degrades to the old record rather
    than to a corrupt one; the reverse order would leave a run that appears to
    have decided nothing, which is actively misleading; and a single transaction
    means threading one connection through ``create_run`` and ``write_signals``
    too, which is a change to the append-only signal path rather than to this
    phase. The failure is loud -- the exception propagates and the command exits
    non-zero -- and a re-run mints a fresh ``run_id`` rather than repairing the
    old one, exactly as a second replay of any range does.
    """
    from strategy_lab.backtests.funding_frame import FundingUnavailable
    from strategy_lab.core.clock import SimClock
    from strategy_lab.core.types import InstrumentId, Mode
    from strategy_lab.engine.runner import StrategyRunner
    from strategy_lab.feeds.base import Subscription
    from strategy_lab.feeds.replay import ReplayFeed

    strategy = get_strategy(strategy_name)
    instrument = InstrumentId(exchange, market_type, symbol)
    subscription = Subscription(instrument, timeframe)

    # The same question ``sweep`` asks, asked the same way: the strategy declares
    # its own inputs, so a run that *needs* funding refuses an uncovered range the
    # way a backtest of it would, and one that does not keeps replaying BTC's
    # permanent leading gap as it always has.
    needs_funding = "crowding" in getattr(strategy, "features", ())
    try:
        feed = ReplayFeed.from_database(
            [subscription],
            start=start,
            end=end,
            limit_bars=limit_bars,
            funding=funding,
            required=needs_funding,
        )
    except FundingUnavailable as exc:
        # Translated the way ``_funding_rates`` translates it for backtest and
        # sweep: the guard's message already names the fetch command and the
        # covered span, and a traceback would bury both.
        raise typer.BadParameter(str(exc)) from exc
    runner = StrategyRunner(
        strategy=strategy, instrument=instrument, timeframe=timeframe, clock=SimClock()
    )

    async def _run() -> list:
        collected = []
        async for event in feed.stream([subscription]):
            collected.extend(runner.on_event(event))
        return collected

    signals = asyncio.run(_run())
    reasons = runner.reasons

    # Reasons alone are enough to mint a run: a state machine that never changed
    # side over the range emitted nothing and still saw something on every bar,
    # and that is the case the per-bar table exists for. A strategy with no
    # `feature_frame` produces neither, so an empty replay still leaves no orphan
    # run header behind.
    if persist and (signals or reasons):
        run_id = _create_run(
            run_id=uuid.uuid4(),
            mode=Mode.REPLAY,
            strategy_id=strategy.name,
            strategy_version=strategy.version,
            config={
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "start": start,
                "end": end,
                "limit_bars": limit_bars,
                "warmup_bars": strategy.warmup_bars,
            },
        )
        written = _write_signals(run_id, Mode.REPLAY, signals)
        message = f"Run {run_id}: emitted {len(signals)} signals, wrote {written}."
        if reasons:
            reason_rows = _write_bar_reasons(run_id, Mode.REPLAY, reasons)
            message += f" Recorded {len(reasons)} bar reasons, wrote {reason_rows}."
        typer.echo(message)
        return

    message = f"Emitted {len(signals)} signals over {len(runner.buffer)} bars (not persisted)."
    if reasons:
        message += f" {len(reasons)} bar reasons not persisted."
    typer.echo(message)


# All three wrappers exist so a test can substitute storage without a database.
def _create_run(**kwargs):
    from strategy_lab.storage.signals import create_run

    return create_run(**kwargs)


def _write_signals(run_id, mode, signals):
    from strategy_lab.storage.signals import write_signals

    return write_signals(run_id, mode, signals)


def _write_bar_reasons(run_id, mode, reasons):
    from strategy_lab.storage.bar_reasons import write_bar_reasons

    return write_bar_reasons(run_id, mode, reasons)


@app.command("paper")
def paper_command(
    exchange: str = typer.Option("binance", help="Venue to poll."),
    market_type: str = typer.Option("perp", help="spot or perp."),
    symbol: str = typer.Option("BTC/USDT", help="Symbol to trade on paper."),
    timeframe: str = typer.Option("15m", help="Candle timeframe."),
    strategy_name: str = typer.Option("donchian", "--strategy", help="Strategy name."),
    exit_mode: str | None = typer.Option(None, help="Engine exit mode; the strategy's own if unset."),
    for_minutes: float = typer.Option(60.0, help="Wall-clock minutes to run for."),
    cash: float = typer.Option(10_000.0, help="Starting cash for the paper book."),
    position_pct: float = typer.Option(0.95, help="Fraction of cash an entry sizes against."),
    poll_seconds: float | None = typer.Option(None, help="Override the derived poll cadence."),
    persist: bool = typer.Option(True, help="Write signals and reasons to Postgres."),
    bars_csv: Path | None = typer.Option(
        None, help="Also log every live bar here, for the delayed oracle to check."
    ),
) -> None:
    """Run a strategy against the live venue on paper, and record what it saw.

    The process R10 and R10g built the parts for: ``LiveFeed`` polls the venue,
    ``StrategyRunner`` turns closed bars into signals over a buffer primed from
    ``backfill``, and ``PaperBook`` turns those into positions. Nothing reaches an
    exchange -- the book is a ledger, and R11 is where real size appears.

    **It writes no candles to ``market_candles``, deliberately.** The delayed
    oracle this phase exists for compares what the live path saw against what the
    venue serves for the same range *later*; a process that stored its own bars
    as the record would be compared against itself, which is not an oracle at
    all. Fetch the range afterwards with ``fetch-perp`` and replay it.

    ``--bars-csv`` is the other half of that and not a contradiction of it: the
    live bars have to be written down *somewhere* or there is nothing to hold the
    later fetch against, and a file beside the run is not the record of a
    dataset. Without it the oracle can still compare signals and per-bar reasons,
    but it cannot tell a venue revision from a closed-bar error, because both
    show up only as a derived difference.

    **A perp run advances its own funding.** Stored settlements move only when a
    funding fetch runs, so a process that only polls candles watches its window
    grow past its coverage until ``LiveFeed`` withholds every poll -- measured
    before this existed, stored funding was 46.6 h stale and no window could be
    covered at all. The top-up runs at startup and once per poll cycle thereafter,
    and it is the same fetch ``server.refresh_candles`` performs for the browser.

    **Bounded by the wall clock rather than by a bar count.** ``stream()`` does
    not terminate, and a bound that waited for bars would hang exactly when the
    feed had stopped producing them -- which is the condition worth observing.

    **Signals and reasons are written as they happen**, not at the end: a run
    measured in hours that flushed once would lose everything to the failure it
    was there to watch. ``write_signals`` is idempotent within a run, so an
    incremental flush is the same record arriving sooner. The run header is
    created up front, unlike ``replay``'s -- a paper run that recorded nothing
    still happened, and that it ran and produced nothing is the thing worth
    keeping.
    """
    from strategy_lab.core.clock import LiveClock
    from strategy_lab.core.types import InstrumentId, Mode
    from strategy_lab.engine.book import PaperBook
    from strategy_lab.engine.runner import StrategyRunner
    from strategy_lab.feeds.base import Subscription
    from strategy_lab.timeframes import timeframe_to_millis

    strategy = get_strategy(strategy_name)
    instrument = InstrumentId(exchange, market_type, symbol)
    subscription = Subscription(instrument, timeframe)
    identity = MarketDataIdentity(
        exchange=exchange, market_type=market_type, symbol=symbol, timeframe=timeframe
    )
    bar_ms = timeframe_to_millis(timeframe)

    settlements = _advance_funding(identity)
    if settlements is not None:
        typer.echo(f"Funding advanced: {settlements} settlements stored.")

    feed = _live_feed(poll_seconds=poll_seconds, clock=LiveClock())
    # One "now" in the process, and the feed owns it. The runner, the backfill
    # window and the funding cadence all read the same clock the polling does, so
    # anything that substitutes the feed substitutes time coherently rather than
    # leaving a second wall clock behind -- which is how a test that thought it
    # had scripted a venue primed zero bars from a window two years wide.
    clock = feed.clock
    runner = StrategyRunner(
        strategy=strategy,
        instrument=instrument,
        timeframe=timeframe,
        clock=clock,
        exit_mode=exit_mode,
    )

    # Warmup plus the lookback, so the first poll's re-read overlaps history
    # rather than landing beside it. `warmup_bars` is what makes a cold start
    # agree with a whole-history backtest, and for an EWM strategy it is ~20x the
    # declared span rather than the span.
    end_ms = clock.now_ms()
    start_ms = end_ms - bar_ms * (strategy.warmup_bars + feed.lookback_bars)

    async def _prime() -> list:
        return [bar async for bar in feed.backfill(subscription, start_ms, end_ms)]

    primed = asyncio.run(_prime())
    runner.prime_bars(primed)
    if primed:
        # Tell the feed where history left off, so the widened perp window does
        # not re-emit bars the buffer would only drop as out-of-order.
        feed.resume_after(subscription, primed[-1].ts_open_ms)
    typer.echo(
        f"Primed {len(primed)} bars ({strategy.name} wants {strategy.warmup_bars}); "
        f"buffer carries funding: {runner.buffer.carries_funding}."
    )
    if len(primed) < strategy.warmup_bars:
        typer.echo(
            "Warning: primed below warmup, so early bars emit nothing and the "
            "first signals will lag the run's start."
        )
    _warn_if_sizing_is_dropped(strategy, runner)

    book = PaperBook(cash=cash, position_pct=position_pct)
    run_id = _create_run(
        run_id=uuid.uuid4(),
        mode=Mode.PAPER,
        strategy_id=strategy.name,
        strategy_version=strategy.version,
        config={
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "timeframe": timeframe,
            "exit_mode": exit_mode,
            "for_minutes": for_minutes,
            "warmup_bars": strategy.warmup_bars,
            "primed_bars": len(primed),
            "cash": cash,
            "position_pct": position_pct,
            "started_ms": end_ms,
        },
    ) if persist else None

    signals: list = []
    written = {"signals": 0, "reasons": 0}
    bars = 0
    bar_log = _open_bar_log(bars_csv)

    async def _consume() -> None:
        nonlocal bars
        async for event in feed.stream([subscription]):
            bars += 1
            if bar_log is not None:
                _log_bar(bar_log, event.bar)
            emitted = runner.on_event(event)
            signals.extend(emitted)
            book.on_bar(
                emitted, close=float(event.bar.close), ts_bar_ms=event.bar.ts_open_ms
            )
            if run_id is not None:
                _flush(run_id, signals, runner.reasons, written)

    async def _keep_funding_current() -> None:
        """Advance settlements beside the stream, never inside it.

        This ran in the consumer loop first, and that is a deadlock by
        construction: a withheld poll yields no event, the loop body is what
        fetches funding, so the fetch that would end a stall can only happen
        while there is no stall. Measured on the first two real runs -- coverage
        lapsed at 00:00, one cadence past the 16:00 settlement, and both stalled
        for every remaining poll (27 and 26) and lost the bar that closed there.

        **A withheld poll is the signal to fetch.** The feed saying it cannot
        cover its window is better evidence that funding is behind than any
        timer, so the counter drives this and the timer is only the floor that
        keeps settlements roughly current when nothing is stalling. The fetch
        blocks the loop for its duration, which at these timeframes costs a
        poll's latency and is why it is not worth a thread.
        """
        withheld = feed.funding_withheld_polls
        elapsed = 0.0
        while True:
            await asyncio.sleep(FUNDING_CHECK_SECONDS)
            elapsed += FUNDING_CHECK_SECONDS
            stalling = feed.funding_withheld_polls > withheld
            withheld = feed.funding_withheld_polls
            if stalling or elapsed * 1000 >= bar_ms:
                _advance_funding(identity)
                elapsed = 0.0

    async def _run() -> None:
        funding = asyncio.create_task(_keep_funding_current())
        try:
            await _consume()
        finally:
            funding.cancel()

    try:
        asyncio.run(asyncio.wait_for(_run(), timeout=for_minutes * 60.0))
    except TimeoutError:
        pass  # the bound, not a failure
    except KeyboardInterrupt:
        typer.echo("Interrupted.")

    if run_id is not None:
        _flush(run_id, signals, runner.reasons, written)
    if bar_log is not None:
        bar_log.close()
        typer.echo(f"Live bars logged to {bars_csv}.")

    typer.echo(
        f"Ran {for_minutes:g} min: {bars} bars, {len(signals)} signals, "
        f"{len(runner.reasons)} reasons, {len(book.trades)} closed trades. "
        f"Withheld polls: {feed.funding_withheld_polls}."
    )
    if run_id is not None:
        typer.echo(
            f"Run {run_id}: wrote {written['signals']} signals, "
            f"{written['reasons']} reasons."
        )
    else:
        typer.echo("Not persisted.")


# How often the funding task looks at whether it is needed. Not how often it
# fetches: it fetches when the feed says it cannot cover its window, or once per
# bar, whichever comes first.
FUNDING_CHECK_SECONDS = 20.0


def _live_feed(**kwargs):
    """The feed, behind the same kind of seam as the three storage writers.

    ``LiveFeed.fetch`` is a dataclass field whose default is bound when the class
    is created, so patching either the attribute or the module function leaves
    the constructed feed pointing at the venue -- measured, a test that thought
    it had substituted a fixture was fetching from Binance. A factory is the seam
    that actually holds.
    """
    from strategy_lab.feeds.live import LiveFeed

    return LiveFeed(**kwargs)


def _open_bar_log(path: Path | None):
    """A csv of what the live path actually received, or ``None``.

    Flushed per bar rather than buffered: the file exists to survive whatever
    ends the run, and a bar still in a buffer when the process dies is exactly
    the bar worth having.
    """
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    handle.write("ts_open_ms,open,high,low,close,volume,funding_rate,is_closed\n")
    return handle


def _log_bar(handle, bar) -> None:
    funding = "" if bar.funding_rate is None else bar.funding_rate
    handle.write(
        f"{bar.ts_open_ms},{bar.open},{bar.high},{bar.low},{bar.close},"
        f"{bar.volume},{funding},{bar.is_closed}\n"
    )
    handle.flush()


def _advance_funding(identity) -> int | None:
    """Top up stored settlements for a perp. ``None`` off-perp, where none exist.

    The browser's own fetch, called rather than copied, so a paper process cannot
    advance funding by a rule the rest of the lab does not use.
    """
    from strategy_lab.server import _fetch_funding

    if identity.market_type != "perp":
        return None
    rows = _fetch_funding(identity, datetime.now(UTC) - timedelta(days=2))
    if rows is None:
        return None
    from strategy_lab.db.funding import upsert_funding

    return upsert_funding(rows)


def _warn_if_sizing_is_dropped(strategy, runner) -> None:
    """Say so when the book cannot size the way a backtest of this would.

    ``StrategyRunner`` emits ``strength=None`` and no ``Signal`` carries a
    ``position_size``, so the book sizes every entry at scale 1.0. For a strategy
    whose ``SignalSet`` carries a per-bar scale that is a real divergence from its
    backtest -- the one property R10g's book was built to have -- and it is worth
    a sentence rather than a quietly different number.
    """
    frame = runner.buffer.frame()
    if frame.empty:
        return
    size = getattr(strategy.generate_signals(frame), "position_size", None)
    if size is not None:
        typer.echo(
            f"Warning: {strategy.name} sizes per bar, and the event path carries no "
            f"size onto a Signal -- this book fills every entry at scale 1.0, so its "
            f"trades will not match a backtest of the same range."
        )


def _flush(run_id, signals: list, reasons: list, written: dict) -> None:
    """Persist whatever has not been persisted yet."""
    if len(signals) > written["signals"]:
        from strategy_lab.core.types import Mode

        _write_signals(run_id, Mode.PAPER, signals[written["signals"] :])
        written["signals"] = len(signals)
    if len(reasons) > written["reasons"]:
        from strategy_lab.core.types import Mode

        _write_bar_reasons(run_id, Mode.PAPER, reasons[written["reasons"] :])
        written["reasons"] = len(reasons)


@app.command("serve")
def serve_command(
    report_root: Path = typer.Option(Path("reports"), help="Backtest report folder to serve."),
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8750, help="Bind port."),
) -> None:
    """Serve the frozen per-run reports, with a candle-refresh API for their charts."""
    from strategy_lab.server import run_server

    typer.echo(f"Serving {report_root} at http://{host}:{port} (Ctrl+C to stop)")
    run_server(report_root=report_root, host=host, port=port)


@app.command("browse")
def browse_command(
    host: str = typer.Option("127.0.0.1", help="Bind host; loopback only."),
    port: int = typer.Option(BROWSER_PORT, help="Bind port."),
) -> None:
    """Open the read-only research browser: any strategy over any stored candle set.

    Opens on the board -- one tile per candle set and strategy, carrying the
    current state, the latest fill, the feature values behind them and the bar
    each is as of -- and every tile opens the single-instrument chart behind it.

    A companion to `serve`, not a replacement for it. `serve` hosts the frozen
    `plot.html` a backtest wrote -- the reproducibility record, dated and
    byte-identical on re-render. This recomputes from stored candles on every
    request and writes nothing, so it can show a strategy that was never run
    and can never become the record of one that was.
    """
    from strategy_lab.api.app import run_api

    typer.echo(f"Research browser at http://{host}:{port} (Ctrl+C to stop)")
    run_api(host=host, port=port)


@app.command("strategies")
def strategies_command() -> None:
    """List available strategy modules."""
    for strategy_name in list_strategies():
        typer.echo(strategy_name)


@app.command("data-sets")
def data_sets() -> None:
    """List candle sets currently stored in Postgres."""
    candle_sets = list_candle_sets()
    if candle_sets.empty:
        typer.echo("No candle data stored yet.")
        raise typer.Exit()

    typer.echo(candle_sets.to_string(index=False))


def _split_symbols(symbols: str) -> list[str]:
    return [symbol.strip() for symbol in symbols.split(",") if symbol.strip()]


def _raise_missing_candles(identity: MarketDataIdentity) -> None:
    candle_sets = list_candle_sets()
    available = "No candle data is stored yet."
    if not candle_sets.empty:
        available = candle_sets.to_string(index=False)

    fetch_hint = _fetch_hint(identity)
    message = (
        "No candles loaded for "
        f"{identity.exchange}/{identity.market_type}/{identity.symbol}/{identity.timeframe}.\n\n"
        f"Available candle sets:\n{available}\n\n"
        f"Fetch this data first:\n{fetch_hint}"
    )
    raise typer.BadParameter(message)


def _fetch_hint(identity: MarketDataIdentity) -> str:
    if identity.exchange == "yahoo" or identity.market_type == "equity":
        return (
            "strategy-lab fetch-stock "
            f"--symbol {identity.symbol} --timeframe {identity.timeframe} --period 2y"
        )

    return (
        "strategy-lab fetch-crypto "
        f"--exchange {identity.exchange} "
        f"--market-type {identity.market_type} "
        f"--symbol {identity.symbol} "
        f"--timeframe {identity.timeframe} "
        "--since 2024-01-01"
    )


if __name__ == "__main__":
    app()
