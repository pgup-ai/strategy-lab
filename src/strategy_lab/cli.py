from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import typer

from strategy_lab.backtests import ExitMode, SizeMode, run_backtest
from strategy_lab.db import init_db, list_candle_sets, load_candles, upsert_candles
from strategy_lab.db.candles import normalize_candle_frame
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.market_data.binance_futures import OPEN_INTEREST_HISTORY_DAYS
from strategy_lab.market_data.binance_futures import SOURCE as BINANCE_FUTURES_SOURCE
from strategy_lab.strategies import get_strategy, list_strategies
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
    exchange: str = typer.Option("binance", help="Venue id used in the stored identity."),
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
    exchange: str = typer.Option("binance", help="Venue id used in the stored identity."),
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
    exchange: str = typer.Option("binance", help="Venue id used in the stored identity."),
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
def _futures_client(**kwargs):
    from strategy_lab.market_data.binance_futures import BinanceFuturesClient

    return BinanceFuturesClient(**kwargs)


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
            "fixed deploys --position-pct on every entry. vol-target scales it by "
            "target / realized volatility, so risk rather than notional is what stays "
            "constant. The estimator is an EWM (span 96) that decays its seed instead "
            "of dropping it, so weights need roughly 20x span -- about 1,900 bars -- to "
            "converge; a shorter frame under-trades its early bars rather than erroring."
        ),
    ),
    vol_target: float = typer.Option(
        0.30,
        "--vol-target",
        min=0.0001,
        help="Annualized volatility to hold under --size-mode vol-target.",
    ),
    max_weight: float = typer.Option(
        2.0,
        "--max-weight",
        min=0.0001,
        help="Cap on the vol-target size multiplier, so a calm stretch cannot lever up.",
    ),
    cost_stress: str = typer.Option(
        "1",
        "--cost-stress",
        help="Comma-separated fee/slippage multiples to compare, for example 1,2,3.",
    ),
    funding: bool = typer.Option(
        True,
        "--funding/--no-funding",
        help="Charge stored perp funding at its settlement times.",
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
        rates = _funding_rates(identity, df) if funding else None
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


def _funding_rates(identity: MarketDataIdentity, df):
    """Stored funding for a perp, bounded to the candle window.

    A perp backtest that quietly skips funding reports a gross number that reads
    exactly like a net one -- and on this instrument the carry is roughly the
    size of buy-and-hold. Missing funding is therefore an error with an explicit
    opt-out, not a silent zero.
    """
    if identity.market_type != "perp":
        return None

    from strategy_lab.db.funding import load_funding

    rates = load_funding(
        exchange=identity.exchange,
        market_type=identity.market_type,
        symbol=identity.symbol,
        start=str(df.index.min()),
        end=str(df.index.max()),
    )
    if rates.empty:
        raise typer.BadParameter(
            f"No stored funding for {identity.exchange}/perp/{identity.symbol} over "
            f"{df.index.min()} -> {df.index.max()}.\n\n"
            "A perp backtest without funding is gross of carry and not a tradeable "
            "number. Fetch it first:\n"
            f"strategy-lab fetch-funding --symbol {identity.symbol} --since 2019-09-01\n\n"
            "Pass --no-funding to run gross of funding on purpose."
        )
    return rates["funding_rate"]


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
    report_root: Path = typer.Option(Path("reports"), help="Report output folder."),
) -> None:
    """Score a strategy across a parameter grid and render the stability surface.

    A single tuned parameter proves nothing; the R0 gate is a broad region where
    neighbouring parameters behave similarly. The stability score and the
    heatmap both exist to make a lone spike look like the overfit it is rather
    than like a result.
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
) -> None:
    """Replay stored candles bar-by-bar through the event engine.

    Same strategy object and same runner the live path will use; only the feed
    differs. That is also why this is slow next to ``backtest`` -- the strategy is
    re-evaluated on every bar rather than once over the range -- and
    ``tests/test_replay_determinism.py`` is what says the two agree anyway.
    """
    from strategy_lab.core.clock import SimClock
    from strategy_lab.core.types import InstrumentId, Mode
    from strategy_lab.engine.runner import StrategyRunner
    from strategy_lab.feeds.base import Subscription
    from strategy_lab.feeds.replay import ReplayFeed

    strategy = get_strategy(strategy_name)
    instrument = InstrumentId(exchange, market_type, symbol)
    subscription = Subscription(instrument, timeframe)

    feed = ReplayFeed.from_database(
        [subscription], start=start, end=end, limit_bars=limit_bars
    )
    runner = StrategyRunner(
        strategy=strategy, instrument=instrument, timeframe=timeframe, clock=SimClock()
    )

    async def _run() -> list:
        collected = []
        async for event in feed.stream([subscription]):
            collected.extend(runner.on_event(event))
        return collected

    signals = asyncio.run(_run())

    if persist and signals:
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
        typer.echo(f"Run {run_id}: emitted {len(signals)} signals, wrote {written}.")
        return

    typer.echo(f"Emitted {len(signals)} signals over {len(runner.buffer)} bars (not persisted).")


# Both wrappers exist so a test can substitute storage without a database.
def _create_run(**kwargs):
    from strategy_lab.storage.signals import create_run

    return create_run(**kwargs)


def _write_signals(run_id, mode, signals):
    from strategy_lab.storage.signals import write_signals

    return write_signals(run_id, mode, signals)


@app.command("serve")
def serve_command(
    report_root: Path = typer.Option(Path("reports"), help="Backtest report folder to serve."),
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8750, help="Bind port."),
) -> None:
    """Serve reports with a candle-refresh API that live-updates report charts."""
    from strategy_lab.server import run_server

    typer.echo(f"Serving {report_root} at http://{host}:{port} (Ctrl+C to stop)")
    run_server(report_root=report_root, host=host, port=port)


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
