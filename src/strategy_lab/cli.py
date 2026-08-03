from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import typer

from strategy_lab.backtests import ExitMode, run_backtest
from strategy_lab.db import init_db, list_candle_sets, load_candles, upsert_candles
from strategy_lab.db.candles import normalize_candle_frame
from strategy_lab.market_data.base import MarketDataIdentity
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
    symbols: str = typer.Option("BTC/USDT", help="Comma-separated symbols."),
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
    report_root: Path = typer.Option(Path("reports"), help="Report output folder."),
) -> None:
    """Run a vectorbt backtest for one or more stored symbols."""
    strategy = get_strategy(strategy_name, allow_shorts=allow_shorts)
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
        )
        typer.echo(f"Wrote report for {symbol}: {result.report_dir}")


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
    differs. That is the point -- and it is why this is slow next to ``backtest``:
    the strategy is re-evaluated on every bar rather than once over the range.
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


def _create_run(**kwargs):
    """Indirection so a test can substitute storage without a database."""
    from strategy_lab.storage.signals import create_run

    return create_run(**kwargs)


def _write_signals(run_id, mode, signals):
    """Indirection so a test can substitute storage without a database."""
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
