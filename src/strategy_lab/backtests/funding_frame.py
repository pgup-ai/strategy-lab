"""The frame a perp strategy should actually see, and the settlements behind it.

Lifted out of ``cli.py`` when the research browser needed the same rule. It is
one rule with two consumers now -- the CLI's ``backtest``/``sweep``/``features``
commands and ``api.analysis`` -- and a second copy is the failure this module
exists to prevent: the browser's whole claim is that it cannot disagree with a
backtest, and a browser that attached funding by its own slightly different rule
would disagree in exactly the way M20 did.

What the CLI keeps is the *exit code*: ``FundingUnavailable`` is a plain
``ValueError`` here, and ``cli._funding_rates`` is the seam that turns it into a
``typer.BadParameter``. A library module that raised a click exception would
make every non-CLI caller catch a CLI framework's type.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from strategy_lab.market_data.base import MarketDataIdentity


class FundingUnavailable(ValueError):
    """Stored funding is missing or does not cover the requested window."""


def funding_rates(
    identity: MarketDataIdentity, df: pd.DataFrame, *, required: bool = True
) -> pd.Series | None:
    """Stored funding for a perp, bounded to the candle window.

    A perp backtest that quietly skips funding reports a gross number that reads
    exactly like a net one -- and on this instrument the carry is roughly the
    size of buy-and-hold. Missing funding is therefore an error with an explicit
    opt-out, not a silent zero. A *partial* history is the same error wearing a
    disguise, so the stored series has to cover the window as well as exist.
    """
    if identity.market_type != "perp":
        return None

    from strategy_lab.backtests.costs import funding_coverage_gaps, window_end
    from strategy_lab.db.funding import load_funding

    # The right bound is the last bar's exclusive right edge, not its opening
    # timestamp: a bar covers an interval, and Binance stamps settlements up to
    # 47 ms past the boundary, so bounding at the open drops a settlement that
    # falls inside the final bar -- either failing the coverage check on a
    # complete history or charging that settlement as zero.
    rates = load_funding(
        exchange=identity.exchange,
        market_type=identity.market_type,
        symbol=identity.symbol,
        start=str(df.index.min()),
        end=str(window_end(df.index)),
    )
    if rates.empty:
        if not required:
            return None
        raise FundingUnavailable(
            f"No stored funding for {identity.exchange}/perp/{identity.symbol} over "
            f"{df.index.min()} -> {df.index.max()}.\n\n"
            "A perp backtest without funding is gross of carry and not a tradeable "
            "number. Fetch it first:\n"
            f"strategy-lab fetch-funding --symbol {identity.symbol} --since 2019-09-01\n\n"
            "Pass --no-funding to proceed without it: on a backtest that means gross of "
            "carry, and on a sweep -- which never charges carry -- it means any "
            "funding-derived feature falls back to neutral."
        )

    gaps = funding_coverage_gaps(funding=rates["funding_rate"], index=df.index)
    if gaps:
        if not required:
            return None
        raise FundingUnavailable(_uncovered_funding(identity, df, gaps))
    return rates["funding_rate"]


def with_funding_column(
    identity: MarketDataIdentity,
    df: pd.DataFrame,
    *,
    enabled: bool,
    required: bool = True,
    load_rates: Callable[..., pd.Series | None] | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """``df`` with the per-bar ``funding_rate`` column, plus the settlement series.

    Funding is two different things to two different consumers, and ``backtest``
    served only the first. It is a *cash flow*, which the engine charges against
    the notional held into each settlement -- that is the returned series. It is
    also an *input*: ``features.flow.Crowding`` is the state machine's fourth
    feature and reads a per-bar ``funding_rate`` column, falling back to a
    neutral 0.5 when the frame has none. Loading funding for the cost ledger and
    then handing the strategy a frame without it ran a *different strategy* than
    the research measured, recording ``crowding_measured: false`` in a corner of
    ``config.json`` while the headline moved. Measured on BTC/USDT perp 4h over
    R5's test half, trained cell: +16.44% / Sharpe +0.801 / 6.08% max drawdown
    without the column, +15.45% / +0.896 / 4.67% with it -- the second is
    charter §9.2's published row.

    Alignment goes through ``align_funding_to_bars``, never a reindex: Binance
    stamps settlements up to 47 ms past the bar boundary, so an equality join
    drops 43% of the stored history and the survivors land on the wrong bars.

    ``load_rates`` is injected so the CLI can wrap the loader's refusal in its
    own exit code without owning a second copy of the attachment rule below. It
    defaults to ``None`` rather than to :func:`funding_rates` because a function
    named as a default argument is captured at import, which would make the
    module attribute look substitutable while binding whatever it was at import
    time.
    """
    if not (enabled and identity.market_type == "perp"):
        return df, None

    from strategy_lab.features.flow import FUNDING_COLUMN, align_funding_to_bars

    rates = (load_rates or funding_rates)(identity, df, required=required)
    if rates is None:
        return df, None
    return df.assign(**{FUNDING_COLUMN: align_funding_to_bars(df.index, rates)}), rates


def _uncovered_funding(identity: MarketDataIdentity, df: pd.DataFrame, gaps) -> str:
    """Names the stretches that fail *and* the window that would not.

    "Narrow the run with --start" is only advice if the reader can work out what
    to narrow it to, and from the gap list alone they cannot -- BTC's leading gap
    is permanent, so the whole frame is refused and the fix is a start date the
    message never mentions. The covered span is one query and turns prose into
    two dates that can be typed.
    """
    from strategy_lab.db.funding import funding_span

    shown = ", ".join(f"{start} -> {end}" for start, end in gaps[:3])
    more = f" (+{len(gaps) - 3} more)" if len(gaps) > 3 else ""
    span = funding_span(
        exchange=identity.exchange, market_type=identity.market_type, symbol=identity.symbol
    )
    covered = (
        f"Stored funding covers {span[0]} -> {span[1]}.\n\n" if span is not None else ""
    )
    return (
        f"Stored funding for {identity.exchange}/perp/{identity.symbol} does not "
        f"cover {df.index.min()} -> {df.index.max()}.\n\n"
        f"Uncovered: {shown}{more}\n\n"
        f"{covered}"
        "Every settlement missing from those stretches is charged as zero, so the "
        "run would report a net-of-funding number that is gross of carry across "
        "them. Backfill the range:\n"
        f"strategy-lab fetch-funding --symbol {identity.symbol} "
        f"--since {gaps[0][0]:%Y-%m-%d}\n\n"
        "If the venue genuinely settled nothing there, narrow the run with --start. "
        "Pass --no-funding to proceed without it: on a backtest that means gross of "
        "carry, and on a sweep -- which never charges carry -- it means any "
        "funding-derived feature falls back to neutral."
    )


__all__ = ["FundingUnavailable", "funding_rates", "with_funding_column"]
