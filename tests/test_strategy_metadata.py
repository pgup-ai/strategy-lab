from __future__ import annotations

import re

import pandas as pd
import pytest

from strategy_lab.strategies.registry import get_strategy, list_strategies
from tests.conftest import synthetic_ohlcv

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")

# Bars probed after warmup. Each probe point costs one generate_signals call on a
# warmup-sized window, so this trades runtime against how many chances the check
# gets to catch a divergence. At 300, warmup=200 fails 6 times on turnaround_v1
# and turnaround_v2 -- enough margin that the check is not seed-luck.
COLD_START_PROBES = 300

SIGNAL_FIELDS = (
    "long_entries",
    "long_exits",
    "short_entries",
    "short_exits",
    "setup_stop_loss",
    "trend_failure_long_exits",
    "trend_failure_short_exits",
    "position_size",
)

# The strategies whose trend indicator is a recursive EMA, and the parameter
# naming its span. ``rolling`` windows are exactly windowed and need no entry
# here; ``ewm`` is recursive from the first bar and never fully forgets its seed.
# For ``ema_cross`` the SLOW span is the binding one: the fast EMA converges far
# sooner, so a warmup that makes the slow one bit-exact makes both exact. It has
# to be listed here, because for ema_cross specifically the signal-level check
# above is blind: measured at warmup=192, the span-192 EMA is wrong on 299/300
# probed bars by up to 2.6e-2 RELATIVE, and the fast-vs-slow comparison still
# comes out identical on 300/300. Only bit-exactness notices.
EWM_TREND_SPANS = {
    "turnaround_v1": "trend_failure_ema_span",
    "turnaround_v2": "ema_trend_span",
    "ema_cross": "slow_span",
}


def _same(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    return bool(a == b)


@pytest.mark.parametrize("name", list_strategies())
def test_every_strategy_declares_a_semver_version(name):
    strategy = get_strategy(name)
    assert SEMVER.match(strategy.version), f"{name} version {strategy.version!r} is not semver"


@pytest.mark.parametrize("name", list_strategies())
def test_declared_warmup_reproduces_whole_history_signals(name):
    """What ``warmup_bars`` actually promises, stated as a test.

    A live process starts cold: it fetches ``warmup_bars`` bars, then decides on
    each new one. A backtest sees everything. Those two agree only if
    ``warmup_bars`` is genuinely enough history, so this replays the cold start
    at every probe point and demands the current bar come out identical.

    ``warmup_bars >= max(declared span)`` does NOT imply this, which is the
    reasoning that set the turnaround strategies to 200 for a span-200 ``ewm``: a
    rolling window of 200 is complete after 200 bars, but ``ewm(adjust=False)``
    recurses from the first element and is still visibly wrong there. Measured at
    warmup=200, both turnaround strategies disagree with the whole-history run on
    6 of these 300 bars -- a signal that fires in a backtest and not in production.
    """
    strategy = get_strategy(name)
    warm = strategy.warmup_bars
    df = synthetic_ohlcv(n=warm + COLD_START_PROBES)
    whole_history = strategy.generate_signals(df)

    probes = range(warm, len(df))
    assert len(probes) == COLD_START_PROBES, "frame must extend past warmup or this proves nothing"

    divergences: list[tuple[int, str]] = []
    for position in probes:
        window = df.iloc[position - warm : position + 1]
        cold = strategy.generate_signals(window)
        for field in SIGNAL_FIELDS:
            expected = getattr(whole_history, field, None)
            actual = getattr(cold, field, None)
            if expected is None or actual is None:
                continue
            if not _same(expected.iloc[position], actual.iloc[-1]):
                divergences.append((position, field))
                break

    assert divergences == [], (
        f"{name} declares warmup_bars={warm}, but a cold start from exactly that "
        f"many bars disagrees with the whole-history run at {divergences[:5]} "
        f"({len(divergences)}/{COLD_START_PROBES} probed bars). Raise warmup_bars."
    )


@pytest.mark.parametrize("name", sorted(EWM_TREND_SPANS))
def test_recursive_ema_is_bit_exact_after_the_declared_warmup(name):
    """For ewm strategies, agreeing signals is not a strong enough bar.

    ``ewm(adjust=False)`` never fully forgets its seed; it decays it by
    ``(1 - 2/(span+1))`` per bar. So "the signals happened to agree over N
    sampled bars" only means no sampled close landed inside the residual error
    band. Measured on the stored BTC/USDT 15m history, the band around the
    span-200 EMA is 7.8e-4 relative at warmup 500 and 4.7e-6 at warmup 1000 --
    small, but a close falling inside it still flips the comparison, and closes
    sit near the trend EMA constantly.

    The only warmup at which divergence is impossible rather than merely
    unobserved is the one where the EMA is bit-identical to the whole-history
    value. For span 200 that is ~4000 bars (20x span, not 10x): the residual is
    1.4e-10 at 2000 and 6.4e-15 at 3000, reaching exactly 0 at 4000. That is
    where these strategies' warmup_bars comes from.
    """
    strategy = get_strategy(name)
    warm = strategy.warmup_bars
    span = getattr(strategy, EWM_TREND_SPANS[name])
    df = synthetic_ohlcv(n=warm + COLD_START_PROBES)

    whole_history = df["close"].ewm(span=span, adjust=False).mean()

    inexact = []
    for position in range(warm, len(df)):
        window = df["close"].iloc[position - warm : position + 1]
        cold = window.ewm(span=span, adjust=False).mean().iloc[-1]
        if cold != whole_history.iloc[position]:
            inexact.append(position)

    assert inexact == [], (
        f"{name}: the span-{span} EMA is not bit-exact after warmup_bars={warm} "
        f"on {len(inexact)}/{COLD_START_PROBES} probed bars; signals can still "
        f"diverge from a backtest on any bar that closes near it."
    )
