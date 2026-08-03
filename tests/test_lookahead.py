from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from strategy_lab.strategies.base import SignalSet, validate_ohlcv
from strategy_lab.strategies.registry import get_strategy, list_strategies
from tests.conftest import synthetic_ohlcv

# Probe frames must be long ENOUGH, not just densely sampled. turnaround_v1/v2 declare
# warmup_bars=200, so a 400-bar frame leaves only 10 probe points after warmup. Measured
# over 40 seeds, that config misses a real lookahead bug in turnaround_v1 entirely on
# 32.5% of seeds (and a full-sample-normalization cheat on 27.5%) -- it passes on the
# default seed by luck. Halving `step` does NOT fix this; lengthening the frame does,
# because detectability depends on covering varied price regimes rather than on sampling
# density. At 1200 bars both miss rates drop to 0/40 for ~0.2s of runtime.
# Do not shrink this without re-running that measurement.
PROBE_BARS = 1200

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


def poison_probe(strategy, df: pd.DataFrame, *, warm: int, step: int = 20) -> list[int]:
    """Return the bar indices whose signals changed when the FUTURE was corrupted.

    A causal strategy cannot see past bar t, so replacing bars t+1.. with garbage
    must leave row t byte-identical. Any returned index is lookahead.
    """
    baseline = strategy.generate_signals(df)
    offenders: list[int] = []
    for t in range(warm, len(df) - 1, step):
        poisoned = df.copy()
        tail = poisoned.index[t + 1 :]
        poisoned.loc[tail, ["open", "high", "low", "close"]] = [1e6, 1.1e6, 0.9e6, 1e6]
        poisoned.loc[tail, "volume"] = 1e9
        probed = strategy.generate_signals(poisoned)
        for field in SIGNAL_FIELDS:
            want = getattr(baseline, field, None)
            got = getattr(probed, field, None)
            if want is None or got is None:
                continue
            if not _same(want.iloc[t], got.iloc[t]):
                offenders.append(t)
                break
    return offenders


def _same(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    return bool(a == b)


@pytest.mark.parametrize("name", list_strategies())
def test_registered_strategies_do_not_look_ahead(name):
    strategy = get_strategy(name)
    df = synthetic_ohlcv(n=PROBE_BARS)
    offenders = poison_probe(strategy, df, warm=strategy.warmup_bars)
    assert offenders == [], f"{name} used future data at bar indices {offenders}"


# --- The probe must be able to fail. These two strategies prove it. ---


@dataclass(frozen=True)
class _BlatantCheat:
    """Enters when the NEXT bar closes higher — textbook shift(-1) lookahead."""

    name: str = "blatant_cheat"
    version: str = "1.0.0"
    warmup_bars: int = 10

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        validate_ohlcv(df)
        longs = (df["close"].shift(-1) > df["close"]).fillna(False)
        flat = pd.Series(False, index=df.index)
        return SignalSet(longs, flat, flat, flat)


@dataclass(frozen=True)
class _SubtleCheat:
    """No shift(-1) anywhere — but normalizes by the FULL-SAMPLE mean."""

    name: str = "subtle_cheat"
    version: str = "1.0.0"
    warmup_bars: int = 10

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        validate_ohlcv(df)
        z = (df["close"] - df["close"].mean()) / df["close"].std()
        longs = (z < -1.0).fillna(False)
        flat = pd.Series(False, index=df.index)
        return SignalSet(longs, flat, flat, flat)


@pytest.mark.parametrize("cheat", [_BlatantCheat(), _SubtleCheat()])
def test_probe_detects_lookahead(cheat):
    df = synthetic_ohlcv(n=PROBE_BARS)
    offenders = poison_probe(cheat, df, warm=cheat.warmup_bars, step=10)
    assert offenders, f"{cheat.name} smuggled future data past the probe"
