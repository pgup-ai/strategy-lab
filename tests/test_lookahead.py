from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


def _poison_flat(poisoned: pd.DataFrame, tail: pd.Index, t: int) -> None:
    """Violent but directionless: every corrupted bar has open == close.

    Effective against indicator-driven fields (EMA/ATR/rolling means), which this
    profile drives far out of range.
    """
    poisoned.loc[tail, ["open", "high", "low", "close"]] = [1e6, 1.1e6, 0.9e6, 1e6]
    poisoned.loc[tail, "volume"] = 1e9


def _poison_directional(poisoned: pd.DataFrame, tail: pd.Index, t: int) -> None:
    """Violent AND directional: corrupted bars are wide green/red candles.

    The flat profile leaves open == close, so `close > open` (green) and `close < open`
    (red) are uniformly False across the corrupted tail -- the exact predicate family
    every strategy in this repo is built on ("two red then a green"). This profile makes
    those predicates take both values so a candle-direction rule reading the future
    actually changes its answer.

    Directions are drawn from a PRNG seeded by the probe index `t`, which keeps the run
    deterministic while varying the pattern's PHASE across probe points. Phase is what
    matters: the tail always starts at bar t+1, so a fixed pattern gives every probe
    point the same leading candles. A strictly alternating profile is therefore no
    better than flat here -- red/green/red never contains "two red then a green" -- and
    a fixed run pattern only ever probes the one setup its leading bars happen to spell.
    """
    green = np.random.default_rng(t).random(len(tail)) < 0.5
    poisoned.loc[tail, "open"] = np.where(green, 0.5e6, 1.5e6)
    poisoned.loc[tail, "close"] = np.where(green, 1.5e6, 0.5e6)
    poisoned.loc[tail, "high"] = 1.6e6
    poisoned.loc[tail, "low"] = 0.4e6
    poisoned.loc[tail, "volume"] = 1e9


# Every probe runs under all profiles; lookahead found by any of them fails the gate.
POISON_PROFILES = (
    ("flat", _poison_flat),
    ("directional", _poison_directional),
)


def poison_probe(
    strategy,
    df: pd.DataFrame,
    *,
    warm: int,
    step: int = 20,
    profiles: tuple = POISON_PROFILES,
) -> list[tuple[str, int]]:
    """Return (profile, bar index) pairs whose signals changed when the FUTURE was corrupted.

    A causal strategy cannot see past bar t, so replacing bars t+1.. with garbage
    must leave row t byte-identical. Any returned pair is lookahead; the profile name
    says which corruption exposed it.
    """
    baseline = strategy.generate_signals(df)
    offenders: list[tuple[str, int]] = []
    for profile_name, poison in profiles:
        for t in range(warm, len(df) - 1, step):
            poisoned = df.copy()
            tail = poisoned.index[t + 1 :]
            poison(poisoned, tail, t)
            probed = strategy.generate_signals(poisoned)
            for field in SIGNAL_FIELDS:
                want = getattr(baseline, field, None)
                got = getattr(probed, field, None)
                if want is None or got is None:
                    continue
                if not _same(want.iloc[t], got.iloc[t]):
                    offenders.append((profile_name, t))
                    break
    return offenders


def _same(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    return bool(a == b)


def _profile(name: str) -> tuple:
    """Select one poison profile by name, for tests that compare profiles head to head."""
    selected = tuple(entry for entry in POISON_PROFILES if entry[0] == name)
    assert selected, f"no poison profile named {name!r} in {[e[0] for e in POISON_PROFILES]}"
    return selected


@pytest.mark.parametrize("name", list_strategies())
def test_registered_strategies_do_not_look_ahead(name):
    strategy = get_strategy(name)
    df = synthetic_ohlcv(n=PROBE_BARS)
    offenders = poison_probe(strategy, df, warm=strategy.warmup_bars)
    assert offenders == [], f"{name} used future data at (profile, bar index) {offenders}"


# --- The probe must be able to fail. These three strategies prove it. ---


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


@dataclass(frozen=True)
class _CandleDirectionCheat:
    """This repo's own three-candle turnaround, every candle read from the FUTURE.

    Purely candle-direction: no EMA, no ATR, no optional SignalSet fields. This is the
    shape the flat profile is weakest against, because a conjunction of direction
    predicates collapses to False on flat-poisoned bars.
    """

    name: str = "candle_direction_cheat"
    version: str = "1.0.0"
    warmup_bars: int = 10

    def generate_signals(self, df: pd.DataFrame) -> SignalSet:
        validate_ohlcv(df)
        red1 = df["close"].shift(-1) < df["open"].shift(-1)
        red2 = df["close"].shift(-2) < df["open"].shift(-2)
        green = df["close"].shift(-3) > df["open"].shift(-3)
        flat = pd.Series(False, index=df.index)
        return SignalSet((red1 & red2 & green).fillna(False), flat, flat, flat)


@pytest.mark.parametrize("cheat", [_BlatantCheat(), _SubtleCheat(), _CandleDirectionCheat()])
def test_probe_detects_lookahead(cheat):
    df = synthetic_ohlcv(n=PROBE_BARS)
    offenders = poison_probe(cheat, df, warm=cheat.warmup_bars, step=10)
    assert offenders, f"{cheat.name} smuggled future data past the probe"


def test_directional_profile_adds_detection_margin():
    """The second profile earns its place on candle-direction lookahead.

    Note the flat profile is NOT blind to this cheat -- measured over 40 seeds it always
    detects it at least once, so the gate was never walkable at PROBE_BARS. What the
    directional profile buys is margin: it fires at strictly more probe points (40/40
    seeds), which is what keeps detection robust if the frame or step is ever tightened.
    """
    cheat = _CandleDirectionCheat()
    df = synthetic_ohlcv(n=PROBE_BARS)
    flat_only = poison_probe(
        cheat, df, warm=cheat.warmup_bars, step=10, profiles=_profile("flat")
    )
    directional_only = poison_probe(
        cheat, df, warm=cheat.warmup_bars, step=10, profiles=_profile("directional")
    )
    assert directional_only, "directional profile failed to catch candle-direction lookahead"
    assert len(directional_only) > len(flat_only), (
        "directional profile should catch candle-direction lookahead at more probe "
        f"points than flat; got directional={len(directional_only)} flat={len(flat_only)}"
    )
