from __future__ import annotations

from strategy_lab.features.base import StateFeature
from strategy_lab.features.flow import Crowding, Participation
from strategy_lab.features.trend import Direction, Persistence, Stability, Strength
from strategy_lab.features.volatility import Compression, CompressionRelease, Energy


def list_features() -> list[str]:
    """Every feature the lookahead probe and the diagnostics run over.

    Registration is manual and in two places, exactly as in
    ``strategies/registry.py``. The cost is remembering both; the benefit is that
    a feature is covered by ``tests/test_feature_lookahead.py`` the moment it is
    named here, and half of these are percentiles -- the one construction that
    leaks the future without a ``shift(-1)`` anywhere in sight.
    """
    return [
        "direction",
        "strength",
        "persistence",
        "stability",
        "energy",
        "compression",
        "compression_release",
        "participation",
        "crowding",
    ]


def get_feature(name: str) -> StateFeature:
    if name == "direction":
        return Direction()
    if name == "strength":
        return Strength()
    if name == "persistence":
        return Persistence()
    if name == "stability":
        return Stability()
    if name == "energy":
        return Energy()
    if name == "compression":
        return Compression()
    if name == "compression_release":
        return CompressionRelease()
    if name == "participation":
        return Participation()
    if name == "crowding":
        return Crowding()
    raise ValueError(f"Unknown feature {name!r}. Available: {', '.join(list_features())}")
