from __future__ import annotations

import re

import pytest

from strategy_lab.strategies.registry import get_strategy, list_strategies

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


@pytest.mark.parametrize("name", list_strategies())
def test_every_strategy_declares_a_semver_version(name):
    strategy = get_strategy(name)
    assert SEMVER.match(strategy.version), f"{name} version {strategy.version!r} is not semver"


@pytest.mark.parametrize("name", list_strategies())
def test_every_strategy_declares_a_positive_warmup(name):
    strategy = get_strategy(name)
    assert isinstance(strategy.warmup_bars, int)
    assert strategy.warmup_bars > 0


@pytest.mark.parametrize("name", list_strategies())
def test_warmup_covers_the_largest_declared_lookback(name):
    """warmup_bars must be >= every span/period parameter the strategy declares."""
    strategy = get_strategy(name)
    spans = [
        value
        for field, value in vars(strategy).items()
        if isinstance(value, int) and (field.endswith("_span") or field.endswith("_period"))
    ]
    assert spans, f"{name} declares no span/period parameters to check against"
    assert strategy.warmup_bars >= max(spans)
