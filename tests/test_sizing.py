from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_lab.backtests.sizing import realized_volatility, volatility_target_weights


def _returns(scale: float, n: int = 500, seed: int = 5) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC", name="timestamp")
    return pd.Series(np.random.default_rng(seed).normal(0, scale, n), index=index)


def test_a_calmer_market_gets_a_larger_weight():
    calm, wild = _returns(0.005), _returns(0.05)
    w_calm = volatility_target_weights(calm, target_annual_vol=0.10, bars_per_year=2190)
    w_wild = volatility_target_weights(wild, target_annual_vol=0.10, bars_per_year=2190)
    assert w_calm.iloc[-1] > w_wild.iloc[-1]


def test_weights_are_capped():
    weights = volatility_target_weights(
        _returns(0.0001), target_annual_vol=0.10, bars_per_year=2190, max_weight=2.0
    )
    assert weights.max() <= 2.0


def test_realized_volatility_annualizes():
    daily = _returns(0.01, n=400)
    annual = realized_volatility(daily, span=100, bars_per_year=2190)
    assert annual.iloc[-1] == pytest.approx(0.01 * (2190**0.5), rel=0.35)


def test_zero_volatility_does_not_produce_infinite_weight():
    flat = pd.Series(
        0.0, index=pd.date_range("2024-01-01", periods=200, freq="4h", tz="UTC", name="timestamp")
    )
    weights = volatility_target_weights(flat, target_annual_vol=0.10, bars_per_year=2190)
    assert np.isfinite(weights).all()


def test_a_ten_times_calmer_market_gets_a_ten_times_larger_weight():
    """Targeting constant *risk* means weight is inversely proportional to volatility."""
    calm, wild = _returns(0.001), _returns(0.01)
    w_calm = volatility_target_weights(
        calm, target_annual_vol=0.10, bars_per_year=2190, max_weight=1e6
    )
    w_wild = volatility_target_weights(
        wild, target_annual_vol=0.10, bars_per_year=2190, max_weight=1e6
    )
    assert w_calm.iloc[-1] / w_wild.iloc[-1] == pytest.approx(10.0, rel=0.02)


def test_doubling_the_target_doubles_the_weight():
    returns = _returns(0.01)
    single = volatility_target_weights(returns, target_annual_vol=0.10, bars_per_year=2190)
    double = volatility_target_weights(returns, target_annual_vol=0.20, bars_per_year=2190)
    assert double.iloc[-1] == pytest.approx(2 * single.iloc[-1], rel=1e-9)


def test_weights_are_causal():
    """Sizing must not read the future -- a later shock cannot change an earlier weight."""
    returns = _returns(0.01)
    poisoned = returns.copy()
    poisoned.iloc[300:] = 5.0
    base = volatility_target_weights(returns, target_annual_vol=0.10, bars_per_year=2190)
    after = volatility_target_weights(poisoned, target_annual_vol=0.10, bars_per_year=2190)
    pd.testing.assert_series_equal(base.iloc[:300], after.iloc[:300])


def test_realized_volatility_scales_with_the_bar_count_per_year():
    """The same bar-level noise annualizes higher when there are more bars in a year."""
    returns = _returns(0.01)
    hourly = realized_volatility(returns, span=100, bars_per_year=8760)
    four_hourly = realized_volatility(returns, span=100, bars_per_year=2190)
    assert hourly.iloc[-1] / four_hourly.iloc[-1] == pytest.approx(2.0, rel=1e-9)


def test_a_missing_return_does_not_poison_every_later_weight():
    returns = _returns(0.01)
    returns.iloc[10] = np.nan
    weights = volatility_target_weights(returns, target_annual_vol=0.10, bars_per_year=2190)
    assert np.isfinite(weights).all()


def test_sizing_is_blind_to_direction():
    """A size multiplier, not a direction: mirroring every return must not move a weight."""
    returns = _returns(0.01)
    up = volatility_target_weights(returns, target_annual_vol=0.10, bars_per_year=2190)
    down = volatility_target_weights(-returns, target_annual_vol=0.10, bars_per_year=2190)
    assert (up >= 0).all()
    pd.testing.assert_series_equal(up, down)
