from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from strategy_lab.backtests.engine import ExitMode, run_backtest
from strategy_lab.backtests.sizing import (
    SizeMode,
    realized_volatility,
    volatility_target_weights,
)
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.strategies import get_strategy


def _returns(scale: float, n: int = 500, seed: int = 5) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC", name="timestamp")
    return pd.Series(np.random.default_rng(seed).normal(0, scale, n), index=index)


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


# --- engine wiring ---------------------------------------------------------

_IDENTITY = MarketDataIdentity(
    exchange="binance", market_type="perp", symbol="BTC/USDT", timeframe="4h"
)


def _regime_shift_frame(n: int = 2400, calm: float = 0.003, wild: float = 0.02) -> pd.DataFrame:
    """A random walk whose second half is far more volatile than its first."""
    rng = np.random.default_rng(3)
    scale = np.where(np.arange(n) < n // 2, calm, wild)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1, n) * scale))
    index = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1000.0,
        },
        index=index,
    )


def _run(tmp_path, df, name="donchian", **kwargs):
    return run_backtest(
        df=df,
        strategy=get_strategy(name),
        identity=_IDENTITY,
        exit_mode=ExitMode.OPPOSITE_SIGNAL_ONLY,
        fees=0.0,
        slippage=0.0,
        report_root=tmp_path / str(len(list(tmp_path.iterdir()))),
        **kwargs,
    )


def _entry_notional(result) -> pd.Series:
    trades = pd.read_csv(result.trades_path, parse_dates=["Entry Timestamp"])
    return pd.Series(
        (trades["Size"] * trades["Avg Entry Price"]).to_numpy(),
        index=trades["Entry Timestamp"],
    )


def test_a_calm_regime_gets_a_larger_entry_than_a_violent_one(tmp_path):
    """The whole claim of vol targeting, measured through the engine's own orders.

    Fixed sizing deploys the same notional whatever the market is doing, which
    is what makes its risk swing with volatility. Vol targeting has to move that
    notional the other way -- and it has to survive the trip through
    ``SignalSet.position_size`` and vectorbt, not merely be correct in the
    module.
    """
    df = _regime_shift_frame()
    split = df.index[len(df) // 2]

    fixed = _entry_notional(_run(tmp_path, df))
    targeted = _entry_notional(_run(tmp_path, df, size_mode=SizeMode.VOL_TARGET))

    assert fixed.to_numpy() == pytest.approx(fixed.iloc[0])
    assert targeted[targeted.index < split].mean() > 3 * targeted[targeted.index >= split].mean()


def test_vol_target_refuses_a_strategy_that_already_sizes_itself(tmp_path):
    """Multiplying two inverse-vol scales targets neither of them, and says nothing."""
    df = _regime_shift_frame()

    with pytest.raises(ValueError, match="sizes its own positions"):
        _run(
            tmp_path,
            df,
            name="trend_rider_v1_deepseek_v4_pro",
            size_mode=SizeMode.VOL_TARGET,
        )


def test_the_sizing_choice_is_recorded_for_reproducibility(tmp_path):
    """``config.json`` is the reproducibility record; a run sized differently must say so."""
    df = _regime_shift_frame()

    fixed = json.loads((_run(tmp_path, df).report_dir / "config.json").read_text())
    targeted = json.loads(
        (
            _run(
                tmp_path, df, size_mode=SizeMode.VOL_TARGET, vol_target=0.25, max_weight=1.5
            ).report_dir
            / "config.json"
        ).read_text()
    )

    assert fixed["size_mode"] == "fixed"
    assert "vol_target" not in fixed
    assert targeted["size_mode"] == "vol-target"
    assert targeted["vol_target"] == 0.25
    assert targeted["max_weight"] == 1.5
    assert targeted["bars_per_year"] == pytest.approx(2191.5)
