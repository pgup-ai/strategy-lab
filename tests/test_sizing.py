from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from strategy_lab.backtests.engine import ExitMode, run_backtest
from strategy_lab.backtests.sizing import (
    SizeMode,
    realized_volatility,
    vol_warmup_bars,
    volatility_target_weights,
)
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.strategies import get_strategy


# Small enough that ``20 x span`` warmup bars leave most of a 500-bar frame
# converged. The production default is 96, whose 1,920-bar warmup would zero
# every weight on these frames and make each assertion below vacuously true.
_SPAN = 20
_WARM = vol_warmup_bars(_SPAN)


def _returns(scale: float, n: int = 500, seed: int = 5) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC", name="timestamp")
    return pd.Series(np.random.default_rng(seed).normal(0, scale, n), index=index)


def _weights(returns: pd.Series, **kwargs) -> pd.Series:
    kwargs.setdefault("target_annual_vol", 0.10)
    kwargs.setdefault("bars_per_year", 2190)
    kwargs.setdefault("span", _SPAN)
    return volatility_target_weights(returns, **kwargs)


def test_weights_are_capped():
    weights = _weights(_returns(0.0001), max_weight=2.0)
    assert weights.max() == pytest.approx(2.0), (
        "a market this calm must request more than the cap, or the ceiling is "
        "never exercised and this test asserts nothing"
    )


def test_realized_volatility_annualizes():
    daily = _returns(0.01, n=400)
    annual = realized_volatility(daily, span=100, bars_per_year=2190)
    assert annual.iloc[-1] == pytest.approx(0.01 * (2190**0.5), rel=0.35)


def test_zero_volatility_does_not_produce_infinite_weight():
    flat = pd.Series(
        0.0, index=pd.date_range("2024-01-01", periods=500, freq="4h", tz="UTC", name="timestamp")
    )
    weights = _weights(flat)
    assert np.isfinite(weights).all()


def test_a_ten_times_calmer_market_gets_a_ten_times_larger_weight():
    """Targeting constant *risk* means weight is inversely proportional to volatility."""
    calm, wild = _returns(0.001), _returns(0.01)
    w_calm = _weights(calm, max_weight=1e6)
    w_wild = _weights(wild, max_weight=1e6)
    assert w_calm.iloc[-1] / w_wild.iloc[-1] == pytest.approx(10.0, rel=0.02)


def test_doubling_the_target_doubles_the_weight():
    returns = _returns(0.01)
    single = _weights(returns)
    double = _weights(returns, target_annual_vol=0.20)
    assert single.iloc[-1] > 0, "a zero weight would make the ratio below vacuous"
    assert double.iloc[-1] == pytest.approx(2 * single.iloc[-1], rel=1e-9)


def test_weights_are_causal():
    """Sizing must not read the future -- a later shock cannot change an earlier weight."""
    poison_at = _WARM + 50
    returns = _returns(0.01)
    poisoned = returns.copy()
    poisoned.iloc[poison_at:] = 5.0
    base = _weights(returns)
    after = _weights(poisoned)
    assert (base.iloc[:poison_at] > 0).any(), (
        "every weight before the poison is zero, so the comparison below holds "
        "for any implementation"
    )
    pd.testing.assert_series_equal(base.iloc[:poison_at], after.iloc[:poison_at])


def test_realized_volatility_scales_with_the_bar_count_per_year():
    """The same bar-level noise annualizes higher when there are more bars in a year."""
    returns = _returns(0.01)
    hourly = realized_volatility(returns, span=100, bars_per_year=8760)
    four_hourly = realized_volatility(returns, span=100, bars_per_year=2190)
    assert hourly.iloc[-1] / four_hourly.iloc[-1] == pytest.approx(2.0, rel=1e-9)


def test_a_missing_return_does_not_poison_every_later_weight():
    returns = _returns(0.01)
    returns.iloc[10] = np.nan
    weights = _weights(returns)
    assert np.isfinite(weights).all()
    assert (weights > 0).any(), "an all-zero series is finite for the wrong reason"


def test_sizing_is_blind_to_direction():
    """A size multiplier, not a direction: mirroring every return must not move a weight."""
    returns = _returns(0.01)
    up = _weights(returns)
    down = _weights(-returns)
    assert (up > 0).any(), "an all-zero series matches its mirror trivially"
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
    """The whole claim of vol *scaling*, measured through the engine's own orders.

    Fixed sizing deploys the same notional whatever the market is doing, which
    is what makes its risk swing with volatility. Vol scaling has to move that
    notional the other way -- and it has to survive the trip through
    ``SignalSet.position_size`` and vectorbt, not merely be correct in the
    module.
    """
    df = _regime_shift_frame()
    split = df.index[len(df) // 2]

    fixed = _entry_notional(_run(tmp_path, df))
    targeted = _entry_notional(
        _run(tmp_path, df, size_mode=SizeMode.VOL_SCALED_ENTRY, vol_span=_SPAN)
    )

    assert targeted.index.min() < split, (
        f"every vol-scaled entry landed after the regime split; the span-{_SPAN} "
        f"estimator's {_WARM}-bar warmup covers the calm half and there is "
        f"nothing to compare"
    )

    assert fixed.to_numpy() == pytest.approx(fixed.iloc[0])
    assert targeted[targeted.index < split].mean() > 3 * targeted[targeted.index >= split].mean()


def test_only_the_entry_bar_weight_lands_and_a_later_one_never_resizes(tmp_path):
    """The mode scales *entries*; it does not retarget a position it already holds.

    ``vbt.Portfolio.from_signals`` defaults to ``accumulate=False``, so a
    repeated same-direction entry while a position is open is ignored and the
    per-bar weight series is consumed on exactly one bar per position -- the one
    that opens it. This test is the guard against someone reading the weights as
    continuous volatility targeting: it pins each fill to the weight at *its own*
    entry bar, on a frame where the weight moves by more than 5x during at least
    one holding period. If the engine ever gains real rebalancing (R6), this test
    is what will fail and say so.

    ``max_weight=1.0`` keeps every requested weight inside the book's buying
    power, so a fill is never clipped for a reason unrelated to what is being
    measured and the notional matches to float precision rather than loosely.
    """
    df = _regime_shift_frame(wild=0.05)
    result = _run(
        tmp_path, df, size_mode=SizeMode.VOL_SCALED_ENTRY, max_weight=1.0, vol_span=_SPAN
    )

    weights = volatility_target_weights(
        df["close"].pct_change(),
        target_annual_vol=0.30,
        bars_per_year=2191.5,
        span=_SPAN,
        max_weight=1.0,
    )
    trades = pd.read_csv(
        result.trades_path, parse_dates=["Entry Timestamp", "Exit Timestamp"]
    )
    assert not trades.empty

    notional = trades["Size"] * trades["Avg Entry Price"]
    entry_weights = weights.reindex(trades["Entry Timestamp"]).to_numpy()
    assert notional.to_numpy() == pytest.approx(10_000.0 * 0.95 * entry_weights, rel=1e-9)

    held = [
        weights.loc[entry_ts:exit_ts]
        for entry_ts, exit_ts in zip(
            trades["Entry Timestamp"], trades["Exit Timestamp"], strict=True
        )
    ]
    assert max((span.max() / span.min()) for span in held if span.min() > 0) > 5.0


def test_from_signals_ignores_a_repeat_entry_which_is_why_rebalancing_is_absent(tmp_path):
    """The upstream fact the whole ``vol-scaled-entry`` naming rests on.

    Pinned directly rather than only inferred through the engine, so that a
    vectorbt upgrade which starts honouring later sizes fails *here* -- next to
    the docstring claiming it does not -- instead of silently making
    ``backtests/sizing.py`` describe behaviour the package no longer has.
    """
    vbt = pytest.importorskip("vectorbt")
    index = pd.date_range("2024-01-01", periods=8, freq="4h", tz="UTC", name="timestamp")
    pf = vbt.Portfolio.from_signals(
        close=pd.Series(100.0, index=index),
        entries=pd.Series(True, index=index),
        exits=pd.Series(False, index=index),
        size=pd.Series([1, 1, 1, 1, 5, 5, 5, 5], index=index, dtype="float64"),
        init_cash=100_000.0,
        freq="4h",
    )

    assert len(pf.orders.records_readable) == 1
    assert pf.assets().to_numpy() == pytest.approx(1.0)


def test_vol_scaling_refuses_a_strategy_that_already_sizes_itself(tmp_path):
    """Multiplying two inverse-vol scales targets neither of them, and says nothing."""
    df = _regime_shift_frame()

    with pytest.raises(ValueError, match="sizes its own positions"):
        _run(
            tmp_path,
            df,
            name="trend_rider_v1_deepseek_v4_pro",
            size_mode=SizeMode.VOL_SCALED_ENTRY,
        )


def test_the_sizing_choice_is_recorded_for_reproducibility(tmp_path):
    """``config.json`` is the reproducibility record; a run sized differently must say so."""
    df = _regime_shift_frame()

    fixed = json.loads((_run(tmp_path, df).report_dir / "config.json").read_text())
    targeted = json.loads(
        (
            _run(
                tmp_path, df, size_mode=SizeMode.VOL_SCALED_ENTRY, vol_target=0.25, max_weight=1.5
            ).report_dir
            / "config.json"
        ).read_text()
    )

    assert fixed["size_mode"] == "fixed"
    assert "vol_target" not in fixed
    assert targeted["size_mode"] == "vol-scaled-entry"
    assert targeted["vol_target"] == 0.25
    assert targeted["max_weight"] == 1.5
    assert targeted["max_weight_effective"] == pytest.approx(1 / 0.95)
    assert targeted["bars_per_year"] == pytest.approx(2191.5)


def test_a_weight_above_buying_power_is_capped_and_named(tmp_path):
    """The advertised ``--max-weight 2.0`` cannot be filled at 95% deployment:
    an entry is sized as cash x position_pct x weight and the book has no
    leverage. Silently clamping is the failure mode -- a config that claims
    something the run did not do."""
    df = _regime_shift_frame()

    with pytest.warns(UserWarning, match="1.053"):
        result = _run(tmp_path, df, size_mode=SizeMode.VOL_SCALED_ENTRY, max_weight=2.0)

    config = json.loads((result.report_dir / "config.json").read_text())
    assert config["max_weight"] == 2.0
    assert config["max_weight_effective"] == pytest.approx(1 / 0.95)


def test_a_weight_the_book_can_fill_is_left_alone(tmp_path):
    """Halving deployment doubles the executable ceiling, so the same cap that
    was clipped above now passes through untouched and unmentioned."""
    df = _regime_shift_frame()

    with warnings.catch_warnings():
        # Only the sizing warning should fail this test -- a bare
        # simplefilter("error") also promotes unrelated FutureWarnings from
        # pandas/vectorbt, failing it for reasons that have nothing to do here.
        warnings.simplefilter("ignore")
        warnings.filterwarnings("error", message=".*max_weight.*")
        result = _run(
            tmp_path, df, size_mode=SizeMode.VOL_SCALED_ENTRY, max_weight=2.0, position_pct=0.5
        )

    config = json.loads((result.report_dir / "config.json").read_text())
    assert config["max_weight_effective"] == 2.0


# --- estimator warmup ------------------------------------------------------


def test_the_estimator_warmup_is_zero_weighted():
    """``ewm().std()`` is finite long before it means anything.

    An estimate exists from the second observation, so the earlier version
    returned a full series whose leading values were a function of where the
    frame happened to start. Measured on the canonical BTC perp 4h frame at the
    production span of 96: the cold-start weight peaks at 1.551 and sits at or
    above the 1.053 the book can actually fill on **104 of the 1,920** warmup
    bars, against a converged mean of 0.644.
    """
    weights = _weights(_returns(0.01), max_weight=1e6)

    assert (weights.iloc[:_WARM] == 0.0).all(), (
        f"weights inside the {_WARM}-bar estimator warmup are non-zero, so an "
        f"entry there is sized off a number that has not converged"
    )
    assert (weights.iloc[_WARM:] > 0.0).all(), (
        "the warmup swallowed the whole series; nothing is left to size with"
    )


def test_a_cold_start_estimate_is_not_merely_noisy_but_biased():
    """Why the prefix is dropped rather than trusted with a wide tolerance.

    An ``ewm(adjust=False)`` variance is seeded from almost nothing and climbs
    towards the truth, so early volatility reads *low* and the weight it implies
    reads *high* -- the direction that puts on the largest positions exactly
    where the estimate is worst.
    """
    returns = _returns(0.01)
    realized = realized_volatility(returns, span=_SPAN, bars_per_year=2190)
    cold = 0.10 / realized.iloc[1:10]
    converged = (0.10 / realized.iloc[_WARM:]).mean()

    assert cold.max() > 2.0 * converged, (
        f"cold-start weight peaks at {cold.max():.3f} against a converged mean "
        f"of {converged:.3f}; if the two agreed there would be nothing to mask"
    )


def test_no_entry_is_sized_off_an_unconverged_estimate(tmp_path):
    """The engine must mask the estimator's warmup, not merely the strategy's.

    ``donchian`` declares 96 bars; the span-96 estimator needs 1,920. Taking the
    strategy's number would leave 1,824 bars of entries sized off a weight that
    is still converging -- and it is the entry *size*, not the entry itself,
    that would be wrong, which no trade count would reveal.
    """
    df = _regime_shift_frame()
    result = _run(tmp_path, df, size_mode=SizeMode.VOL_SCALED_ENTRY, vol_span=_SPAN)

    config = json.loads((result.report_dir / "config.json").read_text())
    assert config["vol_warmup_bars"] == _WARM
    assert config["warmup_bars"] == _WARM, (
        f"the run masked {config['warmup_bars']} bars; the span-{_SPAN} "
        f"estimator needs {_WARM} and is the deeper of the two claims"
    )

    trades = pd.read_csv(result.trades_path, parse_dates=["Entry Timestamp"])
    assert not trades.empty
    assert trades["Entry Timestamp"].min() >= df.index[_WARM], (
        f"first entry at {trades['Entry Timestamp'].min()} precedes the "
        f"estimator warmup ending {df.index[_WARM]}"
    )


def test_a_frame_too_short_for_the_estimator_is_refused(tmp_path):
    """Under-trading silently is the failure this replaces.

    Every weight on such a frame is zero, so every entry is filled at size zero
    and the run reports a flat curve and an empty trade list -- indistinguishable
    from a strategy that found no setups.
    """
    df = _regime_shift_frame(n=_WARM)

    with pytest.raises(ValueError, match="warmup") as raised:
        _run(tmp_path, df, size_mode=SizeMode.VOL_SCALED_ENTRY, vol_span=_SPAN)
    message = str(raised.value)
    assert "volatility estimator" in message and str(_WARM) in message, (
        f"error blames the strategy rather than the estimator that bound: {message}"
    )

    # The same frame is fine under fixed sizing, so the refusal is about the
    # estimator and not about the frame being short in general.
    assert not pd.read_csv(_run(tmp_path, df).trades_path).empty
