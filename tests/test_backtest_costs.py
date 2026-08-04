from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from conftest import synthetic_ohlcv
from strategy_lab.backtests.costs import CostModel
from strategy_lab.backtests.engine import ExitMode, _funding_notional, run_backtest
from strategy_lab.market_data.base import MarketDataIdentity
from strategy_lab.strategies import get_strategy

_IDENTITY = MarketDataIdentity(
    exchange="binance", market_type="perp", symbol="BTC/USDT", timeframe="4h"
)


def _frame() -> pd.DataFrame:
    return synthetic_ohlcv(n=600, seed=11, freq="4h")


def _funding(df: pd.DataFrame, rate: float = 0.0001, offset_ms: int = 0) -> pd.Series:
    """8h settlements over the frame's span, optionally stamped late like the venue's."""
    index = pd.date_range(df.index[0], df.index[-1], freq="8h", tz="UTC", name="timestamp")
    index = index + pd.Timedelta(offset_ms, unit="ms")
    return pd.Series(rate, index=index, dtype="float64")


def _run(tmp_path: Path, name: str = "donchian", **kwargs):
    return run_backtest(
        df=_frame(),
        strategy=get_strategy(name),
        identity=_IDENTITY,
        exit_mode=ExitMode.OPPOSITE_SIGNAL_ONLY,
        report_root=tmp_path / name,
        **kwargs,
    )


def _stats(result) -> dict:
    return json.loads(result.stats_path.read_text())


def _config(result) -> dict:
    return json.loads((result.report_dir / "config.json").read_text())


def _stress(result) -> dict[float, dict]:
    breakdown = json.loads(result.costs_path.read_text())["stress"]
    return {row["multiple"]: row for row in breakdown}


def _equity(result) -> pd.Series:
    frame = pd.read_csv(result.equity_curve_path, index_col=0)
    return frame["equity"]


def test_the_equity_curve_on_disk_is_the_net_one(tmp_path):
    """Funding never enters the simulation, so the whole gap between the two
    runs' curves is the funding charged -- exactly, not merely in the right
    direction. ``equity_curve.csv`` is what a reader plots and compares."""
    without = _run(tmp_path / "a")
    with_funding = _run(tmp_path / "b", funding=_funding(_frame()))
    gap = _equity(without).to_numpy() - _equity(with_funding).to_numpy()
    assert gap[-1] == pytest.approx(_stats(with_funding)["Funding Paid"])
    assert gap[-1] > 0


def _gap_is_exactly_the_funding(result) -> bool:
    """Net differs from vectorbt's fee-and-slippage total by the funding, exactly.

    Asserting only ``net < gross`` would pass on floating-point noise if the
    funding term were dropped from the equity entirely, which is the whole bug
    this file exists to prevent.
    """
    stats, cash = _stats(result), _config(result)["cash"]
    gross = stats["Total Return [%] (gross of funding)"]
    return gross - stats["Net Return [%]"] == pytest.approx(
        stats["Funding Paid"] / cash * 100, rel=1e-9
    )


def test_a_persistently_long_book_pays_funding_when_rates_are_positive(tmp_path):
    result = _run(tmp_path, funding=_funding(_frame()))
    assert _stats(result)["Funding Paid"] > 0
    assert _gap_is_exactly_the_funding(result)


def test_a_negative_rate_pays_the_long_rather_than_charging_it(tmp_path):
    result = _run(tmp_path, funding=_funding(_frame(), rate=-0.0001))
    assert _stats(result)["Funding Paid"] < 0
    assert _gap_is_exactly_the_funding(result)


def test_off_grid_settlements_cost_the_same_as_on_grid_ones(tmp_path):
    """The venue stamps settlements up to 47ms late; an equality join would drop 43% of them."""
    on_grid = _run(tmp_path / "a", funding=_funding(_frame()))
    off_grid = _run(tmp_path / "b", funding=_funding(_frame(), offset_ms=47))
    assert _stats(off_grid)["Funding Paid"] == pytest.approx(_stats(on_grid)["Funding Paid"])
    assert _stats(off_grid)["Funding Paid"] > 0


def test_config_records_the_cost_model_and_whether_funding_was_applied(tmp_path):
    with_funding = _config(_run(tmp_path / "a", funding=_funding(_frame())))
    assert with_funding["funding_applied"] is True
    assert with_funding["funding_settlements"] > 0
    assert with_funding["cost_model"] == {"fee": 0.0005, "slippage": 0.0005}

    without = _config(_run(tmp_path / "b"))
    assert without["funding_applied"] is False
    assert without["funding_settlements"] == 0


def test_a_three_times_cost_run_returns_less_than_a_one_times_run(tmp_path):
    stress = _stress(_run(tmp_path, cost_stress=(1.0, 2.0, 3.0)))
    assert stress[3.0]["net_return_pct"] < stress[2.0]["net_return_pct"]
    assert stress[2.0]["net_return_pct"] < stress[1.0]["net_return_pct"]
    assert stress[3.0]["fees_paid"] == pytest.approx(3 * stress[1.0]["fees_paid"], rel=0.02)


def test_cost_stress_scales_execution_but_never_funding(tmp_path):
    """Charter M8: stressing funding models a different market, not a worse fill.

    Deployment is held to 50% so the book is not cash-constrained and comes out
    bit-identical at both stress levels. The funding charge is then exactly
    equal rather than merely close, which is what makes this assertion sharp.
    """
    stress = _stress(
        _run(
            tmp_path,
            funding=_funding(_frame()),
            cost_stress=(1.0, 3.0),
            position_pct=0.5,
        )
    )
    assert stress[3.0]["funding_paid"] == stress[1.0]["funding_paid"]
    assert stress[1.0]["funding_paid"] > 0
    assert stress[1.0]["slippage_paid"] > 0
    assert stress[3.0]["slippage_paid"] == pytest.approx(3 * stress[1.0]["slippage_paid"])
    # Equal fee and slippage rates buy at almost the same price, so vectorbt's
    # reported fees are the yardstick for the slippage this engine backs out of
    # the fill prices itself.
    assert stress[1.0]["slippage_paid"] == pytest.approx(stress[1.0]["fees_paid"], rel=0.01)


def test_worse_fills_shrink_a_cash_constrained_book(tmp_path):
    """At 95% deployment there is not enough cash left to fill the same size.

    That -- not a scaled rate -- is why the funding column moves at all across
    the stress table. It moves by about a percent; a scaled rate would treble.
    """
    stress = _stress(
        _run(tmp_path, funding=_funding(_frame()), cost_stress=(1.0, 3.0), position_pct=0.95)
    )
    assert stress[3.0]["funding_paid"] < stress[1.0]["funding_paid"]
    assert stress[3.0]["funding_paid"] > 0.95 * stress[1.0]["funding_paid"]


def test_the_stress_table_agrees_with_the_headline_run(tmp_path):
    result = _run(tmp_path, funding=_funding(_frame()), cost_stress=(1.0, 2.0))
    stats, base = _stats(result), _stress(result)[1.0]
    assert base["net_return_pct"] == pytest.approx(stats["Net Return [%]"])
    assert base["funding_paid"] == pytest.approx(stats["Funding Paid"])
    assert _config(result)["cost_stress"] == [1.0, 2.0]


def test_reported_funding_matches_the_settlement_arithmetic(tmp_path):
    """Reconcile against a hand computation: notional held into each settlement bar x rate."""
    result = _run(tmp_path, funding=_funding(_frame()))
    ledger = pd.read_csv(result.funding_path, index_col=0)
    expected = (ledger["notional"] * ledger["rate"]).sum()
    assert _stats(result)["Funding Paid"] == pytest.approx(expected)
    assert ledger["cash_flow"].sum() == pytest.approx(-expected)


def test_the_base_run_is_in_the_table_even_when_not_asked_for(tmp_path):
    """A 3x row with nothing to be stressed against tells a reader nothing."""
    stress = _stress(_run(tmp_path, cost_stress=(3.0,)))
    assert sorted(stress) == [1.0, 3.0]


def test_the_waterfall_closes_on_a_simulated_gross(tmp_path):
    """Gross is a costless re-simulation, so the gap to net is the three costs
    plus the P&L the shrunken book never earned. Naming that remainder is what
    keeps the waterfall an identity."""
    result = _run(tmp_path, funding=_funding(_frame()))
    row, cash = _stress(result)[1.0], _config(result)["cash"]
    deducted = (
        row["fees_paid"] + row["slippage_paid"] + row["funding_paid"] + row["size_effect"]
    )
    assert row["gross_return_pct"] - row["net_return_pct"] == pytest.approx(
        deducted / cash * 100
    )
    assert row["fees_paid"] + row["slippage_paid"] + row["funding_paid"] > 0


def test_gross_is_simulated_rather_than_added_back(tmp_path):
    """Adding the costs back onto net cannot recover a costless book: at 95%
    deployment worse fills buy less, so the cost-bearing run holds a smaller
    position and earns less than the fee it paid. That gap is `size_effect`,
    and it is zero only when there is cash to spare."""
    constrained = _stress(_run(tmp_path / "a", position_pct=0.95))[1.0]
    roomy = _stress(_run(tmp_path / "b", position_pct=0.4))[1.0]
    assert abs(constrained["size_effect"]) > 1.0
    assert roomy["size_effect"] == pytest.approx(0.0, abs=1e-6)


def test_stressing_costs_does_not_move_the_gross_figure(tmp_path):
    """One costless simulation serves every row -- scaling a zero rate changes
    nothing -- so a moving gross column would mean gross was reconstructed from
    the cost-bearing book again."""
    stress = _stress(_run(tmp_path, cost_stress=(1.0, 3.0)))
    assert stress[3.0]["gross_return_pct"] == stress[1.0]["gross_return_pct"]
    assert stress[3.0]["net_return_pct"] < stress[1.0]["net_return_pct"]


def test_funding_is_charged_on_the_position_held_into_the_bar():
    """Fills land at a bar's close, so the book a settlement at bar *t*'s open
    meets is the one established at bar *t-1*. Charging bar *t*'s own position
    would settle funding against a trade made after the settlement happened."""
    df = synthetic_ohlcv(n=4, seed=3, freq="4h")
    entered_at_bar_one = pd.Series([0.0, 2.0, 2.0, 0.0], index=df.index)
    notional = _funding_notional(SimpleNamespace(assets=lambda: entered_at_bar_one), df)

    assert notional.iloc[0] == 0.0
    assert notional.iloc[1] == 0.0
    assert notional.iloc[2] == pytest.approx(2.0 * df["open"].iloc[2])
    assert notional.iloc[3] == pytest.approx(2.0 * df["open"].iloc[3])


def test_a_cost_model_matches_the_legacy_fee_and_slippage_arguments(tmp_path):
    legacy = _run(tmp_path / "a", fees=0.0009, slippage=0.0003)
    modelled = _run(tmp_path / "b", cost_model=CostModel(fee=0.0009, slippage=0.0003))
    assert _stats(modelled) == _stats(legacy)


def test_risk_statistics_are_measured_on_the_curve_that_is_plotted(tmp_path):
    """Funding settles outside the simulation, so ``pf.stats()`` scores a curve
    the report never draws. On a 35%-of-capital funding bill that is the
    difference between a publishable drawdown and a real one."""
    result = _run(tmp_path, funding=_funding(_frame(), rate=0.002))
    stats = _stats(result)
    equity = _equity(result)

    peak = equity.cummax()
    assert stats["Max Drawdown [%] (net of funding)"] == pytest.approx(
        float((1 - equity / peak).max() * 100)
    )
    assert (
        stats["Max Drawdown [%] (net of funding)"]
        > stats["Max Drawdown [%] (gross of funding)"]
    )
    assert stats["Sharpe Ratio (net of funding)"] < stats["Sharpe Ratio (gross of funding)"]


def test_no_path_statistic_is_left_without_a_curve_named(tmp_path):
    """A bare Sharpe on a funded run is the defect: two curves exist and the
    reader has no way to tell which one they are holding."""
    stats = _stats(_run(tmp_path, funding=_funding(_frame())))
    bare = {"Total Return [%]", "Max Drawdown [%]", "Sharpe Ratio", "Sortino Ratio"}
    assert bare.isdisjoint(stats)
    assert "Win Rate [%]" in stats


def test_the_report_shows_both_curves_risk_side_by_side(tmp_path):
    page = _run(tmp_path, funding=_funding(_frame())).plot_path.read_text()
    assert "Gross of funding" in page
    assert "Net of funding" in page
    assert "Sortino" in page
    assert "Max Drawdown (net)" in page


def test_a_run_without_funding_leaves_the_stats_dict_untouched(tmp_path):
    """A funding column on a market that has no funding is noise, and it would
    also break every result of record produced before costs existed."""
    plain = _run(tmp_path / "a")
    with_plumbing = _run(tmp_path / "b", cost_model=CostModel(), cost_stress=(1.0, 3.0))
    assert plain.stats_path.read_bytes() == with_plumbing.stats_path.read_bytes()
    assert "Funding Paid" not in _stats(plain)
    assert "Net Return [%]" not in _stats(plain)
    assert plain.funding_path is None


def test_funding_outside_the_candle_window_is_not_charged(tmp_path):
    df = _frame()
    before = pd.Series(
        0.01,
        index=pd.DatetimeIndex(
            [df.index[0] - pd.Timedelta("8h"), df.index[-1] + pd.Timedelta("8h")],
            tz="UTC",
            name="timestamp",
        ),
    )
    assert _stats(_run(tmp_path, funding=before))["Funding Paid"] == 0.0


def test_the_report_makes_the_funding_drag_visible(tmp_path):
    result = _run(tmp_path, funding=_funding(_frame()), cost_stress=(1.0, 2.0, 3.0))
    page = result.plot_path.read_text()
    assert "Costs" in page
    assert "Funding Paid" in page
    assert "Net Return" in page
    assert "Gross Return" in page
    assert "3x" in page
    assert "http://" not in page.replace("http://www.w3.org", "")


def test_the_headline_chip_is_never_a_gross_number_when_funding_applies(tmp_path):
    """A reader glancing at the top of the page must not read gross as tradeable."""
    funded = _run(tmp_path / "a", funding=_funding(_frame())).plot_path.read_text()
    assert "Gross of Funding" in funded
    net_at = funded.index("Net Return")
    assert net_at < funded.index("<section class=\"costs\">")

    plain = _run(tmp_path / "b").plot_path.read_text()
    assert "Gross of Funding" not in plain
    assert "Total Return" in plain


def test_a_report_without_funding_says_so_rather_than_staying_silent(tmp_path):
    page = _run(tmp_path).plot_path.read_text()
    assert "gross of funding" in page
    assert "not modelled" in page
