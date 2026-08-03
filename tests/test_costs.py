from __future__ import annotations

import pandas as pd
import pytest

from strategy_lab.backtests.costs import CostModel, apply_funding, funding_ledger


def _positions(values, start="2024-01-01", freq="4h"):
    index = pd.date_range(start, periods=len(values), freq=freq, tz="UTC", name="timestamp")
    return pd.Series(values, index=index, dtype="float64")


def _funding(rates, start="2024-01-01", freq="8h"):
    index = pd.date_range(start, periods=len(rates), freq=freq, tz="UTC", name="timestamp")
    return pd.Series(rates, index=index, dtype="float64")


def _funding_at(stamps, rate=0.0001):
    index = pd.DatetimeIndex([pd.Timestamp(s) for s in stamps], tz="UTC", name="timestamp")
    return pd.Series([rate] * len(stamps), index=index, dtype="float64")


def test_a_long_pays_funding_when_the_rate_is_positive():
    cost = apply_funding(positions=_positions([1.0] * 6), funding=_funding([0.0001] * 3))
    assert cost.sum() < 0


def test_a_short_receives_funding_when_the_rate_is_positive():
    cost = apply_funding(positions=_positions([-1.0] * 6), funding=_funding([0.0001] * 3))
    assert cost.sum() > 0


def test_funding_is_charged_only_at_settlement_bars():
    """8h funding against 4h bars must hit 3 of 6 bars, not smear across all six."""
    cost = apply_funding(positions=_positions([1.0] * 6), funding=_funding([0.0001] * 3))
    assert (cost != 0).sum() == 3


def test_a_flat_position_pays_no_funding():
    cost = apply_funding(positions=_positions([0.0] * 6), funding=_funding([0.0001] * 3))
    assert cost.abs().sum() == 0


def test_funding_scales_with_position_size():
    small = apply_funding(positions=_positions([0.5] * 6), funding=_funding([0.0001] * 3))
    full = apply_funding(positions=_positions([1.0] * 6), funding=_funding([0.0001] * 3))
    assert full.sum() == pytest.approx(2 * small.sum())


def test_cost_stress_multiplies_fees_and_slippage_but_not_funding():
    """Funding is a market rate, not an execution cost -- stressing it would model a different market."""
    base = CostModel(fee=0.0004, slippage=0.0005)
    stressed = base.stressed(3.0)
    assert stressed.fee == pytest.approx(0.0012)
    assert stressed.slippage == pytest.approx(0.0015)


def test_an_off_grid_settlement_lands_on_the_bar_that_contains_it():
    """The measured trap: Binance stamps settlements up to 47ms *after* the boundary.

    3,260 of BTC's 7,559 stored settlements are off-grid. Matching them to bars
    by equality against a generated 8h ``date_range`` silently drops 43% of them
    and halves the measured drag.
    """
    on_grid = apply_funding(positions=_positions([1.0] * 6), funding=_funding([0.0001] * 3))
    off_grid = apply_funding(
        positions=_positions([1.0] * 6),
        funding=_funding_at(
            [
                "2024-01-01 00:00:00.047",
                "2024-01-01 08:00:00.013",
                "2024-01-01 16:00:00.001",
            ]
        ),
    )
    assert (off_grid != 0).sum() == 3
    pd.testing.assert_series_equal(off_grid, on_grid)


def test_a_settlement_mid_bar_is_charged_to_the_bar_it_falls_inside():
    """Containment, not nearest-boundary: 09:59 belongs to the 08:00 bar, not the 12:00 one."""
    cost = apply_funding(
        positions=_positions([1.0] * 6), funding=_funding_at(["2024-01-01 09:59:00"])
    )
    charged = cost[cost != 0]
    assert list(charged.index) == [pd.Timestamp("2024-01-01 08:00", tz="UTC")]


def test_several_settlements_inside_one_bar_all_get_charged():
    """8h funding under daily bars settles three times a day; none may be dropped."""
    daily = _positions([1.0] * 2, freq="1D")
    one = apply_funding(positions=daily, funding=_funding_at(["2024-01-01 00:00"]))
    three = apply_funding(
        positions=daily,
        funding=_funding_at(["2024-01-01 00:00", "2024-01-01 08:00", "2024-01-01 16:00"]),
    )
    assert three.sum() == pytest.approx(3 * one.sum())
    assert (three != 0).sum() == 1


def test_a_non_eight_hour_settlement_interval_is_handled():
    """Both stored contracts settle 8-hourly, but that is observed, not guaranteed."""
    cost = apply_funding(
        positions=_positions([1.0] * 6),
        funding=_funding([0.0001] * 12, freq="2h"),
    )
    assert (cost != 0).sum() == 6
    assert cost.sum() == pytest.approx(-12 * 0.0001)


def test_settlements_outside_the_position_window_are_not_charged():
    """Funding loaded over a wider window than the candles must not be charged to the edges."""
    cost = apply_funding(
        positions=_positions([1.0] * 6),
        funding=_funding_at(["2023-12-31 20:00", "2024-01-02 04:00"]),
    )
    assert cost.abs().sum() == 0


def test_the_bar_after_the_last_one_is_still_covered():
    """The final bar spans an interval too -- a settlement inside it is real, not out of range."""
    cost = apply_funding(
        positions=_positions([1.0] * 6), funding=_funding_at(["2024-01-01 21:00"])
    )
    charged = cost[cost != 0]
    assert list(charged.index) == [pd.Timestamp("2024-01-01 20:00", tz="UTC")]


def test_unsorted_positions_are_rejected_rather_than_misassigned():
    """Containment is a binary search; an unsorted index would silently mis-slot every rate."""
    shuffled = _positions([1.0] * 6).iloc[[3, 0, 1, 2, 4, 5]]
    with pytest.raises(ValueError, match="sorted"):
        apply_funding(positions=shuffled, funding=_funding([0.0001] * 3))


def test_missing_rates_are_skipped_rather_than_poisoning_the_series():
    cost = apply_funding(
        positions=_positions([1.0] * 6),
        funding=pd.Series(
            [0.0001, float("nan"), 0.0001],
            index=_funding([0.0] * 3).index,
            dtype="float64",
        ),
    )
    assert cost.notna().all()
    assert (cost != 0).sum() == 2


def test_the_ledger_records_the_bar_each_settlement_landed_on():
    ledger = funding_ledger(
        positions=_positions([1.0] * 6),
        funding=_funding_at(["2024-01-01 00:00:00.047", "2024-01-01 09:59:00"]),
    )
    assert list(ledger["bar"]) == [
        pd.Timestamp("2024-01-01 00:00", tz="UTC"),
        pd.Timestamp("2024-01-01 08:00", tz="UTC"),
    ]
    assert list(ledger.index) == [
        pd.Timestamp("2024-01-01 00:00:00.047", tz="UTC"),
        pd.Timestamp("2024-01-01 09:59:00", tz="UTC"),
    ]


def test_the_ledger_totals_to_the_applied_series():
    """One containment, two views -- the audit trail cannot drift from the charge."""
    positions = _positions([1.0, 0.5, -2.0, 0.0, 1.0, 1.0])
    funding = _funding([0.0001, -0.0002, 0.0003] * 2, freq="4h")
    ledger = funding_ledger(positions=positions, funding=funding)
    applied = apply_funding(positions=positions, funding=funding)
    assert ledger["cash_flow"].sum() == pytest.approx(applied.sum())
    assert len(ledger) == 6

