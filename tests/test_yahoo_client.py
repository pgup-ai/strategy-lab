from __future__ import annotations

from strategy_lab.market_data.yahoo import _yahoo_interval


def test_yahoo_interval_maps_weekly_alias() -> None:
    assert _yahoo_interval("1w") == "1wk"


def test_yahoo_interval_passes_native_intervals_through() -> None:
    assert _yahoo_interval("1wk") == "1wk"
    assert _yahoo_interval("1d") == "1d"
    assert _yahoo_interval("1h") == "1h"
