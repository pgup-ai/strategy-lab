from __future__ import annotations

import json

import pandas as pd

from strategy_lab.backtests.report import _format_duration, render_report_html


def _payload(html: str) -> dict:
    marker = '<script id="payload" type="application/json">'
    start = html.index(marker) + len(marker)
    end = html.index("</script>", start)
    return json.loads(html[start:end])


def _candles() -> pd.DataFrame:
    index = pd.date_range("2024-01-03", periods=6, freq="7D", tz="UTC")
    closes = [100.0, 103.0, 101.0, 106.0, 108.0, 104.0]
    return pd.DataFrame(
        {
            "open": [99.0, 100.0, 103.0, 101.0, 106.0, 108.0],
            "high": [c + 2 for c in closes],
            "low": [c - 2 for c in closes],
            "close": closes,
            "volume": [1_000.0, 1_200.0, 900.0, 1_500.0, 1_100.0, 1_300.0],
        },
        index=index,
    )


_TRADE_COLUMNS = [
    "Exit Trade Id",
    "Column",
    "Size",
    "Entry Timestamp",
    "Avg Entry Price",
    "Entry Fees",
    "Exit Timestamp",
    "Avg Exit Price",
    "Exit Fees",
    "PnL",
    "Return",
    "Direction",
    "Status",
    "Position Id",
]


def _trades(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Exit Trade Id": 0,
                "Column": 0,
                "Size": 95.0,
                "Entry Timestamp": index[1],
                "Avg Entry Price": 103.0,
                "Entry Fees": 4.9,
                "Exit Timestamp": index[4],
                "Avg Exit Price": 108.0,
                "Exit Fees": 5.1,
                "PnL": 465.0,
                "Return": 0.0475,
                "Direction": "Long",
                "Status": "Closed",
                "Position Id": 0,
            }
        ],
        columns=_TRADE_COLUMNS,
    )


def _config() -> dict:
    return {
        "identity": {
            "exchange": "yahoo",
            "market_type": "equity",
            "symbol": "SPY",
            "timeframe": "1w",
        },
        "strategy": "trend_following_deepseek_v4",
        "exit_mode": "trend_structure",
        "cash": 10_000.0,
    }


def _stats() -> dict:
    return {
        "Total Return [%]": 4.65,
        "Benchmark Return [%]": 4.0,
        "Win Rate [%]": 100.0,
        "Max Drawdown [%]": 2.1,
        "Sharpe Ratio": 1.4,
        "Profit Factor": float("inf"),
        "Total Trades": 1,
    }


def test_render_report_embeds_library_candles_and_trade_markers() -> None:
    df = _candles()
    equity = pd.Series(10_000.0, index=df.index)

    html = render_report_html(
        df=df,
        trades=_trades(df.index),
        equity=equity,
        config=_config(),
        stats=_stats(),
    )

    assert "TradingView Lightweight Charts" in html
    assert "SPY" in html
    payload = _payload(html)
    assert len(payload["candles"]) == len(df)
    assert payload["identity"] == {
        "exchange": "yahoo",
        "market_type": "equity",
        "symbol": "SPY",
        "timeframe": "1w",
    }
    assert set(payload["colors"]) == {"upDim", "downDim"}
    shapes = [m["shape"] for m in payload["markers"]]
    assert shapes == ["arrowUp", "arrowDown"]
    assert payload["markers"][0]["time"] == int(df.index[1].timestamp())
    assert payload["markers"][0]["text"] == "B 103.00"
    assert payload["markers"][1]["text"] == "S 108.00"
    assert "103.00" in html
    assert "108.00" in html


def test_render_report_without_trades_still_renders() -> None:
    df = _candles()
    equity = pd.Series(10_000.0, index=df.index)

    html = render_report_html(
        df=df,
        trades=pd.DataFrame(columns=_TRADE_COLUMNS),
        equity=equity,
        config=_config(),
        stats=_stats(),
    )

    assert "TradingView Lightweight Charts" in html
    assert _payload(html)["markers"] == []
    assert "No trades" in html


def test_render_report_marks_open_trades() -> None:
    df = _candles()
    equity = pd.Series(10_000.0, index=df.index)
    trades = _trades(df.index)
    trades.loc[0, "Status"] = "Open"

    html = render_report_html(
        df=df,
        trades=trades,
        equity=equity,
        config=_config(),
        stats=_stats(),
    )

    assert "Open" in html
    payload = _payload(html)
    assert [m["shape"] for m in payload["markers"]] == ["arrowUp"]
    assert payload["zoom"][0]["from"] == payload["zoom"][0]["to"]


def test_format_duration() -> None:
    assert _format_duration(pd.Timedelta(days=448)) == "448d"
    assert _format_duration(pd.Timedelta(days=2, hours=3)) == "2d 3h"
    assert _format_duration(pd.Timedelta(hours=5, minutes=30)) == "5h 30m"
    assert _format_duration(pd.Timedelta(minutes=45)) == "45m"
