from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def synthetic_ohlcv(n: int = 400, seed: int = 7, freq: str = "15min") -> pd.DataFrame:
    """Deterministic random-walk OHLCV with valid high/low ordering."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.003, n)))
    index = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(100, 1000, n),
        },
        index=index,
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return synthetic_ohlcv()


def _postgres_reachable() -> bool:
    import socket
    from urllib.parse import urlparse

    from strategy_lab.config import settings

    parsed = urlparse(settings.database_url.replace("postgresql+psycopg", "postgresql"))
    sock = socket.socket()
    sock.settimeout(1.5)
    try:
        sock.connect((parsed.hostname or "localhost", parsed.port or 5432))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def pytest_collection_modifyitems(config, items):
    if _postgres_reachable():
        return
    skip_db = pytest.mark.skip(
        reason="Postgres not reachable; start it with docker compose up -d postgres"
    )
    for item in items:
        if "db" in item.keywords:
            item.add_marker(skip_db)
