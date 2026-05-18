from __future__ import annotations

from strategy_lab.strategies.turnaround_v1 import TurnaroundV1
from strategy_lab.strategies.turnaround_v2 import TurnaroundV2


def list_strategies() -> list[str]:
    return ["turnaround_v1", "turnaround_v2"]


def get_strategy(name: str):
    if name == "turnaround_v1":
        return TurnaroundV1()
    if name == "turnaround_v2":
        return TurnaroundV2()
    raise ValueError(f"Unknown strategy {name!r}. Available: {', '.join(list_strategies())}")
