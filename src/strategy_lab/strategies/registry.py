from __future__ import annotations

from strategy_lab.strategies.donchian import Donchian
from strategy_lab.strategies.ema_cross import EmaCross
from strategy_lab.strategies.multi_horizon import MultiHorizon
from strategy_lab.strategies.state_machine_v1 import StateMachineV1
from strategy_lab.strategies.trend_following_deepseek_v4 import TrendFollowingDeepseekV4
from strategy_lab.strategies.trend_rider_v1_deepseek_v4_pro import TrendRiderV1DeepseekV4Pro
from strategy_lab.strategies.tsmom import Tsmom
from strategy_lab.strategies.turnaround_v1 import TurnaroundV1
from strategy_lab.strategies.turnaround_v2 import TurnaroundV2


def list_strategies() -> list[str]:
    return [
        "turnaround_v1",
        "turnaround_v2",
        "trend_following_deepseek_v4",
        "trend_rider_v1_deepseek_v4_pro",
        "tsmom",
        "ema_cross",
        "donchian",
        "multi_horizon",
        "state_machine_v1",
    ]


def get_strategy(name: str, *, allow_shorts: bool = True):
    if name == "turnaround_v1":
        return TurnaroundV1(allow_shorts=allow_shorts)
    if name == "turnaround_v2":
        return TurnaroundV2(allow_shorts=allow_shorts)
    if name == "trend_following_deepseek_v4":
        return TrendFollowingDeepseekV4()
    if name == "trend_rider_v1_deepseek_v4_pro":
        return TrendRiderV1DeepseekV4Pro(allow_shorts=allow_shorts)
    if name == "tsmom":
        return Tsmom(allow_shorts=allow_shorts)
    if name == "ema_cross":
        return EmaCross(allow_shorts=allow_shorts)
    if name == "donchian":
        return Donchian(allow_shorts=allow_shorts)
    if name == "multi_horizon":
        return MultiHorizon(allow_shorts=allow_shorts)
    if name == "state_machine_v1":
        return StateMachineV1(allow_shorts=allow_shorts)
    raise ValueError(f"Unknown strategy {name!r}. Available: {', '.join(list_strategies())}")
