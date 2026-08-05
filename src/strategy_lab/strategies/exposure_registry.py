"""The registry for continuous-exposure strategies, separate on purpose.

Manual in two places -- ``list_exposure_strategies`` and
``get_exposure_strategy`` -- exactly as ``strategies/registry.py`` and
``features/registry.py`` are, and for the same reason: naming a strategy here is
what enrols it in ``tests/test_exposure_lookahead.py``, in
``tests/test_exposure_determinism.py``, and in the cold-start warmup check in
``tests/test_strategy_metadata.py``. A strategy that runs but is not listed is a
strategy no safety suite covers.

**It is a second registry rather than a second entry in the first one**, and
that is the load-bearing decision. Six parametrized tests across
``tests/test_lookahead.py``, ``tests/test_replay_determinism.py`` and
``tests/test_strategy_metadata.py`` iterate
``strategies.registry.list_strategies()``, and every one of them calls
``generate_signals``. An exposure strategy has no such method, so listing it
there errors at best -- and at worst is skipped by a ``getattr(..., None)``
guard somewhere and reads as a passing gate, which is the failure mode this
phase is most exposed to. The boolean and continuous contracts are siblings, so
their registries are siblings too.
"""

from __future__ import annotations

from strategy_lab.strategies.exposure import ExposureStrategy
from strategy_lab.strategies.state_machine_v2 import StateMachineV2


def list_exposure_strategies() -> list[str]:
    return ["state_machine_v2"]


def get_exposure_strategy(name: str, *, allow_shorts: bool = True) -> ExposureStrategy:
    if name == "state_machine_v2":
        return StateMachineV2(allow_shorts=allow_shorts)
    raise ValueError(
        f"Unknown exposure strategy {name!r}. Available: "
        f"{', '.join(list_exposure_strategies())}"
    )


__all__ = ["get_exposure_strategy", "list_exposure_strategies"]
