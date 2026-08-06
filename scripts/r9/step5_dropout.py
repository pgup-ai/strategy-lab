"""R9 part 4: feature dropout -- each of the four features at its neutral value.

The machine, the policy and the features are untouched. The wrapper below reads
the strategy's *own* feature frame through ``build_feature_frame``, overwrites
one column with that feature's neutral value, and hands the result to the same
``StateMachine.run`` / ``target_risk_series`` pair ``signed_target`` uses -- so
the only difference from a plain run is the one column.

Two validations, both of which must pass before a dropout row is read:

1. the wrapper with **nothing** dropped reproduces the control run exactly;
2. the wrapper dropping ``crowding`` reproduces the run of the *unmodified*
   strategy over a frame with no ``funding_rate`` column at all -- which is the
   real code path M20 measured at +16.44% / +0.801 / 71 trades.

Neutral values, each in the space the machine reads the feature in:
``direction`` 0.0 (signed, -1..1, neutral at zero); ``strength`` and
``stability`` 0.5 (trailing ranks in 0..1, neutral at the median rank);
``crowding`` 0.5 (``state_machine_core.NEUTRAL_CROWDING``, the repo's own
constant for an unmeasurable carry).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
import r9lib as R
from strategy_lab.features.flow import FUNDING_COLUMN
from strategy_lab.state.policy import target_risk_series
from strategy_lab.strategies.base import validate_ohlcv
from strategy_lab.strategies.state_machine_core import build_feature_frame

OUT = R.OUT
OUT.mkdir(parents=True, exist_ok=True)

NEUTRAL = {"direction": 0.0, "strength": 0.5, "stability": 0.5, "crowding": 0.5}


@dataclass(frozen=True)
class Neutralised:
    """``state_machine_v1`` with one feature column held at its neutral value."""

    inner: object
    dropped: str | None = None

    @property
    def name(self) -> str:
        return f"state_machine_v1_no_{self.dropped or 'nothing'}"

    @property
    def version(self) -> str:
        return self.inner.version

    @property
    def warmup_bars(self) -> int:
        return self.inner.warmup_bars

    def generate_signals(self, df: pd.DataFrame):
        validate_ohlcv(df)
        frame, crowding_measured = build_feature_frame(
            df, features=self.inner.features, rank_window=self.inner.rank_window
        )
        if self.dropped is not None:
            column = frame[self.dropped]
            # Warmup rows stay NaN: "not yet measurable" is a different claim
            # from "measured and neutral", and the machine reads the difference.
            frame[self.dropped] = column.where(column.isna(), NEUTRAL[self.dropped])
        states = self.inner.machine.run(frame)
        target = target_risk_series(
            states=states,
            direction=frame["direction"],
            strength=frame["strength"],
            crowding=frame["crowding"],
        )
        if not self.inner.allow_shorts:
            target = target.clip(lower=0.0)
        return self.inner._signals(
            target,
            metadata={
                "allow_shorts": self.inner.allow_shorts,
                "rank_window": self.inner.rank_window,
                "features": list(self.inner.features),
                "crowding_measured": crowding_measured,
                "dropped_feature": self.dropped,
            },
        )


def headline(row: dict) -> tuple:
    return (
        round(row["net_return_pct"], 6),
        round(row["sharpe_whole_frame"], 6),
        round(row["max_drawdown_pct"], 6),
        row["trades"],
    )


def main() -> None:
    frame = R.load_frame()
    df, rates = frame
    split = int(R.gate.TRAIN_FRACTION * len(df))
    strategies = [R.gate.machine(config) for config in R.gate.declared_machines()]
    deepest = max(strategy.warmup_bars for strategy in strategies)
    inner = R.gate.machine(R.gate.TRAINED_MACHINE)

    def test(strategy, tag, source=frame):
        return R.run(strategy, source, OUT / "dropout" / tag, first_tradeable=split, stop=len(df))

    def train(strategy, tag, source=frame):
        return R.run(
            strategy, source, OUT / "dropout" / tag, first_tradeable=deepest, stop=split
        )

    control = test(inner, "control")
    noop = test(Neutralised(inner), "noop")
    assert headline(noop) == headline(control), (
        f"the wrapper is not a no-op: {headline(noop)} vs {headline(control)}"
    )
    print(f"validation 1 OK -- wrapper with nothing dropped == control {headline(control)}")

    stripped = (df.drop(columns=[FUNDING_COLUMN]), rates)
    no_column = test(inner, "no_funding_column", source=stripped)
    dropped_crowding = test(Neutralised(inner, "crowding"), "crowding_check")
    assert headline(no_column) == headline(dropped_crowding), (
        f"crowding dropout {headline(dropped_crowding)} != no-funding-column "
        f"{headline(no_column)}"
    )
    print(
        f"validation 2 OK -- crowding dropout == strategy on a frame with no funding "
        f"column: net {no_column['net_return_pct']:+.2f}% sharpe "
        f"{no_column['sharpe_whole_frame']:+.3f} dd {no_column['max_drawdown_pct']:.2f}% "
        f"trades {no_column['trades']} (M20 published +16.44 / +0.801 / 6.08 / 71)"
    )

    rows = []
    for dropped in [None, *NEUTRAL]:
        strategy = Neutralised(inner, dropped)
        tag = dropped or "none"
        test_row = test(strategy, f"test_{tag}")
        train_row = train(strategy, f"train_{tag}")
        row = {
            "dropped": tag,
            "neutral": NEUTRAL.get(dropped),
            "train_sharpe_tradeable": train_row["sharpe_tradeable"],
            "train_sharpe_whole_frame": train_row["sharpe_whole_frame"],
            "train_net_pct": train_row["net_return_pct"],
            "train_trades": train_row["trades"],
            "test_sharpe_tradeable": test_row["sharpe_tradeable"],
            "test_sharpe_whole_frame": test_row["sharpe_whole_frame"],
            "test_net_pct": test_row["net_return_pct"],
            "test_max_dd_pct": test_row["max_drawdown_pct"],
            "test_trades": test_row["trades"],
            "test_net_3x": test_row["net_by_stress"][3.0],
        }
        rows.append(row)

        def show(value, spec="+.4f"):
            return "  none  " if value is None else format(value, spec)

        print(
            f"drop {tag:<10} | train {show(row['train_sharpe_whole_frame'])} / "
            f"{show(row['train_net_pct'], '+.2f')}% / {row['train_trades']:>3} trades | "
            f"test {show(row['test_sharpe_whole_frame'])} / "
            f"{show(row['test_net_pct'], '+.2f')}% / "
            f"dd {show(row['test_max_dd_pct'], '.2f')}% / {row['test_trades']:>3} trades / "
            f"3x {show(row['test_net_3x'], '+.2f')}%"
        )

    (OUT / "dropout.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
