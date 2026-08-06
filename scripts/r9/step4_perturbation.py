"""R9 part 3: one step along each axis of the R5 winner, inside the grid.

Training figures are read from the 54-cell surface already computed -- every
neighbour is itself a declared cell -- so only the test-half runs are new.
"""

from __future__ import annotations

import json
from dataclasses import replace

import r9lib as R

OUT = R.OUT
OUT.mkdir(parents=True, exist_ok=True)
WINNER = R.gate.TRAINED_MACHINE


def neighbours() -> list[tuple[str, str, object]]:
    """Every one-step move from the winner that stays inside ``DECLARED_GRID``."""
    out = []
    for axis, values in R.gate.DECLARED_GRID.items():
        ordered = sorted(values)
        current = getattr(WINNER, axis)
        position = min(range(len(ordered)), key=lambda i: abs(ordered[i] - current))
        assert abs(ordered[position] - current) < 1e-12, f"{axis}: winner is off-grid"
        for step, direction in ((-1, "down"), (+1, "up")):
            index = position + step
            if 0 <= index < len(ordered):
                out.append((axis, direction, ordered[index]))
    return out


def main() -> None:
    frame = R.load_frame()
    df, _ = frame
    split = int(R.gate.TRAIN_FRACTION * len(df))
    surface = {row["cell"]: row for row in json.loads((OUT / "training_surface.json").read_text())}

    rows = []
    for axis, direction, value in [("-", "-", None), *neighbours()]:
        config = WINNER if value is None else replace(WINNER, **{axis: value})
        strategy = R.gate.machine(config)
        train = surface[R.cell_key(config)]
        test = R.run(
            strategy,
            frame,
            OUT / "perturb" / f"{axis}_{direction}_{value}",
            first_tradeable=split,
            stop=len(df),
        )
        row = {
            "axis": axis,
            "direction": direction,
            "value": value,
            "label": R.describe(config),
            "warmup_bars": strategy.warmup_bars,
            "train_sharpe_tradeable": train["sharpe_tradeable"],
            "train_sharpe_whole_frame": train["sharpe_whole_frame"],
            "train_net_pct": train["net_return_pct"],
            "train_trades": train["trades"],
            "test_sharpe_tradeable": test["sharpe_tradeable"],
            "test_sharpe_whole_frame": test["sharpe_whole_frame"],
            "test_net_pct": test["net_return_pct"],
            "test_max_dd_pct": test["max_drawdown_pct"],
            "test_trades": test["trades"],
            "test_net_3x": test["net_by_stress"][3.0],
        }
        rows.append(row)
        print(
            f"{axis:>15} {direction:<4} -> {str(value):<8} | train {row['train_sharpe_whole_frame']:+.4f}"
            f" / {row['train_net_pct']:+.2f}%  | test {row['test_sharpe_whole_frame']:+.4f}"
            f" / {row['test_net_pct']:+.2f}% / dd {row['test_max_dd_pct']:.2f}%"
            f" / {row['test_trades']} trades / 3x {row['test_net_3x']:+.2f}%"
        )

    (OUT / "perturbation.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
