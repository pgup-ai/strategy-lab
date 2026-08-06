"""R9 part 1 inputs: the 54-cell training surface, scored both ways."""

from __future__ import annotations

import json
import time

import numpy as np
import r9lib as R

OUT = R.OUT
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    started = time.time()
    frame = R.load_frame()
    df, _ = frame
    split = int(R.gate.TRAIN_FRACTION * len(df))

    cells = R.gate.declared_machines()
    strategies = [R.gate.machine(config) for config in cells]
    first_tradeable = max(strategy.warmup_bars for strategy in strategies)
    print(f"{len(cells)} cells; deepest warmup {first_tradeable}; "
          f"warmups {min(s.warmup_bars for s in strategies)}..{first_tradeable}")
    print(f"training: bars {first_tradeable}..{split} = {split - first_tradeable} tradeable, "
          f"{df.index[first_tradeable]} -> {df.index[split - 1]}")

    rows = []
    returns = {}
    for position, (config, strategy) in enumerate(zip(cells, strategies)):
        row = R.run(
            strategy,
            frame,
            OUT / "surface" / f"cell{position:02d}",
            first_tradeable=first_tradeable,
            stop=split,
        )
        row["cell"] = R.cell_key(config)
        row["label"] = R.describe(config)
        row["enter_strength"] = config.enter_strength
        row["exit_strength"] = config.exit_strength
        row["min_dwell"] = config.min_dwell
        row["cooldown"] = config.cooldown
        returns[row["cell"]] = row["returns"].to_numpy(dtype="float64")
        rows.append(R.slim(row))
        print(f"  [{position:02d}] {row['label']}  whole {row['sharpe_whole_frame']:+.4f}  "
              f"tradeable {row['sharpe_tradeable']:+.4f}  net {row['net_return_pct']:+.2f}%  "
              f"trades {row['trades']}")

    (OUT / "training_surface.json").write_text(json.dumps(rows, indent=2, default=str))
    np.savez_compressed(OUT / "training_returns.npz", **returns)

    for rule in ("sharpe_whole_frame", "sharpe_tradeable"):
        best = max(rows, key=lambda row: row[rule])
        print(f"winner by {rule}: {best['label']} at {best[rule]:+.4f}")
    print(f"elapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
