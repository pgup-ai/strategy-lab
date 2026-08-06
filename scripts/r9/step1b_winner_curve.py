"""The winning cell's *whole-window* training return series.

``r9lib.run`` keeps only the tradeable slice, which is the series M21's scalar
scores. R5's scalar scores the whole window -- warmup zeros included -- so the
deflation under R5's own rule needs that series too, skew, kurtosis and all.
"""

from __future__ import annotations


import numpy as np
import r9lib as R

OUT = R.OUT
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    frame = R.load_frame()
    df, _ = frame
    split = int(R.gate.TRAIN_FRACTION * len(df))
    strategies = [R.gate.machine(config) for config in R.gate.declared_machines()]
    first_tradeable = max(strategy.warmup_bars for strategy in strategies)

    row = R.run(
        R.gate.machine(R.gate.TRAINED_MACHINE),
        frame,
        OUT / "winner_train",
        first_tradeable=first_tradeable,
        stop=split,
    )
    whole = R.returns_of(row["equity"]).to_numpy(dtype="float64")
    np.save(OUT / "winner_whole_window_returns.npy", whole)
    print(f"whole window {len(whole)} bars, warmup {row['warmup_bars']}, "
          f"zeros {(whole == 0.0).sum()}")
    print(f"whole-frame sharpe {row['sharpe_whole_frame']:+.4f}  "
          f"tradeable {row['sharpe_tradeable']:+.4f}  net {row['net_return_pct']:+.2f}%  "
          f"maxDD {row['max_drawdown_pct']:.2f}%  trades {row['trades']}")


if __name__ == "__main__":
    main()
