"""Step 0: reproduce R5's published row before any R9 number is read."""

from __future__ import annotations

import json
import time

import r9lib as R

OUT = R.OUT
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    started = time.time()
    frame = R.load_frame()
    df, rates = frame
    print(f"frame: {len(df)} bars {df.index[0]} -> {df.index[-1]}")
    print(f"funding: {len(rates)} settlements {rates.index[0]} -> {rates.index[-1]}")
    print(f"funding column present: {'funding_rate' in df.columns}")

    split = int(R.gate.TRAIN_FRACTION * len(df))
    print(f"split index {split} at {df.index[split]}, test bars {len(df) - split}")

    strategy = R.gate.machine(R.gate.TRAINED_MACHINE)
    print(f"trained cell: {R.describe(R.gate.TRAINED_MACHINE)} warmup={strategy.warmup_bars}")

    row = R.run(
        strategy,
        frame,
        OUT / "control",
        first_tradeable=split,
        stop=len(df),
        keep=True,
    )
    print(json.dumps(R.slim(row), indent=2, default=str))
    print(f"crowding_measured: {row['config']['strategy_metadata']['crowding_measured']}")
    print(
        f"CONTROL  net {row['net_return_pct']:+.2f}%  sharpe {row['sharpe_whole_frame']:+.3f}  "
        f"maxDD {row['max_drawdown_pct']:.2f}%  trades {row['trades']}"
    )
    print(f"M21 scalar on the same run: sharpe(tradeable) {row['sharpe_tradeable']:+.4f}")
    print(f"elapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
