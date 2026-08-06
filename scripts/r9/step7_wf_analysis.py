"""The walk-forward distribution, plus one declared extra.

The extra: R5's own pinned cell is evaluated on every fold's test block too. It
selects nothing and changes no fold's winner -- it answers "does re-deriving the
cell every six months beat never re-deriving it", which is the question "how
often does the same cell win" turns into once the answer is 'not often'.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import r9lib as R

OUT = R.OUT
OUT.mkdir(parents=True, exist_ok=True)


def chain(returns: list[pd.Series]) -> pd.Series:
    return pd.concat(returns).reset_index(drop=True)


def main() -> None:
    folds = json.loads((OUT / "walkforward.json").read_text())
    frame = R.load_frame()
    df, _ = frame

    sharpes = np.array([fold["test"]["sharpe_tradeable"] for fold in folds])
    whole = np.array([fold["test"]["sharpe_whole_frame"] for fold in folds])
    nets = np.array([fold["test"]["net_return_pct"] for fold in folds])
    winners = [fold["winner"] for fold in folds]

    print("fold | train bars | selected cell                              | OOS sharpe | net %")
    for fold in folds:
        print(f"  {fold['fold']}  | {fold['train_bars']:>10} | {fold['winner_label']:<42} | "
              f"{fold['test']['sharpe_tradeable']:+10.4f} | {fold['test']['net_return_pct']:+6.2f}")

    print(f"\ndistinct winning cells: {len(set(winners))} of {len(folds)} folds")
    for cell in dict.fromkeys(winners):
        which = [fold["fold"] for fold in folds if fold["winner"] == cell]
        label = next(fold["winner_label"] for fold in folds if fold["winner"] == cell)
        print(f"  {label:<42} folds {which}")
    print(f"R5's pinned cell wins folds: "
          f"{[f['fold'] for f in folds if f['winner'] == R.cell_key(R.gate.TRAINED_MACHINE)]}")
    print(f"scalars agree on every fold: {all(fold['scalars_agree'] for fold in folds)}")

    print(f"\nOOS Sharpe (tradeable-bar scalar): mean {sharpes.mean():+.4f}  median "
          f"{np.median(sharpes):+.4f}  sd {sharpes.std(ddof=1):.4f}  positive "
          f"{(sharpes > 0).sum()}/{len(sharpes)}  min {sharpes.min():+.4f}  max {sharpes.max():+.4f}")
    print(f"OOS Sharpe (whole-window scalar): mean {whole.mean():+.4f}  positive "
          f"{(whole > 0).sum()}/{len(whole)}")
    print(f"OOS net %: mean {nets.mean():+.3f}  median {np.median(nets):+.3f}  positive "
          f"{(nets > 0).sum()}/{len(nets)}  sum {nets.sum():+.2f}  "
          f"compounded {(np.prod(1 + nets / 100) - 1) * 100:+.2f}")
    hold = np.array([fold["buy_and_hold_pct"] for fold in folds])
    print(f"buy & hold over the same blocks: compounded "
          f"{(np.prod(1 + hold / 100) - 1) * 100:+.2f}%  positive {(hold > 0).sum()}/{len(hold)}")

    # The declared extra: R5's pinned cell on each block, and the chained curves.
    pinned = R.gate.machine(R.gate.TRAINED_MACHINE)
    rows, wf_returns, pinned_returns = [], [], []
    for fold in folds:
        start = df.index.get_loc(pd.Timestamp(fold["test_start"]))
        stop = df.index.get_loc(pd.Timestamp(fold["test_end"])) + 1
        fixed = R.run(pinned, frame, OUT / "wf_fixed" / f"f{fold['fold']}",
                      first_tradeable=start, stop=stop)
        chosen = R.run(R.gate.machine(_config(fold)), frame,
                       OUT / "wf_chosen" / f"f{fold['fold']}",
                       first_tradeable=start, stop=stop)
        assert abs(chosen["net_return_pct"] - fold["test"]["net_return_pct"]) < 1e-9, (
            "the re-run of a fold's chosen cell does not reproduce the fold"
        )
        identical = bool(np.array_equal(
            chosen["returns"].to_numpy()[-fold["test_bars"]:],
            fixed["returns"].to_numpy()[-fold["test_bars"]:],
        ))
        rows.append({
            "fold": fold["fold"],
            "chosen_sharpe": chosen["sharpe_tradeable"],
            "chosen_net": chosen["net_return_pct"],
            "pinned_sharpe": fixed["sharpe_tradeable"],
            "pinned_net": fixed["net_return_pct"],
            "same_book": identical,
        })
        wf_returns.append(chosen["returns"])
        pinned_returns.append(fixed["returns"])
        print(f"fold {fold['fold']}: chosen {chosen['sharpe_tradeable']:+.4f} / "
              f"{chosen['net_return_pct']:+.2f}%   R5-pinned {fixed['sharpe_tradeable']:+.4f} / "
              f"{fixed['net_return_pct']:+.2f}%   same book: {identical}")

    for label, series in (("walk-forward", wf_returns), ("R5 pinned cell", pinned_returns)):
        joined = chain(series)
        sharpe = float(joined.mean() / joined.std(ddof=1) * np.sqrt(2190.0))
        total = float((1.0 + joined).prod() - 1.0) * 100.0
        print(f"chained 9-block OOS curve, {label}: {len(joined)} bars, Sharpe {sharpe:+.4f}, "
              f"compounded net {total:+.2f}%")

    (OUT / "wf_summary.json").write_text(json.dumps(rows, indent=2))


def _config(fold: dict):
    """The declared cell whose key is ``fold['winner']`` -- matched, not parsed.

    Parsing the key would rebuild ``exit_strength`` as 0.333333 rather than 1/3,
    which is a different machine by 3e-7 of rank.
    """
    for config in R.gate.declared_machines():
        if R.cell_key(config) == fold["winner"]:
            return config
    raise KeyError(fold["winner"])


if __name__ == "__main__":
    main()
