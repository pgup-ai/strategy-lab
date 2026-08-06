"""Step 1: build the forward-ER label, and audit it before anything is scored on it.

The plan's fourth outcome row says an easy win on a question four phases could
not answer is evidence of a bug, and names the two places to look: the ``t+1``
anchoring and the tercile boundaries. This step runs both audits **before** any
R7 figure is read, rather than only if the result looks good -- an audit
conditional on liking the answer is not an audit.

Five things checked here.

1. **The formula**, against a hand-computed value on a short series.
2. **The anchoring, by poison probe** -- the shape ``tests/test_lookahead.py``
   uses. Every bar at or before *t* is replaced; ``ER[t]`` must not move. Then
   bar *t+1* is replaced; it must.
3. **The anchor's cost, measured** -- the ER twin of the module docstring's
   random-walk demonstration. A feature that predicts nothing but reads bar
   *t*'s print error is scored against ER anchored at *t* and at *t+1*.
4. **The tercile boundaries come from the training half**, shown by computing
   them both ways and reporting the gap the test half would have contributed.
5. **The label's own shape**: coverage, distribution, and the base rate of
   "trend" in each half -- which is ~1/3 in training by construction and is a
   free measurement out of sample.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import r7lib as R
from strategy_lab.features import diagnostics as diag


def er_anchored_at_t(close: pd.Series, *, horizon: int) -> pd.Series:
    """The wrong anchor, built deliberately so the right one can be priced.

    ``|close[t+H] - close[t]| / sum(|close[i] - close[i-1]|)`` over
    ``i in (t, t+H]``. Bar *t*'s own print is an endpoint of the numerator and
    the left edge of the first step of the path sum.
    """
    path = close.diff().abs().rolling(horizon).sum().shift(-horizon)
    ratio = (close.shift(-horizon) - close).abs() / path
    return ratio.where(path != 0.0)


def check_formula() -> list[tuple[str, bool]]:
    """Hand-computed ER on a short series, at H = 3."""
    close = pd.Series([100.0, 101.0, 103.0, 102.0, 105.0, 104.0, 108.0])
    got = diag.forward_efficiency_ratio(close, horizon=3)
    # t = 0: entry close[1] = 101, exit close[4] = 105.
    # path = |103-101| + |102-103| + |105-102| = 2 + 1 + 3 = 6; |105-101| = 4.
    want0 = 4.0 / 6.0
    # t = 1: entry close[2] = 103, exit close[5] = 104.
    # path = |102-103| + |105-102| + |104-105| = 1 + 3 + 1 = 5; |104-103| = 1.
    want1 = 1.0 / 5.0
    tail = got.iloc[-4:]
    return [
        ("formula t=0", abs(float(got.iloc[0]) - want0) < 1e-12),
        ("formula t=1", abs(float(got.iloc[1]) - want1) < 1e-12),
        ("last H+1 bars are NaN", bool(tail.isna().all()) and len(tail) == 4),
        ("range is 0..1", bool(((got.dropna() >= 0.0) & (got.dropna() <= 1.0)).all())),
        ("flat window is NaN, not 0.0",
         bool(diag.forward_efficiency_ratio(pd.Series([1.0] * 10), horizon=3).isna().all())),
    ]


def check_poison(close: pd.Series, *, horizon: int, probe: int) -> list[tuple[str, bool]]:
    """Poison the past, then the future, and watch which one moves ``ER[probe]``."""
    baseline = diag.forward_efficiency_ratio(close, horizon=horizon)

    past = close.copy()
    past.iloc[: probe + 1] *= 3.7  # every bar at or before t, bar t included
    poisoned_past = diag.forward_efficiency_ratio(past, horizon=horizon)

    future = close.copy()
    future.iloc[probe + 1] *= 1.05  # the anchor bar itself
    poisoned_future = diag.forward_efficiency_ratio(future, horizon=horizon)

    return [
        (f"ER[{probe}] ignores every bar <= t",
         float(baseline.iloc[probe]) == float(poisoned_past.iloc[probe])),
        (f"ER[{probe}] moves when close[t+1] moves",
         float(baseline.iloc[probe]) != float(poisoned_future.iloc[probe])),
        ("poisoning the past changes no ER at or after the probe",
         bool(baseline.iloc[probe:].equals(poisoned_past.iloc[probe:]))),
    ]


def price_error_feature(
    seed: int, bars: int, *, error: float = 0.03
) -> tuple[pd.Series, pd.Series]:
    """A random walk carrying an i.i.d. print error, and a feature that reads it.

    The feature is bar *t*'s deviation from its own trailing mean, so it
    predicts nothing and is a pure function of ``close[t]``. Anchored at *t*, a
    high print error inflates both the ER numerator's endpoint and the first
    step of its path sum; anchored at *t+1* it can reach neither.

    The print error is three times the walk's per-bar volatility, which is what
    makes the *forward return* control unambiguous rather than merely
    directional: at the shortest declared horizon the error then dominates the
    six bars of walk it is divided into. It is a demonstration of a mechanism,
    not a claim about how noisy real prints are.
    """
    rng = np.random.default_rng(seed)
    walk = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, bars)))
    close = pd.Series(walk * np.exp(rng.normal(0.0, error, bars)))
    feature = np.log(close) - np.log(close).rolling(24).mean()
    return close, feature


def main() -> None:
    started = time.time()
    checks: list[tuple[str, bool]] = []

    print("AUDIT 1-2 -- the formula and the anchoring")
    checks += check_formula()
    rng = np.random.default_rng(7)
    synthetic = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 400))))
    for horizon in R.HORIZONS:
        checks += [(f"H={horizon}: {name}", ok)
                   for name, ok in check_poison(synthetic, horizon=horizon, probe=120)]
    for name, ok in checks:
        print(f"  {R.mark(ok)}  {name}")

    print("\nAUDIT 3 -- what the anchor is worth, on a feature that predicts nothing")
    print("The forward *return* rows are the positive control: they are the "
          "diagnostics module's own")
    print("measurement, and they prove the probe can see anchor leakage when "
          "there is any.")
    print(f"{'seed':>5} {'H':>4} {'target':>8} {'IC at close[t]':>16} "
          f"{'IC at close[t+1]':>18} {'noise 2/sqrt(n/H)':>19}")
    anchor_rows = []
    bars = 4_000
    for seed in (11, 12, 13):
        close, feature = price_error_feature(seed, bars)
        for horizon in R.HORIZONS:
            noise = 2.0 / np.sqrt(bars / horizon)
            wrong_er = diag._ic(diag._paired(feature, er_anchored_at_t(close, horizon=horizon)))
            right_er = diag._ic(diag._paired(feature, R.forward_er(close, horizon=horizon)))
            wrong_ret = diag._ic(diag._paired(
                feature, close.shift(-horizon) / close - 1.0))
            right_ret = diag._ic(diag._paired(
                feature, diag.forward_return(close, horizon=horizon)))
            print(f"{seed:>5} {horizon:>4} {'return':>8} {wrong_ret:>+16.4f} "
                  f"{right_ret:>+18.4f} {noise:>19.4f}")
            print(f"{'':>5} {'':>4} {'ER':>8} {wrong_er:>+16.4f} "
                  f"{right_er:>+18.4f} {noise:>19.4f}")
            anchor_rows.append({
                "seed": seed, "horizon": horizon, "noise_2se": float(noise),
                "er_ic_at_t": wrong_er, "er_ic_at_t1": right_er,
                "return_ic_at_t": wrong_ret, "return_ic_at_t1": right_ret,
            })

    worst_ret_wrong = max(abs(row["return_ic_at_t"]) for row in anchor_rows)
    worst_ret_right = max(abs(row["return_ic_at_t1"]) for row in anchor_rows)
    inside_noise = all(
        abs(row["er_ic_at_t1"]) <= row["noise_2se"] for row in anchor_rows
    )
    # Judged per row against that row's own noise band, not against one flat
    # number: overlapping forward windows leave n/H independent observations, so
    # a |IC| of 0.09 is a leak at H = 6 and is nothing at H = 90.
    checks.append((
        "the probe sees anchor leakage on the forward return (the positive control)",
        all(abs(row["return_ic_at_t"]) > 0.20 for row in anchor_rows)
        and all(abs(row["return_ic_at_t1"]) <= row["noise_2se"] for row in anchor_rows),
    ))
    checks.append((
        "forward ER at the t+1 anchor pays a null feature nothing beyond noise",
        inside_noise,
    ))
    for name, ok in checks[-2:]:
        print(f"  {R.mark(ok)}  {name}")
    print(f"  return: largest |IC| at t {worst_ret_wrong:.4f}, at t+1 "
          f"{worst_ret_right:.4f}")
    print(f"  ER:     largest |IC| at t "
          f"{max(abs(r['er_ic_at_t']) for r in anchor_rows):.4f}, at t+1 "
          f"{max(abs(r['er_ic_at_t1']) for r in anchor_rows):.4f}")
    print("  ER is a ratio, so bar t's print error inflates its numerator and "
          "its path sum together")
    print("  and largely cancels. The t+1 anchor is still the rule; what it "
          "buys on ER is small.")

    # ---- the real frame -------------------------------------------------
    df, _ = R.load_frame()
    halves = R.halves_of(df)
    print(f"\nframe: {len(df)} bars {df.index[0]} -> {df.index[-1]}; "
          f"split {halves.timestamp}")

    print("\nAUDIT 3b -- the same question asked of the real frame and the real "
          "features, H=30")
    print("Every registered feature against ER at both anchors. This is the "
          "check that matters:")
    print("a synthetic demonstration cannot rule out a leak these particular "
          "features would show.")
    features = R.feature_columns(df)
    wrong_er30 = er_anchored_at_t(df["close"], horizon=30)
    right_er30 = R.forward_er(df["close"], horizon=30)
    print(f"{'feature':>20} {'IC at close[t]':>16} {'IC at close[t+1]':>18} "
          f"{'delta':>9}")
    real_anchor = {}
    for name, values in features.items():
        measured = values.dropna()
        wrong = diag._ic(diag._paired(measured, wrong_er30))
        right = diag._ic(diag._paired(measured, right_er30))
        real_anchor[name] = {"ic_at_t": wrong, "ic_at_t1": right}
        print(f"{name:>20} {wrong:>+16.4f} {right:>+18.4f} {wrong - right:>+9.4f}")
    biggest = max(real_anchor.values(), key=lambda row: abs(row["ic_at_t"] - row["ic_at_t1"]))
    checks.append((
        "no real feature's ER IC depends materially on the anchor",
        abs(biggest["ic_at_t"] - biggest["ic_at_t1"]) < 0.02,
    ))
    print(f"  {R.mark(checks[-1][1])}  {checks[-1][0]} "
          f"(largest move {biggest['ic_at_t'] - biggest['ic_at_t1']:+.4f})")

    print("\nAUDIT 4-5 -- the label on the stored frame")
    print(f"{'H':>4} {'defined':>8} {'cov':>7} {'median ER':>10} {'IQR':>7} "
          f"{'train cut (hi)':>15} {'full-sample cut':>16} {'gap':>8} "
          f"{'base train':>11} {'base test':>10}")
    label_rows = {}
    for horizon in R.HORIZONS:
        er = R.forward_er(df["close"], horizon=horizon)
        label, (low, high) = R.trend_label(er, halves)
        # The same boundary computed the wrong way, so the plan's "must not look
        # across the split" is a measured gap rather than an assurance.
        full_high = float(er.dropna().quantile(2 / 3))
        rates = R.rate_table(pd.Series(1.0, index=df.index), label, halves)
        quartiles = er.dropna().quantile([0.25, 0.75])
        print(f"{horizon:>4} {int(er.notna().sum()):>8} {er.notna().mean():>7.1%} "
              f"{er.median():>10.4f} {float(quartiles.iloc[1] - quartiles.iloc[0]):>7.4f} "
              f"{high:>15.4f} {full_high:>16.4f} {full_high - high:>+8.4f} "
              f"{rates['train']['base_rate']:>11.4f} {rates['test']['base_rate']:>10.4f}")
        label_rows[horizon] = {
            "defined": int(er.notna().sum()),
            "coverage": float(er.notna().mean()),
            "median": float(er.median()),
            "cut_low": low,
            "cut_high": high,
            "cut_high_full_sample": full_high,
            "base_rate": {which: rates[which]["base_rate"] for which in rates},
            "n": {which: rates[which]["n"] for which in rates},
        }
        checks.append((f"H={horizon}: training base rate is a tercile",
                       abs(rates["train"]["base_rate"] - 1 / 3) < 0.01))
        checks.append((f"H={horizon}: the cut moves if the test half is let in",
                       abs(full_high - high) > 1e-6))

    for name, ok in checks[-2 * len(R.HORIZONS):]:
        print(f"  {R.mark(ok)}  {name}")

    print("\nAUDIT 6 -- is ER scale-free? The question `energy`'s result turns on.")
    print("`energy` is a rolling percentile of realized volatility. If ER moved "
          "with the level of")
    print("volatility, an energy-vs-ER relationship would be a units artifact "
          "rather than a market")
    print("fact. Under an i.i.d. driftless walk E[ER] ~ 1/sqrt(H) whatever "
          "sigma is; measured:")
    print(f"{'sigma':>8} " + " ".join(f"{f'median ER H={h}':>17}" for h in R.HORIZONS))
    scale_rows = {}
    for sigma in (0.002, 0.01, 0.05):
        walk = pd.Series(100.0 * np.exp(np.cumsum(
            np.random.default_rng(99).normal(0.0, sigma, 60_000)
        )))
        medians = [float(R.forward_er(walk, horizon=h).median()) for h in R.HORIZONS]
        scale_rows[sigma] = medians
        print(f"{sigma:>8.3f} " + " ".join(f"{value:>17.4f}" for value in medians))
    print(f"{'1/sqrt(H)':>8} " + " ".join(
        f"{1 / np.sqrt(h):>17.4f}" for h in R.HORIZONS
    ))
    spread = max(
        max(row[index] for row in scale_rows.values())
        - min(row[index] for row in scale_rows.values())
        for index in range(len(R.HORIZONS))
    )
    checks.append(("ER does not move with the level of volatility", spread < 0.01))
    print(f"  {R.mark(checks[-1][1])}  {checks[-1][0]} "
          f"(largest spread across a 25x range of sigma: {spread:.4f})")
    print(f"{'BTC 4h':>8} " + " ".join(
        f"{label_rows[h]['median']:>17.4f}" for h in R.HORIZONS
    ))
    print("  Compare BTC against the *simulated* medians, not against 1/sqrt(H) "
          "-- that closed form")
    print("  is a ratio of expectations and sits above the median of the ratio "
          "at every horizon.")
    print("  BTC comes in slightly above the driftless walk at all three: "
          "marginally MORE")
    print("  directionally efficient than noise, which is the whole of what a "
          "trend thesis has")
    print("  to work with here.")

    failed = [name for name, ok in checks if not ok]
    print(f"\nLABEL AUDIT: {len(checks) - len(failed)}/{len(checks)} checks pass")
    if failed:
        print(f"FAILED: {', '.join(failed)}")

    R.write("step1_label.json", {
        "checks": dict(checks),
        "anchor_cost": anchor_rows,
        "real_frame_anchor": real_anchor,
        "scale_free": {str(sigma): row for sigma, row in scale_rows.items()},
        "label": label_rows,
    })
    print(f"elapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
