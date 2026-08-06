"""Step 0: reproduce R4's published tables before any R7 number is read.

The label is new, so the harness has to be validated against a target whose
answer is already published. Two controls, both against the forward *return*:

**A. R4's conditional IC table.** ``direction``'s Spearman IC against the
``[t+1, t+31]`` return, split by ``strength`` tercile, as
``state/policy.py``'s docstring carries it to four decimals -- low **+0.0022**,
mid **-0.1128**, high **+0.1314**, unconditional **+0.0385**, on **13,167**
bars. This is the table R7's rate metrics are shaped after.

**B. R4's whole feature table**, §9.1 of the charter: all nine features at
horizons 1, 6 and 30, full sample and both halves, plus each row's bar count.
Produced by calling ``diagnose_features`` -- the same function the ``features``
CLI command runs -- rather than by a private reimplementation.

R4's halves are the **aligned sample cut at its own midpoint**, not the
2023-10-31 split: that is what ``diagnostics._horizon_ic`` does and it is the
convention the published table was produced under. R7's own tables use the
declared split; reproducing a number means reproducing its convention too.

If either control fails, the path is not measuring what R4 measured and every
R7 number through it is uninterpretable.
"""

from __future__ import annotations

import time

import pandas as pd
import r7lib as R
from strategy_lab.features import diagnostics as diag
from strategy_lab.features.registry import get_feature, list_features

# state/policy.py, four decimals.
PUBLISHED_CONDITIONAL = {
    "unconditional": {"ic": +0.0385, "n": 13_167},
    "low": {"ic": +0.0022, "halves": (-0.0361, +0.0468), "n": 4_389},
    "mid": {"ic": -0.1128, "halves": (-0.1201, -0.1101), "n": 4_389},
    "high": {"ic": +0.1314, "halves": (+0.1953, +0.0621), "n": 4_389},
}

# Charter §9.1, three decimals: feature -> horizon -> (full, first half, second half).
PUBLISHED_R4 = {
    "direction": {1: (+0.008, +0.012, +0.003), 6: (+0.015, +0.019, +0.011),
                  30: (+0.039, +0.039, +0.035)},
    "strength": {1: (+0.015, +0.015, +0.015), 6: (+0.048, +0.044, +0.054),
                 30: (+0.057, +0.054, +0.061)},
    "persistence": {1: (+0.008, +0.017, -0.002), 6: (+0.040, +0.055, +0.023),
                    30: (+0.061, +0.084, +0.031)},
    "stability": {1: (-0.006, -0.002, -0.010), 6: (-0.009, +0.003, -0.024),
                  30: (-0.062, -0.032, -0.095)},
    "energy": {1: (+0.010, +0.013, +0.007), 6: (+0.015, +0.019, +0.011),
               30: (+0.059, +0.072, +0.047)},
    "compression": {1: (-0.010, -0.013, -0.007), 6: (-0.015, -0.019, -0.011),
                    30: (-0.059, -0.072, -0.047)},
    "compression_release": {1: (+0.015, +0.010, +0.021), 6: (+0.005, +0.005, +0.004),
                            30: (-0.014, -0.014, -0.013)},
    "participation": {1: (+0.012, +0.013, +0.008), 6: (+0.016, +0.017, +0.011),
                      30: (+0.026, +0.024, +0.024)},
    "crowding": {1: (-0.013, -0.008, -0.017), 6: (-0.026, -0.021, -0.032),
                 30: (-0.021, -0.004, -0.042)},
}
PUBLISHED_BARS = {
    "direction": 13_198, "strength": 15_022, "persistence": 15_022,
    "stability": 15_022, "energy": 14_615, "compression": 14_615,
    "compression_release": 14_614, "participation": 14_639, "crowding": 14_934,
}
PUBLISHED_WARMUP = {
    "direction": 1_920, "strength": 96, "persistence": 96, "stability": 96,
    "energy": 503, "compression": 503, "compression_release": 504,
    "participation": 479, "crowding": 184,
}

# One unit in the published last place. A cell whose disagreement exceeds half
# of that is reported separately: the charter's three-decimal table was rounded
# from a four-decimal intermediate, so a measured -0.00949 appears there as
# -0.010 rather than -0.009. That is a transcription artifact in the doc, not a
# disagreement about the statistic, and it is named rather than absorbed.
PUBLISHED_ULP = 1.0e-3
CLEAN_ROUNDING = 5.0e-4


def conditional_ic(direction: pd.Series, strength: pd.Series, forward: pd.Series) -> dict:
    """R4's table: IC of ``direction`` inside each equal-count ``strength`` tercile.

    The terciles are equal-count buckets of the **paired** sample -- 13,167 rows
    splitting 4,389 / 4,389 / 4,389, which is what the published n says -- and
    each tercile's halves are that tercile's own rows cut at their midpoint, the
    same rule ``_horizon_ic`` applies to the whole.
    """
    paired = pd.DataFrame(
        {"direction": direction, "strength": strength, "forward": forward}
    ).dropna()
    buckets = pd.qcut(paired["strength"], 3, labels=["low", "mid", "high"])
    low, high = (float(x) for x in paired["strength"].quantile([1 / 3, 2 / 3]))

    rows = {
        "unconditional": {
            "ic": diag._ic(paired.rename(columns={"direction": "feature"})),
            "n": int(len(paired)),
        },
        "boundaries": [low, high],
    }
    for name in ("low", "mid", "high"):
        cell = paired[buckets == name].rename(columns={"direction": "feature"})
        midpoint = len(cell) // 2
        rows[name] = {
            "ic": diag._ic(cell),
            "first_half_ic": diag._ic(cell.iloc[:midpoint]),
            "second_half_ic": diag._ic(cell.iloc[midpoint:]),
            "n": int(len(cell)),
        }
    return rows


def main() -> None:
    started = time.time()
    df, rates = R.load_frame()
    print(f"frame: {len(df)} bars {df.index[0]} -> {df.index[-1]}")
    print(f"funding: {len(rates)} settlements; funding column present: "
          f"{'funding_rate' in df.columns}")
    halves = R.halves_of(df)
    print(f"split index {halves.split} at {halves.timestamp}; "
          f"train {halves.split} / test {len(df) - halves.split}")

    checks: list[tuple[str, bool]] = []
    rounding_notes: list[str] = []

    # ---- Control A ------------------------------------------------------
    features = R.feature_columns(df)
    table = conditional_ic(
        features["direction"], features["strength"],
        diag.forward_return(df["close"], horizon=30),
    )
    print("\nCONTROL A -- R4's conditional IC: direction vs [t+1, t+31] return, "
          "by strength tercile")
    print(f"{'cell':>14} {'measured':>9} {'published':>10} {'delta':>10} {'n':>7} "
          f"{'halves measured':>21} {'halves published':>21}")
    for name in ("unconditional", "low", "mid", "high"):
        want = PUBLISHED_CONDITIONAL[name]
        got, n = table[name]["ic"], table[name]["n"]
        got_h = want_h = ""
        if "halves" in want:
            got_h = (f"{table[name]['first_half_ic']:+.4f} / "
                     f"{table[name]['second_half_ic']:+.4f}")
            want_h = f"{want['halves'][0]:+.4f} / {want['halves'][1]:+.4f}"
            for index, key in enumerate(("first_half_ic", "second_half_ic")):
                checks.append((f"A {name} half{index + 1}",
                               abs(table[name][key] - want["halves"][index]) <= 5e-5))
        print(f"{name:>14} {got:>+9.4f} {want['ic']:>+10.4f} {got - want['ic']:>+10.5f} "
              f"{n:>7} {got_h:>21} {want_h:>21}")
        checks.append((f"A {name} ic", abs(got - want["ic"]) <= 5e-5))
        checks.append((f"A {name} n", n == want["n"]))

    # ---- Control B ------------------------------------------------------
    ordered = [get_feature(name) for name in list_features()]
    result = diag.diagnose_features(ordered, df, horizons=(1, 6, 30))
    print("\nCONTROL B -- R4's §9.1 table: nine features vs forward return, "
          "full (first/second half)")
    print(f"{'feature':>20} {'warmup':>7} {'bars':>7} "
          f"{'H=1 measured':>26} {'H=6 measured':>26} {'H=30 measured':>26}")
    for entry in result.diagnostics:
        cells = []
        for ic in entry.ics:
            want = PUBLISHED_R4[entry.name][ic.horizon]
            got = (ic.ic, ic.first_half_ic, ic.second_half_ic)
            cells.append(f"{got[0]:+.4f}({got[1]:+.4f}/{got[2]:+.4f})")
            for index, (g, w) in enumerate(zip(got, want)):
                checks.append((f"B {entry.name}@{ic.horizon}[{index}]",
                               abs(g - w) <= PUBLISHED_ULP))
                if CLEAN_ROUNDING < abs(g - w) <= PUBLISHED_ULP:
                    rounding_notes.append(
                        f"{entry.name}@{ic.horizon} cell{index}: measured {g:+.5f}, "
                        f"charter prints {w:+.3f}"
                    )
        checks.append((f"B {entry.name} bars", entry.observations == PUBLISHED_BARS[entry.name]))
        checks.append((f"B {entry.name} warmup", entry.warmup_bars == PUBLISHED_WARMUP[entry.name]))
        print(f"{entry.name:>20} {entry.warmup_bars:>7} {entry.observations:>7} "
              + " ".join(f"{cell:>26}" for cell in cells))

    # The rank space the machine reads is not the space R4 tabulated, and the
    # boundaries quoted in state/machine.py are over strength's own measured
    # sample rather than direction's paired one -- checked here so R7 does not
    # later conflate the two.
    strength_measured = features["strength"].iloc[PUBLISHED_WARMUP["strength"]:].dropna()
    whole = [float(x) for x in strength_measured.quantile([1 / 3, 2 / 3])]
    print(f"\nraw strength terciles over strength's own {len(strength_measured)} measured "
          f"bars: {whole[0]:.4f} / {whole[1]:.4f}  (state/machine.py: 0.063 / 0.149)")
    print(f"raw strength terciles over direction's paired sample: "
          f"{table['boundaries'][0]:.4f} / {table['boundaries'][1]:.4f}")
    checks.append(("A boundaries (strength's own sample)",
                   abs(whole[0] - 0.063) <= 5e-4 and abs(whole[1] - 0.149) <= 5e-4))

    print()
    failed = [name for name, ok in checks if not ok]
    if rounding_notes:
        print("published-value rounding (charter prints 3dp of a 4dp intermediate):")
        for note in rounding_notes:
            print(f"  {note}")
        print()
    print(f"CONTROL: {len(checks) - len(failed)}/{len(checks)} checks pass")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
    else:
        print("Every published R4 figure reproduces. The harness measures what R4 measured.")

    R.write("step0_control.json", {
        "conditional": table,
        "checks": dict(checks),
        "rounding_notes": rounding_notes,
        "bars": len(df),
        "split_index": halves.split,
        "split_at": str(halves.timestamp),
        "strength_terciles_own_sample": whole,
    })
    print(f"elapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
