"""The univariate diagnostic, and the two ways it could quietly lie.

Both are exercised with purpose-built features rather than registered ones,
because a test that depends on what ``strength`` happens to score on synthetic
data measures the fixture, not the harness.

The two lies, whose reasoning ``features.diagnostics`` carries in full:

1. **A forward return anchored at the feature's own bar** -- no ``shift(-1)`` in
   sight, and
   :func:`test_a_target_anchored_at_the_feature_s_own_bar_manufactures_an_ic`
   measures it: -0.53 against a series with no forward information in it at all.
2. **A split-half that is not a split.** Reporting the full-sample IC twice
   under two labels passes every test that only checks the halves exist, so
   :func:`test_the_two_halves_are_measured_separately_not_copied` builds a
   feature that predicts in the first half and anti-predicts in the second,
   where the full-sample number is the one that is meaningless.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from strategy_lab.features.diagnostics import (
    DEFAULT_HORIZONS,
    REDUNDANT_CORRELATION,
    FeatureDiagnostic,
    diagnose,
    diagnose_features,
    forward_return,
    information_coefficient,
    to_record,
)
from strategy_lab.features.diagnostics_report import render_diagnostics_html
from strategy_lab.features.registry import get_feature, list_features
from tests.conftest import synthetic_ohlcv_with_funding


def _frame(close: np.ndarray) -> pd.DataFrame:
    index = pd.date_range(
        "2024-01-01", periods=len(close), freq="4h", tz="UTC", name="timestamp"
    )
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(len(close), 500.0),
        },
        index=index,
    )


def noisy_price(n: int = 2000, seed: int = 5, noise: float = 0.01) -> pd.DataFrame:
    """A random walk plus an i.i.d. print error -- no forward information at all.

    The print error is the whole point: it is what a target anchored at
    ``close[t]`` divides by, and what an honest one never touches.
    """
    rng = np.random.default_rng(seed)
    walk = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    return _frame(walk * np.exp(rng.normal(0, noise, n)))


@dataclass(frozen=True)
class _CurrentBarDeviation:
    """How far this bar's print sits from its own trailing mean, 0..1.

    Causal -- it reads nothing after bar *t* -- and predictive of nothing. It
    exists to be scored against both target conventions, since it is exactly the
    shape that the anchored one rewards for free.
    """

    name: str = "current_bar_deviation"
    version: str = "1.0.0"
    window: int = 20
    warmup_bars: int = 20

    def compute(self, df: pd.DataFrame) -> pd.Series:
        trailing = df["close"].rolling(self.window)
        return np.tanh((df["close"] - trailing.mean()) / trailing.std())


@dataclass(frozen=True)
class _RegimeSplit:
    """Predicts the next bar in the first half of the sample and inverts in the second.

    Full-sample IC near zero, halves at +1 and -1: the case where the
    full-sample number is the one that means nothing.

    It reads the future on purpose -- it is the target itself, sign-flipped
    halfway -- because the subject here is the harness, not causality. Nothing
    registers it, so the lookahead probe never sees it.
    """

    name: str = "regime_split"
    version: str = "1.0.0"
    warmup_bars: int = 5

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # The target itself: the return from close[t+1] to close[t+2].
        future = df["close"].shift(-2) / df["close"].shift(-1) - 1.0
        half = len(df) // 2
        sign = pd.Series(np.where(np.arange(len(df)) < half, 1.0, -1.0), index=df.index)
        return future * sign


@dataclass(frozen=True)
class _Constant:
    name: str = "constant"
    version: str = "1.0.0"
    warmup_bars: int = 10

    def compute(self, df: pd.DataFrame) -> pd.Series:
        values = pd.Series(0.5, index=df.index)
        values.iloc[: self.warmup_bars] = np.nan
        return values


@dataclass(frozen=True)
class _Echo:
    """A copy of another feature under a different name -- the redundancy case."""

    name: str = "echo"
    version: str = "1.0.0"
    warmup_bars: int = 20

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return _CurrentBarDeviation().compute(df) * 2.0 + 1.0


def test_forward_return_starts_one_bar_after_the_feature_s_own_bar():
    # Deliberately not a constant-ratio series: on one, close[t+1+h]/close[t+1]
    # and close[t+h]/close[t] are the same number and this test sees nothing.
    close = pd.Series([100.0, 200.0, 220.0, 250.0, 300.0])
    # From close[t+1] to close[t+1+h]: at t=0 that is 220/200 - 1.
    assert forward_return(close, horizon=1).iloc[0] == pytest.approx(0.10)
    assert forward_return(close, horizon=2).iloc[0] == pytest.approx(0.25)
    # The feature's own close never appears in its own target: anchored there,
    # horizon 1 would read 200/100 - 1.
    assert forward_return(close, horizon=1).iloc[0] != pytest.approx(1.0)


def test_a_non_positive_horizon_is_refused():
    with pytest.raises(ValueError, match="horizon"):
        forward_return(pd.Series([1.0, 2.0, 3.0]), horizon=0)


def test_a_target_anchored_at_the_feature_s_own_bar_manufactures_an_ic():
    """The mutation this module exists to catch, measured.

    Same causal feature, same prices, two target conventions. Anchored at the
    feature's own close the IC is strongly negative on a series that carries no
    forward information; started one bar later it is noise.
    """
    df = noisy_price()
    values = _CurrentBarDeviation().compute(df)
    close = df["close"]

    honest = information_coefficient(values, forward_return(close, horizon=6))
    anchored = information_coefficient(values, close.shift(-6) / close - 1.0)

    assert abs(honest) < 0.05, f"honest target should carry no information, got {honest:.3f}"
    assert anchored < -0.3, f"anchored target should manufacture one, got {anchored:.3f}"


def test_the_diagnostic_uses_the_honest_target():
    """Same comparison, through ``diagnose`` rather than the helper directly."""
    diagnostic = diagnose(_CurrentBarDeviation(), noisy_price(), horizons=(6,))
    assert abs(diagnostic.ics[0].ic) < 0.05


def test_information_coefficient_is_rank_based_not_linear():
    """Spearman: a monotone squash of a perfect predictor is still perfect."""
    index = pd.date_range("2024-01-01", periods=200, freq="4h", tz="UTC")
    forward = pd.Series(np.linspace(-0.1, 0.1, 200), index=index)
    assert information_coefficient(np.tanh(forward * 50), forward) == pytest.approx(1.0)


def test_information_coefficient_of_a_constant_feature_is_nan_not_zero():
    """No variation is not "no relationship" -- it is no measurement."""
    index = pd.date_range("2024-01-01", periods=200, freq="4h", tz="UTC")
    constant = pd.Series(0.5, index=index)
    forward = pd.Series(np.random.default_rng(1).normal(size=200), index=index)
    assert np.isnan(information_coefficient(constant, forward))


def test_information_coefficient_of_too_few_pairs_is_nan():
    index = pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC")
    values = pd.Series(np.arange(5.0), index=index)
    assert np.isnan(information_coefficient(values, values))


def test_the_two_halves_are_measured_separately_not_copied():
    """A feature that inverts halfway: the halves disagree, the full sample is mush.

    Reporting the full-sample IC twice under two labels would pass any test that
    only checks the fields exist. This one fails.
    """
    diagnostic = diagnose(_RegimeSplit(), noisy_price(n=1200), horizons=(1,))
    [horizon] = diagnostic.ics

    assert horizon.first_half_ic > 0.99
    assert horizon.second_half_ic < -0.99
    assert abs(horizon.ic) < 0.2, "the full-sample number is the meaningless one here"
    assert not horizon.halves_agree
    assert horizon.first_half_ic != pytest.approx(horizon.ic)
    assert horizon.second_half_ic != pytest.approx(horizon.ic)


def test_the_halves_split_the_measured_sample_evenly():
    diagnostic = diagnose(_CurrentBarDeviation(), noisy_price(n=1000), horizons=(1,))
    [horizon] = diagnostic.ics
    assert horizon.first_half_observations + horizon.second_half_observations == (
        horizon.observations
    )
    assert abs(horizon.first_half_observations - horizon.second_half_observations) <= 1


def test_coverage_is_measured_past_warmup_not_over_the_whole_frame():
    """Warmup NaNs are expected and say nothing about whether a feature computes.

    The holey feature pins the denominator: 100 missing bars out of the 490 past
    warmup, not out of the 500 in the frame.
    """

    @dataclass(frozen=True)
    class _Holey:
        name: str = "holey"
        version: str = "1.0.0"
        warmup_bars: int = 10

        def compute(self, df: pd.DataFrame) -> pd.Series:
            values = pd.Series(np.arange(len(df), dtype="float64"), index=df.index)
            values.iloc[: self.warmup_bars] = np.nan
            values.iloc[100:200] = np.nan
            return values

    intact = diagnose(_Constant(), noisy_price(n=500), horizons=(1,))
    assert intact.coverage == pytest.approx(1.0)
    assert intact.observations == 500 - _Constant().warmup_bars

    holey = diagnose(_Holey(), noisy_price(n=500), horizons=(1,))
    assert holey.coverage == pytest.approx(1.0 - 100 / 490)


def test_turnover_is_the_mean_absolute_bar_to_bar_change():
    @dataclass(frozen=True)
    class _Sawtooth:
        name: str = "sawtooth"
        version: str = "1.0.0"
        warmup_bars: int = 0

        def compute(self, df: pd.DataFrame) -> pd.Series:
            return pd.Series(
                np.where(np.arange(len(df)) % 2 == 0, 0.0, 0.25), index=df.index
            )

    diagnostic = diagnose(_Sawtooth(), noisy_price(n=400), horizons=(1,))
    assert diagnostic.turnover == pytest.approx(0.25)
    # A series that alternates every bar has no lag-1 persistence to speak of.
    assert diagnostic.autocorrelation < -0.9


def test_correlations_are_pairwise_over_the_rows_both_features_define():
    features = [_CurrentBarDeviation(), _Echo(), _Constant()]
    result = diagnose_features(features, noisy_price(n=800), horizons=(1,))

    # An affine copy under another name correlates at exactly 1.
    assert result.correlations["current_bar_deviation"]["echo"] == pytest.approx(1.0)
    # A constant has no variance, so no correlation is defined against it.
    assert np.isnan(result.correlations["constant"]["current_bar_deviation"])


def test_max_correlation_names_the_partner_and_keeps_the_sign():
    @dataclass(frozen=True)
    class _Inverse:
        name: str = "inverse"
        version: str = "1.0.0"
        warmup_bars: int = 20

        def compute(self, df: pd.DataFrame) -> pd.Series:
            return -_CurrentBarDeviation().compute(df)

    result = diagnose_features(
        [_CurrentBarDeviation(), _Inverse(), _Constant()], noisy_price(n=800), horizons=(1,)
    )
    partner, value = result.max_correlation("current_bar_deviation")
    assert partner == "inverse"
    assert value == pytest.approx(-1.0)


def test_redundant_pairs_are_reported_once_each_not_twice():
    result = diagnose_features(
        [_CurrentBarDeviation(), _Echo()], noisy_price(n=800), horizons=(1,)
    )
    pairs = result.redundant_pairs()
    assert len(pairs) == 1
    first, second, value = pairs[0]
    assert {first, second} == {"current_bar_deviation", "echo"}
    assert abs(value) >= REDUNDANT_CORRELATION


def test_two_features_sharing_a_name_are_refused():
    """The correlation matrix is keyed by name, so a collision silently drops one."""
    with pytest.raises(ValueError, match="name"):
        diagnose_features([_Constant(), _Constant()], noisy_price(n=600), horizons=(1,))


@pytest.mark.parametrize("name", list_features())
def test_every_registered_feature_can_be_diagnosed(name):
    """The R4 gate in one line: no feature ships without a diagnostic.

    Synthetic data, so the numbers mean nothing -- what is asserted is that the
    harness produces a complete row for every registered feature, at every
    default horizon, rather than a NaN where a statistic should be.
    """
    feature = get_feature(name)
    df = synthetic_ohlcv_with_funding(n=feature.warmup_bars + 900, freq="4h")
    diagnostic = diagnose(feature, df, horizons=DEFAULT_HORIZONS)

    assert isinstance(diagnostic, FeatureDiagnostic)
    assert diagnostic.name == feature.name
    assert diagnostic.coverage == pytest.approx(1.0)
    assert diagnostic.observations == 900
    assert [h.horizon for h in diagnostic.ics] == list(DEFAULT_HORIZONS)
    assert all(np.isfinite(h.ic) for h in diagnostic.ics)
    assert all(np.isfinite(h.first_half_ic) for h in diagnostic.ics)
    assert all(np.isfinite(h.second_half_ic) for h in diagnostic.ics)
    assert np.isfinite(diagnostic.turnover)


def _record_for(features, n=800):
    return to_record(diagnose_features(features, noisy_price(n=n), horizons=(1, 6)))


def test_the_record_is_valid_json_with_nulls_where_nothing_was_measured():
    """``NaN`` is not valid JSON; a page that cannot parse its payload fails silently."""
    record = _record_for([_CurrentBarDeviation(), _Constant()])
    text = json.dumps(record, allow_nan=False)

    assert "NaN" not in text
    constant = next(f for f in record["features"] if f["name"] == "constant")
    assert constant["ic"][0]["ic"] is None
    assert constant["max_correlation"]["r"] is None


def test_the_record_carries_every_feature_and_horizon():
    record = _record_for([_CurrentBarDeviation(), _Echo()])
    assert record["horizons"] == [1, 6]
    assert [f["name"] for f in record["features"]] == ["current_bar_deviation", "echo"]
    for feature in record["features"]:
        assert [entry["horizon"] for entry in feature["ic"]] == [1, 6]
        assert {"ic", "first_half_ic", "second_half_ic"} <= set(feature["ic"][0])
    assert record["redundant_pairs"][0]["features"] == ["current_bar_deviation", "echo"]


def _html(features=None, n=800, config=None):
    features = features or [_CurrentBarDeviation(), _Echo(), _RegimeSplit()]
    result = diagnose_features(features, noisy_price(n=n), horizons=(1, 6))
    return render_diagnostics_html(result=result, config=config or {"candle_count": n})


def test_report_is_self_contained():
    html = _html()
    assert "<script src=" not in html, "must not reference external assets"
    assert "<link" not in html
    assert "http://" not in html and "https://" not in html
    assert html.lstrip().startswith("<!DOCTYPE html>")


def test_report_renders_one_row_per_feature():
    html = _html()
    rows = re.findall(r'<th class="row-label">([^<]+)</th>', html)
    assert rows == ["current_bar_deviation", "echo", "regime_split"]


def test_report_shows_both_halves_beside_every_full_sample_ic():
    """The split is the reading; the full-sample number alone is the lie this catches."""
    result = diagnose_features([_RegimeSplit()], noisy_price(n=1200), horizons=(1,))
    html = render_diagnostics_html(result=result, config={})
    [entry] = result.diagnostics[0].ics

    assert f"{entry.ic:+.3f}" in html
    assert f"{entry.first_half_ic:+.3f} / {entry.second_half_ic:+.3f}" in html
    # And the halves are not the full-sample number wearing two labels.
    assert f"{entry.ic:+.3f} / {entry.ic:+.3f}" not in html


def test_a_feature_whose_halves_disagree_is_marked():
    marked = _html([_RegimeSplit()], n=1200)
    steady = _html([_CurrentBarDeviation()], n=1200)
    assert "cell split" in marked
    assert marked.count("cell split") > steady.count("cell split")


def test_report_flags_a_redundant_pair_by_name():
    html = _html([_CurrentBarDeviation(), _Echo()])
    assert "Redundancy" in html
    assert "current_bar_deviation" in html and "echo" in html
    assert "r = +1.000" in html


def test_report_says_so_when_nothing_is_redundant():
    """Silence would read as 'not checked', which is the one thing it must not."""
    html = _html([_CurrentBarDeviation(), _RegimeSplit()])
    assert "No pair reaches" in html


def test_report_names_what_was_skipped():
    """A feature left out has to appear on the page, or it shipped unexamined."""
    html = _html(config={"skipped": {"crowding": "needs a funding_rate column"}})
    assert "Not examined" in html
    assert "crowding" in html and "funding_rate" in html


def test_report_escapes_interpolated_values():
    html = _html(config={"identity": {"symbol": "<script>alert(1)</script>"}})
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_rendering_no_features_is_refused():
    result = diagnose_features([_Constant()], noisy_price(n=400), horizons=(1,))
    empty = type(result)(horizons=result.horizons, diagnostics=(), correlations={})
    with pytest.raises(ValueError, match="no features"):
        render_diagnostics_html(result=empty, config={})
