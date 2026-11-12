"""Market regime detection: rules, causality and stability."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ml.features.regime import (
    REGIME_COLUMNS,
    REGIMES,
    RegimeConfig,
    _hold,
    _raw_label,
    detect_regimes,
    regime_stability,
)

CFG = RegimeConfig()


def _returns(segments: list[tuple[int, float, float]], seed: int = 0) -> pd.Series:
    """Daily returns built from (sessions, drift, volatility) segments."""
    rng = np.random.default_rng(seed)
    values = np.concatenate([rng.normal(mu, sigma, n) for n, mu, sigma in segments])
    return pd.Series(values, index=pd.bdate_range("2015-01-01", periods=len(values)))


# ------------------------------------------------------------------------- rules


@pytest.mark.parametrize(
    ("vol_level", "trend", "strength", "drawdown", "expected"),
    [
        (0.95, -0.20, 0.40, -0.25, "market_stress"),  # extreme vol and a deep drawdown
        (0.95, -0.02, 0.05, -0.04, "high_volatility"),  # extreme vol, shallow drawdown
        (0.80, 0.10, 0.30, -0.01, "high_volatility"),  # volatility outranks trend
        (0.50, 0.10, 0.30, 0.00, "bull"),
        (0.50, -0.10, 0.30, -0.10, "bear"),
        (0.50, 0.10, 0.02, 0.00, "sideways"),  # a move without direction is not a trend
        (0.10, 0.01, 0.05, 0.00, "low_volatility"),
        (0.10, 0.10, 0.30, 0.00, "bull"),  # a quiet rally is still a bull market
        (0.50, 0.01, 0.05, -0.02, "sideways"),
    ],
)
def test_rules(vol_level, trend, strength, drawdown, expected):
    label, margin = _raw_label(vol_level, trend, strength, drawdown, CFG)
    assert label == expected
    assert label in REGIMES
    assert margin >= 0


def test_undefined_measurements_give_no_label():
    assert _raw_label(np.nan, 0.1, 0.3, 0.0, CFG) == (None, pytest.approx(np.nan, nan_ok=True))


def test_margin_grows_with_distance_from_the_boundary():
    _, near = _raw_label(0.76, 0.0, 0.0, 0.0, CFG)
    _, far = _raw_label(0.99, 0.0, 0.0, 0.0, CFG)
    assert 0 <= near < far <= 1


def test_hold_requires_a_persistent_change():
    raw = [None, "bull", "bull", "sideways", "bull", "sideways", "sideways", "sideways", "bull"]
    assert _hold(raw, 3) == [
        None, "bull", "bull", "bull", "bull", "bull", "bull", "sideways", "sideways",
    ]  # fmt: skip


def test_stress_is_entered_immediately():
    raw = ["bull", "market_stress", "bull", "bull", "bull"]
    assert _hold(raw, 3) == ["bull", "market_stress", "market_stress", "market_stress", "bull"]


def test_config_rejects_inconsistent_thresholds():
    with pytest.raises(ValidationError, match="low_vol_percentile"):
        RegimeConfig(low_vol_percentile=0.8, high_vol_percentile=0.7)
    with pytest.raises(ValidationError):
        RegimeConfig(colour="blue")


# ---------------------------------------------------------------------- detection


@pytest.fixture(scope="module")
def scenario() -> pd.Series:
    return _returns(
        [
            (400, 0.0004, 0.008),  # ordinary market: builds the volatility history
            (120, 0.0025, 0.006),  # steady rally
            (60, -0.012, 0.035),  # crash
            (200, 0.0000, 0.004),  # quiet, directionless
        ]
    )


def test_output_schema_and_warm_up(scenario):
    regimes = detect_regimes(scenario, CFG)
    assert list(regimes.columns) == REGIME_COLUMNS
    assert len(regimes) == len(scenario)
    warm_up = CFG.vol_window + CFG.min_history - 1
    assert regimes["regime"].iloc[: warm_up - 1].isna().all()  # no label before history exists
    assert regimes["regime"].iloc[warm_up:].notna().all()
    labelled = regimes.dropna(subset=["regime"])
    assert set(labelled["regime"]) <= set(REGIMES)
    assert labelled["confidence"].between(0, 1).all()
    assert labelled["volatility_level"].between(0, 1).all()
    assert labelled["trend_strength"].between(0, 1).all()
    assert (regimes["drawdown"] <= 1e-12).all()


def test_known_episodes_are_recognised(scenario):
    regimes = detect_regimes(scenario, CFG).set_index("trade_date")["regime"]
    rally = regimes.iloc[400 + 70 : 400 + 120]
    crash = regimes.iloc[520 + 15 : 520 + 60]
    quiet = regimes.iloc[580 + 120 : 580 + 200]
    assert (rally == "bull").mean() > 0.8
    assert (crash == "market_stress").mean() > 0.8
    assert quiet.isin(["low_volatility", "sideways"]).mean() > 0.8
    # Before the crash, stress is rare (a random walk can briefly qualify).
    assert (regimes.iloc[:500] == "market_stress").mean() < 0.1


def test_regimes_are_causal(scenario):
    """Truncating the future must not change any earlier label or measurement."""
    full = detect_regimes(scenario, CFG)
    for cut in (300, 450, 540, 700):
        prefix = detect_regimes(scenario.iloc[:cut], CFG)
        pd.testing.assert_frame_equal(prefix, full.iloc[:cut].reset_index(drop=True))


def test_future_returns_cannot_change_past_regimes(scenario):
    tampered = scenario.copy()
    tampered.iloc[500:] = tampered.iloc[500:] * 7 - 0.01
    before = detect_regimes(scenario, CFG).iloc[:500]
    after = detect_regimes(tampered, CFG).iloc[:500]
    pd.testing.assert_frame_equal(before, after)


def test_thresholds_are_configurable(scenario):
    strict = RegimeConfig(stress_drawdown=0.6, stress_vol_percentile=0.99)
    assert "market_stress" not in set(detect_regimes(scenario, strict)["regime"].dropna())
    jumpy = detect_regimes(scenario, RegimeConfig(min_duration=1))
    steady = detect_regimes(scenario, RegimeConfig(min_duration=10))
    assert regime_stability(jumpy)["switches"] > regime_stability(steady)["switches"]


def test_unsorted_input_is_rejected(scenario):
    with pytest.raises(ValueError, match="chronological"):
        detect_regimes(scenario.iloc[::-1], CFG)


# ---------------------------------------------------------------------- stability


def test_stability_report_hand_computed():
    labels = ["bull"] * 4 + ["sideways"] * 2 + ["bull"] * 2 + [None]
    frame = pd.DataFrame({"regime": labels})
    report = regime_stability(frame, periods_per_year=8)
    assert report["sessions"] == 8
    assert report["switches"] == 2
    assert report["switches_per_year"] == 2.0
    assert report["mean_run_sessions"] == pytest.approx(8 / 3, abs=0.05)
    assert report["min_run_sessions"] == 2
    assert report["share"] == {"bull": 0.75, "sideways": 0.25}
    assert report["persistence"] == pytest.approx(5 / 7, abs=1e-4)
    assert report["mean_run_by_regime"] == {"bull": 3.0, "sideways": 2.0}
    assert report["transition_matrix"]["sideways"] == {"bull": 0.5, "sideways": 0.5}
    assert regime_stability(pd.DataFrame({"regime": [None, None]})) == {"sessions": 0}


def test_detected_regimes_are_temporally_consistent(scenario):
    report = regime_stability(detect_regimes(scenario, CFG))
    assert report["persistence"] > 0.95
    assert report["mean_run_sessions"] > 15
    assert report["switches_per_year"] < 15
