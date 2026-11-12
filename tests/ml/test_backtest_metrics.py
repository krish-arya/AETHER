"""Performance metrics against hand-computed values."""

import math

import numpy as np
import pandas as pd
import pytest

from ml.backtesting import metrics as m
from ml.backtesting.stats import bootstrap_sharpe

R = pd.Series([0.10, -0.05, 0.02])


def test_total_return_and_cagr():
    growth = 1.10 * 0.95 * 1.02
    assert m.total_return(R) == pytest.approx(growth - 1)
    assert m.cagr(R, periods_per_year=3) == pytest.approx(growth - 1)  # exactly one "year"
    assert m.cagr(R, periods_per_year=252) == pytest.approx(growth ** (252 / 3) - 1)
    assert m.cagr(pd.Series([-1.0, 0.5])) == -1.0  # wiped out


def test_volatility_and_sharpe_hand_computed():
    mean, std = R.mean(), R.std(ddof=1)
    assert m.volatility(R) == pytest.approx(std * math.sqrt(252))
    assert m.sharpe_ratio(R) == pytest.approx(mean / std * math.sqrt(252))
    # A risk-free rate lowers the numerator, not the denominator.
    daily_rf = 1.05 ** (1 / 252) - 1
    assert m.sharpe_ratio(R, risk_free_rate=0.05) == pytest.approx(
        (mean - daily_rf) / std * math.sqrt(252)
    )


def test_sortino_uses_downside_deviation_only():
    downside = math.sqrt((0.05**2) / 3 * 252)  # only the -5% day counts
    assert m.sortino_ratio(R) == pytest.approx(R.mean() * 252 / downside)
    assert math.isnan(m.sortino_ratio(pd.Series([0.01, 0.02, 0.03])))  # no downside at all


def test_drawdown_is_measured_from_the_running_peak_including_starting_capital():
    assert m.max_drawdown(R) == pytest.approx(-0.05)
    np.testing.assert_allclose(m.drawdown_series(R), [0.0, -0.05, 1.10 * 0.95 * 1.02 / 1.10 - 1])
    # A loss on the very first day is a drawdown from the initial capital.
    assert m.max_drawdown(pd.Series([-0.10, 0.05])) == pytest.approx(-0.10)
    # Two losses compound: 0.9 * 0.8 = 0.72 from a peak of 1.2 -> wealth 1.2 -> 0.864
    assert m.max_drawdown(pd.Series([0.2, -0.1, -0.2])) == pytest.approx(0.9 * 0.8 - 1)


def test_calmar():
    assert m.calmar_ratio(R, periods_per_year=3) == pytest.approx((1.10 * 0.95 * 1.02 - 1) / 0.05)
    assert math.isnan(m.calmar_ratio(pd.Series([0.01, 0.02])))  # no drawdown: undefined


def test_undefined_metrics_are_nan_not_zero():
    flat = pd.Series([0.01] * 10)
    assert math.isnan(m.sharpe_ratio(flat))
    assert math.isnan(m.sharpe_ratio(pd.Series([0.01])))
    assert math.isnan(m.volatility(pd.Series([0.01])))
    assert math.isnan(m.trade_win_rate(pd.DataFrame({"side": ["buy"], "realized_pnl": [np.nan]})))


def test_alpha_and_beta_of_a_levered_benchmark():
    rng = np.random.default_rng(0)
    benchmark = pd.Series(rng.normal(0.0004, 0.01, 500))
    strategy = 2.0 * benchmark + 0.0002  # beta 2, alpha 0.0002 per day
    relative = m.relative_metrics(strategy, benchmark)
    assert relative["beta"] == pytest.approx(2.0)
    assert relative["alpha"] == pytest.approx(0.0002 * 252)
    active = strategy - benchmark
    assert relative["tracking_error"] == pytest.approx(active.std(ddof=1) * math.sqrt(252))
    assert relative["information_ratio"] == pytest.approx(
        active.mean() * 252 / relative["tracking_error"]
    )


def test_alpha_accounts_for_the_risk_free_rate():
    rng = np.random.default_rng(1)
    benchmark = pd.Series(rng.normal(0.0004, 0.01, 500))
    daily_rf = 1.06 ** (1 / 252) - 1
    strategy = daily_rf + 1.5 * (benchmark - daily_rf)  # pure beta exposure, no alpha
    relative = m.relative_metrics(strategy, benchmark, risk_free_rate=0.06)
    assert relative["beta"] == pytest.approx(1.5)
    assert relative["alpha"] == pytest.approx(0.0, abs=1e-12)


def test_relative_metrics_align_on_dates():
    index = pd.bdate_range("2024-01-01", periods=6)
    benchmark = pd.Series([0.01, -0.02, 0.03, 0.00, 0.01, -0.01], index=index)
    strategy = benchmark.iloc[1:] * 3
    assert m.relative_metrics(strategy, benchmark)["beta"] == pytest.approx(3.0)


def test_win_rate_counts_closing_trades_only():
    trades = pd.DataFrame(
        {
            "side": ["buy", "sell", "sell", "delist", "buy", "sell"],
            "realized_pnl": [np.nan, 100.0, -50.0, 10.0, np.nan, 0.0],
        }
    )
    assert m.trade_win_rate(trades) == pytest.approx(2 / 4)  # break-even is not a win


def test_compute_metrics_bundle():
    trades = pd.DataFrame({"side": ["buy", "sell"], "realized_pnl": [np.nan, 5.0]})
    out = m.compute_metrics(
        R,
        benchmark=R * 0.5,
        turnover=pd.Series([0.5, 0.0, 0.1]),
        trades=trades,
        costs=pd.Series([100.0, 0.0, 20.0]),
        initial_capital=1_000_000,
    )
    assert out["sessions"] == 3
    assert out["hit_rate"] == pytest.approx(2 / 3)
    assert out["turnover"] == pytest.approx(0.2 * 252)
    assert out["win_rate"] == 1.0 and out["trades"] == 2
    assert out["total_costs"] == 120.0
    assert out["costs_pct_of_capital"] == pytest.approx(0.00012)
    assert out["beta"] == pytest.approx(2.0)


# ---------------------------------------------------------------------- bootstrap


def _returns(seed: int, drift: float, n: int = 1000) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(drift, 0.01, n), index=pd.bdate_range("2020-01-01", periods=n))


def test_bootstrap_is_deterministic_for_a_seed():
    returns = _returns(0, 0.0005)
    first = bootstrap_sharpe(returns, seed=3, samples=500)
    assert first == bootstrap_sharpe(returns, seed=3, samples=500)
    assert first != bootstrap_sharpe(returns, seed=4, samples=500)
    assert first["ci_low"] < first["estimate"] < first["ci_high"]
    assert first["estimate"] == pytest.approx(m.sharpe_ratio(returns))


def test_identical_series_have_no_sharpe_difference():
    returns = _returns(1, 0.0003)
    test = bootstrap_sharpe(returns, benchmark=returns.copy(), samples=500)
    assert test["estimate"] == 0.0
    assert test["ci_low"] == 0.0 and test["ci_high"] == 0.0
    assert test["significant"] is False


def test_noise_is_not_reported_as_an_edge():
    strategy, benchmark = _returns(2, 0.0003), _returns(3, 0.0003)  # same true Sharpe
    test = bootstrap_sharpe(strategy, benchmark=benchmark, samples=1000)
    assert test["ci_low"] < 0 < test["ci_high"]
    assert test["significant"] is False


def test_a_real_edge_is_detected():
    benchmark = _returns(4, 0.0002)
    strategy = benchmark + 0.0008  # same risk, a constant 8 bps/day more return
    test = bootstrap_sharpe(strategy, benchmark=benchmark, samples=1000)
    assert test["estimate"] > 1.0
    assert test["ci_low"] > 0
    assert test["significant"] is True
    assert test["p_value"] < 0.05


def test_bootstrap_requires_enough_observations():
    with pytest.raises(ValueError, match="not enough observations"):
        bootstrap_sharpe(_returns(5, 0.0, n=30), block_sessions=21)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_sharpe(_returns(5, 0.0), confidence=1.5)
