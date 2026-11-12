"""Baseline strategies against hand-computed expectations."""

import numpy as np
import pandas as pd
import pytest

from ml.backtesting.strategies import (
    STRATEGIES,
    BuyAndHold,
    EqualWeight,
    MeanVariance,
    Momentum,
    MovingAverageTrend,
    StrategyContext,
    build_strategy,
    project_capped_simplex,
)


def ctx(prices: dict[str, list[float]], universe: list[str] | None = None) -> StrategyContext:
    frame = pd.DataFrame(
        prices, index=pd.bdate_range("2024-01-01", periods=len(next(iter(prices.values()))))
    )
    return StrategyContext(
        date=frame.index[-1],
        universe=universe if universe is not None else list(frame.columns),
        prices=frame,
        features=None,
        current_weights=pd.Series({"CASH": 1.0}),
    )


def test_registry_contains_every_prd_baseline():
    assert set(STRATEGIES) == {
        "buy_and_hold",
        "equal_weight",
        "momentum",
        "moving_average",
        "mean_variance",
    }
    assert isinstance(build_strategy("momentum", {"top_k": 2}), Momentum)
    with pytest.raises(ValueError, match="Unknown strategy"):
        build_strategy("crystal_ball")


def test_equal_weight_uses_only_the_tradable_universe():
    weights = EqualWeight().target_weights(ctx({"A": [1, 2], "B": [1, 2], "C": [1, 2]}, ["A", "C"]))
    assert weights.to_dict() == {"A": 0.5, "C": 0.5}
    assert EqualWeight().target_weights(ctx({"A": [1, 2]}, [])).empty


def test_buy_and_hold_invests_once_then_holds():
    strategy = BuyAndHold()
    context = ctx({"A": [1, 2], "B": [1, 2]})
    assert strategy.target_weights(context).to_dict() == {"A": 0.5, "B": 0.5}
    assert strategy.target_weights(context) is None
    assert strategy.target_weights(context) is None


def test_momentum_ranks_on_the_lookback_window_and_skips_recent_sessions():
    # lookback 2, skip 1: score = price[-2] / price[-4] - 1 ; the last bar is ignored.
    prices = {
        "A": [10, 10, 12, 12, 1],  # +20%, then collapses in the skipped bar
        "B": [10, 10, 10, 11, 99],  # +10%, then spikes in the skipped bar
        "C": [10, 10, 10, 9, 10],  # -10%
    }
    weights = Momentum(lookback=2, skip=1, top_k=2).target_weights(ctx(prices))
    assert weights.to_dict() == {"A": 0.5, "B": 0.5}

    top_one = Momentum(lookback=2, skip=1, top_k=1).target_weights(ctx(prices))
    assert top_one.to_dict() == {"A": 1.0}


def test_absolute_momentum_leaves_losers_in_cash():
    prices = {"A": [10, 10, 12, 12, 12], "B": [10, 10, 10, 9, 9], "C": [10, 10, 10, 8, 8]}
    weights = Momentum(lookback=2, skip=1, top_k=2, absolute=True).target_weights(ctx(prices))
    assert weights.to_dict() == {"A": 0.5}  # one of two slots filled; the rest is cash


def test_momentum_needs_enough_history_per_asset():
    assert Momentum(lookback=5, skip=1, top_k=1).target_weights(ctx({"A": [1, 2, 3]})).empty
    nan = float("nan")
    prices = {"A": [10, 10, 12, 12, 12], "NEW": [nan, nan, nan, 5, 50]}
    weights = Momentum(lookback=2, skip=1, top_k=2).target_weights(ctx(prices))
    assert weights.to_dict() == {"A": 1.0}  # NEW has no price at the start of the window


def test_moving_average_trend_filter():
    prices = {
        "UP": [10, 10, 10, 13],  # 13 > mean(10,10,13)=11
        "DOWN": [10, 10, 10, 7],  # 7 < 9
        "FLAT": [10, 10, 10, 10],  # not strictly above its average
    }
    weights = MovingAverageTrend(window=3).target_weights(ctx(prices))
    assert weights.to_dict() == {"UP": pytest.approx(1 / 3)}  # two of three slots stay in cash


def test_moving_average_needs_a_full_window():
    nan = float("nan")
    weights = MovingAverageTrend(window=3).target_weights(
        ctx({"A": [10, 10, 10, 13], "NEW": [nan, nan, 5, 9]})
    )
    assert weights.to_dict() == {"A": 0.5}
    assert MovingAverageTrend(window=10).target_weights(ctx({"A": [1, 2, 3]})).empty


# ------------------------------------------------------------------ mean-variance


def _reference_projection(vector: np.ndarray, cap: float) -> np.ndarray:
    low, high = vector.min() - cap, vector.max()
    for _ in range(200):
        mid = (low + high) / 2
        if np.clip(vector - mid, 0, cap).sum() > 1:
            low = mid
        else:
            high = mid
    return np.clip(vector - (low + high) / 2, 0, cap)


@pytest.mark.parametrize("cap", [1.0, 0.5, 0.25, 0.1])
def test_capped_simplex_projection_matches_bisection_reference(cap):
    rng = np.random.default_rng(1)
    for _ in range(200):
        vector = rng.normal(0, 1, 12)
        weights = project_capped_simplex(vector, cap)
        assert weights.sum() == pytest.approx(1.0, abs=1e-12)
        assert weights.min() >= 0 and weights.max() <= cap + 1e-12
        np.testing.assert_allclose(weights, _reference_projection(vector, cap), atol=1e-9)


def test_projection_is_identity_inside_the_feasible_set():
    inside = np.array([0.2, 0.3, 0.5])
    np.testing.assert_allclose(project_capped_simplex(inside, 1.0), inside, atol=1e-12)
    np.testing.assert_allclose(project_capped_simplex(np.zeros(4), 0.01), 0.25)  # cap too tight


def _random_prices(seed: int, vols: list[float], n: int = 300) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    return {
        f"S{i}": list(100 * np.exp(np.cumsum(rng.normal(0, vol, n)))) for i, vol in enumerate(vols)
    }


def test_min_variance_matches_the_two_asset_closed_form():
    context = ctx(_random_prices(4, [0.01, 0.02]))
    strategy = MeanVariance(lookback=252, objective="min_variance", shrinkage=0.0, max_weight=1.0)
    weights = strategy.target_weights(context)

    returns = context.prices.iloc[-253:].pct_change().iloc[1:]
    cov = returns.cov().to_numpy()
    expected = (cov[1, 1] - cov[0, 1]) / (cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    assert weights["S0"] == pytest.approx(expected, abs=1e-6)
    assert weights["S0"] > 0.7  # the calmer asset dominates
    assert weights.sum() == pytest.approx(1.0)


def test_mean_variance_respects_bounds_and_is_deterministic():
    context = ctx(_random_prices(5, [0.01, 0.015, 0.02, 0.012, 0.03]))
    strategy = MeanVariance(lookback=252, max_weight=0.35)
    first, second = strategy.target_weights(context), strategy.target_weights(context)
    pd.testing.assert_series_equal(first, second)
    assert first.sum() == pytest.approx(1.0)
    assert first.min() >= 0 and first.max() <= 0.35 + 1e-9


def test_mean_variance_solution_is_a_constrained_optimum():
    """No feasible two-asset reallocation may improve the objective."""
    context = ctx(_random_prices(6, [0.01, 0.02, 0.015, 0.025]))
    strategy = MeanVariance(lookback=252, risk_aversion=5.0, shrinkage=0.2, max_weight=0.6)
    weights = strategy.target_weights(context).to_numpy()

    returns = context.prices.iloc[-253:].pct_change().iloc[1:].to_numpy()
    mu = returns.mean(axis=0) * 252
    cov = np.cov(returns, rowvar=False, ddof=1) * 252
    cov = 0.8 * cov + 0.2 * np.diag(np.diag(cov))

    def objective(w):
        return mu @ w - 2.5 * w @ cov @ w

    best = objective(weights)
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            step = min(1e-3, weights[i], 0.6 - weights[j])
            if step <= 0:
                continue
            moved = weights.copy()
            moved[i] -= step
            moved[j] += step
            assert objective(moved) <= best + 1e-9


def test_mean_variance_handles_short_history_and_single_assets():
    assert MeanVariance(lookback=252).target_weights(ctx({"A": [1, 2, 3]})).empty
    nan = float("nan")
    prices = _random_prices(7, [0.01], n=300)
    prices["NEW"] = [nan] * 200 + list(np.linspace(10, 20, 100))
    weights = MeanVariance(lookback=252).target_weights(ctx(prices))
    assert weights.to_dict() == {"S0": 1.0}  # NEW lacks a full window


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        Momentum(lookback=0)
    with pytest.raises(ValueError):
        MovingAverageTrend(window=1)
    with pytest.raises(ValueError):
        MeanVariance(objective="max_luck")
    with pytest.raises(ValueError):
        MeanVariance(shrinkage=1.5)
