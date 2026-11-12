"""Risk engine: limits, enforcement, analytics -- against hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from ml.backtesting.costs import CostModel
from ml.backtesting.simulator import SimulationConfig, simulate
from ml.risk import analytics
from ml.risk.fast import FastRiskProjector
from ml.risk.limits import RiskLimits, check_limits, enforce, herfindahl, one_way_turnover
from ml.risk.strategy import RiskManagedStrategy

from .backtest_helpers import Scripted, make_market

SECTORS = pd.Series({"A": "Bank", "B": "Bank", "C": "IT", "D": "IT", "E": "Energy"})
LOOSE = RiskLimits(
    max_weight=1.0, max_sector_weight=1.0, min_cash=0.0, max_drawdown=None, max_turnover=None
)


def w(**weights: float) -> pd.Series:
    return pd.Series(weights, dtype="float64")


def limits(**overrides) -> RiskLimits:
    return LOOSE.model_copy(update=overrides)


# ------------------------------------------------------------------ single limits


def test_compliant_allocation_passes_untouched():
    target = w(A=0.2, B=0.2, C=0.2)
    decision = enforce(target, limits=RiskLimits(), sectors=SECTORS, previous=target)
    assert not decision.rejected and not decision.modified
    assert decision.raw_violations == []
    pd.testing.assert_series_equal(decision.weights.reindex(target.index), target)


def test_asset_cap_moves_the_excess_to_cash():
    decision = enforce(w(A=0.6, C=0.3), limits=limits(max_weight=0.25))
    assert decision.weights.to_dict() == {"A": 0.25, "C": 0.25}
    assert decision.applied == ["max_weight"]
    assert {v.subject for v in decision.raw_violations} == {"A", "C"}
    assert decision.weights.sum() == pytest.approx(0.5)  # not redistributed


def test_sector_cap_scales_the_sector_proportionally():
    decision = enforce(
        w(A=0.4, B=0.2, C=0.3), limits=limits(max_sector_weight=0.3), sectors=SECTORS
    )
    assert decision.weights["A"] == pytest.approx(0.2)  # 0.4 * 0.3 / 0.6
    assert decision.weights["B"] == pytest.approx(0.1)
    assert decision.weights["C"] == pytest.approx(0.3)  # IT is exactly at the cap
    assert decision.applied == ["max_sector_weight"]


def test_cash_reserve_scales_everything():
    decision = enforce(w(A=0.5, C=0.5), limits=limits(min_cash=0.1))
    assert decision.weights.to_dict() == pytest.approx({"A": 0.45, "C": 0.45})
    assert decision.applied == ["min_cash"]


def test_illiquid_assets_are_removed():
    adv = pd.Series({"A": 5e7, "C": 1e5})
    decision = enforce(w(A=0.5, C=0.5), limits=limits(min_adv=1e6), adv=adv)
    assert decision.weights.to_dict() == {"A": 0.5, "C": 0.0}
    assert decision.applied == ["min_adv"]
    # Unknown liquidity is treated as illiquid, not as fine.
    unknown = enforce(w(A=0.5, Z=0.5), limits=limits(min_adv=1e6), adv=adv)
    assert unknown.weights["Z"] == 0.0


def test_drawdown_breach_caps_exposure():
    rules = limits(max_drawdown=0.2, drawdown_exposure_cap=0.4)
    calm = enforce(w(A=0.5, C=0.5), limits=rules, drawdown=-0.10)
    assert calm.weights.sum() == pytest.approx(1.0) and not calm.modified
    breached = enforce(w(A=0.5, C=0.5), limits=rules, drawdown=-0.25)
    assert breached.weights.to_dict() == pytest.approx({"A": 0.2, "C": 0.2})
    assert breached.applied == ["drawdown_exposure_cap"]


def test_negative_weights_are_removed_and_leverage_is_flagged():
    decision = enforce(w(A=0.9, C=0.6, E=-0.3), limits=limits(min_cash=0.0))
    assert (decision.weights >= 0).all()
    assert decision.weights.sum() <= 1 + 1e-9
    assert {"no_short", "no_leverage"} <= {v.limit for v in decision.raw_violations}


def test_concentration_limit():
    rules = limits(max_concentration_hhi=0.3)
    decision = enforce(w(A=0.7, B=0.1, C=0.1, D=0.1), limits=rules)
    assert herfindahl(w(A=0.7, B=0.1, C=0.1, D=0.1)) == pytest.approx(0.52)
    assert herfindahl(decision.weights) <= 0.3 + 1e-6
    assert decision.weights.sum() == pytest.approx(1.0)
    assert decision.applied == ["max_concentration_hhi"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_allocation_is_rejected(bad):
    decision = enforce(w(A=0.5, C=bad), limits=RiskLimits())
    assert decision.rejected and decision.weights is None
    assert "NaN or infinite" in decision.reason


# ---------------------------------------------------------------------- turnover


def test_turnover_definition():
    assert one_way_turnover(w(A=0.5, B=0.5), w(A=0.5, B=0.5)) == 0.0
    assert one_way_turnover(w(A=1.0), w(B=1.0)) == pytest.approx(1.0)  # full rotation
    assert one_way_turnover(w(A=0.4), w()) == pytest.approx(0.4)  # 40% out of cash


def test_turnover_limit_moves_part_of_the_way():
    previous = w(A=0.5, B=0.5)
    decision = enforce(w(C=0.5, D=0.5), limits=limits(max_turnover=0.25), previous=previous)
    # Full rotation needs turnover 1.0; a 0.25 budget moves a quarter of the way.
    assert decision.weights.to_dict() == pytest.approx(
        {"A": 0.375, "B": 0.375, "C": 0.125, "D": 0.125}
    )
    assert decision.turnover == pytest.approx(0.25)
    assert decision.applied == ["max_turnover"]


def test_compliance_takes_priority_over_the_turnover_budget():
    """Holdings that drifted over a hard limit are cut even if that uses the budget."""
    previous = w(A=0.9, C=0.1)
    rules = limits(max_weight=0.3, max_turnover=0.05)
    decision = enforce(w(A=0.3, C=0.3, E=0.3), limits=rules, previous=previous)
    assert not decision.rejected
    assert decision.weights["A"] == pytest.approx(0.3)  # the cap is honoured
    assert decision.weights["E"] == pytest.approx(0.0)  # no budget left for new buys
    assert "max_turnover_exceeded_for_compliance" in decision.applied
    assert check_limits(decision.weights, limits=limits(max_weight=0.3)) == []


# --------------------------------------------------------- invariants (property)


def _random_case(rng: np.random.Generator, symbols: list[str]):
    target = pd.Series(rng.dirichlet(np.ones(len(symbols)) * 0.5) * rng.uniform(0.3, 1.3), symbols)
    target[rng.random(len(symbols)) < 0.1] *= -0.5  # occasional shorts
    previous = pd.Series(rng.dirichlet(np.ones(len(symbols))) * rng.uniform(0.0, 1.0), symbols)
    adv = pd.Series(rng.lognormal(16, 2, len(symbols)), symbols)
    drawdown = -float(rng.uniform(0, 0.4))
    return target, previous, adv, drawdown


def test_enforced_allocations_always_satisfy_every_hard_limit():
    rng = np.random.default_rng(0)
    symbols = [f"S{i:02d}" for i in range(12)]
    sectors = pd.Series({s: f"sec{i % 4}" for i, s in enumerate(symbols)})
    rules = RiskLimits(
        max_weight=0.2, max_sector_weight=0.35, min_cash=0.05, min_adv=5e6,
        max_drawdown=0.25, drawdown_exposure_cap=0.5, max_turnover=0.3,
    )  # fmt: skip
    for _ in range(400):
        target, previous, adv, drawdown = _random_case(rng, symbols)
        decision = enforce(
            target, limits=rules, previous=previous, sectors=sectors, adv=adv, drawdown=drawdown
        )
        assert not decision.rejected
        weights = decision.weights
        assert (weights >= 0).all() and weights.sum() <= 1 + 1e-9
        hard = check_limits(weights, limits=rules, sectors=sectors, adv=adv, drawdown=drawdown)
        assert hard == [], [v.as_dict() for v in hard]


def test_fast_projector_matches_the_reference_implementation():
    rng = np.random.default_rng(1)
    symbols = [f"S{i:02d}" for i in range(12)]
    sector_ids = np.arange(12) % 4
    sectors = pd.Series({s: f"sec{i}" for s, i in zip(symbols, sector_ids, strict=True)})
    rules = RiskLimits(
        max_weight=0.2, max_sector_weight=0.35, min_cash=0.05, min_adv=5e6,
        max_drawdown=0.25, drawdown_exposure_cap=0.5, max_turnover=0.3,
    )  # fmt: skip
    fast = FastRiskProjector(rules, sector_ids)
    for _ in range(300):
        target, previous, adv, drawdown = _random_case(rng, symbols)
        reference = enforce(
            target, limits=rules, previous=previous, sectors=sectors, adv=adv, drawdown=drawdown
        ).weights.reindex(symbols)
        eligible = (adv >= rules.min_adv).to_numpy()
        projected = fast.project(target.to_numpy(), previous.to_numpy(), eligible, drawdown)
        np.testing.assert_allclose(projected, reference.to_numpy(), atol=1e-12)
    with pytest.raises(ValueError):
        FastRiskProjector(limits(max_concentration_hhi=0.3), sector_ids)


# ----------------------------------------------------------------- in a backtest


def test_risk_managed_strategy_constrains_and_logs_every_decision():
    market = make_market(
        {"A": [10, 10, 10, 10, 10], "B": [20, 20, 20, 20, 20], "C": [5, 5, 5, 5, 5]}
    )
    greedy = Scripted({"A": 1.0}, {"A": 1.0}, {"B": float("nan")}, None)
    managed = RiskManagedStrategy(
        greedy, RiskLimits(max_weight=0.4, min_cash=0.1, max_turnover=None)
    )
    result = simulate(
        market, managed, start=market.sessions[0], end=market.sessions[-1],
        decision_dates=market.sessions, cost_model=CostModel.zero(),
        config=SimulationConfig(participation_limit=None),
    )  # fmt: skip
    assert result.weights["A"].max() == pytest.approx(0.4)  # the model asked for 100%
    assert result.weights["CASH"].min() >= 0.1 - 1e-9

    log = managed.risk_log()
    assert len(log) == 3  # the "hold" decision proposes nothing, so there is nothing to check
    assert log.loc[0, "applied"] == ["max_weight"]
    assert log.loc[0, "raw_violations"][0]["limit"] == "max_weight"
    assert bool(log.loc[2, "rejected"]) is True  # NaN allocation: rejected, holdings kept
    assert result.weights["A"].iloc[-1] == pytest.approx(0.4)


# --------------------------------------------------------------------- analytics


def test_var_and_cvar_hand_computed():
    returns = pd.Series(np.arange(-10, 10) / 100)  # -10% .. +9%, 20 observations
    # 5% quantile of 20 evenly spaced points (linear interpolation) = -9.05%
    assert analytics.historical_var(returns, 0.95) == pytest.approx(0.0905)
    assert analytics.conditional_var(returns, 0.95) == pytest.approx(0.10)  # only -10% is beyond
    assert analytics.conditional_var(returns, 0.80) >= analytics.historical_var(returns, 0.80)
    assert analytics.historical_var(pd.Series([0.01, 0.02, 0.03]), 0.95) == 0.0  # no losses
    with pytest.raises(ValueError):
        analytics.historical_var(returns, 1.5)


def test_portfolio_returns_and_volatility():
    asset_returns = pd.DataFrame({"A": [0.01, -0.02, 0.03], "B": [0.00, 0.01, -0.01]})
    weights = w(A=0.5, B=0.25)
    returns = analytics.portfolio_returns(weights, asset_returns)
    np.testing.assert_allclose(returns, [0.005, -0.0075, 0.0125])
    expected = returns.std(ddof=1) * np.sqrt(252)
    assert analytics.portfolio_volatility(weights, asset_returns) == pytest.approx(expected)
    assert analytics.portfolio_returns(w(), asset_returns).eq(0).all()


def test_sector_concentration_and_stress_test():
    weights = w(A=0.3, B=0.2, C=0.1)
    report = analytics.sector_concentration(weights, SECTORS)
    assert report["by_sector"] == pytest.approx({"Bank": 0.5, "IT": 0.1})
    assert report["largest_sector"] == "Bank"
    assert report["herfindahl"] == pytest.approx(0.5**2 + (1 / 3) ** 2 + (1 / 6) ** 2)

    scenarios = pd.DataFrame(
        {"A": [-0.3, -0.1], "B": [-0.2, 0.0], "C": [-0.1, np.nan]}, index=["crash", "partial"]
    )
    losses = analytics.stress_test(weights, scenarios)
    assert losses["crash"] == pytest.approx(-0.14)  # 0.3*-0.3 + 0.2*-0.2 + 0.1*-0.1
    assert np.isnan(losses["partial"])  # a held asset is missing: not assumed unaffected


def test_risk_report_on_a_market_with_a_crash():
    rng = np.random.default_rng(3)
    calm = rng.normal(0.0005, 0.01, (300, 3))
    crash = rng.normal(-0.02, 0.03, (21, 3))
    asset_returns = pd.DataFrame(
        np.vstack([calm, crash, calm[:100]]),
        columns=["A", "B", "C"],
        index=pd.bdate_range("2020-01-01", periods=421),
    )
    weights = w(A=0.3, B=0.3, C=0.3)
    report = analytics.risk_report(
        weights, asset_returns, sectors=SECTORS, market_returns=asset_returns.mean(axis=1)
    )
    assert report["cvar"] >= report["var"] > 0
    assert report["cash"] == pytest.approx(0.1)
    assert report["beta"] == pytest.approx(0.9, abs=1e-9)  # 90% invested in the market itself
    assert report["max_drawdown"] < -0.2
    worst = min(report["stress"]["scenarios"], key=report["stress"]["scenarios"].get)
    start = pd.Timestamp(worst.split("..")[0])
    assert asset_returns.index[285] <= start <= asset_returns.index[305]  # the crash window
    assert report["stress"]["worst"] == pytest.approx(report["stress"]["scenarios"][worst])
    assert len(report["stress"]["scenarios"]) == 5
