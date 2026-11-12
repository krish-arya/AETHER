"""Portfolio risk analytics (PRD 12.2). Reported measures -- none of them is enforced.

All measures are historical: they describe how the *current* weights would have fared
over a past window of asset returns. They say nothing certain about the future, and
historical VaR in particular understates risk when the window contains no crisis.

Conventions
    * Returns are simple daily returns; ``asset_returns`` is wide (dates x symbols).
    * VaR and CVaR are reported as positive loss fractions at the given confidence
      level (``level=0.95`` -> the loss exceeded on the worst 5% of days).
    * Weights are asset weights; the remainder is cash earning zero.
"""

import numpy as np
import pandas as pd

from ml.backtesting.metrics import drawdown_series
from ml.risk.limits import herfindahl

NAN = float("nan")


def portfolio_returns(weights: pd.Series, asset_returns: pd.DataFrame) -> pd.Series:
    """Daily returns of holding ``weights`` (rebalanced daily) through ``asset_returns``.

    Days on which a held asset has no return are dropped rather than treated as zero.
    """
    held = weights[weights.abs() > 0]
    if held.empty:
        return pd.Series(0.0, index=asset_returns.index)
    window = asset_returns.reindex(columns=held.index).dropna(how="any")
    return window @ held


def historical_var(returns: pd.Series, level: float = 0.95) -> float:
    """Loss not exceeded with probability ``level`` (empirical quantile)."""
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    if len(returns) == 0:
        return NAN
    return float(max(-np.quantile(returns.to_numpy(), 1 - level), 0.0))


def conditional_var(returns: pd.Series, level: float = 0.95) -> float:
    """Expected shortfall: mean loss on the days at or beyond the VaR threshold."""
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    if len(returns) == 0:
        return NAN
    values = returns.to_numpy()
    threshold = np.quantile(values, 1 - level)
    tail = values[values <= threshold]
    return float(max(-tail.mean(), 0.0))


def portfolio_volatility(
    weights: pd.Series, asset_returns: pd.DataFrame, periods_per_year: int = 252
) -> float:
    """Annualised volatility from the sample covariance: sqrt(w' Sigma w)."""
    held = weights[weights.abs() > 0]
    window = asset_returns.reindex(columns=held.index).dropna(how="any")
    if held.empty or len(window) < 2:
        return NAN
    covariance = np.atleast_2d(np.cov(window.to_numpy(), rowvar=False, ddof=1))
    vector = held.to_numpy()
    return float(np.sqrt(vector @ covariance @ vector * periods_per_year))


def correlation_matrix(asset_returns: pd.DataFrame) -> pd.DataFrame:
    return asset_returns.corr()


def sector_concentration(weights: pd.Series, sectors: pd.Series) -> dict:
    labels = sectors.reindex(weights.index).fillna("Unknown")
    by_sector = weights.groupby(labels).sum().sort_values(ascending=False)
    return {
        "by_sector": {str(k): float(v) for k, v in by_sector.items() if v > 0},
        "largest_sector": None if by_sector.empty else str(by_sector.index[0]),
        "largest_sector_weight": 0.0 if by_sector.empty else float(by_sector.iloc[0]),
        "herfindahl": herfindahl(weights.clip(lower=0)),
        "largest_position": float(weights.max()) if len(weights) else 0.0,
    }


def rolling_beta(portfolio: pd.Series, market: pd.Series, window: int = 60) -> pd.Series:
    aligned = pd.concat([portfolio, market], axis=1, join="inner").dropna()
    variance = aligned.iloc[:, 1].rolling(window, min_periods=window).var()
    covariance = aligned.iloc[:, 0].rolling(window, min_periods=window).cov(aligned.iloc[:, 1])
    return covariance / variance.where(variance > 0)


def stress_test(weights: pd.Series, scenarios: pd.DataFrame) -> pd.Series:
    """Portfolio return under each scenario.

    ``scenarios``: one row per scenario, one column per symbol, holding the asset
    return in that scenario (e.g. a historical crisis window or a GAN stress path).
    A held asset missing from a scenario makes that scenario's result NaN rather than
    silently assuming it was unaffected.
    """
    held = weights[weights.abs() > 0]
    if held.empty:
        return pd.Series(0.0, index=scenarios.index)
    return scenarios.reindex(columns=held.index).mul(held, axis=1).sum(axis=1, skipna=False)


def historical_stress_scenarios(asset_returns: pd.DataFrame, horizon: int = 21, worst: int = 5):
    """The ``worst`` non-overlapping ``horizon``-session windows for the equal-weight
    market, as cumulative asset returns -- real episodes to stress a portfolio with."""
    market = asset_returns.mean(axis=1)
    cumulative = (1 + market).rolling(horizon).apply(np.prod, raw=True) - 1
    scenarios, used = {}, []
    for end in cumulative.dropna().sort_values().index:
        position = asset_returns.index.get_loc(end)
        if any(abs(position - other) < horizon for other in used):
            continue
        window = asset_returns.iloc[position - horizon + 1 : position + 1]
        start = window.index[0]
        label = f"{start:%Y-%m-%d}..{end:%Y-%m-%d}"
        scenarios[label] = (1 + window).prod(min_count=horizon) - 1
        used.append(position)
        if len(scenarios) == worst:
            break
    return pd.DataFrame(scenarios).T


def risk_report(
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    *,
    sectors: pd.Series | None = None,
    market_returns: pd.Series | None = None,
    level: float = 0.95,
    periods_per_year: int = 252,
    stress_horizon: int = 21,
) -> dict:
    """All PRD 12.2 measures for one allocation over a historical return window."""
    returns = portfolio_returns(weights, asset_returns)
    report = {
        "observations": int(len(returns)),
        "var": historical_var(returns, level),
        "cvar": conditional_var(returns, level),
        "confidence_level": level,
        "volatility": portfolio_volatility(weights, asset_returns, periods_per_year),
        "max_drawdown": float(drawdown_series(returns).min()) if len(returns) else NAN,
        "invested": float(weights.sum()),
        "cash": float(1 - weights.sum()),
    }
    if sectors is not None:
        report["concentration"] = sector_concentration(weights, sectors)
    if market_returns is not None:
        aligned = pd.concat([returns, market_returns], axis=1, join="inner").dropna()
        variance = aligned.iloc[:, 1].var()
        report["beta"] = (
            float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / variance) if variance > 0 else NAN
        )
    if len(asset_returns) >= 2 * stress_horizon:
        scenarios = historical_stress_scenarios(asset_returns, stress_horizon)
        losses = stress_test(weights, scenarios)
        report["stress"] = {
            "horizon_sessions": stress_horizon,
            "scenarios": {str(k): (None if pd.isna(v) else float(v)) for k, v in losses.items()},
            "worst": None if losses.dropna().empty else float(losses.min()),
        }
    return report
