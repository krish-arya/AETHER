"""Performance metrics (PRD 11.3). All inputs are daily; all outputs are plain floats.

Definitions
    total_return     E_end / E_start - 1
    cagr             (E_end / E_start) ** (1 / years) - 1, years = sessions / periods_per_year
    volatility       std(daily returns, ddof=1) * sqrt(periods_per_year)
    sharpe           mean(excess) / std(excess, ddof=1) * sqrt(periods_per_year)
    sortino          mean(excess) * periods_per_year / downside deviation, where the
                     downside deviation is sqrt(mean(min(excess, 0)^2) * periods_per_year)
    max_drawdown     min over time of E / running max(E) - 1   (a negative number)
    calmar           cagr / |max_drawdown|
    turnover         mean daily one-way turnover * periods_per_year
    win_rate         share of closing trades (sells, delistings) with positive realised
                     P&L after costs, on an average-cost basis
    hit_rate         share of sessions with a positive return
    beta, alpha      OLS of strategy excess returns on benchmark excess returns;
                     alpha is annualised
    tracking_error   std(strategy - benchmark, ddof=1) * sqrt(periods_per_year)
    information_ratio  annualised mean active return / tracking_error

``excess`` is the daily return minus the daily risk-free rate
``(1 + risk_free_rate) ** (1 / periods_per_year) - 1``.

A metric that is mathematically undefined (e.g. Sharpe of a constant series) is
returned as NaN, never as 0.
"""

import numpy as np
import pandas as pd

NAN = float("nan")


def _daily_risk_free(risk_free_rate: float, periods_per_year: int) -> float:
    return (1 + risk_free_rate) ** (1 / periods_per_year) - 1


def total_return(returns: pd.Series) -> float:
    return float(np.prod(1 + returns.to_numpy()) - 1)


def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(returns) == 0:
        return NAN
    growth = float(np.prod(1 + returns.to_numpy()))
    if growth <= 0:
        return -1.0
    return growth ** (periods_per_year / len(returns)) - 1


def volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return NAN
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    if len(returns) < 2:
        return NAN
    excess = returns - _daily_risk_free(risk_free_rate, periods_per_year)
    deviation = excess.std(ddof=1)
    if not deviation > 1e-15:
        return NAN
    return float(excess.mean() / deviation * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    if len(returns) < 2:
        return NAN
    excess = returns - _daily_risk_free(risk_free_rate, periods_per_year)
    downside = np.sqrt((np.minimum(excess.to_numpy(), 0.0) ** 2).mean() * periods_per_year)
    if not downside > 1e-15:
        return NAN
    return float(excess.mean() * periods_per_year / downside)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown from the running peak, starting from the capital before the first return."""
    wealth = (1 + returns).cumprod()
    peak = np.maximum(wealth.cummax(), 1.0)
    return wealth / peak - 1


def max_drawdown(returns: pd.Series) -> float:
    if len(returns) == 0:
        return NAN
    return float(drawdown_series(returns).min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    drawdown = max_drawdown(returns)
    if not drawdown < 0:
        return NAN
    return cagr(returns, periods_per_year) / abs(drawdown)


def relative_metrics(
    returns: pd.Series,
    benchmark: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Alpha, beta, tracking error and information ratio versus ``benchmark``."""
    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(aligned) < 3:
        return dict.fromkeys(("alpha", "beta", "tracking_error", "information_ratio"), NAN)
    daily_rf = _daily_risk_free(risk_free_rate, periods_per_year)
    strategy = aligned.iloc[:, 0].to_numpy() - daily_rf
    market = aligned.iloc[:, 1].to_numpy() - daily_rf
    variance = market.var(ddof=1)
    if variance > 1e-18:
        beta = float(np.cov(strategy, market, ddof=1)[0, 1] / variance)
        alpha = float((strategy.mean() - beta * market.mean()) * periods_per_year)
    else:
        beta = alpha = NAN
    active = strategy - market
    tracking = float(active.std(ddof=1) * np.sqrt(periods_per_year))
    information = float(active.mean() * periods_per_year / tracking) if tracking > 1e-15 else NAN
    return {
        "alpha": alpha,
        "beta": beta,
        "tracking_error": tracking,
        "information_ratio": information,
    }


def trade_win_rate(trades: pd.DataFrame) -> float:
    closing = trades[trades["side"].isin(["sell", "delist"])]
    if closing.empty:
        return NAN
    return float((closing["realized_pnl"] > 0).mean())


def compute_metrics(
    returns: pd.Series,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    benchmark: pd.Series | None = None,
    turnover: pd.Series | None = None,
    trades: pd.DataFrame | None = None,
    costs: pd.Series | None = None,
    initial_capital: float | None = None,
) -> dict[str, float]:
    metrics = {
        "sessions": float(len(returns)),
        "total_return": total_return(returns),
        "cagr": cagr(returns, periods_per_year),
        "volatility": volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns, periods_per_year),
        "hit_rate": float((returns > 0).mean()) if len(returns) else NAN,
    }
    if turnover is not None:
        metrics["turnover"] = float(turnover.mean() * periods_per_year) if len(turnover) else NAN
    if trades is not None:
        metrics["win_rate"] = trade_win_rate(trades)
        metrics["trades"] = float(len(trades))
    if costs is not None:
        metrics["total_costs"] = float(costs.sum())
        if initial_capital:
            metrics["costs_pct_of_capital"] = float(costs.sum() / initial_capital)
    if benchmark is not None:
        metrics.update(relative_metrics(returns, benchmark, risk_free_rate, periods_per_year))
    return metrics
