"""Baseline portfolio strategies (PRD 9.6 "Baseline", 11.3 benchmarks).

These are deliberately simple, rule-based reference strategies. They are **not** AI
recommendations; their job is to give the RL agent (Phase 5) honest competition.

Interface
    A strategy receives a :class:`StrategyContext` holding only information available
    at the close of the decision date and returns target weights for the next
    executable session:

    * ``pd.Series`` indexed by symbol, values >= 0, sum <= 1 (the remainder is cash);
    * an empty Series means "all cash";
    * ``None`` means "leave the current holdings untouched".

    ``fit`` is called once with the training window before a walk-forward test window.
    The rule-based baselines have nothing to learn and ignore it.

Adding a strategy: subclass :class:`Strategy` and register it in ``STRATEGIES``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


class StrategyError(ValueError):
    """A strategy returned weights that cannot be traded."""


@dataclass(frozen=True)
class StrategyContext:
    """Everything a strategy may look at when deciding at the close of ``date``."""

    date: pd.Timestamp
    # Symbols listed on ``date`` that also traded on ``date``.
    universe: list[str]
    # Adjusted closes for every session up to and including ``date`` (wide).
    prices: pd.DataFrame
    # Feature cross-section for ``date`` (indexed by symbol), or None.
    features: pd.DataFrame | None
    # End-of-day portfolio weights, including the "CASH" entry.
    current_weights: pd.Series
    # Market regime label for ``date`` (ml.features.regime), if the dataset has one.
    regime: str | None = None
    # Rupee value traded per symbol for every session up to ``date`` (liquidity).
    traded_value: pd.DataFrame | None = None
    # The simulated portfolio's drawdown from its peak at ``date`` (<= 0).
    drawdown: float = 0.0


class Strategy(ABC):
    name: str = "strategy"
    # True for strategies that cannot produce weights before ``fit`` (learned models).
    requires_fit: bool = False

    def fit(self, train: StrategyContext) -> None:  # noqa: B027 - optional hook
        """Learn from the training window. No-op for rule-based strategies."""

    @abstractmethod
    def target_weights(self, ctx: StrategyContext) -> pd.Series | None: ...


def _equal_weights(symbols: list[str]) -> pd.Series:
    if not symbols:
        return pd.Series(dtype="float64")
    return pd.Series(1.0 / len(symbols), index=symbols)


class BuyAndHold(Strategy):
    """Buy the universe in equal weights once, then never trade.

    Weights drift with prices. Proceeds from a delisting stay in cash.
    """

    name = "buy_and_hold"

    def __init__(self) -> None:
        self._invested = False

    def target_weights(self, ctx: StrategyContext) -> pd.Series | None:
        if self._invested:
            return None
        self._invested = True
        return _equal_weights(ctx.universe)


class EqualWeight(Strategy):
    """Rebalance to 1/N across the point-in-time universe at every decision date."""

    name = "equal_weight"

    def target_weights(self, ctx: StrategyContext) -> pd.Series:
        return _equal_weights(ctx.universe)


class Momentum(Strategy):
    """Cross-sectional momentum: hold the ``top_k`` assets by trailing return.

    Score = return over ``lookback`` sessions ending ``skip`` sessions before the
    decision date (the skip avoids short-term reversal). Assets without enough history
    are not eligible. With ``absolute=True`` only assets with a positive score are held
    and the unallocated share stays in cash.
    """

    name = "momentum"

    def __init__(
        self, lookback: int = 126, skip: int = 21, top_k: int = 5, absolute: bool = False
    ) -> None:
        if lookback < 1 or skip < 0 or top_k < 1:
            raise ValueError("lookback >= 1, skip >= 0 and top_k >= 1 are required")
        self.lookback, self.skip, self.top_k, self.absolute = lookback, skip, top_k, absolute

    def target_weights(self, ctx: StrategyContext) -> pd.Series:
        prices = ctx.prices[ctx.universe]
        if len(prices) < self.lookback + self.skip + 1:
            return pd.Series(dtype="float64")
        end = prices.iloc[-1 - self.skip]
        start = prices.iloc[-1 - self.skip - self.lookback]
        score = (end / start - 1).dropna()
        if self.absolute:
            score = score[score > 0]
        winners = score.sort_index().nlargest(self.top_k)
        if winners.empty:
            return pd.Series(dtype="float64")
        weight = 1.0 / self.top_k if self.absolute else 1.0 / len(winners)
        return pd.Series(weight, index=list(winners.index))


class MovingAverageTrend(Strategy):
    """Trend filter: hold an asset only while its close is above its ``window`` SMA.

    Each asset has a fixed 1/N slot (N = universe size); slots of assets in a
    downtrend, or without ``window`` sessions of history, stay in cash.
    """

    name = "moving_average"

    def __init__(self, window: int = 200) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        self.window = window

    def target_weights(self, ctx: StrategyContext) -> pd.Series:
        if not ctx.universe or len(ctx.prices) < self.window:
            return pd.Series(dtype="float64")
        recent = ctx.prices[ctx.universe].iloc[-self.window :]
        complete = recent.notna().all()
        uptrend = complete & (recent.iloc[-1] > recent.mean())
        selected = [symbol for symbol in ctx.universe if uptrend[symbol]]
        return pd.Series(1.0 / len(ctx.universe), index=selected, dtype="float64")


def project_capped_simplex(vector: np.ndarray, cap: float) -> np.ndarray:
    """Euclidean projection onto {w : sum(w) = 1, 0 <= w <= cap}.

    The projection is ``clip(v - shift, 0, cap)`` for the shift at which the weights
    sum to one. That sum is piecewise linear and decreasing in the shift, with kinks at
    ``v_i`` and ``v_i - cap``, so the exact shift is found by evaluating the sum at the
    kinks and interpolating inside the bracketing segment.
    """
    n = len(vector)
    cap = max(cap, 1.0 / n)  # otherwise the set is empty
    kinks = np.sort(np.concatenate([vector, vector - cap]))
    totals = np.clip(vector[None, :] - kinks[:, None], 0.0, cap).sum(axis=1)
    upper = int(np.searchsorted(-totals, -1.0, side="left"))  # first kink with total <= 1
    if upper == 0:
        shift = kinks[0]
    else:
        lower = upper - 1
        fraction = (totals[lower] - 1.0) / (totals[lower] - totals[upper])
        shift = kinks[lower] + fraction * (kinks[upper] - kinks[lower])
    weights = np.clip(vector - shift, 0.0, cap)
    return weights / weights.sum()


class MeanVariance(Strategy):
    """Long-only Markowitz mean-variance optimisation on trailing returns.

    maximise   mu'w - (risk_aversion / 2) w'Sigma w
    subject to sum(w) = 1, 0 <= w <= max_weight

    ``mu`` and ``Sigma`` are the annualised sample mean and covariance of the last
    ``lookback`` daily returns; ``Sigma`` is shrunk towards its diagonal by
    ``shrinkage``. ``objective="min_variance"`` drops the (very noisy) mean term.
    Solved by projected gradient ascent; deterministic.
    """

    name = "mean_variance"

    def __init__(
        self,
        lookback: int = 252,
        objective: str = "mean_variance",
        risk_aversion: float = 5.0,
        shrinkage: float = 0.2,
        max_weight: float = 0.35,
        periods_per_year: int = 252,
        iterations: int = 500,
    ) -> None:
        if objective not in ("mean_variance", "min_variance"):
            raise ValueError("objective must be 'mean_variance' or 'min_variance'")
        if lookback < 20 or risk_aversion <= 0 or not 0 <= shrinkage <= 1 or max_weight <= 0:
            raise ValueError("invalid mean-variance parameters")
        self.lookback, self.objective = lookback, objective
        self.risk_aversion, self.shrinkage, self.max_weight = risk_aversion, shrinkage, max_weight
        self.periods_per_year, self.iterations = periods_per_year, iterations

    def target_weights(self, ctx: StrategyContext) -> pd.Series:
        if len(ctx.prices) < self.lookback + 1:
            return pd.Series(dtype="float64")
        window = ctx.prices[ctx.universe].iloc[-(self.lookback + 1) :]
        window = window.loc[:, window.notna().all()]
        if window.shape[1] == 0:
            return pd.Series(dtype="float64")
        if window.shape[1] == 1:
            return pd.Series(1.0, index=list(window.columns))

        returns = window.pct_change(fill_method=None).iloc[1:].to_numpy()
        mu = returns.mean(axis=0) * self.periods_per_year
        cov = np.cov(returns, rowvar=False, ddof=1) * self.periods_per_year
        cov = (1 - self.shrinkage) * cov + self.shrinkage * np.diag(np.diag(cov))
        if self.objective == "min_variance":
            mu = np.zeros_like(mu)

        n = len(mu)
        weights = np.full(n, 1.0 / n)
        step = 1.0 / (self.risk_aversion * np.linalg.eigvalsh(cov)[-1])
        for _ in range(self.iterations):
            gradient = mu - self.risk_aversion * cov @ weights
            updated = project_capped_simplex(weights + step * gradient, self.max_weight)
            if np.abs(updated - weights).max() < 1e-10:
                weights = updated
                break
            weights = updated
        return pd.Series(weights, index=list(window.columns))


STRATEGIES: dict[str, type[Strategy]] = {
    cls.name: cls for cls in (BuyAndHold, EqualWeight, Momentum, MovingAverageTrend, MeanVariance)
}


def build_strategy(name: str, params: dict | None = None) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Available: {sorted(STRATEGIES)}")
    return STRATEGIES[name](**(params or {}))
