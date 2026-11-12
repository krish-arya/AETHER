"""Risk-managed strategy wrapper: the only route from a model to the simulator.

Wraps any :class:`~ml.backtesting.strategies.Strategy` (in particular the RL policy)
and passes every allocation it proposes through :func:`ml.risk.limits.enforce`. The
simulator therefore only ever sees compliant weights, and every decision -- including
what the model originally asked for and which limits changed it -- is kept in
``decisions`` as the risk log (PRD 12.3 "risk failures must be logged").

A rejected decision becomes "hold": the portfolio is left as it is.
"""

import pandas as pd

from ml.backtesting.strategies import Strategy, StrategyContext
from ml.risk.limits import RiskLimits, enforce

CASH = "CASH"


def average_traded_value(ctx: StrategyContext, window: int) -> pd.Series | None:
    """Mean rupee value traded per symbol over the last ``window`` sessions up to T."""
    if ctx.traded_value is None:
        return None
    return ctx.traded_value.tail(window).mean()


class RiskManagedStrategy(Strategy):
    def __init__(
        self,
        inner: Strategy,
        limits: RiskLimits,
        sectors: pd.Series | None = None,
        adv_window: int = 20,
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self.requires_fit = inner.requires_fit
        self.limits = limits
        self.sectors = sectors
        self.adv_window = adv_window
        self.decisions: list[dict] = []

    def fit(self, train) -> None:
        self.inner.fit(train)

    def target_weights(self, ctx: StrategyContext) -> pd.Series | None:
        proposed = self.inner.target_weights(ctx)
        if proposed is None:
            return None
        previous = ctx.current_weights.drop(labels=[CASH], errors="ignore")
        decision = enforce(
            proposed,
            limits=self.limits,
            previous=previous,
            sectors=self.sectors,
            adv=average_traded_value(ctx, self.adv_window),
            drawdown=ctx.drawdown,
        )
        self.decisions.append({"date": ctx.date.strftime("%Y-%m-%d"), **decision.as_record()})
        if decision.rejected:
            return None
        weights = decision.weights
        return weights[weights > 0]

    def risk_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.decisions)
