"""Adapter that lets a trained policy be backtested like any other strategy.

``PolicyStrategy`` builds the observation for the decision date with the same
``ObservationBuilder`` used in training, asks the policy for an action, and converts it
to weights with the same projection. Wrap it in ``RiskManagedStrategy`` (as
``risk_managed_policy`` does) so the allocation passes through the reference risk
engine and every decision is logged. Results therefore come from
``ml.backtesting.simulator`` with exact costs -- the same yardstick as the baselines.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from ml.backtesting.strategies import Strategy, StrategyContext
from ml.risk.limits import RiskLimits
from ml.risk.strategy import CASH, RiskManagedStrategy
from ml.rl.observation import ObservationBuilder
from ml.rl.projection import action_to_weights

Policy = Callable[[np.ndarray], np.ndarray]


class PolicyStrategy(Strategy):
    name = "rl_policy"
    requires_fit = True

    def __init__(
        self, policy: Policy, builder: ObservationBuilder, logit_scale: float, name: str = "rl"
    ) -> None:
        self.policy = policy
        self.builder = builder
        self.logit_scale = logit_scale
        self.name = name
        self.observations: list[tuple[pd.Timestamp, np.ndarray]] = []

    def target_weights(self, ctx: StrategyContext) -> pd.Series:
        symbols = self.builder.symbols
        row = self.builder.row_of(ctx.date)
        held = ctx.current_weights.drop(labels=[CASH], errors="ignore")
        weights = held.reindex(symbols, fill_value=0.0).to_numpy(dtype="float64")
        observation = self.builder.observe(row, weights, ctx.drawdown)
        self.observations.append((ctx.date, observation))
        tradable = np.isin(symbols, ctx.universe)
        target = action_to_weights(self.policy(observation), tradable, self.logit_scale)
        return pd.Series(target, index=symbols)


def sb3_policy(model) -> Policy:
    """Deterministic action function of a Stable-Baselines3 model."""

    def act(observation: np.ndarray) -> np.ndarray:
        action, _ = model.predict(observation, deterministic=True)
        return action

    return act


def risk_managed_policy(
    policy: Policy,
    builder: ObservationBuilder,
    *,
    limits: RiskLimits,
    sectors: pd.Series | None,
    logit_scale: float,
    adv_window: int = 20,
    name: str = "rl",
) -> RiskManagedStrategy:
    inner = PolicyStrategy(policy, builder, logit_scale, name)
    return RiskManagedStrategy(inner, limits, sectors, adv_window)
