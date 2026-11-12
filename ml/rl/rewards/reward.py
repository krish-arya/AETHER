"""Reward functions for the portfolio environment (PRD 9.5).

    log_return        R = r
    risk_adjusted     R = r - l1 * volatility - l2 * drawdown - l3 * turnover     (PRD)
    drawdown_change   R = r - l1 * volatility - l2 * max(drawdown increase, 0) - l3 * turnover

    r           log return of the portfolio over the step, after trading costs
    volatility  std of the last ``volatility_window`` daily portfolio returns, scaled
                to the step horizon (x sqrt(sessions per step)); 0 until two returns exist
    drawdown    current drawdown from the episode's peak, as a positive fraction
    turnover    one-way turnover of the rebalance that started the step

Trading costs are already inside ``r``; ``lambda_turnover`` is an *additional* penalty
that discourages churn beyond its direct cost. The reward is multiplied by ``scale``
so its magnitude suits PPO/SAC.

``risk_adjusted`` penalises the drawdown *level* at every step, which keeps punishing
the agent for a loss that has already happened; ``drawdown_change`` penalises only new
drawdown. Which works better is an empirical question the experiments report on.
"""

from ml.rl.config import RewardConfig


def compute_reward(
    cfg: RewardConfig,
    *,
    log_return: float,
    volatility: float,
    drawdown: float,
    previous_drawdown: float,
    turnover: float,
) -> float:
    """``drawdown`` and ``previous_drawdown`` are positive fractions (0.1 = 10% below peak)."""
    if cfg.kind == "log_return":
        value = log_return
    else:
        if cfg.kind == "risk_adjusted":
            drawdown_term = drawdown
        else:  # drawdown_change
            drawdown_term = max(drawdown - previous_drawdown, 0.0)
        value = (
            log_return
            - cfg.lambda_volatility * volatility
            - cfg.lambda_drawdown * drawdown_term
            - cfg.lambda_turnover * turnover
        )
    return float(value * cfg.scale)
