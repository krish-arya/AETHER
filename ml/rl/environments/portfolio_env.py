"""Gymnasium portfolio-management environment (PRD 9.2).

One step = one rebalancing decision. At the close of session ``t`` the agent observes
the unified market state and outputs target weights. The environment then advances
``rebalance_every`` sessions:

    session t+1   overnight move (close t -> open t+1) on the old weights; the
                  rebalance trades at the open and pays costs; the open -> close move
                  applies to the new weights
    later ones    close-to-close moves on the drifting weights

so, exactly as in the backtester, a decision is never executed at a price the agent
has already seen. Target weights always pass through the risk limits first.

What is simulated: trading days, portfolio valuation, asset returns, transaction costs
and slippage (as proportional rates), cash, position limits, and trading restrictions
(untradable and illiquid assets, delistings).

Fidelity
    This is a fast, weight-based re-implementation used for *training*. Charges are
    computed per order with the same formula as ``CostModel.charges`` for a portfolio
    of ``reference_capital`` rupees. Two simplifications remain: fills are not capped
    by the day's liquidity, and when cash cannot cover the costs the buys are trimmed
    once rather than iteratively. Reported results never come from here: trained
    policies are evaluated in ``ml.backtesting.simulator`` under the same rules as
    every baseline. ``tests/ml/test_rl_env.py`` checks that the two agree for a fixed
    policy.

Episodes
    Training episodes start at a random session of the window and last
    ``episode_steps`` decisions. With ``random_start=False`` the episode starts at the
    first session and runs to the end of the window (used for validation).

Observation: ``ObservationBuilder.observe`` (float32 vector).
Action: Box(-1, 1, N + 1) -- one entry per asset plus cash (see ``projection``).
Reward: ``ml.rl.rewards.reward.compute_reward``.
"""

import gymnasium as gym
import numpy as np
import pandas as pd

from ml.backtesting.costs import CostModel
from ml.risk.fast import FastRiskProjector
from ml.risk.limits import RiskLimits
from ml.rl.config import EnvConfig, RewardConfig
from ml.rl.observation import ObservationBuilder
from ml.rl.projection import action_to_weights
from ml.rl.rewards.reward import compute_reward


def sector_ids(sectors: pd.Series | None, symbols: list[str]) -> np.ndarray:
    """Integer sector code per symbol (unknown sectors share one code)."""
    if sectors is None:
        return np.zeros(len(symbols), dtype=np.int64)
    labels = sectors.reindex(symbols).fillna("Unknown")
    return pd.factorize(labels)[0].astype(np.int64)


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        builder: ObservationBuilder,
        cost_model: CostModel,
        *,
        start,
        end,
        limits: RiskLimits,
        sectors: pd.Series | None = None,
        env: EnvConfig | None = None,
        reward: RewardConfig | None = None,
        random_start: bool = True,
    ) -> None:
        super().__init__()
        self.builder = builder
        self.market = builder.market
        self.cfg = env or EnvConfig()
        self.reward_cfg = reward or RewardConfig()
        self.random_start = random_start
        self.limits = limits
        self.n = len(builder.symbols)

        sessions = self.market.sessions
        self.lo = int(sessions.searchsorted(pd.Timestamp(start), side="left"))
        self.hi = int(sessions.searchsorted(pd.Timestamp(end), side="right")) - 1
        if self.hi - self.lo < self.cfg.rebalance_every:
            raise ValueError("the window is shorter than one rebalancing step")

        self._open = self.market.open.to_numpy()
        self._close = self.market.close.to_numpy()
        value = self.market.traded_value
        adv = value.rolling(self.cfg.adv_window, min_periods=1).mean().to_numpy()
        self._eligible = builder.tradable & (np.nan_to_num(adv, nan=0.0) >= limits.min_adv)
        # Statutory rates per session: stt_buy, stt_sell, exchange, sebi, stamp_buy,
        # stamp_sell, gst.
        self._rates = np.array(
            [
                (r.stt_buy, r.stt_sell, r.exchange_txn, r.sebi_fee, r.stamp_duty_buy,
                 r.stamp_duty_sell, r.gst)
                for r in (cost_model.rates_on(day) for day in sessions[self.lo : self.hi + 1])
            ]
        )  # fmt: skip
        self._brokerage = cost_model.config.brokerage
        self._flat_sell = cost_model.config.flat_per_sell_order
        self._slippage = cost_model.slippage_rate
        self.projector = FastRiskProjector(limits, sector_ids(sectors, builder.symbols))

        bound = builder.clip + 5.0  # scaled features are clipped; other blocks are in [-1, 1]
        self.observation_space = gym.spaces.Box(-bound, bound, (builder.size,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.n + 1,), np.float32)
        self._reset_state(self.lo)

    # --------------------------------------------------------------------- state

    def _reset_state(self, row: int) -> None:
        self.row = row
        self.weights = np.zeros(self.n)  # asset weights by value; the rest is cash
        self.value = 1.0
        self.peak = 1.0
        self.steps = 0
        self.daily_returns: list[float] = []
        self.history: list[dict] = []

    @property
    def drawdown(self) -> float:
        return self.value / self.peak - 1.0

    def _observation(self) -> np.ndarray:
        return self.builder.observe(self.row, self.weights, self.drawdown)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        row = self.lo
        if options and "start_row" in options:
            row = int(options["start_row"])
        elif self.random_start:
            span = self.cfg.episode_steps * self.cfg.rebalance_every
            latest = max(self.hi - span, self.lo)
            row = int(self.np_random.integers(self.lo, latest + 1))
        self._reset_state(row)
        return self._observation(), {"row": row}

    # ---------------------------------------------------------------------- step

    def _trade_cost(self, delta: np.ndarray, day: int) -> float:
        """Cost of trading ``delta`` (weight changes) as a fraction of portfolio value;
        mirrors ``CostModel.charges`` order by order, plus slippage."""
        scale = self.value * self.cfg.reference_capital
        traded = np.abs(delta) * scale  # rupees per order
        active = traded > 0
        if not active.any():
            return 0.0
        stt_buy, stt_sell, exchange, sebi, stamp_buy, stamp_sell, gst = self._rates[day - self.lo]
        broker = self._brokerage
        brokerage = np.maximum(broker.rate * traded, broker.min_per_order)
        if broker.max_per_order is not None:
            brokerage = np.minimum(brokerage, broker.max_per_order)
        brokerage = np.where(active, brokerage, 0.0)
        buying = delta > 0
        duties = np.where(buying, stt_buy + stamp_buy, stt_sell + stamp_sell) * traded
        fees = (brokerage + (exchange + sebi) * traded) * (1 + gst)
        flat = self._flat_sell * np.count_nonzero(active & ~buying)
        return float((fees.sum() + duties.sum() + self._slippage * traded.sum() + flat) / scale)

    def _grow(self, weights: np.ndarray, gross: np.ndarray) -> np.ndarray:
        """Apply gross asset returns; update value and return the drifted weights."""
        gross = np.where(np.isfinite(gross), gross, 1.0)  # no bar: price unchanged
        growth = 1.0 - weights.sum() + (weights * gross).sum()
        self.daily_factor *= growth
        self.value *= growth
        return weights * gross / growth

    def step(self, action):
        row, k = self.row, self.cfg.rebalance_every
        if row + k > self.hi:
            raise RuntimeError("step() called after the episode ended; call reset()")
        weights = self.weights.copy()
        # A holding that has delisted is closed out at its last price.
        weights[~self.builder.listed[row]] = 0.0

        proposed = action_to_weights(action, self.builder.tradable[row], self.cfg.logit_scale)
        target = self.projector.project(proposed, weights, self._eligible[row], self.drawdown)

        value_before, drawdown_before = self.value, -self.drawdown
        turnover = cost = 0.0
        for offset in range(1, k + 1):
            day = row + offset
            self.daily_factor = 1.0
            if offset == 1:
                weights = self._grow(weights, self._open[day] / self._close[row])
                can_trade = np.isfinite(self._open[day])
                desired = np.where(can_trade, target, weights)
                excess = desired.sum() - 1.0
                tradable_total = desired[can_trade].sum()
                if excess > 0 and tradable_total > 0:  # untradable holdings left too little room
                    shrink = max(1 - excess / tradable_total, 0.0)
                    desired = np.where(can_trade, desired * shrink, desired)
                delta = desired - weights
                cost = self._trade_cost(delta, day)
                cash = 1.0 - desired.sum() - cost
                if cash < 0:  # scale the buys so costs can be paid from cash
                    buys = delta.clip(min=0)
                    desired = desired - buys * min(-cash / max(buys.sum(), 1e-12), 1.0)
                    delta = desired - weights
                    cost = self._trade_cost(delta, day)
                turnover = 0.5 * (np.abs(delta).sum() + abs(delta.sum()))
                self.value *= 1.0 - cost
                self.daily_factor *= 1.0 - cost
                weights = desired / (1.0 - cost)
                weights = self._grow(weights, self._close[day] / self._open[day])
            else:
                weights = self._grow(weights, self._close[day] / self._close[day - 1])
            self.daily_returns.append(self.daily_factor - 1.0)
            self.peak = max(self.peak, self.value)

        self.row, self.weights, self.steps = row + k, weights, self.steps + 1
        window = self.daily_returns[-self.cfg.volatility_window :]
        volatility = float(np.std(window, ddof=1) * np.sqrt(k)) if len(window) >= 2 else 0.0
        reward = compute_reward(
            self.reward_cfg,
            log_return=float(np.log(self.value / value_before)),
            volatility=volatility,
            drawdown=-self.drawdown,
            previous_drawdown=drawdown_before,
            turnover=turnover,
        )
        terminated = self.row + k > self.hi
        truncated = self.random_start and self.steps >= self.cfg.episode_steps and not terminated
        info = {
            "row": self.row,
            "value": self.value,
            "turnover": turnover,
            "cost": cost,
            "drawdown": self.drawdown,
            "invested": float(weights.sum()),
        }
        self.history.append(info)
        return self._observation(), reward, terminated, truncated, info

    # ---------------------------------------------------------------- evaluation

    def run_policy(self, policy) -> pd.Series:
        """Run ``policy(observation) -> action`` once through the whole window and
        return the daily portfolio returns (indexed by session)."""
        observation, info = self.reset(options={"start_row": self.lo})
        first = info["row"]
        done = False
        while not done:
            observation, _, terminated, truncated, _ = self.step(policy(observation))
            done = terminated or truncated
        index = self.market.sessions[first + 1 : first + 1 + len(self.daily_returns)]
        return pd.Series(self.daily_returns, index=index)
