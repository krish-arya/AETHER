"""RL training pipeline (PRD 9.7).

    1. fit the observation scaler on the training window only
    2. build training environments (the real market, plus optional GAN scenarios)
    3. train PPO or SAC with a fixed seed
    4. every ``eval_every`` steps run the deterministic policy through the validation
       window and keep the checkpoint with the best validation Sharpe ratio
    5. reload the best checkpoint and write the model card

The test window is never touched here: checkpoint selection uses validation only.

Artifacts (``<out_dir>/``)
    model.zip        best Stable-Baselines3 checkpoint
    observation.json scaler statistics and observation layout
    model_card.json  registry entry: ids, dataset, config, seed, validation curve,
                     training time, environment (see ml/registry.py)

Known limitations: CPU-only and small step budgets by default; no hyper-parameter
search (the PRD's Optuna step) -- the config values are untuned defaults.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ml.backtesting.costs import CostModel
from ml.backtesting.data import MarketData
from ml.backtesting.metrics import sharpe_ratio
from ml.rl.config import RLConfig
from ml.rl.environments.portfolio_env import PortfolioEnv
from ml.rl.evaluation.strategy import sb3_policy
from ml.rl.observation import ObservationBuilder, ScenarioFeatures


@dataclass
class TrainedAgent:
    model: PPO | SAC
    builder: ObservationBuilder
    cfg: RLConfig
    seed: int
    validation_curve: list[dict]
    best_validation_sharpe: float
    best_step: int
    train_seconds: float
    path: Path | None = None


class _ValidationCallback(BaseCallback):
    """Evaluate on the validation window and remember the best weights."""

    def __init__(self, env: PortfolioEnv, every: int, risk_free_rate: float) -> None:
        super().__init__()
        self.env, self.every, self.risk_free_rate = env, every, risk_free_rate
        self.curve: list[dict] = []
        self.best_sharpe = -np.inf
        self.best_step = 0
        self.best_state: dict | None = None
        self._next = every

    def evaluate(self) -> None:
        returns = self.env.run_policy(sb3_policy(self.model))
        sharpe = sharpe_ratio(returns, self.risk_free_rate)
        total = float(np.prod(1 + returns.to_numpy()) - 1)
        self.curve.append(
            {
                "step": int(self.num_timesteps),
                "sharpe": None if np.isnan(sharpe) else float(sharpe),
                "total_return": total,
            }
        )
        score = -np.inf if np.isnan(sharpe) else sharpe
        if score > self.best_sharpe or self.best_state is None:
            self.best_sharpe, self.best_step = score, int(self.num_timesteps)
            self.best_state = {
                k: v.detach().clone() for k, v in self.model.policy.state_dict().items()
            }

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            self.evaluate()
        return True


def build_model(cfg: RLConfig, env, seed: int, policy_kwargs: dict | None = None) -> PPO | SAC:
    kwargs = dict(policy_kwargs or {})
    if cfg.algorithm == "ppo":
        kwargs.setdefault("net_arch", cfg.ppo.net_arch)
        return PPO(
            "MlpPolicy",
            env,
            n_steps=cfg.ppo.n_steps,
            batch_size=cfg.ppo.batch_size,
            n_epochs=cfg.ppo.n_epochs,
            learning_rate=cfg.ppo.learning_rate,
            gamma=cfg.ppo.gamma,
            gae_lambda=cfg.ppo.gae_lambda,
            ent_coef=cfg.ppo.ent_coef,
            clip_range=cfg.ppo.clip_range,
            policy_kwargs=kwargs,
            seed=seed,
            device="cpu",
            verbose=0,
        )
    kwargs.setdefault("net_arch", cfg.sac.net_arch)
    return SAC(
        "MlpPolicy",
        env,
        learning_rate=cfg.sac.learning_rate,
        buffer_size=cfg.sac.buffer_size,
        batch_size=cfg.sac.batch_size,
        learning_starts=cfg.sac.learning_starts,
        tau=cfg.sac.tau,
        gamma=cfg.sac.gamma,
        policy_kwargs=kwargs,
        seed=seed,
        device="cpu",
        verbose=0,
    )


def _policy_kwargs(cfg: RLConfig, builder: ObservationBuilder) -> dict:
    if cfg.fusion == "early":
        return {}
    from ml.fusion.extractors import fusion_policy_kwargs

    return fusion_policy_kwargs(cfg.fusion, builder.spec, len(builder.symbols))


def train_agent(
    market: MarketData,
    cfg: RLConfig,
    cost_model: CostModel,
    *,
    train_start,
    train_end,
    validation_start,
    validation_end,
    seed: int,
    sectors: pd.Series | None = None,
    news: pd.DataFrame | None = None,
    scenario: ScenarioFeatures | None = None,
    augmentation: list[MarketData] | None = None,
    risk_free_rate: float = 0.0,
    out_dir: Path | None = None,
) -> TrainedAgent:
    """Train one agent. ``augmentation`` holds GAN-generated markets (same symbols,
    with features) that are added as extra training environments; validation always
    uses the real market."""
    torch.set_num_threads(cfg.torch_threads)
    torch.manual_seed(seed)
    np.random.seed(seed)

    builder = ObservationBuilder(market, cfg.features, cfg.groups, news=news, scenario=scenario)
    builder.fit(train_start, train_end)
    common = {"limits": cfg.limits, "sectors": sectors, "env": cfg.env, "reward": cfg.reward}

    def real_env() -> PortfolioEnv:
        return PortfolioEnv(builder, cost_model, start=train_start, end=train_end, **common)

    makers = [real_env]
    for synthetic in augmentation or []:
        synthetic_builder = builder.for_market(synthetic)

        def synthetic_env(b=synthetic_builder) -> PortfolioEnv:
            sessions = b.market.sessions
            return PortfolioEnv(b, cost_model, start=sessions[0], end=sessions[-1], **common)

        makers.append(synthetic_env)
    train_env = DummyVecEnv(makers)
    train_env.seed(seed)

    validation_env = PortfolioEnv(
        builder, cost_model, start=validation_start, end=validation_end, random_start=False,
        **common,
    )  # fmt: skip
    model = build_model(cfg, train_env, seed, _policy_kwargs(cfg, builder))
    callback = _ValidationCallback(validation_env, cfg.eval_every, risk_free_rate)
    total = cfg.ppo.total_timesteps if cfg.algorithm == "ppo" else cfg.sac.total_timesteps

    started = time.perf_counter()
    model.learn(total_timesteps=total, callback=callback)
    callback.evaluate()  # always score the final weights too
    elapsed = time.perf_counter() - started
    model.policy.load_state_dict(callback.best_state)

    agent = TrainedAgent(
        model=model,
        builder=builder,
        cfg=cfg,
        seed=seed,
        validation_curve=callback.curve,
        best_validation_sharpe=float(callback.best_sharpe),
        best_step=callback.best_step,
        train_seconds=elapsed,
    )
    if out_dir is not None:
        save_agent(agent, out_dir)
    return agent


def save_agent(agent: TrainedAgent, out_dir: Path) -> Path:
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    agent.model.save(out_dir / "model.zip")
    with open(out_dir / "observation.json", "w", encoding="utf-8") as handle:
        json.dump(agent.builder.state(), handle)
    agent.path = out_dir
    return out_dir


def load_agent(
    path: Path,
    market: MarketData,
    cfg: RLConfig,
    *,
    news: pd.DataFrame | None = None,
    scenario: ScenarioFeatures | None = None,
):
    """Reload a saved agent for the same dataset: returns ``(model, builder)``."""
    import json

    path = Path(path)
    with open(path / "observation.json", encoding="utf-8") as handle:
        state = json.load(handle)
    builder = ObservationBuilder(market, cfg.features, cfg.groups, news=news, scenario=scenario)
    builder.load_state(state)
    algorithm = PPO if cfg.algorithm == "ppo" else SAC
    return algorithm.load(path / "model.zip", device="cpu"), builder
