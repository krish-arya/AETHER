"""Typed configuration for the RL portfolio optimizer (loaded from configs/rl.yaml)."""

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ml.risk.limits import RiskLimits

# Scale-free market features fed to the agent (see ml/features/engine.py).
DEFAULT_FEATURES = [
    "ret_1d",
    "cum_ret_5d",
    "cum_ret_21d",
    "cum_ret_63d",
    "sma_50_ratio",
    "macd_hist_pct",
    "rsi_14",
    "adx_14",
    "atr_pct_14",
    "hist_vol_20d",
    "bb_pctb_20",
    "obv_norm_20",
    "mfi_14",
    "beta_60d",
]
GROUPS = ("market", "regime", "news", "scenario")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnvConfig(_Strict):
    rebalance_every: int = Field(5, ge=1)  # sessions between decisions
    episode_steps: int = Field(52, ge=2)  # decisions per training episode
    logit_scale: float = Field(4.0, gt=0)  # action in [-1, 1] -> logits in [-scale, scale]
    adv_window: int = Field(20, ge=1)
    volatility_window: int = Field(20, ge=2)  # daily portfolio returns in the risk penalty
    # Portfolio size assumed in training. Per-order charges (brokerage caps, flat
    # fees) depend on order size in rupees, so costs are not scale-free.
    reference_capital: float = Field(1_000_000.0, gt=0)


class RewardConfig(_Strict):
    """Per-step reward (PRD 9.5). The lambdas are starting points, not truths: the
    experiment runner reports results under alternative formulations."""

    kind: Literal["log_return", "risk_adjusted", "drawdown_change"] = "risk_adjusted"
    lambda_volatility: float = Field(0.1, ge=0)
    lambda_drawdown: float = Field(0.02, ge=0)
    lambda_turnover: float = Field(0.002, ge=0)
    scale: float = Field(100.0, gt=0)


class PPOConfig(_Strict):
    total_timesteps: int = Field(60_000, ge=64)
    n_steps: int = Field(512, ge=8)
    batch_size: int = Field(128, ge=8)
    n_epochs: int = Field(10, ge=1)
    learning_rate: float = Field(3e-4, gt=0)
    gamma: float = Field(0.99, gt=0, le=1)
    gae_lambda: float = Field(0.95, gt=0, le=1)
    ent_coef: float = Field(0.005, ge=0)
    clip_range: float = Field(0.2, gt=0)
    net_arch: list[int] = [128, 128]


class SACConfig(_Strict):
    total_timesteps: int = Field(30_000, ge=64)
    learning_rate: float = Field(3e-4, gt=0)
    buffer_size: int = Field(50_000, ge=1000)
    batch_size: int = Field(256, ge=8)
    learning_starts: int = Field(1_000, ge=0)
    tau: float = Field(0.005, gt=0, le=1)
    gamma: float = Field(0.99, gt=0, le=1)
    net_arch: list[int] = [128, 128]


class RLConfig(_Strict):
    algorithm: Literal["ppo", "sac"] = "ppo"
    features: list[str] = DEFAULT_FEATURES
    # Observation groups in use. "portfolio" is always included. Dropping a group is
    # how ablations are run; "news" / "scenario" need the LLM / GAN engines' outputs.
    groups: list[Literal["market", "regime", "news", "scenario"]] = ["market", "regime"]
    # early: one flat vector; late: one encoder per group; attention: encoders whose
    # outputs are combined by learned attention weights (ml/fusion/extractors.py).
    fusion: Literal["early", "late", "attention"] = "early"
    env: EnvConfig = EnvConfig()
    reward: RewardConfig = RewardConfig()
    ppo: PPOConfig = PPOConfig()
    sac: SACConfig = SACConfig()
    limits: RiskLimits = RiskLimits()
    # Evaluate on the validation window this often and keep the best checkpoint.
    eval_every: int = Field(10_000, ge=64)
    seeds: list[int] = [0, 1, 2]
    torch_threads: int = Field(4, ge=1)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RLConfig":
        with open(path, encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))

    def config_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"seeds", "torch_threads"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
