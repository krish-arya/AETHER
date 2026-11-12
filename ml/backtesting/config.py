"""Typed configuration for backtests (loaded from configs/backtesting.yaml)."""

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from ml.backtesting.costs import CostConfig
from ml.backtesting.strategies import STRATEGIES


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_serializer("*", when_used="json", check_fields=False)
    def _portable_paths(self, value):
        return value.as_posix() if isinstance(value, Path) else value


class StrategySpec(_Strict):
    strategy: str
    # Name used in reports; defaults to the strategy name. Must be unique in a run.
    label: str | None = None
    params: dict = {}

    @model_validator(mode="after")
    def _known_strategy(self):
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy '{self.strategy}'; available: {sorted(STRATEGIES)}")
        return self

    @property
    def key(self) -> str:
        return self.label or self.strategy


class WalkForwardConfig(_Strict):
    enabled: bool = True
    train_years: int = Field(5, ge=1)
    validation_years: int = Field(1, ge=1)
    test_years: int = Field(1, ge=1)
    step_years: int = Field(1, ge=1)
    embargo_sessions: int = Field(5, ge=0)


class BootstrapConfig(_Strict):
    samples: int = Field(2000, ge=100)
    block_sessions: int = Field(21, ge=1)
    confidence: float = Field(0.95, gt=0, lt=1)
    seed: int = 7


class BacktestConfig(_Strict):
    name: str = Field("baselines", pattern=r"^[a-z0-9][a-z0-9-]{1,48}$")
    # Dataset snapshot directory (see ml.data.pipeline).
    dataset: Path
    start_date: date | None = None  # None = first session in the dataset
    end_date: date | None = None  # None = last session in the dataset

    initial_capital: float = Field(1_000_000.0, gt=0)
    rebalance: Literal["daily", "weekly", "monthly", "quarterly"] = "monthly"
    execution_price: Literal["open", "close"] = "open"
    participation_limit: float | None = Field(0.05, gt=0, le=1)
    allow_fractional_shares: bool = True
    min_trade_value: float = Field(0.0, ge=0)
    delisting_recovery: float = Field(1.0, ge=0, le=1)

    # Annual risk-free rate used for Sharpe / Sortino / alpha. An assumption the user
    # must set for the period studied; there is no universally right value.
    risk_free_rate: float = 0.0
    periods_per_year: int = Field(252, ge=1)

    costs: CostConfig
    strategies: list[StrategySpec] = Field(min_length=1)
    # Label of the strategy other strategies are compared against.
    benchmark: str

    walk_forward: WalkForwardConfig = WalkForwardConfig()
    bootstrap: BootstrapConfig = BootstrapConfig()
    output_root: Path = Path("artifacts/backtests")

    @model_validator(mode="after")
    def _consistent(self):
        keys = [spec.key for spec in self.strategies]
        if len(set(keys)) != len(keys):
            raise ValueError("strategy labels must be unique")
        if self.benchmark not in keys:
            raise ValueError(f"benchmark '{self.benchmark}' is not one of the strategies {keys}")
        if self.start_date and self.end_date and self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BacktestConfig":
        with open(path, encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))

    def config_hash(self) -> str:
        """Hash of everything that influences results (not where they are written or
        where the dataset happens to be stored -- the dataset's content hash covers that)."""
        payload = self.model_dump(mode="json", exclude={"output_root", "dataset"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
