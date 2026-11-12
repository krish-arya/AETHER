from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ml.backtesting.config import StrategySpec, WalkForwardConfig
from ml.backtesting.costs import CostConfig


class BacktestRequest(BaseModel):
    """Overrides applied on top of the server's default backtest configuration
    (``configs/backtesting.yaml``). Every omitted field keeps its default."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(None, pattern=r"^[a-z0-9][a-z0-9-]{1,48}$")
    dataset_id: str | None = Field(None, description="Defaults to the most recently loaded dataset")
    start_date: date | None = None
    end_date: date | None = None
    initial_capital: float | None = Field(None, gt=0, le=1e13)
    rebalance: Literal["daily", "weekly", "monthly", "quarterly"] | None = None
    risk_free_rate: float | None = Field(None, ge=-0.05, le=0.5)
    strategies: list[StrategySpec] | None = Field(None, min_length=1, max_length=12)
    benchmark: str | None = None
    costs: CostConfig | None = None
    walk_forward: WalkForwardConfig | None = None


class StrategyInfo(BaseModel):
    strategy: str
    description: str
    default_params: dict


class BacktestRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: str
    name: str
    status: Literal["queued", "running", "succeeded", "failed"]
    dataset_id: str
    is_synthetic: bool
    result_id: str | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class EquityCurves(BaseModel):
    dates: list[date]
    series: dict[str, list[float]]


class BacktestRunDetail(BacktestRunOut):
    config: dict
    summary: dict | None
    equity: EquityCurves | None


class BacktestMetrics(BaseModel):
    run_id: str
    status: str
    benchmark: str
    is_synthetic: bool
    caveats: list[str]
    full_period: dict[str, dict[str, float | None]]
    walk_forward: dict[str, dict[str, float | None]]


class ComparisonRow(BaseModel):
    run_id: str
    name: str
    strategy: str
    is_benchmark: bool
    is_synthetic: bool
    metrics: dict[str, float | None]
