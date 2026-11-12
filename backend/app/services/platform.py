"""Shared access to datasets, models and configuration for the model-backed endpoints.

Loaded markets and models are cached in-process: a dataset snapshot is immutable (its
id is its content hash) and a registered model never changes, so the caches cannot go
stale. They are per-process, which is fine for one API worker and for Celery workers.
"""

from functools import lru_cache
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models import DatasetVersion
from ml import registry
from ml.backtesting.config import BacktestConfig
from ml.backtesting.costs import CostModel
from ml.backtesting.data import MarketData
from ml.fusion.experiments import ExperimentConfig
from ml.rl.training.deploy import load_deployed, pick_model

_markets: dict[str, MarketData] = {}
_agents: dict[str, tuple] = {}


class NotAvailable(LookupError):
    """Something the request needs (a dataset, a model) does not exist yet."""


async def latest_dataset(session: AsyncSession, dataset_id: str | None = None) -> DatasetVersion:
    if dataset_id:
        dataset = await session.get(DatasetVersion, dataset_id)
    else:
        dataset = await session.scalar(
            select(DatasetVersion).order_by(DatasetVersion.loaded_at.desc()).limit(1)
        )
    if dataset is None:
        raise NotAvailable("No dataset is loaded; build and load one first")
    return dataset


def load_market(dataset: DatasetVersion) -> MarketData:
    """The dataset's snapshot as a market view (hash-verified on first load)."""
    if dataset.dataset_id not in _markets:
        directory = Path(dataset.manifest_path).parent
        if not directory.is_dir():
            raise NotAvailable(f"Dataset files for '{dataset.dataset_id}' are not on disk")
        _markets[dataset.dataset_id] = MarketData.from_snapshot(directory)
    return _markets[dataset.dataset_id]


def models_root(settings: Settings) -> Path:
    return settings.artifacts_dir / "models"


@lru_cache
def _yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def backtest_defaults(settings: Settings) -> BacktestConfig:
    """Costs, risk-free rate and other shared assumptions (configs/backtesting.yaml)."""
    return BacktestConfig.model_validate({**_yaml(settings.backtest_config), "dataset": "."})


def cost_model(settings: Settings) -> CostModel:
    return CostModel(backtest_defaults(settings).costs)


def experiment_defaults(settings: Settings, dataset_dir: Path) -> ExperimentConfig:
    """RL / GAN settings shared with the research experiments (configs/experiments.yaml)."""
    raw = {**_yaml(settings.experiments_config), "dataset": str(dataset_dir)}
    raw["backtest"] = str(settings.backtest_config)
    raw["models_root"] = str(models_root(settings))
    raw["output_root"] = str(settings.artifacts_dir / "experiments")
    return ExperimentConfig.model_validate(raw)


def list_cards(settings: Settings, kind: str, dataset_id: str | None = None) -> list[dict]:
    cards = registry.list_models(models_root(settings), kind)
    return [c for c in cards if dataset_id is None or c["dataset_id"] == dataset_id]


def get_card(settings: Settings, kind: str, model_id: str) -> dict:
    try:
        return registry.load_card(models_root(settings), kind, model_id)
    except FileNotFoundError as error:
        raise NotAvailable(f"Unknown {kind} model '{model_id}'") from error


def default_rl_card(settings: Settings, dataset_id: str) -> dict:
    card = pick_model(list_cards(settings, "rl"), dataset_id)
    if card is None:
        raise NotAvailable(
            f"No RL model has been trained on dataset '{dataset_id}'. "
            "Train one with POST /api/v1/rl/train or scripts/run_experiments.py."
        )
    return card


def load_agent_for(settings: Settings, card: dict, market: MarketData):
    """``(model, builder, rl_config)`` for a registered RL model (cached)."""
    if card["model_id"] not in _agents:
        try:
            _agents[card["model_id"]] = load_deployed(models_root(settings), card, market)
        except ValueError as error:
            raise NotAvailable(str(error)) from error
    return _agents[card["model_id"]]


def clear_caches() -> None:
    _markets.clear()
    _agents.clear()
    _yaml.cache_clear()
