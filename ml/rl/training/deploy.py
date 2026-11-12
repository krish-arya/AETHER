"""Train the allocation model used for current recommendations and paper trading.

The walk-forward experiments answer "does this approach work out of sample?". This
module trains the model that is actually *used*: fitted on the most recent data
available, with the last ``validation_sessions`` held back to choose the checkpoint.

    train        the ``train_years`` before the validation window
    validation   the last ``validation_sessions`` sessions of the dataset
    test         none -- there is no unseen data left, so this model has **no**
                 out-of-sample track record of its own. Judge the approach by the
                 experiment results, not by this model's validation score.

Several seeds are trained and the one with the best validation Sharpe is kept (seed
selection on validation data is ordinary model selection, not test-set tuning).

``load_deployed`` rebuilds a registered agent -- including the GAN scenario features it
was trained with, found through the model card's lineage -- for inference.
"""

from pathlib import Path

import pandas as pd

from ml import registry
from ml.backtesting.costs import CostModel
from ml.backtesting.data import MarketData
from ml.gan.generation.generate import scenario_features_from_table
from ml.rl.config import RLConfig
from ml.rl.training.train import TrainedAgent, load_agent, save_agent, train_agent

DEPLOYMENT = "deployment"


def deployment_windows(market: MarketData, train_years: int, validation_sessions: int) -> dict:
    sessions = market.sessions
    if len(sessions) < validation_sessions + 252:
        raise ValueError("the dataset is too short to hold out a validation window")
    validation_start = sessions[-validation_sessions]
    train_end = sessions[-validation_sessions - 1]
    earliest = train_end - pd.DateOffset(years=train_years)
    train_start = sessions[sessions.searchsorted(earliest, side="left")]
    return {
        "train": (train_start, train_end),
        "validation": (validation_start, sessions[-1]),
    }


def train_deployment_agent(
    market: MarketData,
    cfg: RLConfig,
    cost_model: CostModel,
    models_root: Path,
    *,
    train_years: int = 5,
    validation_sessions: int = 252,
    risk_free_rate: float = 0.0,
) -> dict:
    """Train on the latest data, register the best seed, return its model card."""
    windows = deployment_windows(market, train_years, validation_sessions)
    sectors = market.securities.get("sector")
    best: TrainedAgent | None = None
    scores = []
    for seed in cfg.seeds:
        agent = train_agent(
            market,
            cfg,
            cost_model,
            train_start=windows["train"][0],
            train_end=windows["train"][1],
            validation_start=windows["validation"][0],
            validation_end=windows["validation"][1],
            seed=seed,
            sectors=sectors,
            risk_free_rate=risk_free_rate,
        )
        scores.append({"seed": seed, "validation_sharpe": agent.best_validation_sharpe})
        if best is None or agent.best_validation_sharpe > best.best_validation_sharpe:
            best = agent

    identifier = registry.model_id(
        "rl", DEPLOYMENT, cfg.config_hash(), market.content_hash, windows["train"][1], best.seed
    )
    directory = registry.register(
        models_root,
        kind="rl",
        model_id=identifier,
        dataset_id=market.dataset_id,
        hyperparameters=cfg.model_dump(mode="json"),
        metrics={
            "validation_sharpe": best.best_validation_sharpe,
            "validation_curve": best.validation_curve,
            "seed_scores": scores,
            "train_seconds": best.train_seconds,
        },
        seed=best.seed,
        config_hash=cfg.config_hash(),
        training_window={
            name: [f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}"] for name, (start, end) in windows.items()
        },
        is_synthetic_data=market.is_synthetic,
        notes="Deployment model: trained on the most recent data. It has no out-of-sample "
        "test of its own; see the experiment results for evidence about the approach.",
        lineage={"experiment": DEPLOYMENT, "generator_id": None, "uses_news": False},
    )
    save_agent(best, directory)
    return registry.load_card(models_root, "rl", identifier)


def load_deployed(models_root: Path, card: dict, market: MarketData):
    """``(model, builder, cfg)`` for a registered RL model on its dataset."""
    if card["dataset_id"] != market.dataset_id:
        raise ValueError(
            f"model {card['model_id']} was trained on dataset {card['dataset_id']}, "
            f"not {market.dataset_id}"
        )
    cfg = RLConfig.model_validate(card["hyperparameters"])
    lineage = card.get("lineage") or {}
    if "news" in cfg.groups:
        raise ValueError("this model needs news features, which are not available for inference")
    scenario = None
    if "scenario" in cfg.groups:
        generator = lineage.get("generator_id")
        table_file = Path(models_root) / "gan" / str(generator) / "scenario_statistics.csv"
        if not generator or not table_file.is_file():
            raise ValueError(f"scenario statistics for generator {generator} are missing")
        scenario = scenario_features_from_table(pd.read_csv(table_file), market.symbols)
    directory = Path(models_root) / "rl" / card["model_id"]
    model, builder = load_agent(directory, market, cfg, scenario=scenario)
    return model, builder, cfg


def pick_model(cards: list[dict], dataset_id: str) -> dict | None:
    """Default model for a dataset: the newest deployment model, otherwise the
    walk-forward model with the latest training window (best validation Sharpe)."""
    usable = [
        c
        for c in cards
        if c["dataset_id"] == dataset_id and "news" not in c["hyperparameters"].get("groups", [])
    ]
    deployed = [c for c in usable if (c.get("lineage") or {}).get("experiment") == DEPLOYMENT]
    if deployed:
        return max(deployed, key=lambda c: c["trained_at"])
    if not usable:
        return None
    latest = max(c["training_window"]["train"][1] for c in usable)
    newest = [c for c in usable if c["training_window"]["train"][1] == latest]
    return max(newest, key=lambda c: c["metrics"].get("validation_sharpe") or float("-inf"))
