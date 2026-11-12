"""Background model jobs (PRD 15: long-running operations run asynchronously).

Kinds
    rl_train      train the deployment allocation model on the latest data
    gan_train     train and evaluate a scenario generator on the latest data
    rl_evaluate   backtest a registered RL model over a period
    experiments   run the research experiments (E0-E6, ablations)

``create_job`` records a queued job; a Celery worker later calls ``execute_job``. The
outcome -- result or error -- is always written back to the job row.
"""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.models import DatasetVersion, Job
from app.services import platform
from app.services.portfolio import _clean
from ml.backtesting.metrics import compute_metrics
from ml.backtesting.simulator import simulate
from ml.backtesting.walk_forward import Fold
from ml.fusion.experiments import _train_gan_for_fold, run_experiments
from ml.rl.evaluation.strategy import risk_managed_policy, sb3_policy
from ml.rl.training.deploy import train_deployment_agent

logger = logging.getLogger(__name__)
KINDS = ("rl_train", "gan_train", "rl_evaluate", "experiments")
ACTIVE = ("queued", "running")


class TooManyJobs(RuntimeError):
    pass


async def create_job(session: AsyncSession, kind: str, params: dict, max_active: int) -> Job:
    active = await session.scalar(
        select(func.count()).select_from(Job).where(Job.status.in_(ACTIVE))
    )
    if active >= max_active:
        raise TooManyJobs(f"{active} model jobs are already queued or running")
    job = Job(
        job_id=uuid4().hex, kind=kind, status="queued", params=params, created_at=datetime.now(UTC)
    )
    session.add(job)
    await session.commit()
    return job


def _run(kind: str, params: dict, dataset: DatasetVersion, settings: Settings) -> dict:
    market = platform.load_market(dataset)
    directory = Path(dataset.manifest_path).parent
    cfg = platform.experiment_defaults(settings, directory)
    defaults = platform.backtest_defaults(settings)
    costs = platform.cost_model(settings)
    root = platform.models_root(settings)

    if kind == "rl_train":
        rl = cfg.rl
        if params.get("total_timesteps"):
            rl = rl.model_copy(
                update={
                    "ppo": rl.ppo.model_copy(update={"total_timesteps": params["total_timesteps"]}),
                    "eval_every": max(params["total_timesteps"] // 4, 64),
                }
            )
        if params.get("seeds"):
            rl = rl.model_copy(update={"seeds": params["seeds"]})
        card = train_deployment_agent(
            market, rl, costs, root, risk_free_rate=defaults.risk_free_rate,
            train_years=params.get("train_years", 5),
            validation_sessions=params.get("validation_sessions", 252),
        )  # fmt: skip
        return {"model_id": card["model_id"], "training_window": card["training_window"],
                "validation_sharpe": card["metrics"]["validation_sharpe"],
                "seed_scores": card["metrics"]["seed_scores"]}  # fmt: skip

    if kind == "gan_train":
        holdout = params.get("validation_sessions", 252)
        sessions = market.sessions
        fold = Fold(0, sessions[0], sessions[-holdout - 1], sessions[-holdout], sessions[-1],
                    sessions[-holdout], sessions[-1])  # fmt: skip
        if params.get("iterations"):
            n = params["iterations"]
            phases = ("embedding_iterations", "supervised_iterations", "joint_iterations")
            gan = cfg.gan.model_copy(update=dict.fromkeys(phases, n))
            cfg = cfg.model_copy(update={"gan": gan})
        trained = _train_gan_for_fold(market, cfg, fold, root)
        return {
            "model_id": trained["model_id"],
            "passes": trained["report"]["passes"],
            "gates": trained["report"]["gates"],
            "excluded_assets": trained["excluded_assets"],
        }

    if kind == "rl_evaluate":
        card = platform.get_card(settings, "rl", params["model_id"])
        model, builder, rl = platform.load_agent_for(settings, card, market)
        sessions = market.sessions_between(
            params.get("start") or sessions_default(market),
            params.get("end") or market.sessions[-1],
        )
        strategy = risk_managed_policy(
            sb3_policy(model), builder, limits=rl.limits, sectors=market.securities.get("sector"),
            logit_scale=rl.env.logit_scale,
        )  # fmt: skip
        result = simulate(
            market, strategy, start=sessions[0], end=sessions[-1],
            decision_dates=sessions[:: rl.env.rebalance_every], cost_model=costs,
        )  # fmt: skip
        metrics = compute_metrics(
            result.returns, risk_free_rate=defaults.risk_free_rate,
            turnover=result.equity["turnover"], trades=result.trades,
            costs=result.equity["costs"], initial_capital=result.initial_capital,
        )  # fmt: skip
        windows = card["training_window"]
        seen_until = max(pd.Timestamp(v[1]) for v in windows.values() if isinstance(v, list))
        return {
            "model_id": card["model_id"],
            "period": [f"{sessions[0]:%Y-%m-%d}", f"{sessions[-1]:%Y-%m-%d}"],
            "metrics": metrics,
            "overlaps_training_or_validation": bool(sessions[0] <= seen_until),
            "note": "Periods the model was trained or validated on are in-sample: the result "
            "is not evidence of out-of-sample performance."
            if sessions[0] <= seen_until
            else "The period is after everything the model was trained or validated on.",
        }

    if kind == "experiments":
        update = {k: params[k] for k in ("experiments", "max_folds") if params.get(k)}
        cfg = cfg.model_copy(update=update)
        if params.get("total_timesteps"):
            cfg.rl.ppo.total_timesteps = params["total_timesteps"]
            cfg.rl.eval_every = max(params["total_timesteps"] // 4, 64)
        report = run_experiments(cfg, market=market)
        return {"run_id": report.run_id, "path": report.path.as_posix()}

    raise ValueError(f"unknown job kind '{kind}'")


def sessions_default(market) -> pd.Timestamp:
    return market.sessions[max(len(market.sessions) - 252, 0)]


async def execute_job(
    job_id: str, session_factory: async_sessionmaker[AsyncSession], settings: Settings
) -> str:
    """Run a queued job to completion and record the outcome. Returns the final status."""
    async with session_factory() as session:
        job = await session.get(Job, job_id)
        if job is None:
            raise LookupError(f"job {job_id} does not exist")
        if job.status != "queued":
            return job.status
        job.status, job.started_at = "running", datetime.now(UTC)
        await session.commit()
        kind, params = job.kind, dict(job.params)
        dataset = await platform.latest_dataset(session, params.get("dataset_id"))

    try:
        result = await asyncio.to_thread(_run, kind, params, dataset, settings)
        status, error = "succeeded", None
    except Exception as failure:  # noqa: BLE001 - every failure is recorded on the job
        logger.exception("job %s (%s) failed", job_id, kind)
        result, status, error = None, "failed", f"{type(failure).__name__}: {failure}"[:2000]

    async with session_factory() as session:
        job = await session.get(Job, job_id)
        job.status, job.error, job.finished_at = status, error, datetime.now(UTC)
        job.result = None if result is None else _clean(result)
        await session.commit()
    return status
