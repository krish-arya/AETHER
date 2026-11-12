"""RL, GAN, experiment and job endpoints (PRD 15.3, 15.4).

Training and evaluation run as background jobs: the POST returns ``202`` with a job id
and ``GET /jobs/{job_id}`` reports status and result. Listing endpoints read the model
registry (``artifacts/models``) and experiment reports (``artifacts/experiments``).
"""

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.database.session import get_session
from app.models import Job
from app.services import jobs, platform, portfolio
from app.services.platform import NotAvailable
from ml.features.regime import REGIMES
from ml.gan.data import CONDITIONS
from ml.gan.generation.generate import PURPOSES, generate_windows, window_returns
from ml.gan.training.train import load_gan

logger = logging.getLogger(__name__)
rl_router = APIRouter(prefix="/rl", tags=["rl"])
gan_router = APIRouter(prefix="/gan", tags=["gan"])
experiments_router = APIRouter(prefix="/experiments", tags=["experiments"])
jobs_router = APIRouter(prefix="/jobs", tags=["jobs"])

Dispatcher = Callable[[str], None]


def get_job_dispatcher() -> Dispatcher:
    from app.workers.celery_app import celery_app
    from app.workers.tasks import JOB_TASK

    def dispatch(job_id: str) -> None:
        celery_app.send_task(JOB_TASK, args=[job_id], retry=False)

    return dispatch


def _not_found(error: Exception) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, str(error))


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    job_id: str
    kind: str
    status: str
    params: dict
    result: dict | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


async def _submit(kind: str, params: dict, session, settings, dispatch) -> Job:
    try:
        await platform.latest_dataset(session)
        job = await jobs.create_job(session, kind, params, settings.max_active_jobs)
    except NotAvailable as error:
        raise _not_found(error) from error
    except jobs.TooManyJobs as error:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(error)) from error
    try:
        await asyncio.to_thread(dispatch, job.job_id)
    except Exception as error:  # noqa: BLE001 - any queue failure must be recorded
        logger.exception("could not enqueue job %s", job.job_id)
        job.status, job.error = "failed", "Job queue unavailable; job not started"
        job.finished_at = datetime.now(UTC)
        await session.commit()
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "The job queue is unavailable."
        ) from error
    return job


# --------------------------------------------------------------------------- RL


class TrainRequest(_Strict):
    total_timesteps: int | None = Field(None, ge=1_000, le=2_000_000)
    seeds: list[int] | None = Field(None, min_length=1, max_length=5)
    train_years: int = Field(5, ge=1, le=20)
    validation_sessions: int = Field(252, ge=60, le=756)


class EvaluateRequest(_Strict):
    model_id: str
    start: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    end: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class AllocateRequest(_Strict):
    model_id: str | None = Field(None, description="Defaults to the newest deployment model")


def _card_summary(card: dict) -> dict:
    metrics = {k: v for k, v in card["metrics"].items() if k != "validation_curve"}
    return {**{k: v for k, v in card.items() if k not in ("hyperparameters", "metrics")},
            "metrics": metrics}  # fmt: skip


@rl_router.get("/models")
async def list_rl_models(
    dataset_id: str | None = None, settings: Settings = Depends(get_settings)
) -> list[dict]:
    """Registered RL models, newest first (PRD 16.1 registry fields)."""
    return [_card_summary(c) for c in platform.list_cards(settings, "rl", dataset_id)]


@rl_router.get("/models/{model_id}")
async def get_rl_model(model_id: str, settings: Settings = Depends(get_settings)) -> dict:
    try:
        return platform.get_card(settings, "rl", model_id)
    except NotAvailable as error:
        raise _not_found(error) from error


@rl_router.post("/train", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
async def train_rl(
    request: TrainRequest,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
    dispatch: Dispatcher = Depends(get_job_dispatcher),
) -> Job:
    """Train the deployment model on the latest data (background job)."""
    return await _submit(
        "rl_train", request.model_dump(exclude_none=True), session, settings, dispatch
    )


@rl_router.post("/evaluate", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
async def evaluate_rl(
    request: EvaluateRequest,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
    dispatch: Dispatcher = Depends(get_job_dispatcher),
) -> Job:
    """Backtest a registered model over a period (background job). The result states
    whether the period overlaps the model's own training data."""
    try:
        platform.get_card(settings, "rl", request.model_id)
    except NotAvailable as error:
        raise _not_found(error) from error
    return await _submit(
        "rl_evaluate", request.model_dump(exclude_none=True), session, settings, dispatch
    )


@rl_router.post("/allocate")
async def allocate(
    request: AllocateRequest,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Target allocation for the next session from a trained RL model, after the risk
    engine, with its explanation record. Stored with a timestamp. This is research
    output, not investment advice, and it is never executed."""
    try:
        allocation = await portfolio.recommend(session, settings, request.model_id)
    except NotAvailable as error:
        raise _not_found(error) from error
    return {
        "allocation_id": allocation.allocation_id,
        "created_at": allocation.created_at.isoformat(),
        **allocation.explanation,
    }


# -------------------------------------------------------------------------- GAN


class GanTrainRequest(_Strict):
    iterations: int | None = Field(None, ge=10, le=20_000)
    validation_sessions: int = Field(252, ge=60, le=756)


class GenerateRequest(_Strict):
    model_id: str | None = None
    regime: str = Field("market_stress")
    scenarios: int = Field(100, ge=10, le=500)
    seed: int = 0
    purpose: str = Field("scenario_exploration")


@gan_router.get("/models")
async def list_gan_models(
    dataset_id: str | None = None, settings: Settings = Depends(get_settings)
) -> list[dict]:
    """Registered scenario generators. ``status`` is ``validated`` only when the
    generator passed its quality gates; otherwise ``rejected``."""
    return platform.list_cards(settings, "gan", dataset_id)


@gan_router.get("/evaluation/{model_id}")
async def gan_evaluation(model_id: str, settings: Settings = Depends(get_settings)) -> dict:
    """Statistical evaluation and memorisation check of a generator (PRD 7.5)."""
    file = platform.models_root(settings) / "gan" / model_id / "evaluation.json"
    if not file.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No evaluation for generator '{model_id}'")
    return json.loads(file.read_text(encoding="utf-8"))


@gan_router.post("/train", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
async def train_gan(
    request: GanTrainRequest,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
    dispatch: Dispatcher = Depends(get_job_dispatcher),
) -> Job:
    return await _submit(
        "gan_train", request.model_dump(exclude_none=True), session, settings, dispatch
    )


def _generate(settings: Settings, market, card: dict, request: GenerateRequest) -> dict:
    gan = load_gan(platform.models_root(settings) / "gan" / card["model_id"], market)
    code = CONDITIONS.index(request.regime)
    raw = generate_windows(gan, np.full(request.scenarios, code), request.seed)
    returns = window_returns(gan, raw)  # [n, L, N]
    market_paths = np.exp(np.cumsum(returns.mean(axis=2), axis=1)) - 1  # equal weight
    total = np.exp(returns.sum(axis=1)) - 1
    summary = {
        "scenario_set_id": uuid4().hex[:12],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "is_synthetic": True,
        "label": "SYNTHETIC scenarios generated by a GAN. Not forecasts.",
        "generator_id": card["model_id"],
        "generator_status": card["status"],
        "dataset_id": card["dataset_id"],
        "regime": request.regime,
        "purpose": request.purpose,
        "scenarios": request.scenarios,
        "horizon_sessions": gan.cfg.sequence_length,
        "seed": request.seed,
        "equal_weight_path": {
            "p05": np.quantile(market_paths, 0.05, axis=0).tolist(),
            "p50": np.quantile(market_paths, 0.50, axis=0).tolist(),
            "p95": np.quantile(market_paths, 0.95, axis=0).tolist(),
        },
        "equal_weight_total_return": {
            "mean": float(market_paths[:, -1].mean()),
            "p05": float(np.quantile(market_paths[:, -1], 0.05)),
            "worst": float(market_paths[:, -1].min()),
        },
        "assets": [
            {
                "symbol": symbol,
                "mean_return": float(total[:, j].mean()),
                "p05_return": float(np.quantile(total[:, j], 0.05)),
                "annualised_volatility": float(returns[:, :, j].std() * np.sqrt(252)),
            }
            for j, symbol in enumerate(gan.windows.symbols)
        ],
    }
    directory = settings.artifacts_dir / "scenarios" / summary["scenario_set_id"]
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "returns.npy", returns.astype(np.float32))
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@gan_router.post("/generate")
async def generate_scenarios(
    request: GenerateRequest,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Generate a labelled set of synthetic scenarios for one regime and store it."""
    if request.regime not in REGIMES:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"regime must be one of {REGIMES}"
        )
    if request.purpose not in PURPOSES:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"purpose must be one of {PURPOSES}"
        )
    try:
        dataset = await platform.latest_dataset(session)
        market = await asyncio.to_thread(platform.load_market, dataset)
        if request.model_id:
            card = platform.get_card(settings, "gan", request.model_id)
        else:
            cards = platform.list_cards(settings, "gan", dataset.dataset_id)
            if not cards:
                raise NotAvailable("No scenario generator has been trained on this dataset")
            card = cards[0]
        return await asyncio.to_thread(_generate, settings, market, card, request)
    except NotAvailable as error:
        raise _not_found(error) from error
    except ValueError as error:
        raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error


@gan_router.get("/scenarios")
async def list_scenarios(settings: Settings = Depends(get_settings)) -> list[dict]:
    """Stored synthetic scenario sets, newest first."""
    root = settings.artifacts_dir / "scenarios"
    if not root.is_dir():
        return []
    sets = [json.loads(f.read_text(encoding="utf-8")) for f in root.glob("*/summary.json")]
    return sorted(sets, key=lambda s: s["created_at"], reverse=True)


# ------------------------------------------------------------------ experiments


def _reports(settings: Settings) -> list[dict]:
    root = settings.artifacts_dir / "experiments"
    if not root.is_dir():
        return []
    reports = [json.loads(f.read_text(encoding="utf-8")) for f in root.glob("*/result.json")]
    return sorted(reports, key=lambda r: r["created_at"], reverse=True)


@experiments_router.get("")
async def list_experiments(settings: Settings = Depends(get_settings)) -> list[dict]:
    """Research experiment runs (PRD 20), newest first, without the per-fold detail."""
    return [
        {k: v for k, v in report.items() if k not in ("per_fold", "config")}
        for report in _reports(settings)
    ]


@experiments_router.get("/{run_id}")
async def get_experiment(run_id: str, settings: Settings = Depends(get_settings)) -> dict:
    for report in _reports(settings):
        if report["run_id"] == run_id:
            return report
    raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown experiment run '{run_id}'")


class ExperimentRequest(_Strict):
    experiments: list[str] | None = Field(None, min_length=1, max_length=12)
    max_folds: int | None = Field(None, ge=1, le=10)
    total_timesteps: int | None = Field(None, ge=1_000, le=2_000_000)


@experiments_router.post("/run", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
async def run_experiment_job(
    request: ExperimentRequest,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
    dispatch: Dispatcher = Depends(get_job_dispatcher),
) -> Job:
    return await _submit(
        "experiments", request.model_dump(exclude_none=True), session, settings, dispatch
    )


# -------------------------------------------------------------------------- jobs


@jobs_router.get("", response_model=list[JobOut])
async def list_jobs(
    limit: int = Query(50, ge=1, le=200), session: AsyncSession = Depends(get_session)
) -> list[Job]:
    return list(await session.scalars(select(Job).order_by(Job.created_at.desc()).limit(limit)))


@jobs_router.get("/{job_id}", response_model=JobOut)
async def get_job(job_id: str, session: AsyncSession = Depends(get_session)) -> Job:
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown job '{job_id}'")
    return job
