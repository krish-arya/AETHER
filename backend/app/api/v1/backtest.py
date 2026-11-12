"""Backtesting endpoints (PRD 15.5).

Backtests run asynchronously: ``POST /backtest/run`` validates the request, records a
job and returns its id immediately (``202``); poll ``GET /backtest/runs/{run_id}`` for
the status and, once ``succeeded``, the results.
"""

import asyncio
import logging
from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.database.session import get_session
from app.models import BacktestRun
from app.models.backtest import STATUS_SUCCEEDED
from app.schemas.backtest import (
    BacktestMetrics,
    BacktestRequest,
    BacktestRunDetail,
    BacktestRunOut,
    ComparisonRow,
    StrategyInfo,
)
from app.services import backtests

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/backtest", tags=["backtest"])

Dispatcher = Callable[[str], None]


def get_dispatcher() -> Dispatcher:
    """Returns a function that hands a run id to the job queue."""
    from app.workers.celery_app import celery_app
    from app.workers.tasks import BACKTEST_TASK

    def dispatch(run_id: str) -> None:
        celery_app.send_task(BACKTEST_TASK, args=[run_id], retry=False)

    return dispatch


async def _get_run(session: AsyncSession, run_id: str) -> BacktestRun:
    run = await session.get(BacktestRun, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown backtest run '{run_id}'")
    return run


@router.get("/strategies", response_model=list[StrategyInfo])
async def list_strategies() -> list[StrategyInfo]:
    """Rule-based baseline strategies available to backtests, with default parameters."""
    return backtests.describe_strategies()


@router.post("/run", response_model=BacktestRunOut, status_code=status.HTTP_202_ACCEPTED)
async def submit_backtest(
    request: BacktestRequest,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
    dispatch: Dispatcher = Depends(get_dispatcher),
) -> BacktestRun:
    dataset = await backtests.resolve_dataset(session, request.dataset_id)
    if dataset is None:
        detail = (
            f"Unknown dataset '{request.dataset_id}'"
            if request.dataset_id
            else "No dataset is loaded; build and load one first"
        )
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail)

    template = backtests.load_template(settings.backtest_config)
    try:
        run = await backtests.create_run(
            session, request, dataset, template, settings.max_active_backtests
        )
    except backtests.BacktestRequestError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
    except backtests.TooManyActiveRuns as error:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(error)) from error

    try:
        await asyncio.to_thread(dispatch, run.run_id)
    except Exception as error:  # noqa: BLE001 - any queue failure must be recorded
        logger.exception("could not enqueue backtest %s", run.run_id)
        await backtests.mark_failed(session, run.run_id, "Job queue unavailable; run not started")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The job queue is unavailable, so the backtest could not be started.",
        ) from error
    return run


@router.get("/runs", response_model=list[BacktestRunOut])
async def list_runs(
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
) -> list[BacktestRun]:
    result = await session.scalars(
        select(BacktestRun).order_by(BacktestRun.created_at.desc()).limit(limit)
    )
    return list(result)


@router.get("/runs/{run_id}", response_model=BacktestRunDetail)
async def get_run(run_id: str, session: AsyncSession = Depends(get_session)) -> BacktestRunDetail:
    run = await _get_run(session, run_id)
    equity = await asyncio.to_thread(backtests.read_equity, run.artifact_path)
    return BacktestRunDetail(
        **BacktestRunOut.model_validate(run).model_dump(),
        config=run.config,
        summary=run.summary,
        equity=equity,
    )


def _require_results(run: BacktestRun) -> dict:
    if run.status != STATUS_SUCCEEDED or run.summary is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Backtest run '{run.run_id}' has no results (status: {run.status})",
        )
    return run.summary


@router.get("/runs/{run_id}/metrics", response_model=BacktestMetrics)
async def get_metrics(run_id: str, session: AsyncSession = Depends(get_session)) -> BacktestMetrics:
    run = await _get_run(session, run_id)
    summary = _require_results(run)
    return BacktestMetrics(
        run_id=run.run_id,
        status=run.status,
        benchmark=summary["benchmark"],
        is_synthetic=run.is_synthetic,
        caveats=summary["caveats"],
        full_period={key: entry["metrics"] for key, entry in summary["strategies"].items()},
        walk_forward={
            key: entry["out_of_sample"]
            for key, entry in summary["walk_forward"]["strategies"].items()
            if "out_of_sample" in entry
        },
    )


@router.get("/compare", response_model=list[ComparisonRow])
async def compare_runs(
    run_ids: list[str] = Query(..., min_length=1, max_length=10, description="Repeatable"),
    session: AsyncSession = Depends(get_session),
) -> list[ComparisonRow]:
    """Full-period metrics of every strategy in the given runs, side by side.

    Runs are only comparable when they share data, period and cost assumptions; each
    row carries its run id so differences in configuration can be looked up.
    """
    rows: list[ComparisonRow] = []
    for run_id in run_ids:
        run = await _get_run(session, run_id)
        summary = _require_results(run)
        for key, entry in summary["strategies"].items():
            rows.append(
                ComparisonRow(
                    run_id=run.run_id,
                    name=run.name,
                    strategy=key,
                    is_benchmark=key == summary["benchmark"],
                    is_synthetic=run.is_synthetic,
                    metrics=entry["metrics"],
                )
            )
    return rows
