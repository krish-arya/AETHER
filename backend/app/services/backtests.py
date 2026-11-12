"""Backtest jobs: build a validated config, record the job, execute it, read results.

The API never runs a backtest inline. ``create_run`` stores a ``queued`` row and the
caller hands the id to the job queue; a worker later calls ``execute_backtest_run``.
Failures are recorded on the row (PRD 18 "record failed model jobs") rather than lost.

Known limitation: a worker that dies mid-run leaves its row in ``running``; there is
no automatic recovery yet (Phase 8).
"""

import asyncio
import inspect
import logging
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import pandas as pd
import yaml
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models import BacktestRun, DatasetVersion
from app.models.backtest import (
    ACTIVE_STATUSES,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
)
from app.schemas.backtest import BacktestRequest, EquityCurves, StrategyInfo
from ml.backtesting.config import BacktestConfig
from ml.backtesting.runner import run_backtest
from ml.backtesting.strategies import STRATEGIES, build_strategy

logger = logging.getLogger(__name__)

OVERRIDABLE = (
    "name",
    "start_date",
    "end_date",
    "initial_capital",
    "rebalance",
    "risk_free_rate",
    "strategies",
    "benchmark",
    "costs",
    "walk_forward",
)
MAX_ERROR_LENGTH = 2000


class BacktestRequestError(ValueError):
    """The request cannot be turned into a valid backtest configuration."""


class TooManyActiveRuns(RuntimeError):
    pass


@lru_cache
def load_template(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def describe_strategies() -> list[StrategyInfo]:
    out = []
    for name, cls in sorted(STRATEGIES.items()):
        defaults = {
            parameter.name: parameter.default
            for parameter in inspect.signature(cls.__init__).parameters.values()
            if parameter.default is not inspect.Parameter.empty
        }
        description = (cls.__doc__ or "").strip().split("\n\n")[0].replace("\n", " ")
        out.append(StrategyInfo(strategy=name, description=description, default_params=defaults))
    return out


def build_config(
    request: BacktestRequest, dataset: DatasetVersion, template: dict
) -> BacktestConfig:
    """Template + request overrides -> a fully validated ``BacktestConfig``."""
    data = dict(template)
    data["dataset"] = Path(dataset.manifest_path).parent.as_posix()
    overrides = request.model_dump(mode="json", exclude_none=True)
    for field in OVERRIDABLE:
        if field in overrides:
            data[field] = overrides[field]
    try:
        cfg = BacktestConfig.model_validate(data)
        for spec in cfg.strategies:
            build_strategy(spec.strategy, spec.params)  # rejects unknown / invalid parameters
    except (ValueError, TypeError) as error:
        raise BacktestRequestError(str(error)) from error
    return cfg


async def resolve_dataset(session: AsyncSession, dataset_id: str | None) -> DatasetVersion | None:
    if dataset_id is not None:
        return await session.get(DatasetVersion, dataset_id)
    return await session.scalar(
        select(DatasetVersion).order_by(DatasetVersion.loaded_at.desc()).limit(1)
    )


async def create_run(
    session: AsyncSession,
    request: BacktestRequest,
    dataset: DatasetVersion,
    template: dict,
    max_active: int,
) -> BacktestRun:
    cfg = build_config(request, dataset, template)
    active = await session.scalar(
        select(func.count()).select_from(BacktestRun).where(BacktestRun.status.in_(ACTIVE_STATUSES))
    )
    if active >= max_active:
        raise TooManyActiveRuns(f"{active} backtests are already queued or running")
    run = BacktestRun(
        run_id=uuid4().hex,
        name=cfg.name,
        status=STATUS_QUEUED,
        dataset_id=dataset.dataset_id,
        is_synthetic=dataset.is_synthetic,
        config=cfg.model_dump(mode="json"),
        created_at=datetime.now(UTC),
    )
    session.add(run)
    await session.commit()
    return run


async def mark_failed(session: AsyncSession, run_id: str, error: str) -> None:
    run = await session.get(BacktestRun, run_id)
    if run is not None:
        run.status = STATUS_FAILED
        run.error = error[:MAX_ERROR_LENGTH]
        run.finished_at = datetime.now(UTC)
        await session.commit()


async def execute_backtest_run(
    run_id: str, session_factory: async_sessionmaker[AsyncSession], output_root: Path
) -> str:
    """Run a queued job to completion and record the outcome. Returns the final status."""
    async with session_factory() as session:
        run = await session.get(BacktestRun, run_id)
        if run is None:
            raise LookupError(f"backtest run {run_id} does not exist")
        if run.status != STATUS_QUEUED:
            return run.status  # already picked up (duplicate delivery)
        run.status = STATUS_RUNNING
        run.started_at = datetime.now(UTC)
        await session.commit()
        config = run.config

    try:
        cfg = BacktestConfig.model_validate(config)
        report = await asyncio.to_thread(run_backtest, cfg, output_root=output_root)
    except Exception as error:  # noqa: BLE001 - every failure must be recorded on the job
        logger.exception("backtest run %s failed", run_id)
        async with session_factory() as session:
            await mark_failed(session, run_id, f"{type(error).__name__}: {error}")
        return STATUS_FAILED

    async with session_factory() as session:
        run = await session.get(BacktestRun, run_id)
        run.status = STATUS_SUCCEEDED
        run.result_id = report.run_id
        run.summary = {k: v for k, v in report.summary.items() if k not in ("config",)}
        run.artifact_path = report.path.as_posix()
        run.finished_at = datetime.now(UTC)
        await session.commit()
    return STATUS_SUCCEEDED


def read_equity(artifact_path: str | None) -> EquityCurves | None:
    """Full-period equity curves written by the runner, if they are still on disk."""
    if not artifact_path:
        return None
    file = Path(artifact_path) / "equity.csv"
    if not file.is_file():
        return None
    frame = pd.read_csv(file, index_col=0, parse_dates=True, float_precision="round_trip")
    return EquityCurves(
        dates=[day.date() for day in frame.index],
        series={column: frame[column].tolist() for column in frame.columns},
    )
