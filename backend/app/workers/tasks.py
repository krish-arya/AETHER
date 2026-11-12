"""Background job implementations executed by Celery workers."""

import asyncio

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.services.backtests import execute_backtest_run
from app.services.jobs import execute_job
from app.workers.celery_app import celery_app

BACKTEST_TASK = "aether.backtest.run"
JOB_TASK = "aether.job.run"


async def _run_backtest(run_id: str) -> str:
    settings = get_settings()
    # A task runs in its own event loop, so it gets its own unpooled engine.
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        return await execute_backtest_run(run_id, factory, settings.artifacts_dir / "backtests")
    finally:
        await engine.dispose()


@celery_app.task(name=BACKTEST_TASK)
def run_backtest_job(run_id: str) -> str:
    """Execute one queued backtest. The outcome is recorded on its ``backtest_runs`` row."""
    return asyncio.run(_run_backtest(run_id))


async def _run_job(job_id: str) -> str:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    try:
        return await execute_job(
            job_id, async_sessionmaker(engine, expire_on_commit=False), settings
        )
    finally:
        await engine.dispose()


@celery_app.task(name=JOB_TASK)
def run_model_job(job_id: str) -> str:
    """Execute one queued model job (training, evaluation, experiments)."""
    return asyncio.run(_run_job(job_id))
