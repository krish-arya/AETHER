"""Celery application for background jobs (model training, backtests, ingestion).

Start a worker with::

    uv run celery -A app.workers.celery_app worker --loglevel=INFO

On Windows add ``--pool=solo`` (Celery's prefork pool is not supported there).
Job implementations live in ``app.workers.tasks``.
"""

from datetime import UTC, datetime

from celery import Celery

from app.core.config import APP_VERSION, get_settings

settings = get_settings()

celery_app = Celery(
    "aether",
    broker=settings.broker_url,
    backend=settings.result_backend,
    include=["app.workers.tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
)


@celery_app.task(name="aether.system.ping")
def ping() -> dict[str, str]:
    """Round-trip check that a worker is consuming the queue."""
    return {"status": "ok", "version": APP_VERSION, "time": datetime.now(UTC).isoformat()}
