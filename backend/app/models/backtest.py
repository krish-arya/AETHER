"""Backtest job records (PRD 5.5 ``backtest_runs``, 15.5).

One row per submitted backtest. The row is the job: ``status`` moves
``queued -> running -> succeeded | failed``. Bulky outputs (equity curves, trades,
weights) stay on disk under ``artifact_path``; the row keeps the config that produced
them and the metric summary, so a result can always be traced to its inputs.
"""

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
ACTIVE_STATUSES = (STATUS_QUEUED, STATUS_RUNNING)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    run_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), index=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.dataset_id"))
    is_synthetic: Mapped[bool] = mapped_column(Boolean)
    # Fully resolved BacktestConfig (ml.backtesting.config) the job runs with.
    config: Mapped[dict] = mapped_column(JSON)
    # Content-derived id from the runner: identical experiments share it.
    result_id: Mapped[str | None] = mapped_column(String(96))
    summary: Mapped[dict | None] = mapped_column(JSON)
    artifact_path: Mapped[str | None] = mapped_column(String(512))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
