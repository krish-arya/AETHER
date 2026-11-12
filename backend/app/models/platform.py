"""Phases 4-7 persistence: allocations, paper trading, background jobs.

``allocations``      every allocation the RL model proposed, with its full explanation
                     record, timestamped (PRD 3.2 principle 3; ``portfolio_allocations``).
``paper_accounts``   a paper-trading account is only a *description*; its positions and
                     P&L are recomputed by replaying the simulator (ml.backtesting.paper).
``paper_orders``     hand-entered simulated orders (target weights). Never sent anywhere.
``jobs``             long-running model jobs (GAN / RL training, evaluation, experiments).
"""

from datetime import date, datetime

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class Allocation(Base):
    __tablename__ = "allocations"

    allocation_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    as_of: Mapped[date] = mapped_column(Date)
    model_id: Mapped[str] = mapped_column(String(128))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.dataset_id"))
    is_synthetic: Mapped[bool] = mapped_column(Boolean)
    weights: Mapped[dict] = mapped_column(JSON)
    explanation: Mapped[dict] = mapped_column(JSON)


class PaperAccount(Base):
    __tablename__ = "paper_accounts"

    account_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), index=True)  # active | closed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.dataset_id"))
    initial_capital: Mapped[float] = mapped_column(Float)
    start_date: Mapped[date] = mapped_column(Date)
    rebalance_every: Mapped[int] = mapped_column(Integer)
    # {"kind": "baseline" | "rl" | "manual", "ref": ..., "params": {...}}
    strategy: Mapped[dict] = mapped_column(JSON)


class PaperOrder(Base):
    __tablename__ = "paper_orders"

    order_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("paper_accounts.account_id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    decision_date: Mapped[date] = mapped_column(Date)
    symbol: Mapped[str] = mapped_column(String(32))
    target_weight: Mapped[float] = mapped_column(Float)


class Job(Base):
    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)  # queued|running|succeeded|failed
    params: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
