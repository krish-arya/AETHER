"""Operational alerts (PRD 16.4): tell the user when inputs are stale or a model is
being used outside the conditions it was validated under.

Each alert is derived from current state -- the loaded dataset, the model registry and
the job table -- at request time. Nothing is stored and nothing is suppressed.
"""

from datetime import UTC, date, datetime, timedelta

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.database.session import get_session
from app.models import DatasetVersion, Job
from app.services import platform

router = APIRouter(tags=["system"])

STALE_AFTER_DAYS = 5  # calendar days without a new bar before market data counts as stale


class Alert(BaseModel):
    severity: str  # critical | warning | info
    code: str
    message: str


class AlertsResponse(BaseModel):
    checked_at: datetime
    alerts: list[Alert]


def dataset_alerts(dataset: DatasetVersion | None, today: date) -> list[Alert]:
    if dataset is None:
        return [Alert(severity="critical", code="no_dataset", message="No dataset is loaded.")]
    alerts = []
    age = (today - dataset.end_date).days
    if age > STALE_AFTER_DAYS:
        alerts.append(
            Alert(
                severity="warning",
                code="stale_market_data",
                message=f"The latest market data is {age} days old (last bar {dataset.end_date}). "
                "Allocations and paper trading reflect that date, not today.",
            )
        )
    if dataset.is_synthetic:
        alerts.append(
            Alert(
                severity="warning",
                code="synthetic_dataset",
                message="The active dataset is synthetic development data, not market data.",
            )
        )
    quality = dataset.quality_summary or {}
    if (quality.get("by_check") or {}).get("survivorship_unverified"):
        alerts.append(
            Alert(
                severity="info",
                code="survivorship_bias",
                message="The universe has no delisted securities: results carry survivorship bias.",
            )
        )
    return alerts


def model_alerts(settings: Settings, dataset: DatasetVersion | None) -> list[Alert]:
    if dataset is None:
        return []
    alerts = []
    generators = platform.list_cards(settings, "gan", dataset.dataset_id)
    if generators and generators[0]["status"] == "rejected":
        alerts.append(
            Alert(
                severity="warning",
                code="generator_failed_quality_gates",
                message="The newest scenario generator failed its quality gates; its scenarios "
                "understate real market stress.",
            )
        )
    rl_models = platform.list_cards(settings, "rl", dataset.dataset_id)
    if not rl_models:
        alerts.append(
            Alert(
                severity="info",
                code="no_rl_model",
                message="No RL model has been trained on the active dataset.",
            )
        )
    elif not any((c.get("lineage") or {}).get("experiment") == "deployment" for c in rl_models):
        alerts.append(
            Alert(
                severity="info",
                code="no_deployment_model",
                message="No deployment model is registered: allocations use a walk-forward "
                "model whose training window ends well before the latest data.",
            )
        )
    if not (settings.llm_base_url and settings.llm_model):
        alerts.append(
            Alert(
                severity="info",
                code="llm_not_configured",
                message="No LLM endpoint is configured for the API; news analysis on demand "
                "is unavailable.",
            )
        )
    return alerts


@router.get("/system/alerts", response_model=AlertsResponse, summary="Operational alerts")
async def system_alerts(
    session: AsyncSession = Depends(get_session), settings: Settings = Depends(get_settings)
) -> AlertsResponse:
    dataset = await session.scalar(
        select(DatasetVersion).order_by(DatasetVersion.loaded_at.desc()).limit(1)
    )
    now = datetime.now(UTC)
    alerts = dataset_alerts(dataset, now.date()) + model_alerts(settings, dataset)
    failed = await session.scalar(
        select(func.count())
        .select_from(Job)
        .where(Job.status == "failed", Job.created_at >= now - timedelta(days=1))
    )
    if failed:
        alerts.append(
            Alert(
                severity="warning",
                code="failed_jobs",
                message=f"{failed} model job(s) failed in the last 24 hours; see /api/v1/jobs.",
            )
        )
    order = {"critical": 0, "warning": 1, "info": 2}
    return AlertsResponse(
        checked_at=now, alerts=sorted(alerts, key=lambda alert: order[alert.severity])
    )
