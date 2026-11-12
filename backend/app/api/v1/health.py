"""Liveness, readiness and system-information endpoints."""

import asyncio
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Response, status

from app.core.config import APP_VERSION, DISCLAIMERS, Settings, get_settings
from app.core.modules import MODULES
from app.core.redis import get_redis
from app.database.session import get_engine
from app.schemas.health import (
    LivenessResponse,
    ModuleOut,
    ReadinessResponse,
    SystemInfo,
)
from app.services.health import HealthProbe, check_database, check_redis

router = APIRouter(tags=["system"])


def get_health_probes(settings: Settings = Depends(get_settings)) -> list[HealthProbe]:
    timeout = settings.health_check_timeout_seconds
    return [
        lambda: check_database(get_engine(), timeout),
        lambda: check_redis(get_redis(), timeout),
    ]


@router.get("/health", response_model=LivenessResponse, summary="Liveness probe")
async def liveness(settings: Settings = Depends(get_settings)) -> LivenessResponse:
    """Process is up. Does not touch any external dependency."""
    return LivenessResponse(
        status="ok",
        service=settings.app_name,
        version=APP_VERSION,
        environment=settings.environment,
        time=datetime.now(UTC),
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse, "description": "A dependency is down"}},
    summary="Readiness probe (database + Redis)",
)
async def readiness(
    response: Response, probes: list[HealthProbe] = Depends(get_health_probes)
) -> ReadinessResponse:
    components = list(await asyncio.gather(*(probe() for probe in probes)))
    healthy = all(component.status == "ok" for component in components)
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ok" if healthy else "degraded",
        components=components,
        time=datetime.now(UTC),
    )


@router.get("/system/info", response_model=SystemInfo, summary="Build and module status")
async def system_info(settings: Settings = Depends(get_settings)) -> SystemInfo:
    return SystemInfo(
        name=settings.app_name,
        version=APP_VERSION,
        environment=settings.environment,
        database_backend=settings.database_backend,
        disclaimers=list(DISCLAIMERS),
        modules=[ModuleOut(**module) for module in MODULES],
    )
