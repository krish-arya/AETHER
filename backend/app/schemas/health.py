from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class LivenessResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    environment: str
    time: datetime


class ComponentStatus(BaseModel):
    name: str
    status: Literal["ok", "down"]
    latency_ms: float | None = None
    detail: str | None = None


class ReadinessResponse(BaseModel):
    status: Literal["ok", "degraded"]
    components: list[ComponentStatus]
    time: datetime


class ModuleOut(BaseModel):
    key: str
    name: str
    phase: int
    status: Literal["available", "in_progress", "planned"]
    summary: str


class SystemInfo(BaseModel):
    name: str
    version: str
    environment: str
    database_backend: str
    disclaimers: list[str]
    modules: list[ModuleOut]
