"""Dependency health probes used by the readiness endpoint.

Probes never raise: a failing dependency is reported as ``down``. Exception details
are logged server-side; only the exception class is returned to clients so connection
strings and hostnames are not leaked.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from time import perf_counter

from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.schemas.health import ComponentStatus

logger = logging.getLogger(__name__)

HealthProbe = Callable[[], Awaitable[ComponentStatus]]


async def _run_probe(name: str, probe: Callable[[], Awaitable[None]], timeout: float):
    started = perf_counter()
    try:
        async with asyncio.timeout(timeout):
            await probe()
    except Exception as exc:  # noqa: BLE001 - a probe must report, not raise
        logger.warning("health probe %s failed: %r", name, exc)
        return ComponentStatus(name=name, status="down", detail=type(exc).__name__)
    latency_ms = round((perf_counter() - started) * 1000, 2)
    return ComponentStatus(name=name, status="ok", latency_ms=latency_ms)


async def check_database(engine: AsyncEngine, timeout: float) -> ComponentStatus:
    async def probe() -> None:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    return await _run_probe("database", probe, timeout)


async def check_redis(client: Redis, timeout: float) -> ComponentStatus:
    async def probe() -> None:
        await client.ping()

    return await _run_probe("redis", probe, timeout)
