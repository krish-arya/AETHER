from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import create_async_engine

from app.api.v1.health import get_health_probes
from app.core.config import APP_VERSION
from app.schemas.health import ComponentStatus
from app.services.health import check_database, check_redis


def _probe(name: str, status: str):
    async def probe() -> ComponentStatus:
        return ComponentStatus(name=name, status=status, latency_ms=1.0)

    return probe


async def test_liveness_reports_ok_without_dependencies(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == APP_VERSION
    assert body["service"] == "Project AETHER"


async def test_readiness_ok_when_all_dependencies_up(app, client):
    app.dependency_overrides[get_health_probes] = lambda: [
        _probe("database", "ok"),
        _probe("redis", "ok"),
    ]
    response = await client.get("/api/v1/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_readiness_returns_503_when_a_dependency_is_down(app, client):
    app.dependency_overrides[get_health_probes] = lambda: [
        _probe("database", "ok"),
        _probe("redis", "down"),
    ]
    response = await client.get("/api/v1/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert {c["name"]: c["status"] for c in body["components"]} == {
        "database": "ok",
        "redis": "down",
    }


async def test_database_probe_ok_against_real_engine(migrated_db_url):
    engine = create_async_engine(migrated_db_url)
    try:
        result = await check_database(engine, timeout=5)
    finally:
        await engine.dispose()
    assert result.status == "ok"
    assert result.latency_ms is not None


async def test_database_probe_reports_down_without_leaking_credentials():
    engine = create_async_engine("postgresql+asyncpg://user:s3cret@127.0.0.1:1/nope")
    try:
        result = await check_database(engine, timeout=3)
    finally:
        await engine.dispose()
    assert result.status == "down"
    assert "s3cret" not in (result.detail or "")


async def test_redis_probe_reports_down_when_unreachable():
    client = Redis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=1)
    try:
        result = await check_redis(client, timeout=3)
    finally:
        await client.aclose()
    assert result.status == "down"


async def test_system_info_exposes_disclaimers_and_module_status(client):
    response = await client.get("/api/v1/system/info")
    assert response.status_code == 200
    body = response.json()
    assert any("SEBI" in d for d in body["disclaimers"])
    assert any("does not guarantee future results" in d for d in body["disclaimers"])
    statuses = {m["key"]: m["status"] for m in body["modules"]}
    # Engines that are not built yet must not be advertised as available.
    assert statuses["data"] == "available"
    # Hardening (auth, monitoring, deployment) is not done and must not claim otherwise.
    assert statuses["hardening"] == "planned"
    summaries = {m["key"]: m["summary"] for m in body["modules"]}
    assert "FAIL their quality gates" in summaries["gan"]  # stated, not hidden
    assert "unverified" in summaries["llm"]
