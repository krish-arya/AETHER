"""Shared backend test fixtures.

Database-backed tests run against a throwaway SQLite file migrated with the real
Alembic migrations, so they need no external services. PostgreSQL-specific checks
are marked ``integration`` and skipped unless a live database is configured.
"""

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database.session import get_session
from app.main import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI = REPO_ROOT / "backend" / "alembic.ini"


def alembic_config(database_url: str) -> Config:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


@pytest.fixture
def sqlite_path(tmp_path: Path) -> Path:
    return tmp_path / "aether_test.db"


@pytest.fixture
def migrated_db_url(sqlite_path: Path) -> Iterator[str]:
    """A SQLite database upgraded to the latest migration (sync: Alembic runs its own loop)."""
    url = f"sqlite+aiosqlite:///{sqlite_path.as_posix()}"
    command.upgrade(alembic_config(url), "head")
    yield url


@pytest.fixture
async def session_factory(migrated_db_url: str) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(migrated_db_url)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def db_client(app, session_factory) -> AsyncIterator[AsyncClient]:
    """API client whose database dependency points at the migrated SQLite database."""

    async def _override() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = _override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
