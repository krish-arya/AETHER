"""PostgreSQL integration tests (the production database engine).

Skipped unless AETHER_TEST_POSTGRES_URL points at a reachable server, e.g. the one
from ``docker compose up -d postgres``::

    AETHER_TEST_POSTGRES_URL=postgresql+asyncpg://aether:aether@localhost:5432/aether \
        uv run pytest -m integration

Each run creates its own scratch database and drops it afterwards; the database named
in the URL is only used to issue CREATE / DROP DATABASE.
"""

import asyncio
import os
from datetime import date
from uuid import uuid4

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import func, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.models  # noqa: F401
from app.database.base import Base
from app.models import CorporateAction, DatasetVersion, MarketData, Security, TechnicalFeature
from app.services.dataset_loader import load_dataset
from ml.data.config import DataConfig
from ml.data.pipeline import build_dataset

from .conftest import alembic_config

pytestmark = pytest.mark.integration

ADMIN_URL = os.environ.get("AETHER_TEST_POSTGRES_URL")


async def _admin(statement: str) -> None:
    engine = create_async_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as connection:
            await connection.execute(text(statement))
    finally:
        await engine.dispose()


@pytest.fixture(scope="module")
def postgres_url():
    if not ADMIN_URL:
        pytest.skip("AETHER_TEST_POSTGRES_URL is not set")
    name = f"aether_test_{uuid4().hex[:10]}"
    asyncio.run(_admin(f'CREATE DATABASE "{name}"'))
    url = make_url(ADMIN_URL).set(database=name).render_as_string(hide_password=False)
    command.upgrade(alembic_config(url), "head")
    yield url
    asyncio.run(_admin(f'DROP DATABASE "{name}" WITH (FORCE)'))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    cfg = DataConfig.model_validate(
        {
            "dataset": {
                "name": "pg-test",
                "start_date": date(2020, 1, 1),
                "end_date": date(2021, 12, 31),
            },
            "provider": {"kind": "synthetic_dev", "synthetic_dev": {"seed": 5, "n_symbols": 4}},
        }
    )
    return build_dataset(cfg, output_root=tmp_path_factory.mktemp("datasets"))


def test_migrations_match_models_on_postgres(postgres_url):
    async def diff():
        engine = create_async_engine(postgres_url)
        try:
            async with engine.connect() as connection:
                return await connection.run_sync(
                    lambda sync: compare_metadata(MigrationContext.configure(sync), Base.metadata)
                )
        finally:
            await engine.dispose()

    assert asyncio.run(diff()) == []


def test_dataset_loads_idempotently_into_postgres(postgres_url, built):
    async def scenario():
        engine = create_async_engine(postgres_url)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with factory() as session:
                first = await load_dataset(session, built.path)
            async with factory() as session:
                second = await load_dataset(session, built.path)
            async with factory() as session:
                counts = {
                    model.__tablename__: await session.scalar(
                        select(func.count()).select_from(model)
                    )
                    for model in (
                        DatasetVersion, Security, MarketData, CorporateAction, TechnicalFeature,
                    )
                }  # fmt: skip
                first_bar_date = await session.scalar(select(func.min(MarketData.trade_date)))
                warm_up = await session.scalar(
                    select(TechnicalFeature.features)
                    .where(TechnicalFeature.symbol == "SYNTH01")
                    .order_by(TechnicalFeature.trade_date)
                    .limit(1)
                )
                last_close = await session.scalar(
                    select(MarketData.close)
                    .where(MarketData.symbol == "SYNTH01")
                    .order_by(MarketData.trade_date.desc())
                    .limit(1)
                )
            return first, second, counts, first_bar_date, warm_up, last_close
        finally:
            await engine.dispose()

    first, second, counts, first_bar_date, warm_up, last_close = asyncio.run(scenario())
    assert first == second
    assert counts == {
        "dataset_versions": 1,
        "securities": 4,
        "market_data": len(built.bars),
        "corporate_actions": built.manifest["counts"]["corporate_actions"],
        "technical_features": len(built.features),
    }
    assert first_bar_date == built.bars["trade_date"].min().date()
    assert warm_up["sma_50_ratio"] is None  # NaN stored as JSON null, not as a number
    source = built.bars[built.bars["symbol"] == "SYNTH01"]
    assert last_close == source["close"].iloc[-1]


def test_downgrade_and_upgrade_round_trip_on_postgres(postgres_url):
    config = alembic_config(postgres_url)
    command.downgrade(config, "base")
    command.upgrade(config, "head")
