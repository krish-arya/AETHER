from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect

import app.models  # noqa: F401
from app.database.base import Base

from .conftest import alembic_config

EXPECTED_TABLES = {
    "dataset_versions",
    "securities",
    "market_data",
    "corporate_actions",
    "technical_features",
    "backtest_runs",
    "market_regimes",
    "financial_news",
    "news_events",
}


def _sync_engine(sqlite_path):
    return create_engine(f"sqlite:///{sqlite_path.as_posix()}")


def test_upgrade_head_creates_phase1_tables(migrated_db_url, sqlite_path):
    engine = _sync_engine(sqlite_path)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert tables >= EXPECTED_TABLES


def test_migrations_match_orm_models(migrated_db_url, sqlite_path):
    """The migration history must produce exactly the schema the models declare."""
    engine = _sync_engine(sqlite_path)
    try:
        with engine.connect() as connection:
            diff = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    finally:
        engine.dispose()
    assert diff == []


def test_downgrade_base_removes_all_tables(migrated_db_url, sqlite_path):
    command.downgrade(alembic_config(migrated_db_url), "base")
    engine = _sync_engine(sqlite_path)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert tables.isdisjoint(EXPECTED_TABLES)
