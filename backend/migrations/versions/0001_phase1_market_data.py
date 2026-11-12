"""Phase 1: dataset versions, securities, market data, corporate actions, features.

Revision ID: 0001
Revises:
Create Date: 2026-11-11
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "dataset_versions",
        sa.Column("dataset_id", sa.String(96), nullable=False),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("is_synthetic", sa.Boolean(), nullable=False),
        sa.Column("license_note", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("feature_set_version", sa.String(16), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("symbol_count", sa.Integer(), nullable=False),
        sa.Column("bar_count", sa.Integer(), nullable=False),
        sa.Column("quality_summary", sa.JSON(), nullable=False),
        sa.Column("manifest_path", sa.String(512), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("loaded_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("dataset_id", name=op.f("pk_dataset_versions")),
    )

    op.create_table(
        "securities",
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("sector", sa.String(64), nullable=True),
        sa.Column("industry", sa.String(128), nullable=True),
        sa.Column("isin", sa.String(12), nullable=True),
        sa.Column("listing_date", sa.Date(), nullable=True),
        sa.Column("delisting_date", sa.Date(), nullable=True),
        sa.Column("is_synthetic", sa.Boolean(), nullable=False),
        sa.Column("source", sa.String(64), nullable=False),
        sa.Column("dataset_id", sa.String(96), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["dataset_versions.dataset_id"],
            name=op.f("fk_securities_dataset_id_dataset_versions"),
        ),
        sa.PrimaryKeyConstraint("symbol", name=op.f("pk_securities")),
    )

    op.create_table(
        "market_data",
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("adjusted_close", sa.Float(), nullable=False),
        sa.Column("adj_factor", sa.Float(), nullable=False),
        sa.Column("split_factor", sa.Float(), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=False),
        sa.Column("turnover", sa.Float(), nullable=True),
        sa.Column("delivery_volume", sa.BigInteger(), nullable=True),
        sa.Column("market_cap", sa.Float(), nullable=True),
        sa.Column("source", sa.String(64), nullable=False),
        sa.Column("is_synthetic", sa.Boolean(), nullable=False),
        sa.Column("dataset_id", sa.String(96), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["symbol"], ["securities.symbol"], name=op.f("fk_market_data_symbol_securities")
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["dataset_versions.dataset_id"],
            name=op.f("fk_market_data_dataset_id_dataset_versions"),
        ),
        sa.PrimaryKeyConstraint("symbol", "trade_date", name=op.f("pk_market_data")),
    )
    op.create_index(op.f("ix_market_data_trade_date"), "market_data", ["trade_date"])

    op.create_table(
        "corporate_actions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("ex_date", sa.Date(), nullable=False),
        sa.Column("action_type", sa.String(16), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("announced_date", sa.Date(), nullable=True),
        sa.Column("source", sa.String(64), nullable=False),
        sa.Column("is_synthetic", sa.Boolean(), nullable=False),
        sa.Column("dataset_id", sa.String(96), nullable=False),
        sa.ForeignKeyConstraint(
            ["symbol"],
            ["securities.symbol"],
            name=op.f("fk_corporate_actions_symbol_securities"),
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["dataset_versions.dataset_id"],
            name=op.f("fk_corporate_actions_dataset_id_dataset_versions"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_corporate_actions")),
        sa.UniqueConstraint(
            "symbol", "ex_date", "action_type", name=op.f("uq_corporate_actions_symbol")
        ),
    )
    op.create_index(op.f("ix_corporate_actions_symbol"), "corporate_actions", ["symbol"])

    op.create_table(
        "technical_features",
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("feature_set_version", sa.String(16), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column("dataset_id", sa.String(96), nullable=False),
        sa.ForeignKeyConstraint(
            ["symbol"],
            ["securities.symbol"],
            name=op.f("fk_technical_features_symbol_securities"),
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["dataset_versions.dataset_id"],
            name=op.f("fk_technical_features_dataset_id_dataset_versions"),
        ),
        sa.PrimaryKeyConstraint(
            "symbol", "trade_date", "feature_set_version", name=op.f("pk_technical_features")
        ),
    )
    op.create_index(op.f("ix_technical_features_trade_date"), "technical_features", ["trade_date"])


def downgrade() -> None:
    op.drop_index(op.f("ix_technical_features_trade_date"), table_name="technical_features")
    op.drop_table("technical_features")
    op.drop_index(op.f("ix_corporate_actions_symbol"), table_name="corporate_actions")
    op.drop_table("corporate_actions")
    op.drop_index(op.f("ix_market_data_trade_date"), table_name="market_data")
    op.drop_table("market_data")
    op.drop_table("securities")
    op.drop_table("dataset_versions")
