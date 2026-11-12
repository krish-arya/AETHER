"""Market regime per session and dataset.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "market_regimes",
        sa.Column("dataset_id", sa.String(96), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("regime", sa.String(24), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("volatility_level", sa.Float(), nullable=True),
        sa.Column("trend_strength", sa.Float(), nullable=True),
        sa.Column("realized_vol", sa.Float(), nullable=True),
        sa.Column("trend_return", sa.Float(), nullable=True),
        sa.Column("drawdown", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["dataset_versions.dataset_id"],
            name=op.f("fk_market_regimes_dataset_id_dataset_versions"),
        ),
        sa.PrimaryKeyConstraint("dataset_id", "trade_date", name=op.f("pk_market_regimes")),
    )


def downgrade() -> None:
    op.drop_table("market_regimes")
