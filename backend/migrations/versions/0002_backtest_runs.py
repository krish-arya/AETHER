"""Phase 2: backtest job records.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-11
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "backtest_runs",
        sa.Column("run_id", sa.String(32), nullable=False),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("dataset_id", sa.String(96), nullable=False),
        sa.Column("is_synthetic", sa.Boolean(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("result_id", sa.String(96), nullable=True),
        sa.Column("summary", sa.JSON(), nullable=True),
        sa.Column("artifact_path", sa.String(512), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["dataset_versions.dataset_id"],
            name=op.f("fk_backtest_runs_dataset_id_dataset_versions"),
        ),
        sa.PrimaryKeyConstraint("run_id", name=op.f("pk_backtest_runs")),
    )
    op.create_index(op.f("ix_backtest_runs_status"), "backtest_runs", ["status"])
    op.create_index(op.f("ix_backtest_runs_created_at"), "backtest_runs", ["created_at"])


def downgrade() -> None:
    op.drop_index(op.f("ix_backtest_runs_created_at"), table_name="backtest_runs")
    op.drop_index(op.f("ix_backtest_runs_status"), table_name="backtest_runs")
    op.drop_table("backtest_runs")
