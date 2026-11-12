"""Allocations, paper trading and background jobs.

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "allocations",
        sa.Column("allocation_id", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("model_id", sa.String(128), nullable=False),
        sa.Column("dataset_id", sa.String(96), nullable=False),
        sa.Column("is_synthetic", sa.Boolean(), nullable=False),
        sa.Column("weights", sa.JSON(), nullable=False),
        sa.Column("explanation", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["dataset_versions.dataset_id"],
            name=op.f("fk_allocations_dataset_id_dataset_versions"),
        ),
        sa.PrimaryKeyConstraint("allocation_id", name=op.f("pk_allocations")),
    )
    op.create_index(op.f("ix_allocations_created_at"), "allocations", ["created_at"])

    op.create_table(
        "paper_accounts",
        sa.Column("account_id", sa.String(32), nullable=False),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("dataset_id", sa.String(96), nullable=False),
        sa.Column("initial_capital", sa.Float(), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("rebalance_every", sa.Integer(), nullable=False),
        sa.Column("strategy", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["dataset_versions.dataset_id"],
            name=op.f("fk_paper_accounts_dataset_id_dataset_versions"),
        ),
        sa.PrimaryKeyConstraint("account_id", name=op.f("pk_paper_accounts")),
    )
    op.create_index(op.f("ix_paper_accounts_status"), "paper_accounts", ["status"])

    op.create_table(
        "paper_orders",
        sa.Column("order_id", sa.String(32), nullable=False),
        sa.Column("account_id", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decision_date", sa.Date(), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("target_weight", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(
            ["account_id"],
            ["paper_accounts.account_id"],
            name=op.f("fk_paper_orders_account_id_paper_accounts"),
        ),
        sa.PrimaryKeyConstraint("order_id", name=op.f("pk_paper_orders")),
    )
    op.create_index(op.f("ix_paper_orders_account_id"), "paper_orders", ["account_id"])

    op.create_table(
        "jobs",
        sa.Column("job_id", sa.String(32), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("job_id", name=op.f("pk_jobs")),
    )
    op.create_index(op.f("ix_jobs_kind"), "jobs", ["kind"])
    op.create_index(op.f("ix_jobs_status"), "jobs", ["status"])
    op.create_index(op.f("ix_jobs_created_at"), "jobs", ["created_at"])


def downgrade() -> None:
    op.drop_index(op.f("ix_jobs_created_at"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_status"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_kind"), table_name="jobs")
    op.drop_table("jobs")
    op.drop_index(op.f("ix_paper_orders_account_id"), table_name="paper_orders")
    op.drop_table("paper_orders")
    op.drop_index(op.f("ix_paper_accounts_status"), table_name="paper_accounts")
    op.drop_table("paper_accounts")
    op.drop_index(op.f("ix_allocations_created_at"), table_name="allocations")
    op.drop_table("allocations")
