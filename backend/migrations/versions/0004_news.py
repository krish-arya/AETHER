"""Phase 3: news articles and extracted financial events.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NEWS_INDEXES = ("published_at", "run_id")
EVENT_INDEXES = ("news_id", "symbol", "sector", "event_type", "publication_timestamp", "run_id")


def upgrade() -> None:
    op.create_table(
        "financial_news",
        sa.Column("news_id", sa.String(96), nullable=False),
        sa.Column("headline", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("has_body", sa.Boolean(), nullable=False),
        sa.Column("source", sa.String(128), nullable=False),
        sa.Column("news_source", sa.String(64), nullable=False),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbols", sa.JSON(), nullable=False),
        sa.Column("duplicate_ids", sa.JSON(), nullable=False),
        sa.Column("duplicate_sources", sa.JSON(), nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("llm_provider", sa.String(64), nullable=False),
        sa.Column("llm_model", sa.String(128), nullable=False),
        sa.Column("prompt_version", sa.String(32), nullable=False),
        sa.Column("loaded_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("news_id", name=op.f("pk_financial_news")),
    )
    for column in NEWS_INDEXES:
        op.create_index(op.f(f"ix_financial_news_{column}"), "financial_news", [column])

    op.create_table(
        "news_events",
        sa.Column("event_id", sa.String(96), nullable=False),
        sa.Column("news_id", sa.String(96), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("company", sa.String(255), nullable=False),
        sa.Column("sector", sa.String(64), nullable=True),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("sentiment", sa.String(16), nullable=False),
        sa.Column("sentiment_score", sa.Float(), nullable=False),
        sa.Column("impact_direction", sa.String(24), nullable=False),
        sa.Column("impact_horizon", sa.String(16), nullable=False),
        sa.Column("severity", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("key_facts", sa.JSON(), nullable=False),
        sa.Column("risks", sa.JSON(), nullable=False),
        sa.Column("source_ids", sa.JSON(), nullable=False),
        sa.Column("flags", sa.JSON(), nullable=False),
        sa.Column("publication_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("publisher", sa.String(128), nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("llm_provider", sa.String(64), nullable=False),
        sa.Column("llm_model", sa.String(128), nullable=False),
        sa.Column("prompt_version", sa.String(32), nullable=False),
        sa.ForeignKeyConstraint(
            ["news_id"],
            ["financial_news.news_id"],
            name=op.f("fk_news_events_news_id_financial_news"),
        ),
        sa.PrimaryKeyConstraint("event_id", name=op.f("pk_news_events")),
    )
    for column in EVENT_INDEXES:
        op.create_index(op.f(f"ix_news_events_{column}"), "news_events", [column])


def downgrade() -> None:
    for column in EVENT_INDEXES:
        op.drop_index(op.f(f"ix_news_events_{column}"), table_name="news_events")
    op.drop_table("news_events")
    for column in NEWS_INDEXES:
        op.drop_index(op.f(f"ix_financial_news_{column}"), table_name="financial_news")
    op.drop_table("financial_news")
