"""Phase 3 persistence models: news articles and the financial events extracted from them.

Rows are loaded from a verified news run directory (``ml.llm.runs``). Every row records
the run, model, provider and prompt version that produced it, so any stored event can
be traced to the article and the model output it came from. Timestamps are UTC.
"""

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class FinancialNews(Base):
    """One de-duplicated news article (PRD 5.3 metadata)."""

    __tablename__ = "financial_news"

    news_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    headline: Mapped[str] = mapped_column(Text)
    # Full text or permitted excerpt; empty for headline-only sources.
    text: Mapped[str] = mapped_column(Text)
    has_body: Mapped[bool] = mapped_column(Boolean)
    # Publisher of the article, and the ingestion source it was fetched through.
    source: Mapped[str] = mapped_column(String(128))
    news_source: Mapped[str] = mapped_column(String(64))
    url: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # Tickers the source attached to the item (hints, not validated links).
    symbols: Mapped[list] = mapped_column(JSON)
    # Articles consolidated into this one by duplicate detection.
    duplicate_ids: Mapped[list] = mapped_column(JSON)
    duplicate_sources: Mapped[list] = mapped_column(JSON)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    llm_provider: Mapped[str] = mapped_column(String(64))
    llm_model: Mapped[str] = mapped_column(String(128))
    prompt_version: Mapped[str] = mapped_column(String(32))
    loaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class NewsEvent(Base):
    """A validated financial event (PRD 8.3) linked to its article."""

    __tablename__ = "news_events"

    event_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    news_id: Mapped[str] = mapped_column(ForeignKey("financial_news.news_id"), index=True)
    # Not a foreign key: news can be analysed before a market dataset is loaded.
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    company: Mapped[str] = mapped_column(String(255))
    sector: Mapped[str | None] = mapped_column(String(64), index=True)
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    sentiment: Mapped[str] = mapped_column(String(16))
    sentiment_score: Mapped[float] = mapped_column(Float)
    impact_direction: Mapped[str] = mapped_column(String(24))
    impact_horizon: Mapped[str] = mapped_column(String(16))
    severity: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    key_facts: Mapped[list] = mapped_column(JSON)
    risks: Mapped[list] = mapped_column(JSON)
    source_ids: Mapped[list] = mapped_column(JSON)
    # [{flag, detail}]: what validation changed or wants a human to look at.
    flags: Mapped[list] = mapped_column(JSON)
    publication_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    publisher: Mapped[str] = mapped_column(String(128))
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    llm_provider: Mapped[str] = mapped_column(String(64))
    llm_model: Mapped[str] = mapped_column(String(128))
    prompt_version: Mapped[str] = mapped_column(String(32))
