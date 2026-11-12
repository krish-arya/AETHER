"""Data contracts of the LLM engine: news articles in, financial events out.

``FinancialEvent`` is the strict schema of PRD 8.3. ``sentiment`` (the tone of the
news) and ``impact_direction`` (the likely effect on the company's finances) are kept
as separate fields on purpose (PRD 8.2-D): revenue can rise while profitability falls.

Every timestamp is timezone-aware and normalised to UTC. A naive timestamp is rejected
rather than guessed, because the point-in-time rules of the aggregation depend on it.
"""

import hashlib
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"
    MIXED = "mixed"


class EventType(StrEnum):
    """PRD 8.2-B event categories, plus ``other`` and ``unknown``.

    ``other``   a real event that fits none of the categories.
    ``unknown`` the article does not contain enough information to tell (PRD 8.4:
                "fallback to unknown when information is insufficient").
    """

    EARNINGS = "earnings"
    MERGER_ACQUISITION = "merger_acquisition"
    MANAGEMENT_CHANGE = "management_change"
    REGULATORY = "regulatory"
    LITIGATION = "litigation"
    DEBT = "debt"
    NEW_CONTRACT = "new_contract"
    PRODUCT_LAUNCH = "product_launch"
    CORPORATE_GOVERNANCE = "corporate_governance"
    DIVIDEND = "dividend"
    FUNDRAISING = "fundraising"
    OTHER = "other"
    UNKNOWN = "unknown"


class ImpactDirection(StrEnum):
    POTENTIALLY_POSITIVE = "potentially_positive"
    POTENTIALLY_NEGATIVE = "potentially_negative"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"


class ImpactHorizon(StrEnum):
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"
    UNKNOWN = "unknown"


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _clean_list(values: list[str]) -> list[str]:
    return [text for text in (" ".join(str(v).split()) for v in values) if text]


class NewsArticle(BaseModel):
    """One news item (PRD 5.3 metadata).

    ``text`` holds the full text or a permitted excerpt; it is empty for sources that
    only provide headlines. ``symbols`` are tickers the *source* attached to the item
    (hints for entity resolution, not trusted labels). ``duplicate_ids`` lists the
    articles that were consolidated into this one (see ``ingestion.dedup``).
    """

    model_config = ConfigDict(extra="forbid")

    article_id: str = Field(min_length=1, max_length=96)
    headline: str = Field(min_length=1, max_length=1000)
    text: str = ""
    source: str = Field(min_length=1, max_length=128)
    url: str | None = None
    published_at: datetime
    retrieved_at: datetime
    symbols: list[str] = []
    duplicate_ids: list[str] = []
    duplicate_sources: list[str] = []

    _utc = field_validator("published_at", "retrieved_at")(_to_utc)

    @field_validator("headline", "source")
    @classmethod
    def _single_line(cls, value: str) -> str:
        return " ".join(value.split())

    @property
    def source_ids(self) -> list[str]:
        """Ids this article stands for: its own plus every consolidated duplicate."""
        return [self.article_id, *self.duplicate_ids]

    @property
    def content(self) -> str:
        """Everything the model is allowed to ground a statement on."""
        return f"{self.headline}\n{self.text}".strip()

    @property
    def headline_only(self) -> bool:
        return not self.text.strip()


def make_article_id(source: str, url: str | None, headline: str, published_at: datetime) -> str:
    """Deterministic id for an article that arrives without one."""
    key = "|".join([source, url or "", " ".join(headline.split()), published_at.isoformat()])
    return "art_" + hashlib.sha256(key.encode()).hexdigest()[:20]


class FinancialEvent(BaseModel):
    """A validated financial event extracted from one or more articles (PRD 8.3)."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1, max_length=96)
    company: str = Field(min_length=1, max_length=255)
    symbol: str = Field(min_length=1, max_length=32)
    event_type: EventType
    sentiment: Sentiment
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    impact_direction: ImpactDirection
    impact_horizon: ImpactHorizon
    severity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    key_facts: list[str] = []
    risks: list[str] = []
    source_ids: list[str] = Field(min_length=1)
    publication_timestamp: datetime

    _utc = field_validator("publication_timestamp")(_to_utc)
    _lists = field_validator("key_facts", "risks")(_clean_list)


def make_event_id(article_id: str, symbol: str) -> str:
    """Deterministic: re-analysing the same article yields the same event id."""
    return "evt_" + hashlib.sha256(f"{article_id}|{symbol}".encode()).hexdigest()[:20]


class RejectionReason(StrEnum):
    """Why a model output (or an article) was refused. Nothing is dropped silently."""

    # article level, before any model call
    ARTICLE_IN_FUTURE = "article_timestamp_in_future"
    NO_KNOWN_ENTITY = "no_known_entity"
    INJECTION_SUSPECTED = "injection_suspected"
    # model call level
    LLM_ERROR = "llm_error"
    SCHEMA_INVALID = "schema_invalid"
    # event level
    UNKNOWN_SYMBOL = "unknown_symbol"
    ENTITY_NOT_IN_ARTICLE = "entity_not_in_article"
    SOURCE_ATTRIBUTION_INVALID = "source_attribution_invalid"
    TIMESTAMP_MISMATCH = "timestamp_mismatch"
    UNGROUNDED_NUMBER = "ungrounded_number"
    SENTIMENT_INCONSISTENT = "sentiment_inconsistent"
    DUPLICATE_EVENT = "duplicate_event"


class Flag(StrEnum):
    """An accepted event that was altered or deserves a second look, and why."""

    UNGROUNDED_FACT_REMOVED = "ungrounded_fact_removed"
    LOW_CONFIDENCE_FALLBACK_UNKNOWN = "low_confidence_fallback_unknown"
    NO_EVIDENCE_FALLBACK_UNKNOWN = "no_evidence_fallback_unknown"
    CONFIDENCE_CAPPED_HEADLINE_ONLY = "confidence_capped_headline_only"
    ENTITY_FROM_HINT_ONLY = "entity_from_hint_only"
    INJECTION_SUSPECTED = "injection_suspected"
    ARTICLE_TRUNCATED = "article_truncated"


class Rejection(BaseModel):
    """One refused output, with enough context to audit it."""

    model_config = ConfigDict(extra="forbid")

    article_id: str
    reason: RejectionReason
    detail: str = ""
    symbol: str | None = None
    raw_output: dict | list | str | None = None


class EventFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flag: Flag
    detail: str = ""


class AnalyzedEvent(BaseModel):
    """An accepted event plus the flags raised while validating it and its provenance."""

    model_config = ConfigDict(extra="forbid")

    event: FinancialEvent
    flags: list[EventFlag] = []
    article_id: str
    source: str
    llm_provider: str
    llm_model: str
    prompt_version: str


def now_utc() -> datetime:
    return datetime.now(UTC)
