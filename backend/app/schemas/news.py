"""Response and request models of the news intelligence API (PRD 15.2).

Every response carries ``llm`` / ``models``: the provider, model and prompt version
that produced the analysis it shows.
"""

from datetime import date, datetime

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class LLMInfo(BaseModel):
    provider: str
    model: str
    prompt_version: str
    run_id: str | None = None


class EventFlagOut(BaseModel):
    flag: str
    detail: str = ""


class EventOut(BaseModel):
    """A validated financial event (PRD 8.3) with its provenance."""

    event_id: str
    news_id: str
    company: str
    symbol: str
    sector: str | None
    event_type: str
    sentiment: str
    sentiment_score: float
    impact_direction: str
    impact_horizon: str
    severity: float
    confidence: float
    key_facts: list[str]
    risks: list[str]
    source_ids: list[str]
    publication_timestamp: datetime
    publisher: str
    flags: list[EventFlagOut]
    llm: LLMInfo


class NewsOut(BaseModel):
    news_id: str
    headline: str
    has_body: bool
    source: str
    news_source: str
    url: str | None
    published_at: datetime
    retrieved_at: datetime
    symbols: list[str]
    duplicate_ids: list[str]
    duplicate_sources: list[str]
    llm: LLMInfo
    events: list[EventOut]


class NewsDetailOut(NewsOut):
    text: str


class NewsListResponse(BaseModel):
    count: int
    models: list[LLMInfo]
    items: list[NewsOut]


class EventsResponse(BaseModel):
    symbol: str
    count: int
    models: list[LLMInfo]
    events: list[EventOut]


class SentimentPoint(BaseModel):
    trade_date: date
    features: dict[str, float]


class SentimentResponse(BaseModel):
    """Point-in-time news features of one symbol (``ml.llm.sentiment.aggregation``)."""

    symbol: str
    start: date
    end: date
    method: str
    market_close: str
    market_timezone: str
    events_used: int
    models: list[LLMInfo]
    current: SentimentPoint
    history: list[SentimentPoint]


class AnalyzeRequest(BaseModel):
    """One article to analyse synchronously. Sizes are bounded (PRD 17)."""

    model_config = ConfigDict(extra="forbid")

    headline: str = Field(min_length=3, max_length=500)
    text: str = Field("", max_length=20_000)
    source: str = Field("user_submitted", min_length=1, max_length=128)
    url: str | None = Field(None, max_length=2000)
    published_at: AwareDatetime
    symbols: list[str] = Field([], max_length=20)


class CandidateOut(BaseModel):
    symbol: str
    evidence: str
    matched: str


class RejectionOut(BaseModel):
    reason: str
    detail: str
    symbol: str | None


class AnalyzeResponse(BaseModel):
    article_id: str
    headline_only: bool
    llm: LLMInfo
    llm_called: bool
    candidates: list[CandidateOut]
    events: list[EventOut]
    rejections: list[RejectionOut]
    persisted: bool
