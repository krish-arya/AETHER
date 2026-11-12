"""News intelligence endpoints (PRD 15.2).

Stored news and events come from news runs loaded with ``scripts/load_news.py``; the
API never invents an article, an event or a score. Every response names the provider,
model and prompt version behind the analysis it shows.

``POST /news/analyze`` runs the configured LLM synchronously on one supplied article
and returns the validated events together with everything that was rejected. It
answers 503 when no LLM is configured or the provider is unreachable, and does not
store anything.
"""

import asyncio
import logging
from datetime import UTC, date, datetime, time, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_session
from app.models import FinancialNews, NewsEvent
from app.schemas.news import (
    AnalyzeRequest,
    AnalyzeResponse,
    CandidateOut,
    EventOut,
    EventsResponse,
    LLMInfo,
    NewsDetailOut,
    NewsListResponse,
    NewsOut,
    RejectionOut,
    SentimentPoint,
    SentimentResponse,
)
from app.services.news import NOT_CONFIGURED, NewsEngine, get_news_engine
from ml.llm.extraction.analyzer import CLOCK_SKEW, NewsAnalyzer
from ml.llm.extraction.prompts import PROMPT_VERSION
from ml.llm.providers import LLMError, LLMUnavailableError
from ml.llm.schema import EventType, FinancialEvent, NewsArticle, make_article_id, now_utc
from ml.llm.sentiment.aggregation import FEATURE_COLUMNS, aggregate_news_features

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/news", tags=["news"])

MAX_ROWS = 500
MAX_SENTIMENT_DAYS = 366
# Simple back-pressure on an expensive endpoint until real rate limiting (Phase 8).
MAX_CONCURRENT_ANALYSES = 2
_active_analyses = 0


def _utc(value: datetime) -> datetime:
    """SQLite returns naive datetimes; everything stored is UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _validate_range(start: date | None, end: date | None) -> None:
    if start and end and start > end:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "start must not be after end")


def _day_bounds(start: date | None, end: date | None) -> tuple[datetime | None, datetime | None]:
    """[start 00:00 UTC, end + 1 day 00:00 UTC) for inclusive calendar-date filters."""
    lower = datetime.combine(start, time.min, UTC) if start else None
    upper = datetime.combine(end + timedelta(days=1), time.min, UTC) if end else None
    return lower, upper


def _event_out(row: NewsEvent) -> EventOut:
    return EventOut(
        event_id=row.event_id,
        news_id=row.news_id,
        company=row.company,
        symbol=row.symbol,
        sector=row.sector,
        event_type=row.event_type,
        sentiment=row.sentiment,
        sentiment_score=row.sentiment_score,
        impact_direction=row.impact_direction,
        impact_horizon=row.impact_horizon,
        severity=row.severity,
        confidence=row.confidence,
        key_facts=row.key_facts,
        risks=row.risks,
        source_ids=row.source_ids,
        publication_timestamp=_utc(row.publication_timestamp),
        publisher=row.publisher,
        flags=row.flags,
        llm=LLMInfo(
            provider=row.llm_provider,
            model=row.llm_model,
            prompt_version=row.prompt_version,
            run_id=row.run_id,
        ),
    )


def _news_fields(row: FinancialNews, events: list[NewsEvent]) -> dict:
    return {
        "news_id": row.news_id,
        "headline": row.headline,
        "has_body": row.has_body,
        "source": row.source,
        "news_source": row.news_source,
        "url": row.url,
        "published_at": _utc(row.published_at),
        "retrieved_at": _utc(row.retrieved_at),
        "symbols": row.symbols,
        "duplicate_ids": row.duplicate_ids,
        "duplicate_sources": row.duplicate_sources,
        "llm": LLMInfo(
            provider=row.llm_provider,
            model=row.llm_model,
            prompt_version=row.prompt_version,
            run_id=row.run_id,
        ),
        "events": [_event_out(event) for event in events],
    }


def _models(infos: list[LLMInfo]) -> list[LLMInfo]:
    """Distinct (provider, model, prompt version) combinations, in first-seen order."""
    seen: dict[tuple, LLMInfo] = {}
    for info in infos:
        key = (info.provider, info.model, info.prompt_version)
        seen.setdefault(key, LLMInfo(provider=key[0], model=key[1], prompt_version=key[2]))
    return list(seen.values())


def _known_symbol(engine: NewsEngine, symbol: str) -> str:
    symbol = symbol.strip().upper()
    if symbol not in engine.master:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown symbol '{symbol}'")
    return symbol


async def _events_by_news(session: AsyncSession, news_ids: list[str]) -> dict[str, list[NewsEvent]]:
    grouped: dict[str, list[NewsEvent]] = {news_id: [] for news_id in news_ids}
    if news_ids:
        stmt = select(NewsEvent).where(NewsEvent.news_id.in_(news_ids)).order_by(NewsEvent.symbol)
        for event in await session.scalars(stmt):
            grouped[event.news_id].append(event)
    return grouped


@router.get("", response_model=NewsListResponse)
async def list_news(
    symbol: str | None = Query(None, description="Only news with an event for this symbol"),
    sector: str | None = Query(None, description="Only news with an event in this sector"),
    event_type: EventType | None = Query(None, description="Only news with such an event"),
    start: date | None = Query(None, description="Published on or after (UTC date)"),
    end: date | None = Query(None, description="Published on or before (UTC date)"),
    limit: int = Query(50, ge=1, le=MAX_ROWS, description="Most recent N articles"),
    session: AsyncSession = Depends(get_session),
) -> NewsListResponse:
    """Analysed news, newest first, with the events extracted from each article."""
    _validate_range(start, end)
    stmt = select(FinancialNews)
    if symbol or sector or event_type:
        matching = select(NewsEvent.news_id)
        if symbol:
            matching = matching.where(NewsEvent.symbol == symbol.strip().upper())
        if sector:
            matching = matching.where(NewsEvent.sector == sector.strip())
        if event_type:
            matching = matching.where(NewsEvent.event_type == event_type.value)
        stmt = stmt.where(FinancialNews.news_id.in_(matching))
    lower, upper = _day_bounds(start, end)
    if lower:
        stmt = stmt.where(FinancialNews.published_at >= lower)
    if upper:
        stmt = stmt.where(FinancialNews.published_at < upper)
    stmt = stmt.order_by(FinancialNews.published_at.desc(), FinancialNews.news_id).limit(limit)
    rows = list(await session.scalars(stmt))
    events = await _events_by_news(session, [row.news_id for row in rows])
    items = [NewsOut(**_news_fields(row, events[row.news_id])) for row in rows]
    return NewsListResponse(
        count=len(items), models=_models([item.llm for item in items]), items=items
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_news(
    request: AnalyzeRequest, engine: NewsEngine = Depends(get_news_engine)
) -> AnalyzeResponse:
    """Analyse one supplied article with the configured LLM (synchronous, not stored)."""
    global _active_analyses
    now = now_utc()
    if request.published_at > now + CLOCK_SKEW:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "published_at must not be in the future"
        )
    if engine.client is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, NOT_CONFIGURED)
    if _active_analyses >= MAX_CONCURRENT_ANALYSES:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, "Too many analyses in progress; retry shortly"
        )

    published = request.published_at.astimezone(UTC)
    article = NewsArticle(
        article_id=make_article_id(request.source, request.url, request.headline, published),
        headline=request.headline,
        text=request.text,
        source=request.source,
        url=request.url,
        published_at=published,
        retrieved_at=now,
        symbols=request.symbols,
    )
    analyzer = NewsAnalyzer(engine.client, engine.master, engine.config.extraction)
    _active_analyses += 1
    try:
        analysis = await asyncio.to_thread(analyzer.analyze, article, now=now)
    except LLMUnavailableError as error:
        logger.warning("news analyze: provider unavailable: %s", error)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"The LLM provider is unavailable: {error}"
        ) from error
    except LLMError as error:
        logger.warning("news analyze: provider error: %s", error)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"The LLM provider returned an error: {error}"
        ) from error
    finally:
        _active_analyses -= 1

    llm = LLMInfo(
        provider=engine.client.provider, model=engine.client.model, prompt_version=PROMPT_VERSION
    )
    events = [
        EventOut(
            **analyzed.event.model_dump(mode="python"),
            news_id=article.article_id,
            sector=engine.master.sector(analyzed.event.symbol),
            publisher=article.source,
            flags=[flag.model_dump(mode="json") for flag in analyzed.flags],
            llm=llm,
        )
        for analyzed in analysis.events
    ]
    candidates = engine.master.referenced(article)
    return AnalyzeResponse(
        article_id=article.article_id,
        headline_only=article.headline_only,
        llm=llm,
        llm_called=analysis.llm_called,
        candidates=[
            CandidateOut(symbol=m.symbol, evidence=m.evidence, matched=m.matched)
            for m in candidates.values()
        ],
        events=events,
        rejections=[
            RejectionOut(reason=r.reason.value, detail=r.detail, symbol=r.symbol)
            for r in analysis.rejections
        ],
        persisted=False,
    )


@router.get("/sentiment/{symbol}", response_model=SentimentResponse)
async def get_sentiment(
    symbol: str,
    start: date | None = Query(None, description="Defaults to 30 days before end"),
    end: date | None = Query(None, description="Defaults to today (market time zone)"),
    session: AsyncSession = Depends(get_session),
    engine: NewsEngine = Depends(get_news_engine),
) -> SentimentResponse:
    """Point-in-time news features per calendar day: only events published at or before
    the market close of a day count for that day. All-zero features with
    ``news_available = 0`` mean "no analysed news", not "neutral news"."""
    symbol = _known_symbol(engine, symbol)
    aggregation = engine.config.aggregation
    if end is None:
        from zoneinfo import ZoneInfo

        end = now_utc().astimezone(ZoneInfo(aggregation.market_timezone)).date()
    start = start or end - timedelta(days=30)
    _validate_range(start, end)
    if (end - start).days > MAX_SENTIMENT_DAYS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"range exceeds {MAX_SENTIMENT_DAYS} days"
        )

    lower = datetime.combine(start, time.min, UTC) - timedelta(days=aggregation.max_age_days + 1)
    upper = datetime.combine(end + timedelta(days=1), time.min, UTC)
    stmt = select(NewsEvent).where(
        NewsEvent.symbol == symbol,
        NewsEvent.publication_timestamp >= lower,
        NewsEvent.publication_timestamp < upper,
    )
    rows = list(await session.scalars(stmt.order_by(NewsEvent.publication_timestamp)))
    events = [
        FinancialEvent(
            event_id=row.event_id, company=row.company, symbol=row.symbol,
            event_type=row.event_type, sentiment=row.sentiment,
            sentiment_score=row.sentiment_score, impact_direction=row.impact_direction,
            impact_horizon=row.impact_horizon, severity=row.severity, confidence=row.confidence,
            key_facts=row.key_facts, risks=row.risks, source_ids=row.source_ids,
            publication_timestamp=_utc(row.publication_timestamp),
        )
        for row in rows
    ]  # fmt: skip
    days = [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
    frame = aggregate_news_features(
        events, [symbol], days, aggregation, publishers={r.event_id: r.publisher for r in rows}
    )
    history = [
        SentimentPoint(
            trade_date=record["trade_date"].date(),
            features={name: float(record[name]) for name in FEATURE_COLUMNS},
        )
        for record in frame.to_dict("records")
    ]
    infos = [
        LLMInfo(provider=r.llm_provider, model=r.llm_model, prompt_version=r.prompt_version)
        for r in rows
    ]
    return SentimentResponse(
        symbol=symbol,
        start=start,
        end=end,
        method="confidence_reliability_recency_weighted",
        market_close=aggregation.market_close,
        market_timezone=aggregation.market_timezone,
        events_used=len(events),
        models=_models(infos),
        current=history[-1],
        history=history,
    )


@router.get("/events/{symbol}", response_model=EventsResponse)
async def get_events(
    symbol: str,
    event_type: EventType | None = None,
    start: date | None = Query(None, description="Published on or after (UTC date)"),
    end: date | None = Query(None, description="Published on or before (UTC date)"),
    limit: int = Query(100, ge=1, le=MAX_ROWS, description="Most recent N events"),
    session: AsyncSession = Depends(get_session),
    engine: NewsEngine = Depends(get_news_engine),
) -> EventsResponse:
    """Validated events for one symbol, newest first."""
    _validate_range(start, end)
    symbol = _known_symbol(engine, symbol)
    stmt = select(NewsEvent).where(NewsEvent.symbol == symbol)
    if event_type:
        stmt = stmt.where(NewsEvent.event_type == event_type.value)
    lower, upper = _day_bounds(start, end)
    if lower:
        stmt = stmt.where(NewsEvent.publication_timestamp >= lower)
    if upper:
        stmt = stmt.where(NewsEvent.publication_timestamp < upper)
    stmt = stmt.order_by(NewsEvent.publication_timestamp.desc(), NewsEvent.event_id).limit(limit)
    events = [_event_out(row) for row in await session.scalars(stmt)]
    return EventsResponse(
        symbol=symbol,
        count=len(events),
        models=_models([event.llm for event in events]),
        events=events,
    )


@router.get("/{news_id}", response_model=NewsDetailOut)
async def get_news(news_id: str, session: AsyncSession = Depends(get_session)) -> NewsDetailOut:
    row = await session.get(FinancialNews, news_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown news item '{news_id}'")
    events = await _events_by_news(session, [news_id])
    return NewsDetailOut(**_news_fields(row, events[news_id]), text=row.text)
