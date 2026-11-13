"""News pipeline: fetch -> de-duplicate -> analyze -> validate -> aggregate -> write.

Produces one versioned run directory (see ``ml.llm.runs``). Every article that enters
the run ends up either with accepted events or in ``rejected.jsonl`` with a reason.

A provider that is unreachable is not papered over: the affected article is recorded
as ``llm_error`` and after ``MAX_CONSECUTIVE_FAILURES`` in a row the run stops with an
error instead of writing an empty result that looks like "no news".
"""

import logging
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from ml.llm.config import LLMConfig
from ml.llm.extraction import prompts
from ml.llm.extraction.analyzer import ArticleAnalysis, NewsAnalyzer
from ml.llm.ingestion.base import NewsSource
from ml.llm.ingestion.dedup import consolidate
from ml.llm.providers import LLMClient, LLMUnavailableError
from ml.llm.runs import write_run
from ml.llm.schema import AnalyzedEvent, NewsArticle, Rejection, RejectionReason, now_utc
from ml.llm.securities import SecuritiesMaster
from ml.llm.sentiment.aggregation import FEATURE_COLUMNS, aggregate_news_features

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_FAILURES = 3


@dataclass
class NewsRun:
    run_id: str
    path: Path
    manifest: dict
    articles: list[NewsArticle]
    events: list[AnalyzedEvent]
    rejections: list[Rejection]
    features: pd.DataFrame


def analyze_articles(
    analyzer: NewsAnalyzer,
    articles: Sequence[NewsArticle],
    *,
    now: datetime,
    progress: Callable[[int, int, ArticleAnalysis], None] | None = None,
    workers: int = 1,
) -> list[ArticleAnalysis]:
    """Analyze articles (``workers`` at a time); stop if the provider keeps being
    unreachable. Results keep the input order whatever ``workers`` is."""

    def attempt(article: NewsArticle) -> ArticleAnalysis | LLMUnavailableError:
        try:
            return analyzer.analyze(article, now=now)
        except LLMUnavailableError as error:
            return error

    results: list[ArticleAnalysis] = []
    consecutive_failures = 0
    batch = max(1, workers) * 4
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for offset in range(0, len(articles), batch):
            chunk = articles[offset : offset + batch]
            outcomes = list(pool.map(attempt, chunk)) if pool else [attempt(a) for a in chunk]
            for article, outcome in zip(chunk, outcomes, strict=True):
                if isinstance(outcome, LLMUnavailableError):
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        raise outcome
                    logger.warning("article %s: %s", article.article_id, outcome)
                    analysis = ArticleAnalysis(article=article, llm_called=True)
                    analysis.rejections.append(
                        Rejection(
                            article_id=article.article_id,
                            reason=RejectionReason.LLM_ERROR,
                            detail=str(outcome)[:500],
                        )
                    )
                else:
                    analysis = outcome
                    if analysis.llm_called:  # an article that needed no call proves nothing
                        consecutive_failures = 0
                results.append(analysis)
                if progress:
                    progress(len(results), len(articles), analysis)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    return results


def run_news_pipeline(
    config: LLMConfig,
    client: LLMClient,
    source: NewsSource,
    master: SecuritiesMaster,
    symbols: Sequence[str],
    *,
    output_root: Path | None = None,
    now: datetime | None = None,
    max_articles: int | None = None,
    progress: Callable[[int, int, ArticleAnalysis], None] | None = None,
    workers: int = 1,
) -> NewsRun:
    """Run the whole pipeline once and write ``<output_root>/<run_id>/``.

    Inputs:  config, an LLM client, a news source, the securities master, the symbols
             to fetch news and build features for. ``max_articles`` keeps only the most
             recent N unique articles (recorded in the manifest).
    Output:  ``NewsRun`` with the run path, manifest and in-memory results.
    """
    now = now or now_utc()
    fetched = source.fetch(symbols)
    unique, groups = consolidate(fetched, config.dedup)
    skipped = 0
    if max_articles is not None and len(unique) > max_articles:
        skipped = len(unique) - max_articles
        unique = unique[-max_articles:]

    analyzer = NewsAnalyzer(client, master, config.extraction)
    analyses = analyze_articles(analyzer, unique, now=now, progress=progress, workers=workers)
    events = [event for analysis in analyses for event in analysis.events]
    rejections = [r for analysis in analyses for r in analysis.rejections]

    zone = ZoneInfo(config.aggregation.market_timezone)
    last_day = pd.Timestamp(now.astimezone(zone).date())
    stamps = [a.event.publication_timestamp.astimezone(zone).date() for a in events]
    first_day = min([pd.Timestamp(d) for d in stamps], default=last_day)
    days = list(pd.bdate_range(first_day, last_day)) or [last_day]
    features = aggregate_news_features(
        [a.event for a in events],
        list(symbols),
        days,
        config.aggregation,
        publishers={a.event.event_id: a.source for a in events},
    )

    config_hash = config.config_hash()
    run_id = f"news-{now:%Y%m%dT%H%M%SZ}-{config_hash[:8]}"
    called = [a for a in analyses if a.llm_called]
    latencies = [a.latency_seconds for a in called if a.latency_seconds is not None]
    manifest = {
        "run_id": run_id,
        "created_at": now.isoformat(),
        "llm": {
            "provider": client.provider,
            "model": client.model,
            "base_url": config.provider.base_url,
        },
        "prompt_version": prompts.PROMPT_VERSION,
        "config_hash": config_hash,
        "config": config.model_dump(mode="json"),
        "source": {**source.info.to_dict(), "fetch_log": source.fetch_log()},
        "symbols": list(symbols),
        "counts": {
            "articles_fetched": len(fetched),
            "articles_unique": len(unique) + skipped,
            "articles_skipped_by_limit": skipped,
            "articles_analyzed": len(unique),
            "duplicate_groups": len(groups),
            "duplicates_merged": sum(len(g.merged) for g in groups),
            "llm_calls": len(called),
            "llm_calls_schema_valid": sum(1 for a in called if a.schema_valid),
            "events": len(events),
            "events_unknown": sum(1 for a in events if a.event.event_type.value == "unknown"),
            "rejections": len(rejections),
        },
        "rejections_by_reason": dict(sorted(Counter(r.reason.value for r in rejections).items())),
        "flags_by_flag": dict(
            sorted(Counter(f.flag.value for a in events for f in a.flags).items())
        ),
        "duplicate_groups": [g.to_dict() for g in groups],
        "llm_seconds_total": round(sum(latencies), 2),
        "feature_columns": FEATURE_COLUMNS,
        "feature_dates": {
            "first": f"{days[0]:%Y-%m-%d}",
            "last": f"{days[-1]:%Y-%m-%d}",
            "note": "Monday-Friday dates; not an exchange calendar.",
        },
    }
    path = Path(output_root or config.output.root) / run_id
    manifest = write_run(path, manifest, unique, events, rejections, features)
    return NewsRun(run_id, path, manifest, unique, events, rejections, features)
