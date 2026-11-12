"""Loads a news run directory (see ``ml.llm.runs``) into the database.

* The run's file hashes are verified against its manifest first; a run that was edited
  after it was written is refused.
* Loading is idempotent: articles and events are upserted on their ids, so re-running a
  load never duplicates rows. An event analysed again by a later run replaces the
  earlier version.
* The whole load is one transaction.

Rejected model outputs are not loaded; they stay in the run's ``rejected.jsonl``.
An event stored by an earlier run is not deleted when a later run rejects it.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import FinancialNews, NewsEvent
from app.services.dataset_loader import _upsert
from ml.llm.runs import load_run
from ml.llm.securities import SecuritiesMaster


@dataclass
class NewsLoadSummary:
    run_id: str
    llm_provider: str
    llm_model: str
    articles: int
    events: int


def _sectors_from_manifest(manifest: dict) -> dict[str, str | None]:
    """symbol -> sector from the universe file the run was made with, if it still exists."""
    path = Path(manifest.get("config", {}).get("universe_file", ""))
    if not path.is_file():
        return {}
    master = SecuritiesMaster.from_yaml(path)
    return {symbol: master.sector(symbol) for symbol in master.symbols}


async def load_news_run(
    session: AsyncSession, run_dir: Path, sectors: Mapping[str, str | None] | None = None
) -> NewsLoadSummary:
    """Upsert one verified news run. ``sectors`` overrides the symbol -> sector lookup."""
    manifest, articles, events = load_run(run_dir)  # raises if any file was modified
    run_id: str = manifest["run_id"]
    llm = manifest["llm"]
    if sectors is None:
        sectors = _sectors_from_manifest(manifest)
    provenance = {
        "run_id": run_id,
        "llm_provider": llm["provider"],
        "llm_model": llm["model"],
        "prompt_version": manifest["prompt_version"],
    }
    now = datetime.now(UTC)

    news_rows = [
        {
            "news_id": article.article_id,
            "headline": article.headline,
            "text": article.text,
            "has_body": not article.headline_only,
            "source": article.source,
            "news_source": manifest["source"]["name"],
            "url": article.url,
            "published_at": article.published_at,
            "retrieved_at": article.retrieved_at,
            "symbols": article.symbols,
            "duplicate_ids": article.duplicate_ids,
            "duplicate_sources": article.duplicate_sources,
            "loaded_at": now,
            **provenance,
        }
        for article in articles
    ]
    await _upsert(session, FinancialNews, news_rows, ["news_id"])

    event_rows = [
        {
            "event_id": analyzed.event.event_id,
            "news_id": analyzed.article_id,
            "symbol": analyzed.event.symbol,
            "company": analyzed.event.company,
            "sector": sectors.get(analyzed.event.symbol),
            "event_type": analyzed.event.event_type.value,
            "sentiment": analyzed.event.sentiment.value,
            "sentiment_score": analyzed.event.sentiment_score,
            "impact_direction": analyzed.event.impact_direction.value,
            "impact_horizon": analyzed.event.impact_horizon.value,
            "severity": analyzed.event.severity,
            "confidence": analyzed.event.confidence,
            "key_facts": analyzed.event.key_facts,
            "risks": analyzed.event.risks,
            "source_ids": analyzed.event.source_ids,
            "flags": [flag.model_dump(mode="json") for flag in analyzed.flags],
            "publication_timestamp": analyzed.event.publication_timestamp,
            "publisher": analyzed.source,
            "run_id": run_id,
            # The event's own provenance (identical to the run's, but authoritative).
            "llm_provider": analyzed.llm_provider,
            "llm_model": analyzed.llm_model,
            "prompt_version": analyzed.prompt_version,
        }
        for analyzed in events
    ]
    await _upsert(session, NewsEvent, event_rows, ["event_id"])

    await session.commit()
    return NewsLoadSummary(
        run_id=run_id,
        llm_provider=llm["provider"],
        llm_model=llm["model"],
        articles=len(news_rows),
        events=len(event_rows),
    )
