"""User-supplied news from a JSON Lines file.

One JSON object per line:

    {"headline": "...", "source": "...", "published_at": "2026-11-10T09:30:00+05:30",
     "text": "...", "url": "...", "symbols": ["RELIANCE"], "article_id": "..."}

``headline``, ``source`` and ``published_at`` (timezone-aware ISO 8601) are required.
``article_id`` defaults to a hash of source, url, headline and publication time;
``retrieved_at`` defaults to the file's modification time.

This is the route for licensed or archived news (with full text and history), which
the free Yahoo source cannot provide. A malformed line is an error, never skipped.
"""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from ml.llm.ingestion.base import NewsSource, NewsSourceError, NewsSourceInfo
from ml.llm.schema import NewsArticle, make_article_id


class JsonlNewsSource(NewsSource):
    def __init__(self, path: str | Path, *, name: str = "user_jsonl", license_note: str = ""):
        self.path = Path(path)
        self.name = name
        self.license_note = license_note or (
            "User-supplied file; the user is responsible for the right to use its content."
        )
        self._log: list[dict] = []

    @property
    def info(self) -> NewsSourceInfo:
        return NewsSourceInfo(
            name=self.name,
            license_note=self.license_note,
            provides_body=True,
            provides_history=True,
        )

    def fetch_log(self) -> list[dict]:
        return list(self._log)

    def _parse(self, line: str, number: int, default_retrieved: datetime) -> NewsArticle:
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("expected a JSON object")
            record.setdefault("retrieved_at", default_retrieved.isoformat())
            if not record.get("article_id"):
                published = datetime.fromisoformat(str(record["published_at"]))
                record["article_id"] = make_article_id(
                    str(record["source"]), record.get("url"), str(record["headline"]), published
                )
            return NewsArticle.model_validate(record)
        except (ValueError, KeyError, TypeError, ValidationError) as error:
            message = " ".join(str(error).split())[:300]
            raise NewsSourceError(f"{self.path}:{number}: invalid article: {message}") from error

    def fetch(self, symbols: Sequence[str] = ()) -> list[NewsArticle]:
        """Every article in the file (``symbols`` is ignored: relevance is decided by
        entity resolution, not by the source)."""
        if not self.path.is_file():
            raise NewsSourceError(f"news file not found: {self.path}")
        default_retrieved = datetime.fromtimestamp(self.path.stat().st_mtime, UTC)
        articles: list[NewsArticle] = []
        with open(self.path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if line.strip():
                    articles.append(self._parse(line, number, default_retrieved))
        self._log.append({"file": self.path.as_posix(), "items": len(articles)})
        return articles
