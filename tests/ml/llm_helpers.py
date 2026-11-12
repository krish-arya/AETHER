"""Shared fakes for the LLM engine tests.

Everything here is fictional: the companies do not exist and the "model" is a scripted
stand-in, so the tests need no network and assert exact behaviour. A fake client is
only ever used in tests -- production code paths take a real ``LLMClient``.
"""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from ml.llm.providers import LLMError
from ml.llm.schema import NewsArticle
from ml.llm.securities import SecuritiesMaster, SecurityInfo

PUBLISHED = datetime(2026, 11, 10, 6, 0, tzinfo=UTC)
NOW = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)


def make_master() -> SecuritiesMaster:
    return SecuritiesMaster(
        [
            SecurityInfo("ZENTARA", "Zentara Infra", "Industrials"),
            SecurityInfo("QUILLON", "Quillon Pharma", "Health Care"),
            SecurityInfo("BRIVEX", "Brivex Software", "Information Technology"),
        ],
        company_aliases={"QUILLON": ["Quillon Pharmaceuticals"]},
        ticker_aliases={"BRVX": "BRIVEX"},
    )


def make_article(**overrides) -> NewsArticle:
    fields = {
        "article_id": "a1",
        "headline": "Zentara Infra wins 450 crore highway contract",
        "text": "Zentara Infra said it won a highway construction contract worth 450 crore "
        "rupees from a state agency. Work will start next quarter.",
        "source": "Fictional Wire",
        "url": "https://example.invalid/a1",
        "published_at": PUBLISHED,
        "retrieved_at": NOW,
        "symbols": [],
    }
    return NewsArticle(**{**fields, **overrides})


def raw_event(**overrides) -> dict:
    event = {
        "company": "Zentara Infra",
        "symbol": "ZENTARA",
        "event_type": "new_contract",
        "sentiment": "positive",
        "sentiment_score": 0.6,
        "impact_direction": "potentially_positive",
        "impact_horizon": "medium_term",
        "severity": 0.5,
        "confidence": 0.9,
        "key_facts": ["Won a highway contract worth 450 crore"],
        "risks": [],
        "source_ids": ["a1"],
        "publication_timestamp": "2026-11-10T06:00:00Z",
    }
    return {**event, **overrides}


class FakeLLM:
    """Scripted ``LLMClient``: returns queued replies and records every call."""

    provider = "fake"
    model = "fake-model"

    def __init__(self, replies: Sequence | Callable | None = None, embeddings: bool = True):
        self.replies = replies if callable(replies) else list(replies or [])
        self.calls: list[tuple[str, str, dict]] = []
        self.embeddings = embeddings

    def complete_json(self, system: str, user: str, json_schema: dict) -> dict:
        self.calls.append((system, user, json_schema))
        reply = (
            self.replies(system, user, json_schema)
            if callable(self.replies)
            else (self.replies.pop(0))
        )
        if isinstance(reply, Exception):
            raise reply
        return reply

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.embeddings:
            raise LLMError("no embedding model is configured")
        # Deterministic bag-of-letters vectors: enough to rank obviously related texts.
        return [[float(text.lower().count(ch)) for ch in "abcdefghijklmnopqrstuvwxyz"]
                for text in texts]  # fmt: skip
