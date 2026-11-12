"""Article -> validated events: entity resolution, one model call, validation.

    article ─► referenced securities (no model) ─► prompt ─► LLM ─► validate_output
                    │ none                                   │ error
                    ▼                                        ▼
            rejected: no_known_entity                rejected: llm_error

The model is only called for articles that reference at least one known security, are
not dated in the future and (by default) contain no instruction-like text aimed at a
model -- those are quarantined as ``injection_suspected``. ``LLMUnavailableError``
(provider down) is raised to the caller, which decides whether to stop; a reply that
is not usable JSON is recorded as a rejection.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ml.llm.config import ExtractionConfig
from ml.llm.extraction import prompts
from ml.llm.extraction.validation import validate_output
from ml.llm.providers import LLMClient, LLMResponseError
from ml.llm.schema import (
    AnalyzedEvent,
    Flag,
    NewsArticle,
    Rejection,
    RejectionReason,
    now_utc,
)
from ml.llm.securities import SecuritiesMaster

CLOCK_SKEW = timedelta(minutes=5)


@dataclass
class ArticleAnalysis:
    """Everything that happened to one article."""

    article: NewsArticle
    events: list[AnalyzedEvent] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    llm_called: bool = False
    # None when the model was not called.
    schema_valid: bool | None = None
    latency_seconds: float | None = None


class NewsAnalyzer:
    """Extracts validated ``FinancialEvent`` records from articles with an LLM client."""

    def __init__(
        self, client: LLMClient, master: SecuritiesMaster, config: ExtractionConfig | None = None
    ) -> None:
        self.client = client
        self.master = master
        self.config = config or ExtractionConfig()
        self.system = prompts.system_prompt()

    def analyze(self, article: NewsArticle, *, now: datetime | None = None) -> ArticleAnalysis:
        result = ArticleAnalysis(article=article)
        now = now or now_utc()
        if article.published_at > now + CLOCK_SKEW:
            result.rejections.append(
                Rejection(
                    article_id=article.article_id,
                    reason=RejectionReason.ARTICLE_IN_FUTURE,
                    detail=f"published_at {article.published_at.isoformat()} is after "
                    f"{now.isoformat()}",
                )
            )
            return result

        candidates = self.master.referenced(article)
        if not candidates:
            result.rejections.append(
                Rejection(
                    article_id=article.article_id,
                    reason=RejectionReason.NO_KNOWN_ENTITY,
                    detail="the article references no security in the universe",
                )
            )
            return result

        suspicious = prompts.looks_like_injection(article.content)
        if suspicious and self.config.injection_policy == "quarantine":
            result.rejections.append(
                Rejection(
                    article_id=article.article_id,
                    reason=RejectionReason.INJECTION_SUSPECTED,
                    detail="the article contains instruction-like text aimed at a model; "
                    "it was not sent to the model (extraction.injection_policy)",
                )
            )
            return result

        user, truncated = prompts.user_prompt(
            article, candidates, self.master, self.config.max_article_chars
        )
        article_flags: list[Flag] = []
        if truncated:
            article_flags.append(Flag.ARTICLE_TRUNCATED)
        if suspicious:
            article_flags.append(Flag.INJECTION_SUSPECTED)

        result.llm_called = True
        started = time.perf_counter()
        try:
            raw = self.client.complete_json(
                self.system, user, prompts.output_schema(list(candidates), [article.article_id])
            )
        except LLMResponseError as error:
            result.latency_seconds = time.perf_counter() - started
            result.schema_valid = False
            result.rejections.append(
                Rejection(
                    article_id=article.article_id,
                    reason=RejectionReason.LLM_ERROR,
                    detail=str(error)[:500],
                )
            )
            return result
        result.latency_seconds = time.perf_counter() - started

        accepted, rejections, schema_valid = validate_output(
            raw, article, candidates, self.master, self.config, article_flags=tuple(article_flags)
        )
        result.schema_valid = schema_valid
        result.rejections = rejections
        result.events = [
            AnalyzedEvent(
                event=event,
                flags=flags,
                article_id=article.article_id,
                source=article.source,
                llm_provider=self.client.provider,
                llm_model=self.client.model,
                prompt_version=prompts.PROMPT_VERSION,
            )
            for event, flags in accepted
        ]
        return result
