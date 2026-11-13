"""Typed configuration of the LLM engine (loaded from ``configs/llm.yaml``).

Unknown keys are rejected. The API key is **never** part of this file: it is read from
the environment (``AETHER_LLM_API_KEY``) by :class:`LLMEnvironment`.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_serializer, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ml.llm.schema import EventType


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_serializer("*", when_used="json", check_fields=False)
    def _portable_paths(self, value):
        return value.as_posix() if isinstance(value, Path) else value


class ProviderConfig(_Strict):
    """An OpenAI-compatible chat-completions endpoint."""

    # Label recorded with every output (e.g. "ollama", "openai"). Not used for routing.
    name: str = Field(min_length=1, max_length=64)
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    # None = the provider has no embedding model; retrieval falls back to lexical search.
    embedding_model: str | None = None
    # None omits the parameter (some hosted models only accept their default).
    temperature: float | None = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(900, ge=16, le=16_000)
    # Hosted OpenAI models have renamed this parameter over time; local servers use
    # "max_tokens".
    max_tokens_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    seed: int | None = None
    timeout_seconds: float = Field(180.0, gt=0)
    max_retries: int = Field(3, ge=0, le=10)
    backoff_seconds: float = Field(1.0, ge=0)
    # First choice of structured-output mode; json_schema falls back to json_object
    # automatically when the server rejects it.
    structured_output: Literal["json_schema", "json_object"] = "json_schema"


class YahooNewsConfig(_Strict):
    cache_dir: Path = Path("data/cache/news")
    exchange_suffix: str = ".NS"
    news_count: int = Field(20, ge=1, le=50)
    # Yahoo's search often returns nothing for an NSE ticker; the company name is tried
    # as a second query when this is on.
    query_company_name: bool = True
    pause_seconds: float = Field(0.5, ge=0)
    max_retries: int = Field(3, ge=0, le=10)
    refresh: bool = False


class DedupConfig(_Strict):
    # Token-set (Jaccard) similarity of normalised headlines at or above which two
    # articles are the same story.
    similarity_threshold: float = Field(0.8, gt=0, le=1)
    # Near-duplicates are only merged when published within this many hours.
    window_hours: float = Field(72.0, gt=0)


class ExtractionConfig(_Strict):
    max_article_chars: int = Field(6000, ge=200, le=100_000)
    max_events_per_article: int = Field(4, ge=1, le=20)
    max_key_facts: int = Field(6, ge=0, le=20)
    # Below this the event is kept but downgraded to event_type "unknown".
    min_confidence: float = Field(0.5, ge=0, le=1)
    # A headline alone cannot support a confident, detailed event.
    headline_only_confidence_cap: float = Field(0.7, ge=0, le=1)
    # Confidence cap when the company is only named by the source's ticker tags and
    # not by the article itself. Below min_confidence by default => fallback to unknown.
    hint_only_confidence_cap: float = Field(0.4, ge=0, le=1)
    # What to do with an article that contains instruction-like text aimed at a model:
    # "quarantine" refuses it before any model call; "flag" analyses it and flags the
    # events. Measured on a small local model, flag-only let injections through.
    injection_policy: Literal["quarantine", "flag"] = "quarantine"
    # Share of a key fact's content words that must occur in the article.
    min_fact_overlap: float = Field(0.5, gt=0, le=1)


class AggregationConfig(_Strict):
    """How events become per-stock daily features (PRD 8.5)."""

    market_timezone: str = "Asia/Kolkata"
    # Decision time on a trading day: only news published at or before it is used.
    market_close: str = Field("15:30", pattern=r"^\d{2}:\d{2}$")
    # Events older than this are ignored entirely.
    max_age_days: float = Field(30.0, gt=0)
    frequency_window_days: float = Field(7.0, gt=0)
    # Recency weight = 0.5 ** (age_days / half_life_days[event_type]).
    default_half_life_days: float = Field(3.0, gt=0)
    half_life_days: dict[EventType, float] = {}
    default_source_reliability: float = Field(1.0, ge=0)
    # Publisher name (case-insensitive) -> reliability weight.
    source_reliability: dict[str, float] = {}

    @field_validator("half_life_days")
    @classmethod
    def _positive(cls, value: dict[EventType, float]) -> dict[EventType, float]:
        if any(days <= 0 for days in value.values()):
            raise ValueError("half-lives must be positive")
        return value

    @field_validator("source_reliability")
    @classmethod
    def _non_negative(cls, value: dict[str, float]) -> dict[str, float]:
        if any(weight < 0 for weight in value.values()):
            raise ValueError("source reliability weights must be >= 0")
        return {name.strip().lower(): weight for name, weight in value.items()}

    def half_life(self, event_type: EventType | str) -> float:
        return self.half_life_days.get(EventType(event_type), self.default_half_life_days)

    def reliability(self, publisher: str | None) -> float:
        if not publisher:
            return self.default_source_reliability
        return self.source_reliability.get(
            publisher.strip().lower(), self.default_source_reliability
        )


class RetrievalConfig(_Strict):
    top_k: int = Field(6, ge=1, le=50)


class SummaryConfig(_Strict):
    max_documents: int = Field(8, ge=1, le=50)
    max_points: int = Field(5, ge=1, le=20)
    max_document_chars: int = Field(1500, ge=100, le=20_000)


class OutputConfig(_Strict):
    root: Path = Path("artifacts/news")


class LLMConfig(_Strict):
    provider: ProviderConfig
    universe_file: Path = Path("configs/universe_nse_largecap.yaml")
    # symbol -> extra names the company is known by (the name in the universe file is
    # always an alias).
    company_aliases: dict[str, list[str]] = {}
    # Foreign tickers for the same company (e.g. ADRs) -> symbol.
    ticker_aliases: dict[str, str] = {}
    # Phrases that contain an alias but name a different company; masked before matching.
    alias_exclusions: list[str] = []
    yahoo_news: YahooNewsConfig = YahooNewsConfig()
    dedup: DedupConfig = DedupConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    aggregation: AggregationConfig = AggregationConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    summary: SummaryConfig = SummaryConfig()
    output: OutputConfig = OutputConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LLMConfig":
        with open(path, encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))

    def with_provider(self, **overrides) -> "LLMConfig":
        """A copy whose provider fields are replaced by the non-None overrides."""
        changes = {key: value for key, value in overrides.items() if value is not None}
        provider = ProviderConfig.model_validate({**self.provider.model_dump(), **changes})
        return self.model_copy(update={"provider": provider})

    def config_hash(self) -> str:
        """Hash of everything that influences the *content* of a run."""
        payload = self.model_dump(mode="json", exclude={"output"})
        for key in ("cache_dir", "refresh", "pause_seconds", "max_retries"):
            payload["yahoo_news"].pop(key, None)
        for key in ("timeout_seconds", "max_retries", "backoff_seconds"):
            payload["provider"].pop(key, None)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class LLMEnvironment(BaseSettings):
    """Deployment-specific overrides and the secret, from ``AETHER_LLM_*`` / ``.env``.

    ``api_key`` is a ``SecretStr``: it does not appear in ``repr`` or logs.
    """

    model_config = SettingsConfigDict(env_prefix="AETHER_LLM_", env_file=".env", extra="ignore")

    provider: str | None = None
    base_url: str | None = None
    model: str | None = None
    embedding_model: str | None = None
    api_key: SecretStr | None = None
    # Hosted reasoning models (e.g. OpenAI gpt-6-luna) reject ``max_tokens`` and any
    # non-default temperature: set MAX_TOKENS_PARAM=max_completion_tokens and
    # TEMPERATURE=default (omit the parameter) for them.
    max_tokens_param: Literal["max_tokens", "max_completion_tokens"] | None = None
    max_tokens: int | None = Field(None, ge=1)
    temperature: str | None = None

    def apply(self, config: LLMConfig) -> LLMConfig:
        updated = config.with_provider(
            name=self.provider or None,
            base_url=self.base_url or None,
            model=self.model or None,
            embedding_model=self.embedding_model or None,
            max_tokens_param=self.max_tokens_param,
            max_tokens=self.max_tokens,
        )
        if self.temperature:
            text = self.temperature.strip().lower()
            value = None if text == "default" else float(text)
            provider = ProviderConfig.model_validate(
                {**updated.provider.model_dump(), "temperature": value}
            )
            updated = updated.model_copy(update={"provider": provider})
        return updated

    @property
    def key(self) -> str | None:
        return self.api_key.get_secret_value() if self.api_key else None
