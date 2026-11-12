"""Hallucination controls: nothing a model returns is trusted until it passes here.

Every event in a model reply is either **accepted** (possibly with flags that say what
was changed) or **rejected** with a reason. Nothing is dropped silently (PRD 8.4).

Rejected (the event is not stored as an event)
    schema_invalid               reply or event does not match the schema / ranges
    unknown_symbol               symbol is not in the securities master
    entity_not_in_article        symbol is known but the article does not reference it
    duplicate_event              a second event for the same company
    source_attribution_invalid   source_ids empty or citing ids that were not supplied
    timestamp_mismatch           publication_timestamp differs from the article's
    sentiment_inconsistent       sign of sentiment_score contradicts the label
    ungrounded_number            a number in key_facts / risks is not in the article

Flagged (the event is stored, altered as described)
    ungrounded_fact_removed          a key fact sharing too few words with the article
    confidence_capped_headline_only  only a headline was available
    entity_from_hint_only            company named only by the source's ticker tags
    low_confidence_fallback_unknown  confidence below threshold: classification reset
                                     to "unknown" / "uncertain"
    no_evidence_fallback_unknown     no key fact survived grounding: same reset (an
                                     assessment without evidence is not kept)
    injection_suspected              the article contains instruction-like text
    article_truncated                the model saw a truncated article

Articles with instruction-like text are quarantined before the model is called (see
``analyzer.py``), because injected text is part of the article and would otherwise
count as "grounding" for whatever it asserts.

Limitations: grounding is lexical. Numbers are compared as digits ("12" matches "12%"
and "12.0"), so numbers written as words are not checked, and a correctly copied number
attached to the wrong quantity passes. Word overlap does not verify meaning.
"""

import re
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ml.llm.config import ExtractionConfig
from ml.llm.schema import (
    EventFlag,
    EventType,
    FinancialEvent,
    Flag,
    ImpactDirection,
    ImpactHorizon,
    NewsArticle,
    Rejection,
    RejectionReason,
    Sentiment,
    make_event_id,
)
from ml.llm.securities import EntityMatch, SecuritiesMaster

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"[a-z][a-z0-9&'-]*")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "had",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "which",
        "who",
        "whom",
        "than",
        "then",
        "there",
        "these",
        "those",
        "after",
        "before",
        "over",
        "under",
        "about",
        "against",
        "between",
        "during",
        "per",
        "via",
        "not",
        "no",
        "also",
        "may",
        "might",
        "can",
        "could",
        "would",
        "should",
        "company",
        "companies",
        "firm",
        "said",
        "says",
        "say",
        "reported",
        "reports",
        "report",
        "announced",
        "announces",
        "announce",
        "according",
    ]
)
_SUFFIXES = ("ing", "ed", "es", "ly", "s")


class RawEvent(BaseModel):
    """One event exactly as the model must emit it (PRD 8.3 minus ``event_id``)."""

    model_config = ConfigDict(extra="forbid")

    company: str = Field(min_length=1, max_length=255)
    symbol: str = Field(min_length=1, max_length=32)
    event_type: EventType
    sentiment: Sentiment
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    impact_direction: ImpactDirection
    impact_horizon: ImpactHorizon
    severity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    key_facts: list[str]
    risks: list[str]
    source_ids: list[str]
    publication_timestamp: str


def _error_summary(error: ValidationError) -> str:
    parts = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()[:6]]
    return "; ".join(parts)


def numbers_in(text: str) -> set[str]:
    """Canonical forms of the numbers in ``text`` ("1,250.50" -> "1250.5")."""
    found = set()
    for match in _NUMBER.findall(text):
        try:
            value = Decimal(match.replace(",", "")).normalize()
        except InvalidOperation:  # pragma: no cover - the regex only matches digits
            continue
        found.add(format(value, "f"))
    return found


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def content_words(text: str) -> set[str]:
    """Lightly stemmed lower-case words of ``text`` without stop words."""
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 1}


def fact_overlap(fact: str, article_words: set[str]) -> float:
    """Share of the fact's content words that occur in the article."""
    words = content_words(fact)
    if not words:
        return 0.0
    return len(words & article_words) / len(words)


def _parse_timestamp(text: str) -> datetime | None:
    try:
        value = datetime.fromisoformat(text.strip())
    except ValueError:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def validate_output(
    raw: object,
    article: NewsArticle,
    candidates: dict[str, EntityMatch],
    master: SecuritiesMaster,
    config: ExtractionConfig,
    *,
    article_flags: tuple[Flag, ...] = (),
) -> tuple[list[tuple[FinancialEvent, list[EventFlag]]], list[Rejection], bool]:
    """Validate one model reply for one article.

    Inputs:  the parsed reply, the article it was produced from, the securities the
             article references (``SecuritiesMaster.referenced``), and the thresholds.
    Outputs: (accepted events with their flags, rejections, whether the reply as a
             whole was schema-valid).
    """

    def reject(reason: RejectionReason, detail: str, symbol: str | None, payload) -> Rejection:
        return Rejection(
            article_id=article.article_id,
            reason=reason,
            detail=detail[:500],
            symbol=symbol,
            raw_output=payload,
        )

    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("events"), list)
        or set(raw) != {"events"}
    ):
        detail = "reply must be an object with exactly one key 'events' holding a list"
        return [], [reject(RejectionReason.SCHEMA_INVALID, detail, None, raw)], False

    accepted: list[tuple[FinancialEvent, list[EventFlag]]] = []
    rejections: list[Rejection] = []
    schema_valid = True
    article_text = article.content
    article_numbers = numbers_in(article_text)
    article_words = content_words(article_text)
    seen: set[str] = set()

    for index, item in enumerate(raw["events"]):
        if index >= config.max_events_per_article:
            schema_valid = False
            detail = f"more than {config.max_events_per_article} events for one article"
            rejections.append(reject(RejectionReason.SCHEMA_INVALID, detail, None, item))
            continue
        try:
            event = RawEvent.model_validate(item)
        except ValidationError as error:
            schema_valid = False
            symbol = item.get("symbol") if isinstance(item, dict) else None
            rejections.append(
                reject(
                    RejectionReason.SCHEMA_INVALID,
                    _error_summary(error),
                    symbol if isinstance(symbol, str) else None,
                    item,
                )
            )
            continue

        symbol = event.symbol.strip().upper()
        security = master.get(symbol)
        if security is None:
            detail = f"'{event.symbol}' is not in the securities master"
            rejections.append(reject(RejectionReason.UNKNOWN_SYMBOL, detail, symbol, item))
            continue
        if symbol not in candidates:
            detail = f"the article does not reference {symbol} ({security.name})"
            rejections.append(reject(RejectionReason.ENTITY_NOT_IN_ARTICLE, detail, symbol, item))
            continue
        if symbol in seen:
            detail = f"second event for {symbol} in one article"
            rejections.append(reject(RejectionReason.DUPLICATE_EVENT, detail, symbol, item))
            continue

        supplied = set(article.source_ids)
        cited = [s.strip() for s in event.source_ids]
        if not cited or not set(cited) <= supplied:
            detail = f"cited {cited}, supplied {sorted(supplied)}"
            rejections.append(
                reject(RejectionReason.SOURCE_ATTRIBUTION_INVALID, detail, symbol, item)
            )
            continue

        stamp = _parse_timestamp(event.publication_timestamp)
        if stamp is None or abs((stamp - article.published_at).total_seconds()) >= 1.0:
            detail = (
                f"model said {event.publication_timestamp!r}, article was published "
                f"{article.published_at.isoformat()}"
            )
            rejections.append(reject(RejectionReason.TIMESTAMP_MISMATCH, detail, symbol, item))
            continue

        score = event.sentiment_score
        if (event.sentiment is Sentiment.POSITIVE and score < 0) or (
            event.sentiment is Sentiment.NEGATIVE and score > 0
        ):
            detail = f"sentiment {event.sentiment.value} with score {score}"
            rejections.append(reject(RejectionReason.SENTIMENT_INCONSISTENT, detail, symbol, item))
            continue

        invented = sorted(numbers_in(" ".join([*event.key_facts, *event.risks])) - article_numbers)
        if invented:
            detail = f"numbers not present in the article: {invented}"
            rejections.append(reject(RejectionReason.UNGROUNDED_NUMBER, detail, symbol, item))
            continue

        flags = [EventFlag(flag=flag) for flag in article_flags]
        key_facts = []
        for fact in event.key_facts:
            overlap = fact_overlap(fact, article_words)
            if overlap >= config.min_fact_overlap:
                key_facts.append(fact)
            elif fact.strip():
                flags.append(
                    EventFlag(
                        flag=Flag.UNGROUNDED_FACT_REMOVED,
                        detail=f"{fact!r} (word overlap {overlap:.2f})",
                    )
                )
        if len(key_facts) > config.max_key_facts:
            for fact in key_facts[config.max_key_facts :]:
                flags.append(
                    EventFlag(flag=Flag.UNGROUNDED_FACT_REMOVED, detail=f"{fact!r} (over limit)")
                )
            key_facts = key_facts[: config.max_key_facts]

        confidence = event.confidence
        if article.headline_only and confidence > config.headline_only_confidence_cap:
            flags.append(
                EventFlag(
                    flag=Flag.CONFIDENCE_CAPPED_HEADLINE_ONLY,
                    detail=f"{confidence:.2f} -> {config.headline_only_confidence_cap:.2f}",
                )
            )
            confidence = config.headline_only_confidence_cap
        if candidates[symbol].evidence == "hint":
            flags.append(
                EventFlag(
                    flag=Flag.ENTITY_FROM_HINT_ONLY,
                    detail=f"tagged {candidates[symbol].matched!r} by the source; not named "
                    "in the text",
                )
            )
            confidence = min(confidence, config.hint_only_confidence_cap)

        fields = {
            "event_type": event.event_type,
            "sentiment": event.sentiment,
            "sentiment_score": score,
            "impact_direction": event.impact_direction,
            "impact_horizon": event.impact_horizon,
            "severity": event.severity,
        }
        reset: EventFlag | None = None
        claimed = f"model said {event.event_type.value}/{event.sentiment.value}"
        if event.event_type is not EventType.UNKNOWN:
            if confidence < config.min_confidence:
                reset = EventFlag(
                    flag=Flag.LOW_CONFIDENCE_FALLBACK_UNKNOWN,
                    detail=f"confidence {confidence:.2f} < {config.min_confidence:.2f}; {claimed}",
                )
            elif not key_facts:
                # An assessment needs supporting evidence (PRD 8.2-D).
                reset = EventFlag(
                    flag=Flag.NO_EVIDENCE_FALLBACK_UNKNOWN,
                    detail=f"no key fact grounded in the article; {claimed}",
                )
        if reset is not None:
            flags.append(reset)
            fields = {
                "event_type": EventType.UNKNOWN,
                "sentiment": Sentiment.UNCERTAIN,
                "sentiment_score": 0.0,
                "impact_direction": ImpactDirection.UNCERTAIN,
                "impact_horizon": ImpactHorizon.UNKNOWN,
                "severity": 0.0,
            }

        seen.add(symbol)
        accepted.append(
            (
                FinancialEvent(
                    event_id=make_event_id(article.article_id, symbol),
                    company=security.name,  # canonical name from the securities master
                    symbol=symbol,
                    confidence=confidence,
                    key_facts=key_facts,
                    risks=event.risks,
                    source_ids=article.source_ids,  # the story and its consolidated copies
                    publication_timestamp=article.published_at,
                    **fields,
                ),
                flags,
            )
        )
    return accepted, rejections, schema_valid
