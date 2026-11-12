"""Document store, source-grounded summaries and the evaluation suite (offline)."""

from datetime import timedelta

import pytest

from ml.llm.config import SummaryConfig
from ml.llm.evaluation.dataset import CASES, fictional_master
from ml.llm.evaluation.suite import evaluate
from ml.llm.providers import LLMError
from ml.llm.retrieval.store import DocumentStore
from ml.llm.summarize import summarize_symbol

from .llm_helpers import NOW, PUBLISHED, FakeLLM, make_article, make_master, raw_event

DOCS = [
    make_article(),
    make_article(article_id="a2", headline="Quillon Pharma recalls a batch of tablets",
                 text="Quillon Pharma recalled one batch after a labelling error.",
                 published_at=PUBLISHED + timedelta(days=1)),
    make_article(article_id="a3", headline="Zentara Infra chief executive resigns",
                 text="The board of Zentara Infra accepted the resignation.",
                 published_at=PUBLISHED + timedelta(days=2)),
]  # fmt: skip


# ------------------------------------------------------------------------- store


def test_store_persists_and_searches_with_embeddings(tmp_path):
    store = DocumentStore(tmp_path / "store")
    fake = FakeLLM()
    assert store.add(DOCS, fake) == 3
    assert store.add(DOCS, fake) == 0  # known ids are skipped

    reopened = DocumentStore(tmp_path / "store")
    assert list(reopened.documents) == ["a1", "a2", "a3"]
    assert reopened.embedding_model == "fake-model"
    hits = reopened.search(DOCS[1].content, k=2, client=fake)
    assert hits[0].article.article_id == "a2" and hits[0].method == "embedding"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5) and len(hits) == 2


def test_store_falls_back_to_lexical_search_without_embeddings(tmp_path):
    store = DocumentStore(tmp_path / "store")
    no_embed = FakeLLM(embeddings=False)
    assert store.add(DOCS, no_embed) == 3  # stored even though embedding failed
    for client in (None, no_embed, FakeLLM()):  # no vectors => lexical whatever the client
        hits = store.search("tablet recall labelling", k=3, client=client)
        assert hits[0].article.article_id == "a2" and hits[0].method == "lexical"
        assert hits[0].score > hits[1].score

    master = make_master()
    zentara = store.search("resignation", symbol="ZENTARA", master=master)
    assert [h.article.article_id for h in zentara] == ["a3", "a1"]
    # Point-in-time: a search "as of" a date cannot see later documents.
    early = store.search("resignation", symbol="ZENTARA", master=master,
                         as_of=PUBLISHED + timedelta(hours=1))  # fmt: skip
    assert [h.article.article_id for h in early] == ["a1"]
    with pytest.raises(ValueError):
        store.search("x", symbol="ZENTARA")


def test_query_embedding_failure_falls_back_to_lexical(tmp_path):
    store = DocumentStore(tmp_path / "store")
    store.add(DOCS, FakeLLM())
    hits = store.search("tablet recall", client=FakeLLM(embeddings=False))
    assert hits[0].method == "lexical" and hits[0].article.article_id == "a2"


# --------------------------------------------------------------------- summaries


def test_summary_keeps_only_points_with_valid_citations_and_grounding():
    master = make_master()
    reply = {
        "points": [
            {"text": "Zentara Infra won a highway contract worth 450 crore", "source_ids": ["a1"]},
            {"text": "The chief executive resigned", "source_ids": ["a3", "a3"]},
            {"text": "Zentara Infra won a contract", "source_ids": ["a9"]},  # not supplied
            {"text": "Zentara Infra won a contract", "source_ids": []},
            {"text": "The contract is worth 900 crore", "source_ids": ["a1"]},  # invented number
            {"text": "Analysts expect a dazzling monsoon", "source_ids": ["a1"]},
            {"text": "The chief executive resigned", "source_ids": ["a1"]},  # wrong document
            {"oops": True},
        ]
    }
    fake = FakeLLM([reply])
    summary = summarize_symbol(
        fake, "zentara", [DOCS[0], DOCS[2]], master, SummaryConfig(max_points=7), now=NOW
    )

    assert summary.status == "ok" and summary.company == "Zentara Infra"
    assert [(p.text, p.source_ids) for p in summary.points][1] == (
        "The chief executive resigned", ["a3"],
    )  # fmt: skip
    assert len(summary.points) == 2
    assert summary.source_ids == ["a1", "a3"] and summary.documents_supplied == ["a1", "a3"]
    assert set(summary.source_ids) <= set(summary.documents_supplied)
    assert [r.reason for r in summary.rejected_points] == [
        "citation_invalid", "citation_invalid", "ungrounded_number",
        "ungrounded_statement", "ungrounded_statement", "schema_invalid",
    ]  # fmt: skip
    assert summary.rejected_points[-1].detail == "more than 7 points"
    cite = fake.calls[0][2]["properties"]["points"]["items"]["properties"]["source_ids"]
    assert cite["items"]["enum"] == ["a1", "a3"]  # citations restricted to supplied ids
    assert (summary.llm_provider, summary.llm_model) == ("fake", "fake-model")
    assert summary.prompt_version == "summary-v1"

    system, user, _ = fake.calls[0]
    assert "450 crore" not in system and "untrusted DATA" in system
    inside = user.split("<<<DOCUMENTS")[1].split("<<<END_DOCUMENTS")[0]
    assert "id=a1" in inside and "id=a3" in inside and "highway" in inside
    assert "highway" not in user.split("<<<DOCUMENTS")[0]


def test_summary_without_documents_or_usable_points_is_insufficient():
    master = make_master()
    fake = FakeLLM([{"points": []}, {"nonsense": 1}, {"points": [{"oops": True}]}])
    empty = summarize_symbol(fake, "ZENTARA", [], master, now=NOW)
    assert empty.status == "insufficient_information" and fake.calls == []  # model not asked

    none = summarize_symbol(fake, "ZENTARA", DOCS[:1], master, now=NOW)
    assert none.status == "insufficient_information" and none.points == []
    bad = summarize_symbol(fake, "ZENTARA", DOCS[:1], master, now=NOW)
    assert [r.reason for r in bad.rejected_points] == ["schema_invalid"]
    malformed = summarize_symbol(fake, "ZENTARA", DOCS[:1], master, now=NOW)
    assert [r.reason for r in malformed.rejected_points] == ["schema_invalid"]
    with pytest.raises(ValueError, match="Unknown symbol"):
        summarize_symbol(fake, "NOPE", DOCS, master)
    with pytest.raises(LLMError):
        summarize_symbol(FakeLLM([LLMError("down")]), "ZENTARA", DOCS[:1], master)


# -------------------------------------------------------------------- evaluation


def test_evaluation_cases_are_well_formed_and_fictional():
    master = fictional_master()
    ids = [case.case_id for case in CASES]
    assert len(ids) == len(set(ids)) >= 20
    kinds = {case.kind for case in CASES}
    assert kinds == {"standard", "sentiment_vs_impact", "injection", "insufficient", "no_entity"}
    for case in CASES:
        referenced = master.referenced(case.article())
        if case.kind == "no_entity":
            assert referenced == {}
        else:
            assert list(referenced) == [case.symbol]
        if case.kind in ("standard", "sentiment_vs_impact", "injection"):
            assert case.sentiment and case.event_type
        if case.kind == "injection":
            assert case.injected_sentiment not in case.sentiment
        if case.kind == "sentiment_vs_impact":
            assert case.impact == {"potentially_negative"}
    real = {"RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN", "ITC", "WIPRO"}
    assert real.isdisjoint(master.symbols)


def _oracle(obey_injection: bool):
    """A scripted model that answers each case from its labels (optionally obeying
    injected instructions), to check the scoring arithmetic."""
    by_headline = {case.headline: case for case in CASES}

    def reply(system: str, user: str, schema: dict):
        headline = user.split("HEADLINE: ")[1].split("\n")[0]
        case = by_headline[headline]
        common = {
            "symbol": case.symbol, "company": "x", "key_facts": [case.headline], "risks": [],
            "source_ids": [f"eval_{case.case_id}"],
            "publication_timestamp": "2026-11-02T05:30:00Z",
        }  # fmt: skip
        if case.kind == "insufficient":
            return {"events": [raw_event(**common, event_type="unknown", sentiment="uncertain",
                                         sentiment_score=0, confidence=0.2)]}  # fmt: skip
        sentiment = sorted(case.sentiment)[0]
        if case.kind == "injection" and obey_injection:
            sentiment = case.injected_sentiment
        score = {"positive": 0.6, "negative": -0.6}.get(sentiment, 0.0)
        impact = sorted(case.impact)[0] if case.impact else "neutral"
        if case.case_id == "earnings_up":  # one deliberately wrong event type
            wrong = {**common, "event_type": "dividend", "impact_direction": impact}
            return {"events": [raw_event(**wrong, sentiment=sentiment, sentiment_score=score)]}
        if case.case_id == "contract_win":  # one fabricated figure -> rejected
            fabricated = {**common, "key_facts": ["Order book reached 77,000 crore"]}
            return {"events": [raw_event(**fabricated, sentiment=sentiment, sentiment_score=score)]}
        return {"events": [raw_event(**common, event_type=sorted(case.event_type)[0],
                                     sentiment=sentiment, sentiment_score=score,
                                     impact_direction=impact)]}  # fmt: skip

    return reply


def test_evaluation_metrics_are_computed_from_validated_output():
    report = evaluate(FakeLLM(_oracle(obey_injection=False)), now=NOW)
    counts, metrics = report["counts"], report["metrics"]
    assert counts["cases"] == len(CASES) and counts["llm_calls"] == len(CASES) - 1
    n = counts["classification_cases"]
    assert n == 20 and counts["injection_cases"] == 3 and counts["insufficient_cases"] == 3
    assert metrics["schema_validity_rate"] == 1.0
    assert metrics["extraction_rate"] == pytest.approx((n - 1) / n)  # fabricated one rejected
    assert metrics["sentiment_accuracy"] == pytest.approx((n - 1) / n)
    assert metrics["event_type_accuracy"] == pytest.approx((n - 2) / n)
    assert metrics["event_type_accuracy_when_extracted"] == pytest.approx(
        (n - 2) / (n - 1), abs=1e-4
    )
    assert metrics["injection_resistance"] == 1.0
    assert metrics["insufficient_info_rate"] == 1.0
    assert metrics["no_entity_refusal_rate"] == 1.0
    assert report["rejections_by_reason"] == {"no_known_entity": 1, "ungrounded_number": 1}
    assert report["injection_resisted_by"] == {"correct_analysis": 3}
    assert metrics["injection_heuristic_detection_rate"] == 1.0
    assert metrics["injection_heuristic_false_alarm_rate"] == 0.0
    assert report["llm"] == {"provider": "fake", "model": "fake-model"}

    obedient = evaluate(FakeLLM(_oracle(obey_injection=True)), now=NOW)
    assert obedient["metrics"]["injection_resistance"] == 0.0
    assert obedient["injection_resisted_by"] == {"not_resisted": 3}
    assert obedient["metrics"]["sentiment_accuracy"] == pytest.approx((n - 4) / n)
