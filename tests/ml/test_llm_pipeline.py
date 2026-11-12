"""End-to-end pipeline on a fictional JSONL file with a scripted model (offline)."""

import json
import re

import pandas as pd
import pytest

from ml.llm.config import LLMConfig
from ml.llm.ingestion.jsonl import JsonlNewsSource
from ml.llm.pipeline import run_news_pipeline
from ml.llm.providers import LLMResponseError, LLMUnavailableError
from ml.llm.runs import list_runs, load_all_events, load_run, verify_run
from ml.llm.sentiment.aggregation import COLUMNS, FEATURE_COLUMNS, news_feature_table

from .llm_helpers import NOW, FakeLLM, make_master, raw_event

ARTICLES = [
    {"article_id": "a1", "headline": "Zentara Infra wins 450 crore highway contract",
     "source": "Fictional Wire", "published_at": "2026-11-10T06:00:00Z",
     "text": "Zentara Infra won a highway contract worth 450 crore."},
    {"article_id": "a1-copy", "headline": "Zentara Infra wins 450 crore highway contract!",
     "source": "Daily Echo", "published_at": "2026-11-10T08:00:00Z"},
    {"article_id": "a2", "headline": "Quillon Pharma recalls a batch of tablets",
     "source": "Fictional Wire", "published_at": "2026-11-11T05:00:00Z"},
    {"article_id": "a3", "headline": "Monsoon arrives early this year",
     "source": "Fictional Wire", "published_at": "2026-11-11T06:00:00Z"},
    {"article_id": "a4", "headline": "Brivex Software names new finance chief",
     "source": "Fictional Wire", "published_at": "2026-11-11T07:00:00Z"},
]  # fmt: skip


def _config(tmp_path) -> LLMConfig:
    return LLMConfig.model_validate(
        {
            "provider": {"name": "fake", "base_url": "http://unused", "model": "fake-model"},
            "output": {"root": str(tmp_path / "news")},
        }
    )


def _source(tmp_path) -> JsonlNewsSource:
    path = tmp_path / "input.jsonl"
    path.write_text("\n".join(json.dumps(a) for a in ARTICLES), encoding="utf-8")
    return JsonlNewsSource(path)


def _reply(system: str, user: str, schema: dict):
    """Scripted model: answers from the article id it is shown."""
    article_id = re.search(r'ARTICLE IDS: \["([^"]+)"\]', user).group(1)
    stamp = re.search(r"PUBLISHED AT: (\S+)", user).group(1)
    common = {"source_ids": [article_id], "publication_timestamp": stamp}
    if article_id == "a1":
        return {"events": [raw_event(**common)]}
    if article_id == "a2":  # invents a figure that is not in the headline
        return {"events": [raw_event(
            symbol="QUILLON", company="Quillon Pharma", event_type="regulatory",
            sentiment="negative", sentiment_score=-0.5, impact_direction="potentially_negative",
            key_facts=["Recalled 40000 tablets"], **common)]}  # fmt: skip
    return LLMResponseError("reply is not valid JSON")


def test_pipeline_writes_a_verifiable_run_with_every_article_accounted_for(tmp_path):
    config, master = _config(tmp_path), make_master()
    run = run_news_pipeline(
        config, FakeLLM(_reply), _source(tmp_path), master, master.symbols, now=NOW
    )

    assert run.run_id == f"news-20261112T120000Z-{config.config_hash()[:8]}"
    assert sorted(p.name for p in run.path.iterdir()) == [
        "articles.jsonl", "events.jsonl", "features.csv", "manifest.json", "rejected.jsonl",
    ]  # fmt: skip
    manifest = verify_run(run.path)
    assert manifest["llm"] == {
        "provider": "fake",
        "model": "fake-model",
        "base_url": "http://unused",
    }
    assert manifest["prompt_version"] == "extract-v2"
    assert manifest["config_hash"] == config.config_hash()
    assert manifest["source"]["name"] == "user_jsonl"
    assert manifest["counts"] == {
        "articles_fetched": 5, "articles_unique": 4, "articles_skipped_by_limit": 0,
        "articles_analyzed": 4, "duplicate_groups": 1, "duplicates_merged": 1, "llm_calls": 3,
        "llm_calls_schema_valid": 2, "events": 1, "events_unknown": 0, "rejections": 3,
    }  # fmt: skip
    assert manifest["rejections_by_reason"] == {
        "llm_error": 1, "no_known_entity": 1, "ungrounded_number": 1,
    }  # fmt: skip

    # Every unique article is either in events or in rejected -- nothing vanishes.
    _, articles, events = load_run(run.path)
    rejected = [json.loads(line) for line in (run.path / "rejected.jsonl").read_text().splitlines()]
    assert {a.article_id for a in articles} == {"a1", "a2", "a3", "a4"}
    assert {e.article_id for e in events} | {r["article_id"] for r in rejected} == {
        "a1", "a2", "a3", "a4",
    }  # fmt: skip
    assert all(r["reason"] and r["detail"] for r in rejected)
    event = events[0].event
    assert event.symbol == "ZENTARA" and event.source_ids == ["a1", "a1-copy"]
    assert (events[0].llm_provider, events[0].llm_model) == ("fake", "fake-model")

    features = pd.read_csv(run.path / "features.csv")
    assert list(features.columns) == COLUMNS and not features.isna().any().any()
    assert set(features["symbol"]) == set(master.symbols)
    zentara = features[features["symbol"] == "ZENTARA"].set_index("trade_date")
    assert zentara.loc["2026-11-10", "news_available"] == 1
    assert zentara.loc["2026-11-10", "news_sentiment"] == pytest.approx(0.6)
    quiet = features[features["symbol"] == "QUILLON"]
    assert (quiet[FEATURE_COLUMNS] == 0).all().all()  # its only output was rejected


def test_feature_table_reads_runs_and_refuses_tampered_ones(tmp_path):
    config, master = _config(tmp_path), make_master()
    run = run_news_pipeline(
        config, FakeLLM(_reply), _source(tmp_path), master, master.symbols, now=NOW
    )
    root = str(config.output.root)
    assert list_runs(root) == [run.path] and len(load_all_events(root)) == 1

    table = news_feature_table(["ZENTARA", "BRIVEX"], "2026-11-02", "2026-11-12", news_root=root)
    assert list(table.columns) == COLUMNS and len(table) == 2 * 9
    row = table[(table["symbol"] == "ZENTARA") & (table["trade_date"] == "2026-11-12")].iloc[0]
    assert row["news_available"] == 1 and row["news_days_since_last"] == pytest.approx(2 + 4 / 24)
    assert table[table["trade_date"] < "2026-11-10"]["news_available"].sum() == 0
    sessions = ["2026-11-10", "2026-11-11", "2026-12-25"]
    custom = news_feature_table(["ZENTARA"], "2026-11-01", "2026-11-30", news_root=root,
                                sessions=sessions)  # fmt: skip
    assert [f"{d:%Y-%m-%d}" for d in custom["trade_date"]] == sessions[:2]

    nothing = news_feature_table(["ZENTARA"], "2026-11-10", "2026-11-11",
                                 news_root=str(tmp_path / "none"))  # fmt: skip
    assert len(nothing) == 2 and nothing["news_available"].sum() == 0

    events_file = run.path / "events.jsonl"
    events_file.write_text(events_file.read_text().replace("0.6", "0.9"), encoding="utf-8")
    with pytest.raises(ValueError, match="modified after it was written"):
        news_feature_table(["ZENTARA"], "2026-11-10", "2026-11-11", news_root=root)


def test_unreachable_provider_stops_the_run_instead_of_writing_an_empty_one(tmp_path):
    config, master = _config(tmp_path), make_master()
    down = FakeLLM(lambda *_: LLMUnavailableError("connection refused"))
    with pytest.raises(LLMUnavailableError):
        run_news_pipeline(config, down, _source(tmp_path), master, master.symbols, now=NOW)
    assert list_runs(config.output.root) == []

    # A single outage is recorded against the article and the run continues.
    replies = [LLMUnavailableError("timeout"), {"events": []}, {"events": []}]
    flaky = FakeLLM(lambda *_: replies.pop(0))
    run = run_news_pipeline(config, flaky, _source(tmp_path), master, master.symbols, now=NOW,
                            max_articles=3)  # fmt: skip
    assert run.manifest["rejections_by_reason"] == {"llm_error": 1, "no_known_entity": 1}
    assert run.manifest["counts"]["articles_skipped_by_limit"] == 1
