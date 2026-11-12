"""Integration: news run -> loader -> database -> news API, and POST /news/analyze.

Offline: the companies are fictional and the model is a scripted stand-in injected
through the ``get_news_engine`` dependency. The production dependency is exercised
only on its "not configured" path.
"""

import json
import re
from datetime import UTC, datetime

import pytest
import yaml
from sqlalchemy import func, select

from app.core.config import Settings, get_settings
from app.models import FinancialNews, NewsEvent
from app.services.news import NewsEngine, get_news_engine
from app.services.news_loader import load_news_run
from ml.llm.config import LLMConfig
from ml.llm.ingestion.jsonl import JsonlNewsSource
from ml.llm.pipeline import run_news_pipeline
from ml.llm.providers import LLMError, LLMResponseError, LLMUnavailableError
from ml.llm.securities import SecuritiesMaster
from ml.llm.sentiment.aggregation import aggregate_news_features

NOW = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)
UNIVERSE = {
    "securities": [
        {"symbol": "ZENTARA", "name": "Zentara Infra", "sector": "Industrials"},
        {"symbol": "QUILLON", "name": "Quillon Pharma", "sector": "Health Care"},
        {"symbol": "BRIVEX", "name": "Brivex Software", "sector": "Information Technology"},
    ]
}
ARTICLES = [
    {"article_id": "a1", "headline": "Zentara Infra wins 450 crore highway contract",
     "source": "Fictional Wire", "published_at": "2026-11-10T06:00:00Z",
     "text": "Zentara Infra won a highway contract worth 450 crore."},
    {"article_id": "a1-copy", "headline": "Zentara Infra wins 450 crore highway contract!",
     "source": "Daily Echo", "published_at": "2026-11-10T08:00:00Z"},
    {"article_id": "a2", "headline": "Quillon Pharma fined by drug regulator",
     "source": "Fictional Wire", "published_at": "2026-11-11T11:00:00Z"},
    {"article_id": "a3", "headline": "Brivex Software results disappoint",
     "source": "Fictional Wire", "published_at": "2026-11-11T12:00:00Z"},
]  # fmt: skip


def _event(symbol: str, company: str, user: str, **fields) -> dict:
    base = {
        "company": company, "symbol": symbol, "event_type": "other", "sentiment": "neutral",
        "sentiment_score": 0.0, "impact_direction": "neutral", "impact_horizon": "short_term",
        "severity": 0.5, "confidence": 0.9, "key_facts": [], "risks": [],
        "source_ids": [re.search(r'ARTICLE IDS: \["([^"]+)"\]', user).group(1)],
        "publication_timestamp": re.search(r"PUBLISHED AT: (\S+)", user).group(1),
    }  # fmt: skip
    return {**base, **fields}


class ScriptedLLM:
    """Test stand-in for an ``LLMClient``; answers from the headline it is shown."""

    provider, model = "fake", "fake-model"

    def __init__(self, failure: Exception | None = None):
        self.failure, self.calls = failure, []

    def complete_json(self, system: str, user: str, json_schema: dict) -> dict:
        self.calls.append(user)
        if self.failure:
            raise self.failure
        if "Zentara" in user:
            return {"events": [_event(
                "ZENTARA", "Zentara Infra", user, event_type="new_contract",
                sentiment="positive", sentiment_score=0.6,
                impact_direction="potentially_positive", key_facts=["Won a highway contract"],
            )]}  # fmt: skip
        if "Quillon" in user:
            return {"events": [_event(
                "QUILLON", "Quillon Pharma", user, event_type="regulatory", sentiment="negative",
                sentiment_score=-0.7, impact_direction="potentially_negative", severity=0.8,
                key_facts=["Fined by drug regulator"],
            )]}  # fmt: skip
        if "Brivex" in user:  # fabricates a figure -> must be rejected
            return {"events": [_event(
                "BRIVEX", "Brivex Software", user, event_type="earnings", sentiment="negative",
                sentiment_score=-0.5, key_facts=["Profit fell 37%"],
            )]}  # fmt: skip
        return {"events": []}

    def embed(self, texts):
        raise LLMError("no embedding model is configured")


@pytest.fixture(scope="module")
def engine_parts(tmp_path_factory):
    root = tmp_path_factory.mktemp("news")
    universe = root / "universe.yaml"
    universe.write_text(yaml.safe_dump(UNIVERSE), encoding="utf-8")
    config = LLMConfig.model_validate(
        {
            "provider": {"name": "fake", "base_url": "http://unused", "model": "fake-model"},
            "universe_file": str(universe),
            "output": {"root": str(root / "runs")},
        }
    )
    return config, SecuritiesMaster.from_yaml(universe), root


@pytest.fixture(scope="module")
def run(engine_parts):
    config, master, root = engine_parts
    source = root / "input.jsonl"
    source.write_text("\n".join(json.dumps(a) for a in ARTICLES), encoding="utf-8")
    return run_news_pipeline(
        config, ScriptedLLM(), JsonlNewsSource(source), master, master.symbols, now=NOW
    )


@pytest.fixture
async def loaded(run, session_factory):
    async with session_factory() as session:
        return await load_news_run(session, run.path)


@pytest.fixture
def engine(app, engine_parts):
    """Injects a news engine with the scripted model; tests may replace ``client``."""
    config, master, _ = engine_parts
    news_engine = NewsEngine(config=config, master=master, client=ScriptedLLM())
    app.dependency_overrides[get_news_engine] = lambda: news_engine
    return news_engine


async def _count(session_factory, model) -> int:
    async with session_factory() as session:
        return await session.scalar(select(func.count()).select_from(model))


# ------------------------------------------------------------------------ loader


async def test_loader_writes_articles_and_events_with_provenance(run, loaded, session_factory):
    assert (loaded.run_id, loaded.llm_provider, loaded.llm_model) == (
        run.run_id,
        "fake",
        "fake-model",
    )
    assert (loaded.articles, loaded.events) == (3, 2)
    assert await _count(session_factory, FinancialNews) == 3  # the duplicate was consolidated
    assert await _count(session_factory, NewsEvent) == 2  # the fabricated one was rejected
    async with session_factory() as session:
        event = await session.scalar(select(NewsEvent).where(NewsEvent.symbol == "QUILLON"))
        news = await session.get(FinancialNews, "a1")
    assert event.sector == "Health Care" and event.run_id == run.run_id
    assert (event.llm_provider, event.llm_model, event.prompt_version) == (
        "fake", "fake-model", "extract-v2",
    )  # fmt: skip
    assert news.duplicate_ids == ["a1-copy"] and news.has_body and news.news_source == "user_jsonl"


async def test_loader_is_idempotent_and_refuses_a_modified_run(
    run, loaded, session_factory, tmp_path
):
    import shutil

    async with session_factory() as session:
        again = await load_news_run(session, run.path)
    assert again == loaded
    assert await _count(session_factory, FinancialNews) == 3
    assert await _count(session_factory, NewsEvent) == 2

    copy = tmp_path / run.run_id
    shutil.copytree(run.path, copy)
    events = copy / "events.jsonl"
    events.write_text(events.read_text().replace("-0.7", "0.9"), encoding="utf-8")
    async with session_factory() as session:
        with pytest.raises(ValueError, match="modified after it was written"):
            await load_news_run(session, copy)


# --------------------------------------------------------------------------- API


async def test_news_list_carries_events_and_the_model_that_produced_them(loaded, db_client):
    response = await db_client.get("/api/v1/news")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert body["models"] == [
        {"provider": "fake", "model": "fake-model", "prompt_version": "extract-v2", "run_id": None}
    ]
    assert [item["news_id"] for item in body["items"]] == ["a3", "a2", "a1"]  # newest first
    first = body["items"][2]
    assert first["llm"]["model"] == "fake-model" and first["llm"]["run_id"] == loaded.run_id
    assert first["duplicate_ids"] == ["a1-copy"] and first["duplicate_sources"] == ["Daily Echo"]
    assert "text" not in first
    event = first["events"][0]
    assert event["symbol"] == "ZENTARA" and event["event_type"] == "new_contract"
    assert event["sentiment"] == "positive" and event["impact_direction"] == "potentially_positive"
    assert event["source_ids"] == ["a1", "a1-copy"]
    assert event["publication_timestamp"].startswith("2026-11-10T06:00:00")
    assert event["llm"] == {"provider": "fake", "model": "fake-model",
                            "prompt_version": "extract-v2", "run_id": loaded.run_id}  # fmt: skip
    assert body["items"][0]["events"] == []  # analysed, but its only event was rejected


async def test_news_list_filters(loaded, db_client):
    async def ids(**params) -> list[str]:
        response = await db_client.get("/api/v1/news", params=params)
        assert response.status_code == 200
        return [item["news_id"] for item in response.json()["items"]]

    assert await ids(symbol="zentara") == ["a1"]
    assert await ids(sector="Health Care") == ["a2"]
    assert await ids(event_type="regulatory") == ["a2"]
    assert await ids(symbol="ZENTARA", event_type="regulatory") == []
    assert await ids(start="2026-11-11") == ["a3", "a2"]
    assert await ids(end="2026-11-10") == ["a1"]
    assert await ids(start="2026-11-10", end="2026-11-10") == ["a1"]
    assert await ids(limit=1) == ["a3"]
    assert await ids(symbol="NOPE") == []

    for bad in ({"event_type": "party"}, {"start": "2026-11-12", "end": "2026-11-10"},
                {"limit": 0}, {"limit": 100000}, {"start": "yesterday"}):  # fmt: skip
        assert (await db_client.get("/api/v1/news", params=bad)).status_code == 422


async def test_news_detail_and_404(loaded, db_client):
    body = (await db_client.get("/api/v1/news/a1")).json()
    assert body["text"] == "Zentara Infra won a highway contract worth 450 crore."
    assert body["llm"]["provider"] == "fake" and len(body["events"]) == 1
    assert (await db_client.get("/api/v1/news/does-not-exist")).status_code == 404


async def test_events_endpoint(loaded, db_client, engine):
    body = (await db_client.get("/api/v1/news/events/quillon")).json()
    assert body["symbol"] == "QUILLON" and body["count"] == 1
    assert body["models"][0]["model"] == "fake-model"
    assert body["events"][0]["event_type"] == "regulatory"
    assert body["events"][0]["sector"] == "Health Care"

    none = await db_client.get("/api/v1/news/events/QUILLON", params={"event_type": "dividend"})
    assert none.json()["events"] == [] and none.json()["models"] == []
    # A known symbol without events is an empty list; an unknown one is a 404.
    assert (await db_client.get("/api/v1/news/events/BRIVEX")).json()["count"] == 0
    assert (await db_client.get("/api/v1/news/events/RELIANCE")).status_code == 404
    bad = await db_client.get("/api/v1/news/events/QUILLON", params={"event_type": "party"})
    assert bad.status_code == 422


async def test_sentiment_endpoint_is_point_in_time_and_matches_the_aggregation(
    run, loaded, db_client, engine
):
    params = {"start": "2026-11-09", "end": "2026-11-12"}
    response = await db_client.get("/api/v1/news/sentiment/QUILLON", params=params)
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "QUILLON" and body["events_used"] == 1
    assert body["market_close"] == "15:30" and body["market_timezone"] == "Asia/Kolkata"
    assert body["models"][0]["provider"] == "fake"
    history = {point["trade_date"]: point["features"] for point in body["history"]}
    assert list(history) == ["2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12"]
    # Published 11:00 UTC on the 11th = 16:30 IST, after the close: first counted on the 12th.
    assert history["2026-11-11"]["news_available"] == 0
    assert all(value == 0 for value in history["2026-11-11"].values())
    assert history["2026-11-12"]["news_available"] == 1
    assert history["2026-11-12"]["news_sentiment"] == pytest.approx(-0.7)
    assert body["current"] == body["history"][-1]

    expected = aggregate_news_features(
        [a.event for a in run.events], ["QUILLON"], ["2026-11-12"], engine.config.aggregation
    ).iloc[0]
    for name, value in history["2026-11-12"].items():
        assert value == pytest.approx(expected[name])

    quiet = (await db_client.get("/api/v1/news/sentiment/BRIVEX", params=params)).json()
    assert quiet["events_used"] == 0 and quiet["models"] == []
    assert quiet["current"]["features"]["news_available"] == 0  # "no news", not "neutral"

    assert (await db_client.get("/api/v1/news/sentiment/RELIANCE")).status_code == 404
    reversed_range = {"start": "2026-11-12", "end": "2026-11-09"}
    assert (
        await db_client.get("/api/v1/news/sentiment/QUILLON", params=reversed_range)
    ).status_code == 422
    too_long = {"start": "2020-01-01", "end": "2026-11-12"}
    assert (
        await db_client.get("/api/v1/news/sentiment/QUILLON", params=too_long)
    ).status_code == 422
    assert (await db_client.get("/api/v1/news/sentiment/QUILLON")).status_code == 200


# ------------------------------------------------------------------ POST /analyze

REQUEST = {
    "headline": "Zentara Infra wins 450 crore highway contract",
    "text": "Zentara Infra won a highway contract worth 450 crore.",
    "source": "Fictional Wire",
    "published_at": "2026-11-10T11:30:00+05:30",
}


async def test_analyze_returns_validated_events_with_model_and_provider(client, engine):
    response = await client.post("/api/v1/news/analyze", json=REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert body["llm"] == {"provider": "fake", "model": "fake-model",
                           "prompt_version": "extract-v2", "run_id": None}  # fmt: skip
    assert body["llm_called"] is True and body["persisted"] is False
    assert body["candidates"] == [
        {"symbol": "ZENTARA", "evidence": "alias", "matched": "Zentara Infra"}
    ]
    event = body["events"][0]
    assert event["symbol"] == "ZENTARA" and event["sector"] == "Industrials"
    assert event["source_ids"] == [body["article_id"]] and event["news_id"] == body["article_id"]
    assert event["publication_timestamp"].startswith("2026-11-10T06:00:00")
    assert event["llm"]["model"] == "fake-model"
    assert body["rejections"] == []
    # The article reached the model only inside the delimited data block.
    assert "<<<ARTICLE_DATA" in engine.client.calls[0]


async def test_analyze_reports_rejected_output_instead_of_passing_it_on(client, engine):
    fabricated = {**REQUEST, "headline": "Brivex Software results disappoint", "text": ""}
    body = (await client.post("/api/v1/news/analyze", json=fabricated)).json()
    assert body["events"] == [] and body["headline_only"] is True
    assert [r["reason"] for r in body["rejections"]] == ["ungrounded_number"]
    assert body["rejections"][0]["symbol"] == "BRIVEX"

    unrelated = {**REQUEST, "headline": "Monsoon arrives early", "text": ""}
    body = (await client.post("/api/v1/news/analyze", json=unrelated)).json()
    assert body["llm_called"] is False
    assert [r["reason"] for r in body["rejections"]] == ["no_known_entity"]


async def test_analyze_is_503_when_no_llm_is_configured(client, engine):
    engine.client = None
    response = await client.post("/api/v1/news/analyze", json=REQUEST)
    assert response.status_code == 503
    assert "AETHER_LLM_BASE_URL" in response.json()["detail"]


async def test_default_settings_have_no_llm_and_never_fake_one(app, client, monkeypatch):
    for name in ("BASE_URL", "MODEL", "API_KEY", "PROVIDER"):
        monkeypatch.delenv(f"AETHER_LLM_{name}", raising=False)
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    response = await client.post("/api/v1/news/analyze", json=REQUEST)
    assert response.status_code == 503
    assert "No LLM is configured" in response.json()["detail"]

    secret = "sk-test-SECRET-123"
    settings = Settings(
        _env_file=None, llm_base_url="http://x/v1", llm_model="m", llm_api_key=secret
    )
    assert secret not in repr(settings) and secret not in settings.model_dump_json()
    built = get_news_engine(settings)
    assert built.client is not None and built.client.provider == "x"
    assert secret not in repr(built.client)

    missing = Settings(_env_file=None, llm_config="configs/does-not-exist.yaml")
    app.dependency_overrides[get_settings] = lambda: missing
    response = await client.post("/api/v1/news/analyze", json=REQUEST)
    assert response.status_code == 503 and "could not be loaded" in response.json()["detail"]


async def test_analyze_maps_provider_failures_to_clear_errors(client, engine):
    engine.client = ScriptedLLM(LLMUnavailableError("ollama at http://x is unavailable"))
    response = await client.post("/api/v1/news/analyze", json=REQUEST)
    assert response.status_code == 503 and "unavailable" in response.json()["detail"]

    engine.client = ScriptedLLM(LLMError("HTTP 400: unsupported parameter"))
    response = await client.post("/api/v1/news/analyze", json=REQUEST)
    assert response.status_code == 502

    # A reply that is not usable JSON is a rejection, not a server error.
    engine.client = ScriptedLLM(LLMResponseError("reply is not valid JSON"))
    body = (await client.post("/api/v1/news/analyze", json=REQUEST)).json()
    assert body["events"] == [] and body["rejections"][0]["reason"] == "llm_error"


@pytest.mark.parametrize(
    "change",
    [
        {"headline": ""},
        {"headline": "x" * 501},
        {"text": "x" * 20_001},
        {"published_at": "2026-11-10T06:00:00"},  # naive
        {"published_at": "2999-01-01T00:00:00Z"},  # future
        {"published_at": "soon"},
        {"symbols": ["S"] * 21},
        {"unexpected": "field"},
    ],
)
async def test_analyze_validates_its_input(client, engine, change):
    response = await client.post("/api/v1/news/analyze", json={**REQUEST, **change})
    assert response.status_code == 422
    assert engine.client.calls == []  # invalid input never reaches the model


async def test_empty_database_returns_empty_lists(db_client, engine):
    body = (await db_client.get("/api/v1/news")).json()
    assert body == {"count": 0, "models": [], "items": []}
    assert (await db_client.get("/api/v1/news/events/ZENTARA")).json()["events"] == []
