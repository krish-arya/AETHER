"""News sources and de-duplication, offline. Yahoo responses are fabricated in Yahoo's
JSON shape; companies and headlines are fictional."""

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from ml.llm.config import DedupConfig
from ml.llm.ingestion.base import NewsSourceError
from ml.llm.ingestion.dedup import consolidate, headline_hash, token_set_similarity
from ml.llm.ingestion.jsonl import JsonlNewsSource
from ml.llm.ingestion.yahoo_news import YahooNewsSource

from .llm_helpers import NOW, PUBLISHED, make_article, make_master


def _item(uuid: str, title: str, tickers=(), publisher="Fictional Wire", when=PUBLISHED) -> dict:
    return {
        "uuid": uuid,
        "title": title,
        "publisher": publisher,
        "link": f"https://example.invalid/{uuid}",
        "providerPublishTime": int(when.timestamp()),
        "type": "STORY",
        "relatedTickers": list(tickers),
    }


class FakeYahoo:
    def __init__(self, news: dict[str, list[dict]], failures: list[int] | None = None):
        self.news, self.failures, self.requests = news, list(failures or []), []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.failures:
            return httpx.Response(self.failures.pop(0))
        return httpx.Response(200, json={"news": self.news.get(request.url.params["q"], [])})

    def source(self, cache, **kwargs) -> YahooNewsSource:
        kwargs.setdefault("clock", lambda: NOW)
        return YahooNewsSource(
            make_master(),
            cache,
            client=httpx.Client(transport=httpx.MockTransport(self)),
            pause_seconds=0.0,
            **kwargs,
        )


# -------------------------------------------------------------------------- Yahoo


def test_yahoo_items_become_headline_only_articles(tmp_path):
    fake = FakeYahoo(
        {
            "ZENTARA.NS": [_item("u1", "Zentara Infra wins contract", ["ZENTARA.NS", "XYZ"])],
            "Zentara Infra": [
                _item("u1", "Zentara Infra wins contract", ["ZENTARA.BO"]),
                _item("u2", "Unrelated company expands", ["OTHER"]),
                {"uuid": "u3", "title": None},  # unusable item
            ],
        }
    )
    source = fake.source(tmp_path)
    articles = source.fetch(["ZENTARA"])

    assert [a.article_id for a in articles] == ["yahoo_u1", "yahoo_u2"]
    first = articles[0]
    assert first.headline_only and first.text == ""
    assert first.published_at == PUBLISHED and first.retrieved_at == NOW
    assert first.source == "Fictional Wire" and first.url.endswith("/u1")
    assert first.symbols == ["ZENTARA.NS", "XYZ", "ZENTARA.BO"]  # tags merged across queries
    # A search hit is not evidence: the queried symbol is not added as a hint.
    assert articles[1].symbols == ["OTHER"]

    assert [r.url.params["q"] for r in fake.requests] == ["ZENTARA.NS", "Zentara Infra"]
    assert fake.requests[0].url.params["newsCount"] == "20"
    log = source.fetch_log()
    assert [(e["query"], e["items"], e["unparseable"]) for e in log] == [
        ("ZENTARA.NS", 1, 0),
        ("Zentara Infra", 3, 1),
    ]
    assert source.info.provides_body is False and source.info.provides_history is False
    assert "not for redistribution" in source.info.license_note


def test_yahoo_responses_are_cached_per_query_and_day(tmp_path):
    fake = FakeYahoo({"ZENTARA.NS": [_item("u1", "Zentara Infra wins contract")]})
    first = fake.source(tmp_path, query_company_name=False).fetch(["ZENTARA"])
    later = fake.source(
        tmp_path, query_company_name=False, clock=lambda: NOW + timedelta(hours=3)
    ).fetch(["ZENTARA"])
    assert len(fake.requests) == 1
    assert later == first  # including the original retrieved_at
    assert [p.name for p in tmp_path.iterdir()] == ["ZENTARA_NS_20261112.json"]

    fake.source(tmp_path, query_company_name=False, refresh=True).fetch(["ZENTARA"])
    next_day = fake.source(
        tmp_path, query_company_name=False, clock=lambda: NOW + timedelta(days=1)
    )
    next_day.fetch(["ZENTARA"])
    assert len(fake.requests) == 3
    assert next_day.fetch_log()[0]["from_cache"] is False


def test_yahoo_retries_then_fails_clearly_without_caching(tmp_path):
    flaky = FakeYahoo({"ZENTARA.NS": [_item("u1", "Zentara Infra wins")]}, failures=[429, 503])
    assert len(flaky.source(tmp_path / "a", query_company_name=False).fetch(["ZENTARA"])) == 1
    assert len(flaky.requests) == 3

    down = FakeYahoo({}, failures=[503] * 10)
    with pytest.raises(NewsSourceError, match="HTTP 503"):
        down.source(tmp_path / "b", max_retries=1).fetch(["ZENTARA"])
    assert len(down.requests) == 2 and not (tmp_path / "b").exists()

    gone = FakeYahoo({}, failures=[404])
    with pytest.raises(NewsSourceError, match="HTTP 404"):
        gone.source(tmp_path / "c").fetch(["ZENTARA"])
    assert len(gone.requests) == 1
    with pytest.raises(NewsSourceError, match="not in the configured universe"):
        FakeYahoo({}).source(tmp_path / "d").fetch(["NOPE"])


# -------------------------------------------------------------------------- JSONL


def test_jsonl_source_reads_articles_and_derives_missing_ids(tmp_path):
    path = tmp_path / "news.jsonl"
    rows = [
        {"article_id": "n1", "headline": "Zentara Infra wins contract", "source": "Wire",
         "published_at": "2026-11-10T11:30:00+05:30", "text": "Body.", "symbols": ["ZENTARA"]},
        {"headline": "Quillon Pharma recalls batch", "source": "Wire",
         "published_at": "2026-11-10T07:00:00Z"},
    ]  # fmt: skip
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    source = JsonlNewsSource(path)
    first, second = source.fetch()
    assert first.article_id == "n1" and first.published_at == PUBLISHED and first.text == "Body."
    assert second.article_id.startswith("art_") and second.headline_only
    assert source.fetch()[1].article_id == second.article_id  # deterministic
    assert source.info.provides_body and source.fetch_log()[0]["items"] == 2


@pytest.mark.parametrize(
    "line",
    [
        "not json",
        '{"headline": "No timestamp", "source": "Wire"}',
        '{"headline": "Naive", "source": "Wire", "published_at": "2026-11-10T07:00:00"}',
        '{"headline": "Extra", "source": "W", "published_at": "2026-11-10T07:00:00Z", "x": 1}',
        "[1, 2]",
    ],
)
def test_jsonl_source_refuses_malformed_lines_instead_of_skipping_them(tmp_path, line):
    path = tmp_path / "news.jsonl"
    good = '{"headline": "Fine", "source": "Wire", "published_at": "2026-11-10T07:00:00Z"}'
    path.write_text(f"{good}\n{line}\n", encoding="utf-8")
    with pytest.raises(NewsSourceError, match=r"news\.jsonl:2"):
        JsonlNewsSource(path).fetch()
    with pytest.raises(NewsSourceError, match="not found"):
        JsonlNewsSource(tmp_path / "missing.jsonl").fetch()


# -------------------------------------------------------------------------- dedup


def _at(hours: float) -> datetime:
    return datetime(2026, 11, 10, 6, 0, tzinfo=UTC) + timedelta(hours=hours)


def test_similarity_helpers():
    assert headline_hash("Zentara Infra wins Contract!") == headline_hash(
        "zentara infra, wins contract"
    )
    assert token_set_similarity("a b c d", "a b c e") == pytest.approx(3 / 5)
    assert token_set_similarity("", "a") == 0.0


def test_duplicates_are_consolidated_into_the_earliest_article():
    original = make_article(article_id="wire", published_at=_at(2), symbols=["ZENTARA.NS"])
    exact = make_article(
        article_id="copy", headline="ZENTARA INFRA wins 450 crore highway contract!",
        source="Daily Echo", published_at=_at(5), text="Later copy with more detail: 999.",
        symbols=["ZENTARA.BO"],
    )  # fmt: skip
    near = make_article(
        article_id="near", headline="Zentara Infra wins 450 crore highway contract today",
        source="Market Minute", published_at=_at(0), text="",
    )  # fmt: skip
    other = make_article(article_id="other", headline="Quillon Pharma recalls a batch",
                         published_at=_at(1))  # fmt: skip

    unique, groups = consolidate([original, exact, near, other])
    assert [a.article_id for a in unique] == ["near", "other"]
    kept = unique[0]
    assert kept.published_at == _at(0)  # earliest publication time
    assert kept.duplicate_ids == ["wire", "copy"] and kept.source_ids == ["near", "wire", "copy"]
    assert kept.duplicate_sources == ["Fictional Wire", "Daily Echo"]
    assert kept.symbols == ["ZENTARA.NS", "ZENTARA.BO"]
    # Text of a later copy is never merged in: it may post-date the earliest article.
    assert kept.text == "" and kept.headline == near.headline
    assert [g.to_dict() for g in groups] == [{"kept": "near", "merged": ["wire", "copy"]}]


def test_recurring_headline_outside_the_window_is_a_new_story():
    monday = make_article(article_id="m", headline="Market wrap: stocks end higher",
                          published_at=_at(0))  # fmt: skip
    next_week = monday.model_copy(update={"article_id": "n", "published_at": _at(24 * 7)})
    unique, groups = consolidate([monday, next_week], DedupConfig(window_hours=72))
    assert len(unique) == 2 and groups == []
    unique, groups = consolidate([monday, next_week], DedupConfig(window_hours=24 * 8))
    assert len(unique) == 1 and unique[0].article_id == "m"


def test_same_article_id_is_always_merged_and_distinct_stories_are_kept():
    first = make_article(
        article_id="x", headline="Zentara Infra wins contract", published_at=_at(0)
    )
    again = make_article(article_id="x", headline="Zentara Infra wins contract (updated)",
                         published_at=_at(24 * 30), symbols=["ZENTARA.NS"])  # fmt: skip
    different = make_article(article_id="y", headline="Zentara Infra loses contract dispute",
                             published_at=_at(1))  # fmt: skip
    unique, _ = consolidate([again, first, different])
    assert [a.article_id for a in unique] == ["x", "y"]
    assert unique[0].duplicate_ids == [] and unique[0].symbols == ["ZENTARA.NS"]
    strict = consolidate([first, different], DedupConfig(similarity_threshold=0.5))[0]
    assert len(strict) == 1  # threshold is configurable
    assert consolidate([]) == ([], [])
