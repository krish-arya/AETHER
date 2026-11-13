"""Google News historical source, concurrent analysis and token accounting."""

import threading
import time
from datetime import UTC, date, datetime

import httpx
import pytest

from ml.llm.ingestion.google_news import (
    GoogleNewsSource,
    end_of_market_day,
    month_windows,
)
from ml.llm.pipeline import analyze_articles
from ml.llm.providers import LLMUnavailableError, OpenAICompatibleClient
from ml.llm.schema import NewsArticle
from ml.llm.securities import SecuritiesMaster, SecurityInfo

FEED = """<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>
<title>search</title>
<item><title>Infosys wins large deal - Mint</title><link>https://x.test/a</link>
<guid>g-a</guid><pubDate>Mon, 04 Mar 2019 08:00:00 GMT</pubDate>
<source url="https://mint.test">Mint</source></item>
<item><title>Infosys shares slip after results - Reuters</title><link>https://x.test/b</link>
<guid>g-b</guid><pubDate>Tue, 05 Mar 2019 20:30:00 GMT</pubDate>
<source url="https://reuters.test">Reuters</source></item>
<item><title>No date here</title><link>https://x.test/c</link><guid>g-c</guid>
<pubDate></pubDate><source>Blog</source></item>
</channel></rss>"""


def _master() -> SecuritiesMaster:
    return SecuritiesMaster([SecurityInfo("INFY", "Infosys", "IT")])


def test_month_windows_and_end_of_day():
    assert month_windows(date(2019, 11, 15), date(2020, 1, 2)) == [
        (date(2019, 11, 1), date(2019, 12, 1)),
        (date(2019, 12, 1), date(2020, 1, 1)),
        (date(2020, 1, 1), date(2020, 2, 1)),
    ]
    # 08:00 UTC = 13:30 IST, before the close: still only usable from the next session.
    moved = end_of_market_day(datetime(2019, 3, 4, 8, 0, tzinfo=UTC))
    assert moved == datetime(2019, 3, 4, 18, 29, 59, tzinfo=UTC)
    # 20:30 UTC is already 5 March in India.
    late = end_of_market_day(datetime(2019, 3, 4, 20, 30, tzinfo=UTC))
    assert late == datetime(2019, 3, 5, 18, 29, 59, tzinfo=UTC)


def test_parse_feed_and_article_mapping():
    items = GoogleNewsSource.parse_feed(FEED)
    assert len(items) == 3 and items[0]["source"] == "Mint"
    retrieved = datetime(2026, 11, 13, tzinfo=UTC)
    first = GoogleNewsSource.to_article(items[0], retrieved)
    assert first.headline == "Infosys wins large deal"  # publisher suffix removed
    assert first.source == "Mint" and first.text == "" and first.symbols == []
    assert first.published_at == datetime(2019, 3, 4, 18, 29, 59, tzinfo=UTC)
    assert first.article_id == GoogleNewsSource.to_article(items[0], retrieved).article_id
    assert GoogleNewsSource.to_article(items[2], retrieved) is None  # no date: dropped


def test_fetch_caches_and_never_tags_the_queried_symbol(tmp_path):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text=FEED)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    source = GoogleNewsSource(
        _master(), tmp_path, date(2019, 3, 1), date(2019, 3, 31), client=client,
        pause_seconds=0,
    )  # fmt: skip
    articles = source.fetch(["INFY"])
    assert len(requests) == 1
    query = requests[0].url.params["q"]
    assert query.startswith('"Infosys"') and "after:2019-02-28" in query
    assert "before:2019-04-01" in query
    assert [a.headline for a in articles] == [
        "Infosys wins large deal",
        "Infosys shares slip after results",
    ]
    assert all(a.symbols == [] for a in articles)
    assert source.info.provides_history and not source.info.provides_body

    again = GoogleNewsSource(
        _master(), tmp_path, date(2019, 3, 1), date(2019, 3, 31), client=client,
        pause_seconds=0,
    )  # fmt: skip
    assert [a.article_id for a in again.fetch(["INFY"])] == [a.article_id for a in articles]
    assert len(requests) == 1 and again.fetch_log()[0]["from_cache"]


def test_fetch_drops_items_outside_the_window(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=FEED)))
    source = GoogleNewsSource(
        _master(), tmp_path, date(2019, 3, 5), date(2019, 3, 31), client=client, pause_seconds=0
    )
    # The 4 March item is outside [5 March, 31 March] in market time.
    assert [a.headline for a in source.fetch(["INFY"])] == ["Infosys shares slip after results"]


def test_fetch_rejects_unknown_symbols(tmp_path):
    source = GoogleNewsSource(_master(), tmp_path, date(2019, 3, 1), date(2019, 3, 31))
    with pytest.raises(Exception, match="not in the configured universe"):
        source.fetch(["NOPE"])


# ------------------------------------------------------------- concurrent analysis


class _SlowAnalyzer:
    def __init__(self, fail_ids=()):
        self.fail_ids = set(fail_ids)
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def analyze(self, article, *, now):
        from ml.llm.extraction.analyzer import ArticleAnalysis

        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
        if article.article_id in self.fail_ids:
            raise LLMUnavailableError("down")
        return ArticleAnalysis(article=article, llm_called=True)


def _articles(n: int) -> list[NewsArticle]:
    stamp = datetime(2019, 3, 4, tzinfo=UTC)
    return [
        NewsArticle(
            article_id=f"a{i}", headline=f"h{i}", source="s", published_at=stamp, retrieved_at=stamp
        )  # fmt: skip
        for i in range(n)
    ]


def test_concurrent_analysis_keeps_order_and_runs_in_parallel():
    analyzer = _SlowAnalyzer(fail_ids={"a3"})
    articles = _articles(12)
    seen: list[int] = []
    results = analyze_articles(
        analyzer, articles, now=datetime.now(UTC), workers=4,
        progress=lambda i, n, a: seen.append(i),
    )  # fmt: skip
    assert [r.article.article_id for r in results] == [a.article_id for a in articles]
    assert seen == list(range(1, 13))
    assert analyzer.peak > 1
    assert results[3].rejections[0].reason.value == "llm_error"


def test_concurrent_analysis_stops_when_the_provider_is_down():
    analyzer = _SlowAnalyzer(fail_ids={f"a{i}" for i in range(12)})
    with pytest.raises(LLMUnavailableError):
        analyze_articles(analyzer, _articles(12), now=datetime.now(UTC), workers=4)


def test_client_sums_token_usage():
    body = {
        "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30},
    }
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=body))
    client = OpenAICompatibleClient(
        base_url="http://llm.test/v1", model="m", api_key=None,
        client=httpx.Client(transport=transport),
    )  # fmt: skip
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    client.complete_json("s", "u", schema)
    client.complete_json("s", "u", schema)
    assert client.usage == {"prompt_tokens": 240, "completion_tokens": 60}
