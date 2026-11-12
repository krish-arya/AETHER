"""Yahoo Finance search endpoint as a free news source (no API key).

What it really is -- and is not
    * An *unofficial* endpoint with no service guarantee. Yahoo's terms allow personal
      research use, not redistribution.
    * **Headlines only**: title, publisher, link, publication time and the tickers
      Yahoo relates to the item. There is no article body, so ``NewsArticle.text`` is
      empty and extraction can only ever produce low-detail events.
    * **No history**: only the most recent items are returned. Historical news features
      for backtests cannot be built from this source.
    * It is a fuzzy search, not a per-company feed. A query for an NSE ticker
      (``RELIANCE.NS``) frequently returns no news at all; a query for the company name
      returns more but includes unrelated stories. The queried symbol is therefore
      **not** recorded as a hint -- an item is only linked to a company if Yahoo tagged
      it with a related ticker or the headline names the company (entity resolution).

Reproducibility
    Each response is cached under ``cache_dir`` per query and UTC day, together with
    the time it was retrieved, and reused for the rest of that day unless
    ``refresh=True``.

Reliability
    Sequential requests with a pause between them; exponential back-off on timeouts,
    HTTP 429 and 5xx.
"""

import json
import logging
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx

from ml.llm.ingestion.base import NewsSource, NewsSourceError, NewsSourceInfo
from ml.llm.schema import NewsArticle, now_utc
from ml.llm.securities import SecuritiesMaster

logger = logging.getLogger(__name__)

SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)
RETRY_STATUS = {429, 500, 502, 503, 504}
LICENSE_NOTE = (
    "Yahoo Finance search API (unofficial, no API key). Personal research use under "
    "Yahoo's terms; not for redistribution. Headlines and metadata only -- no article "
    "text and no historical archive."
)


class YahooNewsSource(NewsSource):
    def __init__(
        self,
        master: SecuritiesMaster,
        cache_dir: str | Path,
        *,
        exchange_suffix: str = ".NS",
        news_count: int = 20,
        query_company_name: bool = True,
        refresh: bool = False,
        client: httpx.Client | None = None,
        pause_seconds: float = 0.5,
        max_retries: int = 3,
        clock: Callable[[], datetime] = now_utc,
    ) -> None:
        self.master = master
        self.cache_dir = Path(cache_dir)
        self.exchange_suffix = exchange_suffix
        self.news_count = news_count
        self.query_company_name = query_company_name
        self.refresh = refresh
        self._client = client
        self.pause_seconds = pause_seconds
        self.max_retries = max_retries
        self._clock = clock
        self._log: list[dict] = []

    @property
    def info(self) -> NewsSourceInfo:
        return NewsSourceInfo(
            name="yahoo_search_news",
            license_note=LICENSE_NOTE,
            provides_body=False,
            provides_history=False,
        )

    def fetch_log(self) -> list[dict]:
        return list(self._log)

    # --------------------------------------------------------------------- fetch

    def _cache_file(self, query: str, day: datetime) -> Path:
        safe = "".join(ch if ch.isalnum() else "_" for ch in query)
        return self.cache_dir / f"{safe}_{day:%Y%m%d}.json"

    def _download(self, query: str) -> dict:
        params = {"q": query, "newsCount": self.news_count, "quotesCount": 0}
        client = self._client or httpx.Client(timeout=30.0, headers={"User-Agent": USER_AGENT})
        failure = "no attempt made"
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    response = client.get(SEARCH_URL, params=params)
                except httpx.TransportError as error:
                    failure = f"{type(error).__name__}: {error}"
                else:
                    if response.status_code == 200:
                        try:
                            return response.json()
                        except ValueError as error:
                            raise NewsSourceError(
                                f"Yahoo news search for {query!r} returned invalid JSON"
                            ) from error
                    failure = f"HTTP {response.status_code}"
                    if response.status_code not in RETRY_STATUS:
                        break
                if attempt < self.max_retries:
                    delay = self.pause_seconds * 2 ** (attempt + 1)
                    logger.warning("yahoo news %r: %s; retrying in %.1fs", query, failure, delay)
                    time.sleep(delay)
        finally:
            if self._client is None:
                client.close()
        raise NewsSourceError(f"Yahoo news search for {query!r} failed: {failure}")

    def _items(self, query: str) -> tuple[list[dict], datetime, bool]:
        """News items for one query, when they were retrieved, and whether from cache."""
        now = self._clock().astimezone(UTC)
        cache = self._cache_file(query, now)
        if cache.is_file() and not self.refresh:
            with open(cache, encoding="utf-8") as handle:
                stored = json.load(handle)
            return stored["news"], datetime.fromisoformat(stored["retrieved_at"]), True
        payload = self._download(query)
        news = payload.get("news")
        if not isinstance(news, list):
            raise NewsSourceError(f"Yahoo news search for {query!r} returned no 'news' list")
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w", encoding="utf-8") as handle:
            json.dump({"query": query, "retrieved_at": now.isoformat(), "news": news}, handle)
        time.sleep(self.pause_seconds)  # be polite between live requests
        return news, now, False

    # --------------------------------------------------------------------- parse

    @staticmethod
    def _article(item: dict, retrieved_at: datetime) -> NewsArticle | None:
        """One Yahoo item as an article, or None when a mandatory field is missing."""
        uuid, title, published = (
            item.get("uuid"),
            item.get("title"),
            item.get("providerPublishTime"),
        )
        if not uuid or not title or not isinstance(published, int | float):
            return None
        return NewsArticle(
            article_id=f"yahoo_{uuid}",
            headline=str(title)[:1000],
            text="",  # the endpoint has no article body
            source=str(item.get("publisher") or "Yahoo Finance")[:128],
            url=item.get("link"),
            published_at=datetime.fromtimestamp(published, UTC),
            retrieved_at=retrieved_at,
            symbols=[str(t) for t in item.get("relatedTickers") or []],
        )

    def _queries(self, symbol: str) -> list[str]:
        queries = [f"{symbol}{self.exchange_suffix}"]
        security = self.master.get(symbol)
        if self.query_company_name and security and security.name != symbol:
            queries.append(security.name)
        return queries

    def fetch(self, symbols: Sequence[str]) -> list[NewsArticle]:
        unknown = sorted(s for s in symbols if s not in self.master)
        if unknown:
            raise NewsSourceError(f"Symbols not in the configured universe: {unknown}")
        articles: dict[str, NewsArticle] = {}
        for symbol in symbols:
            for query in self._queries(symbol):
                items, retrieved_at, cached = self._items(query)
                parsed = [self._article(item, retrieved_at) for item in items]
                kept = [article for article in parsed if article is not None]
                self._log.append(
                    {
                        "symbol": symbol,
                        "query": query,
                        "items": len(items),
                        "unparseable": len(items) - len(kept),
                        "from_cache": cached,
                    }
                )
                for article in kept:
                    known = articles.get(article.article_id)
                    if known is None:
                        articles[article.article_id] = article
                    else:  # same item returned by another query: merge the ticker tags
                        merged = list(dict.fromkeys([*known.symbols, *article.symbols]))
                        articles[article.article_id] = known.model_copy(update={"symbols": merged})
        return sorted(articles.values(), key=lambda a: (a.published_at, a.article_id))
