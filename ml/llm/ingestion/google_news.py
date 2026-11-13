"""Google News RSS search as a free *historical* headline source (no API key).

What it really is -- and is not
    * Google News' public RSS search (``news.google.com/rss/search``) with the
      ``after:`` / ``before:`` date operators. Unofficial for this use and with no
      service guarantee; personal research use only, not redistribution.
    * **Headlines only**: title, publisher, link and a publication date. No article
      body, so ``NewsArticle.text`` is empty.
    * **A sample, not a complete archive.** Each query returns at most about 100 items
      chosen by Google's relevance ranking, typically ~20 per company per month. Coverage
      differs by company and year, and the selection is Google's, made today.
    * The query is the company name; an item is linked to a company only through entity
      resolution on its headline, never because of the query that found it.

Point-in-time safety
    The feed's ``pubDate`` is frequently rounded to a whole hour or day, so the true
    publication time is unknown. Each article's ``published_at`` is therefore moved to
    the **end of its publication day in market time** (23:59:59 Asia/Kolkata): a
    headline is never available to a decision taken at that day's close, only from the
    next session on. This costs up to a day of timeliness and rules out same-day
    look-ahead.

    It does **not** rule out hindsight in the model: a language model trained after
    the news can know how a story turned out. See ``docs/llm-engine.md``.

Reproducibility
    Responses are cached per query and month under ``cache_dir`` and reused (past months
    do not change meaningfully); ``refresh=True`` downloads again.
"""

import hashlib
import json
import logging
import time
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from ml.llm.ingestion.base import NewsSource, NewsSourceError, NewsSourceInfo
from ml.llm.schema import NewsArticle, now_utc
from ml.llm.securities import SecuritiesMaster

logger = logging.getLogger(__name__)

SEARCH_URL = "https://news.google.com/rss/search"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0"
RETRY_STATUS = {429, 500, 502, 503, 504}
MARKET_TZ = ZoneInfo("Asia/Kolkata")
LICENSE_NOTE = (
    "Google News RSS search (unofficial for this use, no API key). Personal research use; "
    "not for redistribution. Headlines and publication dates only; a relevance-ranked "
    "sample per query and month, not a complete archive."
)


def month_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Calendar months covering ``[start, end]`` as (first day, first day of next)."""
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
    return [(m.start_time.date(), (m + 1).start_time.date()) for m in months]


def end_of_market_day(moment: datetime) -> datetime:
    """23:59:59 Asia/Kolkata on the market-local day of ``moment``, in UTC."""
    local_day = moment.astimezone(MARKET_TZ).date()
    closing = datetime(local_day.year, local_day.month, local_day.day, 23, 59, 59, tzinfo=MARKET_TZ)
    return closing.astimezone(UTC)


class GoogleNewsSource(NewsSource):
    def __init__(
        self,
        master: SecuritiesMaster,
        cache_dir: str | Path,
        start: date,
        end: date,
        *,
        max_per_query: int | None = None,
        refresh: bool = False,
        client: httpx.Client | None = None,
        pause_seconds: float = 1.5,
        max_retries: int = 3,
        language: str = "en-IN",
        country: str = "IN",
    ) -> None:
        if end < start:
            raise ValueError("end must not be before start")
        self.master = master
        self.cache_dir = Path(cache_dir)
        self.start, self.end = start, end
        self.max_per_query = max_per_query
        self.refresh = refresh
        self._client = client
        self.pause_seconds = pause_seconds
        self.max_retries = max_retries
        self.language, self.country = language, country
        self._log: list[dict] = []

    @property
    def info(self) -> NewsSourceInfo:
        return NewsSourceInfo(
            name="google_news_rss",
            license_note=LICENSE_NOTE,
            provides_body=False,
            provides_history=True,
        )

    def fetch_log(self) -> list[dict]:
        return list(self._log)

    # ------------------------------------------------------------------ download

    def _query(self, name: str, first: date, after_last: date) -> str:
        # ``before:`` is exclusive and ``after:`` inclusive in practice; one day of
        # overlap at month edges is removed by article-id de-duplication.
        return f'"{name}" after:{first - timedelta(days=1):%Y-%m-%d} before:{after_last:%Y-%m-%d}'

    def _cache_file(self, name: str, first: date) -> Path:
        safe = "".join(ch if ch.isalnum() else "_" for ch in name)
        return self.cache_dir / safe / f"{first:%Y-%m}.json"

    def _download(self, query: str) -> str:
        params = {"q": query, "hl": self.language, "gl": self.country,
                  "ceid": f"{self.country}:{self.language.split('-')[0]}"}  # fmt: skip
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
                        return response.text
                    failure = f"HTTP {response.status_code}"
                    if response.status_code not in RETRY_STATUS:
                        break
                if attempt < self.max_retries:
                    delay = max(self.pause_seconds, 1.0) * 2 ** (attempt + 2)
                    logger.warning("google news %r: %s; retrying in %.0fs", query, failure, delay)
                    time.sleep(delay)
        finally:
            if self._client is None:
                client.close()
        raise NewsSourceError(f"Google News search for {query!r} failed: {failure}")

    @staticmethod
    def parse_feed(xml_text: str) -> list[dict]:
        """Items of an RSS document as dicts (title, link, guid, pubDate, source)."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as error:
            raise NewsSourceError("Google News returned invalid XML") from error
        items = []
        for item in root.iter("item"):
            source = item.find("source")
            items.append(
                {
                    "title": (item.findtext("title") or "").strip(),
                    "link": (item.findtext("link") or "").strip() or None,
                    "guid": (item.findtext("guid") or "").strip() or None,
                    "pubDate": (item.findtext("pubDate") or "").strip(),
                    "source": (source.text or "").strip() if source is not None else "",
                }
            )
        return items

    def _items(self, name: str, first: date, after_last: date) -> tuple[list[dict], datetime, bool]:
        cache = self._cache_file(name, first)
        if cache.is_file() and not self.refresh:
            stored = json.loads(cache.read_text(encoding="utf-8"))
            return stored["items"], datetime.fromisoformat(stored["retrieved_at"]), True
        query = self._query(name, first, after_last)
        items = self.parse_feed(self._download(query))
        retrieved = now_utc()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps({"query": query, "retrieved_at": retrieved.isoformat(), "items": items}),
            encoding="utf-8",
        )
        time.sleep(self.pause_seconds)
        return items, retrieved, False

    # --------------------------------------------------------------------- parse

    @staticmethod
    def to_article(item: dict, retrieved_at: datetime) -> NewsArticle | None:
        """One feed item as an article, or None when a mandatory field is missing."""
        title, source = item.get("title") or "", item.get("source") or ""
        try:
            published = parsedate_to_datetime(item.get("pubDate") or "")
        except (TypeError, ValueError):
            return None
        if not title or published is None:
            return None
        if published.tzinfo is None:
            published = published.replace(tzinfo=UTC)
        suffix = f" - {source}"
        if source and title.endswith(suffix):  # Google appends " - Publisher"
            title = title[: -len(suffix)].strip()
        if not title:
            return None
        key = item.get("guid") or item.get("link") or f"{source}|{title}|{published.isoformat()}"
        return NewsArticle(
            article_id="gnews_" + hashlib.sha256(key.encode()).hexdigest()[:24],
            headline=title[:1000],
            text="",
            source=(source or "Google News")[:128],
            url=item.get("link"),
            published_at=end_of_market_day(published),
            retrieved_at=retrieved_at,
            symbols=[],  # never the queried symbol: entity resolution decides
        )

    def fetch(self, symbols: Sequence[str]) -> list[NewsArticle]:
        unknown = sorted(s for s in symbols if s not in self.master)
        if unknown:
            raise NewsSourceError(f"Symbols not in the configured universe: {unknown}")
        articles: dict[str, NewsArticle] = {}
        windows = month_windows(self.start, self.end)
        lo = datetime(self.start.year, self.start.month, self.start.day, tzinfo=MARKET_TZ)
        hi = datetime(self.end.year, self.end.month, self.end.day, tzinfo=MARKET_TZ) + timedelta(
            days=1
        )
        for symbol in symbols:
            security = self.master.get(symbol)
            name = security.name if security else symbol
            for first, after_last in windows:
                items, retrieved, cached = self._items(name, first, after_last)
                parsed = [self.to_article(item, retrieved) for item in items]
                kept = [a for a in parsed if a is not None and lo <= a.published_at < hi]
                if self.max_per_query is not None:
                    kept = kept[: self.max_per_query]  # feed order = Google's ranking
                entry = {"symbol": symbol, "query": name, "month": f"{first:%Y-%m}"}
                entry.update(items=len(items), kept=len(kept), from_cache=cached)
                self._log.append(entry)
                for article in kept:
                    articles.setdefault(article.article_id, article)
        return sorted(articles.values(), key=lambda a: (a.published_at, a.article_id))
