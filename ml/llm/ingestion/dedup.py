"""Duplicate detection and consolidation of news articles (PRD 5.3, 8.4).

Two articles are the same story when
    * they have the same ``article_id``; or
    * their normalised headlines are identical, or their headline token sets have a
      Jaccard similarity >= ``similarity_threshold``, **and** they were published
      within ``window_hours`` of each other (a recurring headline such as a daily
      market wrap is not a duplicate of last week's).

A group is consolidated into its **earliest** article: that article's id, headline,
text and publication time are kept unchanged, and the others are recorded in
``duplicate_ids`` / ``duplicate_sources``. Text from a later copy is deliberately *not*
merged in -- it could contain information that did not exist at the earliest
publication time. Ticker hints are unioned.

Counting a story once matters: otherwise a wire story syndicated by ten outlets would
weigh ten times as much in the aggregated signal.

Complexity is O(n^2) headline comparisons, which is fine for the thousands of items a
run handles; it is not an index for a large archive.
"""

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import timedelta

from ml.llm.config import DedupConfig
from ml.llm.schema import NewsArticle

_TOKEN = re.compile(r"[a-z0-9]+")


def headline_tokens(headline: str) -> list[str]:
    """Lower-case alphanumeric tokens of a headline, accents and punctuation removed."""
    text = unicodedata.normalize("NFKD", headline).encode("ascii", "ignore").decode()
    return _TOKEN.findall(text.lower())


def headline_hash(headline: str) -> str:
    return hashlib.sha256(" ".join(headline_tokens(headline)).encode()).hexdigest()


def token_set_similarity(a: str, b: str) -> float:
    """Jaccard similarity of two headlines' token sets, in [0, 1]."""
    left, right = set(headline_tokens(a)), set(headline_tokens(b))
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


@dataclass(frozen=True)
class DuplicateGroup:
    kept: str
    merged: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"kept": self.kept, "merged": list(self.merged)}


def consolidate(
    articles: list[NewsArticle], config: DedupConfig | None = None
) -> tuple[list[NewsArticle], list[DuplicateGroup]]:
    """Return (unique articles ordered by publication time, the groups that were merged)."""
    config = config or DedupConfig()
    ordered = sorted(articles, key=lambda a: (a.published_at, a.article_id))
    parent = list(range(len(ordered)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[max(root_i, root_j)] = min(root_i, root_j)  # earliest stays the root

    first_with_id: dict[str, int] = {}
    for index, article in enumerate(ordered):
        union(first_with_id.setdefault(article.article_id, index), index)

    window = timedelta(hours=config.window_hours)
    hashes = [headline_hash(a.headline) for a in ordered]
    token_sets = [set(headline_tokens(a.headline)) for a in ordered]
    for i, first in enumerate(ordered):
        for j in range(i + 1, len(ordered)):
            if ordered[j].published_at - first.published_at > window:
                break  # ordered by time: nothing later can be inside the window
            both = token_sets[i] | token_sets[j]
            similarity = len(token_sets[i] & token_sets[j]) / len(both) if both else 0.0
            if hashes[i] == hashes[j] or similarity >= config.similarity_threshold:
                union(i, j)

    members: dict[int, list[int]] = {}
    for index in range(len(ordered)):
        members.setdefault(find(index), []).append(index)

    unique: list[NewsArticle] = []
    groups: list[DuplicateGroup] = []
    for root, indices in members.items():
        kept = ordered[root]
        others = [ordered[i] for i in indices if i != root]
        if not others:
            unique.append(kept)
            continue
        duplicate_ids = [*kept.duplicate_ids]
        duplicate_sources = [*kept.duplicate_sources]
        symbols = [*kept.symbols]
        for other in others:
            for article_id in other.source_ids:
                if article_id != kept.article_id and article_id not in duplicate_ids:
                    duplicate_ids.append(article_id)
            for source in (other.source, *other.duplicate_sources):
                if source != kept.source and source not in duplicate_sources:
                    duplicate_sources.append(source)
            symbols.extend(s for s in other.symbols if s not in symbols)
        unique.append(
            kept.model_copy(
                update={
                    "duplicate_ids": duplicate_ids,
                    "duplicate_sources": duplicate_sources,
                    "symbols": symbols,
                }
            )
        )
        groups.append(DuplicateGroup(kept.article_id, tuple(a.article_id for a in others)))
    unique.sort(key=lambda a: (a.published_at, a.article_id))
    return unique, groups
