"""A small local document store for retrieval-augmented summaries (PRD 8.4).

On disk (``<path>/``): ``documents.jsonl`` (NewsArticle per line), ``embeddings.npy``
(float32 matrix) and ``meta.json`` (embedding model, the document id of each row).

Search
    * **embedding** -- cosine similarity between the query and document embeddings from
      the client's ``embed``; used when the client can embed and every candidate
      document has a vector.
    * **lexical** -- TF-IDF cosine over lightly stemmed words; the fallback when no
      embedding model is configured, the provider fails, or some documents have no
      vector. The method actually used is returned with every hit.

``as_of`` restricts a search to documents published at or before that instant, so a
summary "as of T" cannot draw on later articles.

Limitations: brute-force search over an in-memory matrix -- fine for thousands of
documents, not a vector database. Embeddings from different models are not comparable;
the store refuses to mix them.
"""

import json
import logging
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from ml.llm.extraction.validation import content_words
from ml.llm.providers import LLMClient, LLMError
from ml.llm.schema import NewsArticle
from ml.llm.securities import SecuritiesMaster

logger = logging.getLogger(__name__)

DOCUMENTS, EMBEDDINGS, META = "documents.jsonl", "embeddings.npy", "meta.json"


@dataclass(frozen=True)
class SearchHit:
    article: NewsArticle
    score: float
    method: str  # "embedding" | "lexical"


class DocumentStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.documents: dict[str, NewsArticle] = {}
        self.embedding_model: str | None = None
        self._vector_ids: list[str] = []
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        if (self.path / DOCUMENTS).is_file():
            self._load()

    # --------------------------------------------------------------- persistence

    def _load(self) -> None:
        with open(self.path / DOCUMENTS, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    article = NewsArticle.model_validate_json(line)
                    self.documents[article.article_id] = article
        if (self.path / META).is_file() and (self.path / EMBEDDINGS).is_file():
            with open(self.path / META, encoding="utf-8") as handle:
                meta = json.load(handle)
            self.embedding_model = meta["embedding_model"]
            self._vector_ids = meta["ids"]
            self._vectors = np.load(self.path / EMBEDDINGS, allow_pickle=False)

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / DOCUMENTS, "w", encoding="utf-8", newline="\n") as handle:
            for article in self.documents.values():
                handle.write(article.model_dump_json() + "\n")
        np.save(self.path / EMBEDDINGS, self._vectors, allow_pickle=False)
        with open(self.path / META, "w", encoding="utf-8") as handle:
            json.dump({"embedding_model": self.embedding_model, "ids": self._vector_ids}, handle)

    # ------------------------------------------------------------------ indexing

    def add(self, articles: Iterable[NewsArticle], client: LLMClient | None = None) -> int:
        """Add new documents (known ids are skipped) and embed them when possible.

        Returns the number of documents added. An embedding failure is logged and the
        documents are stored without vectors (lexical search still works).
        """
        new = [a for a in articles if a.article_id not in self.documents]
        new = list({a.article_id: a for a in new}.values())
        for article in new:
            self.documents[article.article_id] = article
        if new and client is not None:
            model = getattr(client, "embedding_model", None) or client.model
            if self.embedding_model not in (None, model):
                logger.warning(
                    "document store was embedded with %s; not mixing in vectors from %s",
                    self.embedding_model,
                    model,
                )
            else:
                try:
                    vectors = np.asarray(client.embed([a.content for a in new]), dtype=np.float32)
                except LLMError as error:
                    logger.warning("embeddings unavailable (%s); using lexical search", error)
                else:
                    if self._vectors.size == 0:
                        self._vectors = vectors
                    else:
                        self._vectors = np.vstack([self._vectors, vectors])
                    self._vector_ids.extend(a.article_id for a in new)
                    self.embedding_model = model
        if new:
            self.save()
        return len(new)

    # -------------------------------------------------------------------- search

    def _candidates(
        self, symbol: str | None, master: SecuritiesMaster | None, as_of: datetime | None
    ) -> list[NewsArticle]:
        if symbol and master is None:
            raise ValueError("a securities master is needed to filter by symbol")
        chosen = []
        for article in self.documents.values():
            if as_of is not None and article.published_at > as_of:
                continue
            if symbol and symbol not in master.referenced(article):
                continue
            chosen.append(article)
        return chosen

    @staticmethod
    def _lexical(query: str, candidates: list[NewsArticle]) -> list[float]:
        bags = [Counter(content_words(a.content)) for a in candidates]
        frequency = Counter(word for bag in bags for word in bag)
        idf = {w: math.log(1 + len(bags) / count) for w, count in frequency.items()}
        query_bag = Counter(content_words(query))
        query_norm = math.sqrt(sum((n * idf.get(w, 0.0)) ** 2 for w, n in query_bag.items()))
        scores = []
        for bag in bags:
            norm = math.sqrt(sum((n * idf[w]) ** 2 for w, n in bag.items()))
            dot = sum(n * idf[w] * query_bag[w] * idf[w] for w, n in bag.items() if w in query_bag)
            scores.append(dot / (norm * query_norm) if norm and query_norm else 0.0)
        return scores

    def _embedding(
        self, query: str, candidates: list[NewsArticle], client: LLMClient | None
    ) -> list[float] | None:
        rows = {article_id: i for i, article_id in enumerate(self._vector_ids)}
        if client is None or not candidates or any(a.article_id not in rows for a in candidates):
            return None
        try:
            vector = np.asarray(client.embed([query])[0], dtype=np.float32)
        except LLMError as error:
            logger.warning("query embedding unavailable (%s); using lexical search", error)
            return None
        matrix = self._vectors[[rows[a.article_id] for a in candidates]]
        if matrix.shape[1] != vector.shape[0]:
            return None
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
        return [float(x) for x in np.divide(matrix @ vector, norms, where=norms > 0,
                                            out=np.zeros(len(candidates)))]  # fmt: skip

    def search(
        self,
        query: str,
        *,
        k: int = 6,
        symbol: str | None = None,
        master: SecuritiesMaster | None = None,
        as_of: datetime | None = None,
        client: LLMClient | None = None,
    ) -> list[SearchHit]:
        """The ``k`` best documents for ``query`` (ties: newest first), best first."""
        candidates = self._candidates(symbol, master, as_of)
        scores = self._embedding(query, candidates, client)
        method = "embedding"
        if scores is None:
            scores, method = self._lexical(query, candidates), "lexical"
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: (-pair[1], -pair[0].published_at.timestamp(), pair[0].article_id),
        )
        return [SearchHit(article, score, method) for article, score in ranked[:k]]
