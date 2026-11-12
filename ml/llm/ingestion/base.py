"""News source interface.

Every source declares what it really provides (``NewsSourceInfo``) so that documents,
manifests and API responses can state it -- e.g. that a source returns headlines only.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from ml.llm.schema import NewsArticle


class NewsSourceError(RuntimeError):
    """A news source could not be read or returned something unusable."""


@dataclass(frozen=True)
class NewsSourceInfo:
    name: str
    license_note: str
    provides_body: bool
    provides_history: bool

    def to_dict(self) -> dict:
        return asdict(self)


class NewsSource(ABC):
    """Returns the news items a provider currently has for a set of symbols."""

    @property
    @abstractmethod
    def info(self) -> NewsSourceInfo: ...

    @abstractmethod
    def fetch(self, symbols: Sequence[str]) -> list[NewsArticle]:
        """Articles for ``symbols``. May contain duplicates and irrelevant items; both
        are dealt with downstream (``dedup`` and entity resolution)."""

    def fetch_log(self) -> list[dict]:
        """What was requested and how much came back, for the run manifest."""
        return []
