"""Securities master for entity resolution (PRD 8.2-C, 8.4).

Answers one question deterministically, without a model: *which known securities does
this article reference, and on what evidence?* An event the model attributes to any
other symbol is rejected.

Evidence, strongest first
    ``alias``   a company name / alias occurs in the article (case-insensitive, whole
                words), or the NSE symbol occurs as an upper-case token.
    ``hint``    the news source tagged the article with the company's ticker
                (``RELIANCE.NS``, or a configured foreign ticker such as an ADR).

``alias_exclusions`` are phrases that contain an alias but name a *different* company
("SBI Cards" contains the alias "SBI"); they are masked before aliases are matched.

Limitations: alias matching is literal. It does not understand pronouns, subsidiaries
or context, and only knows the aliases and exclusions it is given.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from ml.llm.schema import NewsArticle

EXCHANGE_SUFFIXES = (".NS", ".BO")


@dataclass(frozen=True)
class SecurityInfo:
    symbol: str
    name: str
    sector: str | None = None


@dataclass(frozen=True)
class EntityMatch:
    symbol: str
    evidence: str  # "alias" | "hint"
    matched: str  # the alias or ticker that matched


class SecuritiesMaster:
    """Known securities with their names, sectors and aliases."""

    def __init__(
        self,
        securities: Iterable[SecurityInfo],
        company_aliases: Mapping[str, Iterable[str]] | None = None,
        ticker_aliases: Mapping[str, str] | None = None,
        alias_exclusions: Iterable[str] = (),
    ) -> None:
        self._exclusions = [
            self._word_pattern(phrase, re.IGNORECASE)
            for phrase in alias_exclusions
            if phrase.strip()
        ]
        self._securities = {s.symbol.upper(): s for s in securities}
        if not self._securities:
            raise ValueError("the securities master is empty")
        unknown = sorted(set(company_aliases or {}) - set(self._securities))
        unknown += sorted(set((ticker_aliases or {}).values()) - set(self._securities))
        if unknown:
            raise ValueError(f"aliases refer to symbols that are not in the universe: {unknown}")
        self._ticker_aliases = {t.upper(): s for t, s in (ticker_aliases or {}).items()}
        self._patterns: list[tuple[str, str, re.Pattern[str]]] = []
        for symbol, security in self._securities.items():
            names = [security.name, *(company_aliases or {}).get(symbol, [])]
            for alias in dict.fromkeys(" ".join(n.split()) for n in names if n and n.strip()):
                self._patterns.append((symbol, alias, self._word_pattern(alias, re.IGNORECASE)))
            # The bare symbol only counts in upper case ("LT", "ITC" are ordinary words
            # or fragments otherwise).
            self._patterns.append((symbol, symbol, self._word_pattern(symbol, 0)))

    @staticmethod
    def _word_pattern(text: str, flags: int) -> re.Pattern[str]:
        body = r"\s+".join(re.escape(part) for part in text.split())
        return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", flags)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        company_aliases: Mapping[str, Iterable[str]] | None = None,
        ticker_aliases: Mapping[str, str] | None = None,
        alias_exclusions: Iterable[str] = (),
    ) -> "SecuritiesMaster":
        """Load ``{symbol, name, sector}`` entries from a universe file."""
        with open(path, encoding="utf-8") as handle:
            entries = yaml.safe_load(handle)["securities"]
        securities = [
            SecurityInfo(str(e["symbol"]), str(e.get("name") or e["symbol"]), e.get("sector"))
            for e in entries
        ]
        return cls(securities, company_aliases, ticker_aliases, alias_exclusions)

    @classmethod
    def from_config(cls, config) -> "SecuritiesMaster":
        """The master described by an ``LLMConfig`` (universe file, aliases, exclusions)."""
        return cls.from_yaml(
            config.universe_file,
            config.company_aliases,
            config.ticker_aliases,
            config.alias_exclusions,
        )

    # ------------------------------------------------------------------- lookups

    @property
    def symbols(self) -> list[str]:
        return list(self._securities)

    def __contains__(self, symbol: object) -> bool:
        return isinstance(symbol, str) and symbol.upper() in self._securities

    def get(self, symbol: str) -> SecurityInfo | None:
        return self._securities.get(symbol.upper())

    def sector(self, symbol: str) -> str | None:
        security = self.get(symbol)
        return security.sector if security else None

    def resolve_ticker(self, ticker: str) -> str | None:
        """Map a source ticker (``SBIN.NS``, ``sbin``, an ADR alias) to a known symbol."""
        ticker = ticker.strip().upper()
        if ticker in self._ticker_aliases:
            return self._ticker_aliases[ticker]
        for suffix in EXCHANGE_SUFFIXES:
            if ticker.endswith(suffix):
                ticker = ticker[: -len(suffix)]
                break
        return ticker if ticker in self._securities else None

    def resolve_tickers(self, tickers: Iterable[str]) -> list[str]:
        resolved = (self.resolve_ticker(t) for t in tickers)
        return list(dict.fromkeys(s for s in resolved if s))

    def referenced(self, article: NewsArticle) -> dict[str, EntityMatch]:
        """Known symbols the article references, with the strongest evidence for each."""
        found: dict[str, EntityMatch] = {}
        content = article.content
        for exclusion in self._exclusions:
            content = exclusion.sub(" ", content)
        for symbol, alias, pattern in self._patterns:
            if symbol not in found and pattern.search(content):
                found[symbol] = EntityMatch(symbol, "alias", alias)
        for ticker in article.symbols:
            symbol = self.resolve_ticker(ticker)
            if symbol and symbol not in found:
                found[symbol] = EntityMatch(symbol, "hint", ticker)
        return found
