"""Build a historical news feature table for the research experiments (E3, E5, E6).

    uv run python scripts/build_news_history.py fetch                  # free: headlines
    uv run python scripts/build_news_history.py estimate --sample 40   # LLM cost probe
    uv run python scripts/build_news_history.py analyze --workers 6    # all years
    uv run python scripts/build_news_history.py features --latest      # CSV for E3/E5

Stages
    fetch     Google News RSS headlines per company and month (cached under
              data/cache/google_news); prints how many there are and how many name a
              known company (only those reach the LLM).
    estimate  Analyses a fixed random sample with the configured LLM and projects the
              token use of the whole archive from the measured average. Writes nothing.
    analyze   Runs the validated news pipeline year by year; each year becomes one run
              under artifacts/news_history/ (a year that already has a run is skipped,
              so an interrupted build resumes).
    features  Point-in-time news features on the dataset's trading sessions from every
              run under artifacts/news_history/ -> artifacts/news_history/features-*.csv,
              the file ``run_experiments.py --news-features`` takes.

Headlines are moved to the end of their publication day (see
ml/llm/ingestion/google_news.py), so no headline reaches a decision on the day it
appeared. A model trained after the news may still know how a story ended; see
docs/llm-engine.md before reading anything into results built on these features.
"""

import argparse
import hashlib
import json
import logging
import random
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from ml.llm.config import LLMConfig, LLMEnvironment
from ml.llm.extraction.analyzer import NewsAnalyzer
from ml.llm.ingestion.base import NewsSource, NewsSourceError, NewsSourceInfo
from ml.llm.ingestion.dedup import consolidate
from ml.llm.ingestion.google_news import GoogleNewsSource
from ml.llm.pipeline import analyze_articles, run_news_pipeline
from ml.llm.providers import LLMError, OpenAICompatibleClient
from ml.llm.runs import list_runs
from ml.llm.schema import NewsArticle, now_utc
from ml.llm.securities import SecuritiesMaster

HISTORY_ROOT = Path("artifacts/news_history")
CACHE_DIR = Path("data/cache/google_news")


class FixedNewsSource(NewsSource):
    """Articles fetched earlier, handed to the pipeline as one source."""

    def __init__(self, articles: Sequence[NewsArticle], info: NewsSourceInfo, log: list[dict]):
        self._articles, self._info, self._log = list(articles), info, log

    @property
    def info(self) -> NewsSourceInfo:
        return self._info

    def fetch(self, symbols: Sequence[str]) -> list[NewsArticle]:
        return list(self._articles)

    def fetch_log(self) -> list[dict]:
        return list(self._log)


def _setup(config_path: Path):
    environment = LLMEnvironment()
    config = environment.apply(LLMConfig.from_yaml(config_path))
    return environment, config, SecuritiesMaster.from_config(config)


def _archive(master: SecuritiesMaster, start: date, end: date, refresh: bool):
    source = GoogleNewsSource(master, CACHE_DIR, start, end, refresh=refresh)
    return source, source.fetch(master.symbols)


def _named(articles: Sequence[NewsArticle], master: SecuritiesMaster) -> list[NewsArticle]:
    return [a for a in articles if master.referenced(a)]


def _year_runs() -> dict[int, Path]:
    runs: dict[int, Path] = {}
    for path in list_runs(HISTORY_ROOT):
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        year = manifest.get("history_year")
        if year is not None:
            runs[int(year)] = path
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["fetch", "estimate", "analyze", "features"])
    parser.add_argument("--config", type=Path, default=Path("configs/llm.yaml"))
    parser.add_argument("--start", type=date.fromisoformat, default=date(2017, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--refresh", action="store_true", help="Download again, ignore cache")
    parser.add_argument("--sample", type=int, default=40, help="estimate: articles to probe")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent LLM calls")
    parser.add_argument("--years", type=int, nargs="*", default=None, help="analyze: only these")
    parser.add_argument("--dataset", type=Path, default=None, help="features: dataset dir")
    parser.add_argument("--latest", action="store_true", help="features: newest dataset")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    environment, config, master = _setup(args.config)

    if args.stage == "features":
        return _features(args, config, master)

    try:
        source, articles = _archive(master, args.start, args.end, args.refresh)
    except NewsSourceError as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1
    unique, _ = consolidate(articles, config.dedup)
    named = _named(unique, master)
    print(f"archive  : {args.start} .. {args.end}, {len(master.symbols)} companies")
    print(f"headlines: {len(articles)} fetched, {len(unique)} after de-duplication, "
          f"{len(named)} name a known company (only these reach the LLM)")  # fmt: skip
    if args.stage == "fetch":
        by_year: dict[int, int] = {}
        for article in named:
            by_year[article.published_at.year] = by_year.get(article.published_at.year, 0) + 1
        print("per year :", dict(sorted(by_year.items())))
        return 0

    client = OpenAICompatibleClient.from_config(config.provider, environment.key)
    print(f"provider : {client.provider}  model: {client.model}")

    if args.stage == "estimate":
        sample = random.Random(0).sample(named, min(args.sample, len(named)))
        analyzer = NewsAnalyzer(client, master, config.extraction)
        try:
            results = analyze_articles(analyzer, sample, now=now_utc(), workers=args.workers)
        except LLMError as error:
            print(f"FAILED: {error}", file=sys.stderr)
            return 1
        calls = sum(1 for r in results if r.llm_called) or 1
        prompt = client.usage["prompt_tokens"] / calls
        completion = client.usage["completion_tokens"] / calls
        latency = [r.latency_seconds for r in results if r.latency_seconds is not None]
        print(f"sample   : {len(sample)} articles, {calls} LLM calls, "
              f"{sum(len(r.events) for r in results)} events")  # fmt: skip
        print(f"per call : {prompt:,.0f} prompt + {completion:,.0f} completion tokens, "
              f"{sum(latency) / max(len(latency), 1):.1f}s")  # fmt: skip
        print(f"projected: {len(named):,} calls -> {prompt * len(named) / 1e6:,.1f}M prompt + "
              f"{completion * len(named) / 1e6:,.1f}M completion tokens")  # fmt: skip
        return 0

    done = _year_runs()
    years = args.years or sorted({a.published_at.year for a in unique})
    for year in years:
        if year in done:
            print(f"{year}: already analysed ({done[year].name}), skipped")
            continue
        chunk = [a for a in unique if a.published_at.year == year]
        log = [e for e in source.fetch_log() if e["month"].startswith(str(year))]
        fixed = FixedNewsSource(chunk, source.info, log)
        print(f"{year}: {len(chunk)} headlines ...", flush=True)
        try:
            run = run_news_pipeline(
                config, client, fixed, master, master.symbols, output_root=HISTORY_ROOT,
                workers=args.workers,
            )  # fmt: skip
        except (LLMError, NewsSourceError) as error:
            print(f"FAILED in {year}: {error}", file=sys.stderr)
            return 1
        manifest_path = run.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["history_year"] = year
        manifest["llm_usage_tokens"] = dict(client.usage)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        counts = run.manifest["counts"]
        print(f"{year}: {counts['llm_calls']} calls, {counts['events']} events, "
              f"{counts['rejections']} rejected -> {run.path}  "
              f"(tokens so far {client.usage})", flush=True)  # fmt: skip
    return 0


def _features(args, config, master) -> int:
    from ml.backtesting.data import MarketData
    from ml.llm.sentiment.aggregation import news_feature_table

    dataset = args.dataset
    if args.latest:
        datasets = sorted(Path("artifacts/datasets").glob("*/manifest.json"),
                          key=lambda p: p.stat().st_mtime)  # fmt: skip
        dataset = datasets[-1].parent if datasets else None
    if dataset is None:
        print("FAILED: pass --dataset or --latest", file=sys.stderr)
        return 2
    runs = list_runs(HISTORY_ROOT)
    if not runs:
        print(f"FAILED: no news runs under {HISTORY_ROOT}; run 'analyze' first", file=sys.stderr)
        return 1
    market = MarketData.from_snapshot(dataset)
    table = news_feature_table(
        market.symbols, market.sessions[0], market.sessions[-1], news_root=str(HISTORY_ROOT),
        config=config.aggregation, sessions=market.sessions,
    )  # fmt: skip
    key = hashlib.sha256("|".join(sorted(p.name for p in runs)).encode()).hexdigest()[:10]
    out = HISTORY_ROOT / f"features-{Path(dataset).name}-{key}.csv"
    table.to_csv(out, index=False)
    covered = table[table["news_available"] > 0]
    print(f"features : {out}")
    print(f"rows     : {len(table):,}, with news {len(covered):,} "
          f"({len(covered) / max(len(table), 1):.1%}); runs used: {len(runs)}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
