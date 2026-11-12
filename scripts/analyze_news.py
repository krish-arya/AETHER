"""Fetch news, extract validated financial events with the configured LLM, aggregate.

    uv run python scripts/analyze_news.py                          # Yahoo, whole universe
    uv run python scripts/analyze_news.py --symbols INFY WIPRO --max-articles 20
    uv run python scripts/analyze_news.py --jsonl data/licensed/news.jsonl

fetch -> de-duplicate -> analyze -> validate -> aggregate. Writes
artifacts/news/<run_id>/ (see docs/llm-engine.md). The provider comes from
configs/llm.yaml, overridden by AETHER_LLM_BASE_URL / AETHER_LLM_MODEL /
AETHER_LLM_API_KEY from the environment or .env.
"""

import argparse
import logging
import sys
from pathlib import Path

from ml.llm.config import LLMConfig, LLMEnvironment
from ml.llm.ingestion.base import NewsSourceError
from ml.llm.ingestion.jsonl import JsonlNewsSource
from ml.llm.ingestion.yahoo_news import YahooNewsSource
from ml.llm.pipeline import run_news_pipeline
from ml.llm.providers import LLMError, OpenAICompatibleClient
from ml.llm.securities import SecuritiesMaster


def _progress(index: int, total: int, analysis) -> None:
    outcome = [f"{len(analysis.events)} event(s)"] if analysis.events else []
    outcome += [r.reason.value for r in analysis.rejections]
    seconds = (
        f"{analysis.latency_seconds:5.1f}s" if analysis.latency_seconds is not None else "    --"
    )
    print(f"  [{index:>3}/{total}] {seconds}  {', '.join(outcome) or 'no event':<32} "
          f"{analysis.article.headline[:70]}")  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/llm.yaml"))
    parser.add_argument("--symbols", nargs="*", default=None, help="Default: whole universe")
    parser.add_argument("--jsonl", type=Path, default=None, help="Read articles from a file")
    parser.add_argument("--max-articles", type=int, default=None, help="Most recent N only")
    parser.add_argument("--refresh", action="store_true", help="Ignore today's news cache")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    environment = LLMEnvironment()
    config = environment.apply(LLMConfig.from_yaml(args.config))
    master = SecuritiesMaster.from_config(config)
    symbols = [s.upper() for s in args.symbols] if args.symbols else master.symbols
    unknown = [s for s in symbols if s not in master]
    if unknown:
        print(f"FAILED: not in {config.universe_file}: {unknown}", file=sys.stderr)
        return 2

    if args.jsonl:
        source = JsonlNewsSource(args.jsonl)
    else:
        yahoo = config.yahoo_news
        source = YahooNewsSource(
            master,
            yahoo.cache_dir,
            exchange_suffix=yahoo.exchange_suffix,
            news_count=yahoo.news_count,
            query_company_name=yahoo.query_company_name,
            refresh=args.refresh or yahoo.refresh,
            pause_seconds=yahoo.pause_seconds,
            max_retries=yahoo.max_retries,
        )
    client = OpenAICompatibleClient.from_config(config.provider, environment.key)
    print(f"provider : {client.provider}  model: {client.model}  ({config.provider.base_url})")
    print(f"source   : {source.info.name}  (body text: {source.info.provides_body})")

    try:
        run = run_news_pipeline(
            config, client, source, master, symbols,
            output_root=args.output_root, max_articles=args.max_articles, progress=_progress,
        )  # fmt: skip
    except (LLMError, NewsSourceError) as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1

    counts = run.manifest["counts"]
    print(f"\nrun_id   : {run.run_id}")
    print(f"path     : {run.path}")
    print(f"articles : {counts['articles_fetched']} fetched, {counts['articles_unique']} unique "
          f"({counts['duplicates_merged']} duplicates merged), "
          f"{counts['articles_analyzed']} analyzed")  # fmt: skip
    print(f"llm      : {counts['llm_calls']} calls, {counts['llm_calls_schema_valid']} "
          f"schema-valid, {run.manifest['llm_seconds_total']}s")  # fmt: skip
    print(f"events   : {counts['events']} accepted ({counts['events_unknown']} as 'unknown'), "
          f"{counts['rejections']} rejected")  # fmt: skip
    for reason, count in run.manifest["rejections_by_reason"].items():
        print(f"           rejected  {reason:<28}{count:>4}")
    for flag, count in run.manifest["flags_by_flag"].items():
        print(f"           flagged   {flag:<28}{count:>4}")
    for analyzed in run.events:
        event = analyzed.event
        print(f"  {event.symbol:<11}{event.event_type.value:<22}{event.sentiment.value:<10}"
              f"{event.sentiment_score:+.2f}  conf {event.confidence:.2f}")  # fmt: skip
    if not source.info.provides_body:
        print("\nNote: this source provides headlines only; events are low-detail by design.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
