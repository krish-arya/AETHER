"""Load a news run into the database (idempotent).

    uv run python scripts/load_news.py artifacts/news/<run_id>
    uv run python scripts/load_news.py --latest

The target database is AETHER_DATABASE_URL; run migrations first:

    uv run alembic -c backend/alembic.ini upgrade head
"""

import argparse
import asyncio
import sys
from pathlib import Path

from app.core.config import get_settings
from app.database.session import dispose_engine, get_sessionmaker
from app.services.news_loader import load_news_run
from ml.llm.runs import list_runs


async def _run(run_dir: Path) -> int:
    try:
        async with get_sessionmaker()() as session:
            summary = await load_news_run(session, run_dir)
    finally:
        await dispose_engine()
    print(f"loaded news run : {summary.run_id}")
    print(f"database        : {get_settings().database_backend}")
    print(f"model           : {summary.llm_model} ({summary.llm_provider})")
    print(f"articles        : {summary.articles}")
    print(f"events          : {summary.events}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--latest", action="store_true", help="Load the newest news run")
    parser.add_argument("--root", type=Path, default=Path("artifacts/news"))
    args = parser.parse_args()

    if args.latest == (args.run_dir is not None):
        parser.error("pass exactly one of: a run directory, or --latest")
    if args.latest:
        runs = list_runs(args.root)
        if not runs:
            print(f"FAILED: no news runs in {args.root}. Run scripts/analyze_news.py first.",
                  file=sys.stderr)  # fmt: skip
            return 2
        run_dir = runs[-1]
    else:
        run_dir = args.run_dir
    try:
        return asyncio.run(_run(run_dir))
    except (FileNotFoundError, ValueError) as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
