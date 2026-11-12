"""Load a built dataset snapshot into the database (idempotent).

    uv run python scripts/load_dataset.py artifacts/datasets/<dataset_id>
    uv run python scripts/load_dataset.py --latest

The target database is AETHER_DATABASE_URL; run migrations first:

    uv run alembic -c backend/alembic.ini upgrade head
"""

import argparse
import asyncio
import sys
from pathlib import Path

from app.core.config import get_settings
from app.database.session import dispose_engine, get_sessionmaker
from app.services.dataset_loader import load_dataset
from ml.data.versioning import MANIFEST_NAME


def _latest(root: Path) -> Path:
    candidates = (
        [p for p in root.iterdir() if (p / MANIFEST_NAME).is_file()] if root.is_dir() else []
    )
    if not candidates:
        raise SystemExit(f"No datasets found in {root}. Run scripts/build_dataset.py first.")
    return max(candidates, key=lambda p: (p / MANIFEST_NAME).stat().st_mtime)


async def _run(dataset_dir: Path) -> int:
    try:
        async with get_sessionmaker()() as session:
            summary = await load_dataset(session, dataset_dir)
    finally:
        await dispose_engine()
    if summary.is_synthetic:
        print("*** SYNTHETIC DEVELOPMENT DATA -- not real market data ***")
    print(f"loaded dataset      : {summary.dataset_id}")
    print(f"database            : {get_settings().database_backend}")
    print(f"securities          : {summary.securities}")
    print(f"bars                : {summary.bars}")
    print(f"corporate actions   : {summary.corporate_actions}")
    print(f"feature rows        : {summary.feature_rows}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset_dir", type=Path, nargs="?")
    parser.add_argument("--latest", action="store_true", help="Load the newest dataset")
    parser.add_argument("--root", type=Path, default=Path("artifacts/datasets"))
    args = parser.parse_args()

    if args.latest == (args.dataset_dir is not None):
        parser.error("pass exactly one of: a dataset directory, or --latest")
    dataset_dir = _latest(args.root) if args.latest else args.dataset_dir
    try:
        return asyncio.run(_run(dataset_dir))
    except (FileNotFoundError, ValueError) as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
