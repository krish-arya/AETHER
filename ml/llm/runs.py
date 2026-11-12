"""News run directories: ``artifacts/news/<run_id>/``.

    articles.jsonl   de-duplicated articles that entered the run (NewsArticle)
    events.jsonl     accepted events with flags and provenance (AnalyzedEvent)
    rejected.jsonl   every refused article / model output, with the reason (Rejection)
    features.csv     per-symbol daily news features (sentiment.aggregation)
    manifest.json    model, provider, prompt version, config and its hash, source,
                     counts, rejection counts by reason, SHA-256 of every file

A run is verified against its manifest hashes before it is read or loaded anywhere.
"""

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from ml.data.versioning import MANIFEST_NAME, file_sha256, write_frame, write_json
from ml.llm.schema import AnalyzedEvent, NewsArticle, Rejection

RUN_SCHEMA_VERSION = 1
ARTICLES, EVENTS, REJECTED, FEATURES = (
    "articles.jsonl",
    "events.jsonl",
    "rejected.jsonl",
    "features.csv",
)


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> str:
    """Write one JSON object per line (stable key order) and return the file's SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), sort_keys=True))
            handle.write("\n")
    return file_sha256(path)


def read_jsonl(path: Path, model: type[BaseModel]) -> list:
    with open(path, encoding="utf-8") as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]


def write_run(
    run_dir: Path,
    manifest: dict,
    articles: list[NewsArticle],
    events: list[AnalyzedEvent],
    rejections: list[Rejection],
    features: pd.DataFrame,
) -> dict:
    """Write the run files and the manifest (with file hashes). Returns the manifest."""
    run_dir.mkdir(parents=True, exist_ok=True)
    files = {
        ARTICLES: write_jsonl(run_dir / ARTICLES, articles),
        EVENTS: write_jsonl(run_dir / EVENTS, events),
        REJECTED: write_jsonl(run_dir / REJECTED, rejections),
        FEATURES: write_frame(features, run_dir / FEATURES),
    }
    manifest = {**manifest, "schema_version": RUN_SCHEMA_VERSION, "files": files}
    write_json(manifest, run_dir / MANIFEST_NAME)
    return manifest


def verify_run(run_dir: str | Path) -> dict:
    """Re-hash the run files against the manifest; raise ``ValueError`` on a mismatch."""
    run_dir = Path(run_dir)
    with open(run_dir / MANIFEST_NAME, encoding="utf-8") as handle:
        manifest = json.load(handle)
    for name, expected in manifest["files"].items():
        if file_sha256(run_dir / name) != expected:
            raise ValueError(f"News run {run_dir.name}: {name} was modified after it was written")
    return manifest


def load_run(run_dir: str | Path) -> tuple[dict, list[NewsArticle], list[AnalyzedEvent]]:
    run_dir = Path(run_dir)
    manifest = verify_run(run_dir)
    return (
        manifest,
        read_jsonl(run_dir / ARTICLES, NewsArticle),
        read_jsonl(run_dir / EVENTS, AnalyzedEvent),
    )


def list_runs(news_root: str | Path) -> list[Path]:
    """Run directories under ``news_root``, oldest first (by manifest ``created_at``)."""
    root = Path(news_root)
    if not root.is_dir():
        return []
    runs = []
    for path in root.iterdir():
        if (path / MANIFEST_NAME).is_file():
            with open(path / MANIFEST_NAME, encoding="utf-8") as handle:
                runs.append((json.load(handle).get("created_at", ""), path.name, path))
    return [path for _, _, path in sorted(runs)]


def load_all_events(news_root: str | Path) -> list[AnalyzedEvent]:
    """Events of every run under ``news_root``; a re-analysed event comes from the newest run."""
    latest: dict[str, AnalyzedEvent] = {}
    for run_dir in list_runs(news_root):
        _, _, events = load_run(run_dir)
        for analyzed in events:
            latest[analyzed.event.event_id] = analyzed
    return sorted(latest.values(), key=lambda a: (a.event.publication_timestamp, a.event.event_id))
