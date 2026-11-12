"""File-based model registry (PRD 16.1).

Every trained model gets a ``model_card.json`` next to its artifacts:

    <root>/<kind>/<model_id>/model_card.json

with the fields the PRD requires -- model id, version, training timestamp, dataset
version, hyper-parameters, evaluation metrics, artifact location, status -- plus the
seed, config hash, training window and software environment needed to reproduce it.
Kinds: ``rl``, ``gan``, ``regime``, ``llm``, ``features``.

``model_id`` is derived from the config hash, dataset content hash, training window
and seed, so retraining the same thing yields the same id (and overwrites the card)
while any change yields a new one.

Status
    candidate   trained, not yet judged
    validated   met its acceptance checks (e.g. GAN quality gates)
    rejected    failed them -- kept on record, not hidden

This replaces MLflow in the current setup (MLflow depends on pyarrow, which the
development machine's application-control policy blocks). The cards are plain JSON and
can be imported into MLflow later.
"""

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ml.data.versioning import git_commit

Kind = Literal["rl", "gan", "regime", "llm", "features"]
Status = Literal["candidate", "validated", "rejected"]
CARD_NAME = "model_card.json"
VERSION = 1


def model_id(kind: str, name: str, *parts: object) -> str:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:10]
    return f"{kind}-{name}-{digest}"


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    return value


def register(
    root: Path,
    *,
    kind: Kind,
    model_id: str,
    dataset_id: str | None,
    hyperparameters: dict,
    metrics: dict,
    status: Status = "candidate",
    seed: int | None = None,
    config_hash: str | None = None,
    training_window: dict | None = None,
    notes: str | None = None,
    is_synthetic_data: bool | None = None,
    lineage: dict | None = None,
) -> Path:
    """Write (or overwrite) a model card and return the model directory."""
    directory = Path(root) / kind / model_id
    directory.mkdir(parents=True, exist_ok=True)
    card = _clean(
        {
            "model_id": model_id,
            "kind": kind,
            "version": VERSION,
            "status": status,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "dataset_id": dataset_id,
            "trained_on_synthetic_data": is_synthetic_data,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "artifact_path": directory.as_posix(),
            "seed": seed,
            "config_hash": config_hash,
            "training_window": training_window,
            "notes": notes,
            # Other artifacts this model depends on (e.g. the scenario generator).
            "lineage": lineage or {},
            "environment": {
                "platform": f"{platform.system()}-{platform.machine()}",
                "python": platform.python_version(),
                "git_commit": git_commit(),
            },
        }
    )
    with open(directory / CARD_NAME, "w", encoding="utf-8") as handle:
        json.dump(card, handle, indent=2, sort_keys=True)
    return directory


def load_card(root: Path, kind: str, model_id: str) -> dict:
    with open(Path(root) / kind / model_id / CARD_NAME, encoding="utf-8") as handle:
        return json.load(handle)


def list_models(root: Path, kind: str | None = None) -> list[dict]:
    """All model cards under ``root``, newest first."""
    root = Path(root)
    kinds = (
        [kind] if kind else [p.name for p in root.iterdir() if p.is_dir()] if root.is_dir() else []
    )
    cards = []
    for name in kinds:
        for card_file in sorted((root / name).glob(f"*/{CARD_NAME}")):
            with open(card_file, encoding="utf-8") as handle:
                cards.append(json.load(handle))
    return sorted(cards, key=lambda card: card["trained_at"], reverse=True)
