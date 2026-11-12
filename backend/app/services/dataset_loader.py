"""Loads a versioned dataset snapshot (see ``ml.data.pipeline``) into the database.

* The snapshot's file hashes are verified against its manifest first; a dataset that
  was edited after it was built is refused.
* Loading is idempotent: rows are upserted on their natural keys, so re-running a
  load (or retrying after a failure) never duplicates data.
* The whole load is one transaction -- the database never holds half a dataset.

When two datasets contain the same (symbol, trade_date), the most recently loaded one
wins in the database; both snapshots remain intact on disk.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    CorporateAction,
    DatasetVersion,
    MarketData,
    MarketRegime,
    Security,
    TechnicalFeature,
)
from ml.data.versioning import MANIFEST_NAME, read_frame, verify_dataset

MAX_BIND_PARAMETERS = 30_000  # below both asyncpg's and SQLite's limits


@dataclass
class DatasetLoadSummary:
    dataset_id: str
    is_synthetic: bool
    securities: int
    bars: int
    corporate_actions: int
    feature_rows: int
    regime_rows: int = 0


def _records(frame: pd.DataFrame, integer_columns: tuple[str, ...] = ()) -> list[dict]:
    """DataFrame -> list of dicts of plain Python values, with missing values as None."""
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].dt.date
    for column in integer_columns:
        out[column] = out[column].astype("Int64")
    out = out.astype(object)
    return out.where(out.notna(), None).to_dict("records")


async def _upsert(
    session: AsyncSession, model, rows: list[dict], conflict_columns: list[str]
) -> None:
    if not rows:
        return
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        insert = postgresql_insert
    elif dialect == "sqlite":
        insert = sqlite_insert
    else:  # pragma: no cover - only PostgreSQL and SQLite are supported
        raise RuntimeError(f"Unsupported database dialect: {dialect}")

    columns = list(rows[0])
    chunk_size = max(1, MAX_BIND_PARAMETERS // len(columns))
    for offset in range(0, len(rows), chunk_size):
        statement = insert(model).values(rows[offset : offset + chunk_size])
        statement = statement.on_conflict_do_update(
            index_elements=conflict_columns,
            set_={c: statement.excluded[c] for c in columns if c not in conflict_columns},
        )
        await session.execute(statement)


async def load_dataset(session: AsyncSession, dataset_dir: Path) -> DatasetLoadSummary:
    dataset_dir = Path(dataset_dir)
    manifest = verify_dataset(dataset_dir)  # raises if any file was modified
    dataset_id: str = manifest["dataset_id"]
    provider = manifest["provider"]
    is_synthetic: bool = provider["is_synthetic"]
    source: str = provider["name"]
    now = datetime.now(UTC)

    securities = read_frame(dataset_dir / "securities.csv", ("listing_date", "delisting_date"))
    bars = read_frame(dataset_dir / "bars.csv", ("trade_date",))
    actions = read_frame(dataset_dir / "corporate_actions.csv", ("ex_date", "announced_date"))
    features = read_frame(dataset_dir / "features.csv", ("trade_date",))
    provenance = {"is_synthetic": is_synthetic, "source": source, "dataset_id": dataset_id}

    await _upsert(
        session,
        DatasetVersion,
        [
            {
                "dataset_id": dataset_id,
                "name": manifest["name"],
                "provider": source,
                "is_synthetic": is_synthetic,
                "license_note": provider["license_note"],
                "content_hash": manifest["content_hash"],
                "config_hash": manifest["config_hash"],
                "feature_set_version": manifest["feature_set_version"],
                "start_date": date.fromisoformat(manifest["date_range"]["first_bar"]),
                "end_date": date.fromisoformat(manifest["date_range"]["last_bar"]),
                "as_of": date.fromisoformat(manifest["date_range"]["as_of"]),
                "symbol_count": manifest["counts"]["securities"],
                "bar_count": manifest["counts"]["bars"],
                "quality_summary": manifest["quality"],
                "manifest_path": (dataset_dir / MANIFEST_NAME).as_posix(),
                "created_at": datetime.fromisoformat(manifest["created_at"]),
                "loaded_at": now,
            }
        ],
        ["dataset_id"],
    )

    security_rows = [{**row, **provenance} for row in _records(securities)]
    await _upsert(session, Security, security_rows, ["symbol"])

    bar_frame = bars.rename(columns={"adj_close": "adjusted_close"})[
        [
            "symbol", "trade_date", "open", "high", "low", "close", "adjusted_close",
            "adj_factor", "split_factor", "volume", "turnover", "delivery_volume", "market_cap",
        ]
    ]  # fmt: skip
    bar_rows = [
        {**row, **provenance, "ingested_at": now}
        for row in _records(bar_frame, integer_columns=("volume", "delivery_volume"))
    ]
    await _upsert(session, MarketData, bar_rows, ["symbol", "trade_date"])

    action_rows = [{**row, **provenance} for row in _records(actions)]
    await _upsert(session, CorporateAction, action_rows, ["symbol", "ex_date", "action_type"])

    feature_names = manifest["feature_columns"]
    feature_version = manifest["feature_set_version"]
    feature_values = _records(features[feature_names])
    feature_rows = [
        {
            "symbol": symbol,
            "trade_date": trade_date.date(),
            "feature_set_version": feature_version,
            "features": values,
            "dataset_id": dataset_id,
        }
        for symbol, trade_date, values in zip(
            features["symbol"], features["trade_date"], feature_values, strict=True
        )
    ]
    await _upsert(
        session, TechnicalFeature, feature_rows, ["symbol", "trade_date", "feature_set_version"]
    )

    regime_rows: list[dict] = []
    if (dataset_dir / "regimes.csv").is_file():  # absent in snapshots built before Phase 2
        regimes = read_frame(dataset_dir / "regimes.csv", ("trade_date",))
        regime_rows = [{**row, "dataset_id": dataset_id} for row in _records(regimes)]
        await _upsert(session, MarketRegime, regime_rows, ["dataset_id", "trade_date"])

    await session.commit()
    return DatasetLoadSummary(
        dataset_id=dataset_id,
        is_synthetic=is_synthetic,
        securities=len(security_rows),
        bars=len(bar_rows),
        corporate_actions=len(action_rows),
        feature_rows=len(feature_rows),
        regime_rows=len(regime_rows),
    )
