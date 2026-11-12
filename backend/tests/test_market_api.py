"""Integration: dataset snapshot -> loader -> database -> market API.

Uses the labelled synthetic development provider and a migrated SQLite database, so
values returned by the API are compared against the snapshot they were loaded from.
"""

import math
from datetime import date

import pandas as pd
import pytest
from sqlalchemy import func, select

from app.models import CorporateAction, MarketData, Security, TechnicalFeature
from app.services.dataset_loader import load_dataset
from ml.data.config import DataConfig
from ml.data.pipeline import build_dataset


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    cfg = DataConfig.model_validate(
        {
            "dataset": {
                "name": "api-test",
                "start_date": date(2020, 1, 1),
                "end_date": date(2021, 12, 31),
            },
            "provider": {"kind": "synthetic_dev", "synthetic_dev": {"seed": 5, "n_symbols": 4}},
        }
    )
    return build_dataset(cfg, output_root=tmp_path_factory.mktemp("datasets"))


@pytest.fixture
async def loaded(built, session_factory):
    async with session_factory() as session:
        return await load_dataset(session, built.path)


async def _count(session_factory, model) -> int:
    async with session_factory() as session:
        return await session.scalar(select(func.count()).select_from(model))


# ------------------------------------------------------------------------ loader


async def test_loader_writes_every_row(built, loaded, session_factory):
    assert loaded.dataset_id == built.dataset_id
    assert loaded.is_synthetic is True
    assert await _count(session_factory, Security) == 4
    assert await _count(session_factory, MarketData) == len(built.bars)
    assert await _count(session_factory, TechnicalFeature) == len(built.features)
    assert (
        await _count(session_factory, CorporateAction)
        == built.manifest["counts"]["corporate_actions"]
    )


async def test_loader_is_idempotent(built, loaded, session_factory):
    async with session_factory() as session:
        again = await load_dataset(session, built.path)
    assert again == loaded
    assert await _count(session_factory, MarketData) == len(built.bars)
    assert await _count(session_factory, CorporateAction) == loaded.corporate_actions


async def test_loader_marks_every_row_as_synthetic(loaded, session_factory):
    async with session_factory() as session:
        real_rows = await session.scalar(
            select(func.count()).select_from(MarketData).where(MarketData.is_synthetic.is_(False))
        )
    assert real_rows == 0


async def test_loader_refuses_a_modified_dataset(built, session_factory, tmp_path):
    import shutil

    copy = tmp_path / built.dataset_id
    shutil.copytree(built.path, copy)
    bars = copy / "bars.csv"
    bars.write_text(bars.read_text().replace("SYNTH01", "SYNTH77", 1))
    async with session_factory() as session:
        with pytest.raises(ValueError, match="modified after it was built"):
            await load_dataset(session, copy)
    assert await _count(session_factory, MarketData) == 0  # nothing was written


# --------------------------------------------------------------------------- API


async def test_datasets_endpoint_reports_provenance(built, loaded, db_client):
    body = (await db_client.get("/api/v1/market/datasets")).json()
    assert len(body) == 1
    dataset = body[0]
    assert dataset["dataset_id"] == built.dataset_id
    assert dataset["is_synthetic"] is True
    assert dataset["provider"] == "synthetic_dev"
    assert dataset["content_hash"] == built.manifest["content_hash"]
    assert dataset["bar_count"] == len(built.bars)


async def test_stocks_endpoint_lists_universe(loaded, db_client):
    body = (await db_client.get("/api/v1/market/stocks")).json()
    assert [s["symbol"] for s in body] == ["SYNTH01", "SYNTH02", "SYNTH03", "SYNTH04"]
    assert all(s["is_synthetic"] for s in body)


async def test_stocks_endpoint_is_point_in_time(loaded, db_client):
    early = (await db_client.get("/api/v1/market/stocks", params={"as_of": "2020-01-15"})).json()
    late = (await db_client.get("/api/v1/market/stocks", params={"as_of": "2021-12-15"})).json()
    # SYNTH03 lists mid-history, SYNTH04 delists before the end.
    assert [s["symbol"] for s in early] == ["SYNTH01", "SYNTH02", "SYNTH04"]
    assert [s["symbol"] for s in late] == ["SYNTH01", "SYNTH02", "SYNTH03"]


async def test_stock_detail_matches_snapshot(built, loaded, db_client):
    response = await db_client.get("/api/v1/market/stocks/synth01")  # case-insensitive
    assert response.status_code == 200
    body = response.json()
    source = built.bars[built.bars["symbol"] == "SYNTH01"]
    last = source.iloc[-1]
    assert body["bar_count"] == len(source)
    assert body["first_trade_date"] == source["trade_date"].min().strftime("%Y-%m-%d")
    assert body["latest_bar"]["trade_date"] == last["trade_date"].strftime("%Y-%m-%d")
    assert body["latest_bar"]["close"] == last["close"]
    assert body["latest_bar"]["volume"] == int(last["volume"])


async def test_history_returns_exact_snapshot_values(built, loaded, db_client):
    response = await db_client.get(
        "/api/v1/market/stocks/SYNTH02/history",
        params={"start": "2020-03-01", "end": "2020-03-31", "limit": 5000},
    )
    assert response.status_code == 200
    body = response.json()
    source = built.bars[
        (built.bars["symbol"] == "SYNTH02")
        & (built.bars["trade_date"] >= "2020-03-01")
        & (built.bars["trade_date"] <= "2020-03-31")
    ]
    assert body["is_synthetic"] is True and body["source"] == "synthetic_dev"
    assert body["count"] == len(source) > 15
    dates = [bar["trade_date"] for bar in body["bars"]]
    assert dates == sorted(dates)
    assert [bar["close"] for bar in body["bars"]] == source["close"].tolist()
    assert [bar["adjusted_close"] for bar in body["bars"]] == source["adj_close"].tolist()
    assert [bar["volume"] for bar in body["bars"]] == source["volume"].tolist()


async def test_history_limit_returns_most_recent_bars(built, loaded, db_client):
    body = (
        await db_client.get("/api/v1/market/stocks/SYNTH01/history", params={"limit": 3})
    ).json()
    source = built.bars[built.bars["symbol"] == "SYNTH01"]
    assert [b["trade_date"] for b in body["bars"]] == [
        d.strftime("%Y-%m-%d") for d in source["trade_date"].iloc[-3:]
    ]


async def test_indicators_match_snapshot_and_keep_undefined_as_null(built, loaded, db_client):
    response = await db_client.get("/api/v1/market/indicators/SYNTH01", params={"limit": 5000})
    assert response.status_code == 200
    body = response.json()
    source = built.features[built.features["symbol"] == "SYNTH01"].reset_index(drop=True)

    assert body["feature_set_version"] == built.manifest["feature_set_version"]
    assert body["dataset_id"] == built.dataset_id
    assert body["feature_names"] == sorted(built.manifest["feature_columns"])
    assert body["count"] == len(source)

    first, last = body["rows"][0], body["rows"][-1]
    assert first["features"]["sma_50_ratio"] is None  # warm-up: undefined, not filled
    assert first["features"]["ret_1d"] is None
    for name, value in last["features"].items():
        expected = source[name].iloc[-1]
        if pd.isna(expected):
            assert value is None
        else:
            assert math.isclose(value, expected, rel_tol=0, abs_tol=0)


async def test_regime_endpoint_matches_snapshot(built, loaded, db_client):
    assert loaded.regime_rows == len(built.regimes) > 400
    response = await db_client.get("/api/v1/market/regime", params={"limit": 5000})
    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == built.dataset_id
    assert body["is_synthetic"] is True
    assert len(body["history"]) == len(built.regimes)

    source = built.regimes.iloc[-1]
    current = body["current"]
    assert current == body["history"][-1]
    assert current["timestamp"] == source["trade_date"].strftime("%Y-%m-%d")
    assert current["regime"] == source["regime"]
    assert current["volatility_level"] == source["volatility_level"]
    assert set(current) >= {
        "timestamp",
        "regime",
        "confidence",
        "volatility_level",
        "trend_strength",
    }
    # Warm-up sessions carry no label rather than a guessed one.
    assert body["history"][0]["regime"] is None

    recent = (await db_client.get("/api/v1/market/regime", params={"limit": 3})).json()
    assert [p["timestamp"] for p in recent["history"]] == [
        d.strftime("%Y-%m-%d") for d in built.regimes["trade_date"].iloc[-3:]
    ]


async def test_regime_endpoint_404s(loaded, db_client):
    missing = await db_client.get("/api/v1/market/regime", params={"dataset_id": "nope"})
    assert missing.status_code == 404
    empty = await db_client.get("/api/v1/market/regime", params={"end": "1990-01-01"})
    assert empty.status_code == 404


async def test_stocks_can_be_filtered_by_dataset(built, loaded, db_client):
    mine = await db_client.get("/api/v1/market/stocks", params={"dataset_id": built.dataset_id})
    assert len(mine.json()) == 4
    other = await db_client.get("/api/v1/market/stocks", params={"dataset_id": "nope"})
    assert other.json() == []


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/market/stocks/RELIANCE",
        "/api/v1/market/stocks/RELIANCE/history",
        "/api/v1/market/indicators/RELIANCE",
    ],
)
async def test_unknown_symbol_is_404_not_invented_data(loaded, db_client, path):
    assert (await db_client.get(path)).status_code == 404


async def test_indicators_unknown_feature_set_is_404(loaded, db_client):
    response = await db_client.get(
        "/api/v1/market/indicators/SYNTH01", params={"feature_set_version": "deadbeef"}
    )
    assert response.status_code == 404


async def test_invalid_query_parameters_are_rejected(loaded, db_client):
    base = "/api/v1/market/stocks/SYNTH01/history"
    assert (
        await db_client.get(base, params={"start": "2021-01-01", "end": "2020-01-01"})
    ).status_code == 422
    assert (await db_client.get(base, params={"limit": 0})).status_code == 422
    assert (await db_client.get(base, params={"limit": 999999})).status_code == 422
    assert (await db_client.get(base, params={"start": "yesterday"})).status_code == 422


async def test_empty_database_returns_empty_lists(db_client):
    assert (await db_client.get("/api/v1/market/stocks")).json() == []
    assert (await db_client.get("/api/v1/market/datasets")).json() == []
