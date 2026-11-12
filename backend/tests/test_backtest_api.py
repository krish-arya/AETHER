"""Backtest API and job lifecycle: submit -> queued -> executed -> results.

The job queue is replaced by a recorder; jobs are then executed directly through the
same service function the Celery task calls.
"""

from datetime import date

import pytest
from sqlalchemy import select

from app.api.v1.backtest import get_dispatcher
from app.core.config import Settings, get_settings
from app.models import BacktestRun
from app.services.backtests import execute_backtest_run
from app.services.dataset_loader import load_dataset
from ml.data.config import DataConfig
from ml.data.pipeline import build_dataset

STRATEGIES = [
    {"strategy": "equal_weight"},
    {"strategy": "momentum", "params": {"lookback": 60, "skip": 5, "top_k": 2}},
]
REQUEST = {
    "name": "api-test",
    "strategies": STRATEGIES,
    "benchmark": "equal_weight",
    "walk_forward": {"train_years": 1, "validation_years": 1, "test_years": 1},
}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    cfg = DataConfig.model_validate(
        {
            "dataset": {
                "name": "bt-api",
                "start_date": date(2018, 1, 1),
                "end_date": date(2021, 12, 31),
            },
            "provider": {"kind": "synthetic_dev", "synthetic_dev": {"seed": 9, "n_symbols": 5}},
        }
    )
    return build_dataset(cfg, output_root=tmp_path_factory.mktemp("datasets"))


@pytest.fixture
async def loaded(built, session_factory):
    async with session_factory() as session:
        return await load_dataset(session, built.path)


@pytest.fixture
def queue(app):
    """Stand-in for the Celery queue: records the run ids that were dispatched."""
    dispatched: list[str] = []
    app.dependency_overrides[get_dispatcher] = lambda: dispatched.append
    return dispatched


async def _submit(db_client, **overrides):
    return await db_client.post("/api/v1/backtest/run", json={**REQUEST, **overrides})


async def test_strategy_catalogue(db_client):
    body = (await db_client.get("/api/v1/backtest/strategies")).json()
    by_name = {item["strategy"]: item for item in body}
    assert set(by_name) == {
        "buy_and_hold", "equal_weight", "momentum", "moving_average", "mean_variance",
    }  # fmt: skip
    assert by_name["momentum"]["default_params"] == {
        "lookback": 126, "skip": 21, "top_k": 5, "absolute": False,
    }  # fmt: skip
    assert "Cross-sectional momentum" in by_name["momentum"]["description"]


async def test_submit_records_a_queued_job_and_dispatches_it(built, loaded, db_client, queue):
    response = await _submit(db_client)
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["dataset_id"] == built.dataset_id  # defaulted to the latest loaded dataset
    assert body["is_synthetic"] is True
    assert queue == [body["run_id"]]

    listed = (await db_client.get("/api/v1/backtest/runs")).json()
    assert [run["run_id"] for run in listed] == [body["run_id"]]

    detail = (await db_client.get(f"/api/v1/backtest/runs/{body['run_id']}")).json()
    assert detail["summary"] is None and detail["equity"] is None
    assert detail["config"]["rebalance"] == "monthly"  # inherited from the template
    assert detail["config"]["costs"]["statutory"][0]["stt_buy"] == 0.001
    assert [s["strategy"] for s in detail["config"]["strategies"]] == ["equal_weight", "momentum"]

    # Results are not available -- and not invented -- before the job has run.
    metrics = await db_client.get(f"/api/v1/backtest/runs/{body['run_id']}/metrics")
    assert metrics.status_code == 409


async def test_job_runs_to_completion_and_results_are_served(
    built, loaded, db_client, queue, session_factory, tmp_path
):
    run_id = (await _submit(db_client)).json()["run_id"]
    assert await execute_backtest_run(run_id, session_factory, tmp_path) == "succeeded"

    detail = (await db_client.get(f"/api/v1/backtest/runs/{run_id}")).json()
    assert detail["status"] == "succeeded"
    assert detail["error"] is None
    assert detail["started_at"] and detail["finished_at"]
    assert detail["result_id"].startswith("api-test-")
    summary = detail["summary"]
    assert summary["dataset"]["dataset_id"] == built.dataset_id
    assert any("SYNTHETIC" in caveat for caveat in summary["caveats"])
    assert len(summary["walk_forward"]["folds"]) == 2

    equity = detail["equity"]
    assert set(equity["series"]) == {"equal_weight", "momentum"}
    assert len(equity["dates"]) == summary["period"]["sessions"]
    assert all(len(values) == len(equity["dates"]) for values in equity["series"].values())
    assert equity["series"]["equal_weight"][0] == 1_000_000.0
    final = summary["strategies"]["equal_weight"]["metrics"]["final_equity"]
    assert equity["series"]["equal_weight"][-1] == pytest.approx(final)

    metrics = (await db_client.get(f"/api/v1/backtest/runs/{run_id}/metrics")).json()
    assert metrics["benchmark"] == "equal_weight"
    assert metrics["is_synthetic"] is True
    assert (
        set(metrics["full_period"]) == set(metrics["walk_forward"]) == {"equal_weight", "momentum"}
    )
    assert "alpha" in metrics["full_period"]["momentum"]
    assert "alpha" not in metrics["full_period"]["equal_weight"]

    rows = (await db_client.get("/api/v1/backtest/compare", params={"run_ids": [run_id]})).json()
    assert [(row["strategy"], row["is_benchmark"]) for row in rows] == [
        ("equal_weight", True),
        ("momentum", False),
    ]
    assert rows[0]["metrics"]["sharpe"] == metrics["full_period"]["equal_weight"]["sharpe"]

    # A duplicate delivery of the same job must not run it again.
    assert await execute_backtest_run(run_id, session_factory, tmp_path) == "succeeded"


async def test_failed_job_is_recorded_with_its_error(
    loaded, db_client, queue, session_factory, tmp_path
):
    costs = {
        "statutory": [
            {
                "effective_from": "2030-01-01",  # schedule starts after the data
                "stt_buy": 0.001, "stt_sell": 0.001, "exchange_txn": 0, "sebi_fee": 0,
                "stamp_duty_buy": 0, "stamp_duty_sell": 0, "gst": 0,
            }
        ]
    }  # fmt: skip
    run_id = (await _submit(db_client, costs=costs)).json()["run_id"]
    assert await execute_backtest_run(run_id, session_factory, tmp_path) == "failed"

    detail = (await db_client.get(f"/api/v1/backtest/runs/{run_id}")).json()
    assert detail["status"] == "failed"
    assert "No statutory cost rates configured" in detail["error"]
    assert detail["finished_at"] is not None
    assert detail["summary"] is None
    assert (await db_client.get(f"/api/v1/backtest/runs/{run_id}/metrics")).status_code == 409


@pytest.mark.parametrize(
    "overrides",
    [
        {"strategies": [{"strategy": "crystal_ball"}]},  # unknown strategy
        {"strategies": [{"strategy": "momentum", "params": {"top_k": 0}}], "benchmark": "momentum"},
        {
            "strategies": [{"strategy": "momentum", "params": {"lokback": 5}}],
            "benchmark": "momentum",
        },
        {"benchmark": "nifty50_tri"},  # benchmark is not one of the strategies
        {"start_date": "2021-01-01", "end_date": "2020-01-01"},
        {"initial_capital": -5},
        {"rebalance": "hourly"},
        {"leverage": 3},  # unknown field
    ],
)
async def test_invalid_requests_are_rejected_before_anything_is_queued(
    loaded, db_client, queue, session_factory, overrides
):
    response = await _submit(db_client, **overrides)
    assert response.status_code == 422
    assert queue == []
    async with session_factory() as session:
        assert (await session.scalars(select(BacktestRun))).all() == []


async def test_unknown_or_missing_dataset_is_404(loaded, db_client, queue):
    response = await _submit(db_client, dataset_id="no-such-dataset")
    assert response.status_code == 404
    assert queue == []


async def test_submit_without_any_loaded_dataset_is_404(db_client, queue):
    response = await _submit(db_client)
    assert response.status_code == 404
    assert "No dataset is loaded" in response.json()["detail"]


async def test_queue_outage_fails_the_run_and_returns_503(loaded, app, db_client):
    def broken(run_id: str) -> None:
        raise ConnectionError("redis is down")

    app.dependency_overrides[get_dispatcher] = lambda: broken
    response = await _submit(db_client)
    assert response.status_code == 503

    runs = (await db_client.get("/api/v1/backtest/runs")).json()
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"  # not left looking as if it were queued
    assert "Job queue unavailable" in runs[0]["error"]


async def test_active_run_limit(loaded, app, db_client, queue):
    app.dependency_overrides[get_settings] = lambda: Settings(max_active_backtests=1)
    assert (await _submit(db_client)).status_code == 202
    second = await _submit(db_client)
    assert second.status_code == 429
    assert len(queue) == 1


async def test_unknown_run_is_404(db_client):
    assert (await db_client.get("/api/v1/backtest/runs/nope")).status_code == 404
    assert (await db_client.get("/api/v1/backtest/runs/nope/metrics")).status_code == 404
    compare = await db_client.get("/api/v1/backtest/compare", params={"run_ids": ["nope"]})
    assert compare.status_code == 404


def test_backtest_task_is_registered_with_celery():
    import app.workers.tasks  # noqa: F401
    from app.workers.celery_app import celery_app

    assert "aether.backtest.run" in celery_app.tasks
    assert "app.workers.tasks" in celery_app.conf.include
