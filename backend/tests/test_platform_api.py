"""RL / GAN / risk / portfolio / paper-trading / job endpoints.

A tiny RL deployment model and a tiny scenario generator are trained once per module
(through the same job functions the workers run) into a temporary artifacts folder;
each test then loads the dataset into its own migrated database.
"""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
import yaml

from app.api.v1.models_api import get_job_dispatcher
from app.core.config import Settings, get_settings
from app.models import DatasetVersion, Job
from app.services import jobs, platform, portfolio
from app.services.dataset_loader import load_dataset
from ml.data.config import DataConfig
from ml.data.pipeline import build_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
API = "/api/v1"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    cfg = DataConfig.model_validate(
        {
            "dataset": {
                "name": "platform-test",
                "start_date": date(2018, 1, 1),
                "end_date": date(2021, 12, 31),
            },
            "provider": {"kind": "synthetic_dev", "synthetic_dev": {"seed": 4, "n_symbols": 6}},
        }
    )
    return build_dataset(cfg, output_root=tmp_path_factory.mktemp("datasets"))


@pytest.fixture(scope="module")
def settings(tmp_path_factory) -> Settings:
    root = tmp_path_factory.mktemp("platform")
    experiments = yaml.safe_load((REPO_ROOT / "configs" / "experiments.yaml").read_text())
    experiments["rl"].update(
        {
            "features": ["ret_1d", "cum_ret_21d", "rsi_14", "hist_vol_20d"],
            "seeds": [0],
            "eval_every": 256,
            "torch_threads": 1,
            "ppo": {"total_timesteps": 512, "n_steps": 128, "batch_size": 64, "n_epochs": 1,
                    "net_arch": [16]},
        }
    )  # fmt: skip
    experiments["rl"]["env"]["episode_steps"] = 12
    experiments["rl"]["limits"]["max_weight"] = 0.4
    experiments["gan"] = {
        "sequence_length": 12, "volume_window": 20, "hidden_dim": 8, "num_layers": 1,
        "noise_dim": 4, "batch_size": 32, "embedding_iterations": 3, "supervised_iterations": 3,
        "joint_iterations": 3, "calibration_windows": 50, "min_regime_windows": 5,
        "evaluation_windows": 60, "torch_threads": 1,
    }  # fmt: skip
    (root / "experiments.yaml").write_text(yaml.safe_dump(experiments))
    return Settings(
        artifacts_dir=root / "artifacts",
        experiments_config=root / "experiments.yaml",
        backtest_config=REPO_ROOT / "configs" / "backtesting.yaml",
        max_active_jobs=2,
    )


def _dataset_row(built) -> DatasetVersion:
    return DatasetVersion(
        dataset_id=built.dataset_id,
        is_synthetic=True,
        manifest_path=(built.path / "manifest.json").as_posix(),
    )


@pytest.fixture(scope="module")
def trained(built, settings) -> dict:
    """Train the deployment RL model and a generator exactly as the worker jobs do."""
    platform.clear_caches()
    portfolio._states.clear()
    row = _dataset_row(built)
    rl = jobs._run("rl_train", {}, row, settings)
    gan = jobs._run("gan_train", {}, row, settings)
    return {"rl": rl, "gan": gan}


@pytest.fixture
async def api(app, db_client, session_factory, built, settings, trained):
    app.dependency_overrides[get_settings] = lambda: settings
    portfolio._states.clear()
    async with session_factory() as session:
        await load_dataset(session, built.path)
    return db_client


async def _initialize(api, **body):
    payload = {"strategy": {"kind": "baseline", "ref": "equal_weight"}, **body}
    return await api.post(f"{API}/paper/initialize", json=payload)


# ----------------------------------------------------------------------- models


async def test_registered_models_are_listed_with_registry_fields(api, trained, built):
    models = (await api.get(f"{API}/rl/models")).json()
    assert [m["model_id"] for m in models] == [trained["rl"]["model_id"]]
    card = models[0]
    for field in ("model_id", "version", "trained_at", "dataset_id", "artifact_path", "status",
                  "metrics", "training_window"):  # fmt: skip
        assert field in card
    assert card["dataset_id"] == built.dataset_id
    assert card["lineage"]["experiment"] == "deployment"
    assert "no out-of-sample" in card["notes"]

    detail = (await api.get(f"{API}/rl/models/{card['model_id']}")).json()
    assert detail["hyperparameters"]["algorithm"] == "ppo"
    assert detail["metrics"]["validation_curve"]
    assert (await api.get(f"{API}/rl/models/nope")).status_code == 404
    assert (await api.get(f"{API}/rl/models", params={"dataset_id": "other"})).json() == []


async def test_generator_listing_and_evaluation_report_its_real_verdict(api, trained):
    generators = (await api.get(f"{API}/gan/models")).json()
    assert len(generators) == 1
    generator = generators[0]
    # Three training iterations cannot produce usable scenarios, and the API says so.
    assert trained["gan"]["passes"] is False
    assert generator["status"] == "rejected"
    report = (await api.get(f"{API}/gan/evaluation/{generator['model_id']}")).json()
    assert report["passes"] is False
    assert {"discriminative", "memorization", "gates", "synthetic_vs_heldout_real"} <= set(report)
    assert (await api.get(f"{API}/gan/evaluation/nope")).status_code == 404


async def test_generated_scenarios_are_labelled_synthetic_and_stored(api, trained):
    response = await api.post(
        f"{API}/gan/generate", json={"regime": "market_stress", "scenarios": 20, "seed": 1}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["is_synthetic"] is True and "Not forecasts" in body["label"]
    assert body["generator_status"] == "rejected"  # carried along, not hidden
    assert body["scenarios"] == 20 and len(body["equal_weight_path"]["p50"]) == 12
    assert body["equal_weight_total_return"]["worst"] <= body["equal_weight_total_return"]["mean"]
    stored = (await api.get(f"{API}/gan/scenarios")).json()
    assert [s["scenario_set_id"] for s in stored] == [body["scenario_set_id"]]
    bad = await api.post(f"{API}/gan/generate", json={"regime": "moon"})
    assert bad.status_code == 422


# ------------------------------------------------------------------- allocation


async def test_allocation_is_risk_checked_explained_and_recorded(api, trained, built, settings):
    response = await api.post(f"{API}/rl/allocate", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == trained["rl"]["model_id"]
    assert body["dataset_id"] == built.dataset_id and body["is_synthetic"] is True
    assert body["is_recommendation_for_real_trading"] is False
    weights = body["allocation"]
    assert 0 < sum(weights.values()) <= 0.98 + 1e-9  # the 2% cash reserve
    assert max(weights.values()) <= 0.4 + 1e-9  # the configured asset cap
    assert body["evidence"]["risk_engine"]["rejected"] is False
    assert body["model"]["experiment"] == "deployment"
    assert any("not a causal explanation" in line for line in body["summary"]["narrative"])
    assert body["as_of"] == built.bars["trade_date"].max().strftime("%Y-%m-%d")
    assert body["executable_from"] is None  # no later session exists in the data yet

    # Same data and model -> the same allocation; both calls are stored with timestamps.
    again = (await api.post(f"{API}/rl/allocate", json={})).json()
    assert again["allocation"] == weights
    assert again["allocation_id"] != body["allocation_id"]
    assert (await api.post(f"{API}/rl/allocate", json={"model_id": "nope"})).status_code == 404


async def test_allocation_without_a_trained_model_explains_what_to_do(api, settings, app, tmp_path):
    empty = settings.model_copy(update={"artifacts_dir": tmp_path / "empty"})
    app.dependency_overrides[get_settings] = lambda: empty
    response = await api.post(f"{API}/rl/allocate", json={})
    assert response.status_code == 404
    assert "No RL model has been trained" in response.json()["detail"]


# ------------------------------------------------------------------------- risk


async def test_risk_validation_reports_violations_and_the_compliant_result(api):
    response = await api.post(
        f"{API}/risk/validate", json={"weights": {"SYNTH01": 0.8, "SYNTH02": 0.3}}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["compliant"] is False and body["rejected"] is False
    assert {v["limit"] for v in body["raw_violations"]} >= {"max_weight", "no_leverage"}
    assert body["weights"]["SYNTH01"] <= 0.4 + 1e-9
    assert sum(body["weights"].values()) <= 0.98 + 1e-9

    fine = await api.post(f"{API}/risk/validate", json={"weights": {"SYNTH01": 0.2}})
    assert fine.json()["compliant"] is True and fine.json()["applied"] == []
    unknown = await api.post(f"{API}/risk/validate", json={"weights": {"RELIANCE": 0.2}})
    assert unknown.status_code == 422


async def test_portfolio_endpoints_need_a_portfolio(api):
    paths = (
        "/portfolio",
        "/portfolio/allocation",
        "/portfolio/performance",
        "/paper/positions",
        "/paper/orders",
        "/paper/performance",
        "/risk/metrics",
    )
    for path in paths:
        response = await api.get(f"{API}{path}")
        assert response.status_code == 404, path


# ---------------------------------------------------------------- paper trading


async def test_baseline_paper_account_is_replayed_and_analysed(api, built):
    created = await _initialize(api, name="ew", initial_capital=500_000, start_date="2021-01-04")
    assert created.status_code == 201
    summary = created.json()
    assert summary["simulated"] is True and "No order is ever sent" in summary["note"]
    assert summary["start_date"] == "2021-01-04" and summary["is_synthetic"] is True
    assert summary["value"] == pytest.approx(summary["cash"] + summary["invested"])

    positions = (await api.get(f"{API}/paper/positions")).json()
    held = positions["holdings"]
    assert len(held) == summary["positions"] > 0
    assert sum(p["value"] for p in held) == pytest.approx(summary["invested"], rel=1e-9)

    performance = (await api.get(f"{API}/paper/performance")).json()
    equity = performance["equity"]
    assert equity["dates"][0] == "2021-01-04" and equity["values"][0] == 500_000
    assert equity["values"][-1] == pytest.approx(summary["value"])
    assert performance["metrics"]["total_costs"] > 0
    assert len(performance["daily_pnl"]) == 30

    orders = (await api.get(f"{API}/paper/orders")).json()
    assert orders["manual_orders"] == []
    assert len(orders["executed_trades"]) > 0
    assert orders["executed_trades"][0]["trade_date"] > "2021-01-04"  # never the decision day

    overview = (await api.get(f"{API}/portfolio")).json()
    assert overview["value"] == pytest.approx(summary["value"])
    allocation = (await api.get(f"{API}/portfolio/allocation")).json()
    assert allocation["cash_weight"] == pytest.approx(summary["cash"] / summary["value"])
    assert allocation["recommended"] is None

    risk = (await api.get(f"{API}/risk/metrics")).json()
    assert "paper account 'ew'" in risk["source"]
    assert risk["cvar"] >= risk["var"] > 0 and risk["limit_violations"] == []
    stress = (await api.get(f"{API}/risk/stress-test")).json()
    assert len(stress["historical"]) == 5
    assert "SYNTHETIC" in stress["generated"]["label"]
    assert "FAILED its quality gates" in stress["generated"]["label"]


async def test_manual_orders_are_simulated_risk_checked_and_never_filled_same_day(api, built):
    created = await _initialize(api, strategy={"kind": "manual"}, start_date="2021-06-01")
    assert created.status_code == 201 and created.json()["positions"] == 0

    first = await api.post(
        f"{API}/paper/orders",
        json={"symbol": "synth01", "target_weight": 0.25, "decision_date": "2021-06-01"},
    )
    assert first.status_code == 201
    assert first.json()["status"] == "executed" and first.json()["simulated"] is True

    greedy = await api.post(
        f"{API}/paper/orders",
        json={"symbol": "SYNTH02", "target_weight": 0.9, "decision_date": "2021-06-08"},
    )
    assert greedy.json()["status"] == "executed"
    assert "risk engine applied" in greedy.json()["detail"]

    pending = await api.post(
        f"{API}/paper/orders", json={"symbol": "SYNTH01", "target_weight": 0.1}
    )
    assert pending.json()["status"] == "pending"  # the latest session has no "next open" yet

    positions = {
        p["symbol"]: p for p in (await api.get(f"{API}/paper/positions")).json()["holdings"]
    }
    assert set(positions) == {"SYNTH01", "SYNTH02"}
    assert positions["SYNTH02"]["weight"] < 0.9  # capped by the risk engine, then drifted

    book = (await api.get(f"{API}/paper/orders")).json()
    assert [o["status"] for o in book["manual_orders"]] == ["executed", "executed", "pending"]
    fills = [t for t in book["executed_trades"] if t["symbol"] == "SYNTH01"]
    assert fills[0]["trade_date"] > "2021-06-01"  # filled at the next session's open
    assert any(d["applied"] for d in book["risk_decisions"])

    assert (
        await api.post(f"{API}/paper/orders", json={"symbol": "RELIANCE", "target_weight": 0.1})
    ).status_code == 422
    assert (
        await api.post(
            f"{API}/paper/orders",
            json={"symbol": "SYNTH01", "target_weight": 0.1, "decision_date": "2021-06-05"},
        )
    ).status_code == 422  # a Saturday


async def test_rl_paper_account_and_request_validation(api, trained):
    created = await _initialize(
        api, strategy={"kind": "rl", "ref": trained["rl"]["model_id"]}, start_date="2021-03-01"
    )
    assert created.status_code == 201
    assert created.json()["positions"] > 0
    refused = await api.post(
        f"{API}/paper/orders", json={"symbol": "SYNTH01", "target_weight": 0.1}
    )
    assert refused.status_code == 422  # orders only on manual accounts

    assert (await _initialize(api, strategy={"kind": "rl", "ref": "nope"})).status_code == 404
    assert (
        await _initialize(api, strategy={"kind": "baseline", "ref": "crystal_ball"})
    ).status_code == 422
    assert (await _initialize(api, start_date="2030-01-01")).status_code == 422
    assert (await _initialize(api, strategy={"kind": "broker"})).status_code == 422
    assert (await _initialize(api, initial_capital=-1)).status_code == 422


# ------------------------------------------------------------------------- jobs


@pytest.fixture
def queue(app):
    dispatched: list[str] = []
    app.dependency_overrides[get_job_dispatcher] = lambda: dispatched.append
    return dispatched


async def test_model_job_lifecycle(api, queue, session_factory, settings, trained):
    response = await api.post(
        f"{API}/rl/evaluate",
        json={"model_id": trained["rl"]["model_id"], "start": "2021-06-01", "end": "2021-12-31"},
    )
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "queued" and queue == [job["job_id"]]

    assert await jobs.execute_job(job["job_id"], session_factory, settings) == "succeeded"
    done = (await api.get(f"{API}/jobs/{job['job_id']}")).json()
    assert done["status"] == "succeeded" and done["error"] is None
    assert done["result"]["period"] == ["2021-06-01", "2021-12-31"]
    # The model was validated on 2021, so this "evaluation" is in-sample -- and says so.
    assert done["result"]["overlaps_training_or_validation"] is True
    assert "in-sample" in done["result"]["note"]
    assert (
        await jobs.execute_job(job["job_id"], session_factory, settings) == "succeeded"
    )  # no rerun

    assert (await api.get(f"{API}/jobs")).json()[0]["job_id"] == job["job_id"]
    assert (await api.get(f"{API}/jobs/nope")).status_code == 404
    assert (await api.post(f"{API}/rl/evaluate", json={"model_id": "nope"})).status_code == 404


async def test_failed_job_is_recorded_and_limits_apply(api, queue, session_factory, settings):
    async with session_factory() as session:
        session.add(
            Job(job_id="broken", kind="rl_evaluate", status="queued",
                params={"model_id": "missing"}, created_at=datetime.now(UTC))
        )  # fmt: skip
        await session.commit()
    assert await jobs.execute_job("broken", session_factory, settings) == "failed"
    failed = (await api.get(f"{API}/jobs/broken")).json()
    assert failed["status"] == "failed" and "Unknown rl model" in failed["error"]
    assert failed["finished_at"] is not None and failed["result"] is None

    assert (await api.post(f"{API}/rl/train", json={"total_timesteps": 1000})).status_code == 202
    assert (await api.post(f"{API}/gan/train", json={})).status_code == 202
    assert (await api.post(f"{API}/rl/train", json={})).status_code == 429  # two already active
    assert (await api.post(f"{API}/rl/train", json={"total_timesteps": 5})).status_code == 422


async def test_queue_outage_fails_the_job_visibly(api, app):
    def broken(job_id: str) -> None:
        raise ConnectionError("redis is down")

    app.dependency_overrides[get_job_dispatcher] = lambda: broken
    assert (await api.post(f"{API}/gan/train", json={})).status_code == 503
    listed = (await api.get(f"{API}/jobs")).json()
    assert listed[0]["status"] == "failed" and "queue unavailable" in listed[0]["error"]


# ------------------------------------------------------------------ experiments


async def test_experiment_reports_are_served(api, settings):
    directory = settings.artifacts_dir / "experiments" / "run-1"
    directory.mkdir(parents=True, exist_ok=True)
    report = {"run_id": "run-1", "created_at": "2026-11-12T00:00:00+00:00", "caveats": ["x"],
              "experiments": {"E0": {"label": "Equal weight"}}, "per_fold": [{"fold": 0}],
              "config": {"name": "t"}}  # fmt: skip
    (directory / "result.json").write_text(json.dumps(report))
    listed = (await api.get(f"{API}/experiments")).json()
    assert listed[0]["run_id"] == "run-1" and "per_fold" not in listed[0]
    full = (await api.get(f"{API}/experiments/run-1")).json()
    assert full["per_fold"] == [{"fold": 0}]
    assert (await api.get(f"{API}/experiments/nope")).status_code == 404


# ----------------------------------------------------------------------- alerts


async def test_alerts_describe_the_real_state_of_data_and_models(api, db_client, app, settings):
    alerts = {a["code"]: a for a in (await api.get(f"{API}/system/alerts")).json()["alerts"]}
    # The test dataset ends in 2021 and is synthetic; the generator failed its gates.
    assert alerts["stale_market_data"]["severity"] == "warning"
    assert "2021-12-31" in alerts["stale_market_data"]["message"]
    assert "synthetic_dataset" in alerts
    assert "generator_failed_quality_gates" in alerts
    assert "llm_not_configured" in alerts
    assert "no_rl_model" not in alerts and "no_deployment_model" not in alerts


async def test_alerts_without_any_dataset(db_client, app, settings):
    app.dependency_overrides[get_settings] = lambda: settings
    body = (await db_client.get(f"{API}/system/alerts")).json()
    assert [a["code"] for a in body["alerts"]] == ["no_dataset"]
    assert body["alerts"][0]["severity"] == "critical"
