"""Research experiment runner, fusion networks and model registry (tiny budgets)."""

import json

import numpy as np
import pytest
import torch
import yaml
from gymnasium import spaces

from ml import registry
from ml.fusion.experiments import ExperimentConfig, experiment_definitions, run_experiments
from ml.fusion.extractors import GroupFusionExtractor, fusion_policy_kwargs
from ml.gan.config import GANConfig
from ml.risk.limits import RiskLimits
from ml.rl.config import EnvConfig, PPOConfig, RLConfig

from .test_backtest_engine import SHIPPED
from .test_rl_env import FEATURES

SPEC = {"market": (0, 8), "regime": (8, 12), "portfolio": (12, 18)}


# ------------------------------------------------------------------------ fusion


def _observations(batch: int = 5) -> torch.Tensor:
    return torch.randn(batch, 18, generator=torch.Generator().manual_seed(0))


def test_late_fusion_concatenates_group_embeddings():
    space = spaces.Box(-10, 10, (18,), np.float32)
    extractor = GroupFusionExtractor(space, SPEC, mode="late", embed_dim=16)
    out = extractor(_observations())
    assert out.shape == (5, 48) and extractor.features_dim == 48
    assert extractor.attention_by_group() is None


def test_attention_fusion_weights_groups_and_exposes_the_weights():
    space = spaces.Box(-10, 10, (18,), np.float32)
    extractor = GroupFusionExtractor(space, SPEC, mode="attention", embed_dim=16)
    out = extractor(_observations(1))
    assert out.shape == (1, 16)
    weights = extractor.attention_by_group()
    assert set(weights) == {"market", "regime", "portfolio"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(w > 0 for w in weights.values())
    # The weights depend on the input (they are not a fixed mixing vector).
    extractor(_observations(1) * 5 + 2)
    assert extractor.attention_by_group() != weights
    with pytest.raises(ValueError):
        GroupFusionExtractor(space, SPEC, mode="magic")


def test_each_group_encoder_sees_only_its_own_slice():
    space = spaces.Box(-10, 10, (18,), np.float32)
    extractor = GroupFusionExtractor(space, SPEC, mode="late", embed_dim=4)
    base = _observations(1)
    changed = base.clone()
    changed[:, 8:12] += 3.0  # perturb the regime block only
    a, b = extractor(base), extractor(changed)
    assert torch.equal(a[:, :4], b[:, :4])  # market embedding untouched
    assert not torch.equal(a[:, 4:8], b[:, 4:8])  # regime embedding changed
    assert torch.equal(a[:, 8:], b[:, 8:])  # portfolio embedding untouched
    kwargs = fusion_policy_kwargs("attention", SPEC, n_assets=2)
    assert kwargs["features_extractor_class"] is GroupFusionExtractor


# ---------------------------------------------------------------------- registry


def test_registry_round_trip(tmp_path):
    identifier = registry.model_id("rl", "e2", "cfg", "data", 0)
    assert identifier == registry.model_id("rl", "e2", "cfg", "data", 0)  # stable
    assert identifier != registry.model_id("rl", "e2", "cfg", "data", 1)
    directory = registry.register(
        tmp_path, kind="rl", model_id=identifier, dataset_id="ds-1",
        hyperparameters={"lr": 3e-4}, metrics={"sharpe": float("nan"), "n": np.int64(3)},
        seed=0, config_hash="abc", training_window={"start": "2018-01-01"},
    )  # fmt: skip
    card = registry.load_card(tmp_path, "rl", identifier)
    for required in ("model_id", "version", "trained_at", "dataset_id", "hyperparameters",
                     "metrics", "artifact_path", "status"):  # fmt: skip
        assert required in card  # the fields PRD 16.1 lists
    assert card["metrics"] == {"sharpe": None, "n": 3}  # NaN is null, not a fake number
    assert card["status"] == "candidate" and card["artifact_path"] == directory.as_posix()
    registry.register(tmp_path, kind="gan", model_id="gan-x", dataset_id="ds-1",
                      hyperparameters={}, metrics={}, status="rejected")  # fmt: skip
    assert {c["kind"] for c in registry.list_models(tmp_path)} == {"rl", "gan"}
    assert [c["model_id"] for c in registry.list_models(tmp_path, "gan")] == ["gan-x"]
    assert registry.list_models(tmp_path / "missing") == []


# ------------------------------------------------------------------- experiments


def test_experiment_definitions_cover_the_prd_table():
    without_news = experiment_definitions(news_available=False)
    assert {"E0", "E1", "E2", "E3", "E4", "E5", "E6"} <= set(without_news)
    assert without_news["E2"].groups == ("market", "regime") and not without_news["E2"].uses_gan
    assert without_news["E3"].uses_news and without_news["E5"].uses_news
    assert without_news["E4"].augmentation and "scenario" in without_news["E4"].groups
    assert without_news["E6"].fusion == "attention" and not without_news["E6"].uses_news
    assert "regime" not in without_news["A_no_regime"].groups
    assert without_news["A_no_synthetic"].augmentation is False
    assert without_news["A_no_risk_penalty"].reward_kind == "log_return"
    assert experiment_definitions(news_available=True)["E6"].uses_news


@pytest.fixture(scope="module")
def experiment_config(market, tmp_path_factory):
    root = tmp_path_factory.mktemp("experiments")
    backtest = {
        **SHIPPED,
        "walk_forward": {"train_years": 1, "validation_years": 1, "test_years": 1,
                         "embargo_sessions": 5},
        "bootstrap": {"samples": 200, "block_sessions": 10, "seed": 1},
    }  # fmt: skip
    (root / "backtest.yaml").write_text(yaml.safe_dump(backtest))
    return ExperimentConfig(
        name="test-experiments",
        dataset=root,  # unused: the market is passed in
        backtest=root / "backtest.yaml",
        experiments=["E0", "E1", "E2", "E3", "E4", "E6", "A_no_risk_penalty"],
        max_folds=1,
        augmentation_markets=1,
        augmentation_sessions=160,
        rl=RLConfig(
            features=FEATURES,
            env=EnvConfig(rebalance_every=5, episode_steps=12),
            ppo=PPOConfig(
                total_timesteps=256, n_steps=64, batch_size=32, n_epochs=1, net_arch=[16]
            ),
            eval_every=128,
            limits=RiskLimits(max_weight=0.4),
            seeds=[0, 1],
            torch_threads=1,
        ),
        gan=GANConfig(
            sequence_length=12, volume_window=20, hidden_dim=8, num_layers=1, noise_dim=4,
            batch_size=32, embedding_iterations=3, supervised_iterations=3, joint_iterations=3,
            calibration_windows=50, min_regime_windows=5, evaluation_windows=60, torch_threads=1,
        ),
        models_root=root / "models",
        output_root=root / "runs",
    )  # fmt: skip


@pytest.fixture(scope="module")
def report(market, experiment_config):
    return run_experiments(experiment_config, market=market)


def test_report_is_valid_json_with_provenance_and_caveats(report):
    text = (report.path / "result.json").read_text()
    summary = json.loads(text, parse_constant=lambda token: pytest.fail(f"invalid JSON: {token}"))
    assert summary["run_id"] == report.run_id
    assert summary["dataset"]["is_synthetic"] is True
    assert any("SYNTHETIC" in caveat for caveat in summary["caveats"])
    assert len(summary["folds"]) == 1 and summary["seeds"] == [0, 1]
    assert summary["benchmark"] == "E0"


def test_every_seed_is_reported_not_just_the_best(report):
    experiments = report.summary["experiments"]
    assert experiments["E0"]["seeds"] == [0]  # deterministic baseline
    for key in ("E2", "E4", "E6", "A_no_risk_penalty"):
        entry = experiments[key]
        assert entry["seeds"] == [0, 1]
        assert len(entry["per_seed"]) == 2
        sharpes = [row["sharpe"] for row in entry["per_seed"]]
        assert entry["min"]["sharpe"] == pytest.approx(min(sharpes))
        assert entry["max"]["sharpe"] == pytest.approx(max(sharpes))
        assert entry["mean"]["sharpe"] == pytest.approx(np.mean(sharpes))
        assert "beta" in entry["mean"]  # measured against E0
        assert entry["risk_engine"]["decisions"] > 0
    assert experiments["E6"]["fusion"] == "attention"
    assert "scenario" in experiments["E4"]["groups"] and experiments["E4"]["augmentation"]
    assert set(report.returns.columns) >= {"E0|seed0", "E2|seed0", "E2|seed1", "E4|seed1"}


def test_experiments_that_need_news_are_skipped_with_a_reason(report):
    skipped = report.summary["experiments"]["E3"]
    assert "skipped" in skipped and "news" in skipped["skipped"]
    assert "E3|seed0" not in report.returns.columns
    assert any("Skipped: E3" in caveat for caveat in report.summary["caveats"])


def test_generator_is_evaluated_and_its_verdict_is_reported(report):
    generator = report.summary["generator"]
    assert len(generator) == 1
    assert set(generator[0]["gates"]) >= {"return_ks", "discriminative_score", "memorization_ratio"}
    # Four iterations of training cannot pass the gates -- and the report must say so.
    assert generator[0]["passes"] is False
    assert any("failed its quality gates" in caveat for caveat in report.summary["caveats"])


def test_models_are_registered_with_their_lineage(report, experiment_config):
    root = experiment_config.models_root
    rl_cards = registry.list_models(root, "rl")
    gan_cards = registry.list_models(root, "gan")
    assert len(rl_cards) == 4 * 2  # four RL configurations x two seeds
    assert len(gan_cards) == 1 and gan_cards[0]["status"] == "rejected"
    card = rl_cards[0]
    assert card["training_window"]["train"][1] < card["training_window"]["test"][0]
    assert card["metrics"]["validation_curve"]
    directory = root / "rl" / card["model_id"]
    assert {"model.zip", "observation.json", "model_card.json"} <= {
        p.name for p in directory.iterdir()
    }
    assert (root / "gan" / gan_cards[0]["model_id"] / "evaluation.json").is_file()


def test_misconfigured_experiments_are_rejected(market, experiment_config):
    with pytest.raises(ValueError, match="unknown experiments"):
        run_experiments(
            experiment_config.model_copy(update={"experiments": ["E0", "E9"]}), market=market
        )
    with pytest.raises(ValueError, match="benchmark"):
        run_experiments(experiment_config.model_copy(update={"experiments": ["E2"]}), market=market)


# ---------------------------------------------------------------- explainability


def test_allocation_explanation_is_built_from_the_models_actual_output(
    market, report, experiment_config
):
    import pandas as pd

    from ml.fusion.explain import ASSOCIATION_NOTE, AllocationInputs, explain_allocation
    from ml.rl.training.train import load_agent

    root = experiment_config.models_root
    card = next(c for c in registry.list_models(root, "rl") if "experiment E2" in c["notes"])
    cfg = RLConfig.model_validate(card["hyperparameters"])
    model, builder = load_agent(root / "rl" / card["model_id"], market, cfg)
    inputs = AllocationInputs(
        model=model, builder=builder, limits=cfg.limits, logit_scale=cfg.env.logit_scale,
        sectors=market.securities["sector"], adv=None, model_id=card["model_id"],
        dataset_id=market.dataset_id,
    )  # fmt: skip
    day = market.sessions[-30]
    previous = pd.Series({"SYNTH01": 0.3, "SYNTH02": 0.1})
    record = explain_allocation(inputs, day, previous, drawdown=-0.03)

    allocation = pd.Series(record["allocation"])
    assert record["as_of"] == day.strftime("%Y-%m-%d")
    assert record["executable_from"] == market.sessions[-29].strftime("%Y-%m-%d")  # next session
    assert record["model_id"] == card["model_id"] and record["dataset_id"] == market.dataset_id
    assert record["is_recommendation_for_real_trading"] is False
    assert allocation.max() <= cfg.limits.max_weight + 1e-9  # passed through the risk engine
    assert record["summary"]["cash"] == pytest.approx(1 - allocation.sum())
    assert record["summary"]["cash"] >= cfg.limits.min_cash - 1e-9
    assert record["previous_allocation"] == {"SYNTH01": 0.3, "SYNTH02": 0.1}

    evidence = record["evidence"]
    assert evidence["risk_engine"]["rejected"] is False
    assert evidence["risk_limits"]["max_weight"] == cfg.limits.max_weight
    top = evidence["positions"][0]
    assert top["weight"] == pytest.approx(allocation.max())
    assert top["change"] == pytest.approx(top["weight"] - top["previous_weight"])
    assert len(top["most_influential_features"]) == 3
    assert {f["feature"] for f in top["most_influential_features"]} <= set(FEATURES)
    assert evidence["policy_action_spread"] > 0

    sensitivity = record["sensitivity"]
    assert set(sensitivity["by_information_source"]) == {"market", "regime"}
    assert all(0 <= v <= 1 for v in sensitivity["by_information_source"].values())
    assert sensitivity["note"] == ASSOCIATION_NOTE and "not causal" in ASSOCIATION_NOTE

    # Every number quoted in the narrative comes from the record itself.
    text = " ".join(record["summary"]["narrative"])
    assert f"{allocation.sum():.1%} invested" in text
    assert "not a causal explanation" in text
    assert record["summary"]["regime"]["regime"].replace("_", " ") in text

    # Deterministic: the same inputs give the same record.
    assert explain_allocation(inputs, day, previous, drawdown=-0.03) == record
