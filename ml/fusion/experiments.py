"""Research experiments (PRD 20): does adding GAN scenarios, LLM news intelligence and
richer fusion to an RL allocator actually help, compared with simple alternatives?

    E0   equal-weight benchmark
    E1   traditional quantitative strategy (cross-sectional momentum)
    E2   RL only                         market + regime features
    E3   RL + LLM                        adds news features
    E4   RL + GAN                        adds scenario features and GAN-augmented training
    E5   RL + GAN + LLM
    E6   RL + GAN + LLM + attention fusion

    Ablations (one component removed from the fullest available system)
    A_no_regime         without regime features
    A_no_synthetic      without GAN-augmented training (scenario features kept)
    A_no_risk_penalty   reward is the plain log return
    ("no GAN" is E2/E3, "no LLM" is E4, "no attention" is E5/E4.)

Protocol -- identical for every configuration
    * chronological walk-forward folds; models are fitted on ``train``, the checkpoint
      is chosen on ``validation``, and ``test`` is used once, for reporting;
    * every strategy -- baselines included -- trades on the same dates, pays the same
      costs in the same simulator and passes through the same risk limits;
    * the GAN is trained per fold on that fold's training window only, so no scenario
      can carry information from the validation or test periods;
    * RL configurations are trained with several seeds and **all** seeds are reported
      (mean, standard deviation, range) -- never only the best run;
    * Sharpe differences against E0 come with block-bootstrap confidence intervals.

Experiments that need news features are skipped (and say so) when no news feature
table is supplied: the LLM engine only has recent headlines, so there is no historical
news archive to build point-in-time features from.

The aim is not to show that the three-model system wins. A result where it does not
beat equal weight is a finding, and is reported as such.
"""

import hashlib
import json
import platform
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ml import registry
from ml.backtesting.config import BacktestConfig
from ml.backtesting.costs import CostModel
from ml.backtesting.data import MarketData
from ml.backtesting.metrics import compute_metrics
from ml.backtesting.runner import (
    PERFORMANCE_CAVEAT,
    SYNTHETIC_CAVEAT,
    _json_safe,
    simulation_config,
)
from ml.backtesting.simulator import simulate
from ml.backtesting.stats import bootstrap_sharpe
from ml.backtesting.strategies import build_strategy
from ml.backtesting.walk_forward import Fold, walk_forward_folds
from ml.data.versioning import git_commit, write_json
from ml.gan.config import GANConfig
from ml.gan.data import build_windows
from ml.gan.evaluation.metrics import evaluate
from ml.gan.generation.generate import generate_windows, scenario_features, synthetic_market
from ml.gan.training.train import save_gan, train_timegan
from ml.risk.strategy import RiskManagedStrategy
from ml.rl.config import RLConfig
from ml.rl.evaluation.strategy import risk_managed_policy, sb3_policy
from ml.rl.training.train import save_agent, train_agent

BENCHMARK = "E0"


@dataclass(frozen=True)
class ExperimentDef:
    key: str
    label: str
    kind: str  # "baseline" | "rl"
    strategy: str | None = None
    params: dict = field(default_factory=dict)
    groups: tuple[str, ...] = ("market", "regime")
    fusion: str = "early"
    reward_kind: str | None = None
    augmentation: bool = False

    @property
    def uses_gan(self) -> bool:
        return self.augmentation or "scenario" in self.groups

    @property
    def uses_news(self) -> bool:
        return "news" in self.groups


def experiment_definitions(news_available: bool) -> dict[str, ExperimentDef]:
    """The PRD's configurations. E6 and the ablations build on the fullest system that
    can actually be run: with news when a news feature table exists, otherwise without."""
    full = (
        ("market", "regime", "scenario", "news")
        if news_available
        else ("market", "regime", "scenario")
    )
    suffix = "" if news_available else " (no LLM features available)"
    without_regime = tuple(g for g in full if g != "regime")
    return {
        d.key: d
        for d in (
            ExperimentDef("E0", "Equal weight", "baseline", "equal_weight"),
            ExperimentDef("E1", "Momentum (traditional quant)", "baseline", "momentum",
                          {"lookback": 126, "skip": 21, "top_k": 5}),
            ExperimentDef("E2", "RL only", "rl"),
            ExperimentDef("E3", "RL + LLM", "rl", groups=("market", "regime", "news")),
            ExperimentDef("E4", "RL + GAN", "rl", groups=("market", "regime", "scenario"),
                          augmentation=True),
            ExperimentDef("E5", "RL + GAN + LLM", "rl",
                          groups=("market", "regime", "scenario", "news"), augmentation=True),
            ExperimentDef("E6", "RL + GAN + LLM + attention fusion" + suffix, "rl", groups=full,
                          fusion="attention", augmentation=True),
            ExperimentDef("A_no_regime", "Ablation: no regime features" + suffix, "rl",
                          groups=without_regime, augmentation=True),
            ExperimentDef("A_no_synthetic", "Ablation: no synthetic training data" + suffix, "rl",
                          groups=full, augmentation=False),
            ExperimentDef("A_no_risk_penalty", "Ablation: no risk penalty in the reward" + suffix,
                          "rl", groups=full, reward_kind="log_return", augmentation=True),
        )
    }  # fmt: skip


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(_Strict):
    name: str = Field("aether-experiments", pattern=r"^[a-z0-9][a-z0-9-]{1,48}$")
    dataset: Path
    backtest: Path = Path("configs/backtesting.yaml")  # costs, capital, folds, bootstrap
    rl: RLConfig = RLConfig()
    gan: GANConfig = GANConfig()
    experiments: list[str] = ["E0", "E1", "E2", "E4", "E6"]
    max_folds: int | None = Field(3, ge=1)  # most recent folds; None = all
    augmentation_markets: int = Field(2, ge=1)
    augmentation_sessions: int = Field(756, ge=120)
    news_features: Path | None = None  # CSV: symbol, trade_date, news_* columns
    models_root: Path = Path("artifacts/models")
    output_root: Path = Path("artifacts/experiments")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        import yaml

        with open(path, encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))

    def config_hash(self) -> str:
        payload = self.model_dump(
            mode="json", exclude={"output_root", "models_root", "dataset", "rl", "gan"}
        )
        payload["rl"], payload["gan"] = self.rl.config_hash(), self.gan.config_hash()
        payload["seeds"] = self.rl.seeds
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class ExperimentReport:
    run_id: str
    path: Path
    summary: dict
    returns: pd.DataFrame


def _log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def _train_gan_for_fold(market, cfg: ExperimentConfig, fold: Fold, models_root: Path) -> dict:
    """Train and evaluate the fold's generator; build scenario features and markets."""
    gan = train_timegan(
        market, cfg.gan, train_start=fold.train_start, train_end=fold.train_end, seed=fold.index
    )
    windows = gan.windows
    heldout = build_windows(
        market, fold.validation_start, fold.validation_end,
        sequence_length=cfg.gan.sequence_length, volume_window=cfg.gan.volume_window,
        clip_sigma=cfg.gan.clip_sigma, scaler=windows,
    )  # fmt: skip
    rng = np.random.default_rng(fold.index)
    codes = rng.choice(windows.conditions, cfg.gan.evaluation_windows)
    report = evaluate(
        windows,
        train_raw=windows.inverse(windows.windows.astype("float64")),
        heldout_raw=heldout.inverse(heldout.windows.astype("float64")),
        synthetic_raw=generate_windows(gan, codes, seed=fold.index + 1),
        gates=cfg.gan.gates,
        seed=fold.index,
    )
    identifier = registry.model_id(
        "gan", "timegan", cfg.gan.config_hash(), market.content_hash, fold.train_start,
        fold.train_end, fold.index,
    )  # fmt: skip
    directory = registry.register(
        models_root,
        kind="gan",
        model_id=identifier,
        dataset_id=market.dataset_id,
        hyperparameters=cfg.gan.model_dump(mode="json"),
        metrics={"passes": report["passes"], "gates": report["gates"],
                 "train_seconds": gan.train_seconds},
        status="validated" if report["passes"] else "rejected",
        seed=fold.index,
        config_hash=cfg.gan.config_hash(),
        training_window={"start": gan.train_start, "end": gan.train_end},
        is_synthetic_data=market.is_synthetic,
        notes=None if report["passes"] else "Failed quality gates; research comparison only.",
    )  # fmt: skip
    save_gan(gan, directory)
    write_json(_json_safe(report), directory / "evaluation.json")
    features, table = scenario_features(gan, market.symbols, seed=fold.index)
    table.to_csv(directory / "scenario_statistics.csv", index=False)
    markets = [
        synthetic_market(
            gan, market, cfg.augmentation_sessions, seed=1000 * (fold.index + 1) + k,
            features=None, regime=None,
        )
        for k in range(cfg.augmentation_markets)
    ]  # fmt: skip
    return {"model_id": identifier, "report": report, "features": features, "markets": markets,
            "excluded_assets": windows.excluded}  # fmt: skip


def run_experiments(
    cfg: ExperimentConfig,
    *,
    market: MarketData | None = None,
    output_root: Path | None = None,
    models_root: Path | None = None,
) -> ExperimentReport:
    started = time.perf_counter()
    market = market or MarketData.from_snapshot(cfg.dataset)
    backtest = BacktestConfig.model_validate(
        {**_load_yaml(cfg.backtest), "dataset": str(cfg.dataset)}
    )
    costs = CostModel(backtest.costs)
    sim_cfg = simulation_config(backtest)
    sectors = market.securities.get("sector")
    models_root = Path(models_root or cfg.models_root)
    step = cfg.rl.env.rebalance_every

    news = None
    if cfg.news_features is not None:
        news = pd.read_csv(cfg.news_features, parse_dates=["trade_date"])
    definitions = experiment_definitions(news_available=news is not None)
    unknown = [key for key in cfg.experiments if key not in definitions]
    if unknown:
        raise ValueError(f"unknown experiments {unknown}; available: {sorted(definitions)}")
    if BENCHMARK not in cfg.experiments:
        raise ValueError(f"{BENCHMARK} (the benchmark) must be one of the experiments")
    selected = [definitions[key] for key in cfg.experiments]
    skipped = {
        d.key: "needs news features; no historical news feature table was supplied"
        for d in selected
        if d.uses_news and news is None
    }
    runnable = [d for d in selected if d.key not in skipped]

    wf = backtest.walk_forward
    folds = walk_forward_folds(
        market.sessions, train_years=wf.train_years, validation_years=wf.validation_years,
        test_years=wf.test_years, step_years=wf.step_years, embargo_sessions=wf.embargo_sessions,
    )  # fmt: skip
    if cfg.max_folds:
        folds = folds[-cfg.max_folds :]
    if not folds:
        raise ValueError("the dataset is too short for the configured walk-forward windows")

    series: dict[tuple[str, int], list[pd.Series]] = {}
    fold_rows: list[dict] = []
    gan_reports: list[dict] = []
    risk_summary: dict[str, dict] = {}

    for fold in folds:
        _log(f"fold {fold.index}: train {fold.train_start:%Y-%m-%d}..{fold.train_end:%Y-%m-%d}, "
             f"test {fold.test_start:%Y-%m-%d}..{fold.test_end:%Y-%m-%d}")  # fmt: skip
        sessions = market.sessions_between(fold.test_start, fold.test_end)
        decisions = sessions[::step]
        gan = None
        if any(d.uses_gan for d in runnable):
            _log("  training the scenario generator")
            gan = _train_gan_for_fold(market, cfg, fold, models_root)
            gan_reports.append(
                {"fold": fold.index, "model_id": gan["model_id"], "passes": gan["report"]["passes"],
                 "gates": gan["report"]["gates"], "excluded_assets": gan["excluded_assets"],
                 "discriminative": gan["report"]["discriminative"],
                 "memorization": gan["report"]["memorization"]}
            )  # fmt: skip
            _log(f"  generator gates passed: {gan['report']['passes']}")

        def evaluate_strategy(
            strategy,
            key: str,
            seed: int,
            extra: dict,
            fold=fold,
            sessions=sessions,
            decisions=decisions,
        ) -> pd.Series:
            result = simulate(
                market, strategy, start=sessions[0], end=sessions[-1],
                decision_dates=decisions, cost_model=costs, config=sim_cfg,
            )  # fmt: skip
            metrics = compute_metrics(
                result.returns, risk_free_rate=backtest.risk_free_rate,
                periods_per_year=backtest.periods_per_year, turnover=result.equity["turnover"],
                trades=result.trades, costs=result.equity["costs"],
                initial_capital=result.initial_capital,
            )  # fmt: skip
            log = strategy.risk_log()
            entry = risk_summary.setdefault(key, {"decisions": 0, "modified": 0, "rejected": 0})
            entry["decisions"] += len(log)
            if len(log):
                entry["modified"] += int(log["applied"].map(bool).sum())
                entry["rejected"] += int(log["rejected"].sum())
            fold_rows.append(
                {"experiment": key, "seed": seed, "fold": fold.index, **metrics, **extra}
            )
            series.setdefault((key, seed), []).append(result.returns)
            return result.returns

        for definition in runnable:
            if definition.kind == "baseline":
                strategy = RiskManagedStrategy(
                    build_strategy(definition.strategy, definition.params), cfg.rl.limits, sectors
                )
                evaluate_strategy(strategy, definition.key, 0, {})
                continue
            rl_cfg = cfg.rl.model_copy(
                update={"groups": list(definition.groups), "fusion": definition.fusion}
            )
            if definition.reward_kind:
                rl_cfg = rl_cfg.model_copy(
                    update={
                        "reward": rl_cfg.reward.model_copy(update={"kind": definition.reward_kind})
                    }
                )
            for seed in cfg.rl.seeds:
                identifier = registry.model_id(
                    "rl", definition.key.lower(), rl_cfg.config_hash(), market.content_hash,
                    fold.train_start, fold.train_end, seed,
                    gan["model_id"] if gan and definition.uses_gan else "",
                )  # fmt: skip
                agent = train_agent(
                    market, rl_cfg, costs,
                    train_start=fold.train_start, train_end=fold.train_end,
                    validation_start=fold.validation_start, validation_end=fold.validation_end,
                    seed=seed, sectors=sectors,
                    news=news if definition.uses_news else None,
                    scenario=gan["features"] if gan and "scenario" in definition.groups else None,
                    augmentation=gan["markets"] if gan and definition.augmentation else None,
                    risk_free_rate=backtest.risk_free_rate,
                )  # fmt: skip
                strategy = risk_managed_policy(
                    sb3_policy(agent.model), agent.builder, limits=rl_cfg.limits, sectors=sectors,
                    logit_scale=rl_cfg.env.logit_scale, adv_window=rl_cfg.env.adv_window,
                    name=definition.key,
                )  # fmt: skip
                returns = evaluate_strategy(
                    strategy, definition.key, seed,
                    {"validation_sharpe": agent.best_validation_sharpe,
                     "best_step": agent.best_step, "train_seconds": agent.train_seconds,
                     "model_id": identifier},
                )  # fmt: skip
                directory = registry.register(
                    models_root,
                    kind="rl",
                    model_id=identifier,
                    dataset_id=market.dataset_id,
                    hyperparameters=rl_cfg.model_dump(mode="json"),
                    metrics={
                        "validation_sharpe": agent.best_validation_sharpe,
                        "validation_curve": agent.validation_curve,
                        "test_total_return": float(np.prod(1 + returns.to_numpy()) - 1),
                        "train_seconds": agent.train_seconds,
                    },
                    seed=seed,
                    config_hash=rl_cfg.config_hash(),
                    training_window={
                        "train": [f"{fold.train_start:%Y-%m-%d}", f"{fold.train_end:%Y-%m-%d}"],
                        "validation": [f"{fold.validation_start:%Y-%m-%d}",
                                       f"{fold.validation_end:%Y-%m-%d}"],
                        "test": [f"{fold.test_start:%Y-%m-%d}", f"{fold.test_end:%Y-%m-%d}"],
                    },
                    is_synthetic_data=market.is_synthetic,
                    notes=f"experiment {definition.key}: {definition.label}; "
                    + (f"generator {gan['model_id']}" if gan and definition.uses_gan
                       else "no generator"),
                    lineage={
                        "experiment": definition.key,
                        "fold": fold.index,
                        "generator_id": gan["model_id"] if gan and definition.uses_gan else None,
                        "uses_news": definition.uses_news,
                    },
                )  # fmt: skip
                save_agent(agent, directory)
                _log(f"  {definition.key} seed {seed}: validation Sharpe "
                     f"{agent.best_validation_sharpe:.2f}, test return "
                     f"{np.prod(1 + returns.to_numpy()) - 1:+.2%} "
                     f"({agent.train_seconds:.0f}s)")  # fmt: skip

    # ------------------------------------------------------------- aggregation
    stitched = {key: pd.concat(parts) for key, parts in series.items()}
    benchmark = stitched[(BENCHMARK, 0)]
    experiments: dict[str, dict] = {}
    columns: dict[str, pd.Series] = {}
    for definition in selected:
        if definition.key in skipped:
            experiments[definition.key] = {
                "label": definition.label,
                "skipped": skipped[definition.key],
            }
            continue
        seeds = sorted(seed for key, seed in stitched if key == definition.key)
        per_seed = []
        for seed in seeds:
            returns = stitched[(definition.key, seed)]
            columns[f"{definition.key}|seed{seed}"] = returns
            per_seed.append(
                {"seed": seed, **compute_metrics(
                    returns, risk_free_rate=backtest.risk_free_rate,
                    periods_per_year=backtest.periods_per_year,
                    benchmark=None if definition.key == BENCHMARK else benchmark,
                )}
            )  # fmt: skip
        names = [k for k in per_seed[0] if k != "seed"]
        values = {k: np.array([row[k] for row in per_seed], dtype="float64") for k in names}
        entry = {
            "label": definition.label,
            "kind": definition.kind,
            "groups": list(definition.groups) if definition.kind == "rl" else None,
            "fusion": definition.fusion if definition.kind == "rl" else None,
            "augmentation": definition.augmentation,
            "seeds": seeds,
            "mean": {k: _stat(np.mean, v) for k, v in values.items()},
            "std": {
                k: _stat(lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0, v)
                for k, v in values.items()
            },
            "min": {k: _stat(np.min, v) for k, v in values.items()},
            "max": {k: _stat(np.max, v) for k, v in values.items()},
            "per_seed": per_seed,
            "risk_engine": risk_summary.get(definition.key),
        }
        if definition.key != BENCHMARK:
            # Averaging the seeds' daily returns gives the "typical" run; the interval
            # is for that average, and the per-seed spread is reported alongside it.
            average = pd.concat([stitched[(definition.key, s)] for s in seeds], axis=1).mean(axis=1)
            try:
                entry["sharpe_difference_vs_E0"] = bootstrap_sharpe(
                    average, benchmark=benchmark, risk_free_rate=backtest.risk_free_rate,
                    periods_per_year=backtest.periods_per_year, samples=backtest.bootstrap.samples,
                    block_sessions=backtest.bootstrap.block_sessions,
                    confidence=backtest.bootstrap.confidence, seed=backtest.bootstrap.seed,
                )  # fmt: skip
            except ValueError:
                entry["sharpe_difference_vs_E0"] = None
        experiments[definition.key] = entry

    caveats = [PERFORMANCE_CAVEAT]
    if market.is_synthetic:
        caveats.insert(0, SYNTHETIC_CAVEAT)
    if market.provider == "yahoo_chart":
        caveats.append(
            "The universe was chosen from today's large caps (survivorship bias): absolute "
            "returns of every strategy, including the benchmark, are overstated."
        )
    if gan_reports and not all(r["passes"] for r in gan_reports):
        caveats.append(
            "The scenario generator failed its quality gates in at least one fold. Results "
            "that use GAN scenarios show what this generator does, not what a good one could do."
        )
    if skipped:
        caveats.append("Skipped: " + "; ".join(f"{k} ({v})" for k, v in skipped.items()))

    identity = f"{cfg.config_hash()}:{market.content_hash}"
    run_id = f"{cfg.name}-{hashlib.sha256(identity.encode()).hexdigest()[:12]}"
    summary = _json_safe(
        {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "dataset": {"dataset_id": market.dataset_id, "is_synthetic": market.is_synthetic,
                        "provider": market.provider, "content_hash": market.content_hash},
            "benchmark": BENCHMARK,
            "folds": [fold.as_dict() for fold in folds],
            "rebalance_every_sessions": step,
            "seeds": cfg.rl.seeds,
            "experiments": experiments,
            "per_fold": fold_rows,
            "generator": gan_reports,
            "caveats": caveats,
            "config": cfg.model_dump(mode="json"),
            "config_hash": cfg.config_hash(),
            "duration_seconds": round(time.perf_counter() - started, 1),
            "environment": {"platform": f"{platform.system()}-{platform.machine()}",
                            "python": platform.python_version(), "git_commit": git_commit()},
        }
    )  # fmt: skip

    root = Path(output_root or cfg.output_root)
    path = root / run_id
    path.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(columns)
    frame.index.name = "trade_date"
    frame.to_csv(path / "returns.csv", date_format="%Y-%m-%d", lineterminator="\n")
    write_json(summary, path / "result.json")
    return ExperimentReport(run_id, path, summary, frame)


def _stat(function, values: np.ndarray) -> float:
    """``function`` over the defined values; NaN when a metric is undefined for all seeds."""
    defined = values[np.isfinite(values)]
    return float(function(defined)) if len(defined) else float("nan")


def _load_yaml(path: Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)
