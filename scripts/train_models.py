"""Train the scenario generator or the deployment RL model from the command line.

    uv run python scripts/train_models.py gan --latest
    uv run python scripts/train_models.py rl --latest --timesteps 60000

These call the same functions as the API's background jobs (POST /api/v1/gan/train,
POST /api/v1/rl/train). Models and their cards are written to artifacts/models/.

    gan   TimeGAN on everything except the last ``--holdout`` sessions, evaluated on
          those held-out sessions. The card says ``validated`` or ``rejected``.
    rl    PPO on the most recent data with the last ``--holdout`` sessions as the
          validation window; the best seed is registered as the deployment model.
          It has no out-of-sample test of its own -- see scripts/run_experiments.py.
"""

import argparse
from pathlib import Path

from ml.backtesting.config import BacktestConfig
from ml.backtesting.costs import CostModel
from ml.backtesting.data import MarketData
from ml.backtesting.walk_forward import Fold
from ml.data.versioning import MANIFEST_NAME
from ml.fusion.experiments import ExperimentConfig, _load_yaml, _train_gan_for_fold
from ml.rl.training.deploy import train_deployment_agent


def _latest(root: Path) -> Path:
    candidates = (
        [p for p in root.iterdir() if (p / MANIFEST_NAME).is_file()] if root.is_dir() else []
    )
    if not candidates:
        raise SystemExit(f"No datasets found in {root}. Run scripts/build_dataset.py first.")
    return max(candidates, key=lambda p: (p / MANIFEST_NAME).stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=["gan", "rl"])
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--latest", action="store_true", help="Use the newest built dataset")
    parser.add_argument("--holdout", type=int, default=252, help="Held-out sessions at the end")
    parser.add_argument("--timesteps", type=int, default=None, help="RL: PPO total timesteps")
    parser.add_argument("--iterations", type=int, default=None, help="GAN: iterations per phase")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.latest:
        cfg.dataset = _latest(Path("artifacts/datasets"))
    elif args.dataset:
        cfg.dataset = args.dataset
    market = MarketData.from_snapshot(cfg.dataset)
    if market.is_synthetic:
        print("*** SYNTHETIC development data: the model learns nothing about real markets ***")
    print(f"dataset : {market.dataset_id}")

    if args.model == "gan":
        if args.iterations:
            phases = ("embedding_iterations", "supervised_iterations", "joint_iterations")
            cfg.gan = cfg.gan.model_copy(update=dict.fromkeys(phases, args.iterations))
        sessions = market.sessions
        fold = Fold(0, sessions[0], sessions[-args.holdout - 1], sessions[-args.holdout],
                    sessions[-1], sessions[-args.holdout], sessions[-1])  # fmt: skip
        trained = _train_gan_for_fold(market, cfg, fold, cfg.models_root)
        report = trained["report"]
        print(f"model   : {trained['model_id']}")
        print(f"trained : {fold.train_start:%Y-%m-%d} .. {fold.train_end:%Y-%m-%d}; "
              f"evaluated on the following {args.holdout} sessions")  # fmt: skip
        for name, gate in report["gates"].items():
            verdict = "pass" if gate["passed"] else "FAIL"
            print(f"  {name:<24}{gate['value']:8.3f}  (needs {gate['rule']} {gate['threshold']})"
                  f"  {verdict}")  # fmt: skip
        print("status  :", "validated" if report["passes"] else "REJECTED (failed quality gates)")
        return 0

    backtest = BacktestConfig.model_validate({**_load_yaml(cfg.backtest), "dataset": "."})
    if args.timesteps:
        cfg.rl.ppo.total_timesteps = args.timesteps
        cfg.rl.eval_every = max(args.timesteps // 4, 64)
    card = train_deployment_agent(
        market, cfg.rl, CostModel(backtest.costs), cfg.models_root,
        validation_sessions=args.holdout, risk_free_rate=backtest.risk_free_rate,
    )  # fmt: skip
    print(f"model   : {card['model_id']}")
    print(f"windows : {card['training_window']}")
    for score in card["metrics"]["seed_scores"]:
        print(f"  seed {score['seed']}: validation Sharpe {score['validation_sharpe']:.2f}")
    print(f"kept    : seed {card['seed']} (best on validation)")
    print("note    :", card["notes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
