"""Run the research experiments E0-E6 and the ablations (PRD 20).

    uv run python scripts/run_experiments.py --latest
    uv run python scripts/run_experiments.py --latest --experiments E0 E1 E2 --max-folds 1

Trains RL agents (and a scenario generator per fold) on the walk-forward training
windows and evaluates everything on unseen test windows in the shared backtester.
Results: artifacts/experiments/<run_id>/result.json. Models: artifacts/models/.
"""

import argparse
from pathlib import Path

from ml.data.versioning import MANIFEST_NAME
from ml.fusion.experiments import ExperimentConfig, run_experiments


def _latest(root: Path) -> Path:
    candidates = (
        [p for p in root.iterdir() if (p / MANIFEST_NAME).is_file()] if root.is_dir() else []
    )
    if not candidates:
        raise SystemExit(f"No datasets found in {root}. Run scripts/build_dataset.py first.")
    return max(candidates, key=lambda p: (p / MANIFEST_NAME).stat().st_mtime)


def _cell(value, percent: bool) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%" if percent else f"{value:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--latest", action="store_true", help="Use the newest built dataset")
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None, help="Override PPO total_timesteps")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.latest:
        cfg.dataset = _latest(Path("artifacts/datasets"))
    elif args.dataset:
        cfg.dataset = args.dataset
    if args.experiments:
        cfg.experiments = args.experiments
    if args.max_folds:
        cfg.max_folds = args.max_folds
    if args.timesteps:
        cfg.rl.ppo.total_timesteps = args.timesteps
        cfg.rl.eval_every = max(args.timesteps // 4, 64)
    if args.seeds:
        cfg.rl.seeds = args.seeds

    report = run_experiments(cfg)
    summary = report.summary
    print()
    for caveat in summary["caveats"]:
        print(f"*** {caveat}")
    folds = summary["folds"]
    print(f"\nrun_id  : {report.run_id}\npath    : {report.path}")
    print(f"dataset : {summary['dataset']['dataset_id']}")
    print(f"test    : {folds[0]['test_start']} .. {folds[-1]['test_end']} ({len(folds)} folds), "
          f"seeds {summary['seeds']}, {summary['duration_seconds'] / 60:.1f} min")  # fmt: skip
    for row in summary["generator"]:
        gates = ", ".join(f"{k}={v['value']:.2f}{'' if v['passed'] else ' FAIL'}"
                          for k, v in row["gates"].items())  # fmt: skip
        verdict = "passes" if row["passes"] else "FAILS"
        print(f"generator fold {row['fold']}: {verdict} gates ({gates})")

    print(f"\n{'experiment':<20}{'CAGR':>9}{'+/-':>8}{'Sharpe':>8}{'+/-':>7}{'range':>15}"
          f"{'max DD':>9}{'vs E0 Sharpe (95% CI)':>30}")  # fmt: skip
    for key, entry in summary["experiments"].items():
        if "skipped" in entry:
            print(f"{key:<20}skipped: {entry['skipped']}")
            continue
        mean, std = entry["mean"], entry["std"]
        spread = f"[{_cell(entry['min']['sharpe'], False)}, {_cell(entry['max']['sharpe'], False)}]"
        test = entry.get("sharpe_difference_vs_E0")
        versus = "-"
        if test:
            flag = "" if test["significant"] else " n.s."
            versus = (
                f"{test['estimate']:+.2f} [{test['ci_low']:+.2f}, {test['ci_high']:+.2f}]{flag}"
            )
        print(f"{key:<20}{_cell(mean['cagr'], True):>9}{_cell(std['cagr'], True):>8}"
              f"{_cell(mean['sharpe'], False):>8}{_cell(std['sharpe'], False):>7}{spread:>15}"
              f"{_cell(mean['max_drawdown'], True):>9}{versus:>30}")  # fmt: skip
    print("\n+/- and range are across training seeds; n.s. = not statistically significant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
