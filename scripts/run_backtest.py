"""Run the configured baseline strategies through the backtester.

    uv run python scripts/run_backtest.py --latest
    uv run python scripts/run_backtest.py --dataset artifacts/datasets/<dataset_id>

Results are written to artifacts/backtests/<run_id>/ (see ml/backtesting/runner.py).
"""

import argparse
import sys
from pathlib import Path

from ml.backtesting.config import BacktestConfig
from ml.backtesting.runner import run_backtest
from ml.data.versioning import MANIFEST_NAME

MISSING = "    -"


def _latest(root: Path) -> Path:
    candidates = (
        [p for p in root.iterdir() if (p / MANIFEST_NAME).is_file()] if root.is_dir() else []
    )
    if not candidates:
        raise SystemExit(f"No datasets found in {root}. Run scripts/build_dataset.py first.")
    return max(candidates, key=lambda p: (p / MANIFEST_NAME).stat().st_mtime)


def _pct(value) -> str:
    return MISSING if value is None else f"{value * 100:7.2f}%"


def _num(value) -> str:
    return MISSING if value is None else f"{value:7.2f}"


def _table(title: str, rows: dict[str, dict], benchmark: str) -> None:
    columns = [
        ("CAGR", "cagr", _pct, 9),
        ("vol", "volatility", _pct, 9),
        ("Sharpe", "sharpe", _num, 8),
        ("Sortino", "sortino", _num, 8),
        ("max DD", "max_drawdown", _pct, 9),
        ("Calmar", "calmar", _num, 8),
        ("turnover", "turnover", _num, 10),
        ("alpha", "alpha", _pct, 9),
        ("beta", "beta", _num, 8),
    ]
    print(f"\n{title}")
    print(f"  {'strategy':<16}" + "".join(f"{label:>{width}}" for label, _, _, width in columns))
    for key, metrics in rows.items():
        name = key + (" *" if key == benchmark else "")
        cells = "".join(f"{fmt(metrics.get(field)):>{width}}" for _, field, fmt, width in columns)
        print(f"  {name:<16}{cells}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/backtesting.yaml"))
    parser.add_argument("--dataset", type=Path, default=None, help="Dataset snapshot directory")
    parser.add_argument("--latest", action="store_true", help="Use the newest built dataset")
    parser.add_argument("--datasets-root", type=Path, default=Path("artifacts/datasets"))
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    cfg = BacktestConfig.from_yaml(args.config)
    if args.latest and args.dataset:
        parser.error("pass --dataset or --latest, not both")
    if args.latest:
        cfg.dataset = _latest(args.datasets_root)
    elif args.dataset:
        cfg.dataset = args.dataset
    if not (cfg.dataset / MANIFEST_NAME).is_file():
        print(f"FAILED: no dataset snapshot at {cfg.dataset} (use --latest or --dataset)",
              file=sys.stderr)  # fmt: skip
        return 2

    report = run_backtest(cfg, output_root=args.output_root)
    summary = report.summary

    for caveat in summary["caveats"]:
        print(f"*** {caveat}")
    period = summary["period"]
    print(f"\nrun_id   : {report.run_id}")
    print(f"path     : {report.path}")
    print(f"dataset  : {summary['dataset']['dataset_id']}")
    print(f"period   : {period['start']} .. {period['end']} ({period['sessions']} sessions)")
    print(f"benchmark: {summary['benchmark']} (marked *)")

    full = {key: entry["metrics"] for key, entry in summary["strategies"].items()}
    _table("Full period",
           full, summary["benchmark"])  # fmt: skip

    walk_forward = summary["walk_forward"]
    if walk_forward["folds"]:
        first, last = walk_forward["folds"][0], walk_forward["folds"][-1]
        oos = {k: v["out_of_sample"] for k, v in walk_forward["strategies"].items()}
        _table(
            f"Walk-forward, stitched test windows ({len(walk_forward['folds'])} folds, "
            f"{first['test_start']} .. {last['test_end']})",
            oos,
            summary["benchmark"],
        )
        print("\nSharpe difference vs benchmark (walk-forward, block bootstrap)")
        for key, entry in walk_forward["strategies"].items():
            test = entry.get("sharpe_difference_vs_benchmark")
            if test:
                verdict = "distinguishable from zero" if test["significant"] else "not significant"
                print(f"  {key:<16}{test['estimate']:+6.2f}  "
                      f"CI [{test['ci_low']:+.2f}, {test['ci_high']:+.2f}]  "
                      f"p={test['p_value']:.2f}  {verdict}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
