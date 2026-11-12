"""Backtest orchestration: full-period comparison + walk-forward evaluation + artifacts.

    scripts/run_backtest.py --config configs/backtesting.yaml

For every configured strategy the runner produces

1. a **full-period** simulation over ``start_date .. end_date``;
2. a **walk-forward** evaluation: for each chronological fold the strategy is fitted on
   the training window and simulated, with fresh capital, on the unseen test window.
   The per-fold test returns are stitched into one out-of-sample series.

Each strategy is compared with the configured benchmark strategy (alpha, beta,
tracking error, and a block-bootstrap interval for the Sharpe difference). All
strategies share the same data, costs, schedule and execution assumptions.

Output: ``<output_root>/<run_id>/`` where ``run_id`` is derived from the config hash and
the dataset's content hash, so re-running the same experiment on the same data yields
the same id.

    result.json                 config, provenance, metrics, folds, caveats
    equity.csv / returns.csv    full-period series, one column per strategy
    walk_forward_returns.csv    stitched out-of-sample daily returns (+ fold index)
    <strategy>/trades.csv       every fill with its cost breakdown
    <strategy>/events.csv       partial fills, untradable orders, stale prices, ...
    <strategy>/weights.csv      end-of-day weights

Known limitations: see docs/backtesting.md.
"""

import hashlib
import math
import platform
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml.backtesting.config import BacktestConfig, StrategySpec
from ml.backtesting.costs import CostModel
from ml.backtesting.data import MarketData
from ml.backtesting.metrics import compute_metrics
from ml.backtesting.simulator import CASH, SimulationConfig, SimulationResult, simulate
from ml.backtesting.stats import bootstrap_sharpe
from ml.backtesting.strategies import StrategyContext, build_strategy
from ml.backtesting.walk_forward import Fold, rebalance_dates, walk_forward_folds
from ml.data.versioning import git_commit, write_json

SYNTHETIC_CAVEAT = (
    "Computed on SYNTHETIC development data. These numbers demonstrate that the engine "
    "works; they say nothing about any real market or strategy."
)
PERFORMANCE_CAVEAT = "Historical and simulated performance does not guarantee future results."


@dataclass
class BacktestReport:
    run_id: str
    path: Path
    summary: dict
    results: dict[str, SimulationResult]
    walk_forward_returns: pd.DataFrame


def _json_safe(value):
    """Replace NaN / infinity (not valid JSON) with null, recursively."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def simulation_config(cfg: BacktestConfig) -> SimulationConfig:
    return SimulationConfig(
        initial_capital=cfg.initial_capital,
        execution_price=cfg.execution_price,
        participation_limit=cfg.participation_limit,
        allow_fractional_shares=cfg.allow_fractional_shares,
        min_trade_value=cfg.min_trade_value,
        delisting_recovery=cfg.delisting_recovery,
    )


def _fit_context(market: MarketData, train_end: pd.Timestamp) -> StrategyContext:
    row = market.sessions.get_loc(train_end)
    close = market.close.iloc[row]
    listed = market.listed_on(train_end)
    return StrategyContext(
        date=train_end,
        universe=[s for s in market.symbols if listed[s] and np.isfinite(close[s])],
        prices=market.close.iloc[: row + 1],
        features=market.features_on(train_end),
        current_weights=pd.Series({CASH: 1.0}),
        regime=market.regime_on(train_end),
    )


def run_window(
    market: MarketData,
    spec: StrategySpec,
    cfg: BacktestConfig,
    cost_model: CostModel,
    start,
    end,
    *,
    train_end: pd.Timestamp | None = None,
) -> SimulationResult:
    """Simulate one strategy over ``start..end`` (fitting on data up to ``train_end``)."""
    strategy = build_strategy(spec.strategy, spec.params)
    strategy.name = spec.key
    if train_end is not None:
        strategy.fit(_fit_context(market, train_end))
    sessions = market.sessions_between(start, end)
    return simulate(
        market,
        strategy,
        start=start,
        end=end,
        decision_dates=rebalance_dates(sessions, cfg.rebalance),
        cost_model=cost_model,
        config=simulation_config(cfg),
    )


def _metrics(result: SimulationResult, cfg: BacktestConfig, benchmark: pd.Series | None) -> dict:
    metrics = compute_metrics(
        result.returns,
        risk_free_rate=cfg.risk_free_rate,
        periods_per_year=cfg.periods_per_year,
        benchmark=benchmark,
        turnover=result.equity["turnover"],
        trades=result.trades,
        costs=result.equity["costs"],
        initial_capital=result.initial_capital,
    )
    metrics["final_equity"] = float(result.equity["equity"].iloc[-1])
    metrics["slippage_cost"] = float(result.trades["slippage_cost"].sum())
    return metrics


def _significance(returns: pd.Series, benchmark: pd.Series, cfg: BacktestConfig) -> dict | None:
    try:
        return bootstrap_sharpe(
            returns,
            benchmark=benchmark,
            risk_free_rate=cfg.risk_free_rate,
            periods_per_year=cfg.periods_per_year,
            samples=cfg.bootstrap.samples,
            block_sessions=cfg.bootstrap.block_sessions,
            confidence=cfg.bootstrap.confidence,
            seed=cfg.bootstrap.seed,
        )
    except ValueError:
        return None  # window too short for the configured block length


def _walk_forward(
    market: MarketData, cfg: BacktestConfig, cost_model: CostModel, sessions: pd.DatetimeIndex
) -> tuple[list[Fold], dict, pd.DataFrame]:
    wf = cfg.walk_forward
    folds = walk_forward_folds(
        sessions,
        train_years=wf.train_years,
        validation_years=wf.validation_years,
        test_years=wf.test_years,
        step_years=wf.step_years,
        embargo_sessions=wf.embargo_sessions,
    )
    per_strategy: dict[str, dict] = {spec.key: {"folds": []} for spec in cfg.strategies}
    stitched: dict[str, list[pd.Series]] = {spec.key: [] for spec in cfg.strategies}
    fold_index: list[pd.Series] = []

    for fold in folds:
        results = {
            spec.key: run_window(
                market, spec, cfg, cost_model, fold.test_start, fold.test_end,
                train_end=fold.train_end,
            )
            for spec in cfg.strategies
        }  # fmt: skip
        benchmark = results[cfg.benchmark].returns
        fold_index.append(pd.Series(fold.index, index=benchmark.index))
        for key, result in results.items():
            relative = None if key == cfg.benchmark else benchmark
            per_strategy[key]["folds"].append(
                {"fold": fold.index, **_metrics(result, cfg, relative)}
            )
            stitched[key].append(result.returns)

    frame = pd.DataFrame()
    if folds:
        frame = pd.DataFrame({key: pd.concat(parts) for key, parts in stitched.items()})
        benchmark = frame[cfg.benchmark]
        for key in per_strategy:
            relative = None if key == cfg.benchmark else benchmark
            per_strategy[key]["out_of_sample"] = compute_metrics(
                frame[key],
                risk_free_rate=cfg.risk_free_rate,
                periods_per_year=cfg.periods_per_year,
                benchmark=relative,
            )
            if relative is not None:
                per_strategy[key]["sharpe_difference_vs_benchmark"] = _significance(
                    frame[key], benchmark, cfg
                )
        frame.insert(0, "fold", pd.concat(fold_index))
        frame.index.name = "trade_date"
    return folds, per_strategy, frame


def run_backtest(
    cfg: BacktestConfig,
    *,
    market: MarketData | None = None,
    output_root: Path | None = None,
) -> BacktestReport:
    market = market or MarketData.from_snapshot(cfg.dataset)
    start = pd.Timestamp(cfg.start_date) if cfg.start_date else market.sessions[0]
    end = pd.Timestamp(cfg.end_date) if cfg.end_date else market.sessions[-1]
    sessions = market.sessions_between(start, end)
    if len(sessions) < 2:
        raise ValueError("the configured period contains fewer than two sessions")
    cost_model = CostModel(cfg.costs)
    cost_model.rates_on(sessions[0])  # fail early if the cost schedule starts too late

    results = {
        spec.key: run_window(market, spec, cfg, cost_model, start, end) for spec in cfg.strategies
    }
    benchmark = results[cfg.benchmark].returns
    strategies: dict[str, dict] = {}
    for spec in cfg.strategies:
        result = results[spec.key]
        relative = None if spec.key == cfg.benchmark else benchmark
        entry = {
            "strategy": spec.strategy,
            "params": spec.params,
            "metrics": _metrics(result, cfg, relative),
            "events": result.events["event"].value_counts().sort_index().to_dict(),
        }
        if relative is not None:
            entry["sharpe_difference_vs_benchmark"] = _significance(result.returns, benchmark, cfg)
        strategies[spec.key] = entry

    folds: list[Fold] = []
    walk_forward: dict = {}
    wf_returns = pd.DataFrame()
    if cfg.walk_forward.enabled:
        folds, walk_forward, wf_returns = _walk_forward(market, cfg, cost_model, sessions)

    caveats = [PERFORMANCE_CAVEAT]
    if market.is_synthetic:
        caveats.insert(0, SYNTHETIC_CAVEAT)
    if cfg.walk_forward.enabled and not folds:
        caveats.append("Walk-forward produced no folds: the period is too short for the windows.")

    identity = f"{cfg.config_hash()}:{market.content_hash}"
    run_id = f"{cfg.name}-{hashlib.sha256(identity.encode()).hexdigest()[:12]}"
    summary = _json_safe(
        {
            "run_id": run_id,
            "name": cfg.name,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "dataset": {
                "dataset_id": market.dataset_id,
                "is_synthetic": market.is_synthetic,
                "provider": market.provider,
                "content_hash": market.content_hash,
            },
            "period": {
                "start": sessions[0].strftime("%Y-%m-%d"),
                "end": sessions[-1].strftime("%Y-%m-%d"),
                "sessions": len(sessions),
            },
            "benchmark": cfg.benchmark,
            "strategies": strategies,
            "walk_forward": {
                "enabled": cfg.walk_forward.enabled,
                "folds": [fold.as_dict() for fold in folds],
                "strategies": walk_forward,
            },
            "caveats": caveats,
            "config": cfg.model_dump(mode="json"),
            "config_hash": cfg.config_hash(),
            "environment": {
                "platform": f"{platform.system()}-{platform.machine()}",
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "git_commit": git_commit(),
            },
        }
    )

    path = _write(Path(output_root or cfg.output_root), run_id, summary, results, wf_returns)
    return BacktestReport(run_id, path, summary, results, wf_returns)


def _write(
    root: Path,
    run_id: str,
    summary: dict,
    results: dict[str, SimulationResult],
    wf_returns: pd.DataFrame,
) -> Path:
    staging = root / f".staging-{run_id}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    csv = {"date_format": "%Y-%m-%d", "lineterminator": "\n"}
    pd.DataFrame({k: r.equity["equity"] for k, r in results.items()}).to_csv(
        staging / "equity.csv", **csv
    )
    pd.DataFrame({k: r.returns for k, r in results.items()}).to_csv(staging / "returns.csv", **csv)
    if not wf_returns.empty:
        wf_returns.to_csv(staging / "walk_forward_returns.csv", **csv)
    for key, result in results.items():
        folder = staging / key
        folder.mkdir()
        result.trades.to_csv(folder / "trades.csv", index=False, **csv)
        result.events.to_csv(folder / "events.csv", index=False, **csv)
        result.weights.to_csv(folder / "weights.csv", **csv)
    write_json(summary, staging / "result.json")

    final = root / run_id
    if final.exists():
        shutil.rmtree(final)
    staging.rename(final)
    return final
