"""Walk-forward splits, the backtest runner, and whole-engine invariants on a dataset
built by the real Phase 1 pipeline (synthetic development provider)."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from ml.backtesting.config import BacktestConfig
from ml.backtesting.costs import CostModel
from ml.backtesting.data import MarketData
from ml.backtesting.runner import run_backtest, run_window, simulation_config
from ml.backtesting.simulator import CASH, simulate
from ml.backtesting.strategies import Strategy, build_strategy
from ml.backtesting.walk_forward import rebalance_dates, walk_forward_folds
from ml.data.config import DataConfig
from ml.data.pipeline import build_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED = yaml.safe_load((REPO_ROOT / "configs" / "backtesting.yaml").read_text())


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    cfg = DataConfig.model_validate(
        {
            "dataset": {
                "name": "bt-test",
                "start_date": date(2018, 1, 1),
                "end_date": date(2021, 12, 31),
            },
            "provider": {"kind": "synthetic_dev", "synthetic_dev": {"seed": 3, "n_symbols": 6}},
        }
    )
    return build_dataset(cfg, output_root=tmp_path_factory.mktemp("datasets"))


@pytest.fixture(scope="module")
def market(dataset) -> MarketData:
    return MarketData.from_snapshot(dataset.path)


def make_config(dataset_path, **overrides) -> BacktestConfig:
    base = {
        "name": "engine-test",
        "dataset": str(dataset_path),
        "risk_free_rate": 0.05,
        "costs": SHIPPED["costs"],
        "strategies": [
            {"strategy": "equal_weight"},
            {"strategy": "buy_and_hold"},
            {"strategy": "momentum", "params": {"lookback": 60, "skip": 5, "top_k": 2}},
            {"strategy": "moving_average", "params": {"window": 50}},
            {"strategy": "mean_variance", "params": {"lookback": 120}},
        ],
        "benchmark": "equal_weight",
        "walk_forward": {
            "train_years": 1,
            "validation_years": 1,
            "test_years": 1,
            "embargo_sessions": 5,
        },
        "bootstrap": {"samples": 200, "block_sessions": 10, "seed": 1},
    }
    return BacktestConfig.model_validate({**base, **overrides})


@pytest.fixture(scope="module")
def report(dataset, tmp_path_factory):
    return run_backtest(make_config(dataset.path), output_root=tmp_path_factory.mktemp("runs"))


# ------------------------------------------------------------------- market view


def test_market_view_matches_the_snapshot(dataset, market):
    assert market.dataset_id == dataset.dataset_id
    assert market.is_synthetic is True
    assert market.symbols == sorted(dataset.bars["symbol"].unique())
    row = dataset.bars.iloc[1234]
    assert market.close.loc[row["trade_date"], row["symbol"]] == row["adj_close"]
    assert market.open.loc[row["trade_date"], row["symbol"]] == row["adj_open"]
    assert market.traded_value.loc[row["trade_date"], row["symbol"]] == row["turnover"]
    assert market.close.isna().any().any()  # late listing / delisting leave gaps


def test_market_view_rejects_a_tampered_snapshot(dataset, tmp_path):
    import shutil

    copy = tmp_path / dataset.dataset_id
    shutil.copytree(dataset.path, copy)
    bars = copy / "bars.csv"
    bars.write_text(bars.read_text().replace("SYNTH01", "SYNTH99", 1))
    with pytest.raises(ValueError, match="modified after it was built"):
        MarketData.from_snapshot(copy)


# ---------------------------------------------------------------------- schedules


def test_rebalance_schedules(market):
    sessions = market.sessions
    assert rebalance_dates(sessions, "daily").equals(sessions)

    monthly = rebalance_dates(sessions, "monthly")
    assert monthly[0] == sessions[0]  # capital is deployed at the start
    month_ends = pd.Series(sessions, index=sessions).groupby(sessions.to_period("M")).max()
    assert set(monthly[1:]) == set(month_ends) - {sessions[0]}
    assert len(monthly) in (48, 49)

    weekly = rebalance_dates(sessions, "weekly")
    assert 200 < len(weekly) < 215
    assert set(weekly) <= set(sessions)
    assert len(rebalance_dates(sessions, "quarterly")) in (16, 17)
    with pytest.raises(ValueError):
        rebalance_dates(sessions, "hourly")


def test_walk_forward_folds_reproduce_the_prd_example():
    sessions = pd.bdate_range("2015-01-01", "2023-12-29")
    folds = walk_forward_folds(sessions, train_years=5, validation_years=1, test_years=1)
    spans = [
        (f.train_start.year, f.train_end.year, f.validation_start.year, f.test_start.year)
        for f in folds
    ]
    assert spans == [(2015, 2019, 2020, 2021), (2016, 2020, 2021, 2022), (2017, 2021, 2022, 2023)]


def test_walk_forward_folds_are_chronological_and_embargoed(market):
    plain = walk_forward_folds(market.sessions, train_years=1, validation_years=1, test_years=1)
    embargoed = walk_forward_folds(
        market.sessions, train_years=1, validation_years=1, test_years=1, embargo_sessions=5
    )
    assert len(plain) == len(embargoed) == 2
    for fold in embargoed:
        assert fold.train_start < fold.train_end < fold.validation_start
        assert fold.validation_end < fold.test_start < fold.test_end
        for bound in (fold.train_start, fold.train_end, fold.test_start, fold.test_end):
            assert bound in market.sessions
    for without, with_embargo in zip(plain, embargoed, strict=True):
        skipped = market.sessions_between(without.test_start, with_embargo.test_start)
        assert len(skipped) == 6  # 5 embargoed sessions + the new first session
        gap = market.sessions_between(with_embargo.validation_end, with_embargo.test_start)
        assert len(gap) - 2 == 5  # sessions strictly between validation and test
    # Later folds move forward in time; test windows never overlap.
    assert embargoed[0].test_end < embargoed[1].test_start


def test_walk_forward_never_tests_beyond_the_data(market):
    assert (
        walk_forward_folds(market.sessions, train_years=3, validation_years=1, test_years=1) == []
    )
    assert (
        walk_forward_folds(market.sessions[:100], train_years=1, validation_years=1, test_years=1)
        == []
    )
    with pytest.raises(ValueError):
        walk_forward_folds(market.sessions, train_years=0, validation_years=1, test_years=1)


# ----------------------------------------------------- whole-engine invariants


@pytest.fixture(scope="module")
def full_runs(dataset, market):
    cfg = make_config(dataset.path)
    costs = CostModel(cfg.costs)
    start, end = market.sessions[0], market.sessions[-1]
    return {spec.key: run_window(market, spec, cfg, costs, start, end) for spec in cfg.strategies}


def test_accounting_identities_hold_for_every_strategy(full_runs, market):
    for key, result in full_runs.items():
        equity = result.equity
        np.testing.assert_allclose(result.weights.sum(axis=1), 1.0, atol=1e-9, err_msg=key)
        np.testing.assert_allclose(equity["equity"], equity["cash"] + equity["invested"])
        assert equity["cash"].min() >= -1e-6, key  # never borrows
        assert (result.weights.drop(columns=CASH) >= -1e-12).all().all(), key  # never short
        assert equity["costs"].sum() == pytest.approx(result.trades["total_cost"].sum()), key
        assert equity["equity"].iloc[0] == result.initial_capital  # nothing trades on day 0
        growth = (1 + result.returns).prod()
        assert growth == pytest.approx(equity["equity"].iloc[-1] / result.initial_capital)


def test_every_fill_is_on_a_real_bar_inside_the_listing_window(full_runs, market):
    for key, result in full_runs.items():
        fills = result.trades[result.trades["side"].isin(["buy", "sell"])]
        assert len(fills) > 0, key
        assert (fills["trade_date"] > market.sessions[0]).all()
        for trade in fills.itertuples():
            assert np.isfinite(market.open.loc[trade.trade_date, trade.symbol])
            assert market.listed_on(trade.trade_date)[trade.symbol]
        limit = 0.05 * np.array(
            [market.traded_value.loc[t.trade_date, t.symbol] for t in fills.itertuples()]
        )
        assert (fills["value"].to_numpy() <= limit * (1 + 1e-3)).all(), key


def test_delisted_holdings_are_closed_out(full_runs, market):
    gone = market.securities["delisting_date"].dropna()
    assert len(gone) == 1
    symbol, delisted_on = gone.index[0], gone.iloc[0]
    for key, result in full_runs.items():
        after = result.weights.loc[result.weights.index >= delisted_on, symbol]
        assert (after == 0).all(), key
    held = full_runs["buy_and_hold"]
    assert "delist" in set(held.trades["side"])
    assert set(held.trades["side"]) == {"buy", "delist"}  # it never rebalances


def test_costs_reduce_returns(dataset, market):
    cfg = make_config(dataset.path, rebalance="weekly")
    spec = cfg.strategies[0]
    start, end = market.sessions[0], market.sessions[-1]
    with_costs = run_window(market, spec, cfg, CostModel(cfg.costs), start, end)
    free = run_window(market, spec, cfg, CostModel.zero(), start, end)
    drag = free.equity["equity"].iloc[-1] - with_costs.equity["equity"].iloc[-1]
    assert drag > 0
    explicit = with_costs.trades["total_cost"].sum() + with_costs.trades["slippage_cost"].sum()
    assert drag > 0.5 * explicit  # costs compound, so the drag is at least of this order


class _Spy(Strategy):
    """Wraps a strategy and records the latest date visible at each decision."""

    def __init__(self, inner: Strategy) -> None:
        self.inner, self.name, self.violations, self.calls = inner, inner.name, [], 0

    def target_weights(self, ctx):
        self.calls += 1
        if ctx.prices.index.max() > ctx.date:
            self.violations.append(("prices", ctx.date))
        if ctx.features is not None and len(ctx.features) and "trade_date" in ctx.features:
            self.violations.append(("features", ctx.date))
        return self.inner.target_weights(ctx)


@pytest.mark.parametrize("name", ["equal_weight", "momentum", "moving_average", "mean_variance"])
def test_future_prices_cannot_change_past_results(dataset, market, name):
    """Scramble every price after a cut-off: equity up to the cut-off must be identical."""
    cfg = make_config(dataset.path)
    spec = next(s for s in cfg.strategies if s.key == name)
    costs = CostModel(cfg.costs)
    start, end = market.sessions[0], market.sessions[-1]
    cut = market.sessions[600]

    rng = np.random.default_rng(42)
    future = market.sessions > cut
    scrambled = {}
    for field in ("open", "close", "traded_value"):
        frame = getattr(market, field).copy()
        frame.loc[future] = frame.loc[future] * rng.uniform(0.3, 3.0, frame.loc[future].shape)
        scrambled[field] = frame
    tampered = MarketData(
        sessions=market.sessions, symbols=market.symbols, securities=market.securities,
        features=market.features, **scrambled,
    )  # fmt: skip

    def run(view):
        spy = _Spy(build_strategy(spec.strategy, spec.params))
        result = simulate(
            view, spy, start=start, end=end,
            decision_dates=rebalance_dates(view.sessions_between(start, end), cfg.rebalance),
            cost_model=costs, config=simulation_config(cfg),
        )  # fmt: skip
        assert spy.calls > 40 and spy.violations == []
        return result

    original, altered = run(market), run(tampered)
    pd.testing.assert_frame_equal(original.equity.loc[:cut], altered.equity.loc[:cut])
    pd.testing.assert_frame_equal(original.weights.loc[:cut], altered.weights.loc[:cut])
    assert not np.allclose(original.equity["equity"].iloc[-1], altered.equity["equity"].iloc[-1])


def test_features_handed_to_strategies_are_for_the_decision_date_only(market):
    day = market.sessions[300]
    cross_section = market.features_on(day)
    assert cross_section.index.name == "symbol"
    expected = market.features.xs(day, level="trade_date")
    pd.testing.assert_frame_equal(cross_section, expected)
    assert market.features_on(pd.Timestamp("1999-01-01")).empty


# ------------------------------------------------------------------------ runner


def test_report_layout_and_provenance(report, dataset):
    files = {p.name for p in report.path.iterdir()}
    assert {"result.json", "equity.csv", "returns.csv", "walk_forward_returns.csv"} <= files
    for key in ("equal_weight", "buy_and_hold", "momentum", "moving_average", "mean_variance"):
        assert {"trades.csv", "events.csv", "weights.csv"} == {
            p.name for p in (report.path / key).iterdir()
        }

    text = (report.path / "result.json").read_text()
    summary = json.loads(text, parse_constant=lambda token: pytest.fail(f"invalid JSON: {token}"))
    assert summary["run_id"] == report.run_id == report.path.name
    assert summary["dataset"]["dataset_id"] == dataset.dataset_id
    assert summary["dataset"]["is_synthetic"] is True
    assert any("SYNTHETIC" in caveat for caveat in summary["caveats"])
    assert any("does not guarantee future results" in caveat for caveat in summary["caveats"])
    assert summary["period"]["sessions"] == len(pd.read_csv(report.path / "equity.csv"))


def test_report_metrics_are_complete_and_consistent(report):
    strategies = report.summary["strategies"]
    for key, entry in strategies.items():
        metrics = entry["metrics"]
        for name in ("cagr", "total_return", "sharpe", "sortino", "max_drawdown", "calmar",
                     "volatility", "win_rate", "turnover", "total_costs"):  # fmt: skip
            assert name in metrics, (key, name)
        assert -1 <= metrics["max_drawdown"] <= 0
        result = report.results[key]
        assert metrics["final_equity"] == pytest.approx(result.equity["equity"].iloc[-1])
        assert metrics["total_return"] == pytest.approx(
            result.equity["equity"].iloc[-1] / result.initial_capital - 1
        )
        if key == "equal_weight":  # the benchmark is not compared with itself
            assert "alpha" not in metrics and "sharpe_difference_vs_benchmark" not in entry
        else:
            assert {"alpha", "beta", "tracking_error", "information_ratio"} <= set(metrics)
            assert entry["sharpe_difference_vs_benchmark"]["observations"] > 900
    assert (
        strategies["buy_and_hold"]["metrics"]["turnover"]
        < strategies["momentum"]["metrics"]["turnover"]
    )


def test_walk_forward_is_out_of_sample_and_stitched(report, market):
    walk_forward = report.summary["walk_forward"]
    folds = walk_forward["folds"]
    assert len(folds) == 2
    returns = report.walk_forward_returns
    assert returns.index.is_unique and returns.index.is_monotonic_increasing
    for fold in folds:
        inside = returns[returns["fold"] == fold["index"]]
        assert inside.index.min() == pd.Timestamp(fold["test_start"])
        assert inside.index.max() == pd.Timestamp(fold["test_end"])
        assert pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["validation_start"])
        assert pd.Timestamp(fold["validation_end"]) < inside.index.min()
        # Fresh capital each fold: nothing is traded on the fold's first session.
        assert (inside.drop(columns="fold").iloc[0] == 0).all()
    for key, entry in walk_forward["strategies"].items():
        assert len(entry["folds"]) == 2
        assert entry["out_of_sample"]["sessions"] == len(returns)
        if key != "equal_weight":
            assert "beta" in entry["out_of_sample"]


def test_runs_are_reproducible(report, dataset, tmp_path):
    again = run_backtest(make_config(dataset.path), output_root=tmp_path)
    assert again.run_id == report.run_id
    assert again.summary["strategies"] == report.summary["strategies"]
    assert again.summary["walk_forward"] == report.summary["walk_forward"]
    for key, result in report.results.items():
        pd.testing.assert_frame_equal(result.equity, again.results[key].equity)
    assert [p.name for p in tmp_path.iterdir()] == [report.run_id]  # no staging leftovers


def test_run_id_changes_with_config_but_not_with_output_location(report, dataset, tmp_path):
    moved = make_config(dataset.path, output_root=str(tmp_path / "elsewhere"))
    assert moved.config_hash() == make_config(dataset.path).config_hash()
    changed = make_config(dataset.path, rebalance="weekly", walk_forward={"enabled": False})
    other = run_backtest(changed, output_root=tmp_path)
    assert other.run_id != report.run_id
    assert other.summary["walk_forward"]["folds"] == []
    assert not (other.path / "walk_forward_returns.csv").exists()


def test_cost_schedule_must_cover_the_period(dataset, tmp_path):
    late = {**SHIPPED["costs"], "statutory": SHIPPED["costs"]["statutory"][-1:]}  # from Oct 2024
    with pytest.raises(ValueError, match="No statutory cost rates configured"):
        run_backtest(make_config(dataset.path, costs=late), output_root=tmp_path)


def test_invalid_configs_are_rejected(dataset):
    with pytest.raises(ValidationError, match="unknown strategy"):
        make_config(dataset.path, strategies=[{"strategy": "crystal_ball"}])
    with pytest.raises(ValidationError, match="benchmark"):
        make_config(dataset.path, benchmark="nifty50_tri")
    with pytest.raises(ValidationError, match="unique"):
        make_config(
            dataset.path, strategies=[{"strategy": "equal_weight"}, {"strategy": "equal_weight"}]
        )
    with pytest.raises(ValidationError):
        make_config(dataset.path, rebalance="hourly")
    with pytest.raises(ValidationError):
        make_config(dataset.path, leverage=2.0)  # unknown keys are refused


def test_shipped_config_is_valid():
    cfg = BacktestConfig.model_validate(SHIPPED)
    assert cfg.benchmark == "equal_weight"
    assert {s.key for s in cfg.strategies} >= {
        "buy_and_hold", "equal_weight", "momentum", "moving_average", "mean_variance",
    }  # fmt: skip
