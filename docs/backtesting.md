# Backtesting engine (Phase 2)

Module documentation for `ml/backtesting`, in the format required by PRD section 24
(Rule 6).

## Purpose

Simulate portfolio strategies on historical data under realistic trading assumptions,
and compare them fairly. It provides the rule-based baselines the RL agent (Phase 5)
has to beat, and the evaluation harness every later model will be judged with.

```text
dataset snapshot ─► MarketData ─► simulate(strategy) ─► metrics ─► result.json
                                       ▲
              cost model, rebalance schedule, walk-forward folds
```

| File | Responsibility |
|---|---|
| `data.py` | Read-only market view built from a verified dataset snapshot |
| `costs.py` | Dated Indian delivery-equity cost model |
| `simulator.py` | Daily portfolio simulator with cash accounting |
| `strategies.py` | Strategy interface and the five baselines |
| `metrics.py` | Performance and risk metrics |
| `walk_forward.py` | Rebalance schedules and chronological folds |
| `stats.py` | Block-bootstrap significance tests |
| `runner.py` | Orchestration and result artifacts |
| `config.py` | Typed configuration |

## Input schema

A dataset snapshot directory produced by `scripts/build_dataset.py`
(see [data-pipeline.md](data-pipeline.md)). The snapshot is hash-verified before use.
From it the engine reads adjusted open/close, traded value, the securities master
(listing windows) and the feature table.

A strategy receives a `StrategyContext` for each decision date `T`:

| Field | Content |
|---|---|
| `date` | The decision date `T` |
| `universe` | Symbols listed on `T` that traded on `T` |
| `prices` | Adjusted closes for all sessions up to and including `T` |
| `features` | Feature cross-section for `T` |
| `current_weights` | End-of-day weights, including `CASH` |

and returns a `pandas.Series` of target weights (≥ 0, sum ≤ 1; the rest is cash), an
empty Series for "all cash", or `None` for "do not trade".

## Output schema

`artifacts/backtests/<run_id>/`:

| File | Contents |
|---|---|
| `result.json` | Config, dataset provenance, per-strategy metrics, folds, caveats, environment |
| `equity.csv`, `returns.csv` | Full-period series, one column per strategy |
| `walk_forward_returns.csv` | Stitched out-of-sample daily returns with the fold index |
| `<strategy>/trades.csv` | Every fill: shares, price, value, each cost component, slippage, realised P&L |
| `<strategy>/events.csv` | Partial fills, untradable orders, stale prices, delistings |
| `<strategy>/weights.csv` | End-of-day weights |

`run_id = <name>-<hash of config + dataset content hash>`: the same experiment on the
same data always gets the same id.

Metrics reported: total return, CAGR, volatility, Sharpe, Sortino, maximum drawdown,
Calmar, turnover, trade win rate, daily hit rate, total costs, and — versus the
benchmark strategy — alpha, beta, tracking error and information ratio. Definitions are
in the docstring of [metrics.py](../ml/backtesting/metrics.py). An undefined metric is
`null`, never `0`.

## Configuration

[configs/backtesting.yaml](../configs/backtesting.yaml), validated by
[config.py](../ml/backtesting/config.py). Unknown keys are rejected.

| Key | Meaning |
|---|---|
| `dataset`, `start_date`, `end_date` | What to test on |
| `initial_capital`, `rebalance`, `execution_price` | Portfolio and schedule |
| `participation_limit`, `min_trade_value`, `allow_fractional_shares`, `delisting_recovery` | Trading frictions |
| `risk_free_rate` | Annual rate for Sharpe / Sortino / alpha — **an assumption you must set** |
| `costs` | Slippage, brokerage, and the dated statutory schedule |
| `strategies`, `benchmark` | What to run and what to compare against |
| `walk_forward` | Train / validation / test window lengths, step, embargo |
| `bootstrap` | Samples, block length, confidence, seed |

### Cost model

Per order on traded value `V`: brokerage `min(max(rate·V, min), max)`; STT; exchange
transaction charge; SEBI fee; stamp duty; GST on (brokerage + exchange + SEBI fee); an
optional flat charge per sell order. Rates come from a dated schedule and the entry in
force on the trade date is applied. Slippage is an adverse move of the fill price.

> The shipped rates are **indicative**. They capture the large items (STT 0.1% each
> side, stamp duty 0.015% on buys from July 2020, GST 18%) but not every small revision
> of exchange and SEBI charges, and pre-2020 stamp duty was state-specific. Verify them
> against current NSE, SEBI and broker schedules before relying on results.

## Execution model

| Rule | Behaviour |
|---|---|
| Timing | Decide at the close of `T`; execute at the open of `T+1` (configurable: close) |
| Order of fills | Sells first, then buys |
| Funding | Buys are scaled down together if cash cannot cover value + costs; no leverage, no shorting |
| No bar on execution day | Order is not placed; logged; not carried forward |
| Liquidity | Each order capped at `participation_limit` × that day's traded value; remainder dropped and logged |
| Missing bar while held | Position valued at last close; logged as `stale_price` |
| Delisting | Position closed at `delisting_recovery` × last close on the delisting date |
| Dividends | Reinvested (total-return adjusted prices); untaxed |
| Cash | Earns no interest |

## Strategies

Rule-based baselines — not AI.

| Name | Rule |
|---|---|
| `buy_and_hold` | Equal weights once, then never trade |
| `equal_weight` | Rebalance to 1/N over the point-in-time universe |
| `momentum` | Hold the `top_k` assets by return over `lookback` sessions, skipping the last `skip` |
| `moving_average` | Hold each asset's 1/N slot only while its close is above its `window` SMA |
| `mean_variance` | Long-only Markowitz on trailing returns with a weight cap; `objective: min_variance` drops the mean |

Add one by subclassing `Strategy` and registering it in `STRATEGIES`.

## Running it

There is nothing to train in this phase; `Strategy.fit` exists for Phase 5.

```bash
uv run python scripts/run_backtest.py --latest            # newest dataset, default config
uv run python scripts/run_backtest.py --dataset artifacts/datasets/<dataset_id>
```

Through the API (asynchronous; needs Redis and a worker):

```bash
curl -X POST localhost:8000/api/v1/backtest/run -H 'Content-Type: application/json' \
     -d '{"name": "weekly-test", "rebalance": "weekly"}'        # -> 202 {"run_id": ...}
curl localhost:8000/api/v1/backtest/runs/<run_id>               # status, config, results
curl localhost:8000/api/v1/backtest/runs/<run_id>/metrics
```

or from the dashboard's **Backtesting Lab** page.

## Evaluation methodology

- **Full period** — every strategy over the whole window with identical data, costs and
  execution rules.
- **Walk-forward** — chronological folds (`train | validation | test`). A strategy is
  fitted on `train` only and simulated with fresh capital on the unseen `test` window;
  the test windows are stitched into one out-of-sample series. `validation` is reserved
  for hyper-parameter selection. An embargo drops sessions at the start of validation
  and test windows.
- **Significance** — a paired circular block bootstrap gives a confidence interval for
  the Sharpe difference between a strategy and the benchmark. A difference whose
  interval contains zero is reported as *not significant*.
- **The test set is never used for tuning.** The baselines have no fitted parameters;
  their window lengths are stated in the config, not searched.

## Verification

```bash
uv run pytest tests/ml -k backtest
```

| Test file | Verifies |
|---|---|
| `test_backtest_costs.py` | Every charge against hand-computed values; dated schedule; no silent defaults |
| `test_backtest_simulator.py` | Execution timing, cash accounting, costs, slippage, liquidity cap, missing bars, delisting, integer shares — on tiny hand-computed markets |
| `test_backtest_strategies.py` | Each baseline against hand-computed selections; optimiser against the closed form and optimality conditions |
| `test_backtest_metrics.py` | Each metric by hand; bootstrap finds a real edge and does not flag noise |
| `test_backtest_engine.py` | Accounting identities for every strategy; every fill on a real bar inside the listing window; **scrambling all prices after a cut-off leaves earlier equity and weights bit-identical**; folds; reproducible run ids |

## Known limitations

- **Results so far are on synthetic development data** and carry no information about
  real markets. Every report says so.
- **Benchmark:** NIFTY 50 TRI needs licensed index data. Until then strategies are
  compared with the equal-weight universe portfolio.
- **Liquidity cap uses the execution day's total traded value**, which is only known
  after the close — the common, slightly optimistic, convention.
- Unfilled orders are dropped, not worked over following days. Market impact is a flat
  slippage assumption, not a function of order size.
- Share counts are in adjusted units; values and costs are exact, but lot-level realism
  (odd lots, circuit limits, auction sessions) is not modelled.
- Delisting at 100% of the last close is optimistic; set `delisting_recovery` lower for
  a conservative view.
- Dividends are reinvested untaxed; capital-gains tax and interest on cash are ignored.
- Daily bars only: no intraday execution, no partial-day effects.
- No Deflated Sharpe Ratio or other multiple-testing correction yet. That becomes
  necessary as soon as many configurations are compared (Phase 5 hyper-parameter search).
- Purged splits for overlapping supervised labels are not needed by the baselines and
  are not implemented; the embargo is.
- A worker that dies mid-job leaves its run in `running`; there is no automatic recovery.
- Market regime detection (PRD 6.3) is not built yet.
