# Model engines: regimes, risk, RL, GAN, fusion, explainability, paper trading

Module documentation for `ml/features/regime.py`, `ml/risk`, `ml/rl`, `ml/gan`,
`ml/fusion`, `ml/registry.py` and `ml/backtesting/paper.py`, in the format required by
PRD section 24 (Rule 6). The LLM engine has its own document:
[llm-engine.md](llm-engine.md). Data and backtesting:
[data-pipeline.md](data-pipeline.md), [backtesting.md](backtesting.md).

How the pieces connect:

```text
dataset snapshot ──► MarketData ──► ObservationBuilder ──► RL policy ──► risk engine ──► simulator
   (bars, features,      │              ▲      ▲                             │
    regimes)             │              │      └── news features (LLM)       └─► explanation
                         └─► TimeGAN ───┴── scenario features, synthetic training markets
```

---

## 1. Market regime detection — `ml/features/regime.py`

**Purpose.** Label each session bull / bear / sideways / high-volatility /
low-volatility / market-stress, as an input to the generator, the agent and the
dashboard.

**Method.** A rule system on three causal measurements of the equal-weight universe
return: the expanding percentile of 21-session realised volatility, the 63-session
return with its Kaufman efficiency ratio, and the drawdown from the running peak. A
new label must hold for `min_duration` sessions before the regime switches (stress is
entered immediately).

**Input.** Daily market-proxy returns. **Output.** One row per session: `regime`,
`confidence`, `volatility_level`, `trend_strength`, `realized_vol`, `trend_return`,
`drawdown`; written to `regimes.csv` in every dataset snapshot.

**Configuration.** `regime:` section of the data config (`RegimeConfig`).

**Evaluation.** `regime_stability()` (in each manifest): on the NSE dataset, 9.4
switches a year, mean run 26 sessions, 96% day-to-day persistence; the stress label
covers Feb–Mar 2016, Oct–Nov 2018, 9 Mar – 22 Jun 2020, and episodes in 2022 and 2026.
Tests assert causality by truncation and by corrupting the future.

**Limitations.** Thresholds are judgement calls, not estimated. `confidence` is a
margin score, not a probability. One market-wide regime. The statistical alternatives
in the PRD (GMM, HMM, k-means) are not implemented.

---

## 2. Risk engine — `ml/risk`

**Purpose.** Deterministic controls that no model can bypass (PRD 12), and risk
analytics.

**Limits** (`RiskLimits`): maximum asset weight, maximum sector weight, optional
concentration (HHI) cap, minimum cash, minimum liquidity, drawdown threshold with an
exposure cap, maximum turnover per rebalance.

**Enforcement** (`enforce`). A non-finite allocation is rejected. Otherwise hard limits
are applied in a fixed order, with removed weight going to cash; the turnover budget
then limits how far the portfolio moves from compliant holdings towards the compliant
target. The result is re-validated; if it still breaks a hard limit the decision is
rejected and holdings are kept. Every decision returns a record of what was violated
and what was applied — the risk log.

**Analytics** (`analytics.py`): historical VaR and expected shortfall, portfolio
volatility, correlation matrix, sector concentration, rolling beta, drawdown, and
stress losses on the worst historical windows and on GAN stress scenarios.

**Where it sits.** `RiskManagedStrategy` wraps any strategy, so the simulator only ever
receives compliant weights. Training uses `FastRiskProjector`, a NumPy re-implementation
tested for equality with `enforce`.

**Evaluation.** `tests/ml/test_risk.py`: each limit by hand; 400 random allocations
always end compliant; the fast projector matches the reference to 1e-12.

**Limitations.** Limits act on weights at decision time; positions drift until the next
rebalance, so a limit can be exceeded slightly between decisions (the risk report shows
it). VaR is historical and understates risk when the window has no crisis. Correlations
are reported, not used by the limits.

---

## 3. RL portfolio optimizer — `ml/rl`

**Purpose.** Learn target weights from the unified market state.

**Observation** (`ObservationBuilder`). Groups: `market` (N assets × z-scored technical
features, scaler fitted on the training window only), `regime` (9 values), `news` and
`scenario` (when available), `portfolio` (weights, cash, drawdown, tradable mask).

**Action.** One number per asset plus cash in [-1, 1] → masked softmax → long-only
weights that sum to one → risk engine.

**Environment** (`PortfolioEnv`, Gymnasium). One step is one rebalance: decide at the
close of T, trade at the open of T+1 with per-order costs and slippage, then hold for
`rebalance_every` sessions. For a fixed policy it reproduces the backtest simulator to
within rounding (`tests/ml/test_rl_env.py`), which is what makes training and
evaluation consistent.

**Reward.** `log_return`, `risk_adjusted` (PRD formula: return − λ₁·volatility −
λ₂·drawdown − λ₃·turnover) or `drawdown_change`. The lambdas are configuration, and the
`A_no_risk_penalty` experiment measures their effect.

**Algorithms.** PPO (default) and SAC through Stable-Baselines3. DQN is not included:
the action space is continuous.

**Training** (`training/train.py`). Fixed seed; every `eval_every` steps the
deterministic policy is run through the validation window and the checkpoint with the
best validation Sharpe is kept. The test window is never touched during training.

```bash
uv run python scripts/train_models.py rl --latest        # deployment model
uv run python scripts/run_experiments.py --latest        # walk-forward evidence
```

**Evaluation.** A trained policy is wrapped as a strategy and backtested in the same
simulator, with the same costs and risk limits, as every baseline.

**Configuration.** `rl:` section of [configs/experiments.yaml](../configs/experiments.yaml).

**Limitations.** CPU only, small step budgets. **No hyper-parameter search** (the PRD's
Optuna step): the values are untuned defaults. The training window holds only a few
hundred weekly decisions, so over-fitting is the expected failure mode. The deployment
model has no out-of-sample test of its own.

---

## 4. GAN scenario engine — `ml/gan`

**Purpose.** Generate alternative market paths for training augmentation, stress
testing and scenario exploration. Scenarios are not forecasts and never replace
out-of-sample tests on real data.

**Model.** Conditional TimeGAN in PyTorch (embedder, recovery, generator, supervisor,
discriminator; GRU networks). The condition is the one-hot market regime of the window.

**Variables per asset and session:** overnight gap, open-to-close move, wick, and
traded value relative to its recent median. OHLCV is derived from them so that
`low ≤ open, close ≤ high` and `volume ≥ 1` hold by construction. A full synthetic
market is passed through the real feature engine and regime detector, so it can be used
wherever a real one can.

**Calibration.** After training, generated values are rescaled per regime to the mean
and standard deviation of the training windows of that regime. This corrects scale
only; it does not add structure the generator failed to learn.

**Training.**

```bash
uv run python scripts/train_models.py gan --latest
```

**Evaluation** (`evaluation/metrics.py`), against held-out real windows: return and
volatility distributions (KS, Wasserstein), autocorrelation of returns and absolute
returns, cross-asset correlation, tail risk, max-drawdown distribution, a
real-vs-synthetic classifier, and a nearest-neighbour memorisation check. Each measure
is shown next to the same distance between two real periods. Quality gates decide
whether a generator is registered as `validated` or `rejected`.

**Current result — the generators fail their gates.** On the NSE dataset (trained
through 2021, held-out 2022):

| Check | Value | Needed | |
|---|---|---|---|
| Return distribution KS | 0.09 | ≤ 0.10 | pass |
| Memorisation ratio | 0.99 | ≥ 0.60 | pass |
| Near-copies of training data | 0% | 0% | pass |
| Cross-asset correlation error | 0.25 | ≤ 0.15 | **fail** |
| Real-vs-synthetic classifier | 100% accuracy | ≤ 75% | **fail** |

The scale of returns is right and nothing is memorised, but the generator does not
reproduce co-movement or volatility clustering: a classifier separates real from
synthetic windows perfectly, and "market-stress" scenarios lose at most about 3% over
24 sessions where the real 2020 window lost 33%. Results that use these scenarios show
what *this* generator does.

**Limitations.** Temporal dependence only inside a 24-session window (long paths are
stitched). Small data (about 1,100 training windows) and a small CPU-trained network.
Assets missing more than 5% of the period are not modelled.

---

## 5. Fusion and experiments — `ml/fusion`

**Fusion** (`extractors.py`). `early`: the policy reads the flat observation. `late`:
one encoder per group, embeddings concatenated. `attention`: encoders combined by a
learned, input-dependent softmax whose weights are exposed for explanations.

**Experiments** (`experiments.py`, PRD 20). E0 equal weight, E1 momentum, E2 RL,
E3 RL+LLM, E4 RL+GAN, E5 RL+GAN+LLM, E6 with attention fusion, and three ablations.
Chronological walk-forward folds; the generator is trained per fold on that fold's
training window; all strategies share dates, costs and risk limits; several seeds, all
reported; block-bootstrap intervals for the Sharpe difference against E0.

```bash
uv run python scripts/run_experiments.py --latest
```

**Output.** `artifacts/experiments/<run_id>/result.json` and `returns.csv`; models and
their cards under `artifacts/models/`.

**Limitations.** E3 and E5 cannot run: there is no historical news archive, so there
are no point-in-time news features. E6 and the ablations therefore run without LLM
features and say so. No multiple-testing correction (Deflated Sharpe) is applied across
the configurations.

---

## 6. Explainability — `ml/fusion/explain.py`

For one allocation: the regime, the proposed weights and the changes, the model and
dataset ids, the raw policy output, the risk engine's record, feature values for the
largest positions, the policy's action spread, attention weights when available, and
an occlusion sensitivity per information source.

The narrative is assembled from those numbers — no language model writes it. The
sensitivities are labelled as statistical associations for this model on this date,
not causal explanations. Every allocation is stored with a timestamp.

---

## 7. Model registry — `ml/registry.py`

A `model_card.json` beside each model's artifacts with the PRD 16.1 fields plus seed,
config hash, training window, lineage and environment. Status is `candidate`,
`validated` or `rejected`; rejected models stay on record. This stands in for MLflow,
which depends on a library the development machine's security policy blocks.

---

## 8. Paper trading — `ml/backtesting/paper.py`

A paper account is a description (capital, start date, and a baseline strategy, an RL
model, or hand-entered orders). Its state is recomputed by replaying the backtest
simulator from the start date to the latest session in the loaded data, so paper
results use the same costs, timing and risk limits as backtests. Orders fill at the
next session's open; an order placed on the latest session stays pending until newer
data is loaded. **There is no broker integration anywhere in the platform.**

**Limitations.** The account advances only when a newer dataset is built and loaded —
there is no scheduled ingestion yet. One active account at a time.
