# Implementation checklist

Living record of progress against [prd.md](../prd.md). An item is ticked only after it
has been **verified** — the "Evidence" column says how. Update this file, and the module
registry in [backend/app/core/modules.py](../backend/app/core/modules.py), whenever a
phase item is completed.

Legend: `[x]` done and verified · `[~]` partly done (see note) · `[ ]` not done

_Last updated: 2026-11-12_

## Verification snapshot (2026-11-12)

| Check | Result |
|---|---|
| `uv run pytest` | 487 passed, 3 skipped. The skipped ones are the PostgreSQL tests: migrations 0003–0005 and the new tables have run on SQLite only |
| `uv run ruff check .` | clean |
| Frontend `tsc --noEmit`, `eslint`, `next build` | clean; 9 static routes |
| Real dataset build (Yahoo, 20 NSE large caps, 2015 → 2026-10-30) | 58,379 bars, 344 dividends, 41 rows quarantined (placeholder / non-session bars), survivorship flagged |
| Baseline backtest on real data | runs; equal weight 14.7% CAGR, Sharpe 0.55 (survivorship-biased universe, 6.5% risk-free) |
| RL environment vs backtest simulator, same policy, real costs | identical to 4 decimals (11.94% vs 11.94%) |
| Research experiments on real data | <!-- experiments:start -->full run (3 folds × 3 seeds) in progress when this was written; see `artifacts/experiments/`<!-- experiments:end --> |
| LLM engine, live against local `qwen2.5:3b` | 24-case evaluation and a real 14-article run; see [llm-engine.md](llm-engine.md) |
| API + dashboard on SQLite with real data | market, regime, news, allocation, risk, stress test, paper trading verified by request; overview, news, scenarios, paper pages viewed in headless Edge |
| Docker Compose stack | verified through Phase 2 on 2026-11-11. **Not re-run since** — Docker Desktop was not running when the model engines were finished |

Not verified: OpenAI as the LLM provider (no key); the Compose stack and the worker
jobs with the model engines; the GitHub Actions workflow (no remote); clicking through
the dashboard by hand; the cost-schedule rates against official circulars.

---

## Section 25 — Project bootstrap

All thirteen items done and verified on 2026-11-11 (repository, Python environment,
FastAPI, PostgreSQL, Redis, Next.js, Docker Compose, migrations, health check,
dashboard shell, tests, documentation).

## Phase 1 — Foundation and data infrastructure

| Deliverable | Status | Evidence / note |
|---|---|---|
| Market data provider interface | [x] | `synthetic_dev`, `csv_directory`, `yahoo_chart` |
| Historical data ingestion | [x] | Real NSE data from Yahoo Finance (the project owner's choice of a free source); cached raw responses; idempotent database load |
| Data validation | [x] | 9 hard checks incl. placeholder bars found in the real data; quality gates |
| Corporate-action adjustment | [~] | Splits, bonus, dividends implemented and tested. With Yahoo only dividends are applied (its prices are already split-adjusted); rights issues and demergers are not modelled |
| Trading calendar, symbol mapping | [x] | Calendar inferred from traded bars; zero-volume filler days excluded |
| Survivorship-bias controls | [~] | Point-in-time universe by listing window. **The Yahoo universe has no delisted names and no index-membership history**, so it is survivorship-biased — flagged on every build |
| Point-in-time validation, features, versioning | [x] | Three look-ahead tests; 32 scale-free features; content-addressed snapshots |

Not started: fundamentals (PRD 5.2), macroeconomic data (5.4), `/market/overview`.

## Phase 2 — Quantitative baseline and backtesting

| Deliverable | Status | Evidence / note |
|---|---|---|
| Simulator, cost model, liquidity, rebalancing | [x] | Hand-computed tests; accounting identities. **Cost rates are indicative** (B6) |
| Benchmark strategies | [~] | Five baselines. NIFTY 50 TRI needs index data (B1) |
| Metrics, walk-forward, look-ahead tests, bootstrap | [x] | All eleven PRD metrics; embargoed folds; paired block bootstrap |
| Asynchronous backtest jobs + API | [x] | Verified on the Compose stack on 2026-11-11 |
| Market regime detection (PRD 6.3) | [x] | Rule-based, causal; labels the 2016, 2018, 2020 and 2022 stress episodes; stability reported. GMM / HMM alternatives not implemented |

## Phase 3 — LLM intelligence engine

| Deliverable | Status | Evidence / note |
|---|---|---|
| Provider-agnostic LLM client, structured output | [~] | One OpenAI-compatible client. Verified against local Ollama; **OpenAI (`gpt-6-luna`) unverified — no key** |
| News ingestion, de-duplication | [~] | Yahoo headlines (recent only, no bodies) and user JSONL. **No historical news archive** |
| Event extraction, sentiment, entity mapping, impact | [x] | Schema per PRD 8.3; live run on real headlines |
| Hallucination controls | [x] | Schema, entity, timestamp, source and grounding checks; rejections recorded with reasons |
| Prompt-injection defence | [~] | Article isolated as data and a heuristic flags all three test attempts — but **the small local model obeyed two of three** |
| Signal aggregation, point-in-time | [x] | Event-type half-lives; future news cannot change past features (tested) |
| Summaries, retrieval | [~] | Implemented and unit-tested; not exercised live |
| Evaluation suite | [x] | 24 fictional cases; measured: sentiment 85%, event type 65%, schema validity 100% |
| News API + storage | [x] | PRD 15.2 endpoints; 14 articles / 16 events loaded and served |

**Exit criterion — "structured, timestamped and traceable financial intelligence":**
met for recent headlines with a local model. Not met for a production LLM or for
historical news.

## Phase 4 — GAN engine

| Deliverable | Status | Evidence / note |
|---|---|---|
| TimeGAN baseline (PyTorch) | [x] | Trains on real data; deterministic per seed |
| Conditional / regime-conditioned generation | [x] | One-hot regime condition; regime Markov chain for long paths |
| Valid synthetic OHLCV; enters the feature pipeline | [x] | Valid by construction; synthetic markets run through the real feature engine |
| Statistical evaluation (PRD 7.5, all eight) | [x] | Against held-out real windows, with a real-vs-real baseline |
| Memorisation / nearest-neighbour check | [x] | Detects a replaying generator in tests; real generators show no copying |
| Stress scenarios, scenario features | [x] | Used by the risk report and the RL observation |

**Exit criterion — "synthetic sequences that pass statistical quality and memorization
checks": NOT MET.** The generators pass the memorisation and return-distribution checks
and fail the correlation and discriminative checks; their stress scenarios are far
milder than real crises. They are registered as `rejected` and every use is labelled.

## Phase 5 — RL portfolio optimization

| Deliverable | Status | Evidence / note |
|---|---|---|
| Gymnasium environment | [x] | Passes `check_env`; matches the simulator for a fixed policy |
| PPO baseline, continuous allocation | [x] | Stable-Baselines3; SAC selectable but not evaluated |
| Reward function (three formulations) | [x] | Hand-computed tests; ablation in the experiments |
| Risk constraints, never bypassed | [x] | Independent engine; 400-case property test |
| Training pipeline, checkpoints | [x] | Validation-selected checkpoint; reproducible per seed |
| Model registry | [x] | JSON model cards with PRD 16.1 fields (MLflow not used — B2) |
| Hyper-parameter tuning (Optuna) | [ ] | Not done: configuration values are untuned defaults |

**Exit criterion — "a reproducible RL strategy that outperforms or provides useful
evidence against simple baselines on genuinely unseen data":** see the experiment
results in the verification snapshot.

## Phase 6 — Unified intelligence architecture

| Deliverable | Status | Evidence / note |
|---|---|---|
| Feature fusion: early, late, attention | [x] | Unit-tested; attention weights exposed |
| Unified market state | [x] | One builder shared by training, backtests and live allocation |
| GAN + LLM + RL integration | [~] | GAN ↔ RL integrated and run. LLM features are wired in but **cannot be used in experiments** (no historical news) |
| Ablation experiments, multi-seed, significance | [x] | E0–E6 + three ablations; every seed reported |
| Explainability engine | [x] | Narrative built from the model's own numbers; sensitivities labelled as associations |

## Phase 7 — Dashboard and paper trading

| Deliverable | Status | Evidence / note |
|---|---|---|
| Overview, Market, News, GAN, Optimizer, Backtesting, Experiments, Paper pages | [~] | All eight read live API data. **Tables only — no charts.** No regime timeline, sector view or correlation matrices |
| Paper trading | [x] | Replay through the backtest simulator; baseline, RL and manual accounts; orders risk-checked; never sent anywhere |
| Risk monitoring | [x] | VaR, expected shortfall, concentration, limit checks, historical and generated stress tests; alerts |

## Phase 8 — Production hardening

| Deliverable | Status | Evidence / note |
|---|---|---|
| Authentication and authorisation | [ ] | **None.** The API must not be exposed beyond localhost |
| Rate limiting | [~] | Caps on concurrent backtests, model jobs and LLM analyses only |
| Monitoring | [~] | `/system/alerts` (stale data, rejected generator, failed jobs). No Prometheus / Grafana, no drift detection |
| CI/CD | [~] | Workflow written, never run |
| Cloud deployment, object storage, performance tests, security review | [ ] | Not done |
| Routine ingestion and inference without manual steps | [ ] | Data and models are refreshed by running scripts |

---

## Acceptance criteria (PRD §22)

**Data** — [x] ingestion · [~] corporate actions (see Phase 1) · [x] quality validation ·
[x] reproducible features · [~] point-in-time controls (no index-membership history)

**GAN** — [x] independent training · [x] generation · [x] statistical evaluation ·
[x] regime conditioning · [x] memorisation checks (quality gates **failed**)

**LLM** — [x] news processed · [x] entities extracted · [x] structured events ·
[x] source attribution · [x] invalid output rejected or flagged (verified with a local
model, not OpenAI)

**RL** — [x] environment · [x] PPO training · [x] continuous allocation ·
[x] transaction costs · [x] risk constraints · [x] checkpoints

**Integration** — [x] GAN output in the feature pipeline · [~] LLM output in the feature
pipeline (code path tested; no historical data to feed it) · [x] unified observations ·
[x] allocations pass the risk engine · [x] decisions stored and inspectable

**Backtesting** — [x] benchmarks · [x] walk-forward · [x] look-ahead checks ·
[x] configurable costs · [x] risk-adjusted metrics

**Frontend** — [~] dashboard (no charts) · [x] experiments can be monitored ·
[x] backtests can be launched · [x] allocation recommendations visible · [x] paper
trading without real-money execution

---

## Known blockers and decisions needed

**B1 — Data limits of the free source.** Yahoo gives split-adjusted prices for current
listings only: no delisted names, no index-membership history, no NIFTY 50 TRI, no
fundamentals, and news as recent headlines only. Fixing survivorship bias or building
historical news features needs a licensed source.

**B2 — Application-control policy on the development machine.** Unsigned native code is
blocked. uv's standalone Python and pyarrow are affected (hence CSV snapshots and no
MLflow); PyTorch, scikit-learn and SciPy are not.

**B3 — No GPU.** Training budgets are small (PPO 40,000 steps, TimeGAN 600 iterations
per phase). Better models need more compute.

**B4 — OpenAI key.** Put `AETHER_LLM_API_KEY` in `.env`, then run
`scripts/evaluate_llm.py` to measure the configured model before trusting it.

**B5 — Memory headroom.** 16 GB RAM, usually 1–3 GB free.

**B6 — Cost schedule and risk-free rate need review.** `configs/backtesting.yaml` ships
indicative rates; check them against current NSE, SEBI and broker schedules.

**B7 — Docker stack unverified with the model engines.** Start Docker Desktop and run
`docker compose up -d --build`; the worker image now includes PyTorch.

## Deviations from the PRD (deliberate)

| PRD | What was done | Why |
|---|---|---|
| Licensed data source | Yahoo Finance, unofficial | Project owner's instruction: use a free source |
| Raw prices + all corporate actions | Yahoo's split-adjusted prices + dividends | The source offers nothing else |
| Parquet / object storage | CSV snapshots on local disk | pyarrow blocked (B2) |
| MLflow, DVC | JSON model cards; manifest-based dataset versioning | MLflow needs pyarrow (B2); no extra tooling |
| Optuna tuning | None | Compute and time; values are untuned |
| LangChain / LangGraph, vector database | Direct HTTP client; small on-disk store | Not needed at this size |
| TimescaleDB, pgvector | Plain PostgreSQL 16 | Not beneficial at this size |
| Prometheus, Grafana, S3 | Not added | Phase 8 not done |
| NIFTY 50 TRI benchmark | Equal-weight universe portfolio | No index data (B1) |
| Charts in the dashboard | Tables only | Time; the data behind every chart is served by the API |
| Features as indicator levels | Scale-free forms | Level features leak future corporate actions through back-adjustment |
| GMM / HMM regime models | Rule-based classifier | Transparent, causal, nothing to over-fit |
