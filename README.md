# Project AETHER

An integrated GAN + Reinforcement Learning + LLM framework for quantitative market
intelligence and portfolio optimization on the National Stock Exchange of India (NSE).

AETHER is a **research and decision-support platform**. It is not a SEBI-registered
investment adviser or research analyst, it does not give investment advice, and it
never places real orders — there is no broker integration. Historical and simulated
performance does not guarantee future results.

The full specification is in [prd.md](prd.md). Progress against it is tracked, item by
item, in [docs/CHECKLIST.md](docs/CHECKLIST.md).

## Status

| Phase | Scope | Status |
|---|---|---|
| Bootstrap | Repo, FastAPI, PostgreSQL, Redis, Next.js, Docker Compose, migrations, tests | Done |
| 1 | Data: provider interface, ingestion, validation, features | Done — real NSE data through Yahoo Finance (free, unofficial) |
| 2 | Baselines, backtesting, market regimes | Done |
| 3 | LLM news intelligence | Built; verified with a small local model only — **OpenAI not yet verified** (no key); headlines only, no news history |
| 4 | GAN scenario engine | Built; **the trained generators fail their quality gates** |
| 5 | RL portfolio optimizer, risk engine, registry | Built; untuned hyper-parameters |
| 6 | Fusion, experiments E0–E6, explainability | Built; experiments that need news features cannot run |
| 7 | Dashboard, paper trading | Eight pages on live data (tables, no charts); replay-based paper trading |
| 8 | Production hardening | **Not done**: no authentication, no cloud deployment, no monitoring stack |

What the research shows so far is in [Results](#results). The short version: the
platform works end to end, and it does **not** show that the three-model system beats
simple strategies.

## Requirements

| Tool | Version | Notes |
|---|---|---|
| Python | 3.12+ | Use the python.org installer on Windows (see [Windows notes](#windows-notes)) |
| [uv](https://docs.astral.sh/uv/) | 0.5+ | Dependency and environment manager; installs CPU-only PyTorch |
| Node.js | 20.9+ (24 tested) | For the dashboard |
| Docker Desktop | recent | Optional: PostgreSQL, Redis, background workers |

## Quick start (no Docker)

Everything except background jobs works with a SQLite file and no other services.

```bash
uv sync

# 1. Real NSE data (Yahoo Finance, no key) -> validated, versioned dataset
uv run python scripts/build_dataset.py --config configs/data_nse.yaml

# 2. Database
export AETHER_DATABASE_URL="sqlite+aiosqlite:///./aether_dev.db"   # PowerShell: $env:AETHER_DATABASE_URL = "..."
uv run alembic -c backend/alembic.ini upgrade head
uv run python scripts/load_dataset.py --latest

# 3. Baselines, models, experiments (each writes under artifacts/)
uv run python scripts/run_backtest.py --latest          # rule-based baselines
uv run python scripts/train_models.py rl --latest       # allocation model (minutes, CPU)
uv run python scripts/train_models.py gan --latest      # scenario generator
uv run python scripts/run_experiments.py --latest       # E0-E6 + ablations (about 2 hours, CPU)

# 4. News intelligence (needs an OpenAI-compatible LLM; see "LLM" below)
uv run python scripts/analyze_news.py --max-articles 20
uv run python scripts/load_news.py --latest

# 5. API (http://localhost:8000/docs) and dashboard (http://localhost:3000)
uv run uvicorn app.main:app --reload
cd frontend && npm install && npm run dev
```

Without Redis, `GET /api/v1/health/ready` reports `degraded` and endpoints that queue
background jobs return `503`; use the scripts above instead.

## Quick start (Docker)

```bash
docker compose up -d --build        # postgres, redis, migrations, API, worker, dashboard
docker compose exec backend python scripts/build_dataset.py --config configs/data_nse.yaml
docker compose exec backend python scripts/load_dataset.py --latest
```

Ports are published on `127.0.0.1` only. The Compose stack was verified through
Phase 2; **it has not been re-run since the model engines (and PyTorch) were added.**

## Data

`configs/data_nse.yaml` builds a dataset of 20 NSE large caps from Yahoo Finance's
chart API. Read [ml/data/ingestion/yahoo.py](ml/data/ingestion/yahoo.py) before relying
on it:

- it is an unofficial endpoint, for personal research use, not redistribution;
- prices are already split-adjusted by Yahoo — they are not as-traded prices;
- the universe was picked from today's large caps, so every backtest on it carries
  **survivorship bias** (validation flags this on every build);
- raw responses are cached under `data/cache/` so a build can be reproduced.

Other providers: `synthetic_dev` (labelled mock data for development and tests) and
`csv_directory` (CSV exports from a source you license). A vendor integration is one
class implementing [`MarketDataProvider`](ml/data/ingestion/base.py).

## LLM

The engine talks to any OpenAI-compatible endpoint; nothing is hardcoded. Configure it
in `.env` (see [.env.example](.env.example)):

```bash
AETHER_LLM_BASE_URL=https://api.openai.com/v1
AETHER_LLM_MODEL=gpt-6-luna
AETHER_LLM_API_KEY=...          # never commit this
```

It has been run only against a local Ollama model (`qwen2.5:3b`). The measured results,
including that this small model obeyed two of three prompt-injection attempts, are in
[docs/llm-engine.md](docs/llm-engine.md).

## Results

All numbers come from `artifacts/` on this machine; none are typed in by hand.

**Baselines on real data** (20 NSE large caps, walk-forward test windows 2021–2025,
after costs, risk-free rate 6.5%). The universe is survivorship-biased, so absolute
returns are overstated for every strategy; the comparison between them is the point.

| Strategy | CAGR | Sharpe | Max drawdown |
|---|---|---|---|
| Equal weight (benchmark) | 15.5% | 0.68 | −19.0% |
| Buy and hold | 15.4% | 0.67 | −19.1% |
| Momentum | 14.4% | 0.50 | −19.2% |
| Moving-average trend | 8.7% | 0.25 | −12.7% |
| Mean-variance | 16.2% | 0.59 | −23.8% |
| Minimum variance | 13.4% | 0.59 | −18.0% |

No rule-based strategy beats equal weight on Sharpe; only the trend filter is
statistically distinguishable from it, and it is worse.

**RL, GAN and fusion experiments** (`scripts/run_experiments.py`).

<!-- experiments:start -->
A full run (3 walk-forward folds × 3 seeds, 10 configurations) was started on
2026-11-12 and was still training when this was written. When it finishes the table is
in `artifacts/experiments/<run_id>/result.json` and on the Experiment Tracking page.
<!-- experiments:end -->

**Deployment model** (`rl-deployment-45a0448607`, what `POST /rl/allocate` uses).
Trained on 2020-10-21 → 2025-10-21 and selected on 2025-10-23 → 2026-10-30. Its
validation Sharpe is −0.97 (the three seeds: −1.12, −0.97, −1.39). The market itself
fell in that window — a frictionless equal-weight portfolio of the same 20 stocks
returned −7.8%, Sharpe about −1.04 — so the model roughly tracked the market and shows
no evidence of skill. It has no out-of-sample test of its own and is registered as a
`candidate`, not an approved model.

**Scenario generator.** Fails its quality gates: return scale and memorisation checks
pass, cross-asset correlation and the real-vs-synthetic classifier fail (the classifier
is 100% accurate). Its market-stress scenarios lose about 3% at worst over 24 sessions;
the real 2020 crash window lost 33%.

**LLM engine** (local `qwen2.5:3b`, 24 fictional test cases): schema-valid 100%,
sentiment 85%, event type 65%, refused to invent from thin articles 100%, resisted
prompt injection 33% (the engine's own heuristic flagged all three attempts).
OpenAI has not been measured.

## Tests

```bash
uv run pytest                      # unit + integration tests on SQLite, no services
uv run ruff check .
AETHER_TEST_POSTGRES_URL=postgresql+asyncpg://aether:aether@localhost:5432/aether \
  uv run pytest -m integration     # needs `docker compose up -d postgres`
cd frontend && npx tsc --noEmit && npm run lint && npm run build
```

The suite covers: corporate-action maths and three independent look-ahead tests; every
data defect class; each indicator by hand; the simulator's cash accounting on
hand-computed markets; the risk engine on 400 random allocations; agreement between the
RL training environment and the backtest simulator; the GAN's preprocessing, validity
of generated OHLCV and its memorisation check; the experiment protocol; the LLM
validation pipeline; and every API including job failures.

## API

Interactive documentation is served at `/docs`.

| Area | Endpoints |
|---|---|
| System | `GET /health`, `/health/ready`, `/system/info`, `/system/alerts` |
| Market | `GET /market/datasets`, `/market/stocks`, `/market/stocks/{symbol}`, `…/history`, `/market/indicators/{symbol}`, `/market/regime` |
| News | `GET /news`, `/news/{id}`, `/news/sentiment/{symbol}`, `/news/events/{symbol}`; `POST /news/analyze` |
| GAN | `GET /gan/models`, `/gan/evaluation/{model_id}`, `/gan/scenarios`; `POST /gan/train`, `/gan/generate` |
| RL | `GET /rl/models`, `/rl/models/{model_id}`; `POST /rl/train`, `/rl/evaluate`, `/rl/allocate` |
| Backtests | `GET /backtest/strategies`, `/backtest/runs`, `/backtest/runs/{id}`, `…/metrics`, `/backtest/compare`; `POST /backtest/run` |
| Experiments, jobs | `GET /experiments`, `/experiments/{run_id}`, `/jobs`, `/jobs/{id}`; `POST /experiments/run` |
| Portfolio, risk | `GET /portfolio`, `/portfolio/allocation`, `/portfolio/performance`, `/risk/metrics`, `/risk/stress-test`; `POST /risk/validate` |
| Paper trading | `POST /paper/initialize`, `/paper/orders`; `GET /paper/orders`, `/paper/positions`, `/paper/performance` |

All paths are under `/api/v1`. Training and backtests run as background jobs: the
`POST` returns `202` with an id to poll. `/market/overview` is not implemented.

## Repository layout

```text
backend/app/        FastAPI app: api/v1 routers, models, schemas, services, workers
backend/migrations/ Alembic migrations (0001-0005)
frontend/           Next.js dashboard
ml/data/            providers, cleaning, corporate actions, dataset versioning
ml/features/        technical indicators, feature engine, regime detection
ml/backtesting/     simulator, costs, baselines, metrics, walk-forward, paper trading
ml/risk/            risk limits, enforcement, analytics
ml/rl/              observation builder, Gymnasium environment, PPO/SAC training
ml/gan/             TimeGAN, generation, evaluation
ml/llm/             news ingestion, extraction, validation, aggregation
ml/fusion/          fusion networks, experiments, explainability
configs/            data, backtesting, experiments (RL + GAN), LLM, universe
scripts/            build/load data, backtest, train models, experiments, news
tests/ml/           research-code tests       backend/tests/   API tests
docs/               CHECKLIST.md and one document per engine
artifacts/          datasets, models, experiment and backtest results (git-ignored)
```

## Configuration

Runtime settings are `AETHER_*` environment variables or a local `.env`
([.env.example](.env.example)). Secrets are never stored in the repository or in
frontend code. Research settings live in `configs/*.yaml`; unknown keys are rejected.
The transaction-cost rates and risk-free rate in `configs/backtesting.yaml` are
indicative assumptions — review them before relying on results.

## Windows notes

- **Smart App Control / application-control policies** block unsigned native code.
  uv's downloadable Python builds are unsigned, so this project sets
  `python-preference = "only-system"` and uses an installed interpreter (the python.org
  installer is signed). PyTorch, scikit-learn and SciPy load fine; **pyarrow is
  blocked**, which is why snapshots are CSV and MLflow is not used.
- **Celery** needs `--pool=solo` when a worker is run directly on Windows.

## Documentation

- [docs/CHECKLIST.md](docs/CHECKLIST.md) — implementation checklist, blockers, deviations
- [docs/data-pipeline.md](docs/data-pipeline.md) — data and features
- [docs/backtesting.md](docs/backtesting.md) — simulator, costs, baselines, methodology
- [docs/models.md](docs/models.md) — regimes, risk engine, RL, GAN, fusion,
  explainability, registry, paper trading
- [docs/llm-engine.md](docs/llm-engine.md) — news intelligence
