# Project AETHER

An integrated GAN + Reinforcement Learning + LLM framework for quantitative market
intelligence and portfolio optimization on the National Stock Exchange of India (NSE).

AETHER is a **research and decision-support platform**. It is not a SEBI-registered
investment adviser or research analyst, it does not give investment advice, and it
never places real orders. Historical and simulated performance does not guarantee
future results.

The full specification is in [prd.md](prd.md). Progress against it is tracked, item by
item, in [docs/CHECKLIST.md](docs/CHECKLIST.md).

## Status

| Phase | Scope | Status |
|---|---|---|
| Bootstrap | Repo, FastAPI, PostgreSQL, Redis, Next.js, Docker Compose, migrations, tests | Done |
| 1 | Data infrastructure: provider interface, ingestion, validation, features | Done on synthetic dev data — **real NSE data source not yet connected** |
| 2 | Quant baselines, backtesting | Done on synthetic dev data. Market regime detection still open |
| 3 | LLM intelligence engine | Not started |
| 4 | GAN scenario engine | Not started |
| 5 | RL portfolio optimizer | Not started |
| 6 | Feature fusion, explainability | Not started |
| 7 | Full dashboard, paper trading | Shell + Market and Backtesting Lab pages (no charts yet) |
| 8 | Production hardening | Not started |

No GAN, RL or LLM model exists yet. The dashboard says so on every page that depends
on one; nothing is stubbed with invented numbers.

> **About the data.** The only data source wired up so far is `synthetic_dev`: a
> seeded generator of artificial prices used to build and test the pipeline. Its
> symbols are `SYNTH01…`, every row is flagged `is_synthetic`, and the UI badges it
> "Synthetic · not market data". It has no research value. To use real data, supply
> CSV exports from a source you are licensed to use — see
> [Using real market data](#using-real-market-data).

## Requirements

| Tool | Version | Notes |
|---|---|---|
| Python | 3.12+ | Use the python.org installer on Windows (see [Windows notes](#windows-notes)) |
| [uv](https://docs.astral.sh/uv/) | 0.5+ | Python dependency and environment manager |
| Node.js | 20.9+ (24 tested) | For the dashboard |
| Docker Desktop | recent | Optional; needed for PostgreSQL / Redis / the full stack |

## Quick start (Docker)

```bash
docker compose up -d --build        # postgres, redis, migrations, API, worker, dashboard
docker compose ps                   # wait until backend is "healthy"

# Build the development dataset and load it into PostgreSQL
docker compose exec backend python scripts/build_dataset.py
docker compose exec backend python scripts/load_dataset.py --latest

# Run the baseline strategies (or use the Backtesting Lab page)
docker compose exec backend python scripts/run_backtest.py --latest
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:3000 |
| API | http://localhost:8000 |
| Interactive API docs | http://localhost:8000/docs |

All ports are published on `127.0.0.1` only. Stop with `docker compose down`
(add `-v` to delete the database volume as well).

## Local development (without Docker)

Everything except Redis-backed jobs works with no external services, using a SQLite
file in place of PostgreSQL.

```bash
uv sync                                               # create .venv, install dependencies

# 1. Build a versioned dataset (ingest -> validate -> adjust -> features)
uv run python scripts/build_dataset.py --config configs/data.yaml

# 2. Create the database and load the dataset
export AETHER_DATABASE_URL="sqlite+aiosqlite:///./aether_dev.db"     # PowerShell: $env:AETHER_DATABASE_URL = "..."
uv run alembic -c backend/alembic.ini upgrade head
uv run python scripts/load_dataset.py --latest

# 3. Backtest the baseline strategies (writes artifacts/backtests/<run_id>/)
uv run python scripts/run_backtest.py --latest

# 4. Run the API (http://localhost:8000/docs)
uv run uvicorn app.main:app --reload

# 5. Run the dashboard (http://localhost:3000) in a second terminal
cd frontend && npm install && npm run dev
```

Without Redis, `GET /api/v1/health/ready` correctly reports `degraded` (Redis down) and
the dashboard shows it; the market endpoints still work. Submitting a backtest through
the API needs Redis and a worker (it returns `503` without them) — use
`scripts/run_backtest.py` instead.

A hybrid setup also works: `docker compose up -d postgres redis`, leave
`AETHER_DATABASE_URL` at its default, and run the API and dashboard on the host.

## Tests

```bash
uv run pytest                      # unit + integration tests on SQLite (no services needed)
uv run ruff check .                # lint

# PostgreSQL integration tests (need `docker compose up -d postgres`)
AETHER_TEST_POSTGRES_URL=postgresql+asyncpg://aether:aether@localhost:5432/aether \
  uv run pytest -m integration

cd frontend && npx tsc --noEmit && npm run lint && npm run build
```

What the suite covers:

- **Financial correctness** — corporate-action factors against hand-computed values;
  three independent look-ahead-bias tests; point-in-time universe; as-of cut-offs.
- **Data quality** — every defect class is quarantined or flagged, never silently fixed.
- **Indicators** — each indicator against hand-computed values or an independent reference.
- **Reproducibility** — the same config yields the same dataset id; tampering is detected.
- **Backtesting** — costs, fills and cash accounting against hand-computed values;
  accounting identities for every strategy; scrambling future prices cannot change past
  results; metrics by hand; walk-forward folds; reproducible run ids.
- **Backend** — health probes, migrations matching the ORM models (SQLite and
  PostgreSQL), idempotent dataset loading, API responses equal to the source snapshot,
  the backtest job lifecycle including failures.

## Using real market data

AETHER does not download or scrape market data. A publicly reachable endpoint does not
imply permission for automated collection or redistribution, so the data source is an
explicit operator decision.

1. Obtain daily OHLCV (unadjusted) and corporate actions from a source whose licence
   covers your use.
2. Export to CSV in the layout documented in
   [ml/data/ingestion/csv_provider.py](ml/data/ingestion/csv_provider.py), under
   `data/licensed/` (git-ignored).
3. In [configs/data.yaml](configs/data.yaml) set `provider.kind: csv_directory`, fill in
   `source_name` and `license_note` (both are required and recorded in the dataset
   manifest), and rename `dataset.name`.
4. `uv run python scripts/build_dataset.py`, then review
   `artifacts/datasets/<id>/quality_report.json` before using the dataset.

A direct vendor integration is one new class implementing
[`MarketDataProvider`](ml/data/ingestion/base.py); nothing else changes.

## API

Interactive documentation is served at `/docs` (OpenAPI at `/openapi.json`).

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Liveness (no dependencies) |
| GET | `/api/v1/health/ready` | Readiness: database + Redis; `503` when degraded |
| GET | `/api/v1/system/info` | Version, disclaimers, per-module build status |
| GET | `/api/v1/market/datasets` | Loaded dataset versions with provenance and quality summary |
| GET | `/api/v1/market/stocks` | Securities; `?as_of=YYYY-MM-DD` gives the point-in-time universe |
| GET | `/api/v1/market/stocks/{symbol}` | Security detail and latest bar |
| GET | `/api/v1/market/stocks/{symbol}/history` | Daily bars; `start`, `end`, `limit` |
| GET | `/api/v1/market/indicators/{symbol}` | Technical features; `start`, `end`, `limit`, `feature_set_version` |
| GET | `/api/v1/backtest/strategies` | Available baseline strategies and default parameters |
| POST | `/api/v1/backtest/run` | Submit a backtest job; returns `202` with a `run_id` |
| GET | `/api/v1/backtest/runs` | Jobs, newest first |
| GET | `/api/v1/backtest/runs/{run_id}` | Job status, config, summary and equity curves |
| GET | `/api/v1/backtest/runs/{run_id}/metrics` | Full-period and walk-forward metrics |
| GET | `/api/v1/backtest/compare?run_ids=…` | Metrics of several runs side by side |

Backtests run as background jobs: the `POST` returns immediately and the job's status
is polled. Endpoints for later phases (news, GAN, RL, portfolio, risk, paper trading,
`/market/overview`, `/market/regime`) are deliberately absent until their engines exist.

## Repository layout

```text
backend/
  app/
    main.py            FastAPI application factory
    api/v1/            Versioned REST routers
    core/              Settings, logging, Redis client, module status registry
    database/          SQLAlchemy base, async engine and sessions
    models/            ORM models
    schemas/           Pydantic response models
    services/          Health probes, dataset loader, backtest jobs
    workers/           Celery application and job tasks
  migrations/          Alembic migrations
  tests/               Backend tests
frontend/              Next.js dashboard (app/, components/, hooks/, services/, lib/)
ml/
  data/                Provider interface, ingestion, cleaning, adjustment, versioning
  features/            Technical indicator library and feature engine
  backtesting/         Simulator, cost model, baseline strategies, metrics, walk-forward
  gan/ llm/ rl/ fusion/ risk/                   (empty packages for later phases)
configs/               YAML configuration (data.yaml, backtesting.yaml)
scripts/               build_dataset.py, load_dataset.py, run_backtest.py
tests/ml/              Data pipeline, feature and backtesting tests
docs/                  CHECKLIST.md, data-pipeline.md, backtesting.md
artifacts/             Generated datasets, backtest results and models (git-ignored)
```

## Configuration

Runtime settings come from `AETHER_*` environment variables or a local `.env`
(template: [.env.example](.env.example)). Secrets are never stored in the repository
or in frontend code. Dataset builds are configured in [configs/data.yaml](configs/data.yaml)
and backtests in [configs/backtesting.yaml](configs/backtesting.yaml); unknown keys are
rejected. The transaction-cost rates and the risk-free rate in the backtest config are
indicative assumptions — review them before relying on results.

## Windows notes

- **Smart App Control / application-control policies** block unsigned executables and
  DLLs. uv's downloadable Python builds are unsigned, so this project sets
  `python-preference = "only-system"` and uses an installed interpreter (the python.org
  installer is signed). If `uv sync` reports *"An Application Control policy has blocked
  this file"*, install Python 3.12 from python.org and re-run it.
- **pyarrow** is blocked by the same policy, which is why dataset snapshots are CSV
  rather than Parquet. Other native ML packages (notably PyTorch, needed from Phase 4)
  may be blocked too; if so, model training will have to run in Docker or on another
  machine. This is tracked as a known blocker in the checklist.
- **Celery** needs `--pool=solo` when a worker is run directly on Windows.

## Further documentation

- [docs/CHECKLIST.md](docs/CHECKLIST.md) — implementation checklist and known blockers
- [docs/data-pipeline.md](docs/data-pipeline.md) — data and feature pipeline: purpose,
  schemas, configuration, evaluation, limitations
- [docs/backtesting.md](docs/backtesting.md) — backtesting engine: execution model, cost
  model, strategies, evaluation methodology, limitations
