# Implementation checklist

Living record of progress against [prd.md](../prd.md). An item is ticked only after it
has been **verified** — the "Evidence" column says how. Update this file, and the module
registry in [backend/app/core/modules.py](../backend/app/core/modules.py), whenever a
phase item is completed.

Legend: `[x]` done and verified · `[~]` partly done (see note) · `[ ]` not started

_Last updated: 2026-11-11_

## Verification snapshot (2026-11-11)

| Check | Result |
|---|---|
| `uv run pytest` | 267 passed, 3 skipped (PostgreSQL tests need `AETHER_TEST_POSTGRES_URL`) |
| `uv run pytest` with PostgreSQL available | 270 passed |
| `uv run ruff check .` / `ruff format --check .` | clean |
| Frontend `tsc --noEmit`, `eslint`, `next build` | clean; 9 static routes |
| `docker compose up -d --build` | postgres, redis, backend healthy; migrate exited 0; worker and frontend running |
| Celery round trip through Redis | `aether.system.ping` returned from the worker |
| Dataset build + load into PostgreSQL (in container) | 12 securities, 28,376 bars, 119 corporate actions, 28,376 feature rows |
| API latency on 127.0.0.1 | 7–33 ms typical; 125 ms for a 2.7 MB response (target < 500 ms) |
| Dashboard in headless Edge (1440 px and 520 px) | renders live data; synthetic data badged |
| Baseline backtest via CLI (6 strategies, 10 years, 4 walk-forward folds) | 23 s; same `run_id` and metrics on every run |
| Backtest via API on the stack | `202` → queued in Redis → Celery worker (11.5 s) → results in PostgreSQL and served; metrics equal the CLI run |
| Backtesting Lab page | lists runs, shows status and metric tables from the API; form request replayed with curl (button click itself not automated) |
| No-Docker workflow (SQLite, no Redis) | works; readiness correctly reports `503 degraded` |

Not verified: the GitHub Actions workflow (no remote yet); dashboard below ~500 px
width; clicking through the dashboard in a real browser session; any behaviour on real
NSE data; the cost-schedule rates against official circulars (see B6).

---

## Section 25 — Project bootstrap

| # | Task | Status | Evidence |
|---|---|---|---|
| 1 | Inspect the development environment | [x] | Python 3.12.10, Node 24, uv, Docker 29; no GPU; Smart App Control enforced (see blockers) |
| 2 | Create the repository structure | [x] | Layout per PRD §23 |
| 3 | Set up the Python environment | [x] | `pyproject.toml` + `uv.lock`; `uv sync` |
| 4 | Configure FastAPI | [x] | `backend/app/main.py`; `/docs` serves OpenAPI |
| 5 | Configure PostgreSQL | [x] | Async SQLAlchemy + asyncpg; integration tests against PostgreSQL 16 |
| 6 | Configure Redis | [x] | Readiness probe; Celery broker/back end round trip |
| 7 | Configure Next.js | [x] | Next 16 + TypeScript + Tailwind 4 + TanStack Query; production build |
| 8 | Create Docker Compose | [x] | Full stack starts and passes health checks |
| 9 | Establish database migrations | [x] | Alembic `0001`; schema equals ORM models on SQLite and PostgreSQL; downgrade tested |
| 10 | Create the initial API health check | [x] | `/api/v1/health`, `/api/v1/health/ready`, `/api/v1/system/info` |
| 11 | Create a basic dashboard shell | [x] | 8 pages; Overview and Market read live API data; others state "not built" |
| 12 | Add automated tests | [x] | 163 tests; CI workflow written (unverified) |
| 13 | Document all setup instructions | [x] | `README.md`, `.env.example`, this file, `docs/data-pipeline.md` |

## Phase 1 — Foundation and data infrastructure

| Deliverable | Status | Evidence / note |
|---|---|---|
| Repository structure, Docker Compose, PostgreSQL, Redis, FastAPI, Next.js | [x] | See bootstrap |
| Market data provider interface | [x] | `MarketDataProvider`; `synthetic_dev` and `csv_directory` providers |
| Historical data ingestion | [~] | Pipeline + idempotent database loader work end to end. **No real NSE source connected** — needs a data-licensing decision (blocker B1) |
| Data validation | [x] | 8 hard checks, 7 flags, quality gates; every defect class tested |
| Corporate-action adjustment | [x] | Splits, bonus, dividends; hand-computed tests. Rights/demergers not modelled |
| Trading calendar alignment | [x] | Provider or inferred calendar; missing sessions reported, never filled |
| Symbol mapping | [x] | `universe.symbol_map` |
| Survivorship-bias controls | [~] | Listing windows, `universe_as_of`, warning when no delisted names. Index constituent history not modelled |
| Point-in-time data validation | [x] | `as_of` cut-off; three independent look-ahead tests |
| Feature engineering (PRD §6.2, all five families) | [x] | 32 causal, scale-free features; configurable; versioned |
| Dataset versioning | [x] | Content-addressed snapshots + manifest; tamper detection |
| Market API (`/market/stocks`, `/history`, `/indicators`) | [x] | Responses compared value-for-value with the snapshot |

**Exit criterion — "a reproducible historical market dataset with validated features":**
met for the pipeline (deterministic, validated, tested), demonstrated on **synthetic
development data only**. It is not met for real market data until B1 is resolved.

Not in Phase 1's deliverable list, not started: fundamentals (§5.2), news (§5.3),
macro data (§5.4), market regime detection (§6.3, still open), `/market/overview`,
`/market/regime`.

## Phase 2 — Quantitative baseline and backtesting

| Deliverable | Status | Evidence / note |
|---|---|---|
| Technical indicator engine | [x] | Delivered in Phase 1 (`ml/features`) |
| Portfolio simulator | [x] | Decide at close T, execute at open T+1; cash accounting; hand-computed tests; accounting identities for every strategy |
| Transaction-cost model | [x] | Brokerage, STT, exchange charge, SEBI fee, stamp duty, GST, slippage; dated schedule in config. **Rates are indicative** (B6) |
| Liquidity restrictions, cash management, corporate actions | [x] | Participation cap, no leverage/shorting, total-return adjusted prices, delisting close-out |
| Rebalancing frequencies | [x] | Daily, weekly, monthly, quarterly |
| Benchmark strategies | [~] | Buy-and-hold, equal weight, momentum, moving average, mean-variance (+ min-variance). **NIFTY 50 TRI needs licensed index data** (B1) |
| Performance metrics (PRD §11.3, all eleven) | [x] | Each checked by hand; undefined metrics are null |
| Walk-forward backtesting | [x] | Chronological folds, embargo, fresh capital per test window, stitched out-of-sample series |
| Look-ahead-bias tests | [x] | Scrambling all prices after a cut-off leaves earlier equity and weights bit-identical |
| Point-in-time universe, survivorship controls | [x] | Universe = listed and traded on the decision date; delisted names included |
| Statistical significance | [~] | Paired block bootstrap of the Sharpe difference. Deflated Sharpe / multiple-testing correction not yet (needed from Phase 5) |
| Multiple random seeds | [ ] | Not applicable to the deterministic baselines; required for RL (Phase 5) |
| Asynchronous jobs + API (PRD §15.5) | [x] | `backtest_runs` table, Celery task, submit/list/detail/metrics/compare; failures recorded |
| Reproducibility | [x] | `run_id` derived from config hash + dataset content hash |
| Market regime detection (PRD §6.3) | [ ] | **Not built.** Not in the Phase 2 deliverable list but needed before Phase 4/5 |

**Exit criterion — "reliable baseline strategies and reproducible backtesting":** met
for the engine. Results exist only on synthetic development data, so nothing can yet be
said about how any strategy performs on NSE.

## Phase 3 — LLM intelligence engine

- [ ] Provider-agnostic LLM abstraction, structured outputs, schema validation
- [ ] News / filings ingestion, de-duplication, entity resolution
- [ ] Event extraction, sentiment, impact estimation, signal aggregation
- [ ] Retrieval store, source attribution, prompt-injection defences
- [ ] LLM evaluation suite

## Phase 4 — GAN engine

- [ ] PyTorch TimeGAN baseline, conditional / regime-conditioned generation
- [ ] Statistical evaluation suite, memorisation check
- [ ] Synthetic OHLCV derivation, scenario explorer API

## Phase 5 — RL portfolio optimization

- [ ] Gymnasium environment, action projection, configurable reward
- [ ] PPO, then SAC; Optuna tuning on validation data only
- [ ] Independent risk engine (§12), model registry, MLflow tracking

## Phase 6 — Unified intelligence architecture

- [ ] Early / late / attention fusion, temporal alignment
- [ ] Experiments E0–E6, ablations, statistical validation
- [ ] Explainability engine

## Phase 7 — Dashboard and paper trading

- [~] Dashboard shell (Overview, Market Intelligence, Backtesting Lab live; five pages state "not built")
- [~] Backtesting Lab: launch a baseline run, job status, metric tables. No charts, no asset/date/strategy/cost editors yet
- [ ] Charts, regime timeline, news, GAN explorer, optimizer, experiments
- [ ] Paper trading (simulated only)

## Phase 8 — Production hardening

- [ ] Authentication and authorisation, rate limiting
- [~] CI (workflow written, not yet run)
- [ ] Monitoring (Prometheus/Grafana), drift and staleness alerts
- [ ] Object storage, cloud deployment, security review, performance tests

---

## Acceptance criteria (PRD §22) — Data

- [~] Historical market data can be ingested — pipeline and loader verified; real source pending (B1)
- [x] Corporate actions are handled — splits, bonus issues, cash dividends
- [x] Data quality validation is implemented
- [x] Features are reproducible
- [~] Point-in-time data controls are implemented — bars, actions, universe by listing window; index membership and fundamentals timestamps pending

## Acceptance criteria (PRD §22) — Backtesting

- [x] Benchmark strategies are implemented — NIFTY 50 TRI pending data (B1)
- [x] Walk-forward validation works
- [x] Look-ahead bias checks exist
- [x] Transaction costs are configurable
- [x] Risk-adjusted metrics are reported

## Acceptance criteria (PRD §22) — Frontend

- [~] Dashboard is operational — three live pages
- [ ] Model experiments can be monitored
- [~] Backtests can be launched — baseline runs with capital and rebalance options
- [ ] Allocation recommendations are visible
- [ ] Paper trading works without real-money execution

The GAN, LLM, RL and Integration sections of §22 are open.

---

## Known blockers and decisions needed

**B1 — Real market data source (blocks real results in every later phase).**
AETHER does not scrape. A decision is needed on which licensed or authorised source to
use for NSE daily OHLCV, corporate actions and (later) index constituents, the NIFTY 50
TRI series, fundamentals and news. Until then only synthetic development data exists.
Options: CSV exports from a vendor you already license (works today via
`csv_directory`), or a dedicated provider class for a vendor API.

**B2 — Application-control policy on the development machine.**
Windows Smart App Control blocks unsigned native code. Effects so far: uv's standalone
Python (worked around by using the signed python.org interpreter) and pyarrow (dropped;
snapshots are CSV). **PyTorch and other ML wheels needed from Phase 4 may be blocked as
well.** If so, training must run inside Docker (CPU only on this machine) or on another
host. To be tested at the start of Phase 4, before any model code is written.

**B3 — No GPU.** Intel integrated graphics only. TimeGAN and RL training at research
scale will be slow on CPU; plan for small configurations locally and a GPU host for
real experiments.

**B4 — LLM provider (Phase 3).** Needs a provider choice and API key; the abstraction
will keep it swappable.

**B6 — Cost schedule and risk-free rate need review.**
`configs/backtesting.yaml` ships indicative Indian delivery-equity rates and a
placeholder risk-free rate (6.5%). The large items are right in structure (STT 0.1%
each side, 0.015% buy-side stamp duty from July 2020, 18% GST), but small exchange/SEBI
revisions are not itemised and pre-2020 stamp duty is approximated. Someone should
check them against current NSE, SEBI and broker schedules, and set brokerage to the
intended broker's plan, before any result is quoted.

**B5 — Memory headroom.** 16 GB RAM with roughly 1 GB free at bootstrap time. Docker
Desktop plus training jobs will be tight.

## Deviations from the PRD (deliberate)

| PRD | What was done | Why |
|---|---|---|
| Repository root `project-aether/` | Repository root is the existing `AETHER/` folder | Avoid a redundant nested directory |
| Parquet / object storage for datasets | CSV snapshots on local disk | pyarrow blocked (B2); object storage is Phase 8 |
| DVC for dataset versioning | Manifest-based content addressing | "or equivalent"; no extra tooling; can be registered with DVC later |
| TimescaleDB, pgvector | Plain PostgreSQL 16 | Not beneficial at tens of symbols; revisit at scale / Phase 3 |
| MLflow, Prometheus, Grafana, S3 in Compose | Not added yet | Introduced with the phases that use them |
| `/market/overview`, `/market/regime` | Not registered | Would have to return invented data until regime detection exists |
| Backtesting Lab is a Phase 7 item | A minimal version was built with Phase 2 | Makes the engine usable without curl; charts and editors remain in Phase 7 |
| NIFTY 50 TRI benchmark | Equal-weight universe portfolio as benchmark | No licensed index data yet (B1) |
| Backtests on raw prices with explicit corporate-action cash flows | Total-return adjusted prices | Standard for daily research; values and costs stay exact (see docs/backtesting.md) |
| Features as classic indicator levels | Scale-free forms (ratios, percentages) | Level features leak future corporate actions through back-adjustment |
