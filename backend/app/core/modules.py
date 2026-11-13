"""Single source of truth for which platform modules are actually implemented.

The API and dashboard read this registry so the UI can never present an unbuilt
component as working (PRD section 24, Rule 2: no fake functionality). Update the
status here -- and in docs/CHECKLIST.md -- only once a module is verified.
"""

from typing import Literal, TypedDict

ModuleStatus = Literal["available", "in_progress", "planned"]


class ModuleInfo(TypedDict):
    key: str
    name: str
    phase: int
    status: ModuleStatus
    summary: str


MODULES: tuple[ModuleInfo, ...] = (
    {
        "key": "data",
        "name": "Data infrastructure",
        "phase": 1,
        "status": "available",
        "summary": "Provider interface, ingestion, validation, corporate-action "
        "adjustment, technical features, versioned datasets.",
    },
    {
        "key": "backtesting",
        "name": "Quant baselines & backtesting",
        "phase": 2,
        "status": "available",
        "summary": "Portfolio simulator, Indian delivery cost model, five baseline "
        "strategies, risk-adjusted metrics, walk-forward evaluation, bootstrap "
        "significance; runs as background jobs. Market regime detection is not built yet.",
    },
    {
        "key": "llm",
        "name": "LLM intelligence engine",
        "phase": 3,
        "status": "available",
        "summary": "Headline ingestion, event extraction with schema / entity / grounding "
        "checks, point-in-time news features. Verified live with a small local model only; "
        "Verified with OpenAI gpt-6-luna and local qwen2.5:3b. No historical news archive.",
    },
    {
        "key": "gan",
        "name": "GAN scenario engine",
        "phase": 4,
        "status": "available",
        "summary": "Regime-conditioned TimeGAN, synthetic OHLCV markets, statistical and "
        "memorisation evaluation. The generators trained so far FAIL their quality gates.",
    },
    {
        "key": "rl",
        "name": "RL portfolio optimizer",
        "phase": 5,
        "status": "available",
        "summary": "Gymnasium environment, PPO (SAC selectable), risk-aware rewards, "
        "independent risk engine, model registry. No hyper-parameter search yet.",
    },
    {
        "key": "fusion",
        "name": "Unified intelligence & explainability",
        "phase": 6,
        "status": "available",
        "summary": "Early / late / attention fusion, experiments E0-E6 with ablations and "
        "seeds, allocation explanations. News features are not in the experiments (no "
        "historical news).",
    },
    {
        "key": "paper",
        "name": "Dashboard & paper trading",
        "phase": 7,
        "status": "available",
        "summary": "Eight dashboard pages on live API data (tables, no charts yet) and "
        "replay-based paper trading. Simulated only: there is no broker integration.",
    },
    {
        "key": "hardening",
        "name": "Production hardening",
        "phase": 8,
        "status": "planned",
        "summary": "Authentication, monitoring, CI/CD, cloud deployment, security review.",
    },
)
