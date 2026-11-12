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
        "status": "planned",
        "summary": "News ingestion, event extraction, sentiment, entity mapping, "
        "source-grounded summaries.",
    },
    {
        "key": "gan",
        "name": "GAN scenario engine",
        "phase": 4,
        "status": "planned",
        "summary": "TimeGAN baseline, regime-conditioned generation, statistical and "
        "memorization evaluation.",
    },
    {
        "key": "rl",
        "name": "RL portfolio optimizer",
        "phase": 5,
        "status": "planned",
        "summary": "Gymnasium environment, PPO/SAC, risk-aware reward, model registry.",
    },
    {
        "key": "fusion",
        "name": "Unified intelligence & explainability",
        "phase": 6,
        "status": "planned",
        "summary": "Feature fusion, ablations, multi-regime evaluation, decision evidence.",
    },
    {
        "key": "paper",
        "name": "Dashboard & paper trading",
        "phase": 7,
        "status": "in_progress",
        "summary": "Dashboard shell is live; analytics pages and paper trading follow "
        "the engines they depend on.",
    },
    {
        "key": "hardening",
        "name": "Production hardening",
        "phase": 8,
        "status": "planned",
        "summary": "Authentication, monitoring, CI/CD, cloud deployment, security review.",
    },
)
