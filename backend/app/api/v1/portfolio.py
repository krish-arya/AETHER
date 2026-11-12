"""Portfolio, risk and paper-trading endpoints (PRD 15.6, 15.7).

"The portfolio" is the active paper-trading account: a simulated portfolio replayed
through the backtest simulator. Nothing here places a real order.
"""

import asyncio
from datetime import date

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.database.session import get_session
from app.services import platform, portfolio
from app.services.platform import NotAvailable
from app.services.portfolio import InvalidRequest
from ml.risk.limits import enforce

portfolio_router = APIRouter(prefix="/portfolio", tags=["portfolio"])
risk_router = APIRouter(prefix="/risk", tags=["risk"])
paper_router = APIRouter(prefix="/paper", tags=["paper trading"])


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _http(error: Exception) -> HTTPException:
    if isinstance(error, NotAvailable):
        return HTTPException(status.HTTP_404_NOT_FOUND, str(error))
    return HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error))


async def _account_view(session: AsyncSession, settings: Settings):
    account = await portfolio.require_account(session)
    state, market, dataset, orders = await portfolio.account_state(session, account, settings)
    return account, state, market, dataset, orders


# -------------------------------------------------------------------- portfolio


@portfolio_router.get("")
async def get_portfolio(
    session: AsyncSession = Depends(get_session), settings: Settings = Depends(get_settings)
) -> dict:
    try:
        account, state, _, dataset, _ = await _account_view(session, settings)
    except (NotAvailable, InvalidRequest) as error:
        raise _http(error) from error
    return portfolio.account_summary(account, state, dataset)


@portfolio_router.get("/allocation")
async def get_allocation(
    session: AsyncSession = Depends(get_session), settings: Settings = Depends(get_settings)
) -> dict:
    """Current weights of the paper portfolio and the latest recommended allocation."""
    try:
        account, state, market, dataset, _ = await _account_view(session, settings)
    except (NotAvailable, InvalidRequest) as error:
        raise _http(error) from error
    positions = portfolio.positions_payload(state, market)
    latest = await portfolio.latest_allocation(session)
    recommended = None
    if latest is not None and latest.dataset_id == dataset.dataset_id:
        recommended = {
            "allocation_id": latest.allocation_id,
            "created_at": latest.created_at.isoformat(),
            "as_of": latest.as_of.isoformat(),
            "model_id": latest.model_id,
            "weights": latest.weights,
        }
    return {
        "as_of": state.as_of.strftime("%Y-%m-%d"),
        "is_synthetic": dataset.is_synthetic,
        "cash_weight": 1 - sum(p["weight"] for p in positions),
        "positions": positions,
        "recommended": recommended,
    }


@portfolio_router.get("/performance")
async def get_performance(
    session: AsyncSession = Depends(get_session), settings: Settings = Depends(get_settings)
) -> dict:
    try:
        _, state, _, dataset, _ = await _account_view(session, settings)
    except (NotAvailable, InvalidRequest) as error:
        raise _http(error) from error
    return {"is_synthetic": dataset.is_synthetic, **portfolio.performance_payload(state, settings)}


# ------------------------------------------------------------------------- risk


class ValidateRequest(_Strict):
    weights: dict[str, float] = Field(min_length=1, max_length=500)
    previous: dict[str, float] | None = None
    drawdown: float = Field(0.0, le=0, ge=-1)


@risk_router.get("/metrics")
async def get_risk_metrics(
    session: AsyncSession = Depends(get_session), settings: Settings = Depends(get_settings)
) -> dict:
    """VaR, expected shortfall, volatility, concentration, beta, drawdown and limit
    checks for the paper portfolio (or the latest recommended allocation)."""
    try:
        context = await portfolio.portfolio_weights(session, settings)
    except (NotAvailable, InvalidRequest) as error:
        raise _http(error) from error
    return await asyncio.to_thread(portfolio.risk_metrics, *context, settings)


@risk_router.get("/stress-test")
async def get_stress_test(
    session: AsyncSession = Depends(get_session), settings: Settings = Depends(get_settings)
) -> dict:
    """Portfolio loss in the worst historical windows and, when a scenario generator
    exists, in GAN market-stress scenarios (clearly labelled synthetic)."""
    try:
        weights, _, source, dataset, market = await portfolio.portfolio_weights(session, settings)
    except (NotAvailable, InvalidRequest) as error:
        raise _http(error) from error
    return await asyncio.to_thread(
        portfolio.stress_report, weights, source, dataset, market, settings
    )


@risk_router.post("/validate")
async def validate_allocation(
    request: ValidateRequest,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Run an allocation through the risk engine: what it violates, what the engine
    would change, and the compliant result (or a rejection)."""
    try:
        dataset = await platform.latest_dataset(session)
        market = await asyncio.to_thread(platform.load_market, dataset)
    except NotAvailable as error:
        raise _http(error) from error
    unknown = sorted(set(request.weights) - set(market.symbols))
    if unknown:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"unknown symbols: {unknown}")
    limits = portfolio.default_limits(settings, dataset)
    decision = enforce(
        pd.Series(request.weights, dtype="float64"),
        limits=limits,
        previous=None if request.previous is None else pd.Series(request.previous, dtype="float64"),
        sectors=market.securities.get("sector"),
        adv=market.traded_value.tail(20).mean(),
        drawdown=request.drawdown,
    )
    return {
        **decision.as_record(),
        "compliant": not decision.raw_violations,
        "weights": None
        if decision.weights is None
        else {s: float(w) for s, w in decision.weights.items() if w > 1e-9},
        "limits": limits.model_dump(mode="json"),
    }


# ---------------------------------------------------------------- paper trading


class StrategyChoice(_Strict):
    kind: str = Field(pattern="^(baseline|rl|manual)$")
    ref: str | None = Field(None, description="Baseline strategy name, or RL model id")
    params: dict = {}


class InitializeRequest(_Strict):
    name: str = Field("paper", min_length=1, max_length=64)
    initial_capital: float = Field(1_000_000.0, gt=0, le=1e13)
    strategy: StrategyChoice
    start_date: date | None = Field(None, description="Defaults to about six months back")
    rebalance_every: int = Field(5, ge=1, le=63)


class OrderRequest(_Strict):
    symbol: str = Field(min_length=1, max_length=32)
    target_weight: float = Field(ge=0, le=1)
    decision_date: date | None = Field(None, description="Defaults to the latest session")


@paper_router.post("/initialize", status_code=status.HTTP_201_CREATED)
async def initialize_paper_account(
    request: InitializeRequest,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Create the paper account (replacing any active one). Its history from
    ``start_date`` is simulated immediately; it advances as newer data is loaded."""
    try:
        account = await portfolio.create_account(
            session,
            settings,
            name=request.name,
            initial_capital=request.initial_capital,
            strategy=request.strategy.model_dump(),
            start_date=request.start_date,
            rebalance_every=request.rebalance_every,
        )
        state, _, dataset, _ = await portfolio.account_state(session, account, settings)
    except (NotAvailable, InvalidRequest) as error:
        raise _http(error) from error
    return portfolio.account_summary(account, state, dataset)


@paper_router.post("/orders", status_code=status.HTTP_201_CREATED)
async def place_paper_order(
    request: OrderRequest,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Enter a simulated order (a target weight) on a manual paper account."""
    try:
        order = await portfolio.place_order(
            session,
            settings,
            symbol=request.symbol,
            target_weight=request.target_weight,
            decision_date=request.decision_date,
        )
        _, state, _, _, _ = await _account_view(session, settings)
    except (NotAvailable, InvalidRequest) as error:
        raise _http(error) from error
    return {
        "order_id": order.order_id,
        "symbol": order.symbol,
        "target_weight": order.target_weight,
        "decision_date": order.decision_date.isoformat(),
        "simulated": True,
        **state.order_status.get(order.order_id, {"status": "pending", "detail": None}),
    }


@paper_router.get("/orders")
async def list_paper_orders(
    session: AsyncSession = Depends(get_session), settings: Settings = Depends(get_settings)
) -> dict:
    try:
        _, state, _, _, orders = await _account_view(session, settings)
    except (NotAvailable, InvalidRequest) as error:
        raise _http(error) from error
    return portfolio.orders_payload(state, orders)


@paper_router.get("/positions")
async def list_paper_positions(
    session: AsyncSession = Depends(get_session), settings: Settings = Depends(get_settings)
) -> dict:
    try:
        account, state, market, dataset, _ = await _account_view(session, settings)
    except (NotAvailable, InvalidRequest) as error:
        raise _http(error) from error
    return {
        **portfolio.account_summary(account, state, dataset),
        "holdings": portfolio.positions_payload(state, market),
    }


@paper_router.get("/performance")
async def paper_performance(
    session: AsyncSession = Depends(get_session), settings: Settings = Depends(get_settings)
) -> dict:
    try:
        _, state, _, dataset, _ = await _account_view(session, settings)
    except (NotAvailable, InvalidRequest) as error:
        raise _http(error) from error
    return {"is_synthetic": dataset.is_synthetic, **portfolio.performance_payload(state, settings)}
