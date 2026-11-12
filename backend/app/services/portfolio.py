"""Paper trading, portfolio views, allocation recommendations and risk reports.

Everything here is simulation or analysis. No function in this module -- or anywhere
in the platform -- can send an order to a broker.
"""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models import Allocation, DatasetVersion, PaperAccount, PaperOrder
from app.services import platform
from app.services.platform import NotAvailable
from ml.backtesting.data import MarketData
from ml.backtesting.metrics import compute_metrics, drawdown_series
from ml.backtesting.paper import ManualOrder, ManualOrders, PaperState, replay
from ml.backtesting.strategies import STRATEGIES, build_strategy
from ml.fusion.explain import AllocationInputs, explain_allocation
from ml.gan.generation.generate import stress_scenarios
from ml.gan.training.train import load_gan
from ml.risk import analytics
from ml.risk.limits import RiskLimits, check_limits
from ml.rl.evaluation.strategy import PolicyStrategy, sb3_policy

CASH = "CASH"
RISK_WINDOW = 500  # sessions of history behind the risk measures
_states: dict[tuple, PaperState] = {}


class InvalidRequest(ValueError):
    pass


def _clean(value):
    """NaN / numpy scalars -> JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def default_limits(settings: Settings, dataset: DatasetVersion) -> RiskLimits:
    return platform.experiment_defaults(settings, _dataset_dir(dataset)).rl.limits


def _dataset_dir(dataset: DatasetVersion):
    from pathlib import Path

    return Path(dataset.manifest_path).parent


# ---------------------------------------------------------------- paper accounts


async def active_account(session: AsyncSession) -> PaperAccount | None:
    return await session.scalar(
        select(PaperAccount)
        .where(PaperAccount.status == "active")
        .order_by(PaperAccount.created_at.desc())
        .limit(1)
    )


async def require_account(session: AsyncSession) -> PaperAccount:
    account = await active_account(session)
    if account is None:
        raise NotAvailable("No paper account exists; create one with POST /api/v1/paper/initialize")
    return account


async def account_orders(session: AsyncSession, account_id: str) -> list[PaperOrder]:
    result = await session.scalars(
        select(PaperOrder)
        .where(PaperOrder.account_id == account_id)
        .order_by(PaperOrder.created_at)
    )
    return list(result)


def validate_strategy(strategy: dict, settings: Settings, market: MarketData) -> None:
    kind = strategy.get("kind")
    if kind == "baseline":
        if strategy.get("ref") not in STRATEGIES:
            raise InvalidRequest(f"unknown baseline strategy; available: {sorted(STRATEGIES)}")
        try:
            build_strategy(strategy["ref"], strategy.get("params") or {})
        except (TypeError, ValueError) as error:
            raise InvalidRequest(str(error)) from error
    elif kind == "rl":
        card = platform.get_card(settings, "rl", strategy.get("ref") or "")
        platform.load_agent_for(settings, card, market)
    elif kind != "manual":
        raise InvalidRequest("strategy.kind must be 'baseline', 'rl' or 'manual'")


async def create_account(
    session: AsyncSession,
    settings: Settings,
    *,
    name: str,
    initial_capital: float,
    strategy: dict,
    start_date,
    rebalance_every: int,
) -> PaperAccount:
    dataset = await platform.latest_dataset(session)
    market = await asyncio.to_thread(platform.load_market, dataset)
    sessions = market.sessions
    start = pd.Timestamp(start_date) if start_date else sessions[max(len(sessions) - 127, 0)]
    position = sessions.searchsorted(start, side="left")
    if position >= len(sessions) - 1:
        raise InvalidRequest(
            f"start_date must be before the last session in the data ({sessions[-1]:%Y-%m-%d})"
        )
    await asyncio.to_thread(validate_strategy, strategy, settings, market)

    for existing in await session.scalars(
        select(PaperAccount).where(PaperAccount.status == "active")
    ):
        existing.status = "closed"
    account = PaperAccount(
        account_id=uuid4().hex,
        name=name,
        status="active",
        created_at=datetime.now(UTC),
        dataset_id=dataset.dataset_id,
        initial_capital=initial_capital,
        start_date=sessions[position].date(),
        rebalance_every=rebalance_every,
        strategy=strategy,
    )
    session.add(account)
    await session.commit()
    return account


def _replay(
    account: PaperAccount, orders: list[PaperOrder], settings, dataset, market
) -> PaperState:
    limits = default_limits(settings, dataset)
    sessions = market.sessions_between(account.start_date, market.sessions[-1])
    kind = account.strategy["kind"]
    manual = [
        ManualOrder(o.order_id, pd.Timestamp(o.decision_date), o.symbol, o.target_weight)
        for o in orders
    ]
    if kind == "manual":
        strategy, decisions = ManualOrders(manual), sessions
    elif kind == "baseline":
        strategy = build_strategy(account.strategy["ref"], account.strategy.get("params") or {})
        decisions = sessions[:: account.rebalance_every]
    else:
        card = platform.get_card(settings, "rl", account.strategy["ref"])
        model, builder, cfg = platform.load_agent_for(settings, card, market)
        strategy = PolicyStrategy(sb3_policy(model), builder, cfg.env.logit_scale, name="rl")
        decisions, limits = sessions[:: account.rebalance_every], cfg.limits
    return replay(
        market,
        strategy,
        start=account.start_date,
        initial_capital=account.initial_capital,
        cost_model=platform.cost_model(settings),
        limits=limits,
        decision_dates=decisions,
        orders=manual,
    )


async def account_state(session: AsyncSession, account: PaperAccount, settings: Settings):
    """``(state, market, dataset, orders)`` -- the account replayed to the latest session."""
    dataset = await platform.latest_dataset(session, account.dataset_id)
    market = await asyncio.to_thread(platform.load_market, dataset)
    orders = await account_orders(session, account.account_id)
    key = (account.account_id, dataset.dataset_id, tuple(o.order_id for o in orders))
    if key not in _states:
        _states[key] = await asyncio.to_thread(_replay, account, orders, settings, dataset, market)
    return _states[key], market, dataset, orders


def holdings(state: PaperState) -> tuple[pd.Series, float]:
    """Latest asset weights and the account's current drawdown."""
    weights = state.result.weights.iloc[-1].drop(labels=[CASH])
    drawdown = float(drawdown_series(state.result.returns).iloc[-1])
    return weights[weights > 1e-9], drawdown


def account_summary(account: PaperAccount, state: PaperState, dataset: DatasetVersion) -> dict:
    equity = state.result.equity
    weights, drawdown = holdings(state)
    return _clean(
        {
            "account_id": account.account_id,
            "name": account.name,
            "strategy": account.strategy,
            "dataset_id": dataset.dataset_id,
            "is_synthetic": dataset.is_synthetic,
            "start_date": account.start_date.isoformat(),
            "as_of": state.as_of,
            "initial_capital": account.initial_capital,
            "value": equity["equity"].iloc[-1],
            "cash": equity["cash"].iloc[-1],
            "invested": equity["invested"].iloc[-1],
            "daily_return": equity["daily_return"].iloc[-1],
            "cumulative_return": equity["equity"].iloc[-1] / account.initial_capital - 1,
            "drawdown": drawdown,
            "positions": int(len(weights)),
            "simulated": True,
            "note": "Simulated portfolio. No order is ever sent to a broker.",
        }
    )


def positions_payload(state: PaperState, market: MarketData) -> list[dict]:
    weights, _ = holdings(state)
    equity = float(state.result.equity["equity"].iloc[-1])
    sectors = market.securities.get("sector")
    last_close = market.close.iloc[-1]
    rows = [
        {
            "symbol": symbol,
            "weight": weight,
            "value": weight * equity,
            "sector": None if sectors is None else sectors.get(symbol),
            "last_close": last_close.get(symbol),
        }
        for symbol, weight in weights.sort_values(ascending=False).items()
    ]
    return _clean(rows)


def performance_payload(state: PaperState, settings: Settings) -> dict:
    result = state.result
    defaults = platform.backtest_defaults(settings)
    metrics = compute_metrics(
        result.returns,
        risk_free_rate=defaults.risk_free_rate,
        periods_per_year=defaults.periods_per_year,
        turnover=result.equity["turnover"],
        trades=result.trades,
        costs=result.equity["costs"],
        initial_capital=result.initial_capital,
    )
    equity = result.equity
    pnl = equity["equity"].diff().fillna(equity["equity"].iloc[0] - result.initial_capital)
    return _clean(
        {
            "as_of": state.as_of,
            "metrics": metrics,
            "slippage_cost": float(result.trades["slippage_cost"].sum()),
            "equity": {
                "dates": [d.strftime("%Y-%m-%d") for d in equity.index],
                "values": equity["equity"].tolist(),
            },
            "daily_pnl": [
                {"date": d.strftime("%Y-%m-%d"), "pnl": v, "return": r}
                for d, v, r in zip(
                    equity.index[-30:], pnl.iloc[-30:], result.returns.iloc[-30:], strict=True
                )
            ],
        }
    )


def orders_payload(state: PaperState, orders: list[PaperOrder]) -> dict:
    trades = state.result.trades.copy()
    trades["trade_date"] = pd.to_datetime(trades["trade_date"]).dt.strftime("%Y-%m-%d")
    events = state.result.events.copy()
    events["trade_date"] = pd.to_datetime(events["trade_date"]).dt.strftime("%Y-%m-%d")
    columns = ["trade_date", "symbol", "side", "shares", "price", "value", "total_cost",
               "slippage_cost", "realized_pnl"]  # fmt: skip
    return _clean(
        {
            "manual_orders": [
                {
                    "order_id": o.order_id,
                    "created_at": o.created_at.isoformat(),
                    "decision_date": o.decision_date.isoformat(),
                    "symbol": o.symbol,
                    "target_weight": o.target_weight,
                    **state.order_status.get(o.order_id, {"status": "pending", "detail": None}),
                }
                for o in orders
            ],
            "executed_trades": trades[columns].tail(200).to_dict("records"),
            "rejected_or_limited": events.tail(200).to_dict("records"),
            "risk_decisions": state.strategy.decisions[-50:],
        }
    )


async def place_order(
    session: AsyncSession, settings: Settings, *, symbol: str, target_weight: float, decision_date
) -> PaperOrder:
    account = await require_account(session)
    if account.strategy["kind"] != "manual":
        raise InvalidRequest("orders can only be entered on a manual paper account")
    dataset = await platform.latest_dataset(session, account.dataset_id)
    market = await asyncio.to_thread(platform.load_market, dataset)
    symbol = symbol.strip().upper()
    if symbol not in market.symbols:
        raise InvalidRequest(f"unknown symbol '{symbol}'")
    day = pd.Timestamp(decision_date) if decision_date else market.sessions[-1]
    if day not in market.sessions or day.date() < account.start_date:
        raise InvalidRequest("decision_date must be a trading session on or after the start date")
    order = PaperOrder(
        order_id=uuid4().hex,
        account_id=account.account_id,
        created_at=datetime.now(UTC),
        decision_date=day.date(),
        symbol=symbol,
        target_weight=target_weight,
    )
    session.add(order)
    await session.commit()
    return order


# ------------------------------------------------------------------ allocations


async def current_holdings(session: AsyncSession, settings: Settings, dataset: DatasetVersion):
    """Weights and drawdown of the active paper account on this dataset (else all cash)."""
    account = await active_account(session)
    if account is None or account.dataset_id != dataset.dataset_id:
        return pd.Series(dtype="float64"), 0.0
    state, *_ = await account_state(session, account, settings)
    return holdings(state)


def _explain(settings: Settings, card: dict, market: MarketData, previous, drawdown) -> dict:
    model, builder, cfg = platform.load_agent_for(settings, card, market)
    inputs = AllocationInputs(
        model=model,
        builder=builder,
        limits=cfg.limits,
        logit_scale=cfg.env.logit_scale,
        sectors=market.securities.get("sector"),
        adv=market.traded_value.tail(cfg.env.adv_window).mean(),
        model_id=card["model_id"],
        dataset_id=market.dataset_id,
    )
    return explain_allocation(inputs, market.sessions[-1], previous, drawdown)


async def recommend(session: AsyncSession, settings: Settings, model_id: str | None) -> Allocation:
    dataset = await platform.latest_dataset(session)
    market = await asyncio.to_thread(platform.load_market, dataset)
    card = (
        platform.get_card(settings, "rl", model_id)
        if model_id
        else platform.default_rl_card(settings, dataset.dataset_id)
    )
    previous, drawdown = await current_holdings(session, settings, dataset)
    record = await asyncio.to_thread(_explain, settings, card, market, previous, drawdown)
    record["model"] = {
        "model_id": card["model_id"],
        "trained_at": card["trained_at"],
        "status": card["status"],
        "training_window": card["training_window"],
        "validation_sharpe": card["metrics"].get("validation_sharpe"),
        "experiment": (card.get("lineage") or {}).get("experiment"),
        "notes": card.get("notes"),
    }
    record["is_synthetic"] = dataset.is_synthetic
    allocation = Allocation(
        allocation_id=uuid4().hex,
        created_at=datetime.now(UTC),
        as_of=market.sessions[-1].date(),
        model_id=card["model_id"],
        dataset_id=dataset.dataset_id,
        is_synthetic=dataset.is_synthetic,
        weights=record["allocation"],
        explanation=_clean(record),
    )
    session.add(allocation)
    await session.commit()
    return allocation


async def latest_allocation(session: AsyncSession) -> Allocation | None:
    return await session.scalar(select(Allocation).order_by(Allocation.created_at.desc()).limit(1))


# ------------------------------------------------------------------------- risk


async def portfolio_weights(session: AsyncSession, settings: Settings):
    """``(weights, drawdown, source, dataset, market)`` for the portfolio risk is measured on:
    the active paper account, or failing that the latest recommended allocation."""
    account = await active_account(session)
    if account is not None:
        state, market, dataset, _ = await account_state(session, account, settings)
        weights, drawdown = holdings(state)
        return weights, drawdown, f"paper account '{account.name}'", dataset, market
    allocation = await latest_allocation(session)
    if allocation is None:
        raise NotAvailable(
            "There is no portfolio to analyse: create a paper account or request an allocation"
        )
    dataset = await platform.latest_dataset(session, allocation.dataset_id)
    market = await asyncio.to_thread(platform.load_market, dataset)
    weights = pd.Series(allocation.weights, dtype="float64")
    return weights, 0.0, f"recommended allocation of {allocation.as_of}", dataset, market


def risk_metrics(weights, drawdown, source, dataset, market, settings) -> dict:
    returns = market.close.pct_change(fill_method=None).tail(RISK_WINDOW)
    sectors = market.securities.get("sector")
    limits = default_limits(settings, dataset)
    report = analytics.risk_report(
        weights, returns, sectors=sectors, market_returns=returns.mean(axis=1)
    )
    violations = check_limits(weights, limits=limits, sectors=sectors, drawdown=drawdown)
    return _clean(
        {
            "source": source,
            "dataset_id": dataset.dataset_id,
            "is_synthetic": dataset.is_synthetic,
            "as_of": market.sessions[-1],
            "window_sessions": int(len(returns)),
            "current_drawdown": drawdown,
            **report,
            "limits": limits.model_dump(mode="json"),
            "limit_violations": [v.as_dict() for v in violations],
            "note": "Historical measures over the window shown; they describe the past, "
            "not a forecast.",
        }
    )


def stress_report(weights, source, dataset, market, settings) -> dict:
    returns = market.close.pct_change(fill_method=None).tail(RISK_WINDOW * 4)
    historical = analytics.historical_stress_scenarios(returns, horizon=21, worst=5)
    losses = analytics.stress_test(weights, historical)
    out = {
        "source": source,
        "dataset_id": dataset.dataset_id,
        "is_synthetic": dataset.is_synthetic,
        "horizon_sessions": 21,
        "historical": [
            {"scenario": name, "portfolio_return": value, "kind": "historical window"}
            for name, value in losses.items()
        ],
        "generated": None,
    }
    cards = platform.list_cards(settings, "gan", dataset.dataset_id)
    if cards:
        card = cards[0]
        gan = load_gan(platform.models_root(settings) / "gan" / card["model_id"], market)
        generated = analytics.stress_test(weights, stress_scenarios(gan, 200, seed=0)).dropna()
        out["generated"] = {
            "generator_id": card["model_id"],
            "generator_status": card["status"],
            "scenarios": int(len(generated)),
            "horizon_sessions": gan.cfg.sequence_length,
            "mean": generated.mean(),
            "percentile_5": generated.quantile(0.05),
            "worst": generated.min(),
            "label": "SYNTHETIC scenarios from the GAN, conditioned on market stress. "
            "Not a forecast."
            + (
                "" if card["status"] == "validated" else " This generator FAILED its quality gates."
            ),
        }
    return _clean(out)
