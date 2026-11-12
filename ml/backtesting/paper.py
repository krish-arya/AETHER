"""Paper trading: a simulated portfolio followed forward in time. No broker, ever.

Design: **replay**. A paper account is just a description -- starting capital, start
date, and what decides its trades (a baseline strategy, a registered RL model, or
orders entered by hand). Its state is recomputed by running the ordinary backtest
simulator from the start date to the latest session in the loaded dataset. So:

    * paper results use exactly the costs, execution timing and risk limits of the
      backtests -- there is no second, subtly different engine;
    * the state is always reproducible from the account description and the data;
    * the account "advances" when a newer dataset is loaded;
    * nothing here can reach a real market: orders are rows in a table and fills are
      computed by ``ml.backtesting.simulator``.

Manual orders are target weights decided at the close of ``decision_date`` and filled
at the next session's open. An order placed on the latest session stays ``pending``
until data for the next session exists. Every order passes through the risk engine; a
change that breaks a limit is scaled back or rejected, and the reason is recorded.
"""

from dataclasses import dataclass

import pandas as pd

from ml.backtesting.costs import CostModel
from ml.backtesting.data import MarketData
from ml.backtesting.simulator import CASH, SimulationConfig, SimulationResult, simulate
from ml.backtesting.strategies import Strategy, StrategyContext
from ml.risk.limits import RiskLimits
from ml.risk.strategy import RiskManagedStrategy


@dataclass(frozen=True)
class ManualOrder:
    order_id: str
    decision_date: pd.Timestamp
    symbol: str
    target_weight: float


class ManualOrders(Strategy):
    """Applies hand-entered target weights on their decision dates."""

    name = "manual"

    def __init__(self, orders: list[ManualOrder]) -> None:
        self.by_date: dict[pd.Timestamp, list[ManualOrder]] = {}
        for order in orders:
            self.by_date.setdefault(pd.Timestamp(order.decision_date), []).append(order)

    def target_weights(self, ctx: StrategyContext) -> pd.Series | None:
        todays = self.by_date.get(ctx.date)
        if not todays:
            return None
        target = ctx.current_weights.drop(labels=[CASH], errors="ignore").copy()
        target = target[target > 0]
        for order in todays:
            target[order.symbol] = order.target_weight
        return target


@dataclass
class PaperState:
    result: SimulationResult
    strategy: RiskManagedStrategy
    order_status: dict[str, dict]
    as_of: pd.Timestamp


def replay(
    market: MarketData,
    strategy: Strategy,
    *,
    start,
    initial_capital: float,
    cost_model: CostModel,
    limits: RiskLimits,
    decision_dates: pd.DatetimeIndex,
    orders: list[ManualOrder] | None = None,
) -> PaperState:
    """Recompute a paper account's full history up to the latest session."""
    sessions = market.sessions_between(start, market.sessions[-1])
    if len(sessions) < 2:
        raise ValueError("paper trading needs at least two sessions after the start date")
    managed = RiskManagedStrategy(strategy, limits, market.securities.get("sector"))
    result = simulate(
        market,
        managed,
        start=sessions[0],
        end=sessions[-1],
        decision_dates=decision_dates,
        cost_model=cost_model,
        config=SimulationConfig(initial_capital=initial_capital),
    )
    return PaperState(result, managed, _order_status(orders or [], result, managed, sessions), sessions[-1])


def _order_status(
    orders: list[ManualOrder],
    result: SimulationResult,
    managed: RiskManagedStrategy,
    sessions: pd.DatetimeIndex,
) -> dict[str, dict]:
    """pending / executed / rejected (with the reason) for each manual order."""
    risk_log = {row["date"]: row for row in managed.decisions}
    events = result.events
    status: dict[str, dict] = {}
    for order in orders:
        day = pd.Timestamp(order.decision_date)
        position = sessions.searchsorted(day)
        in_range = position < len(sessions) and sessions[position] == day
        if not in_range:
            status[order.order_id] = {"status": "rejected", "detail": "not a session of this account"}
            continue
        if position == len(sessions) - 1:
            status[order.order_id] = {
                "status": "pending",
                "detail": "will be filled at the next session's open once its data is loaded",
            }
            continue
        fill_day = sessions[position + 1]
        decision = risk_log.get(day.strftime("%Y-%m-%d"), {})
        if decision.get("rejected"):
            status[order.order_id] = {"status": "rejected", "detail": decision.get("reason")}
            continue
        blocked = events[(events["trade_date"] == fill_day) & (events["symbol"] == order.symbol)]
        fills = result.trades[
            (result.trades["trade_date"] == fill_day) & (result.trades["symbol"] == order.symbol)
        ]
        applied = decision.get("applied") or []
        note = f"risk engine applied: {', '.join(applied)}" if applied else None
        if "no_bar_on_execution" in set(blocked["event"]):
            status[order.order_id] = {"status": "rejected", "detail": "symbol did not trade that day"}
        elif len(fills):
            filled = float(fills["value"].sum())
            detail = f"filled {fill_day:%Y-%m-%d} for Rs {filled:,.0f}"
            if "partial_fill" in set(blocked["event"]):
                detail += " (partially: liquidity limit)"
            status[order.order_id] = {"status": "executed", "detail": "; ".join(x for x in (detail, note) if x)}
        else:
            status[order.order_id] = {
                "status": "executed",
                "detail": "; ".join(x for x in ("no trade was needed", note) if x),
            }
    return status
