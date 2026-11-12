"""Daily portfolio simulator with explicit cash accounting (PRD 11, section 24 rule 5).

Timeline for a decision date T
    close of T        the strategy sees data up to and including T and returns targets
    session T+1       orders execute at that session's open (or close, if configured),
                      moved against the trader by the slippage assumption
    close of T+1      the portfolio is marked to market

A decision is never executed at a price the strategy could have seen. Decisions made
on the final session of the window are discarded (nothing left to execute on).

Execution rules
    * Targets are converted to rupee values using the portfolio value at the execution
      price; sells are processed before buys.
    * A symbol with no bar on the execution day cannot be traded (logged, not carried).
    * ``participation_limit`` caps each order at that share of the day's traded value;
      the unfilled part is dropped and logged. (The day's total value is only known
      after the close -- the usual, slightly optimistic, backtest convention.)
    * Buys are scaled down together if cash cannot cover value + costs. Cash never goes
      negative; there is no leverage and no short selling.
    * A held symbol with no bar is valued at its last close and flagged ``stale_price``.
    * On or after a symbol's delisting date the position is closed at
      ``delisting_recovery`` x last close (default 1.0 -- an optimistic assumption).
    * Idle cash earns no interest.

Outputs (:class:`SimulationResult`)
    equity   per session: equity, cash, invested, costs, turnover, daily_return
    weights  end-of-day weights per symbol plus CASH
    trades   every fill with its full cost breakdown and realised P&L (average cost)
    events   everything that did not go to plan (partial fills, no bar, stale, ...)
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from ml.backtesting.costs import COST_COMPONENTS, CostBreakdown, CostModel
from ml.backtesting.data import MarketData
from ml.backtesting.strategies import Strategy, StrategyContext, StrategyError

CASH = "CASH"
WEIGHT_TOLERANCE = 1e-9
# Buys are always trimmed slightly to pay for costs; only log a material shortfall.
MATERIAL_SCALING = 0.99


@dataclass(frozen=True)
class SimulationConfig:
    initial_capital: float = 1_000_000.0
    execution_price: Literal["open", "close"] = "open"
    participation_limit: float | None = 0.05
    allow_fractional_shares: bool = True
    min_trade_value: float = 0.0
    delisting_recovery: float = 1.0

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.participation_limit is not None and not 0 < self.participation_limit <= 1:
            raise ValueError("participation_limit must be in (0, 1]")
        if not 0 <= self.delisting_recovery <= 1:
            raise ValueError("delisting_recovery must be in [0, 1]")


@dataclass
class SimulationResult:
    strategy: str
    initial_capital: float
    equity: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    events: pd.DataFrame

    @property
    def returns(self) -> pd.Series:
        return self.equity["daily_return"]


TRADE_COLUMNS = [
    "trade_date",
    "symbol",
    "side",
    "shares",
    "price",
    "value",
    *COST_COMPONENTS,
    "total_cost",
    "slippage_cost",
    "realized_pnl",
]
EVENT_COLUMNS = ["trade_date", "symbol", "event", "detail"]


def _validated_target(
    target: pd.Series, ctx: StrategyContext, column_of: dict[str, int], n: int
) -> np.ndarray:
    """Turn a strategy's Series into a dense weight vector, rejecting untradable output."""
    if not isinstance(target, pd.Series):
        raise StrategyError("target_weights must return a pandas Series or None")
    if target.index.has_duplicates:
        raise StrategyError("target weights contain duplicate symbols")
    values = target.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise StrategyError("target weights contain NaN or infinite values")
    if (values < -WEIGHT_TOLERANCE).any():
        raise StrategyError("negative target weights: short selling is not supported")
    # A position already held may keep a weight even if it cannot trade today.
    held = set(ctx.current_weights.index[ctx.current_weights.to_numpy() > WEIGHT_TOLERANCE])
    allowed = set(ctx.universe) | held
    outside = sorted(set(target.index[values > WEIGHT_TOLERANCE]) - allowed)
    if outside:
        raise StrategyError(f"weights for symbols outside the tradable universe: {outside}")
    if values.sum() > 1 + 1e-6:
        raise StrategyError(f"target weights sum to {values.sum():.6f} > 1: leverage not supported")

    dense = np.zeros(n)
    for symbol, weight in zip(target.index, np.clip(values, 0.0, None), strict=True):
        if symbol in column_of:
            dense[column_of[symbol]] = weight
    return dense


def simulate(
    market: MarketData,
    strategy: Strategy,
    *,
    start,
    end,
    decision_dates,
    cost_model: CostModel,
    config: SimulationConfig | None = None,
) -> SimulationResult:
    config = config or SimulationConfig()
    sessions = market.sessions_between(start, end)
    if len(sessions) < 2:
        raise ValueError("the simulation window must contain at least two sessions")

    symbols = market.symbols
    n = len(symbols)
    column_of = {symbol: j for j, symbol in enumerate(symbols)}
    rows = market.sessions.get_indexer(sessions)
    close_matrix = market.close.to_numpy()
    exec_matrix = market.open.to_numpy() if config.execution_price == "open" else close_matrix
    value_matrix = market.traded_value.to_numpy()
    delisting = [
        None if pd.isna(d) else pd.Timestamp(d) for d in market.securities["delisting_date"]
    ]
    decisions = set(pd.DatetimeIndex(decision_dates))
    slip = cost_model.slippage_rate
    fractional = config.allow_fractional_shares

    shares = np.zeros(n)
    average_cost = np.zeros(n)
    last_price = np.full(n, np.nan)
    stale = np.zeros(n, dtype=bool)
    cash = float(config.initial_capital)
    peak_equity = float(config.initial_capital)
    pending: np.ndarray | None = None

    trades: list[tuple] = []
    events: list[tuple] = []
    equity_rows: list[tuple] = []
    weight_rows: list[np.ndarray] = []

    def record(day, j, side, quantity, price, value, costs: CostBreakdown, slippage, realized):
        trades.append(
            (day, symbols[j], side, quantity, price, value, *costs.as_dict().values(),
             costs.total, slippage, realized)
        )  # fmt: skip

    for k, day in enumerate(sessions):
        row = rows[k]
        costs_today = 0.0
        traded_today = 0.0
        portfolio_value = np.nan

        # (a) Positions in securities that have delisted are closed out.
        for j in np.flatnonzero(shares > 0):
            if delisting[j] is not None and day >= delisting[j]:
                price = last_price[j] * config.delisting_recovery
                proceeds = shares[j] * price
                realized = proceeds - shares[j] * average_cost[j]
                record(day, j, "delist", shares[j], price, proceeds, CostBreakdown(), 0.0, realized)
                events.append(
                    (day, symbols[j], "delisted",
                     f"closed at {config.delisting_recovery:.0%} of last close")
                )  # fmt: skip
                cash += proceeds
                shares[j] = average_cost[j] = 0.0

        # (b) Execute the targets decided at the previous session's close.
        if pending is not None:
            price = exec_matrix[row]
            tradable = np.isfinite(price)
            mark = np.where(tradable, price, last_price)
            held_value = np.where(shares > 0, shares * mark, 0.0)
            portfolio_value = cash + held_value.sum()
            delta = pending * portfolio_value - held_value
            tolerance = max(config.min_trade_value, 1e-9 * portfolio_value)
            if config.participation_limit is None:
                cap = np.full(n, np.inf)
            else:
                day_value = value_matrix[row]
                cap = np.where(np.isfinite(day_value), config.participation_limit * day_value, 0.0)

            for j in np.flatnonzero(np.abs(delta) > tolerance):
                if not tradable[j]:
                    events.append((day, symbols[j], "no_bar_on_execution", "order not placed"))

            # Sells first, so their proceeds can fund the buys.
            for j in np.flatnonzero((delta < -tolerance) & tradable):
                full_exit = pending[j] == 0.0
                quantity = shares[j] if full_exit else min(-delta[j] / price[j], shares[j])
                limit = cap[j] / price[j]
                if quantity > limit:
                    events.append(
                        (day, symbols[j], "partial_fill",
                         f"sell limited to {limit / quantity:.1%} of the order by liquidity")
                    )  # fmt: skip
                    quantity = limit
                if not fractional:
                    quantity = np.floor(quantity)
                if quantity <= 0:
                    continue
                fill = price[j] * (1 - slip)
                proceeds = quantity * fill
                costs = cost_model.charges("sell", proceeds, day)
                realized = (fill - average_cost[j]) * quantity - costs.total
                record(day, j, "sell", quantity, fill, proceeds, costs,
                       quantity * price[j] * slip, realized)  # fmt: skip
                cash += proceeds - costs.total
                shares[j] -= quantity
                if shares[j] <= 1e-12:
                    shares[j] = average_cost[j] = 0.0
                costs_today += costs.total
                traded_today += proceeds

            wanted: list[tuple[int, float]] = []
            for j in np.flatnonzero((delta > tolerance) & tradable):
                value = delta[j]
                if value > cap[j]:
                    events.append(
                        (day, symbols[j], "partial_fill",
                         f"buy limited to {cap[j] / value:.1%} of the order by liquidity")
                    )  # fmt: skip
                    value = cap[j]
                if value > 0:
                    wanted.append((j, value))

            def plan(scale: float, wanted=wanted, price=price, day=day):
                orders, required = [], 0.0
                for j, value in wanted:
                    quantity = scale * value / price[j]
                    if not fractional:
                        quantity = np.floor(quantity)
                    if quantity <= 0:
                        continue
                    fill = price[j] * (1 + slip)
                    cost_value = quantity * fill
                    costs = cost_model.charges("buy", cost_value, day)
                    orders.append((j, quantity, fill, cost_value, costs))
                    required += cost_value + costs.total
                return orders, required

            available = max(cash, 0.0)
            scale = 1.0
            orders, required = plan(scale)
            for _ in range(25):
                if required <= available or not orders:
                    break
                scale *= available / required * (1 - 1e-9)
                orders, required = plan(scale)
            if required > available:
                orders = []
            if wanted and (not orders or scale < MATERIAL_SCALING):
                events.append(
                    (day, None, "buys_scaled_to_cash", f"buy orders scaled to {scale:.1%}")
                )
            for j, quantity, fill, cost_value, costs in orders:
                record(day, j, "buy", quantity, fill, cost_value, costs,
                       quantity * price[j] * slip, np.nan)  # fmt: skip
                average_cost[j] = (shares[j] * average_cost[j] + cost_value) / (
                    shares[j] + quantity
                )
                shares[j] += quantity
                cash -= cost_value + costs.total
                costs_today += costs.total
                traded_today += cost_value
            pending = None

        # (c) Mark to market at the close.
        close = close_matrix[row]
        has_close = np.isfinite(close)
        last_price = np.where(has_close, close, last_price)
        now_stale = (shares > 0) & ~has_close
        for j in np.flatnonzero(now_stale & ~stale):
            events.append((day, symbols[j], "stale_price", "no bar; valued at last close"))
        stale = now_stale
        position_value = np.where(shares > 0, shares * last_price, 0.0)
        invested = position_value.sum()
        equity = cash + invested
        peak_equity = max(peak_equity, equity)
        turnover = 0.5 * traded_today / portfolio_value if traded_today else 0.0
        equity_rows.append((equity, cash, invested, costs_today, turnover))
        weight_rows.append(np.append(position_value, cash) / equity)

        # (d) Decide at the close, for execution on the next session.
        if day in decisions and k < len(sessions) - 1:
            listed = market.listed_on(day).to_numpy()
            universe = [symbols[j] for j in np.flatnonzero(listed & has_close)]
            ctx = StrategyContext(
                date=day,
                universe=universe,
                prices=market.close.iloc[: row + 1],
                features=market.features_on(day),
                current_weights=pd.Series(weight_rows[-1], index=[*symbols, CASH]),
                regime=market.regime_on(day),
                traded_value=market.traded_value.iloc[: row + 1],
                drawdown=equity / peak_equity - 1,
            )
            target = strategy.target_weights(ctx)
            pending = None if target is None else _validated_target(target, ctx, column_of, n)

    equity_frame = pd.DataFrame(
        equity_rows, index=sessions, columns=["equity", "cash", "invested", "costs", "turnover"]
    )
    equity_frame.index.name = "trade_date"
    previous = equity_frame["equity"].shift(1, fill_value=config.initial_capital)
    equity_frame["daily_return"] = equity_frame["equity"] / previous - 1

    weights_frame = pd.DataFrame(weight_rows, index=sessions, columns=[*symbols, CASH])
    weights_frame.index.name = "trade_date"
    return SimulationResult(
        strategy=strategy.name,
        initial_capital=config.initial_capital,
        equity=equity_frame,
        weights=weights_frame,
        trades=pd.DataFrame(trades, columns=TRADE_COLUMNS),
        events=pd.DataFrame(events, columns=EVENT_COLUMNS),
    )
