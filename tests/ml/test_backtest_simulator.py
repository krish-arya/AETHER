"""Portfolio simulator: execution timing, cash accounting, costs and trading frictions,
checked against hand-computed numbers on tiny markets."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ml.backtesting.costs import CostConfig, CostModel
from ml.backtesting.simulator import CASH, SimulationConfig, simulate
from ml.backtesting.strategies import StrategyError

from .backtest_helpers import Scripted, make_market

CAPITAL = 1_000_000.0
FREE = CostModel.zero()


def _run(market, strategy, *, costs=FREE, decisions=None, **config):
    sessions = market.sessions
    return simulate(
        market,
        strategy,
        start=sessions[0],
        end=sessions[-1],
        decision_dates=sessions if decisions is None else sessions[decisions],
        cost_model=costs,
        config=SimulationConfig(initial_capital=CAPITAL, **config),
    )


def _stt_only(rate=0.001, slippage_bps=0.0) -> CostModel:
    return CostModel(
        CostConfig.model_validate(
            {
                "slippage_bps": slippage_bps,
                "statutory": [
                    {
                        "effective_from": date(2000, 1, 1),
                        "stt_buy": rate,
                        "stt_sell": rate,
                        "exchange_txn": 0,
                        "sebi_fee": 0,
                        "stamp_duty_buy": 0,
                        "stamp_duty_sell": 0,
                        "gst": 0,
                    }
                ],
            }
        )
    )


# ------------------------------------------------------------- timing and returns


def test_decision_at_close_executes_at_next_open():
    # Close of day 0 is 10, but the order can only fill at day 1's open of 12.
    market = make_market({"A": [10, 12, 12, 12]}, {"A": [10, 12, 13.2, 13.2]})
    result = _run(market, Scripted({"A": 1.0}))

    trade = result.trades.iloc[0]
    assert trade["trade_date"] == market.sessions[1]  # not the decision day
    assert trade["price"] == 12.0  # next open, not the close the strategy saw
    assert trade["shares"] == pytest.approx(CAPITAL / 12)
    np.testing.assert_allclose(result.equity["equity"], [1e6, 1e6, 1.1e6, 1.1e6])
    np.testing.assert_allclose(result.returns, [0.0, 0.0, 0.1, 0.0], atol=1e-12)


def test_execution_at_close_when_configured():
    market = make_market({"A": [10, 12, 12]}, {"A": [10, 15, 15]})
    result = _run(market, Scripted({"A": 1.0}), execution_price="close")
    assert result.trades.iloc[0]["price"] == 15.0


def test_first_session_is_never_traded_and_last_decision_is_discarded():
    market = make_market({"A": [10, 10, 10, 10]})
    result = _run(market, Scripted({"A": 1.0}, None, None, {"A": 0.0}))
    assert (result.trades["trade_date"] > market.sessions[0]).all()
    # The 4th decision falls on the final session: nothing left to execute it on.
    assert list(result.trades["side"]) == ["buy"]
    assert result.weights["A"].iloc[-1] == pytest.approx(1.0)


def test_two_asset_portfolio_matches_hand_computation():
    market = make_market(
        {"A": [100, 100, 110, 121], "B": [50, 50, 50, 40]},
        {"A": [100, 110, 121, 121], "B": [50, 50, 40, 40]},
    )
    result = _run(market, Scripted({"A": 0.6, "B": 0.2}), decisions=[0])
    # Day 1 open: 6,000 A @100, 4,000 B @50, cash 200,000.
    np.testing.assert_allclose(
        result.equity["equity"],
        [
            1e6,
            6000 * 110 + 4000 * 50 + 2e5,
            6000 * 121 + 4000 * 40 + 2e5,
            6000 * 121 + 4000 * 40 + 2e5,
        ],
    )
    np.testing.assert_allclose(result.equity["cash"].iloc[1:], 2e5)
    final = result.weights.iloc[-1]
    assert final["A"] == pytest.approx(726_000 / 1_086_000)
    assert final[CASH] == pytest.approx(200_000 / 1_086_000)


def test_rotation_sells_before_buying():
    market = make_market({"A": [10, 10, 10, 10], "B": [20, 20, 20, 20]})
    result = _run(market, Scripted({"A": 1.0}, {"B": 1.0}))
    assert list(result.trades["side"]) == ["buy", "sell", "buy"]
    final = result.weights.iloc[-1]
    assert final["B"] == pytest.approx(1.0) and final["A"] == 0.0
    assert result.equity["equity"].iloc[-1] == pytest.approx(CAPITAL)


def test_none_means_hold_and_empty_means_all_cash():
    market = make_market({"A": [10, 10, 11, 12, 12]})
    held = _run(market, Scripted({"A": 1.0}, None, None, None))
    assert len(held.trades) == 1

    flat = _run(market, Scripted({"A": 1.0}, {}))
    assert list(flat.trades["side"]) == ["buy", "sell"]
    assert flat.weights[CASH].iloc[-1] == pytest.approx(1.0)
    assert flat.weights["A"].iloc[-1] == 0.0  # full exit leaves no dust


# ----------------------------------------------------------------- costs and cash


def test_costs_are_paid_from_cash_and_buys_are_scaled_to_fit():
    market = make_market({"A": [10, 10, 10]})
    result = _run(market, Scripted({"A": 1.0}), costs=_stt_only(0.001))

    trade = result.trades.iloc[0]
    # value + 0.1% STT must equal the available cash.
    assert trade["value"] == pytest.approx(CAPITAL / 1.001, rel=1e-8)
    assert trade["stt"] == pytest.approx(trade["value"] * 0.001)
    assert trade["total_cost"] == pytest.approx(trade["stt"])
    assert result.equity["cash"].min() >= -1e-6
    assert result.equity["equity"].iloc[-1] == pytest.approx(
        CAPITAL - trade["total_cost"], rel=1e-8
    )
    assert result.equity["costs"].sum() == pytest.approx(result.trades["total_cost"].sum())


def test_round_trip_loses_exactly_costs_plus_slippage():
    market = make_market({"A": [100, 100, 100, 100]})
    costs = _stt_only(0.001, slippage_bps=10)  # 0.1% slippage per side
    result = _run(market, Scripted({"A": 1.0}, {}), costs=costs)

    buy, sell = result.trades.iloc[0], result.trades.iloc[1]
    assert buy["price"] == pytest.approx(100.1)  # pays up
    assert sell["price"] == pytest.approx(99.9)  # sells down
    assert buy["slippage_cost"] == pytest.approx(buy["shares"] * 100 * 0.001)

    lost = CAPITAL - result.equity["equity"].iloc[-1]
    explicit = result.trades["total_cost"].sum() + result.trades["slippage_cost"].sum()
    assert lost == pytest.approx(explicit, rel=1e-9)
    assert sell["realized_pnl"] == pytest.approx(
        (99.9 - 100.1) * sell["shares"] - sell["total_cost"]
    )
    assert sell["realized_pnl"] < 0


def test_whole_shares_only_when_fractional_trading_is_off():
    market = make_market({"A": [333, 333, 333], "B": [77, 77, 77]})
    result = _run(
        market, Scripted({"A": 0.5, "B": 0.5}), costs=_stt_only(), allow_fractional_shares=False
    )
    assert (result.trades["shares"] % 1 == 0).all()
    assert result.equity["cash"].min() >= 0
    assert result.weights[CASH].iloc[-1] < 0.01  # only the rounding remainder is idle


def test_min_trade_value_suppresses_small_orders():
    market = make_market({"A": [10, 10, 10, 10]})
    targets = Scripted({"A": 0.5}, {"A": 0.5001})
    assert len(_run(market, targets, min_trade_value=1_000).trades) == 1
    assert len(_run(market, Scripted({"A": 0.5}, {"A": 0.5001})).trades) == 2


# --------------------------------------------------------------------- frictions


def test_participation_limit_caps_orders_and_logs_partial_fills():
    market = make_market({"A": [10, 10, 10]}, traded_value=2_000_000.0)
    result = _run(market, Scripted({"A": 1.0}), participation_limit=0.05)

    assert result.trades.iloc[0]["value"] == pytest.approx(100_000.0)  # 5% of 2,000,000
    assert result.weights["A"].iloc[-1] == pytest.approx(0.1)
    assert list(result.events["event"]) == ["partial_fill"]


def test_symbol_without_a_bar_cannot_be_traded_and_is_valued_at_last_close():
    nan = float("nan")
    market = make_market(
        {"A": [10, 10, nan, nan, 12], "B": [5, 5, 5, 5, 5]},
        {"A": [10, 11, nan, nan, 12], "B": [5, 5, 5, 5, 5]},
    )
    # Buy A on day 1; ask to sell it on day 2 when A has no bar.
    result = _run(market, Scripted({"A": 0.5}, {"A": 0.0, "B": 0.4}, None, None))

    assert set(result.events["event"]) == {"no_bar_on_execution", "stale_price"}
    a_trades = result.trades[result.trades["symbol"] == "A"]
    assert list(a_trades["side"]) == ["buy"]  # the sell never happened
    # Held through the gap at the last close of 11, then marked at 12.
    value = result.weights["A"] * result.equity["equity"]
    np.testing.assert_allclose(value.iloc[1:], [550_000, 550_000, 550_000, 600_000])


def test_untradable_symbols_are_not_in_the_strategy_universe():
    nan = float("nan")
    market = make_market(
        {"A": [10, 10, 10], "B": [nan, 5, 5], "C": [7, 7, 7]},
        listing={"C": "2024-01-03"},  # lists on the third session
    )
    strategy = Scripted(None, None)
    _run(market, strategy)
    assert strategy.contexts[0].universe == ["A"]  # B has no bar, C is not yet listed
    assert strategy.contexts[1].universe == ["A", "B"]


def test_delisted_holding_is_closed_at_recovery_value():
    nan = float("nan")
    market = make_market(
        {"A": [10, 10, 10, nan, nan], "B": [5, 5, 5, 5, 5]},  # B keeps the sessions going
        {"A": [10, 10, 8, nan, nan], "B": [5, 5, 5, 5, 5]},
        delisting={"A": "2024-01-04"},
    )
    full = _run(market, Scripted({"A": 1.0}))
    assert list(full.trades["side"]) == ["buy", "delist"]
    assert full.trades.iloc[-1]["trade_date"] == pd.Timestamp("2024-01-04")
    assert full.equity["equity"].iloc[-1] == pytest.approx(800_000.0)
    assert full.weights[CASH].iloc[-1] == pytest.approx(1.0)
    assert "delisted" in set(full.events["event"])

    haircut = _run(market, Scripted({"A": 1.0}), delisting_recovery=0.5)
    assert haircut.equity["equity"].iloc[-1] == pytest.approx(400_000.0)
    assert haircut.trades.iloc[-1]["realized_pnl"] == pytest.approx(-600_000.0)


# -------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ({"A": -0.1}, "short selling"),
        ({"A": 0.7, "B": 0.6}, "leverage"),
        ({"A": float("nan")}, "NaN"),
        ({"ZZZ": 0.5}, "outside the tradable universe"),
    ],
)
def test_untradable_targets_are_rejected(target, message):
    market = make_market({"A": [10, 10, 10], "B": [5, 5, 5]})
    with pytest.raises(StrategyError, match=message):
        _run(market, Scripted(target))


def test_window_needs_two_sessions():
    market = make_market({"A": [10, 10, 10]})
    with pytest.raises(ValueError, match="at least two sessions"):
        simulate(
            market,
            Scripted(),
            start=market.sessions[0],
            end=market.sessions[0],
            decision_dates=market.sessions,
            cost_model=FREE,
        )


def test_invalid_simulation_config_is_rejected():
    with pytest.raises(ValueError):
        SimulationConfig(initial_capital=0)
    with pytest.raises(ValueError):
        SimulationConfig(participation_limit=1.5)
    with pytest.raises(ValueError):
        SimulationConfig(delisting_recovery=-0.1)


def test_strategy_context_contains_nothing_after_the_decision_date():
    market = make_market({"A": [10, 11, 12, 13, 14], "B": [5, 5, 5, 5, 5]})
    strategy = Scripted(None, None, None, None)
    _run(market, strategy)
    assert len(strategy.contexts) == 4  # the final session's decision is skipped
    for ctx in strategy.contexts:
        assert ctx.prices.index.max() == ctx.date
        assert ctx.current_weights.sum() == pytest.approx(1.0)
    assert [len(ctx.prices) for ctx in strategy.contexts] == [1, 2, 3, 4]


def test_weights_and_equity_identity_hold_every_day():
    market = make_market(
        {"A": [10, 10, 12, 9, 11, 13], "B": [20, 21, 19, 22, 25, 24]},
        {"A": [10, 11, 10, 10, 12, 13], "B": [20, 20, 21, 24, 24, 23]},
    )
    result = _run(
        market,
        Scripted({"A": 0.5, "B": 0.5}, {"A": 0.2, "B": 0.7}, {"B": 1.0}, {"A": 1.0}, None),
        costs=_stt_only(0.001, slippage_bps=5),
    )
    np.testing.assert_allclose(result.weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        result.equity["equity"], result.equity["cash"] + result.equity["invested"]
    )
    assert result.equity["cash"].min() >= -1e-6
    assert isinstance(result.equity.index, pd.DatetimeIndex)
