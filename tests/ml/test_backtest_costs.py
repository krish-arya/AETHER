"""Transaction-cost model against hand-computed charges."""

from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ml.backtesting.costs import CostConfig, CostModel

REPO_ROOT = Path(__file__).resolve().parents[2]


def _rates(effective_from, **overrides):
    base = {
        "effective_from": effective_from,
        "stt_buy": 0.001,
        "stt_sell": 0.001,
        "exchange_txn": 0.0000325,
        "sebi_fee": 0.000001,
        "stamp_duty_buy": 0.00015,
        "stamp_duty_sell": 0.0,
        "gst": 0.18,
    }
    return {**base, **overrides}


@pytest.fixture
def model() -> CostModel:
    return CostModel(
        CostConfig.model_validate(
            {
                "slippage_bps": 5,
                "brokerage": {"rate": 0.0003, "max_per_order": 20.0},
                "flat_per_sell_order": 15.0,
                "statutory": [
                    _rates(date(2017, 7, 1), stamp_duty_buy=0.0001, stamp_duty_sell=0.0001),
                    _rates(date(2020, 7, 1)),
                    _rates(date(2024, 10, 1), exchange_txn=0.0000297),
                ],
            }
        )
    )


def test_buy_charges_hand_computed(model):
    costs = model.charges("buy", 100_000.0, "2021-03-15")
    assert costs.brokerage == pytest.approx(20.0)  # 0.03% = 30, capped at 20
    assert costs.stt == pytest.approx(100.0)  # 0.1%
    assert costs.exchange_txn == pytest.approx(3.25)
    assert costs.sebi_fee == pytest.approx(0.10)
    assert costs.stamp_duty == pytest.approx(15.0)  # 0.015%, buy side
    assert costs.gst == pytest.approx(0.18 * (20.0 + 3.25 + 0.10))  # 4.203
    assert costs.flat == 0.0
    assert costs.total == pytest.approx(142.553)


def test_sell_charges_hand_computed(model):
    costs = model.charges("sell", 100_000.0, "2021-03-15")
    assert costs.stt == pytest.approx(100.0)
    assert costs.stamp_duty == 0.0  # buy side only since July 2020
    assert costs.flat == 15.0
    assert costs.total == pytest.approx(20.0 + 100.0 + 3.25 + 0.10 + 4.203 + 15.0)


def test_small_order_pays_proportional_brokerage(model):
    costs = model.charges("buy", 10_000.0, "2021-03-15")
    assert costs.brokerage == pytest.approx(3.0)  # below the cap


def test_rates_follow_the_dated_schedule(model):
    before = model.charges("sell", 100_000.0, "2019-06-03")
    assert before.stamp_duty == pytest.approx(10.0)  # pre-July-2020 schedule entry
    boundary = model.charges("sell", 100_000.0, "2020-07-01")
    assert boundary.stamp_duty == 0.0  # effective on the date itself
    assert model.charges("buy", 100_000.0, "2024-09-30").exchange_txn == pytest.approx(3.25)
    assert model.charges("buy", 100_000.0, "2024-10-01").exchange_txn == pytest.approx(2.97)


def test_trade_before_the_schedule_is_an_error_not_a_default(model):
    with pytest.raises(ValueError, match="No statutory cost rates configured"):
        model.charges("buy", 100_000.0, "2015-01-05")


def test_no_trade_no_cost(model):
    assert model.charges("buy", 0.0, "2021-03-15").total == 0.0
    assert model.charges("sell", -5.0, "2021-03-15").total == 0.0


def test_zero_model_is_frictionless():
    zero = CostModel.zero()
    assert zero.charges("sell", 1e9, "1999-01-01").total == 0.0
    assert zero.slippage_rate == 0.0


def test_slippage_rate(model):
    assert model.slippage_rate == pytest.approx(0.0005)


def test_schedule_must_be_ascending_and_unique():
    with pytest.raises(ValidationError, match="ascending"):
        CostConfig.model_validate(
            {"statutory": [_rates(date(2020, 7, 1)), _rates(date(2017, 7, 1))]}
        )
    with pytest.raises(ValidationError):
        CostConfig.model_validate({"statutory": []})
    with pytest.raises(ValidationError):
        CostConfig.model_validate({"statutory": [_rates(date(2020, 7, 1), stt_buy=-0.1)]})


def test_shipped_cost_schedule_loads_and_covers_the_default_dataset():
    raw = yaml.safe_load((REPO_ROOT / "configs" / "backtesting.yaml").read_text())
    model = CostModel(CostConfig.model_validate(raw["costs"]))
    for day in ("2015-01-01", "2018-01-01", "2021-01-01", "2024-12-31"):
        assert model.charges("buy", 100_000.0, day).total > 100.0  # STT alone is 100
    # Round trip of a 1,00,000 position costs roughly 0.25% including slippage.
    buy = model.charges("buy", 100_000.0, "2024-12-31").total
    sell = model.charges("sell", 100_000.0, "2024-12-31").total
    slippage = 2 * 100_000.0 * model.slippage_rate
    assert 0.002 < (buy + sell + slippage) / 100_000.0 < 0.005
