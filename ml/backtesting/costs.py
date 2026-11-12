"""Transaction-cost model for Indian cash-equity **delivery** trades (PRD 11.2).

Charges per order, on the traded value ``V``:

    brokerage      min(max(rate x V, min_per_order), max_per_order)
    STT            stt_buy / stt_sell x V
    exchange       exchange_txn x V          (exchange transaction charge)
    SEBI fee       sebi_fee x V              (turnover fee)
    stamp duty     stamp_duty_buy / stamp_duty_sell x V
    GST            gst x (brokerage + exchange + SEBI fee)
    flat           flat_per_sell_order       (e.g. depository charge), sells only

Statutory rates change over time, so they are supplied as a dated schedule and the
rate in force on the trade date is used. **No rate is hardcoded here** -- they live in
``configs/backtesting.yaml``. A trade dated before the first schedule entry is an
error rather than a silent default.

Slippage is not a fee: it is modelled by the simulator as an adverse move of the
execution price (``slippage_bps``), and reported separately.

Not modelled: intraday / F&O products, state-specific stamp duty before July 2020
beyond what the schedule states, broker-specific minimums, market-impact beyond the
flat slippage assumption, taxes on gains.
"""

from dataclasses import dataclass, fields
from datetime import date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

Side = Literal["buy", "sell"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StatutoryRates(_Strict):
    """Rates in force from ``effective_from`` (fractions of traded value, not percent)."""

    effective_from: date
    stt_buy: float = Field(ge=0)
    stt_sell: float = Field(ge=0)
    exchange_txn: float = Field(ge=0)
    sebi_fee: float = Field(ge=0)
    stamp_duty_buy: float = Field(ge=0)
    stamp_duty_sell: float = Field(ge=0)
    gst: float = Field(ge=0)


class BrokerageConfig(_Strict):
    rate: float = Field(0.0, ge=0)
    min_per_order: float = Field(0.0, ge=0)
    max_per_order: float | None = Field(None, ge=0)


class CostConfig(_Strict):
    slippage_bps: float = Field(0.0, ge=0)
    brokerage: BrokerageConfig = BrokerageConfig()
    flat_per_sell_order: float = Field(0.0, ge=0)
    statutory: list[StatutoryRates] = Field(min_length=1)

    @model_validator(mode="after")
    def _sorted_schedule(self):
        dates = [entry.effective_from for entry in self.statutory]
        if dates != sorted(set(dates)):
            raise ValueError("statutory schedule must have unique, ascending effective_from")
        return self


@dataclass(frozen=True)
class CostBreakdown:
    brokerage: float = 0.0
    stt: float = 0.0
    exchange_txn: float = 0.0
    sebi_fee: float = 0.0
    stamp_duty: float = 0.0
    gst: float = 0.0
    flat: float = 0.0

    @property
    def total(self) -> float:
        return sum(getattr(self, f.name) for f in fields(self))

    def as_dict(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


COST_COMPONENTS = [f.name for f in fields(CostBreakdown)]


class CostModel:
    def __init__(self, config: CostConfig) -> None:
        self.config = config
        self._starts = pd.DatetimeIndex([pd.Timestamp(e.effective_from) for e in config.statutory])

    @classmethod
    def zero(cls) -> "CostModel":
        """Frictionless model -- for tests and for isolating the effect of costs."""
        rates = StatutoryRates(
            effective_from=date(1900, 1, 1),
            stt_buy=0,
            stt_sell=0,
            exchange_txn=0,
            sebi_fee=0,
            stamp_duty_buy=0,
            stamp_duty_sell=0,
            gst=0,
        )
        return cls(CostConfig(statutory=[rates]))

    @property
    def slippage_rate(self) -> float:
        return self.config.slippage_bps / 10_000

    def rates_on(self, trade_date) -> StatutoryRates:
        position = self._starts.searchsorted(pd.Timestamp(trade_date), side="right") - 1
        if position < 0:
            raise ValueError(
                f"No statutory cost rates configured for {pd.Timestamp(trade_date).date()} "
                f"(schedule starts {self._starts[0].date()})"
            )
        return self.config.statutory[position]

    def charges(self, side: Side, traded_value: float, trade_date) -> CostBreakdown:
        if traded_value <= 0:
            return CostBreakdown()
        rates = self.rates_on(trade_date)
        broker = self.config.brokerage
        brokerage = max(broker.rate * traded_value, broker.min_per_order)
        if broker.max_per_order is not None:
            brokerage = min(brokerage, broker.max_per_order)
        exchange = rates.exchange_txn * traded_value
        sebi = rates.sebi_fee * traded_value
        is_buy = side == "buy"
        return CostBreakdown(
            brokerage=brokerage,
            stt=(rates.stt_buy if is_buy else rates.stt_sell) * traded_value,
            exchange_txn=exchange,
            sebi_fee=sebi,
            stamp_duty=(rates.stamp_duty_buy if is_buy else rates.stamp_duty_sell) * traded_value,
            gst=rates.gst * (brokerage + exchange + sebi),
            flat=0.0 if is_buy else self.config.flat_per_sell_order,
        )
