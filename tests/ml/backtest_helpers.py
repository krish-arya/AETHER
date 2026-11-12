"""Tiny hand-built markets and scripted strategies for backtester tests."""

import numpy as np
import pandas as pd

from ml.backtesting.data import MarketData
from ml.backtesting.strategies import Strategy


def make_market(
    open_: dict[str, list[float]],
    close: dict[str, list[float]] | None = None,
    *,
    start: str = "2024-01-01",
    traded_value: float = 1e12,
    listing: dict[str, str] | None = None,
    delisting: dict[str, str] | None = None,
) -> MarketData:
    """Market from per-symbol price lists on consecutive business days.

    ``None`` / NaN entries mean "no bar that day". ``close`` defaults to ``open``.
    Prices are used as both raw and adjusted (no corporate actions).
    """
    close = close or open_
    length = len(next(iter(open_.values())))
    dates = pd.bdate_range(start, periods=length)
    rows = []
    for symbol in open_:
        for day, o, c in zip(dates, open_[symbol], close[symbol], strict=True):
            if o is None or c is None or np.isnan(o) or np.isnan(c):
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": day,
                    "close": c,
                    "adj_open": o,
                    "adj_close": c,
                    "volume": 1,
                    "turnover": traded_value,
                }
            )
    securities = pd.DataFrame(
        {
            "symbol": list(open_),
            "listing_date": [pd.Timestamp((listing or {}).get(s, pd.NaT)) for s in open_],
            "delisting_date": [pd.Timestamp((delisting or {}).get(s, pd.NaT)) for s in open_],
        }
    )
    return MarketData.from_frames(pd.DataFrame(rows), securities)


class Scripted(Strategy):
    """Returns pre-set weights by decision index; ``None`` entries mean "hold"."""

    name = "scripted"

    def __init__(self, *targets: dict[str, float] | None) -> None:
        self.targets = list(targets)
        self.contexts = []

    def target_weights(self, ctx):
        self.contexts.append(ctx)
        position = len(self.contexts) - 1
        if position >= len(self.targets):
            return None
        target = self.targets[position]
        return None if target is None else pd.Series(target, dtype="float64")
