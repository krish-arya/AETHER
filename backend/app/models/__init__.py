"""ORM models. Importing this package registers every table on ``Base.metadata``."""

from app.models.backtest import BacktestRun
from app.models.market import (
    CorporateAction,
    DatasetVersion,
    MarketData,
    MarketRegime,
    Security,
    TechnicalFeature,
)
from app.models.news import FinancialNews, NewsEvent
from app.models.platform import Allocation, Job, PaperAccount, PaperOrder

__all__ = [
    "Allocation",
    "BacktestRun",
    "CorporateAction",
    "DatasetVersion",
    "FinancialNews",
    "Job",
    "MarketData",
    "MarketRegime",
    "NewsEvent",
    "PaperAccount",
    "PaperOrder",
    "Security",
    "TechnicalFeature",
]
