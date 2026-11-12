from datetime import date, datetime

from pydantic import BaseModel, ConfigDict


class DatasetVersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    dataset_id: str
    name: str
    provider: str
    is_synthetic: bool
    license_note: str
    content_hash: str
    config_hash: str
    feature_set_version: str
    start_date: date
    end_date: date
    as_of: date
    symbol_count: int
    bar_count: int
    quality_summary: dict
    created_at: datetime
    loaded_at: datetime


class SecurityOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    name: str
    sector: str | None
    industry: str | None
    isin: str | None
    listing_date: date | None
    delisting_date: date | None
    is_synthetic: bool
    source: str
    dataset_id: str


class BarOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    trade_date: date
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: int
    turnover: float | None
    delivery_volume: int | None
    market_cap: float | None


class SecurityDetailOut(SecurityOut):
    first_trade_date: date | None
    last_trade_date: date | None
    bar_count: int
    latest_bar: BarOut | None


class HistoryResponse(BaseModel):
    symbol: str
    is_synthetic: bool
    source: str
    count: int
    bars: list[BarOut]


class FeatureRow(BaseModel):
    trade_date: date
    features: dict[str, float | None]


class RegimePoint(BaseModel):
    """One session's market regime (PRD 6.3 output schema)."""

    timestamp: date
    regime: str | None
    confidence: float | None
    volatility_level: float | None
    trend_strength: float | None
    realized_vol: float | None
    trend_return: float | None
    drawdown: float | None


class RegimeResponse(BaseModel):
    dataset_id: str
    is_synthetic: bool
    method: str
    current: RegimePoint | None
    history: list[RegimePoint]


class IndicatorsResponse(BaseModel):
    symbol: str
    is_synthetic: bool
    feature_set_version: str
    dataset_id: str
    feature_names: list[str]
    count: int
    rows: list[FeatureRow]
