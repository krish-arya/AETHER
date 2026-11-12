"""Phase 1 persistence models: securities, daily bars, corporate actions, features.

Prices in ``market_data`` are stored *as traded* (unadjusted). ``adj_factor`` and
``split_factor`` are the back-adjustment multipliers computed by the data pipeline as
of the owning dataset version, so adjusted series can always be re-derived and audited.
Every row carries ``is_synthetic`` so mock development data can never be mistaken for
real market data.
"""

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class DatasetVersion(Base):
    """One immutable, content-addressed dataset snapshot produced by the pipeline."""

    __tablename__ = "dataset_versions"

    dataset_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    provider: Mapped[str] = mapped_column(String(64))
    is_synthetic: Mapped[bool] = mapped_column(Boolean)
    license_note: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    config_hash: Mapped[str] = mapped_column(String(64))
    feature_set_version: Mapped[str] = mapped_column(String(16))
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    as_of: Mapped[date] = mapped_column(Date)
    symbol_count: Mapped[int] = mapped_column(Integer)
    bar_count: Mapped[int] = mapped_column(Integer)
    quality_summary: Mapped[dict] = mapped_column(JSON)
    manifest_path: Mapped[str] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    loaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Security(Base):
    __tablename__ = "securities"

    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    sector: Mapped[str | None] = mapped_column(String(64))
    industry: Mapped[str | None] = mapped_column(String(128))
    isin: Mapped[str | None] = mapped_column(String(12))
    # Listing window drives point-in-time universe reconstruction (survivorship control).
    listing_date: Mapped[date | None] = mapped_column(Date)
    delisting_date: Mapped[date | None] = mapped_column(Date)
    is_synthetic: Mapped[bool] = mapped_column(Boolean)
    source: Mapped[str] = mapped_column(String(64))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.dataset_id"))


class MarketData(Base):
    __tablename__ = "market_data"

    symbol: Mapped[str] = mapped_column(ForeignKey("securities.symbol"), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True, index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    adjusted_close: Mapped[float] = mapped_column(Float)
    adj_factor: Mapped[float] = mapped_column(Float)
    split_factor: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(BigInteger)
    turnover: Mapped[float | None] = mapped_column(Float)
    delivery_volume: Mapped[int | None] = mapped_column(BigInteger)
    market_cap: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(64))
    is_synthetic: Mapped[bool] = mapped_column(Boolean)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.dataset_id"))
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class CorporateAction(Base):
    __tablename__ = "corporate_actions"
    __table_args__ = (UniqueConstraint("symbol", "ex_date", "action_type"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(ForeignKey("securities.symbol"), index=True)
    ex_date: Mapped[date] = mapped_column(Date)
    # split | bonus | dividend
    action_type: Mapped[str] = mapped_column(String(16))
    # split/bonus: shares held after the action per share held before (e.g. 2.0).
    # dividend: cash amount per share in INR.
    value: Mapped[float] = mapped_column(Float)
    announced_date: Mapped[date | None] = mapped_column(Date)
    source: Mapped[str] = mapped_column(String(64))
    is_synthetic: Mapped[bool] = mapped_column(Boolean)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.dataset_id"))


class MarketRegime(Base):
    """Market-wide regime per session for one dataset (ml.features.regime)."""

    __tablename__ = "market_regimes"

    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_versions.dataset_id"), primary_key=True
    )
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    # None during the warm-up window, before enough volatility history exists.
    regime: Mapped[str | None] = mapped_column(String(24))
    confidence: Mapped[float | None] = mapped_column(Float)
    volatility_level: Mapped[float | None] = mapped_column(Float)
    trend_strength: Mapped[float | None] = mapped_column(Float)
    realized_vol: Mapped[float | None] = mapped_column(Float)
    trend_return: Mapped[float | None] = mapped_column(Float)
    drawdown: Mapped[float | None] = mapped_column(Float)


class TechnicalFeature(Base):
    __tablename__ = "technical_features"

    symbol: Mapped[str] = mapped_column(ForeignKey("securities.symbol"), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True, index=True)
    feature_set_version: Mapped[str] = mapped_column(String(16), primary_key=True)
    # {feature_name: value | null}. Null means "undefined at this timestamp"
    # (warm-up window or mathematically undefined) -- never a filled value.
    features: Mapped[dict] = mapped_column(JSON)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.dataset_id"))
