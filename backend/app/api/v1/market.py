"""Market data endpoints (PRD 15.1), served from datasets loaded into the database.

Every response states whether its rows are synthetic development data. Endpoints
that depend on later phases (``/market/overview``) are not registered until those
modules exist.
"""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_session
from app.models import DatasetVersion, MarketData, MarketRegime, Security, TechnicalFeature
from app.schemas.market import (
    BarOut,
    DatasetVersionOut,
    FeatureRow,
    HistoryResponse,
    IndicatorsResponse,
    RegimePoint,
    RegimeResponse,
    SecurityDetailOut,
    SecurityOut,
)

router = APIRouter(prefix="/market", tags=["market"])

MAX_ROWS = 5000


async def _get_security(session: AsyncSession, symbol: str) -> Security:
    security = await session.get(Security, symbol.strip().upper())
    if security is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown symbol '{symbol}'")
    return security


def _validate_range(start: date | None, end: date | None) -> None:
    if start and end and start > end:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "start must not be after end")


@router.get("/datasets", response_model=list[DatasetVersionOut])
async def list_datasets(session: AsyncSession = Depends(get_session)) -> list[DatasetVersion]:
    """Dataset versions loaded into the database, newest first."""
    result = await session.scalars(select(DatasetVersion).order_by(DatasetVersion.loaded_at.desc()))
    return list(result)


@router.get("/stocks", response_model=list[SecurityOut])
async def list_stocks(
    as_of: date | None = Query(
        None, description="Return only securities listed (and not delisted) on this date"
    ),
    dataset_id: str | None = Query(None, description="Only securities of this dataset"),
    session: AsyncSession = Depends(get_session),
) -> list[Security]:
    stmt = select(Security).order_by(Security.symbol)
    if dataset_id is not None:
        stmt = stmt.where(Security.dataset_id == dataset_id)
    if as_of is not None:
        stmt = stmt.where(
            (Security.listing_date.is_(None)) | (Security.listing_date <= as_of),
            (Security.delisting_date.is_(None)) | (Security.delisting_date > as_of),
        )
    return list(await session.scalars(stmt))


@router.get("/stocks/{symbol}", response_model=SecurityDetailOut)
async def get_stock(symbol: str, session: AsyncSession = Depends(get_session)):
    security = await _get_security(session, symbol)
    first, last, count = (
        await session.execute(
            select(
                func.min(MarketData.trade_date),
                func.max(MarketData.trade_date),
                func.count(),
            ).where(MarketData.symbol == security.symbol)
        )
    ).one()
    latest = None
    if last is not None:
        latest = await session.get(MarketData, (security.symbol, last))
    return SecurityDetailOut(
        **SecurityOut.model_validate(security).model_dump(),
        first_trade_date=first,
        last_trade_date=last,
        bar_count=count,
        latest_bar=BarOut.model_validate(latest) if latest else None,
    )


@router.get("/stocks/{symbol}/history", response_model=HistoryResponse)
async def get_history(
    symbol: str,
    start: date | None = None,
    end: date | None = None,
    limit: int = Query(500, ge=1, le=MAX_ROWS, description="Most recent N bars in range"),
    session: AsyncSession = Depends(get_session),
) -> HistoryResponse:
    _validate_range(start, end)
    security = await _get_security(session, symbol)
    stmt = select(MarketData).where(MarketData.symbol == security.symbol)
    if start:
        stmt = stmt.where(MarketData.trade_date >= start)
    if end:
        stmt = stmt.where(MarketData.trade_date <= end)
    rows = list(await session.scalars(stmt.order_by(MarketData.trade_date.desc()).limit(limit)))
    rows.reverse()
    return HistoryResponse(
        symbol=security.symbol,
        is_synthetic=security.is_synthetic,
        source=security.source,
        count=len(rows),
        bars=[BarOut.model_validate(row) for row in rows],
    )


@router.get("/regime", response_model=RegimeResponse)
async def get_regime(
    dataset_id: str | None = Query(None, description="Defaults to the latest loaded dataset"),
    start: date | None = None,
    end: date | None = None,
    limit: int = Query(250, ge=1, le=MAX_ROWS, description="Most recent N sessions in range"),
    session: AsyncSession = Depends(get_session),
) -> RegimeResponse:
    """Market regime per session: a rule-based, causal classification of the market
    proxy (see ``ml/features/regime.py``). ``confidence`` is a margin score, not a
    probability."""
    _validate_range(start, end)
    if dataset_id is None:
        dataset = await session.scalar(
            select(DatasetVersion).order_by(DatasetVersion.loaded_at.desc()).limit(1)
        )
    else:
        dataset = await session.get(DatasetVersion, dataset_id)
    if dataset is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such dataset is loaded")

    stmt = select(MarketRegime).where(MarketRegime.dataset_id == dataset.dataset_id)
    if start:
        stmt = stmt.where(MarketRegime.trade_date >= start)
    if end:
        stmt = stmt.where(MarketRegime.trade_date <= end)
    rows = list(await session.scalars(stmt.order_by(MarketRegime.trade_date.desc()).limit(limit)))
    if not rows:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Dataset '{dataset.dataset_id}' has no regime data for this range",
        )
    rows.reverse()
    history = [
        RegimePoint(
            timestamp=row.trade_date,
            regime=row.regime,
            confidence=row.confidence,
            volatility_level=row.volatility_level,
            trend_strength=row.trend_strength,
            realized_vol=row.realized_vol,
            trend_return=row.trend_return,
            drawdown=row.drawdown,
        )
        for row in rows
    ]
    return RegimeResponse(
        dataset_id=dataset.dataset_id,
        is_synthetic=dataset.is_synthetic,
        method="volatility_trend_rules",
        current=history[-1],
        history=history,
    )


@router.get("/indicators/{symbol}", response_model=IndicatorsResponse)
async def get_indicators(
    symbol: str,
    start: date | None = None,
    end: date | None = None,
    feature_set_version: str | None = Query(
        None, description="Defaults to the most recently loaded feature set"
    ),
    limit: int = Query(250, ge=1, le=MAX_ROWS, description="Most recent N rows in range"),
    session: AsyncSession = Depends(get_session),
) -> IndicatorsResponse:
    _validate_range(start, end)
    security = await _get_security(session, symbol)

    if feature_set_version is None:
        feature_set_version = await session.scalar(
            select(TechnicalFeature.feature_set_version)
            .join(DatasetVersion, DatasetVersion.dataset_id == TechnicalFeature.dataset_id)
            .where(TechnicalFeature.symbol == security.symbol)
            .order_by(DatasetVersion.loaded_at.desc())
            .limit(1)
        )
        if feature_set_version is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"No features loaded for '{security.symbol}'"
            )

    stmt = select(TechnicalFeature).where(
        TechnicalFeature.symbol == security.symbol,
        TechnicalFeature.feature_set_version == feature_set_version,
    )
    if start:
        stmt = stmt.where(TechnicalFeature.trade_date >= start)
    if end:
        stmt = stmt.where(TechnicalFeature.trade_date <= end)
    rows = list(
        await session.scalars(stmt.order_by(TechnicalFeature.trade_date.desc()).limit(limit))
    )
    if not rows:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No features for '{security.symbol}' with feature set '{feature_set_version}'",
        )
    rows.reverse()
    return IndicatorsResponse(
        symbol=security.symbol,
        is_synthetic=security.is_synthetic,
        feature_set_version=feature_set_version,
        dataset_id=rows[-1].dataset_id,
        feature_names=sorted(rows[-1].features),
        count=len(rows),
        rows=[FeatureRow(trade_date=row.trade_date, features=row.features) for row in rows],
    )
