from fastapi import APIRouter

from app.api.v1 import backtest, health, market, news

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(market.router)
api_router.include_router(backtest.router)
api_router.include_router(news.router)
