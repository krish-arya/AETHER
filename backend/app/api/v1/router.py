from fastapi import APIRouter

from app.api.v1 import alerts, backtest, health, market, models_api, news, portfolio

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(alerts.router)
api_router.include_router(market.router)
api_router.include_router(backtest.router)
api_router.include_router(news.router)
api_router.include_router(models_api.rl_router)
api_router.include_router(models_api.gan_router)
api_router.include_router(models_api.experiments_router)
api_router.include_router(models_api.jobs_router)
api_router.include_router(portfolio.portfolio_router)
api_router.include_router(portfolio.risk_router)
api_router.include_router(portfolio.paper_router)
