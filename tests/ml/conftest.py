"""Shared fixtures for the ML / data pipeline tests.

All fixtures use the synthetic development provider, so tests need no network, no
licensed data and no external services.

Symbols in the 4-symbol provider:
    SYNTH01  plain history with annual dividends
    SYNTH02  has a split
    SYNTH03  lists part-way through the history
    SYNTH04  has a bonus issue and delists before the history ends
"""

from datetime import date

import pandas as pd
import pytest

from ml.data.config import DataConfig
from ml.data.ingestion.synthetic import SyntheticDevProvider
from ml.data.validation.calendar import TradingCalendar

START, END = "2020-01-01", "2021-12-31"


@pytest.fixture(scope="session")
def provider() -> SyntheticDevProvider:
    return SyntheticDevProvider(seed=11, n_symbols=4, history_start=START, history_end=END)


@pytest.fixture(scope="session")
def securities(provider) -> pd.DataFrame:
    return provider.list_securities()


@pytest.fixture(scope="session")
def symbols(securities) -> list[str]:
    return list(securities["symbol"])


@pytest.fixture(scope="session")
def calendar(provider) -> TradingCalendar:
    return TradingCalendar.from_sessions(provider.get_trading_sessions(START, END), "provider")


@pytest.fixture
def raw_bars(provider, symbols) -> pd.DataFrame:
    """A fresh copy per test so tests can inject defects freely."""
    return provider.get_daily_bars(symbols, START, END).copy()


@pytest.fixture(scope="session")
def raw_actions(provider, symbols) -> pd.DataFrame:
    return provider.get_corporate_actions(symbols, START, END)


@pytest.fixture(scope="session")
def market(tmp_path_factory):
    """Four years of the synthetic development market (6 symbols) as a ``MarketData``
    view, shared by the model tests (RL, GAN, experiments)."""
    from ml.backtesting.data import MarketData
    from ml.data.pipeline import build_dataset

    cfg = DataConfig.model_validate(
        {
            "dataset": {
                "name": "model-test",
                "start_date": date(2018, 1, 1),
                "end_date": date(2021, 12, 31),
            },
            "provider": {"kind": "synthetic_dev", "synthetic_dev": {"seed": 3, "n_symbols": 6}},
        }
    )
    built = build_dataset(cfg, output_root=tmp_path_factory.mktemp("model-datasets"))
    return MarketData.from_snapshot(built.path)


def make_config(**dataset_overrides) -> DataConfig:
    dataset = {
        "name": "test-synthetic",
        "start_date": date(2020, 1, 1),
        "end_date": date(2021, 12, 31),
        **dataset_overrides,
    }
    return DataConfig.model_validate(
        {
            "dataset": dataset,
            "provider": {"kind": "synthetic_dev", "synthetic_dev": {"seed": 11, "n_symbols": 4}},
        }
    )


@pytest.fixture
def config() -> DataConfig:
    return make_config()
