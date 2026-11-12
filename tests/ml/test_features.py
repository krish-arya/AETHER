import numpy as np
import pandas as pd
import pytest

from ml.data.preprocessing.clean import clean_bars
from ml.data.preprocessing.corporate_actions import apply_adjustments, validate_actions
from ml.data.validation.report import DataQualityReport
from ml.features.config import FeatureConfig
from ml.features.engine import (
    compute_features,
    feature_columns,
    market_proxy_returns,
    validate_features,
)

from .conftest import END

CFG = FeatureConfig()


@pytest.fixture(scope="module")
def adjusted(provider, symbols, securities, calendar):
    bars = provider.get_daily_bars(symbols, "2020-01-01", END)
    cleaned = clean_bars(bars, securities=securities, calendar=calendar, as_of=END)
    actions = validate_actions(
        provider.get_corporate_actions(symbols, "2020-01-01", END),
        known_symbols=set(symbols),
        as_of=END,
        report=DataQualityReport(),
    )
    return apply_adjustments(cleaned.bars, actions, DataQualityReport())


@pytest.fixture(scope="module")
def features(adjusted):
    return compute_features(adjusted, CFG)


def test_one_feature_row_per_bar(adjusted, features):
    assert len(features) == len(adjusted)
    pd.testing.assert_frame_equal(
        features[["symbol", "trade_date"]], adjusted[["symbol", "trade_date"]]
    )


def test_feature_library_covers_every_prd_indicator_family(features):
    columns = set(feature_columns(features))
    expected = {
        # trend
        "sma_20_ratio", "sma_50_ratio", "ema_12_ratio", "ema_26_ratio",
        "macd_pct", "macd_signal_pct", "macd_hist_pct", "adx_14", "plus_di_14", "minus_di_14",
        # momentum
        "rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "williams_r_14",
        # volatility
        "atr_pct_14", "bb_pctb_20", "bb_width_20", "hist_vol_20d", "hist_vol_60d", "ret_std_20d",
        # volume
        "obv_norm_20", "vwap_ratio_20", "volume_change_1d", "mfi_14",
        # returns
        "ret_1d", "log_ret_1d", "cum_ret_5d", "cum_ret_21d", "cum_ret_63d",
        "beta_60d", "corr_mkt_60d",
    }  # fmt: skip
    assert columns == expected


def test_windows_are_configurable_and_change_the_version(adjusted):
    custom = FeatureConfig(sma_windows=[10], rsi_window=7, cum_return_windows=[3])
    columns = set(feature_columns(compute_features(adjusted, custom)))
    assert {"sma_10_ratio", "rsi_7", "cum_ret_3d"} <= columns
    assert "sma_20_ratio" not in columns and "rsi_14" not in columns
    assert custom.version() != CFG.version()
    assert FeatureConfig().version() == CFG.version()  # stable for equal configs


def test_returns_match_adjusted_closes(adjusted, features):
    for symbol, group in adjusted.groupby("symbol"):
        expected = group["adj_close"].pct_change().to_numpy()
        actual = features.loc[features["symbol"] == symbol, "ret_1d"].to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-12, equal_nan=True)
        assert np.isnan(actual[0])  # never carried across symbols


def test_warm_up_rows_are_nan_not_filled(features):
    for _, group in features.groupby("symbol"):
        assert group["sma_50_ratio"].iloc[:49].isna().all()
        assert group["sma_50_ratio"].iloc[49:].notna().all()
        assert group["cum_ret_63d"].iloc[:63].isna().all()
        assert group["beta_60d"].iloc[:60].isna().all()
        assert group["beta_60d"].iloc[60:].notna().all()


def test_market_proxy_is_cross_sectional_mean(adjusted):
    market = market_proxy_returns(adjusted)
    day = adjusted["trade_date"].unique()[300]
    todays = adjusted[adjusted["trade_date"] == day]
    returns = []
    for symbol in todays["symbol"]:
        series = adjusted[adjusted["symbol"] == symbol].set_index("trade_date")["adj_close"]
        returns.append(series.pct_change().loc[day])
    assert market.loc[day] == pytest.approx(np.nanmean(returns))


def test_synthetic_features_pass_validation(features):
    report = DataQualityReport()
    validate_features(features, report)
    assert not report.has("feature_infinite")
    assert not report.has("feature_out_of_bounds")
    assert report.has("feature_undefined")  # warm-up NaNs are counted, not hidden


def test_validation_catches_infinite_and_out_of_range_values(features):
    broken = features.copy()
    broken.loc[broken.index[100], "rsi_14"] = 140.0
    broken.loc[broken.index[101], "macd_pct"] = np.inf
    report = DataQualityReport()
    validate_features(broken, report)
    assert report.count("feature_out_of_bounds") == 1
    assert report.count("feature_infinite") == 1


def test_duplicate_bars_are_rejected(adjusted):
    with pytest.raises(ValueError, match="duplicate"):
        compute_features(pd.concat([adjusted, adjusted.iloc[[0]]]), CFG)


def test_missing_adjusted_columns_are_rejected(adjusted):
    with pytest.raises(ValueError, match="missing columns"):
        compute_features(adjusted.drop(columns=["adj_close"]), CFG)
