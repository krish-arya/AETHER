"""Look-ahead bias tests (PRD 6.2, 10.3, 19 "Financial Correctness Testing").

Three independent ways of asserting that nothing at time T depends on data after T:

1. every indicator, run on a truncated series, reproduces the prefix of the full run;
2. corrupting all bars after T leaves every feature dated <= T bit-for-bit unchanged;
3. rebuilding the dataset as it would have been known at T (bars *and* corporate
   actions truncated, so back-adjustment factors differ) reproduces the same features.
"""

import numpy as np
import pandas as pd
import pytest

from ml.data.preprocessing.clean import clean_bars
from ml.data.preprocessing.corporate_actions import apply_adjustments, validate_actions
from ml.data.validation.report import DataQualityReport
from ml.features import indicators as ind
from ml.features.config import FeatureConfig
from ml.features.engine import compute_features, feature_columns

from .conftest import END

CFG = FeatureConfig()


def _adjust(raw_bars, raw_actions, securities, calendar, as_of=END):
    cleaned = clean_bars(raw_bars, securities=securities, calendar=calendar, as_of=as_of)
    actions = validate_actions(
        raw_actions,
        known_symbols=set(cleaned.bars["symbol"]),
        as_of=as_of,
        report=DataQualityReport(),
    )
    return apply_adjustments(cleaned.bars, actions, DataQualityReport())


@pytest.fixture(scope="module")
def adjusted(provider, symbols, securities, calendar):
    bars = provider.get_daily_bars(symbols, "2020-01-01", END)
    actions = provider.get_corporate_actions(symbols, "2020-01-01", END)
    return _adjust(bars, actions, securities, calendar)


@pytest.fixture(scope="module")
def full_features(adjusted):
    return compute_features(adjusted, CFG)


def _series(adjusted, symbol="SYNTH01"):
    g = adjusted[adjusted["symbol"] == symbol].reset_index(drop=True)
    return g["adj_high"], g["adj_low"], g["adj_close"], g["adj_volume"]


INDICATORS = {
    "sma": lambda h, lo, c, v: ind.sma(c, 20),
    "ema": lambda h, lo, c, v: ind.ema(c, 12),
    "macd": lambda h, lo, c, v: ind.macd(c)[2],
    "adx": lambda h, lo, c, v: ind.adx(h, lo, c, 14)[0],
    "rsi": lambda h, lo, c, v: ind.rsi(c, 14),
    "stochastic": lambda h, lo, c, v: ind.stochastic(h, lo, c)[1],
    "roc": lambda h, lo, c, v: ind.roc(c, 10),
    "williams_r": lambda h, lo, c, v: ind.williams_r(h, lo, c, 14),
    "atr": lambda h, lo, c, v: ind.atr(h, lo, c, 14),
    "bollinger": lambda h, lo, c, v: ind.bollinger_bands(c)[2],
    "hist_vol": lambda h, lo, c, v: ind.historical_volatility(c, 20),
    "obv": lambda h, lo, c, v: ind.obv(c, v),
    "obv_norm": lambda h, lo, c, v: ind.obv_normalized(c, v, 20),
    "vwap": lambda h, lo, c, v: ind.rolling_vwap(h, lo, c, v, 20),
    "volume_change": lambda h, lo, c, v: ind.volume_change(v),
    "mfi": lambda h, lo, c, v: ind.money_flow_index(h, lo, c, v, 14),
    "beta": lambda h, lo, c, v: ind.rolling_beta(ind.simple_returns(c), ind.simple_returns(h), 60),
}


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_indicator_is_causal(adjusted, name):
    high, low, close, volume = _series(adjusted)
    full = INDICATORS[name](high, low, close, volume)
    for cut in (90, 201, 333):
        prefix = INDICATORS[name](high[:cut], low[:cut], close[:cut], volume[:cut])
        np.testing.assert_allclose(
            prefix.to_numpy(), full[:cut].to_numpy(), rtol=1e-10, atol=1e-12, equal_nan=True
        )


@pytest.mark.parametrize("cut", ["2020-09-15", "2021-03-10", "2021-10-01"])
def test_corrupting_the_future_does_not_change_past_features(adjusted, full_features, cut):
    cut = pd.Timestamp(cut)
    rng = np.random.default_rng(99)
    tampered = adjusted.copy()
    future = tampered["trade_date"] > cut
    scale = rng.uniform(0.2, 5.0, int(future.sum()))
    for column in ("adj_open", "adj_high", "adj_low", "adj_close"):
        tampered.loc[future, column] = tampered.loc[future, column] * scale
    tampered.loc[future, "adj_volume"] = rng.uniform(1, 1e7, int(future.sum()))

    features = compute_features(tampered, CFG)
    past = features["trade_date"] <= cut
    assert past.sum() > 0 and (~past).sum() > 0
    pd.testing.assert_frame_equal(
        features[past].reset_index(drop=True),
        full_features[full_features["trade_date"] <= cut].reset_index(drop=True),
        check_exact=True,
    )
    # Sanity: the corruption really did change the future features.
    assert not np.allclose(
        features.loc[~past, "ret_1d"].to_numpy(),
        full_features.loc[full_features["trade_date"] > cut, "ret_1d"].to_numpy(),
        equal_nan=True,
    )


def test_features_match_a_dataset_built_with_knowledge_only_up_to_T(
    provider, symbols, securities, calendar, raw_actions, adjusted, full_features
):
    """Rebuild as of dates before each split/bonus: adjusted price *levels* differ from
    the full build, yet every (scale-free) feature must be identical."""
    structural = raw_actions[raw_actions["action_type"].isin(["split", "bonus"])]
    assert len(structural) >= 2
    cuts = [d - pd.Timedelta(days=7) for d in structural["ex_date"]] + [pd.Timestamp("2021-11-30")]

    for cut in cuts:
        bars = provider.get_daily_bars(symbols, "2020-01-01", cut)
        actions = provider.get_corporate_actions(symbols, "2020-01-01", cut)
        as_known_then = _adjust(bars, actions, securities, calendar, as_of=cut)

        # The adjusted levels really are different (otherwise this test proves nothing).
        merged = as_known_then.merge(
            adjusted, on=["symbol", "trade_date"], suffixes=("_then", "_now")
        )
        if cut < structural["ex_date"].max():
            assert not np.allclose(merged["adj_close_then"], merged["adj_close_now"])

        then = compute_features(as_known_then, CFG)
        now = full_features[full_features["trade_date"] <= cut].reset_index(drop=True)
        assert len(then) == len(now)
        for column in feature_columns(then):
            np.testing.assert_allclose(
                then[column].to_numpy(),
                now[column].to_numpy(),
                rtol=1e-8,
                atol=1e-10,
                equal_nan=True,
                err_msg=f"{column} differs when rebuilt as of {cut.date()}",
            )


def test_market_proxy_uses_same_day_returns_only(adjusted, full_features):
    """beta/corr at T must not move when another symbol's *later* prices change."""
    cut = pd.Timestamp("2021-02-01")
    tampered = adjusted.copy()
    mask = (tampered["symbol"] == "SYNTH02") & (tampered["trade_date"] > cut)
    tampered.loc[mask, "adj_close"] *= 3.0
    features = compute_features(tampered, CFG)
    for column in (f"beta_{CFG.beta_window}d", f"corr_mkt_{CFG.corr_window}d"):
        before = features.loc[
            (features["symbol"] == "SYNTH01") & (features["trade_date"] <= cut), column
        ]
        reference = full_features.loc[
            (full_features["symbol"] == "SYNTH01") & (full_features["trade_date"] <= cut), column
        ]
        np.testing.assert_array_equal(before.to_numpy(), reference.to_numpy())
