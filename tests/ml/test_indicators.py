"""Indicator correctness against hand-computed values and independent references."""

import numpy as np
import pandas as pd
import pytest

from ml.features import indicators as ind

NAN = np.nan


def s(*values) -> pd.Series:
    return pd.Series(values, dtype="float64")


def check(actual: pd.Series, expected) -> None:
    np.testing.assert_allclose(actual.to_numpy(), np.asarray(expected, dtype=float), rtol=1e-12)


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 400
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    open_ = close * np.exp(rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.006, n)))
    volume = rng.integers(1_000, 100_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


# ------------------------------------------------------------------------- trend


def test_sma():
    check(ind.sma(s(1, 2, 3, 4, 5), 3), [NAN, NAN, 2, 3, 4])


def test_ema_recursion():
    # span 3 -> alpha 0.5, seeded with the first value: 1, 1.5, 2.25, 3.125
    check(ind.ema(s(1, 2, 3, 4), 3), [NAN, NAN, 2.25, 3.125])


def test_wilder_smoothing_seeds_with_simple_mean():
    # seed = mean(1,2,3) = 2 ; (2*2+4)/3 = 8/3 ; (8/3*2+5)/3 = 31/9
    check(ind.wilder_smooth(s(1, 2, 3, 4, 5), 3), [NAN, NAN, 2, 8 / 3, 31 / 9])


def test_wilder_smoothing_restarts_after_missing_value():
    out = ind.wilder_smooth(s(1, 2, 3, NAN, 4, 5, 6), 3)
    check(out, [NAN, NAN, 2, NAN, NAN, NAN, 5])  # NaN is never bridged


def test_macd_is_difference_of_emas(ohlcv):
    line, signal, histogram = ind.macd(ohlcv["close"], 12, 26, 9)
    expected = ind.ema(ohlcv["close"], 12) - ind.ema(ohlcv["close"], 26)
    pd.testing.assert_series_equal(line, expected)
    pd.testing.assert_series_equal(histogram, line - signal)
    assert line.isna().sum() == 25 and signal.isna().sum() == 25 + 8


def test_adx_is_100_in_a_perfect_uptrend():
    steps = np.arange(60, dtype=float)
    adx, plus_di, minus_di = ind.adx(
        pd.Series(11 + steps), pd.Series(9 + steps), pd.Series(10 + steps), 14
    )
    assert adx.dropna().iloc[-1] == pytest.approx(100.0)
    assert (minus_di.dropna() == 0).all() and (plus_di.dropna() > 0).all()
    assert adx.isna().sum() == 27  # 1 (no previous bar) + 13 (DI seed) + 13 (ADX seed)


def test_adx_is_zero_without_directional_movement():
    flat_high, flat_low, close = (
        pd.Series([11.0] * 60),
        pd.Series([9.0] * 60),
        pd.Series([10.0] * 60),
    )
    adx, _, _ = ind.adx(flat_high, flat_low, close, 14)
    assert (adx.dropna() == 0).all()


def test_adx_bounds(ohlcv):
    for series in ind.adx(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14):
        values = series.dropna()
        assert len(values) > 300 and values.between(0, 100).all()


# ---------------------------------------------------------------------- momentum


def _reference_rsi(close: np.ndarray, n: int) -> np.ndarray:
    """Textbook Wilder RSI written independently of the library."""
    out = np.full(len(close), np.nan)
    deltas = np.diff(close)
    gains, losses = np.clip(deltas, 0, None), np.clip(-deltas, 0, None)
    average_gain, average_loss = gains[:n].mean(), losses[:n].mean()
    for i in range(n, len(close)):
        if i > n:
            average_gain = (average_gain * (n - 1) + gains[i - 1]) / n
            average_loss = (average_loss * (n - 1) + losses[i - 1]) / n
        out[i] = 100.0 if average_loss == 0 else 100 - 100 / (1 + average_gain / average_loss)
    return out


def test_rsi_matches_textbook_reference(ohlcv):
    actual = ind.rsi(ohlcv["close"], 14)
    np.testing.assert_allclose(actual.to_numpy(), _reference_rsi(ohlcv["close"].to_numpy(), 14))


def test_rsi_extremes():
    rising = pd.Series(np.arange(1.0, 40.0))
    assert (ind.rsi(rising, 14).dropna() == 100).all()
    assert (ind.rsi(rising[::-1].reset_index(drop=True), 14).dropna() == 0).all()
    assert ind.rsi(pd.Series([5.0] * 40), 14).isna().all()  # flat: undefined, not 50


def test_stochastic_and_williams_hand_computed():
    high, low, close = s(10, 12, 11, 13), s(8, 9, 9, 10), s(9, 11, 10, 12)
    k, d = ind.stochastic(high, low, close, k=3, d=2)
    # idx2: HH=12 LL=8 -> (10-8)/4 = 50 ; idx3: HH=13 LL=9 -> (12-9)/4 = 75
    check(k, [NAN, NAN, 50, 75])
    check(d, [NAN, NAN, NAN, 62.5])
    check(ind.williams_r(high, low, close, 3), [NAN, NAN, -50, -25])


def test_stochastic_is_undefined_on_a_zero_range():
    flat = s(5, 5, 5, 5)
    k, _ = ind.stochastic(flat, flat, flat, k=3, d=2)
    assert k.isna().all()


def test_roc():
    check(ind.roc(s(100, 110, 121), 1), [NAN, 10, 10])
    check(ind.roc(s(100, 110, 121), 2), [NAN, NAN, 21])


# -------------------------------------------------------------------- volatility


def test_true_range_and_atr_hand_computed():
    high, low, close = s(10, 11, 12.5), s(9, 9.5, 10.5), s(9.5, 10.5, 11.5)
    # TR1 = max(1.5, |11-9.5|, |9.5-9.5|) = 1.5 ; TR2 = max(2.0, |12.5-10.5|, 0) = 2.0
    check(ind.true_range(high, low, close), [NAN, 1.5, 2.0])
    check(ind.atr(high, low, close, 2), [NAN, NAN, 1.75])


def test_true_range_uses_gap_from_previous_close():
    # Gap up: today's range is 1 but the move from yesterday's close is 6.
    check(ind.true_range(s(10, 16), s(9, 15), s(10, 15.5)), [NAN, 6])


def test_bollinger_bands_use_population_std():
    lower, middle, upper = ind.bollinger_bands(s(1, 2, 3, 4, 5), n=5, num_std=2.0)
    assert middle.iloc[-1] == pytest.approx(3.0)
    assert upper.iloc[-1] == pytest.approx(3 + 2 * np.sqrt(2))
    assert lower.iloc[-1] == pytest.approx(3 - 2 * np.sqrt(2))
    assert middle.iloc[:4].isna().all()


def test_historical_volatility():
    constant_growth = pd.Series(100 * 1.01 ** np.arange(30))
    assert ind.historical_volatility(constant_growth, 20).dropna().abs().max() < 1e-12

    log_ret = np.array([0.01, -0.02, 0.03, 0.0, -0.01])
    close = pd.Series(100 * np.exp(np.concatenate([[0], np.cumsum(log_ret)])))
    expected = np.std(log_ret, ddof=1) * np.sqrt(252)
    assert ind.historical_volatility(close, 5).iloc[-1] == pytest.approx(expected)


# ------------------------------------------------------------------------ volume


def test_obv_hand_computed():
    close, volume = s(10, 11, 10, 10, 12), s(100, 200, 300, 400, 500)
    check(ind.obv(close, volume), [0, 200, -100, -100, 400])


def test_obv_normalized_hand_computed():
    close, volume = s(10, 11, 10, 10, 12), s(100, 200, 300, 400, 500)
    # signed volume: nan, +200, -300, 0, +500
    check(ind.obv_normalized(close, volume, 2), [NAN, NAN, -100 / 500, -300 / 700, 500 / 900])


def test_rolling_vwap_hand_computed():
    high, low, close, volume = s(11, 13), s(9, 11), s(10, 12), s(100, 300)
    # typical prices 10 and 12 -> (10*100 + 12*300) / 400 = 11.5
    check(ind.rolling_vwap(high, low, close, volume, 2), [NAN, 11.5])


def test_volume_change_is_undefined_after_zero_volume():
    check(ind.volume_change(s(100, 150, 0, 50)), [NAN, 0.5, -1.0, NAN])


def test_money_flow_index_hand_computed():
    typical, volume = s(10, 11, 10.5), s(100, 100, 100)
    # positive flow 1100 (up day), negative flow 1050 (down day)
    mfi = ind.money_flow_index(typical, typical, typical, volume, 2)
    check(mfi, [NAN, NAN, 100 * 1100 / 2150])


def test_money_flow_index_extremes_and_bounds(ohlcv):
    rising = pd.Series(np.arange(1.0, 30.0))
    volume = pd.Series([10.0] * 29)
    assert (ind.money_flow_index(rising, rising, rising, volume, 14).dropna() == 100).all()
    mfi = ind.money_flow_index(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"], 14)
    assert mfi.dropna().between(0, 100).all()


# ----------------------------------------------------------------------- returns


def test_return_definitions():
    close = s(100, 110, 99)
    check(ind.simple_returns(close), [NAN, 0.10, -0.10])
    check(ind.log_returns(close), [NAN, np.log(1.1), np.log(0.9)])
    check(ind.cumulative_return(close, 2), [NAN, NAN, -0.01])


def test_beta_and_correlation_of_a_levered_market():
    rng = np.random.default_rng(0)
    market = pd.Series(rng.normal(0, 0.01, 200))
    asset = 2.0 * market + 0.001
    beta = ind.rolling_beta(asset, market, 60).dropna()
    corr = ind.rolling_correlation(asset, market, 60).dropna()
    assert len(beta) == 141
    np.testing.assert_allclose(beta, 2.0, rtol=1e-9)
    np.testing.assert_allclose(corr, 1.0, rtol=1e-9)
    np.testing.assert_allclose(ind.rolling_beta(-asset, market, 60).dropna(), -2.0, rtol=1e-9)
