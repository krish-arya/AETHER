"""Technical indicator library (PRD 6.2).

Every function is **causal**: the value at row ``t`` depends only on rows ``<= t``.
No centred windows, no back-filling, no look-ahead. This is enforced by
``tests/ml/test_no_lookahead.py``.

Inputs are pandas Series for one symbol in chronological order. Rows inside the
warm-up window are NaN, as are values that are mathematically undefined (for example
a stochastic oscillator over a zero-width range). Nothing is ever filled.

Conventions
    * ``ema`` is the recursive EMA seeded with the first observation
      (``adjust=False``); values exist from the ``n``-th observation.
    * Wilder smoothing (RSI, ATR, ADX) is seeded with the simple mean of the first
      ``n`` values, as in Wilder's original definition and TA-Lib.
    * Bollinger Bands use the population standard deviation (ddof=0), per Bollinger.
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------- helpers


def wilder_smooth(series: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing: SMA seed over the first ``n`` values, then
    ``s_t = (s_{t-1} * (n - 1) + x_t) / n``.

    A NaN input yields NaN and restarts the seed (``n`` fresh values are required),
    so an undefined input is never bridged.
    """
    values = series.to_numpy(dtype="float64")
    out = np.full(len(values), np.nan)
    smoothed = np.nan
    run, total = 0, 0.0
    for i, value in enumerate(values):
        if np.isnan(value):
            smoothed, run, total = np.nan, 0, 0.0
        elif np.isnan(smoothed):
            run += 1
            total += value
            if run == n:
                smoothed = total / n
                out[i] = smoothed
        else:
            smoothed = (smoothed * (n - 1) + value) / n
            out[i] = smoothed
    return pd.Series(out, index=series.index)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    )
    # Undefined on the first bar: there is no previous close.
    return ranges.max(axis=1).where(previous_close.notna())


# ------------------------------------------------------------------------- trend


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns ``(macd_line, signal_line, histogram)``."""
    line = ema(close, fast) - ema(close, slow)
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, signal_line, line - signal_line


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns ``(adx, plus_di, minus_di)``, each in [0, 100]."""
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0).where(up.notna())
    minus_dm = down.where((down > up) & (down > 0), 0.0).where(down.notna())
    average_tr = wilder_smooth(true_range(high, low, close), n)
    average_tr = average_tr.where(average_tr > 0)
    plus_di = 100 * wilder_smooth(plus_dm, n) / average_tr
    minus_di = 100 * wilder_smooth(minus_dm, n) / average_tr
    total = plus_di + minus_di
    # No directional movement at all means no trend: DX is 0, not undefined.
    dx = (100 * (plus_di - minus_di).abs() / total.where(total > 0)).where(total != 0, 0.0)
    dx = dx.where(total.notna())
    return wilder_smooth(dx, n), plus_di, minus_di


# ---------------------------------------------------------------------- momentum


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = wilder_smooth(gain, n)
    average_loss = wilder_smooth(loss, n)
    total = average_gain + average_loss
    # Equivalent to 100 - 100 / (1 + avg_gain / avg_loss); undefined when flat.
    return 100 * average_gain / total.where(total > 0)


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Returns ``(%K, %D)``. Undefined (NaN) when the ``k``-bar range is zero."""
    highest = high.rolling(k, min_periods=k).max()
    lowest = low.rolling(k, min_periods=k).min()
    width = highest - lowest
    percent_k = 100 * (close - lowest) / width.where(width > 0)
    return percent_k, percent_k.rolling(d, min_periods=d).mean()


def roc(close: pd.Series, n: int = 10) -> pd.Series:
    """Rate of change in percent."""
    return 100 * (close / close.shift(n) - 1)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    highest = high.rolling(n, min_periods=n).max()
    lowest = low.rolling(n, min_periods=n).min()
    width = highest - lowest
    return -100 * (highest - close) / width.where(width > 0)


# -------------------------------------------------------------------- volatility


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return wilder_smooth(true_range(high, low, close), n)


def bollinger_bands(
    close: pd.Series, n: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns ``(lower, middle, upper)``."""
    middle = sma(close, n)
    deviation = close.rolling(n, min_periods=n).std(ddof=0)
    return middle - num_std * deviation, middle, middle + num_std * deviation


def historical_volatility(close: pd.Series, n: int = 20, periods_per_year: int = 252) -> pd.Series:
    """Annualised standard deviation of daily log returns."""
    return log_returns(close).rolling(n, min_periods=n).std(ddof=1) * np.sqrt(periods_per_year)


def rolling_std(series: pd.Series, n: int = 20) -> pd.Series:
    return series.rolling(n, min_periods=n).std(ddof=1)


# ------------------------------------------------------------------------ volume


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Classic On-Balance Volume (starts at 0).

    Its level depends on where the history starts and on share-count changes, so it is
    not fed to models directly -- see :func:`obv_normalized`.
    """
    direction = np.sign(close.diff()).fillna(0.0)  # first bar has no direction
    return (direction * volume).cumsum()


def obv_normalized(close: pd.Series, volume: pd.Series, n: int = 20) -> pd.Series:
    """Windowed OBV: net signed volume over ``n`` bars / total volume, in [-1, 1]."""
    signed = np.sign(close.diff()) * volume
    total = volume.rolling(n, min_periods=n).sum()
    return signed.rolling(n, min_periods=n).sum() / total.where(total > 0)


def rolling_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 20
) -> pd.Series:
    """Volume-weighted average of the typical price over ``n`` daily bars.

    With daily bars this is a rolling approximation, not the intraday session VWAP.
    """
    typical = (high + low + close) / 3
    total = volume.rolling(n, min_periods=n).sum()
    return (typical * volume).rolling(n, min_periods=n).sum() / total.where(total > 0)


def volume_change(volume: pd.Series) -> pd.Series:
    """Day-over-day fractional change in volume; undefined after a zero-volume bar."""
    previous = volume.shift(1)
    return volume / previous.where(previous > 0) - 1


def money_flow_index(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 14
) -> pd.Series:
    typical = (high + low + close) / 3
    flow = typical * volume
    delta = typical.diff()
    positive = flow.where(delta > 0, 0.0).where(delta.notna())
    negative = flow.where(delta < 0, 0.0).where(delta.notna())
    positive_sum = positive.rolling(n, min_periods=n).sum()
    total = positive_sum + negative.rolling(n, min_periods=n).sum()
    return 100 * positive_sum / total.where(total > 0)


# ----------------------------------------------------------------------- returns


def simple_returns(close: pd.Series) -> pd.Series:
    return close / close.shift(1) - 1


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def cumulative_return(close: pd.Series, n: int) -> pd.Series:
    """Return over the trailing ``n`` bars."""
    return close / close.shift(n) - 1


def rolling_beta(asset_returns: pd.Series, market_returns: pd.Series, n: int = 60) -> pd.Series:
    variance = market_returns.rolling(n, min_periods=n).var()
    covariance = asset_returns.rolling(n, min_periods=n).cov(market_returns)
    return covariance / variance.where(variance > 0)


def rolling_correlation(a: pd.Series, b: pd.Series, n: int = 60) -> pd.Series:
    return a.rolling(n, min_periods=n).corr(b)
