"""Feature engine: turns adjusted bars into a model-ready, point-in-time feature table.

Input
    Adjusted bars from ``ml.data.preprocessing.corporate_actions.apply_adjustments``
    (columns ``symbol, trade_date, adj_open, adj_high, adj_low, adj_close, adj_volume``).

Output
    Long frame ``symbol, trade_date, <features...>`` with one row per input bar.

Point-in-time guarantees
    * Each value at date T uses only bars dated <= T (see ``indicators``).
    * The market proxy on date T is the cross-sectional mean return *of date T*.
    * Every emitted feature is **scale-free** (a ratio, percentage or bounded
      oscillator). Back-adjusted price *levels* change when a later split or dividend
      occurs, so a level-based feature would leak the existence of future corporate
      actions. Level indicators are therefore emitted relative to price:
      ``sma_N_ratio = close / SMA_N - 1``, ``macd_pct = MACD / close``,
      ``atr_pct_N = ATR_N / close``, and so on.

    A feature row for date T is known after the close of T; a decision based on it can
    be executed no earlier than the next session (PRD 10.3).

Missing values
    NaN means "not yet defined" (warm-up) or "mathematically undefined". They are left
    as NaN; consumers must mask them explicitly.

Known limitations
    * Indicators run over each symbol's consecutive *available* bars. A missing session
      (reported by validation) makes the adjacent "1-day" return span more than a day.
    * The market proxy is the equal-weight mean of the universe, which includes the
      asset itself; with a small universe this biases beta towards 1.
"""

import numpy as np
import pandas as pd

from ml.data.validation.report import DataQualityReport, Severity
from ml.features import indicators as ind
from ml.features.config import FeatureConfig

REQUIRED_COLUMNS = [
    "symbol",
    "trade_date",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "adj_volume",
]
KEY_COLUMNS = ["symbol", "trade_date"]

# Feature-name prefix -> closed interval the values must lie in.
BOUNDS: dict[str, tuple[float, float]] = {
    "rsi_": (0, 100),
    "stoch_k_": (0, 100),
    "stoch_d_": (0, 100),
    "mfi_": (0, 100),
    "adx_": (0, 100),
    "plus_di_": (0, 100),
    "minus_di_": (0, 100),
    "williams_r_": (-100, 0),
    "corr_mkt_": (-1, 1),
    "obv_norm_": (-1, 1),
    "bb_width_": (0, np.inf),
    "atr_pct_": (0, np.inf),
    "hist_vol_": (0, np.inf),
    "ret_std_": (0, np.inf),
}
BOUND_TOLERANCE = 1e-9


def market_proxy_returns(bars: pd.DataFrame) -> pd.Series:
    """Equal-weight cross-sectional mean of same-day simple returns, indexed by date."""
    ordered = bars.sort_values(KEY_COLUMNS)
    returns = ordered.groupby("symbol")["adj_close"].pct_change(fill_method=None)
    return returns.groupby(ordered["trade_date"]).mean()


def _symbol_features(group: pd.DataFrame, market: pd.Series, cfg: FeatureConfig) -> pd.DataFrame:
    g = group.reset_index(drop=True)
    high, low, close, volume = (g[c] for c in ("adj_high", "adj_low", "adj_close", "adj_volume"))
    out: dict[str, pd.Series] = {"symbol": g["symbol"], "trade_date": g["trade_date"]}

    # Returns
    returns = ind.simple_returns(close)
    out["ret_1d"] = returns
    out["log_ret_1d"] = ind.log_returns(close)
    for n in cfg.cum_return_windows:
        out[f"cum_ret_{n}d"] = ind.cumulative_return(close, n)

    # Trend
    for n in cfg.sma_windows:
        out[f"sma_{n}_ratio"] = close / ind.sma(close, n) - 1
    for n in cfg.ema_windows:
        out[f"ema_{n}_ratio"] = close / ind.ema(close, n) - 1
    line, signal, histogram = ind.macd(close, cfg.macd.fast, cfg.macd.slow, cfg.macd.signal)
    out["macd_pct"] = line / close
    out["macd_signal_pct"] = signal / close
    out["macd_hist_pct"] = histogram / close
    adx_value, plus_di, minus_di = ind.adx(high, low, close, cfg.adx_window)
    out[f"adx_{cfg.adx_window}"] = adx_value
    out[f"plus_di_{cfg.adx_window}"] = plus_di
    out[f"minus_di_{cfg.adx_window}"] = minus_di

    # Momentum
    out[f"rsi_{cfg.rsi_window}"] = ind.rsi(close, cfg.rsi_window)
    percent_k, percent_d = ind.stochastic(high, low, close, cfg.stochastic.k, cfg.stochastic.d)
    out[f"stoch_k_{cfg.stochastic.k}"] = percent_k
    out[f"stoch_d_{cfg.stochastic.k}"] = percent_d
    out[f"roc_{cfg.roc_window}"] = ind.roc(close, cfg.roc_window)
    out[f"williams_r_{cfg.williams_window}"] = ind.williams_r(high, low, close, cfg.williams_window)

    # Volatility
    out[f"atr_pct_{cfg.atr_window}"] = ind.atr(high, low, close, cfg.atr_window) / close
    lower, middle, upper = ind.bollinger_bands(close, cfg.bollinger.window, cfg.bollinger.num_std)
    band = upper - lower
    out[f"bb_pctb_{cfg.bollinger.window}"] = (close - lower) / band.where(band > 0)
    out[f"bb_width_{cfg.bollinger.window}"] = band / middle
    for n in cfg.hist_vol_windows:
        out[f"hist_vol_{n}d"] = ind.historical_volatility(close, n, cfg.trading_days_per_year)
    out[f"ret_std_{cfg.rolling_std_window}d"] = ind.rolling_std(returns, cfg.rolling_std_window)

    # Volume
    out[f"obv_norm_{cfg.obv_window}"] = ind.obv_normalized(close, volume, cfg.obv_window)
    vwap = ind.rolling_vwap(high, low, close, volume, cfg.vwap_window)
    out[f"vwap_ratio_{cfg.vwap_window}"] = close / vwap - 1
    out["volume_change_1d"] = ind.volume_change(volume)
    out[f"mfi_{cfg.mfi_window}"] = ind.money_flow_index(high, low, close, volume, cfg.mfi_window)

    # Relationship to the market
    market_returns = pd.Series(market.reindex(g["trade_date"]).to_numpy(), index=g.index)
    out[f"beta_{cfg.beta_window}d"] = ind.rolling_beta(returns, market_returns, cfg.beta_window)
    out[f"corr_mkt_{cfg.corr_window}d"] = ind.rolling_correlation(
        returns, market_returns, cfg.corr_window
    )
    return pd.DataFrame(out)


def compute_features(bars: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"adjusted bars are missing columns: {missing}")
    if bars.duplicated(KEY_COLUMNS).any():
        raise ValueError("bars contain duplicate (symbol, trade_date) rows; clean them first")

    ordered = bars.sort_values(KEY_COLUMNS).reset_index(drop=True)
    market = market_proxy_returns(ordered)
    frames = [
        _symbol_features(group, market, cfg) for _, group in ordered.groupby("symbol", sort=True)
    ]
    if not frames:
        return pd.DataFrame(columns=KEY_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c not in KEY_COLUMNS]


def validate_features(features: pd.DataFrame, report: DataQualityReport) -> None:
    """Record infinite values, bound violations (errors) and undefined values (info)."""
    for column in feature_columns(features):
        values = features[column]
        infinite = np.isinf(values)
        if infinite.any():
            report.add(
                "feature_infinite",
                Severity.ERROR,
                int(infinite.sum()),
                detail=column,
                dates=features.loc[infinite, "trade_date"],
            )
        for prefix, (lower, upper) in BOUNDS.items():
            if column.startswith(prefix):
                outside = (values < lower - BOUND_TOLERANCE) | (values > upper + BOUND_TOLERANCE)
                if outside.any():
                    report.add(
                        "feature_out_of_bounds",
                        Severity.ERROR,
                        int(outside.sum()),
                        detail=f"{column} outside [{lower}, {upper}]",
                        dates=features.loc[outside, "trade_date"],
                    )
        report.add(
            "feature_undefined",
            Severity.INFO,
            int(values.isna().sum()),
            detail=f"{column}: NaN during warm-up or where undefined (left as NaN)",
        )
