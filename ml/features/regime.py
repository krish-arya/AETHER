"""Market regime detection (PRD 6.3): a transparent, causal, rule-based classifier.

Of the candidate approaches in the PRD (GMM, HMM, k-means, volatility rules) this is
the volatility/trend rule system. It was chosen first because every label can be
traced to three numbers, it needs no fitting (so it cannot overfit or leak), and it is
deterministic. A statistical model (GMM / HMM) can be added behind the same output
schema and compared with it.

Inputs
    Daily returns of a market proxy (the equal-weight universe return, or an index).

Measurements -- each uses data up to and including T only
    realized_vol       annualised std of the last ``vol_window`` daily returns
    volatility_level   percentile rank of today's realized_vol among all realized_vol
                       values observed so far (expanding window), in [0, 1]
    trend_return       return of the proxy over the last ``trend_window`` sessions
    trend_strength     Kaufman efficiency ratio over the same window:
                       |net move| / sum(|daily moves|), in [0, 1]
    drawdown           fall of the proxy from its running peak (<= 0)

Rules, first match wins
    market_stress      volatility_level >= stress_vol_percentile and
                       drawdown <= -stress_drawdown
    high_volatility    volatility_level >= high_vol_percentile
    bull               trend_return >= +trend_return_threshold and
                       trend_strength >= min_trend_strength
    bear               trend_return <= -trend_return_threshold and
                       trend_strength >= min_trend_strength
    low_volatility     volatility_level <= low_vol_percentile
    sideways           otherwise

    No regime is assigned until ``min_history`` sessions of volatility history exist.

Stability
    A new label must hold for ``min_duration`` consecutive sessions before the regime
    switches (stress is entered immediately). This removes one-day flips at the cost of
    reacting a few sessions late.

Output: one row per session with ``regime``, ``confidence``, ``volatility_level``,
``trend_strength`` plus the raw measurements.

``confidence`` is a margin score in [0, 1] -- how far the measurements are from the
nearest rule boundary -- **not a calibrated probability**. It is 0 on days when the
day's own measurements point to a different label than the one being held.

Known limitations
    * Thresholds are judgement calls exposed in the config, not estimated from data.
    * One market-wide regime; no per-sector or per-asset regimes.
    * With the equal-weight proxy the regime describes the chosen universe, not NIFTY.
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

REGIMES = ("bull", "bear", "sideways", "high_volatility", "low_volatility", "market_stress")
REGIME_COLUMNS = [
    "trade_date",
    "regime",
    "confidence",
    "volatility_level",
    "trend_strength",
    "realized_vol",
    "trend_return",
    "drawdown",
]


class RegimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vol_window: int = Field(21, ge=5)
    trend_window: int = Field(63, ge=5)
    min_history: int = Field(252, ge=20)
    high_vol_percentile: float = Field(0.75, gt=0, lt=1)
    low_vol_percentile: float = Field(0.25, gt=0, lt=1)
    stress_vol_percentile: float = Field(0.90, gt=0, lt=1)
    stress_drawdown: float = Field(0.10, gt=0, lt=1)
    trend_return_threshold: float = Field(0.05, gt=0)
    min_trend_strength: float = Field(0.10, ge=0, le=1)
    min_duration: int = Field(5, ge=1)
    periods_per_year: int = Field(252, ge=1)

    @model_validator(mode="after")
    def _ordered_percentiles(self):
        if not self.low_vol_percentile < self.high_vol_percentile <= self.stress_vol_percentile:
            raise ValueError(
                "require low_vol_percentile < high_vol_percentile <= stress_vol_percentile"
            )
        return self


def _raw_label(vol_level, trend_return, strength, drawdown, cfg: RegimeConfig):
    """(label, margin) for one day's measurements; (None, nan) while undefined."""
    if not np.isfinite([vol_level, trend_return, strength, drawdown]).all():
        return None, np.nan
    if vol_level >= cfg.stress_vol_percentile and drawdown <= -cfg.stress_drawdown:
        vol_margin = (vol_level - cfg.stress_vol_percentile) / (1 - cfg.stress_vol_percentile)
        dd_margin = (-drawdown - cfg.stress_drawdown) / (1 - cfg.stress_drawdown)
        return "market_stress", min(vol_margin, dd_margin * 5)
    if vol_level >= cfg.high_vol_percentile:
        return "high_volatility", (vol_level - cfg.high_vol_percentile) / (
            1 - cfg.high_vol_percentile
        )
    trending = strength >= cfg.min_trend_strength
    if trending and abs(trend_return) >= cfg.trend_return_threshold:
        return_margin = abs(trend_return) / cfg.trend_return_threshold - 1
        strength_margin = (strength - cfg.min_trend_strength) / max(
            1 - cfg.min_trend_strength, 1e-9
        )
        return ("bull" if trend_return > 0 else "bear"), min(return_margin, strength_margin * 5)
    if vol_level <= cfg.low_vol_percentile:
        return "low_volatility", (cfg.low_vol_percentile - vol_level) / cfg.low_vol_percentile
    # Sideways: far from both the volatility bands and the trend threshold.
    band = min(vol_level - cfg.low_vol_percentile, cfg.high_vol_percentile - vol_level)
    band /= (cfg.high_vol_percentile - cfg.low_vol_percentile) / 2
    flat = 1 - min(abs(trend_return) / cfg.trend_return_threshold, 1) if trending else 1.0
    return "sideways", min(band, flat)


def _hold(raw: list, min_duration: int) -> list:
    """Switch regime only after ``min_duration`` consecutive days of a new label."""
    held, current, candidate, run = [], None, None, 0
    for label in raw:
        if label is None:
            held.append(current)
            continue
        if current is None or label == "market_stress":
            current, candidate, run = label, None, 0
        elif label == current:
            candidate, run = None, 0
        else:
            run = run + 1 if label == candidate else 1
            candidate = label
            if run >= min_duration:
                current, candidate, run = label, None, 0
        held.append(current)
    return held


def detect_regimes(market_returns: pd.Series, cfg: RegimeConfig | None = None) -> pd.DataFrame:
    """Classify every session of ``market_returns`` (indexed by date, chronological)."""
    cfg = cfg or RegimeConfig()
    returns = market_returns.astype("float64")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("market_returns must be in chronological order")

    level = (1 + returns.fillna(0.0)).cumprod()
    realized = returns.rolling(cfg.vol_window, min_periods=cfg.vol_window).std(ddof=1)
    realized = realized * np.sqrt(cfg.periods_per_year)
    vol_level = realized.expanding(min_periods=cfg.min_history).rank(pct=True)
    trend_return = level / level.shift(cfg.trend_window) - 1
    path = level.diff().abs().rolling(cfg.trend_window, min_periods=cfg.trend_window).sum()
    strength = (level - level.shift(cfg.trend_window)).abs() / path.where(path > 0)
    drawdown = level / level.cummax() - 1

    raw, margins = [], []
    for values in zip(vol_level, trend_return, strength, drawdown, strict=True):
        label, margin = _raw_label(*values, cfg)
        raw.append(label)
        margins.append(margin)
    held = _hold(raw, cfg.min_duration)
    confidence = [
        float(np.clip(margin, 0.0, 1.0)) if label is not None and label == kept else
        (0.0 if kept is not None else np.nan)
        for label, kept, margin in zip(raw, held, margins, strict=True)
    ]  # fmt: skip

    out = pd.DataFrame(
        {
            "trade_date": returns.index,
            "regime": held,
            "confidence": confidence,
            "volatility_level": vol_level.to_numpy(),
            "trend_strength": strength.to_numpy(),
            "realized_vol": realized.to_numpy(),
            "trend_return": trend_return.to_numpy(),
            "drawdown": drawdown.to_numpy(),
        }
    )
    return out[REGIME_COLUMNS]


def regime_stability(regimes: pd.DataFrame, periods_per_year: int = 252) -> dict:
    """Stability and temporal-consistency diagnostics for a regime series (PRD 6.3)."""
    labels = regimes["regime"].dropna().reset_index(drop=True)
    if labels.empty:
        return {"sessions": 0}
    changed = labels != labels.shift(1)
    run_id = changed.cumsum()
    runs = labels.groupby(run_id).agg(["first", "size"])
    switches = int(changed.sum()) - 1
    previous = labels.shift(1)
    transitions = pd.crosstab(previous.iloc[1:], labels.iloc[1:], normalize="index")
    return {
        "sessions": int(len(labels)),
        "share": labels.value_counts(normalize=True).round(4).to_dict(),
        "switches": switches,
        "switches_per_year": round(switches / (len(labels) / periods_per_year), 2),
        "mean_run_sessions": round(float(runs["size"].mean()), 1),
        "median_run_sessions": float(runs["size"].median()),
        "min_run_sessions": int(runs["size"].min()),
        "mean_run_by_regime": runs.groupby("first")["size"].mean().round(1).to_dict(),
        # Probability of staying in the same regime from one session to the next.
        "persistence": round(float((labels == previous).iloc[1:].mean()), 4),
        "transition_matrix": {
            row: {col: round(float(value), 4) for col, value in cols.items() if value > 0}
            for row, cols in transitions.to_dict("index").items()
        },
    }
