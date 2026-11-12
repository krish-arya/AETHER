"""Statistical evaluation of generated scenarios (PRD 7.5) -- never by eye alone.

Synthetic windows are compared with **held-out real windows** (a period the generator
never saw, standardised with the training statistics):

    return_distribution       KS statistic and Wasserstein distance of daily returns
    volatility_distribution   same, for per-window realised volatility
    autocorrelation           mean |difference| of ACF(1..5) for returns and |returns|
    cross_correlation         mean |difference| between the asset correlation matrices
    tail_risk                 1% / 5% quantiles, 5% CVaR and excess kurtosis, side by side
    drawdown_distribution     KS statistic of the equal-weight max drawdown per window
    discriminative_score      |accuracy - 0.5| of classifiers trained to tell real from
                              synthetic windows, measured on unseen windows (0 = cannot
                              tell them apart, 0.5 = trivially separable)
    memorization              nearest-neighbour distances from synthetic windows to the
                              training windows, relative to the same distances for
                              unseen real windows; plus the share of near-copies

``real_baseline`` runs the same comparison between the training period and the
held-out period. Markets change, so even real data differs from real data: a synthetic
score should be read against that baseline, not against zero.

``passes`` applies the configured quality gates. A failed gate is reported as failed;
it is the caller's decision whether to use the generator anyway.
"""

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.gan.config import QualityGates
from ml.gan.data import WindowSet


def _returns(windows: WindowSet, raw: np.ndarray) -> np.ndarray:
    parts = windows.split(raw)
    return parts["gap"] + parts["intra"]  # [n, L, N]


def _acf(returns: np.ndarray, lags: int = 5) -> np.ndarray:
    """Mean autocorrelation at lags 1..lags over windows and assets."""
    centred = returns - returns.mean(axis=1, keepdims=True)
    variance = (centred**2).mean(axis=1)
    out = []
    for lag in range(1, lags + 1):
        covariance = (centred[:, lag:] * centred[:, :-lag]).mean(axis=1)
        out.append(np.nanmean(covariance / np.where(variance > 0, variance, np.nan)))
    return np.asarray(out)


def _max_drawdown(returns: np.ndarray) -> np.ndarray:
    """Max drawdown of the equal-weight portfolio in each window."""
    wealth = np.exp(np.cumsum(returns.mean(axis=2), axis=1))
    peak = np.maximum.accumulate(np.maximum(wealth, 1.0), axis=1)
    return (wealth / peak - 1).min(axis=1)


def _tail(values: np.ndarray) -> dict:
    q01, q05 = np.quantile(values, [0.01, 0.05])
    return {
        "q01": float(q01),
        "q05": float(q05),
        "cvar05": float(values[values <= q05].mean()),
        "excess_kurtosis": float(stats.kurtosis(values)),
    }


def _summary(returns: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Per-window summary statistics used by the tree-based discriminator."""
    centred = returns - returns.mean(axis=1, keepdims=True)
    lag1 = (centred[:, 1:] * centred[:, :-1]).mean(axis=1) / (centred.var(axis=1) + 1e-12)
    market = returns.mean(axis=2)
    return np.column_stack(
        [
            returns.mean(axis=1),
            returns.std(axis=1),
            lag1,
            stats.skew(returns, axis=1),
            market.std(axis=1),
            np.abs(returns).mean(axis=(1, 2)),
            raw.std(axis=1),
        ]
    )


def discriminative_score(
    windows: WindowSet, real_raw: np.ndarray, synthetic_raw: np.ndarray, seed: int
) -> dict:
    n = min(len(real_raw), len(synthetic_raw))
    rng = np.random.default_rng(seed)
    real = real_raw[rng.choice(len(real_raw), n, replace=False)]
    synthetic = synthetic_raw[rng.choice(len(synthetic_raw), n, replace=False)]
    labels = np.r_[np.ones(n), np.zeros(n)]
    both = np.concatenate([real, synthetic])
    flat = ((both - windows.mean) / windows.std).reshape(len(both), -1)
    summary = _summary(_returns(windows, both), both)

    scores = {}
    for name, features, model in (
        ("logistic_on_windows", flat,
         make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.1))),
        ("boosting_on_statistics", summary,
         HistGradientBoostingClassifier(max_iter=100, random_state=seed)),
    ):  # fmt: skip
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=seed, stratify=labels
        )
        accuracy = float(model.fit(x_train, y_train).score(x_test, y_test))
        scores[name] = {"accuracy": accuracy, "score": abs(accuracy - 0.5)}
    scores["score"] = max(v["score"] for v in scores.values())
    return scores


def _nearest_distance(queries: np.ndarray, reference: np.ndarray, block: int = 256) -> np.ndarray:
    out = np.empty(len(queries))
    squared = (reference**2).sum(axis=1)
    for start in range(0, len(queries), block):
        chunk = queries[start : start + block]
        distances = (chunk**2).sum(axis=1)[:, None] + squared[None, :] - 2 * chunk @ reference.T
        out[start : start + block] = np.sqrt(np.maximum(distances.min(axis=1), 0.0))
    return out


def memorization_check(
    windows: WindowSet, train_raw: np.ndarray, heldout_raw: np.ndarray, synthetic_raw: np.ndarray
) -> dict:
    """Does the generator replay its training data?

    ``ratio`` < 1 means synthetic windows sit closer to the training set than unseen
    real windows do; values far below 1 indicate memorisation. ``copied_fraction`` is
    the share of synthetic windows that are near-duplicates of a training window
    (closer than 1% of the typical real nearest-neighbour distance).
    """

    def flat(raw: np.ndarray) -> np.ndarray:
        return ((raw - windows.mean) / windows.std).reshape(len(raw), -1)

    reference = flat(train_raw)
    synthetic = _nearest_distance(flat(synthetic_raw), reference)
    heldout = _nearest_distance(flat(heldout_raw), reference)
    typical = float(np.median(heldout))
    return {
        "median_distance_synthetic": float(np.median(synthetic)),
        "median_distance_heldout_real": typical,
        "ratio": float(np.median(synthetic) / typical) if typical > 0 else float("nan"),
        "min_distance_synthetic": float(synthetic.min()),
        "copied_fraction": float((synthetic < 0.01 * typical).mean()),
        "closer_than_5pct_of_real": float((synthetic < np.quantile(heldout, 0.05)).mean()),
    }


def compare(windows: WindowSet, real_raw: np.ndarray, other_raw: np.ndarray) -> dict:
    """Distributional distances between two sets of raw windows."""
    real, other = _returns(windows, real_raw), _returns(windows, other_raw)
    real_flat, other_flat = real.ravel(), other.ravel()
    real_vol, other_vol = real.std(axis=1).ravel(), other.std(axis=1).ravel()
    real_corr = np.corrcoef(real.reshape(-1, real.shape[2]), rowvar=False)
    other_corr = np.corrcoef(other.reshape(-1, other.shape[2]), rowvar=False)
    off_diagonal = ~np.eye(real.shape[2], dtype=bool)
    real_dd, other_dd = _max_drawdown(real), _max_drawdown(other)
    return {
        "return_distribution": {
            "ks": float(stats.ks_2samp(real_flat, other_flat).statistic),
            "wasserstein": float(stats.wasserstein_distance(real_flat, other_flat)),
            "real_std": float(real_flat.std()),
            "other_std": float(other_flat.std()),
        },
        "volatility_distribution": {
            "ks": float(stats.ks_2samp(real_vol, other_vol).statistic),
            "wasserstein": float(stats.wasserstein_distance(real_vol, other_vol)),
        },
        "autocorrelation": {
            "returns_error": float(np.abs(_acf(real) - _acf(other)).mean()),
            "abs_returns_error": float(np.abs(_acf(np.abs(real)) - _acf(np.abs(other))).mean()),
        },
        "cross_correlation": {
            "mean_abs_error": float(np.abs(real_corr - other_corr)[off_diagonal].mean()),
            "real_mean": float(real_corr[off_diagonal].mean()),
            "other_mean": float(other_corr[off_diagonal].mean()),
        },
        "tail_risk": {"real": _tail(real_flat), "other": _tail(other_flat)},
        "drawdown_distribution": {
            "ks": float(stats.ks_2samp(real_dd, other_dd).statistic),
            "real_mean": float(real_dd.mean()),
            "other_mean": float(other_dd.mean()),
        },
    }


def evaluate(
    windows: WindowSet,
    *,
    train_raw: np.ndarray,
    heldout_raw: np.ndarray,
    synthetic_raw: np.ndarray,
    gates: QualityGates,
    seed: int = 0,
) -> dict:
    """Full evaluation report with gate verdicts."""
    synthetic = compare(windows, heldout_raw, synthetic_raw)
    discriminative = discriminative_score(windows, heldout_raw, synthetic_raw, seed)
    memorization = memorization_check(windows, train_raw, heldout_raw, synthetic_raw)
    checks = {
        "return_ks": (synthetic["return_distribution"]["ks"], "<=", gates.max_return_ks),
        "correlation_error": (
            synthetic["cross_correlation"]["mean_abs_error"], "<=", gates.max_correlation_error,
        ),
        "discriminative_score": (discriminative["score"], "<=", gates.max_discriminative_score),
        "memorization_ratio": (memorization["ratio"], ">=", gates.min_memorization_ratio),
        "copied_fraction": (memorization["copied_fraction"], "<=", gates.max_copied_fraction),
    }  # fmt: skip
    verdicts = {
        name: {
            "value": float(value),
            "threshold": float(limit),
            "rule": rule,
            "passed": bool(value <= limit if rule == "<=" else value >= limit),
        }
        for name, (value, rule, limit) in checks.items()
    }
    return {
        "windows": {
            "train": int(len(train_raw)),
            "heldout_real": int(len(heldout_raw)),
            "synthetic": int(len(synthetic_raw)),
        },
        "synthetic_vs_heldout_real": synthetic,
        "real_baseline_train_vs_heldout": compare(windows, heldout_raw, train_raw),
        "discriminative": discriminative,
        "memorization": memorization,
        "gates": verdicts,
        "passes": all(v["passed"] for v in verdicts.values()),
    }
