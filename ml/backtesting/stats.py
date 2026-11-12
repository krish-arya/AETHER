"""Statistical validation of backtest results (PRD 11.4, 20.4).

A Sharpe ratio from a few years of daily data is a noisy estimate. Before saying one
strategy "beats" another, the difference should be distinguishable from noise.

Method: circular block bootstrap. Daily returns are resampled in contiguous blocks
(wrapping around the end) so that volatility clustering and short-range
autocorrelation are preserved. Strategy and benchmark are resampled with the *same*
blocks, which keeps their cross-correlation intact (a paired test).

Not yet implemented: Deflated Sharpe Ratio / multiple-testing corrections. They become
necessary once many configurations are tried (hyper-parameter search, Phase 5).
"""

import numpy as np
import pandas as pd

from ml.backtesting.metrics import sharpe_ratio


def _block_indices(n: int, block: int, samples: int, rng: np.random.Generator) -> np.ndarray:
    """``samples`` x ``n`` matrix of resampled positions (circular blocks)."""
    blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(samples, blocks))
    offsets = np.arange(block)
    return ((starts[:, :, None] + offsets) % n).reshape(samples, -1)[:, :n]


def _sharpe_matrix(matrix: np.ndarray, daily_rf: float, periods_per_year: int) -> np.ndarray:
    excess = matrix - daily_rf
    deviation = excess.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            deviation > 1e-15, excess.mean(axis=1) / deviation * np.sqrt(periods_per_year), np.nan
        )


def bootstrap_sharpe(
    returns: pd.Series,
    *,
    benchmark: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    samples: int = 2000,
    block_sessions: int = 21,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap confidence interval for a Sharpe ratio, or for a Sharpe difference.

    Without ``benchmark``: interval for the strategy's Sharpe ratio.
    With ``benchmark``: interval for ``sharpe(strategy) - sharpe(benchmark)`` plus
    ``p_value``, the two-sided bootstrap p-value for "the difference is zero".
    ``significant`` is True when the interval excludes zero.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    frame = returns.to_frame("strategy")
    if benchmark is not None:
        frame = frame.join(benchmark.rename("benchmark"), how="inner")
    frame = frame.dropna()
    n = len(frame)
    if n < 2 * block_sessions:
        raise ValueError("not enough observations for the requested block length")

    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    rng = np.random.default_rng(seed)
    positions = _block_indices(n, block_sessions, samples, rng)
    strategy = frame["strategy"].to_numpy()
    point = sharpe_ratio(frame["strategy"], risk_free_rate, periods_per_year)
    draws = _sharpe_matrix(strategy[positions], daily_rf, periods_per_year)
    if benchmark is not None:
        market = frame["benchmark"].to_numpy()
        point -= sharpe_ratio(frame["benchmark"], risk_free_rate, periods_per_year)
        draws = draws - _sharpe_matrix(market[positions], daily_rf, periods_per_year)

    draws = draws[np.isfinite(draws)]
    if len(draws) == 0 or not np.isfinite(point):
        nan = float("nan")
        return {"estimate": nan, "ci_low": nan, "ci_high": nan, "p_value": nan,
                "significant": False, "observations": float(n)}  # fmt: skip
    tail = (1 - confidence) / 2
    low, high = np.quantile(draws, [tail, 1 - tail])
    # Two-sided p-value: how often the centred bootstrap distribution is as extreme
    # as the observed estimate.
    centred = draws - draws.mean()
    p_value = float((np.abs(centred) >= abs(point)).mean())
    return {
        "estimate": float(point),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": p_value,
        "significant": bool(low > 0 or high < 0),
        "observations": float(n),
    }
