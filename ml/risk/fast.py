"""NumPy implementation of the hard risk limits for the RL training loop.

``ml.risk.limits.enforce`` is the reference implementation (pandas, with a full
decision record). Training calls the projection hundreds of thousands of times, so
this module re-implements exactly the same arithmetic on arrays. The two are tested
against each other on random allocations (``tests/ml/test_risk.py``); evaluation and
live allocation always go through the reference implementation.

Not supported here: ``max_concentration_hhi`` (training configs must leave it unset).
"""

import numpy as np

from ml.risk.limits import TOLERANCE, RiskLimits


class FastRiskProjector:
    def __init__(self, limits: RiskLimits, sector_ids: np.ndarray) -> None:
        if limits.max_concentration_hhi is not None:
            raise ValueError("FastRiskProjector does not support max_concentration_hhi")
        self.limits = limits
        self.sector_ids = np.asarray(sector_ids, dtype=np.int64)
        self.n_sectors = int(self.sector_ids.max()) + 1 if len(self.sector_ids) else 0

    def _hard(self, weights: np.ndarray, eligible: np.ndarray, drawdown: float) -> np.ndarray:
        limits = self.limits
        out = np.clip(weights, 0.0, None)
        out = np.where(eligible, out, 0.0)
        out = np.minimum(out, limits.max_weight)
        totals = np.bincount(self.sector_ids, weights=out, minlength=self.n_sectors)
        scale = np.where(
            totals > limits.max_sector_weight + TOLERANCE,
            limits.max_sector_weight / np.maximum(totals, TOLERANCE),
            1.0,
        )
        out = out * scale[self.sector_ids]
        cap = 1.0 - limits.min_cash
        if limits.max_drawdown is not None and -drawdown >= limits.max_drawdown:
            cap = min(cap, limits.drawdown_exposure_cap)
        invested = out.sum()
        if invested > cap + TOLERANCE:
            out = out * (cap / invested)
        return out

    @staticmethod
    def _turnover(new: np.ndarray, old: np.ndarray) -> float:
        return 0.5 * (np.abs(new - old).sum() + abs(new.sum() - old.sum()))

    def project(
        self,
        target: np.ndarray,
        previous: np.ndarray,
        eligible: np.ndarray,
        drawdown: float = 0.0,
    ) -> np.ndarray:
        """Compliant asset weights. ``eligible`` marks assets that pass the liquidity
        limit (and are tradable); ineligible assets get zero weight."""
        compliant = self._hard(target, eligible, drawdown)
        limit = self.limits.max_turnover
        if limit is not None:
            holdings = self._hard(previous, eligible, drawdown)
            required = self._turnover(holdings, previous)
            desired = self._turnover(compliant, holdings)
            if required + desired > limit + TOLERANCE:
                budget = max(limit - required, 0.0)
                step = budget / desired if desired > TOLERANCE else 0.0
                compliant = holdings + step * (compliant - holdings)
        return np.clip(compliant, 0.0, None)
