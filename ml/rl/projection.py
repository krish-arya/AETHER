"""Action projection (PRD 9.4): raw policy output -> feasible long-only weights.

The policy emits one number per asset plus one for cash, each in [-1, 1]. They are
scaled to logits and passed through a softmax, which guarantees non-negative weights
that sum to one. Assets that cannot be traded are masked out before the softmax.
The result is then handed to the risk engine, which applies exposure limits.

The same function is used by the training environment and by the evaluation strategy.
"""

import numpy as np


def action_to_weights(action: np.ndarray, tradable: np.ndarray, logit_scale: float) -> np.ndarray:
    """Asset weights (length N; cash is ``1 - sum``) from an action of length N + 1."""
    action = np.asarray(action, dtype="float64").ravel()
    n = len(tradable)
    if len(action) != n + 1:
        raise ValueError(f"expected an action of length {n + 1}, got {len(action)}")
    logits = np.clip(np.nan_to_num(action, nan=0.0), -1.0, 1.0) * logit_scale
    mask = np.append(np.asarray(tradable, dtype=bool), True)  # cash is always available
    logits = np.where(mask, logits, -np.inf)
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    return weights[:n]
