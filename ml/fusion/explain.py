"""Explainability engine (PRD 13): what the allocator did, and the evidence for it.

``explain_allocation`` produces, for one decision date:

    summary     market regime, proposed allocation, cash, changes from the previous
                allocation, and a plain-language description assembled from those numbers
    evidence    model id, dataset id, decision timestamp, the raw policy output, the risk
                engine's record (limits violated and applied), feature values for the
                largest positions, and how certain the policy was (action spread)
    sensitivity how much the allocation changes when one information source -- or one
                feature of one asset -- is replaced by its training-period average

Honesty rules built into this module
    * Every sentence in ``narrative`` is generated from the numbers in the same record.
      No language model writes it, so nothing can be invented.
    * ``sensitivity`` is an occlusion measure: a *statistical association* between
      inputs and this model's output. It is not a causal explanation of the market or
      proof of why the policy "decided" something, and the record says so.
    * Attention weights (when the policy uses attention fusion) are reported as what
      the network attended to, nothing more.

Known limitations: occlusion treats inputs independently and replaces them with an
average the model may rarely have seen; interacting features can be under- or
over-credited. The numbers describe one model on one date.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ml.risk.limits import RiskLimits, enforce
from ml.rl.observation import ObservationBuilder
from ml.rl.projection import action_to_weights

ASSOCIATION_NOTE = (
    "Sensitivities are statistical associations for this model on this date (occlusion "
    "analysis). They are not causal explanations and do not predict returns."
)


@dataclass
class AllocationInputs:
    model: object  # Stable-Baselines3 model
    builder: ObservationBuilder
    limits: RiskLimits
    logit_scale: float
    sectors: pd.Series | None
    adv: pd.Series | None
    model_id: str
    dataset_id: str | None


def _propose(inputs: AllocationInputs, observation: np.ndarray, tradable: np.ndarray) -> np.ndarray:
    action, _ = inputs.model.predict(observation, deterministic=True)
    return action_to_weights(action, tradable, inputs.logit_scale)


def _policy_spread(model, observation: np.ndarray) -> float | None:
    """Mean standard deviation of the policy's action distribution (PPO); a rough
    indication of how settled the policy is. None when the algorithm has no such thing."""
    try:
        tensor, _ = model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            distribution = model.policy.get_distribution(tensor)
        return float(distribution.distribution.stddev.mean())
    except (AttributeError, NotImplementedError):
        return None


def _attention(model) -> dict[str, float] | None:
    extractor = getattr(model.policy, "features_extractor", None)
    if extractor is None or not hasattr(extractor, "attention_by_group"):
        return None
    return extractor.attention_by_group()


def explain_allocation(
    inputs: AllocationInputs,
    day,
    previous_weights: pd.Series,
    drawdown: float = 0.0,
    top: int = 5,
) -> dict:
    """Allocation for the session after ``day`` with its explanation record.

    ``previous_weights``: current asset weights by symbol (missing symbols = 0).
    """
    builder = inputs.builder
    symbols = builder.symbols
    row = builder.row_of(day)
    tradable = builder.tradable[row]
    previous = previous_weights.reindex(symbols, fill_value=0.0).astype("float64")
    observation = builder.observe(row, previous.to_numpy(), drawdown)

    raw = pd.Series(_propose(inputs, observation, tradable), index=symbols)
    decision = enforce(
        raw, limits=inputs.limits, previous=previous, sectors=inputs.sectors, adv=inputs.adv,
        drawdown=drawdown,
    )  # fmt: skip
    final = previous if decision.rejected else decision.weights.reindex(symbols, fill_value=0.0)
    changes = final - previous

    # ------------------------------------------------------ sensitivity (occlusion)
    def weights_for(modified: np.ndarray) -> np.ndarray:
        return _propose(inputs, modified.astype(np.float32), tradable)

    baseline = raw.to_numpy()
    by_group = {}
    for group, (lo, hi) in builder.spec.items():
        if group == "portfolio" or hi <= lo:
            continue
        occluded = observation.copy()
        occluded[lo:hi] = 0.0  # the training-period average of every scaled input
        by_group[group] = float(0.5 * np.abs(weights_for(occluded) - baseline).sum())

    holdings = final.sort_values(ascending=False)
    holdings = holdings[holdings > 1e-4].head(top)
    features = builder.feature_names
    lo, _ = builder.spec.get("market", (0, 0))
    raw_values = builder._features[row]  # unscaled feature values [N, F]
    positions = []
    for symbol, weight in holdings.items():
        j = symbols.index(symbol)
        impacts = []
        if "market" in builder.spec:
            for f, name in enumerate(features):
                occluded = observation.copy()
                occluded[lo + j * len(features) + f] = 0.0
                delta = float(weights_for(occluded)[j] - baseline[j])
                value = raw_values[j, f]
                impacts.append(
                    {
                        "feature": name,
                        "value": None if not np.isfinite(value) else float(value),
                        "weight_change_if_average": delta,
                    }
                )
            impacts.sort(key=lambda item: abs(item["weight_change_if_average"]), reverse=True)
        positions.append(
            {
                "symbol": symbol,
                "weight": float(weight),
                "previous_weight": float(previous[symbol]),
                "change": float(changes[symbol]),
                "policy_weight_before_risk": float(raw[symbol]),
                "sector": None if inputs.sectors is None else inputs.sectors.get(symbol),
                "most_influential_features": impacts[:3],
            }
        )

    # --------------------------------------------------------------------- regime
    regime = None
    if builder.market.regimes is not None and pd.Timestamp(day) in builder.market.regimes.index:
        record = builder.market.regimes.loc[pd.Timestamp(day)]
        regime = {
            key: (
                None
                if pd.isna(record[key])
                else (record[key] if key == "regime" else float(record[key]))
            )
            for key in ("regime", "confidence", "volatility_level", "trend_strength", "drawdown")
        }

    next_session = (
        builder.market.sessions[row + 1].strftime("%Y-%m-%d")
        if row + 1 < len(builder.market.sessions)
        else None
    )
    increases = changes[changes > 0.005].sort_values(ascending=False).head(3)
    decreases = changes[changes < -0.005].sort_values().head(3)
    record = decision.as_record()
    narrative = _narrative(
        day, regime, final, increases, decreases, record, by_group, decision.rejected
    )

    return {
        "as_of": pd.Timestamp(day).strftime("%Y-%m-%d"),
        "executable_from": next_session,
        "model_id": inputs.model_id,
        "dataset_id": inputs.dataset_id,
        "is_recommendation_for_real_trading": False,
        "summary": {
            "regime": regime,
            "invested": float(final.sum()),
            "cash": float(1 - final.sum()),
            "positions": int((final > 1e-4).sum()),
            "turnover": float(decision.turnover),
            "largest_increases": {s: float(v) for s, v in increases.items()},
            "largest_decreases": {s: float(v) for s, v in decreases.items()},
            "narrative": narrative,
        },
        "allocation": {s: float(w) for s, w in final.items() if w > 1e-6},
        "previous_allocation": {s: float(w) for s, w in previous.items() if w > 1e-6},
        "evidence": {
            "positions": positions,
            "policy_output_before_risk": {s: float(w) for s, w in raw.items() if w > 1e-4},
            "risk_engine": record,
            "risk_limits": inputs.limits.model_dump(mode="json"),
            "policy_action_spread": _policy_spread(inputs.model, observation),
            "attention_by_source": _attention(inputs.model),
            "observation_groups": [g for g in builder.spec if g != "portfolio"],
        },
        "sensitivity": {
            "by_information_source": by_group,
            "definition": "Share of the portfolio that would be allocated differently if "
            "that source were replaced by its training-period average.",
            "note": ASSOCIATION_NOTE,
        },
    }


def _narrative(day, regime, final, increases, decreases, risk, by_group, rejected) -> list[str]:
    """Plain-language description; each sentence restates numbers from the record."""
    lines = []
    date = pd.Timestamp(day).strftime("%d %b %Y")
    if regime and regime["regime"]:
        lines.append(
            f"On {date} the market regime was {regime['regime'].replace('_', ' ')} "
            f"(volatility at the {regime['volatility_level']:.0%} percentile of its history)."
        )
    else:
        lines.append(f"On {date} no market regime was defined yet.")
    if rejected:
        lines.append(
            "The risk engine rejected the model's proposal, so the current holdings are kept: "
            f"{risk['reason']}."
        )
    else:
        top = final.sort_values(ascending=False).head(3)
        holdings = ", ".join(f"{s} {w:.1%}" for s, w in top.items() if w > 1e-4)
        lines.append(
            f"The model proposes {final.sum():.1%} invested across {(final > 1e-4).sum()} "
            f"positions and {1 - final.sum():.1%} cash; the largest are {holdings or 'none'}."
        )
    if len(increases) or len(decreases):
        up = ", ".join(f"{s} +{v:.1%}" for s, v in increases.items())
        down = ", ".join(f"{s} {v:.1%}" for s, v in decreases.items())
        lines.append(
            "Changes from the current allocation: " + "; ".join(x for x in (up, down) if x) + "."
        )
    else:
        lines.append("No position changes by more than 0.5% of the portfolio.")
    if risk["applied"]:
        lines.append(
            "The risk engine modified the model's raw proposal to satisfy: "
            + ", ".join(sorted(set(risk["applied"]))).replace("_", " ")
            + "."
        )
    elif not rejected:
        lines.append("The model's raw proposal already satisfied every risk limit.")
    if by_group:
        source, value = max(by_group.items(), key=lambda item: item[1])
        lines.append(
            f"The allocation is most sensitive to the {source} inputs: replacing them with "
            f"their average would move {value:.1%} of the portfolio. This is an association "
            "for this model, not a causal explanation."
        )
    return lines
