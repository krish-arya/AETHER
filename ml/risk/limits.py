"""Deterministic risk controls (PRD 12). Independent of every model.

No AI component can bypass this module: it takes a proposed allocation and returns a
compliant one (or rejects the decision), using only arithmetic on the inputs.

Limits (:class:`RiskLimits`, all configurable)
    max_weight               largest weight of a single asset
    max_sector_weight        largest combined weight of one sector
    max_concentration_hhi    cap on the Herfindahl index of the invested portfolio
    min_cash                 minimum cash reserve
    min_adv                  assets whose average daily traded value is below this
                             (rupees) may not be held
    max_drawdown             once the portfolio's drawdown reaches this, invested
                             exposure is capped at ``drawdown_exposure_cap``
    max_turnover             one-way turnover allowed per rebalance

Enforcement (:func:`enforce`), PRD 12.3
    1. A non-finite raw allocation is rejected outright.
    2. The raw allocation's violations are recorded.
    3. Hard limits are applied in a fixed order: no shorts -> liquidity -> asset cap ->
       sector cap -> concentration -> gross exposure (cash reserve / drawdown cap).
       Weight removed by a limit goes to cash; nothing is redistributed, so applying a
       later limit can never re-break an earlier one.
    4. Turnover: the current holdings are first made compliant with the same hard
       limits (the minimum trade that risk requires), then the portfolio moves from
       there towards the compliant target only as far as the turnover budget allows.
       A blend of two compliant portfolios is compliant, so hard limits always hold;
       if compliance alone needs more turnover than the budget, compliance wins and
       the overrun is logged.
    5. The result is re-validated. If it still violates a hard limit the decision is
       rejected (the caller keeps its current holdings).

Every decision is returned as a :class:`RiskDecision` that records what was violated
and what was applied, for the risk log.

Known limitations: limits act on weights at decision time, not on intraday exposure;
sector labels come from the securities master; correlations are not used by the
limits (see ``analytics`` for VaR / stress measures, which are reported, not enforced).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

TOLERANCE = 1e-9
HARD_LIMITS = (
    "no_short",
    "min_adv",
    "max_weight",
    "max_sector_weight",
    "max_concentration_hhi",
    "min_cash",
    "drawdown_exposure_cap",
)


class RiskLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_weight: float = Field(0.20, gt=0, le=1)
    max_sector_weight: float = Field(0.40, gt=0, le=1)
    max_concentration_hhi: float | None = Field(None, gt=0, le=1)
    min_cash: float = Field(0.02, ge=0, lt=1)
    min_adv: float = Field(0.0, ge=0)
    max_drawdown: float | None = Field(0.25, gt=0, lt=1)
    drawdown_exposure_cap: float = Field(0.5, ge=0, le=1)
    max_turnover: float | None = Field(0.5, gt=0, le=1)


@dataclass(frozen=True)
class Violation:
    limit: str
    value: float
    threshold: float
    subject: str | None = None  # the asset or sector concerned

    def as_dict(self) -> dict:
        return {
            "limit": self.limit,
            "value": round(float(self.value), 6),
            "threshold": float(self.threshold),
            "subject": self.subject,
        }


@dataclass
class RiskDecision:
    """Outcome of one enforcement. ``weights`` is None when the decision was rejected."""

    weights: pd.Series | None
    rejected: bool
    raw_violations: list[Violation] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)
    final_violations: list[Violation] = field(default_factory=list)
    reason: str | None = None
    turnover: float = 0.0

    @property
    def modified(self) -> bool:
        return bool(self.applied)

    def as_record(self) -> dict:
        return {
            "rejected": self.rejected,
            "reason": self.reason,
            "applied": list(self.applied),
            "raw_violations": [v.as_dict() for v in self.raw_violations],
            "final_violations": [v.as_dict() for v in self.final_violations],
            "turnover": round(float(self.turnover), 6),
            "cash": None if self.weights is None else round(float(1 - self.weights.sum()), 6),
        }


def one_way_turnover(new: pd.Series, old: pd.Series) -> float:
    """Half the sum of absolute weight changes, cash included."""
    index = new.index.union(old.index)
    a = new.reindex(index, fill_value=0.0)
    b = old.reindex(index, fill_value=0.0)
    return float(0.5 * ((a - b).abs().sum() + abs(a.sum() - b.sum())))


def herfindahl(weights: pd.Series) -> float:
    """HHI of the invested part (1/N for N equal positions, 1 for a single position)."""
    invested = weights.sum()
    if invested <= TOLERANCE:
        return 0.0
    shares = weights / invested
    return float((shares**2).sum())


def _exposure_cap(drawdown: float, limits: RiskLimits) -> tuple[float, str]:
    cap, name = 1.0 - limits.min_cash, "min_cash"
    in_breach = limits.max_drawdown is not None and -drawdown >= limits.max_drawdown
    if in_breach and limits.drawdown_exposure_cap < cap:
        cap, name = limits.drawdown_exposure_cap, "drawdown_exposure_cap"
    return cap, name


def check_limits(
    weights: pd.Series,
    *,
    limits: RiskLimits,
    sectors: pd.Series | None = None,
    adv: pd.Series | None = None,
    drawdown: float = 0.0,
    previous: pd.Series | None = None,
) -> list[Violation]:
    """Every limit the allocation breaks (empty list = compliant)."""
    found: list[Violation] = []
    for symbol, weight in weights.items():
        if weight < -TOLERANCE:
            found.append(Violation("no_short", weight, 0.0, str(symbol)))
        if weight > limits.max_weight + TOLERANCE:
            found.append(Violation("max_weight", weight, limits.max_weight, str(symbol)))
    if adv is not None and limits.min_adv > 0:
        liquidity = adv.reindex(weights.index)
        for symbol in weights.index[(weights > TOLERANCE) & ~(liquidity >= limits.min_adv)]:
            found.append(
                Violation("min_adv", liquidity.get(symbol, np.nan), limits.min_adv, str(symbol))
            )
    if sectors is not None:
        by_sector = weights.groupby(sectors.reindex(weights.index).fillna("Unknown")).sum()
        for sector, weight in by_sector.items():
            if weight > limits.max_sector_weight + TOLERANCE:
                found.append(
                    Violation("max_sector_weight", weight, limits.max_sector_weight, str(sector))
                )
    if limits.max_concentration_hhi is not None:
        hhi = herfindahl(weights.clip(lower=0))
        if hhi > limits.max_concentration_hhi + TOLERANCE:
            found.append(Violation("max_concentration_hhi", hhi, limits.max_concentration_hhi))
    cap, name = _exposure_cap(drawdown, limits)
    invested = float(weights.sum())
    if invested > cap + TOLERANCE:
        found.append(Violation(name, invested, cap))
    if previous is not None and limits.max_turnover is not None:
        turnover = one_way_turnover(weights, previous)
        if turnover > limits.max_turnover + TOLERANCE:
            found.append(Violation("max_turnover", turnover, limits.max_turnover))
    return found


def _apply_hard_limits(
    weights: pd.Series,
    *,
    limits: RiskLimits,
    sectors: pd.Series | None,
    adv: pd.Series | None,
    drawdown: float,
) -> tuple[pd.Series, list[str]]:
    """Project onto the hard limits. Only ever reduces weights (excess goes to cash)."""
    applied: list[str] = []
    out = weights.astype("float64").copy()

    if (out < 0).any():
        out = out.clip(lower=0.0)
        applied.append("no_short")
    if adv is not None and limits.min_adv > 0:
        illiquid = (out > TOLERANCE) & ~(adv.reindex(out.index) >= limits.min_adv)
        if illiquid.any():
            out[illiquid] = 0.0
            applied.append("min_adv")
    if (out > limits.max_weight + TOLERANCE).any():
        out = out.clip(upper=limits.max_weight)
        applied.append("max_weight")
    if sectors is not None:
        labels = sectors.reindex(out.index).fillna("Unknown")
        totals = out.groupby(labels).transform("sum")
        over = totals > limits.max_sector_weight + TOLERANCE
        if over.any():
            out = out.where(~over, out * limits.max_sector_weight / totals)
            applied.append("max_sector_weight")
    too_concentrated = (
        limits.max_concentration_hhi is not None
        and out.sum() > TOLERANCE
        and herfindahl(out) > limits.max_concentration_hhi + TOLERANCE
    )
    if too_concentrated:
        # Shrink every position towards the equal-weight portfolio of the held
        # assets. This can lift small positions, so in rare cases a sector cap is
        # re-broken; the final validation in ``enforce`` then rejects the decision.
        held = out > TOLERANCE
        equal = pd.Series(0.0, index=out.index)
        equal[held] = out.sum() / held.sum()
        equal = equal.clip(upper=out.max())
        low, high = 0.0, 1.0
        for _ in range(60):
            mid = (low + high) / 2
            if herfindahl((1 - mid) * out + mid * equal) > limits.max_concentration_hhi:
                low = mid
            else:
                high = mid
        out = (1 - high) * out + high * equal
        applied.append("max_concentration_hhi")
    cap, name = _exposure_cap(drawdown, limits)
    invested = out.sum()
    if invested > cap + TOLERANCE:
        out = out * (cap / invested)
        applied.append(name)
    return out, applied


def enforce(
    target: pd.Series,
    *,
    limits: RiskLimits,
    previous: pd.Series | None = None,
    sectors: pd.Series | None = None,
    adv: pd.Series | None = None,
    drawdown: float = 0.0,
) -> RiskDecision:
    """Turn a proposed allocation into a compliant one, or reject it.

    ``target`` / ``previous``: asset weights indexed by symbol (cash is the remainder).
    ``sectors``: sector label per symbol. ``adv``: average daily traded value per
    symbol. ``drawdown``: the portfolio's current drawdown as a non-positive fraction.
    """
    values = target.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        return RiskDecision(None, True, reason="allocation contains NaN or infinite weights")
    if target.index.has_duplicates:
        return RiskDecision(None, True, reason="allocation contains duplicate symbols")

    context = {"limits": limits, "sectors": sectors, "adv": adv, "drawdown": drawdown}
    previous = (
        pd.Series(0.0, index=target.index)
        if previous is None
        else previous.astype("float64").clip(lower=0.0)
    )
    index = target.index.union(previous.index)
    raw = target.reindex(index, fill_value=0.0).astype("float64")
    previous = previous.reindex(index, fill_value=0.0)

    raw_violations = check_limits(raw, previous=previous, **context)
    if raw.sum() > 1 + 1e-6:
        raw_violations.append(Violation("no_leverage", float(raw.sum()), 1.0))
    compliant, applied = _apply_hard_limits(raw, **context)

    if limits.max_turnover is not None:
        holdings, _ = _apply_hard_limits(previous, **context)
        required = one_way_turnover(holdings, previous)
        desired = one_way_turnover(compliant, holdings)
        if required + desired > limits.max_turnover + TOLERANCE:
            budget = max(limits.max_turnover - required, 0.0)
            step = budget / desired if desired > TOLERANCE else 0.0
            compliant = holdings + step * (compliant - holdings)
            applied.append("max_turnover")
            if required > limits.max_turnover + TOLERANCE:
                applied.append("max_turnover_exceeded_for_compliance")

    compliant = compliant.clip(lower=0.0)
    final = [v for v in check_limits(compliant, previous=None, **context) if v.limit in HARD_LIMITS]
    turnover = one_way_turnover(compliant, previous)
    if final:
        return RiskDecision(
            None, True, raw_violations, applied, final,
            reason="no compliant allocation could be produced", turnover=0.0,
        )  # fmt: skip
    return RiskDecision(compliant, False, raw_violations, applied, [], turnover=turnover)
