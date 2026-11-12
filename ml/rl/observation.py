"""Unified market state: turns everything known at the close of T into one vector.

This is the single place where market, regime, news (LLM), scenario (GAN) and
portfolio information are assembled for the agent (PRD 9.3, 10). The training
environment and the evaluation strategy both call it, so the policy sees exactly the
same representation in training, backtests and live allocation.

Layout (``spec`` maps each group to its slice of the vector)

    market      N x F   z-scored technical features (statistics from the training
                        window only), clipped; 0 for undefined values and for assets
                        that cannot be traded
    regime      9       one-hot of the 6 regimes + volatility_level, trend_strength,
                        confidence (all 0 while the regime is undefined)
    news        N x K   LLM news features for date T (0 with ``news_available`` = 0
                        where there is no news)
    scenario    N x S   GAN scenario-risk statistics for the regime at T
    portfolio   2N + 2  current asset weights, cash, drawdown, tradable mask

Point-in-time: row ``t`` of every block uses information dated <= session ``t``; the
scaler is fitted on the training window and frozen. Groups not listed in ``groups`` are
simply absent, which is how ablation studies remove a component.

Known limitation: NaN features become 0 after scaling (the feature's training mean);
the tradable mask in the portfolio block is the agent's only signal that an asset is
unavailable.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.backtesting.data import MarketData
from ml.features.regime import REGIMES

REGIME_EXTRAS = ("volatility_level", "trend_strength", "confidence")


@dataclass(frozen=True)
class ScenarioFeatures:
    """Per-regime, per-asset scenario statistics from the GAN (``names`` x N)."""

    names: list[str]
    by_regime: dict[str, np.ndarray]  # regime -> array [N, len(names)], already scaled


class ObservationBuilder:
    def __init__(
        self,
        market: MarketData,
        feature_names: list[str],
        groups: list[str],
        *,
        news: pd.DataFrame | None = None,
        scenario: ScenarioFeatures | None = None,
        clip: float = 5.0,
    ) -> None:
        if market.features is None:
            raise ValueError("the market has no feature table")
        missing = [name for name in feature_names if name not in market.features.columns]
        if missing:
            raise ValueError(f"features not in the dataset: {missing}")
        if "news" in groups and news is None:
            raise ValueError('the "news" group needs news features (LLM engine output)')
        if "scenario" in groups and scenario is None:
            raise ValueError('the "scenario" group needs scenario features (GAN engine output)')

        self.market = market
        self.symbols = list(market.symbols)
        self.feature_names = list(feature_names)
        self.groups = [g for g in ("market", "regime", "news", "scenario") if g in groups]
        self.clip = clip
        self.scenario = scenario
        sessions, n = market.sessions, len(self.symbols)

        self._features = np.stack(
            [
                market.features[name]
                .unstack("symbol")
                .reindex(index=sessions, columns=self.symbols)
                .to_numpy(dtype="float64")
                for name in self.feature_names
            ],
            axis=-1,
        )  # [T, N, F]

        listing = market.securities["listing_date"].to_numpy()
        delisting = market.securities["delisting_date"].to_numpy()
        days = sessions.to_numpy()[:, None]
        listed = (pd.isna(listing) | (listing <= days)) & (pd.isna(delisting) | (days < delisting))
        self.listed = np.asarray(listed, dtype=bool)
        self.tradable = self.listed & np.isfinite(market.close.to_numpy())

        self._regime = np.zeros((len(sessions), len(REGIMES) + len(REGIME_EXTRAS)))
        self._regime_labels: list[str | None] = [None] * len(sessions)
        if market.regimes is not None:
            regimes = market.regimes.reindex(sessions)
            for column, label in enumerate(REGIMES):
                self._regime[:, column] = (regimes["regime"] == label).to_numpy(dtype="float64")
            for offset, name in enumerate(REGIME_EXTRAS):
                self._regime[:, len(REGIMES) + offset] = regimes[name].fillna(0.0).to_numpy()
            self._regime_labels = [None if pd.isna(x) else str(x) for x in regimes["regime"]]

        self.news_names: list[str] = []
        self._news = np.zeros((len(sessions), n, 0))
        if news is not None:
            self.news_names = [c for c in news.columns if c not in ("symbol", "trade_date")]
            indexed = news.set_index(["trade_date", "symbol"])
            self._news = np.stack(
                [
                    indexed[name]
                    .unstack("symbol")
                    .reindex(index=sessions, columns=self.symbols)
                    .fillna(0.0)
                    .to_numpy(dtype="float64")
                    for name in self.news_names
                ],
                axis=-1,
            )

        self.mean = np.zeros(len(self.feature_names))
        self.std = np.ones(len(self.feature_names))
        self.fitted = False
        self.spec = self._layout()
        self.size = max(stop for _, stop in self.spec.values())

    # --------------------------------------------------------------------- layout

    def _layout(self) -> dict[str, tuple[int, int]]:
        n = len(self.symbols)
        sizes = {
            "market": n * len(self.feature_names),
            "regime": self._regime.shape[1],
            "news": n * len(self.news_names),
            "scenario": n * len(self.scenario.names) if self.scenario else 0,
        }
        spec, cursor = {}, 0
        for group in self.groups:
            spec[group] = (cursor, cursor + sizes[group])
            cursor += sizes[group]
        spec["portfolio"] = (cursor, cursor + 2 * n + 2)
        return spec

    # --------------------------------------------------------------------- scaler

    def fit(self, start, end) -> "ObservationBuilder":
        """Fit the feature scaler on sessions in ``start..end`` (the training window)."""
        sessions = self.market.sessions
        lo = sessions.searchsorted(pd.Timestamp(start), side="left")
        hi = sessions.searchsorted(pd.Timestamp(end), side="right")
        if hi - lo < 2:
            raise ValueError("the scaler window contains fewer than two sessions")
        window = self._features[lo:hi].reshape(-1, len(self.feature_names))
        self.mean = np.nanmean(window, axis=0)
        std = np.nanstd(window, axis=0)
        self.std = np.where(std > 1e-12, std, 1.0)
        self.mean = np.where(np.isfinite(self.mean), self.mean, 0.0)
        self.fitted = True
        return self

    def for_market(self, market: MarketData, news: pd.DataFrame | None = None):
        """A builder over another market (e.g. a GAN scenario) sharing this scaler."""
        other = ObservationBuilder(
            market, self.feature_names, self.groups, news=news, scenario=self.scenario,
            clip=self.clip,
        )  # fmt: skip
        if other.symbols != self.symbols:
            raise ValueError("markets must contain the same symbols in the same order")
        other.mean, other.std, other.fitted = self.mean.copy(), self.std.copy(), self.fitted
        return other

    def state(self) -> dict:
        """Everything needed to rebuild the builder for the same dataset."""
        return {
            "symbols": self.symbols,
            "feature_names": self.feature_names,
            "groups": self.groups,
            "clip": self.clip,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "size": self.size,
            "spec": {group: list(bounds) for group, bounds in self.spec.items()},
        }

    def load_state(self, state: dict) -> "ObservationBuilder":
        if state["symbols"] != self.symbols or state["feature_names"] != self.feature_names:
            raise ValueError("saved observation state does not match this dataset / config")
        self.mean, self.std = np.asarray(state["mean"]), np.asarray(state["std"])
        self.fitted = True
        return self

    # ---------------------------------------------------------------- observation

    def row_of(self, day) -> int:
        return int(self.market.sessions.get_loc(pd.Timestamp(day)))

    def regime_label(self, row: int) -> str | None:
        return self._regime_labels[row]

    def observe(self, row: int, asset_weights: np.ndarray, drawdown: float) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("call fit() with the training window before observing")
        tradable = self.tradable[row]
        parts = []
        for group in self.groups:
            if group == "market":
                scaled = (self._features[row] - self.mean) / self.std
                scaled = np.clip(np.nan_to_num(scaled, nan=0.0), -self.clip, self.clip)
                parts.append((scaled * tradable[:, None]).ravel())
            elif group == "regime":
                parts.append(self._regime[row])
            elif group == "news":
                parts.append(self._news[row].ravel())
            elif group == "scenario":
                label = self._regime_labels[row]
                block = self.scenario.by_regime.get(label) if label else None
                if block is None:
                    block = np.zeros((len(self.symbols), len(self.scenario.names)))
                parts.append(block.ravel())
        weights = np.asarray(asset_weights, dtype="float64")
        parts.append(
            np.concatenate([weights, [1.0 - weights.sum(), drawdown], tradable.astype("float64")])
        )
        return np.concatenate(parts).astype(np.float32)
