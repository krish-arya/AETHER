"""Scenario generation: windows -> valid OHLCV -> a full synthetic market.

Every object produced here is synthetic and says so (``is_synthetic=True``,
``provider="timegan"``). Synthetic scenarios are for training augmentation, stress
testing, exploration and robustness checks. They are **not** forecasts, and they are
never a substitute for out-of-sample tests on real data (PRD 7.1, 7.6).

OHLCV derivation (PRD 7.4) from the generated variables, per asset:

    open_t   = close_{t-1} * exp(gap_t)
    close_t  = open_t * exp(intra_t)
    high_t   = max(open_t, close_t) * exp(wick_t / 2)
    low_t    = min(open_t, close_t) * exp(-wick_t / 2)
    value_t  = base_value * exp(volume_t)          (rupees traded)
    volume_t = max(round(value_t / close_t), 1)    (shares)

so ``low <= open, close <= high`` and ``volume >= 1`` hold by construction.

Long paths are built by stitching windows end to end. The regime of each window
follows a Markov chain estimated from the training period's own window regimes, so
regimes persist and change as they did historically. Known limitation: temporal
dependence is modelled inside a window only; there is a seam every
``sequence_length`` sessions.
"""

import numpy as np
import pandas as pd
import torch

from ml.backtesting.data import MarketData
from ml.features.config import FeatureConfig
from ml.features.engine import compute_features, market_proxy_returns
from ml.features.regime import REGIMES, RegimeConfig, detect_regimes
from ml.gan.data import CONDITIONS
from ml.gan.training.train import TrainedGAN, _one_hot
from ml.rl.observation import ScenarioFeatures

PURPOSES = (
    "training_augmentation",
    "stress_testing",
    "scenario_exploration",
    "robustness_evaluation",
)
SCENARIO_STATS = ("scenario_volatility", "scenario_cvar", "scenario_mean")


def generate_windows(gan: TrainedGAN, condition_codes: np.ndarray, seed: int) -> np.ndarray:
    """Raw variables [n, L, 4N] for the given per-window condition codes."""
    generator = torch.Generator().manual_seed(seed)
    standardised = gan.model.sample(
        _one_hot(np.asarray(condition_codes)), gan.cfg.sequence_length, generator
    ).numpy()
    standardised = standardised.astype("float64")
    if gan.calibration is not None:
        standardised = gan.calibration.apply(standardised, np.asarray(condition_codes))
    standardised = np.clip(standardised, -gan.cfg.clip_sigma, gan.cfg.clip_sigma)
    return gan.windows.inverse(standardised)


def window_returns(gan: TrainedGAN, raw: np.ndarray) -> np.ndarray:
    """Close-to-close log returns [n, L, N] implied by raw windows."""
    parts = gan.windows.split(raw)
    return parts["gap"] + parts["intra"]


def regime_transition_matrix(codes: np.ndarray, step: int) -> np.ndarray:
    """Transition probabilities between window regimes ``step`` windows apart
    (rows without observations fall back to staying put)."""
    k = len(CONDITIONS)
    counts = np.zeros((k, k))
    for a, b in zip(codes[:-step], codes[step:], strict=True):
        counts[a, b] += 1
    totals = counts.sum(axis=1, keepdims=True)
    return np.where(totals > 0, counts / np.maximum(totals, 1), np.eye(k))


def sample_regime_chain(gan: TrainedGAN, n_windows: int, rng: np.random.Generator) -> np.ndarray:
    """Regime per stitched window, following the training period's regime dynamics."""
    codes = gan.windows.conditions
    defined = codes[codes != CONDITIONS.index("undefined")]
    source = defined if len(defined) else codes
    transitions = regime_transition_matrix(source, gan.cfg.sequence_length)
    chain = [int(rng.choice(source))]
    for _ in range(n_windows - 1):
        chain.append(int(rng.choice(len(CONDITIONS), p=transitions[chain[-1]])))
    return np.asarray(chain)


def windows_to_bars(gan: TrainedGAN, raw: np.ndarray, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Stitch raw windows [n, L, 4N] into one OHLCV bar table (long format)."""
    path = raw.reshape(-1, raw.shape[-1])[: len(sessions)]
    parts = gan.windows.split(path)
    w = gan.windows
    close = w.last_close * np.exp(np.cumsum(parts["gap"] + parts["intra"], axis=0))
    open_ = close / np.exp(parts["intra"])
    high = np.maximum(open_, close) * np.exp(parts["wick"] / 2)
    low = np.minimum(open_, close) * np.exp(-parts["wick"] / 2)
    value = w.base_value * np.exp(parts["volume"])
    volume = np.maximum(np.round(value / close), 1.0)

    frames = []
    for j, symbol in enumerate(w.symbols):
        frame = pd.DataFrame(
            {
                "symbol": symbol,
                "trade_date": sessions[: len(path)],
                "open": open_[:, j],
                "high": high[:, j],
                "low": low[:, j],
                "close": close[:, j],
                "volume": volume[:, j].astype("int64"),
                "turnover": volume[:, j] * close[:, j],
            }
        )
        frames.append(frame)
    bars = pd.concat(frames, ignore_index=True)
    # No corporate actions in a synthetic market: adjusted = unadjusted.
    bars["adj_factor"] = bars["split_factor"] = 1.0
    for column in ("open", "high", "low", "close"):
        bars[f"adj_{column}"] = bars[column]
    bars["adj_volume"] = bars["volume"].astype("float64")
    return bars


def synthetic_market(
    gan: TrainedGAN,
    template: MarketData,
    n_sessions: int,
    seed: int,
    *,
    features: FeatureConfig | None = None,
    regime: RegimeConfig | None = None,
    regime_codes: np.ndarray | None = None,
) -> MarketData:
    """A complete synthetic market with the template's symbols, features and regimes.

    It runs through the same feature engine and regime detector as real data, so it
    can be used anywhere a real ``MarketData`` can. Session dates are borrowed from the
    start of the generator's training period (only so dated cost schedules apply); the
    prices on those dates are not real.
    """
    length = gan.cfg.sequence_length
    n_windows = int(np.ceil(n_sessions / length))
    rng = np.random.default_rng(seed)
    codes = regime_codes if regime_codes is not None else sample_regime_chain(gan, n_windows, rng)
    raw = generate_windows(gan, codes, seed)

    sessions = template.sessions[template.sessions >= pd.Timestamp(gan.train_start)][:n_sessions]
    if len(sessions) < n_sessions:
        sessions = pd.bdate_range(gan.train_start, periods=n_sessions)
    bars = windows_to_bars(gan, raw, sessions)

    feature_table = compute_features(bars, features or FeatureConfig())
    regimes = detect_regimes(market_proxy_returns(bars), regime or RegimeConfig())
    securities = template.securities.reset_index()[["symbol", "sector"]].assign(
        listing_date=pd.NaT, delisting_date=pd.NaT
    )
    return MarketData.from_frames(
        bars,
        securities,
        feature_table,
        regimes,
        symbols=list(template.symbols),
        dataset_id=f"timegan-seed{seed}",
        is_synthetic=True,
        provider="timegan",
    )


def scenario_statistics(gan: TrainedGAN, n_per_regime: int, seed: int) -> pd.DataFrame:
    """Per regime and asset: statistics of generated ``sequence_length``-session paths.

    scenario_volatility   annualised volatility of daily returns
    scenario_cvar         mean of the worst 5% window returns (a loss is negative)
    scenario_mean         mean window return
    """
    rows = []
    for code, label in enumerate(REGIMES):
        raw = generate_windows(gan, np.full(n_per_regime, code), seed + code)
        returns = window_returns(gan, raw)  # [n, L, N]
        total = np.exp(returns.sum(axis=1)) - 1  # [n, N]
        threshold = np.quantile(total, 0.05, axis=0)
        for j, symbol in enumerate(gan.windows.symbols):
            tail = total[:, j][total[:, j] <= threshold[j]]
            rows.append(
                {
                    "regime": label,
                    "symbol": symbol,
                    "scenario_volatility": float(returns[:, :, j].std() * np.sqrt(252)),
                    "scenario_cvar": float(tail.mean()),
                    "scenario_mean": float(total[:, j].mean()),
                }
            )
    return pd.DataFrame(rows)


def scenario_features(
    gan: TrainedGAN, symbols: list[str], n_per_regime: int = 400, seed: int = 0
) -> tuple[ScenarioFeatures, pd.DataFrame]:
    """GAN features for the RL state (PRD 9.3): regime-conditional scenario risk per
    asset, standardised across all (regime, asset) pairs. Assets the generator does
    not model get zeros. Returns the features and the unscaled table."""
    table = scenario_statistics(gan, n_per_regime, seed)
    return scenario_features_from_table(table, symbols), table


def scenario_features_from_table(table: pd.DataFrame, symbols: list[str]) -> ScenarioFeatures:
    """Rebuild the RL scenario features from a saved ``scenario_statistics`` table."""
    names = list(SCENARIO_STATS)
    centre, spread = table[names].mean(), table[names].std(ddof=0).replace(0, 1.0)
    by_regime = {}
    for label, group in table.groupby("regime"):
        scaled = ((group.set_index("symbol")[names] - centre) / spread).reindex(symbols)
        by_regime[label] = scaled.fillna(0.0).to_numpy()
    return ScenarioFeatures(names=names, by_regime=by_regime)


def stress_scenarios(gan: TrainedGAN, n: int, seed: int) -> pd.DataFrame:
    """Asset returns over ``sequence_length`` sessions in generated market-stress
    windows (one row per scenario) -- input for ``ml.risk.analytics.stress_test``."""
    code = CONDITIONS.index("market_stress")
    returns = window_returns(gan, generate_windows(gan, np.full(n, code), seed))
    total = np.exp(returns.sum(axis=1)) - 1
    index = [f"timegan-stress-{i:03d}" for i in range(n)]
    return pd.DataFrame(total, index=index, columns=gan.windows.symbols)
