"""GAN scenario engine: preprocessing, TimeGAN training, generation and evaluation.

Training here uses a few iterations of a tiny network: the tests check mechanics and
invariants, not scenario quality (which ``ml.gan.evaluation`` measures on real runs).
"""

import numpy as np
import pandas as pd
import pytest

from ml.gan.config import GANConfig, QualityGates
from ml.gan.data import CONDITIONS, VARIABLES, build_windows, raw_variables
from ml.gan.evaluation.metrics import compare, evaluate, memorization_check
from ml.gan.generation.generate import (
    generate_windows,
    regime_transition_matrix,
    sample_regime_chain,
    scenario_features,
    stress_scenarios,
    synthetic_market,
    window_returns,
    windows_to_bars,
)
from ml.gan.training.train import load_gan, save_gan, train_timegan
from ml.risk.analytics import stress_test

from .test_rl_env import TRAIN, VALIDATION

CFG = GANConfig(
    sequence_length=12,
    volume_window=20,
    hidden_dim=8,
    num_layers=1,
    noise_dim=4,
    batch_size=32,
    embedding_iterations=4,
    supervised_iterations=4,
    joint_iterations=4,
    calibration_windows=60,
    min_regime_windows=5,
    evaluation_windows=80,
    torch_threads=1,
)


def _windows(market, period=TRAIN, scaler=None):
    return build_windows(
        market, *period, sequence_length=CFG.sequence_length, volume_window=CFG.volume_window,
        clip_sigma=CFG.clip_sigma, scaler=scaler,
    )  # fmt: skip


@pytest.fixture(scope="module")
def gan(market):
    return train_timegan(market, CFG, train_start=TRAIN[0], train_end=TRAIN[1], seed=0)


# ------------------------------------------------------------------ preprocessing


def test_windows_cover_only_assets_with_a_complete_history(market):
    windows = _windows(market)
    late = market.securities["listing_date"].dropna().index[0]
    assert late in windows.excluded  # listed part-way through: reported, not filled
    assert late not in windows.symbols
    n = len(windows.symbols)
    assert windows.windows.shape[1:] == (CFG.sequence_length, len(VARIABLES) * n)
    assert np.isfinite(windows.windows).all()
    assert set(windows.conditions) <= set(range(len(CONDITIONS)))
    assert windows.end_dates.max() <= pd.Timestamp(TRAIN[1])
    assert windows.end_dates.min() >= pd.Timestamp(TRAIN[0])


def test_standardisation_uses_training_statistics_only(market):
    train = _windows(market)
    flat = train.windows.reshape(-1, train.dim)
    assert np.abs(flat.mean(axis=0)).max() < 0.25  # overlapping windows, so not exactly 0
    assert 0.7 < flat.std(axis=0).mean() < 1.1
    heldout = _windows(market, VALIDATION, scaler=train)
    np.testing.assert_array_equal(heldout.mean, train.mean)  # never re-fitted
    np.testing.assert_array_equal(heldout.std, train.std)
    assert heldout.symbols == train.symbols
    assert heldout.end_dates.min() > pd.Timestamp(TRAIN[1])


def test_variables_reconstruct_the_real_prices(market):
    """gap/intra/wick must be an exact re-parameterisation of OHLC."""
    table, _ = raw_variables(market, CFG.volume_window)
    symbol = "SYNTH01"
    gap, intra, wick = (table[(name, symbol)] for name in ("gap", "intra", "wick"))
    close = market.close[symbol]
    rebuilt_close = close.shift(1) * np.exp(gap + intra)
    np.testing.assert_allclose(rebuilt_close.dropna(), close[rebuilt_close.notna()], rtol=1e-12)
    rebuilt_range = np.log(market.high[symbol] / market.low[symbol])
    np.testing.assert_allclose(
        (wick + intra.abs()).dropna(), rebuilt_range[wick.notna()], rtol=1e-9
    )
    assert (wick.dropna() >= 0).all()


def test_inverse_restores_raw_units(market):
    windows = _windows(market)
    raw = windows.inverse(windows.windows.astype("float64"))
    parts = windows.split(raw)
    assert set(parts) == set(VARIABLES)
    assert (parts["wick"] >= 0).all()
    daily = parts["gap"] + parts["intra"]
    assert 0.005 < daily.std() < 0.05  # daily log returns, not z-scores


# ---------------------------------------------------------------------- training


def test_training_is_deterministic_and_records_losses(market, gan):
    again = train_timegan(market, CFG, train_start=TRAIN[0], train_end=TRAIN[1], seed=0)
    assert again.losses == gan.losses
    assert {k: len(v) for k, v in gan.losses.items()} == {
        "reconstruction": 4, "supervised": 4, "generator": 4, "discriminator": 4,
    }  # fmt: skip
    codes = np.zeros(5, dtype=int)
    np.testing.assert_array_equal(
        generate_windows(gan, codes, seed=3), generate_windows(again, codes, seed=3)
    )
    other = train_timegan(market, CFG, train_start=TRAIN[0], train_end=TRAIN[1], seed=1)
    assert other.losses != gan.losses


def test_saved_generator_reloads_identically(market, gan, tmp_path):
    save_gan(gan, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"model.pt", "scaler.npz", "meta.json"}
    loaded = load_gan(tmp_path, market)
    codes = np.array([0, 1, 5, 5])
    np.testing.assert_allclose(
        generate_windows(loaded, codes, seed=9), generate_windows(gan, codes, seed=9), rtol=1e-6
    )


# -------------------------------------------------------------------- generation


def test_generated_windows_are_conditioned_and_calibrated(gan):
    stress = CONDITIONS.index("market_stress")
    bull = CONDITIONS.index("bull")
    raw = generate_windows(gan, np.r_[np.full(200, bull), np.full(200, stress)], seed=1)
    n = len(gan.windows.symbols)
    assert raw.shape == (400, CFG.sequence_length, 4 * n)
    assert np.isfinite(raw).all()
    assert not np.allclose(raw[:200].mean(axis=0), raw[200:].mean(axis=0))  # regime matters

    # Calibration: per-regime spread of the generated variables matches the training
    # windows of that regime (or all training windows where the regime is too rare).
    train_std = {}
    for code in (bull, stress):
        members = gan.windows.windows[gan.windows.conditions == code]
        source = members if len(members) >= CFG.min_regime_windows else gan.windows.windows
        train_std[code] = source.reshape(-1, gan.windows.dim).std(axis=0)
    for code, block in ((bull, raw[:200]), (stress, raw[200:])):
        standardised = (block - gan.windows.mean) / gan.windows.std
        generated = standardised.reshape(-1, gan.windows.dim).std(axis=0)
        returns_columns = slice(0, 2 * n)  # gap and intra are unaffected by the wick floor
        np.testing.assert_allclose(
            generated[returns_columns], train_std[code][returns_columns], rtol=0.25
        )


def test_derived_ohlcv_is_always_valid(gan, market):
    raw = generate_windows(gan, np.zeros(20, dtype=int), seed=2)
    sessions = pd.bdate_range("2018-01-01", periods=20 * CFG.sequence_length)
    bars = windows_to_bars(gan, raw, sessions)
    assert len(bars) == len(sessions) * len(gan.windows.symbols)
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars[["open", "high", "low", "close"]] > 0).all().all()
    assert (bars["volume"] >= 1).all() and bars["volume"].dtype == "int64"
    # Prices continue from the last real close and follow the generated returns.
    first = bars[bars["symbol"] == gan.windows.symbols[0]].reset_index(drop=True)
    returns = window_returns(gan, raw).reshape(-1, len(gan.windows.symbols))[:, 0]
    assert first["close"].iloc[0] == pytest.approx(gan.windows.last_close[0] * np.exp(returns[0]))
    np.testing.assert_allclose(np.log(first["close"]).diff().iloc[1:], returns[1:], rtol=1e-9)


def test_synthetic_market_is_a_labelled_full_market(gan, market):
    synthetic = synthetic_market(gan, market, 150, seed=4)
    assert synthetic.is_synthetic is True and synthetic.provider == "timegan"
    assert synthetic.symbols == market.symbols  # same columns, so the RL agent can use it
    assert len(synthetic.sessions) == 150
    excluded = gan.windows.excluded
    assert synthetic.close[excluded].isna().all().all()  # unmodelled assets stay empty
    modelled = gan.windows.symbols
    assert synthetic.close[modelled].notna().all().all()
    assert (synthetic.high[modelled] >= synthetic.close[modelled]).all().all()
    assert set(synthetic.features.columns) == set(market.features.columns)
    assert synthetic.regimes is not None and len(synthetic.regimes) == 150
    again = synthetic_market(gan, market, 150, seed=4)
    pd.testing.assert_frame_equal(again.close, synthetic.close)
    assert not synthetic_market(gan, market, 150, seed=5).close.equals(synthetic.close)


def test_regime_chain_follows_training_dynamics(gan):
    codes = np.array([0, 0, 0, 1, 1, 0, 0, 5, 5, 5])
    matrix = regime_transition_matrix(codes, step=1)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
    assert matrix[0, 0] == pytest.approx(3 / 5) and matrix[1, 0] == pytest.approx(0.5)
    assert matrix[3, 3] == 1.0  # never observed: stays put
    chain = sample_regime_chain(gan, 50, np.random.default_rng(0))
    observed = set(gan.windows.conditions) - {CONDITIONS.index("undefined")}
    assert set(chain) <= observed  # only regimes that occurred in training


def test_scenario_features_and_stress_scenarios(gan, market):
    features, table = scenario_features(gan, market.symbols, n_per_regime=60, seed=0)
    assert features.names == ["scenario_volatility", "scenario_cvar", "scenario_mean"]
    assert set(features.by_regime) == {c for c in CONDITIONS if c != "undefined"}
    block = features.by_regime["market_stress"]
    assert block.shape == (len(market.symbols), 3) and np.isfinite(block).all()
    excluded = [market.symbols.index(s) for s in gan.windows.excluded]
    assert (block[excluded] == 0).all()  # no scenario information for unmodelled assets
    assert (table["scenario_volatility"] > 0).all()
    assert (table["scenario_cvar"] <= table["scenario_mean"]).all()

    scenarios = stress_scenarios(gan, 25, seed=1)
    assert scenarios.shape == (25, len(gan.windows.symbols))
    weights = pd.Series(1 / len(gan.windows.symbols), index=gan.windows.symbols)
    losses = stress_test(weights, scenarios)
    assert len(losses) == 25 and np.isfinite(losses).all()


# -------------------------------------------------------------------- evaluation


@pytest.fixture(scope="module")
def raw_sets(market, gan):
    train = gan.windows
    heldout = _windows(market, VALIDATION, scaler=train)
    return (
        train.inverse(train.windows.astype("float64")),
        heldout.inverse(heldout.windows.astype("float64")),
    )


def test_identical_samples_score_as_indistinguishable(gan, raw_sets):
    _, heldout = raw_sets
    same = compare(gan.windows, heldout, heldout.copy())
    assert same["return_distribution"]["ks"] == 0.0
    assert same["cross_correlation"]["mean_abs_error"] == pytest.approx(0.0, abs=1e-12)
    assert same["drawdown_distribution"]["ks"] == 0.0
    assert same["autocorrelation"]["returns_error"] == pytest.approx(0.0, abs=1e-12)


def test_evaluation_separates_good_from_bad_scenarios(gan, raw_sets):
    train, heldout = raw_sets
    rng = np.random.default_rng(0)
    half = len(heldout) // 2
    real_like = evaluate(
        gan.windows, train_raw=train, heldout_raw=heldout[:half], synthetic_raw=heldout[half:],
        gates=QualityGates(),
    )  # fmt: skip
    noise = rng.normal(0, 0.2, heldout[half:].shape)  # 20% daily moves: absurd
    absurd = evaluate(
        gan.windows, train_raw=train, heldout_raw=heldout[:half], synthetic_raw=noise,
        gates=QualityGates(),
    )  # fmt: skip
    assert set(real_like["gates"]) == {
        "return_ks", "correlation_error", "discriminative_score", "memorization_ratio",
        "copied_fraction",
    }  # fmt: skip
    assert real_like["synthetic_vs_heldout_real"]["return_distribution"]["ks"] < 0.1
    assert absurd["synthetic_vs_heldout_real"]["return_distribution"]["ks"] > 0.4
    assert absurd["discriminative"]["score"] > 0.4  # trivially separable
    assert real_like["discriminative"]["score"] < absurd["discriminative"]["score"]
    assert absurd["passes"] is False
    assert "real_baseline_train_vs_heldout" in real_like


def test_memorisation_check_catches_a_generator_that_replays_training_data(gan, raw_sets):
    train, heldout = raw_sets
    replay = train[::3] + np.random.default_rng(0).normal(0, 1e-7, train[::3].shape)
    copied = memorization_check(gan.windows, train, heldout, replay)
    assert copied["copied_fraction"] == 1.0
    assert copied["ratio"] < 0.01

    report = evaluate(
        gan.windows, train_raw=train, heldout_raw=heldout, synthetic_raw=replay,
        gates=QualityGates(),
    )  # fmt: skip
    assert report["gates"]["memorization_ratio"]["passed"] is False
    assert report["gates"]["copied_fraction"]["passed"] is False
    assert report["passes"] is False

    fresh = memorization_check(
        gan.windows, train, heldout[: len(heldout) // 2], heldout[len(heldout) // 2 :]
    )
    assert fresh["copied_fraction"] == 0.0
    assert fresh["ratio"] > 0.6  # unseen real data is not "close" to the training set
