"""RL environment, observation builder, projection, reward and training pipeline."""

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from ml.backtesting.costs import CostModel
from ml.backtesting.data import MarketData
from ml.backtesting.simulator import SimulationConfig, simulate
from ml.risk.limits import RiskLimits
from ml.rl.config import EnvConfig, PPOConfig, RewardConfig, RLConfig
from ml.rl.environments.portfolio_env import PortfolioEnv
from ml.rl.evaluation.strategy import risk_managed_policy, sb3_policy
from ml.rl.observation import ObservationBuilder
from ml.rl.projection import action_to_weights
from ml.rl.rewards.reward import compute_reward
from ml.rl.training.train import load_agent, train_agent

from .test_backtest_engine import SHIPPED

TRAIN = ("2018-01-01", "2019-12-31")
VALIDATION = ("2020-01-01", "2020-12-31")
TEST = ("2021-01-01", "2021-12-31")
FEATURES = ["ret_1d", "cum_ret_21d", "rsi_14", "hist_vol_20d"]
LOOSE = RiskLimits(
    max_weight=1.0, max_sector_weight=1.0, min_cash=0.0, max_drawdown=None, max_turnover=None
)


@pytest.fixture(scope="module")
def costs() -> CostModel:
    from ml.backtesting.costs import CostConfig

    return CostModel(CostConfig.model_validate(SHIPPED["costs"]))


@pytest.fixture(scope="module")
def builder(market) -> ObservationBuilder:
    return ObservationBuilder(market, FEATURES, ["market", "regime"]).fit(*TRAIN)


def _env(builder, costs, window=TRAIN, limits=LOOSE, **kwargs) -> PortfolioEnv:
    return PortfolioEnv(builder, costs, start=window[0], end=window[1], limits=limits, **kwargs)


# ------------------------------------------------------------------- observation


def test_observation_layout(builder, market):
    n = len(market.symbols)
    assert builder.spec == {
        "market": (0, n * len(FEATURES)),
        "regime": (n * len(FEATURES), n * len(FEATURES) + 9),
        "portfolio": (n * len(FEATURES) + 9, n * len(FEATURES) + 9 + 2 * n + 2),
    }
    row = builder.row_of("2019-06-03")
    weights = np.full(n, 0.1)
    observation = builder.observe(row, weights, -0.05)
    assert observation.shape == (builder.size,) and observation.dtype == np.float32
    assert np.isfinite(observation).all()
    lo, hi = builder.spec["portfolio"]
    portfolio = observation[lo:hi]
    np.testing.assert_allclose(portfolio[:n], 0.1, rtol=1e-6)
    assert portfolio[n] == pytest.approx(1 - 0.1 * n)  # cash
    assert portfolio[n + 1] == pytest.approx(-0.05)  # drawdown
    regime = observation[slice(*builder.spec["regime"])]
    assert regime[:6].sum() == 1.0  # exactly one regime is active


def test_scaler_uses_the_training_window_only(market, builder):
    """Statistics must not move when data after the training window changes."""
    tampered_features = market.features.copy()
    later = tampered_features.index.get_level_values("trade_date") > pd.Timestamp(TRAIN[1])
    tampered_features.loc[later, FEATURES] = tampered_features.loc[later, FEATURES] * 50 + 3
    tampered = MarketData(
        sessions=market.sessions, symbols=market.symbols, open=market.open, close=market.close,
        traded_value=market.traded_value, securities=market.securities,
        features=tampered_features, regimes=market.regimes,
    )  # fmt: skip
    other = ObservationBuilder(tampered, FEATURES, ["market", "regime"]).fit(*TRAIN)
    np.testing.assert_array_equal(other.mean, builder.mean)
    np.testing.assert_array_equal(other.std, builder.std)

    row = builder.row_of("2019-06-03")
    weights = np.zeros(len(market.symbols))
    np.testing.assert_array_equal(
        other.observe(row, weights, 0.0), builder.observe(row, weights, 0.0)
    )
    late = builder.row_of("2021-06-01")
    assert not np.array_equal(
        other.observe(late, weights, 0.0), builder.observe(late, weights, 0.0)
    )


def test_untradable_assets_are_masked(market, builder):
    late_listing = market.securities["listing_date"].dropna()
    symbol, listed_on = late_listing.index[0], late_listing.iloc[0]
    column = market.symbols.index(symbol)
    row = builder.row_of(market.sessions[market.sessions < listed_on][-1])
    assert not builder.tradable[row, column]
    block = builder.observe(row, np.zeros(len(market.symbols)), 0.0)[
        slice(*builder.spec["market"])
    ].reshape(len(market.symbols), len(FEATURES))
    assert (block[column] == 0).all()
    mask = builder.observe(row, np.zeros(len(market.symbols)), 0.0)[-len(market.symbols) :]
    assert mask[column] == 0.0 and mask.sum() == len(market.symbols) - 1


def test_builder_requires_fit_and_known_features(market):
    fresh = ObservationBuilder(market, FEATURES, ["market"])
    with pytest.raises(RuntimeError, match="fit"):
        fresh.observe(10, np.zeros(len(market.symbols)), 0.0)
    with pytest.raises(ValueError, match="not in the dataset"):
        ObservationBuilder(market, ["crystal_ball"], ["market"])
    with pytest.raises(ValueError, match="news"):
        ObservationBuilder(market, FEATURES, ["market", "news"])


# -------------------------------------------------------- projection and reward


def test_action_projection_is_a_masked_softmax():
    tradable = np.array([True, True, False])
    weights = action_to_weights(np.zeros(4), tradable, logit_scale=4.0)
    np.testing.assert_allclose(weights, [1 / 3, 1 / 3, 0.0])  # two assets + cash share equally
    tilted = action_to_weights(np.array([1.0, -1.0, 1.0, -1.0]), tradable, logit_scale=4.0)
    assert tilted[0] > 0.99 and tilted[2] == 0.0
    assert (weights >= 0).all() and weights.sum() <= 1
    all_cash = action_to_weights(np.array([-1.0, -1.0, -1.0, 1.0]), tradable, logit_scale=4.0)
    assert all_cash.sum() < 0.001
    nan_safe = action_to_weights(np.array([np.nan, 0.0, 0.0, 0.0]), tradable, logit_scale=4.0)
    assert np.isfinite(nan_safe).all()
    with pytest.raises(ValueError):
        action_to_weights(np.zeros(3), tradable, logit_scale=4.0)


def test_reward_formulations_hand_computed():
    args = {
        "log_return": 0.01,
        "volatility": 0.02,
        "drawdown": 0.10,
        "previous_drawdown": 0.04,
        "turnover": 0.5,
    }
    lambdas = {
        "lambda_volatility": 0.1,
        "lambda_drawdown": 0.2,
        "lambda_turnover": 0.01,
        "scale": 100.0,
    }
    plain = compute_reward(RewardConfig(kind="log_return", **lambdas), **args)
    assert plain == pytest.approx(1.0)
    adjusted = compute_reward(RewardConfig(kind="risk_adjusted", **lambdas), **args)
    assert adjusted == pytest.approx(100 * (0.01 - 0.002 - 0.02 - 0.005))
    change = compute_reward(RewardConfig(kind="drawdown_change", **lambdas), **args)
    assert change == pytest.approx(100 * (0.01 - 0.002 - 0.2 * 0.06 - 0.005))
    recovering = compute_reward(
        RewardConfig(kind="drawdown_change", **lambdas), **{**args, "drawdown": 0.02}
    )
    assert recovering == pytest.approx(100 * (0.01 - 0.002 - 0.005))  # no penalty for recovering


# -------------------------------------------------------------------- environment


def test_environment_follows_the_gymnasium_api(builder, costs):
    check_env(_env(builder, costs), skip_render_check=True)


def test_all_cash_policy_earns_exactly_nothing(builder, costs):
    env = _env(builder, costs, random_start=False)
    all_cash = np.append(-np.ones(env.n), 1.0)
    returns = env.run_policy(lambda _: all_cash)
    assert len(returns) > 400
    # The softmax leaves ~0.2% invested, so returns are tiny but not exactly zero.
    assert np.abs(returns).max() < 2e-4
    assert env.value == pytest.approx(1.0, abs=2e-3)


def test_episode_boundaries(builder, costs):
    cfg = EnvConfig(rebalance_every=5, episode_steps=10)
    env = _env(builder, costs, env=cfg)
    observation, info = env.reset(seed=1)
    assert env.lo <= info["row"] <= env.hi - 50
    steps, done = 0, False
    while not done:
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        steps += 1
        done = terminated or truncated
        assert np.isfinite(reward) and np.isfinite(observation).all()
    assert steps == 10
    assert info["row"] <= env.hi

    again, _ = env.reset(seed=1)
    first, _ = env.reset(seed=1)
    np.testing.assert_array_equal(again, first)  # seeded resets are reproducible


def test_risk_limits_apply_inside_the_environment(builder, costs):
    rules = RiskLimits(max_weight=0.25, min_cash=0.10, max_turnover=None, max_drawdown=None)
    env = _env(builder, costs, limits=rules, random_start=False)
    greedy = np.append(np.array([1.0] + [-1.0] * (env.n - 1)), -1.0)  # everything in asset 0
    env.reset(options={"start_row": env.lo})
    for _ in range(20):
        _, _, terminated, _, info = env.step(greedy)
        # Weights drift between rebalances, so allow for a week of price movement.
        assert env.weights[0] < 0.25 * 1.35
        assert info["invested"] < 0.35
        if terminated:
            break


def test_environment_agrees_with_the_backtester_for_a_fixed_policy(market, builder, costs):
    """Same policy, same decision dates: the fast environment must track the exact
    simulator closely (it differs only in its proportional cost approximation)."""
    cfg = EnvConfig(rebalance_every=5)
    env = _env(builder, costs, window=VALIDATION, random_start=False, env=cfg)
    rng = np.random.default_rng(0)
    actions = iter(rng.uniform(-1, 1, (200, env.n + 1)))
    used: list[np.ndarray] = []

    def policy(_observation):
        used.append(next(actions))
        return used[-1]

    env_returns = env.run_policy(policy)

    replay = iter(used)
    strategy = risk_managed_policy(
        lambda _obs: next(replay), builder, limits=LOOSE, sectors=None,
        logit_scale=cfg.logit_scale,
    )  # fmt: skip
    sessions = market.sessions_between(*VALIDATION)
    result = simulate(
        market, strategy, start=sessions[0], end=sessions[-1],
        decision_dates=sessions[::5][: len(used)], cost_model=costs,
        config=SimulationConfig(initial_capital=1e6, participation_limit=None),
    )  # fmt: skip
    simulated = result.returns.loc[env_returns.index]

    assert len(used) > 40
    env_total = np.prod(1 + env_returns) - 1
    sim_total = np.prod(1 + simulated) - 1
    assert abs(env_total - sim_total) < 0.001
    assert np.corrcoef(env_returns, simulated)[0, 1] > 0.999
    assert np.abs(env_returns - simulated).max() < 0.0002


# ----------------------------------------------------------------------- training


@pytest.fixture(scope="module")
def tiny_config() -> RLConfig:
    return RLConfig(
        features=FEATURES,
        groups=["market", "regime"],
        env=EnvConfig(rebalance_every=5, episode_steps=20),
        ppo=PPOConfig(total_timesteps=1024, n_steps=256, batch_size=64, n_epochs=2, net_arch=[32]),
        eval_every=512,
        limits=RiskLimits(max_weight=0.4, max_turnover=None),
        torch_threads=1,
    )


@pytest.fixture(scope="module")
def agent(market, costs, tiny_config, tmp_path_factory):
    return train_agent(
        market, tiny_config, costs,
        train_start=TRAIN[0], train_end=TRAIN[1],
        validation_start=VALIDATION[0], validation_end=VALIDATION[1],
        seed=0, out_dir=tmp_path_factory.mktemp("agent"),
    )  # fmt: skip


def test_training_produces_a_validated_checkpoint(agent):
    assert [point["step"] for point in agent.validation_curve][:2] == [512, 1024]
    assert agent.best_step in {point["step"] for point in agent.validation_curve}
    best = max(p["sharpe"] for p in agent.validation_curve if p["sharpe"] is not None)
    assert agent.best_validation_sharpe == pytest.approx(best)
    assert agent.train_seconds > 0
    assert {p.name for p in agent.path.iterdir()} == {"model.zip", "observation.json"}


def test_training_is_reproducible_for_a_seed(market, costs, tiny_config, agent):
    again = train_agent(
        market, tiny_config, costs,
        train_start=TRAIN[0], train_end=TRAIN[1],
        validation_start=VALIDATION[0], validation_end=VALIDATION[1], seed=0,
    )  # fmt: skip
    assert again.validation_curve == agent.validation_curve
    other = train_agent(
        market, tiny_config, costs,
        train_start=TRAIN[0], train_end=TRAIN[1],
        validation_start=VALIDATION[0], validation_end=VALIDATION[1], seed=1,
    )  # fmt: skip
    assert other.validation_curve != agent.validation_curve


def test_saved_agent_reloads_to_the_same_policy(market, tiny_config, agent):
    model, builder = load_agent(agent.path, market, tiny_config)
    np.testing.assert_array_equal(builder.mean, agent.builder.mean)
    row = builder.row_of("2021-03-01")
    observation = builder.observe(row, np.zeros(len(market.symbols)), 0.0)
    np.testing.assert_array_equal(
        sb3_policy(model)(observation), sb3_policy(agent.model)(observation)
    )


def test_trained_policy_runs_through_the_backtester_under_risk_control(
    market, costs, tiny_config, agent
):
    strategy = risk_managed_policy(
        sb3_policy(agent.model), agent.builder, limits=tiny_config.limits,
        sectors=market.securities["sector"], logit_scale=tiny_config.env.logit_scale,
    )  # fmt: skip
    sessions = market.sessions_between(*TEST)
    result = simulate(
        market, strategy, start=sessions[0], end=sessions[-1],
        decision_dates=sessions[::5], cost_model=costs,
    )  # fmt: skip
    assets = result.weights.drop(columns="CASH")
    assert len(result.trades) > 0
    assert assets.max().max() < 0.4 * 1.4  # the cap, plus drift between rebalances
    assert result.equity["cash"].min() >= -1e-6
    log = strategy.risk_log()
    assert len(log) == len(sessions[::5]) - (1 if sessions[::5][-1] == sessions[-1] else 0)
    assert not log["rejected"].any()
    # Decisions on the first day after each rebalance respect the cap exactly.
    first_day = result.weights.loc[sessions[1]].drop("CASH")
    assert first_day.max() <= 0.4 * 1.1
