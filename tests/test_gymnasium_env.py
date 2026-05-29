import math

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")

from snfs_traffic.rl.episode import EpisodeConfig
from snfs_traffic.rl.env import SnfsTrafficEnv


def test_env_constructs_with_reference_backend():
    env = SnfsTrafficEnv(backend="reference")
    assert env.action_space.n == 3
    assert env.backend_name == "reference"


def test_reset_returns_obs_info_and_observation_is_contained():
    env = SnfsTrafficEnv(backend="reference")
    obs, info = env.reset(seed=123)

    assert isinstance(obs, dict)
    assert env.observation_space.contains(obs)
    assert info["step_index"] == 0
    assert "controlled_vehicle_id" in info
    assert info["backend_name"] == "reference"


def test_action_space_sample_is_contained():
    env = SnfsTrafficEnv(backend="reference")
    assert env.action_space.contains(env.action_space.sample())


def test_one_valid_step_returns_gymnasium_tuple_and_info_schema():
    env = SnfsTrafficEnv(backend="reference")
    env.reset(seed=123)

    obs, reward, terminated, truncated, info = env.step(0)

    assert env.observation_space.contains(obs)
    assert math.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["step_index"] == 1
    assert "reward_components" in info
    assert "total" in info["reward_components"]
    assert "mean_speed" in info
    assert "num_lane_changes" in info


def test_episode_truncates_at_configured_max_steps():
    env = SnfsTrafficEnv(backend="reference", episode_config=EpisodeConfig(max_steps=2))
    env.reset(seed=123)

    _, _, terminated1, truncated1, _ = env.step(0)
    _, _, terminated2, truncated2, info = env.step(0)

    assert terminated1 is False
    assert truncated1 is False
    assert terminated2 is False
    assert truncated2 is True
    assert info["truncated_reason"] == "max_steps"


def test_same_seed_and_actions_are_reproducible():
    actions = [0, 1, 0, 2]
    env1 = SnfsTrafficEnv(backend="reference", episode_config=EpisodeConfig(max_steps=10))
    env2 = SnfsTrafficEnv(backend="reference", episode_config=EpisodeConfig(max_steps=10))

    obs1, info1 = env1.reset(seed=321)
    obs2, info2 = env2.reset(seed=321)
    np.testing.assert_allclose(obs1["obs"], obs2["obs"])
    np.testing.assert_array_equal(obs1["action_mask"], obs2["action_mask"])
    assert info1 == info2

    for action in actions:
        step1 = env1.step(action)
        step2 = env2.step(action)
        np.testing.assert_allclose(step1[0]["obs"], step2[0]["obs"])
        np.testing.assert_array_equal(step1[0]["action_mask"], step2[0]["action_mask"])
        assert step1[1:] == step2[1:]


def test_different_seeds_can_produce_different_initial_observations_or_ids():
    env1 = SnfsTrafficEnv(backend="reference")
    env2 = SnfsTrafficEnv(backend="reference")

    obs1, info1 = env1.reset(seed=10)
    obs2, info2 = env2.reset(seed=11)

    assert (
        not np.array_equal(obs1["obs"], obs2["obs"])
        or not np.array_equal(obs1["action_mask"], obs2["action_mask"])
        or info1["controlled_vehicle_id"] != info2["controlled_vehicle_id"]
    )


def test_invalid_action_raises_clear_exception():
    env = SnfsTrafficEnv(backend="reference")
    env.reset(seed=123)

    with pytest.raises(ValueError, match="invalid action"):
        env.step(3)


def test_step_after_done_requires_reset():
    env = SnfsTrafficEnv(backend="reference", episode_config=EpisodeConfig(max_steps=1))
    env.reset(seed=123)
    env.step(0)

    with pytest.raises(RuntimeError, match="call reset"):
        env.step(0)


def test_base_package_import_still_works_without_importing_env_module():
    import snfs_traffic
    import snfs_traffic.rl as rl

    assert snfs_traffic.__version__
    assert rl.RewardConfig is not None
