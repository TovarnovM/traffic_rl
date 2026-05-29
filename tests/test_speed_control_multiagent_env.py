import subprocess
import sys

import numpy as np
import pytest


def test_speed_control_env_is_lazily_imported_with_rl_package():
    code = """
import sys
import snfs_traffic
import snfs_traffic.rl
assert 'gymnasium' not in sys.modules
assert 'snfs_traffic.rl.speed_control_multiagent_env' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_speed_control_env_import_path_when_gymnasium_available():
    pytest.importorskip("gymnasium")
    from snfs_traffic.rl.speed_control_multiagent_env import SnfsTrafficSpeedControlMultiAgentEnv

    assert SnfsTrafficSpeedControlMultiAgentEnv.__name__ == "SnfsTrafficSpeedControlMultiAgentEnv"


@pytest.fixture
def speed_env_cls():
    pytest.importorskip("gymnasium")
    from snfs_traffic.rl.speed_control_multiagent_env import SnfsTrafficSpeedControlMultiAgentEnv

    return SnfsTrafficSpeedControlMultiAgentEnv


def _assert_same_mapping_arrays(left, right):
    assert set(left) == set(right)
    for key in left:
        if isinstance(left[key], dict):
            _assert_same_mapping_arrays(left[key], right[key])
        elif isinstance(left[key], np.ndarray):
            np.testing.assert_array_equal(left[key], right[key])
        else:
            assert left[key] == right[key]


def _state_arrays(env):
    state = env._sim.state
    return tuple(arr.copy() for arr in (state.lane, state.pos, state.vel, state.alive, state.controlled))


def test_speed_control_env_constructs_with_multidiscrete_action_space(speed_env_cls):
    env = speed_env_cls(backend="reference", num_controlled=3, density=0.3)

    assert env.single_agent_action_space.nvec.tolist() == [3, 3]
    assert env.action_space.nvec.tolist() == [3, 3]
    assert env.agents == []
    assert env.backend_name == "reference"


def test_speed_control_env_reset_and_step(speed_env_cls):
    env = speed_env_cls(backend="reference", num_controlled=3, density=0.3)
    obs, infos = env.reset(seed=123)
    prev_agents = list(env.agents)

    assert set(obs) == set(prev_agents)
    assert set(infos) == set(prev_agents)
    for agent_obs in obs.values():
        assert env.single_agent_observation_space.contains(agent_obs)
        assert agent_obs["action_mask"].shape == (2, 3)

    actions = {agent_id: np.asarray([0, 1], dtype=np.int64) for agent_id in env.agents}
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    assert set(rewards) == set(prev_agents)
    assert set(infos) == set(prev_agents)
    assert set(terminateds) == set(prev_agents) | {"__all__"}
    assert set(truncateds) == set(prev_agents) | {"__all__"}
    for info in infos.values():
        assert info["controlled_speed_delta"] == 0
        assert "desired_velocity" in info
        assert "applied_velocity" in info
        assert "speed_clipped_by_safety" in info
    for agent_obs in obs.values():
        assert env.single_agent_observation_space.contains(agent_obs)


@pytest.mark.parametrize(
    "bad_action",
    [
        np.asarray([3, 1], dtype=np.int64),
        np.asarray([0, 3], dtype=np.int64),
        0,
        np.asarray([0], dtype=np.int64),
        np.asarray([0, 1, 2], dtype=np.int64),
    ],
)
def test_speed_control_env_rejects_invalid_actions(speed_env_cls, bad_action):
    env = speed_env_cls(backend="reference", num_controlled=2, density=0.3)
    env.reset(seed=123)
    actions = {agent: np.asarray([0, 1], dtype=np.int64) for agent in env.agents}
    actions[env.agents[0]] = bad_action

    with pytest.raises(ValueError, match="invalid action"):
        env.step(actions)


def test_speed_control_env_reproducibility(speed_env_cls):
    kwargs = dict(backend="reference", num_controlled=3, density=0.3)
    env1 = speed_env_cls(**kwargs)
    env2 = speed_env_cls(**kwargs)

    obs1, infos1 = env1.reset(seed=123)
    obs2, infos2 = env2.reset(seed=123)
    _assert_same_mapping_arrays(obs1, obs2)
    _assert_same_mapping_arrays(infos1, infos2)

    sequence = [
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([0, 2], dtype=np.int64),
        np.asarray([0, 0], dtype=np.int64),
    ]
    for action in sequence:
        assert env1.agents == env2.agents
        actions1 = {agent: action.copy() for agent in env1.agents}
        actions2 = {agent: action.copy() for agent in env2.agents}
        result1 = env1.step(actions1)
        result2 = env2.step(actions2)
        obs1, rewards1, terminateds1, truncateds1, infos1 = result1
        obs2, rewards2, terminateds2, truncateds2, infos2 = result2

        assert env1.agents == env2.agents
        assert rewards1 == rewards2
        assert terminateds1 == terminateds2
        assert truncateds1 == truncateds2
        _assert_same_mapping_arrays(obs1, obs2)
        _assert_same_mapping_arrays(infos1, infos2)
        for left, right in zip(_state_arrays(env1), _state_arrays(env2)):
            np.testing.assert_array_equal(left, right)
        if terminateds1["__all__"] or truncateds1["__all__"]:
            break


def test_old_env_action_spaces_remain_lateral_only():
    pytest.importorskip("gymnasium")
    from snfs_traffic.rl.env import SnfsTrafficEnv
    from snfs_traffic.rl.multiagent_env import SnfsTrafficMultiAgentEnv

    single = SnfsTrafficEnv(backend="reference")
    multi = SnfsTrafficMultiAgentEnv(backend="reference", num_controlled=2, density=0.3)

    assert single.action_space.n == 3
    assert multi.single_agent_action_space.n == 3
    assert multi.action_space.n == 3
