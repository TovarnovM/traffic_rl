import math
import subprocess
import sys

import numpy as np
import pytest


def test_base_and_rl_imports_do_not_import_gymnasium_or_env_modules():
    code = """
import sys
import snfs_traffic
import snfs_traffic.rl
assert 'gymnasium' not in sys.modules
assert 'snfs_traffic.rl.env' not in sys.modules
assert 'snfs_traffic.rl.multiagent_env' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


from snfs_traffic.rl.episode import EpisodeConfig


@pytest.fixture
def env_cls():
    pytest.importorskip("gymnasium")
    from snfs_traffic.rl.multiagent_env import SnfsTrafficMultiAgentEnv

    return SnfsTrafficMultiAgentEnv


@pytest.fixture
def gym_spaces():
    gymnasium = pytest.importorskip("gymnasium")
    return gymnasium.spaces


def _controlled_alive_ids(env):
    state = env._sim.state
    return tuple(int(v) for v in state.vehicle_id[state.alive & state.controlled])


def _assert_valid_observations(env, observations):
    assert set(observations) == set(env.agents)
    for agent_obs in observations.values():
        assert env.single_agent_observation_space.contains(agent_obs)


def test_multiagent_env_constructs_with_reference_backend(env_cls):
    env = env_cls(backend="reference", num_controlled=3)

    assert env.single_agent_action_space.n == 3
    assert env.action_space.n == 3
    assert env.backend_name == "reference"
    assert env.agents == []
    assert env._controlled_vehicle_ids == ()
    assert env._agent_to_vehicle_id == {}
    assert env._vehicle_id_to_agent == {}
    assert len(env.possible_agents) == env.params.num_lanes * env.params.road_length
    assert env.get_action_space("vehicle_0").contains(0)
    assert env.get_observation_space("vehicle_0") is env.single_agent_observation_space


@pytest.mark.parametrize("agent_id", ["vehicle_", "vehicle_x", "car_0", 0])
def test_multiagent_space_lookup_rejects_invalid_agent_id(env_cls, agent_id):
    env = env_cls(backend="reference")

    with pytest.raises(ValueError, match="vehicle_<vehicle_id>"):
        env.get_action_space(agent_id)


def test_reset_selects_requested_number_of_agents(env_cls):
    env = env_cls(backend="reference", num_controlled=3, density=0.3)

    obs, infos = env.reset(seed=123)
    controlled_ids = _controlled_alive_ids(env)

    assert len(env.agents) == 3
    assert set(obs) == set(env.agents)
    assert set(infos) == set(env.agents)
    assert len(controlled_ids) == 3
    assert controlled_ids == env._controlled_vehicle_ids
    assert tuple(env._agent_to_vehicle_id[agent] for agent in env.agents) == env._controlled_vehicle_ids
    for agent_id, agent_obs in obs.items():
        assert env.single_agent_observation_space.contains(agent_obs)
        vehicle_id = env._agent_to_vehicle_id[agent_id]
        assert agent_id == f"vehicle_{vehicle_id}"
        assert infos[agent_id]["agent_id"] == agent_id
        assert infos[agent_id]["controlled_vehicle_id"] == vehicle_id


def test_explicit_controlled_vehicle_ids_are_selected(env_cls):
    env = env_cls(backend="reference", density=0.3, controlled_vehicle_ids=[0, 1])

    obs, infos = env.reset(seed=123)

    assert env.agents == ["vehicle_0", "vehicle_1"]
    assert env._controlled_vehicle_ids == (0, 1)
    assert _controlled_alive_ids(env) == (0, 1)
    assert set(obs) == {"vehicle_0", "vehicle_1"}
    assert {info["controlled_vehicle_id"] for info in infos.values()} == {0, 1}


def test_one_valid_simultaneous_step(env_cls):
    env = env_cls(backend="reference", num_controlled=3, density=0.3)
    env.reset(seed=123)
    prev_agents = list(env.agents)

    obs, rewards, terminateds, truncateds, infos = env.step({agent: 0 for agent in env.agents})

    assert set(rewards) == set(prev_agents)
    assert set(infos) == set(prev_agents)
    assert set(terminateds) == set(prev_agents) | {"__all__"}
    assert set(truncateds) == set(prev_agents) | {"__all__"}
    assert all(math.isfinite(reward) for reward in rewards.values())
    for info in infos.values():
        assert "reward_components" in info
        assert "total" in info["reward_components"]
    _assert_valid_observations(env, obs)
    assert env._step_index == 1


def test_missing_action_raises_clear_value_error(env_cls):
    env = env_cls(backend="reference", num_controlled=2, density=0.3)
    env.reset(seed=123)
    actions = {agent: 0 for agent in env.agents[:-1]}

    with pytest.raises(ValueError, match="missing actions|active agents"):
        env.step(actions)


def test_extra_action_raises_clear_value_error(env_cls):
    env = env_cls(backend="reference", num_controlled=2, density=0.3)
    env.reset(seed=123)
    actions = {agent: 0 for agent in env.agents}
    actions["vehicle_999999"] = 0

    with pytest.raises(ValueError, match="unknown|inactive|extra"):
        env.step(actions)


def test_invalid_action_raises_clear_value_error(env_cls):
    env = env_cls(backend="reference", num_controlled=2, density=0.3)
    env.reset(seed=123)
    actions = {agent: 0 for agent in env.agents}
    actions[env.agents[0]] = 3

    with pytest.raises(ValueError, match="invalid action"):
        env.step(actions)


def test_reproducibility_with_same_seed_and_actions(env_cls):
    kwargs = dict(backend="reference", num_controlled=3, density=0.3)
    env1 = env_cls(**kwargs)
    env2 = env_cls(**kwargs)

    obs1, infos1 = env1.reset(seed=123)
    obs2, infos2 = env2.reset(seed=123)
    _assert_same_multiagent_result(env1, env2, obs1, obs2, infos1, infos2)

    for _ in range(4):
        actions1 = {agent: 0 for agent in env1.agents}
        actions2 = {agent: 0 for agent in env2.agents}
        assert actions1 == actions2
        result1 = env1.step(actions1)
        result2 = env2.step(actions2)
        obs1, rewards1, terminateds1, truncateds1, infos1 = result1
        obs2, rewards2, terminateds2, truncateds2, infos2 = result2
        assert env1.agents == env2.agents
        assert rewards1 == rewards2
        assert terminateds1 == terminateds2
        assert truncateds1 == truncateds2
        _assert_same_multiagent_result(env1, env2, obs1, obs2, infos1, infos2)
        if terminateds1["__all__"] or truncateds1["__all__"]:
            break


def _assert_same_multiagent_result(env1, env2, obs1, obs2, infos1, infos2):
    assert env1.agents == env2.agents
    assert set(obs1) == set(obs2)
    assert set(infos1) == set(infos2)
    for agent in obs1:
        np.testing.assert_array_equal(obs1[agent]["obs"], obs2[agent]["obs"])
        np.testing.assert_array_equal(obs1[agent]["action_mask"], obs2[agent]["action_mask"])
    for agent in infos1:
        assert infos1[agent]["controlled_vehicle_id"] == infos2[agent]["controlled_vehicle_id"]


def test_max_steps_truncates_globally_and_requires_reset(env_cls):
    env = env_cls(
        backend="reference",
        num_controlled=2,
        density=0.3,
        episode_config=EpisodeConfig(max_steps=1),
    )
    env.reset(seed=123)

    obs, rewards, terminateds, truncateds, infos = env.step({agent: 0 for agent in env.agents})

    assert truncateds["__all__"] is True
    assert terminateds["__all__"] is False
    assert env.agents == []
    assert obs == {}
    with pytest.raises(RuntimeError, match="call reset"):
        env.step({})


def test_custom_reward_hook_is_used(env_cls):
    class ConstantMultiRewardEnv(env_cls):
        def _compute_agent_reward(
            self,
            prev_state,
            next_state,
            *,
            agent_id,
            controlled_vehicle_id,
            action,
            action_applied,
        ):
            return 42.0, {"custom_reward": 42.0, "total": 42.0}

    env = ConstantMultiRewardEnv(backend="reference", num_controlled=2, density=0.3)
    env.reset(seed=123)
    prev_agents = list(env.agents)

    _, rewards, _, _, infos = env.step({agent: 0 for agent in env.agents})

    assert set(rewards) == set(prev_agents)
    assert all(reward == 42.0 for reward in rewards.values())
    for info in infos.values():
        assert info["reward_components"]["custom_reward"] == 42.0
        assert info["reward_components"]["total"] == 42.0


def test_custom_controlled_selection_hook_is_used(env_cls):
    class MaxVehicleIdsEnv(env_cls):
        def _select_controlled(self, state):
            alive_ids = state.vehicle_id[state.alive]
            if alive_ids.size < 2:
                raise ValueError("expected at least two alive vehicles")
            selected = tuple(sorted(int(v) for v in alive_ids)[-2:])
            state = state.copy()
            state.controlled[:] = False
            for vehicle_id in selected:
                matches = np.flatnonzero(state.vehicle_id == vehicle_id)
                state.controlled[int(matches[0])] = True
            self._controlled_vehicle_ids = selected
            self._set_active_agents(selected)
            return state

    env = MaxVehicleIdsEnv(backend="reference", num_controlled=2, density=0.3)

    obs, infos = env.reset(seed=123)
    state = env._sim.state
    expected = tuple(sorted(int(v) for v in state.vehicle_id[state.alive])[-2:])

    assert tuple(env._agent_to_vehicle_id[agent] for agent in env.agents) == expected
    assert env.agents == [f"vehicle_{vehicle_id}" for vehicle_id in expected]
    assert _controlled_alive_ids(env) == expected
    assert int(np.sum(state.controlled & state.alive)) == 2
    _assert_valid_observations(env, obs)
    assert {info["controlled_vehicle_id"] for info in infos.values()} == set(expected)

    step_obs, rewards, terminateds, truncateds, step_infos = env.step({agent: 0 for agent in env.agents})
    assert set(rewards) == {f"vehicle_{vehicle_id}" for vehicle_id in expected}
    assert set(step_infos) == set(rewards)
    if not terminateds["__all__"] and not truncateds["__all__"]:
        _assert_valid_observations(env, step_obs)


def test_custom_observation_hook_is_used(env_cls, gym_spaces):
    class TinyObsMultiEnv(env_cls):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.single_agent_observation_space = gym_spaces.Dict(
                {
                    "obs": gym_spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                    "action_mask": gym_spaces.MultiBinary(3),
                }
            )
            self.observation_space = self.single_agent_observation_space

        def _build_observations(self):
            base = super()._build_observations()
            return {
                agent_id: {
                    "obs": np.asarray([0.25, -0.25], dtype=np.float32),
                    "action_mask": obs["action_mask"],
                }
                for agent_id, obs in base.items()
            }

    env = TinyObsMultiEnv(backend="reference", num_controlled=2, density=0.3)

    reset_obs, _ = env.reset(seed=123)
    step_obs, _, terminateds, truncateds, _ = env.step({agent: 0 for agent in env.agents})

    for observations in [reset_obs, step_obs]:
        for agent_obs in observations.values():
            assert agent_obs["obs"].shape == (2,)
            assert env.single_agent_observation_space.contains(agent_obs)
    assert set(reset_obs)
    if not terminateds["__all__"] and not truncateds["__all__"]:
        assert set(step_obs)
