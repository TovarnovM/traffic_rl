from __future__ import annotations

import math

import numpy as np
import pytest


gymnasium = pytest.importorskip("gymnasium")

from snfs_traffic.rl.centralized_pv_env import (  # noqa: E402
    CentralizedPvEnv,
    PvSpeedRewardConfig,
    compute_pv_speed_reward,
)
from snfs_traffic.observations import PvGraphConfig  # noqa: E402
from snfs_traffic.scenarios import AV_VEH_TYPE  # noqa: E402


def _env(**overrides):
    config = {
        "num_lanes": 4,
        "road_length": 100,
        "density": 0.20,
        "av_fraction": 0.95,
        "front_distance": 10,
        "back_distance": 5,
        "sensor_distance": 20,
        "cooldown_steps": 5,
        "warmup_steps": 0,
        "episode_steps": 3,
        "backend": "reference",
        "validate": True,
        "seed": 123,
    }
    config.update(overrides)
    return CentralizedPvEnv(config)


def test_reward_is_pv_speed_with_only_two_count_regularizers():
    reward, components = compute_pv_speed_reward(
        pv_speed=5,
        pv_vmax=6,
        applied_count=3,
        rejected_count=2,
        config=PvSpeedRewardConfig(
            applied_penalty=0.001,
            rejected_penalty=0.002,
        ),
    )

    expected = 5 / 6 - 3 * 0.001 - 2 * 0.002
    assert reward == pytest.approx(expected)
    assert components == {
        "pv_speed_reward": pytest.approx(5 / 6),
        "applied_regularizer": pytest.approx(-0.003),
        "rejected_regularizer": pytest.approx(-0.004),
        "applied_count": 3.0,
        "rejected_count": 2.0,
        "total": pytest.approx(expected),
    }


def test_graph_cooldown_default_is_one_step_and_zero_remains_valid():
    assert PvGraphConfig().cooldown_steps == 1
    assert PvGraphConfig(cooldown_steps=0).cooldown_steps == 0


def test_reset_returns_fixed_padded_graph_and_never_sets_controlled_flag():
    env = _env()
    observation, info = env.reset(seed=77)

    assert env.observation_space.contains(observation)
    assert env.action_space.shape == (env.graph_config.max_nodes(env.params),)
    assert info["priority_vehicle_id"] == env.priority_vehicle_id
    state = env._sim.state
    assert not np.any(state.controlled)
    active_ids = env._current_graph.vehicle_id[env._current_graph.node_mask.astype(bool)]
    id_to_index = {int(vehicle_id): idx for idx, vehicle_id in enumerate(state.vehicle_id)}
    assert all(
        int(state.veh_type[id_to_index[int(vehicle_id)]]) == AV_VEH_TYPE
        for vehicle_id in active_ids
    )


def test_all_stay_step_is_valid_and_reward_matches_reported_pv_speed():
    env = _env()
    env.reset(seed=77)
    action = np.ones(env.action_space.shape, dtype=np.int64)

    observation, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(observation)
    assert not terminated
    assert not truncated
    assert math.isfinite(reward)
    assert info["applied_count"] == 0
    assert info["rejected_count"] == 0
    assert reward == pytest.approx(info["pv_speed"] / env.params.vmax_controlled)
    assert info["episode_mean_pv_speed"] == pytest.approx(info["pv_speed"])
    assert info["density"] == pytest.approx(0.20)
    assert info["av_fraction"] == pytest.approx(0.95)
    assert info["condition_key"] == "rho_0p200_av_0p950"


def test_episode_truncates_and_requires_reset():
    env = _env(episode_steps=1)
    env.reset(seed=77)
    action = np.ones(env.action_space.shape, dtype=np.int64)

    _observation, _reward, terminated, truncated, _info = env.step(action)

    assert not terminated
    assert truncated
    with pytest.raises(RuntimeError, match="call reset"):
        env.step(action)


def test_applied_lane_change_starts_vehicle_id_keyed_cooldown():
    env = _env(
        density=0.10,
        av_fraction=1.0,
        front_distance=15,
        back_distance=10,
        warmup_steps=2,
    )
    observation, _info = env.reset(seed=0)
    candidates = np.argwhere(
        (observation["node_mask"][:, None] > 0)
        & (observation["action_mask"] > 0)
        & (np.arange(3)[None, :] != 1)
    )
    assert candidates.size > 0
    slot, action_index = (int(value) for value in candidates[0])
    vehicle_id = int(env._current_graph.vehicle_id[slot])
    action = np.ones(env.action_space.shape, dtype=np.int64)
    action[slot] = action_index

    observation, _reward, _terminated, _truncated, info = env.step(action)

    assert info["applied_count"] == 1
    assert env._cooldown_by_vehicle_id[vehicle_id] == env.graph_config.cooldown_steps
    next_slot = int(np.flatnonzero(env._current_graph.vehicle_id == vehicle_id)[0])
    np.testing.assert_array_equal(observation["action_mask"][next_slot], [0, 1, 0])


def test_equal_seeds_and_actions_are_reproducible():
    first = _env()
    second = _env()
    first_obs, first_info = first.reset(seed=99)
    second_obs, second_info = second.reset(seed=99)
    for key in first_obs:
        np.testing.assert_array_equal(first_obs[key], second_obs[key])
    assert first_info == second_info

    action = np.ones(first.action_space.shape, dtype=np.int64)
    first_step = first.step(action)
    second_step = second.step(action)
    for key in first_step[0]:
        np.testing.assert_array_equal(first_step[0][key], second_step[0][key])
    assert first_step[1:] == second_step[1:]


def test_reset_grid_covers_every_pair_and_caches_warmup_once_per_density():
    first = _env(
        density_values=[0.10, 0.20],
        av_fraction_values=[0.50, 1.0],
        warmup_steps=2,
        warmup_cache=True,
    )
    second = _env(
        density_values=[0.10, 0.20],
        av_fraction_values=[0.50, 1.0],
        warmup_steps=2,
        warmup_cache=True,
    )

    def reset_sequence(env):
        infos = []
        for index in range(4):
            _observation, info = env.reset(seed=91 if index == 0 else None)
            infos.append(info)
        return infos

    first_infos = reset_sequence(first)
    second_infos = reset_sequence(second)
    first_pairs = [
        (info["density"], info["av_fraction"]) for info in first_infos
    ]
    second_pairs = [
        (info["density"], info["av_fraction"]) for info in second_infos
    ]

    assert set(first_pairs) == {
        (0.10, 0.50),
        (0.10, 1.0),
        (0.20, 0.50),
        (0.20, 1.0),
    }
    assert first_pairs == second_pairs
    assert sum(not info["warmup_cache_hit"] for info in first_infos) == 2
    assert sum(info["warmup_cache_hit"] for info in first_infos) == 2
    assert first_infos[-1]["warmup_cache_size"] == 2
