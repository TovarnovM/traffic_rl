import numpy as np

from snfs_traffic.core import SimulationParams, empty_state
from snfs_traffic.rl.rewards import RewardConfig, compute_controlled_reward


def _state_with_controlled_vel(vel: int):
    state = empty_state(1)
    state.controlled[0] = True
    state.vel[0] = vel
    return state


def test_reward_is_finite_and_components_include_total():
    params = SimulationParams(num_lanes=2, road_length=20)
    prev = _state_with_controlled_vel(0)
    nxt = _state_with_controlled_vel(3)

    reward, components = compute_controlled_reward(
        prev,
        nxt,
        controlled_vehicle_id=0,
        action={0: 0},
        action_applied=True,
        params=params,
        config=RewardConfig(),
    )

    assert np.isfinite(reward)
    assert set(components) == {
        "speed_reward",
        "lane_change_penalty",
        "blocked_action_penalty",
        "stopped_penalty",
        "total",
    }
    assert components["total"] == reward


def test_higher_velocity_gives_higher_speed_component():
    params = SimulationParams(num_lanes=2, road_length=20)
    prev = _state_with_controlled_vel(0)
    slow = _state_with_controlled_vel(1)
    fast = _state_with_controlled_vel(4)

    _, slow_components = compute_controlled_reward(
        prev,
        slow,
        controlled_vehicle_id=0,
        action={0: 0},
        action_applied=True,
        params=params,
        config=RewardConfig(),
    )
    _, fast_components = compute_controlled_reward(
        prev,
        fast,
        controlled_vehicle_id=0,
        action={0: 0},
        action_applied=True,
        params=params,
        config=RewardConfig(),
    )

    assert fast_components["speed_reward"] > slow_components["speed_reward"]


def test_lane_change_penalty_applied_when_controlled_vehicle_changed_lane():
    params = SimulationParams(num_lanes=2, road_length=20)
    prev = _state_with_controlled_vel(2)
    nxt = _state_with_controlled_vel(2)
    nxt.lane[0] = 1

    _, components = compute_controlled_reward(
        prev,
        nxt,
        controlled_vehicle_id=0,
        action={0: 1},
        action_applied=True,
        params=params,
        config=RewardConfig(lane_change_penalty=0.25),
    )

    assert components["lane_change_penalty"] == -0.25


def test_stopped_penalty_applied_when_configured():
    params = SimulationParams(num_lanes=2, road_length=20)
    prev = _state_with_controlled_vel(2)
    nxt = _state_with_controlled_vel(0)

    _, components = compute_controlled_reward(
        prev,
        nxt,
        controlled_vehicle_id=0,
        action={0: 0},
        action_applied=True,
        params=params,
        config=RewardConfig(stopped_penalty=0.5),
    )

    assert components["stopped_penalty"] == -0.5


def test_blocked_action_penalty_applied_when_requested_action_not_applied():
    params = SimulationParams(num_lanes=2, road_length=20)
    prev = _state_with_controlled_vel(2)
    nxt = _state_with_controlled_vel(2)

    _, components = compute_controlled_reward(
        prev,
        nxt,
        controlled_vehicle_id=0,
        action={0: 1},
        action_applied=False,
        params=params,
        config=RewardConfig(blocked_action_penalty=0.4),
    )

    assert components["blocked_action_penalty"] == -0.4


def test_reward_function_does_not_mutate_states():
    params = SimulationParams(num_lanes=2, road_length=20)
    prev = _state_with_controlled_vel(2)
    nxt = _state_with_controlled_vel(3)
    prev_copy = prev.copy()
    next_copy = nxt.copy()

    compute_controlled_reward(
        prev,
        nxt,
        controlled_vehicle_id=0,
        action={0: 0},
        action_applied=True,
        params=params,
        config=RewardConfig(),
    )

    for field in prev.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(prev, field), getattr(prev_copy, field))
        np.testing.assert_array_equal(getattr(nxt, field), getattr(next_copy, field))
