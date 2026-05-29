import numpy as np
import pytest

from snfs_traffic.control import (
    LANE_LEFT,
    LANE_RIGHT,
    LANE_STAY,
    ControlledVehicleAction,
    normalize_controlled_action,
    step_with_controlled_lateral_actions_reference,
)
from snfs_traffic.core import SimulationParams, empty_state
from snfs_traffic.topology import RingTopology


def _state(*, pos=(0,), vel=(0,), controlled=(True,), length=None):
    state = empty_state(len(pos))
    state.lane[:] = 0
    state.pos[:] = np.asarray(pos, dtype=state.pos.dtype)
    state.vel[:] = np.asarray(vel, dtype=state.vel.dtype)
    state.controlled[:] = np.asarray(controlled, dtype=state.controlled.dtype)
    if length is not None:
        state.length[:] = np.asarray(length, dtype=state.length.dtype)
    return state


def _params(**kwargs):
    defaults = dict(num_lanes=1, road_length=30, vmax_default=0, vmax_controlled=6, p_lane_change=0.0)
    defaults.update(kwargs)
    return SimulationParams(**defaults)


def _step(state, action, params=None):
    params = params or _params()
    topology = RingTopology(num_lanes=params.num_lanes, length=params.road_length)
    return step_with_controlled_lateral_actions_reference(
        state,
        params,
        topology,
        np.random.default_rng(123),
        {0: action},
    )


@pytest.mark.parametrize("lane_delta", [LANE_STAY, LANE_LEFT, LANE_RIGHT, np.int64(LANE_STAY)])
def test_normalize_controlled_action_accepts_legacy_lateral_actions(lane_delta):
    action = normalize_controlled_action(lane_delta)

    assert action.lane_delta == int(lane_delta)
    assert action.speed_delta is None


@pytest.mark.parametrize("speed_delta", [-1, 0, 1, np.int64(1)])
def test_normalize_controlled_action_accepts_explicit_speed_actions(speed_delta):
    action = normalize_controlled_action(ControlledVehicleAction(LANE_STAY, speed_delta))

    assert action.lane_delta == LANE_STAY
    assert action.speed_delta == int(speed_delta)


@pytest.mark.parametrize(
    "raw_action",
    [
        ControlledVehicleAction(2, None),
        ControlledVehicleAction(-2, None),
        ControlledVehicleAction(LANE_STAY, 2),
        ControlledVehicleAction(LANE_STAY, -2),
        ControlledVehicleAction(LANE_STAY, True),
        ControlledVehicleAction(True, None),
        2,
        True,
    ],
)
def test_normalize_controlled_action_rejects_invalid_values(raw_action):
    with pytest.raises(ValueError):
        normalize_controlled_action(raw_action)


def test_legacy_int_action_matches_structured_none_speed_action():
    params = _params(vmax_default=5, p_lane_change=0.0)
    topology = RingTopology(num_lanes=1, length=params.road_length)
    state = _state(pos=(0, 12), vel=(2, 0), controlled=(True, False))

    old, _ = step_with_controlled_lateral_actions_reference(
        state, params, topology, np.random.default_rng(7), {0: LANE_STAY}
    )
    new, result = step_with_controlled_lateral_actions_reference(
        state, params, topology, np.random.default_rng(7), {0: ControlledVehicleAction(LANE_STAY, None)}
    )

    np.testing.assert_array_equal(new.lane, old.lane)
    np.testing.assert_array_equal(new.pos, old.pos)
    np.testing.assert_array_equal(new.vel, old.vel)
    np.testing.assert_array_equal(new.alive, old.alive)
    np.testing.assert_array_equal(new.controlled, old.controlled)
    assert result.requested_speed_delta is None


def test_speed_delta_brake_reduces_velocity_by_one_with_enough_gap():
    state = _state(pos=(0, 20), vel=(3, 0), controlled=(True, False))

    out, result = _step(state, ControlledVehicleAction(LANE_STAY, -1))

    assert int(out.vel[0]) == 2
    assert int(result.desired_velocity[0]) == 2
    assert int(result.applied_velocity[0]) == 2
    assert bool(result.speed_clipped_by_safety[0]) is False


def test_speed_delta_keep_preserves_velocity_with_enough_gap():
    state = _state(pos=(0, 20), vel=(3, 0), controlled=(True, False))

    out, result = _step(state, ControlledVehicleAction(LANE_STAY, 0))

    assert int(out.vel[0]) == 3
    assert int(result.desired_velocity[0]) == 3
    assert bool(result.speed_clipped_by_safety[0]) is False


def test_speed_delta_accelerate_increases_velocity_with_enough_gap():
    state = _state(pos=(0, 20), vel=(3, 0), controlled=(True, False))

    out, result = _step(state, ControlledVehicleAction(LANE_STAY, 1))

    assert int(out.vel[0]) == 4
    assert int(result.desired_velocity[0]) == 4
    assert bool(result.speed_clipped_by_safety[0]) is False


def test_speed_delta_accelerate_is_clipped_by_vmax_controlled():
    params = _params(vmax_controlled=3)
    state = _state(pos=(0, 20), vel=(3, 0), controlled=(True, False))

    out, result = _step(state, ControlledVehicleAction(LANE_STAY, 1), params=params)

    assert int(out.vel[0]) == 3
    assert int(result.desired_velocity[0]) == 3


def test_speed_delta_accelerate_is_clipped_by_safety_gap_and_avoids_overlap():
    params = _params(road_length=10, vmax_default=0, vmax_controlled=6)
    state = _state(pos=(0, 2), vel=(1, 0), controlled=(True, False))

    out, result = _step(state, ControlledVehicleAction(LANE_STAY, 1), params=params)

    assert int(result.desired_velocity[0]) == 2
    assert int(out.vel[0]) <= 1
    assert int(out.pos[0]) != int(out.pos[1])
    assert bool(result.speed_clipped_by_safety[0]) is True
