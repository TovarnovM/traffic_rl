import numpy as np
import pytest

from snfs_traffic.core import (
    SimulationParams,
    TrafficState,
    empty_state,
    max_supported_velocity,
    validate_state,
)


def _valid_state() -> TrafficState:
    return TrafficState(
        vehicle_id=np.array([10, 11, 12], dtype=np.int32),
        lane=np.array([0, 1, 3], dtype=np.int16),
        pos=np.array([0, 10, 1499], dtype=np.int32),
        vel=np.array([0, 5, 6], dtype=np.int16),
        length=np.array([1, 1, 1], dtype=np.int16),
        veh_type=np.array([0, 0, 0], dtype=np.int16),
        behavior_id=np.array([0, 0, 0], dtype=np.int16),
        alive=np.array([True, True, True], dtype=np.bool_),
        last_lane_delta=np.array([0, -1, 1], dtype=np.int8),
        changed_lane=np.array([False, True, True], dtype=np.bool_),
        controlled=np.array([False, False, True], dtype=np.bool_),
    )


def test_simulation_params_defaults() -> None:
    params = SimulationParams(num_lanes=4, road_length=1500)
    assert params.vmax_default == 5
    assert params.vmax_controlled == 6
    assert params.G == 15
    assert params.q == 0.99
    assert params.r == 0.99
    assert params.S == 2
    assert params.P1 == 0.999
    assert params.P2 == 0.99
    assert params.P3 == 0.98
    assert params.P4 == 0.01
    assert params.p_lane_change == 0.5


@pytest.mark.parametrize(
    ("kwargs", "field_name"),
    [
        ({"num_lanes": 0, "road_length": 1500}, "num_lanes"),
        ({"num_lanes": 4, "road_length": 0}, "road_length"),
        ({"num_lanes": 4, "road_length": 1500, "vmax_default": -1}, "vmax_default"),
        ({"num_lanes": 4, "road_length": 1500, "S": 0}, "S"),
        ({"num_lanes": 4, "road_length": 1500, "q": -0.1}, "q"),
        ({"num_lanes": 4, "road_length": 1500, "q": 1.1}, "q"),
        ({"num_lanes": 4, "road_length": 1500, "P1": 1.5}, "P1"),
        ({"num_lanes": 4, "road_length": 1500, "p_lane_change": -0.1}, "p_lane_change"),
        ({"num_lanes": 4.0, "road_length": 1500}, "num_lanes"),
    ],
)
def test_simulation_params_invalid(kwargs: dict, field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        SimulationParams(**kwargs)




def test_simulation_params_probability_type_strictness() -> None:
    with pytest.raises(ValueError, match="q"):
        SimulationParams(num_lanes=4, road_length=1500, q=True)

    with pytest.raises(ValueError, match="q"):
        SimulationParams(num_lanes=4, road_length=1500, q="0.5")

    with pytest.raises(ValueError, match="q"):
        SimulationParams(num_lanes=4, road_length=1500, q="bad")

    with pytest.raises(ValueError, match="p_lane_change"):
        SimulationParams(num_lanes=4, road_length=1500, p_lane_change=None)


def test_max_supported_velocity() -> None:
    params = SimulationParams(num_lanes=4, road_length=1500, vmax_default=5, vmax_controlled=6)
    assert max_supported_velocity(params) == 6

    params = SimulationParams(num_lanes=4, road_length=1500, vmax_default=8, vmax_controlled=6)
    assert max_supported_velocity(params) == 8

def test_validate_state_valid() -> None:
    params = SimulationParams(num_lanes=4, road_length=1500)
    validate_state(_valid_state(), params)


def test_validate_state_mismatched_shapes() -> None:
    params = SimulationParams(num_lanes=4, road_length=1500)
    state = _valid_state()
    state.controlled = np.array([False, True], dtype=np.bool_)
    with pytest.raises(ValueError, match="controlled"):
        validate_state(state, params)


def test_validate_state_wrong_dtype() -> None:
    params = SimulationParams(num_lanes=4, road_length=1500)
    state = _valid_state()
    state.lane = state.lane.astype(np.int32)
    with pytest.raises(ValueError, match="lane"):
        validate_state(state, params)

    state = _valid_state()
    state.alive = state.alive.astype(np.int8)
    with pytest.raises(ValueError, match="alive"):
        validate_state(state, params)




def test_validate_state_non_contiguous_rejected() -> None:
    params = SimulationParams(num_lanes=4, road_length=1500)
    state = _valid_state()
    state.lane = np.array([0, 9, 1, 9, 3, 9], dtype=np.int16)[::2]
    with pytest.raises(ValueError, match="lane|C-contiguous"):
        validate_state(state, params)

def test_validate_state_non_1d_rejected() -> None:
    params = SimulationParams(num_lanes=4, road_length=1500)
    state = _valid_state()
    state.pos = state.pos.reshape(3, 1)
    with pytest.raises(ValueError, match="pos"):
        validate_state(state, params)


def test_validate_state_invalid_values_rejected() -> None:
    params = SimulationParams(num_lanes=4, road_length=1500)

    state = _valid_state()
    state.lane[0] = 4
    with pytest.raises(ValueError, match="lane"):
        validate_state(state, params)

    state = _valid_state()
    state.pos[0] = 1500
    with pytest.raises(ValueError, match="pos"):
        validate_state(state, params)

    state = _valid_state()
    state.vel[0] = -1
    with pytest.raises(ValueError, match="vel"):
        validate_state(state, params)

    state = _valid_state()
    state.vel[0] = 7
    with pytest.raises(ValueError, match="vel"):
        validate_state(state, params)

    state = _valid_state()
    state.length[0] = 0
    with pytest.raises(ValueError, match="length"):
        validate_state(state, params)

    state = _valid_state()
    state.vehicle_id[0] = state.vehicle_id[1]
    with pytest.raises(ValueError, match="vehicle_id"):
        validate_state(state, params)

    state = _valid_state()
    state.last_lane_delta[0] = 2
    with pytest.raises(ValueError, match="last_lane_delta"):
        validate_state(state, params)

    state = _valid_state()
    state.changed_lane[0] = True
    with pytest.raises(ValueError, match="changed_lane"):
        validate_state(state, params)


def test_validate_state_inactive_slot_out_of_range_values_allowed() -> None:
    params = SimulationParams(num_lanes=4, road_length=1500)
    state = _valid_state()
    state.alive[2] = False
    state.lane[2] = 99
    state.pos[2] = 9000
    state.vel[2] = 120
    validate_state(state, params)


def test_state_copy_is_deep() -> None:
    state = _valid_state()
    state_copy = state.copy()

    state_copy.lane[0] = 2

    assert state.lane[0] == 0
    assert not np.shares_memory(state.lane, state_copy.lane)
    assert not np.shares_memory(state.pos, state_copy.pos)


def test_empty_state_schema_and_validation() -> None:
    params = SimulationParams(num_lanes=4, road_length=1500)
    state = empty_state(3)

    assert isinstance(state, TrafficState)

    arrays = [
        state.vehicle_id,
        state.lane,
        state.pos,
        state.vel,
        state.length,
        state.veh_type,
        state.behavior_id,
        state.alive,
        state.last_lane_delta,
        state.changed_lane,
        state.controlled,
    ]
    assert all(arr.shape == (3,) for arr in arrays)

    assert state.vehicle_id.dtype == np.int32
    assert state.lane.dtype == np.int16
    assert state.pos.dtype == np.int32
    assert state.vel.dtype == np.int16
    assert state.length.dtype == np.int16
    assert state.veh_type.dtype == np.int16
    assert state.behavior_id.dtype == np.int16
    assert state.alive.dtype == np.bool_
    assert state.last_lane_delta.dtype == np.int8
    assert state.changed_lane.dtype == np.bool_
    assert state.controlled.dtype == np.bool_

    assert np.array_equal(state.vehicle_id, np.arange(3, dtype=np.int32))
    assert np.all(state.alive)
    assert np.all(state.length == 1)
    assert np.all(state.last_lane_delta == 0)
    assert np.all(state.changed_lane == 0)

    validate_state(state, params)


def test_validate_state_rejects_uncontrolled_velocity_above_vmax_default() -> None:
    params = SimulationParams(num_lanes=4, road_length=1500, vmax_default=4, vmax_controlled=6)
    state = _valid_state()
    state.controlled[:] = np.array([False, False, True], dtype=np.bool_)
    state.vel[:] = np.array([5, 0, 0], dtype=np.int16)

    with pytest.raises(ValueError, match="uncontrolled"):
        validate_state(state, params)


def test_validate_state_rejects_controlled_velocity_above_vmax_controlled_with_lower_controlled_cap() -> None:
    params = SimulationParams(num_lanes=4, road_length=1500, vmax_default=7, vmax_controlled=5)
    state = _valid_state()
    state.controlled[:] = np.array([True, False, False], dtype=np.bool_)
    state.vel[:] = np.array([6, 0, 0], dtype=np.int16)

    with pytest.raises(ValueError, match="controlled"):
        validate_state(state, params)


def test_simulation_params_reject_vmax_values_outside_velocity_dtype_range() -> None:
    vmax_too_large = int(np.iinfo(np.int16).max) + 1

    with pytest.raises(ValueError, match="vmax_default"):
        SimulationParams(num_lanes=2, road_length=50, vmax_default=vmax_too_large)

    with pytest.raises(ValueError, match="vmax_controlled"):
        SimulationParams(num_lanes=2, road_length=50, vmax_controlled=vmax_too_large)
