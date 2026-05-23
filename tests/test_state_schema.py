import numpy as np
import pytest

from snfs_traffic.core import SimulationParams, TrafficState, validate_state


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
