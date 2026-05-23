import numpy as np
import pytest

from snfs_traffic.core import SimulationParams, validate_state
from snfs_traffic.core.types import (
    BEHAVIOR_ID_DTYPE,
    BOOL_DTYPE,
    LANE_DELTA_DTYPE,
    LANE_DTYPE,
    LENGTH_DTYPE,
    POSITION_DTYPE,
    VEHICLE_ID_DTYPE,
    VEHICLE_TYPE_DTYPE,
    VELOCITY_DTYPE,
)
from snfs_traffic.scenarios import (
    AV_BEHAVIOR_ID,
    AV_VEH_TYPE,
    BUS_BEHAVIOR_ID,
    BUS_VEH_TYPE,
    CONTROLLED_AV_BEHAVIOR_ID,
    HDV_BEHAVIOR_ID,
    HDV_VEH_TYPE,
    VehicleMix,
    make_uniform_random_state,
)


def _state_fields(state):
    return (
        "vehicle_id",
        "lane",
        "pos",
        "vel",
        "length",
        "veh_type",
        "behavior_id",
        "alive",
        "last_lane_delta",
        "changed_lane",
        "controlled",
    )


def test_basic_state_generation_and_schema():
    state = make_uniform_random_state(num_lanes=4, road_length=100, density=0.25, seed=123)

    assert state.n_vehicles == 100
    for field in _state_fields(state):
        assert getattr(state, field).shape == (100,)

    validate_state(state, SimulationParams(num_lanes=4, road_length=100))


def test_exact_dtypes_and_c_contiguous_and_ranges():
    num_lanes = 4
    road_length = 100
    state = make_uniform_random_state(num_lanes=num_lanes, road_length=road_length, density=0.25, seed=123)

    assert state.vehicle_id.dtype == np.dtype(VEHICLE_ID_DTYPE)
    assert state.lane.dtype == np.dtype(LANE_DTYPE)
    assert state.pos.dtype == np.dtype(POSITION_DTYPE)
    assert state.vel.dtype == np.dtype(VELOCITY_DTYPE)
    assert state.length.dtype == np.dtype(LENGTH_DTYPE)
    assert state.veh_type.dtype == np.dtype(VEHICLE_TYPE_DTYPE)
    assert state.behavior_id.dtype == np.dtype(BEHAVIOR_ID_DTYPE)
    assert state.alive.dtype == np.dtype(BOOL_DTYPE)
    assert state.last_lane_delta.dtype == np.dtype(LANE_DELTA_DTYPE)
    assert state.changed_lane.dtype == np.dtype(BOOL_DTYPE)
    assert state.controlled.dtype == np.dtype(BOOL_DTYPE)

    for field in _state_fields(state):
        assert getattr(state, field).flags.c_contiguous

    assert np.all((state.lane >= 0) & (state.lane < num_lanes))
    assert np.all((state.pos >= 0) & (state.pos < road_length))
    assert np.all(state.vel == 0)
    assert np.all(state.length >= 1)
    assert np.all(state.alive)
    assert np.all(state.last_lane_delta == 0)
    assert np.all(state.changed_lane == 0)


def test_no_duplicate_head_cells():
    road_length = 100
    state = make_uniform_random_state(num_lanes=4, road_length=road_length, density=0.25, seed=123)
    flat = state.lane.astype(np.int64) * road_length + state.pos.astype(np.int64)
    assert np.unique(flat).size == state.n_vehicles


def test_reproducibility_and_different_seed_changes_cells():
    config = dict(num_lanes=4, road_length=100, density=0.3)
    s1 = make_uniform_random_state(**config, seed=123)
    s2 = make_uniform_random_state(**config, seed=123)
    s3 = make_uniform_random_state(**config, seed=456)

    for field in _state_fields(s1):
        np.testing.assert_array_equal(getattr(s1, field), getattr(s2, field))

    flat1 = s1.lane.astype(np.int64) * config["road_length"] + s1.pos.astype(np.int64)
    flat3 = s3.lane.astype(np.int64) * config["road_length"] + s3.pos.astype(np.int64)
    assert not np.array_equal(flat1, flat3)


def test_density_zero_returns_empty_valid_state():
    state = make_uniform_random_state(num_lanes=4, road_length=100, density=0.0, seed=123)
    assert state.n_vehicles == 0
    validate_state(state, SimulationParams(num_lanes=4, road_length=100))


def test_density_one_fills_every_cell_exactly_once():
    num_lanes = 2
    road_length = 5
    state = make_uniform_random_state(num_lanes=num_lanes, road_length=road_length, density=1.0, seed=123)

    assert state.n_vehicles == 10
    flat = state.lane.astype(np.int64) * road_length + state.pos.astype(np.int64)
    assert np.unique(flat).size == state.n_vehicles
    assert set(flat.tolist()) == set(range(num_lanes * road_length))
    validate_state(state, SimulationParams(num_lanes=num_lanes, road_length=road_length))


@pytest.mark.parametrize(
    ("kwargs", "field_name"),
    [
        ({"num_lanes": 0, "road_length": 100, "density": 0.5, "seed": 1}, "num_lanes"),
        ({"num_lanes": -1, "road_length": 100, "density": 0.5, "seed": 1}, "num_lanes"),
        ({"num_lanes": True, "road_length": 100, "density": 0.5, "seed": 1}, "num_lanes"),
        ({"num_lanes": 4.0, "road_length": 100, "density": 0.5, "seed": 1}, "num_lanes"),
        ({"num_lanes": 4, "road_length": 0, "density": 0.5, "seed": 1}, "road_length"),
        ({"num_lanes": 4, "road_length": -10, "density": 0.5, "seed": 1}, "road_length"),
        ({"num_lanes": 4, "road_length": True, "density": 0.5, "seed": 1}, "road_length"),
        ({"num_lanes": 4, "road_length": 100.0, "density": 0.5, "seed": 1}, "road_length"),
        ({"num_lanes": 4, "road_length": 100, "density": -0.01, "seed": 1}, "density"),
        ({"num_lanes": 4, "road_length": 100, "density": 1.01, "seed": 1}, "density"),
        ({"num_lanes": 4, "road_length": 100, "density": True, "seed": 1}, "density"),
        ({"num_lanes": 4, "road_length": 100, "density": "0.5", "seed": 1}, "density"),
        ({"num_lanes": 4, "road_length": 100, "density": 0.5, "seed": True}, "seed"),
        ({"num_lanes": 4, "road_length": 100, "density": 0.5, "seed": 1.5}, "seed"),
    ],
)
def test_invalid_make_uniform_random_state_inputs(kwargs, field_name):
    with pytest.raises(ValueError, match=field_name):
        make_uniform_random_state(**kwargs)


def test_vehicle_mix_defaults():
    mix = VehicleMix()
    state = make_uniform_random_state(num_lanes=1, road_length=20, density=1.0, seed=123, vehicle_mix=mix)

    assert np.all(state.veh_type == HDV_VEH_TYPE)
    assert np.all(state.behavior_id == HDV_BEHAVIOR_ID)
    assert not np.any(state.controlled)
    assert np.all(state.length == 1)


def test_vehicle_mix_deterministic_counts():
    mix = VehicleMix(av_fraction=0.2, controlled_fraction=0.1, bus_fraction=0.05, bus_length=3)
    state = make_uniform_random_state(num_lanes=1, road_length=100, density=1.0, seed=123, vehicle_mix=mix)

    assert np.sum(state.controlled) == 10
    assert np.sum(state.behavior_id == CONTROLLED_AV_BEHAVIOR_ID) == 10
    assert np.sum(state.behavior_id == AV_BEHAVIOR_ID) == 20
    assert np.sum(state.behavior_id == BUS_BEHAVIOR_ID) == 5
    assert np.sum(state.behavior_id == HDV_BEHAVIOR_ID) == 65
    assert np.sum(state.veh_type == AV_VEH_TYPE) == 30
    assert np.sum(state.veh_type == BUS_VEH_TYPE) == 5
    assert np.sum(state.length == 3) == 5
    assert np.sum(state.length == 1) == 95

    validate_state(state, SimulationParams(num_lanes=1, road_length=100))


@pytest.mark.parametrize(
    ("kwargs", "field_name"),
    [
        ({"av_fraction": -0.1}, "av_fraction"),
        ({"av_fraction": 1.1}, "av_fraction"),
        ({"av_fraction": True}, "av_fraction"),
        ({"controlled_fraction": -0.1}, "controlled_fraction"),
        ({"bus_fraction": 1.1}, "bus_fraction"),
        ({"bus_length": 0}, "bus_length"),
        ({"bus_length": -1}, "bus_length"),
        ({"bus_length": True}, "bus_length"),
        ({"bus_length": 3.0}, "bus_length"),
        (
            {"av_fraction": 0.5, "controlled_fraction": 0.5, "bus_fraction": 0.1},
            "av_fraction",
        ),
    ],
)
def test_vehicle_mix_invalid_values(kwargs, field_name):
    with pytest.raises(ValueError, match=field_name):
        VehicleMix(**kwargs)


def test_public_imports():
    from snfs_traffic.scenarios import VehicleMix as ImportedVehicleMix
    from snfs_traffic.scenarios import make_uniform_random_state as imported_make_uniform_random_state

    assert ImportedVehicleMix is VehicleMix
    assert imported_make_uniform_random_state is make_uniform_random_state
