import numpy as np
import pytest

from snfs_traffic.core import (
    SimulationParams,
    TrafficState,
    build_occupancy,
    step_reference,
    validate_runtime_invariants,
)
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
from snfs_traffic.scenarios import VehicleMix, make_uniform_random_state
from snfs_traffic.topology import RingTopology


def _params(*, num_lanes: int, road_length: int) -> SimulationParams:
    return SimulationParams(
        num_lanes=num_lanes,
        road_length=road_length,
        vmax_default=5,
        vmax_controlled=6,
        p_lane_change=0.5,
    )


def test_validate_runtime_invariants_public_import_callable():
    assert callable(validate_runtime_invariants)


@pytest.mark.parametrize(
    ("num_lanes", "road_length", "density", "seed"),
    [
        (1, 10, 0.0, 0),
        (1, 10, 1.0, 1),
        (2, 5, 1.0, 2),
        (2, 30, 0.5, 3),
        (3, 50, 0.2, 4),
        (4, 100, 0.8, 5),
    ],
)
def test_validate_runtime_invariants_initial_random_states(num_lanes, road_length, density, seed):
    params = _params(num_lanes=num_lanes, road_length=road_length)
    topology = RingTopology(num_lanes=num_lanes, length=road_length)
    state = make_uniform_random_state(
        num_lanes=num_lanes,
        road_length=road_length,
        density=density,
        seed=seed,
    )
    validate_runtime_invariants(state, params, topology)


@pytest.mark.parametrize("density", [0.05, 0.2, 0.5, 0.8, 1.0])
def test_validate_runtime_invariants_random_rollouts(density):
    params = _params(num_lanes=3, road_length=100)
    topology = RingTopology(num_lanes=3, length=100)
    seed = int(1000 * density)
    state = make_uniform_random_state(num_lanes=3, road_length=100, density=density, seed=seed)
    rng = np.random.default_rng(seed + 99)

    for _ in range(50):
        validate_runtime_invariants(state, params, topology)
        state = step_reference(state, params, topology, rng)
        validate_runtime_invariants(state, params, topology)


def test_validate_runtime_invariants_random_rollouts_multiple_seeds_smoke():
    params = _params(num_lanes=4, road_length=40)
    topology = RingTopology(num_lanes=4, length=40)

    for seed in [0, 1, 2, 3, 4]:
        state = make_uniform_random_state(num_lanes=4, road_length=40, density=0.35, seed=seed)
        rng = np.random.default_rng(seed)
        for _ in range(30):
            validate_runtime_invariants(state, params, topology)
            state = step_reference(state, params, topology, rng)
            validate_runtime_invariants(state, params, topology)


def test_validate_runtime_invariants_density_one_stays_valid_and_no_lane_changes():
    params = _params(num_lanes=2, road_length=5)
    topology = RingTopology(num_lanes=2, length=5)
    state = make_uniform_random_state(num_lanes=2, road_length=5, density=1.0, seed=123)
    rng = np.random.default_rng(321)

    for _ in range(20):
        validate_runtime_invariants(state, params, topology)
        occupancy = build_occupancy(state, params)
        assert int((occupancy >= 0).sum()) == int(state.alive.sum())
        assert not state.changed_lane.any()
        assert np.all(state.last_lane_delta == 0)

        state = step_reference(state, params, topology, rng)


def _manual_state_two_duplicates() -> TrafficState:
    return TrafficState(
        vehicle_id=np.array([0, 1], dtype=VEHICLE_ID_DTYPE),
        lane=np.array([0, 0], dtype=LANE_DTYPE),
        pos=np.array([3, 3], dtype=POSITION_DTYPE),
        vel=np.array([0, 0], dtype=VELOCITY_DTYPE),
        length=np.array([1, 1], dtype=LENGTH_DTYPE),
        veh_type=np.array([0, 0], dtype=VEHICLE_TYPE_DTYPE),
        behavior_id=np.array([0, 0], dtype=BEHAVIOR_ID_DTYPE),
        alive=np.array([True, True], dtype=BOOL_DTYPE),
        last_lane_delta=np.array([0, 0], dtype=LANE_DELTA_DTYPE),
        changed_lane=np.array([False, False], dtype=BOOL_DTYPE),
        controlled=np.array([False, False], dtype=BOOL_DTYPE),
    )


def test_validate_runtime_invariants_rejects_duplicate_head_cell():
    state = _manual_state_two_duplicates()
    params = _params(num_lanes=1, road_length=10)
    topology = RingTopology(num_lanes=1, length=10)

    with pytest.raises(ValueError, match="duplicate|occup"):
        validate_runtime_invariants(state, params, topology)


def test_validate_runtime_invariants_rejects_mismatched_topology():
    params = _params(num_lanes=2, road_length=10)
    state = make_uniform_random_state(num_lanes=2, road_length=10, density=0.2, seed=7)

    with pytest.raises(ValueError, match="topology.length"):
        validate_runtime_invariants(state, params, RingTopology(num_lanes=2, length=11))

    with pytest.raises(ValueError, match="topology.num_lanes"):
        validate_runtime_invariants(state, params, RingTopology(num_lanes=3, length=10))


def test_validate_runtime_invariants_rejects_invalid_lane_change_flags():
    params = _params(num_lanes=2, road_length=10)
    topology = RingTopology(num_lanes=2, length=10)
    state = make_uniform_random_state(num_lanes=2, road_length=10, density=0.5, seed=8)

    bad = state.copy()
    bad.changed_lane[0] = True
    bad.last_lane_delta[0] = 0
    with pytest.raises(ValueError, match="changed_lane"):
        validate_runtime_invariants(bad, params, topology)

    bad2 = state.copy()
    bad2.last_lane_delta[0] = 2
    bad2.changed_lane[0] = True
    with pytest.raises(ValueError, match="last_lane_delta"):
        validate_runtime_invariants(bad2, params, topology)


def test_validate_runtime_invariants_rejects_inactive_lane_change_flags():
    params = _params(num_lanes=2, road_length=10)
    topology = RingTopology(num_lanes=2, length=10)
    state = make_uniform_random_state(num_lanes=2, road_length=10, density=0.5, seed=9)

    bad = state.copy()
    bad.alive[0] = False
    bad.changed_lane[0] = True
    bad.last_lane_delta[0] = 1

    with pytest.raises(ValueError, match="inactive"):
        validate_runtime_invariants(bad, params, topology)


def test_validate_runtime_invariants_rejects_invalid_velocity_bounds():
    params = _params(num_lanes=3, road_length=20)
    topology = RingTopology(num_lanes=3, length=20)
    state = make_uniform_random_state(num_lanes=3, road_length=20, density=0.5, seed=10)

    bad = state.copy()
    bad.controlled[:] = False
    bad.vel[0] = params.vmax_default + 1
    with pytest.raises(ValueError, match="uncontrolled"):
        validate_runtime_invariants(bad, params, topology)

    bad3 = state.copy()
    bad3.controlled[0] = True
    bad3.vel[0] = params.vmax_controlled + 1
    with pytest.raises(ValueError, match="vel out of range|controlled"):
        validate_runtime_invariants(bad3, params, topology)


def test_runtime_invariants_intentionally_ignore_bus_body_cells():
    params = _params(num_lanes=3, road_length=30)
    topology = RingTopology(num_lanes=3, length=30)
    state = make_uniform_random_state(
        num_lanes=3,
        road_length=30,
        density=0.4,
        seed=11,
        vehicle_mix=VehicleMix(bus_fraction=1.0, bus_length=3),
    )
    assert np.all(state.length[state.alive] == 3)

    rng = np.random.default_rng(12)
    for _ in range(20):
        validate_runtime_invariants(state, params, topology)
        occupancy = build_occupancy(state, params)
        assert int((occupancy >= 0).sum()) == int(state.alive.sum())
        state = step_reference(state, params, topology, rng)


def test_validate_runtime_invariants_does_not_mutate_state():
    params = _params(num_lanes=3, road_length=50)
    topology = RingTopology(num_lanes=3, length=50)
    state = make_uniform_random_state(num_lanes=3, road_length=50, density=0.4, seed=13)

    before = {field: getattr(state, field).copy() for field in state.__dataclass_fields__}
    validate_runtime_invariants(state, params, topology)

    for field, arr_before in before.items():
        np.testing.assert_array_equal(getattr(state, field), arr_before)
