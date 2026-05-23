import numpy as np

from snfs_traffic.core import (
    ReferenceBackend,
    SimulationParams,
    StepBackend,
    get_reference_backend,
    step_reference,
    validate_runtime_invariants,
)
from snfs_traffic.scenarios import VehicleMix, make_uniform_random_state
from snfs_traffic.topology import RingTopology


def assert_states_equal(actual, expected):
    assert actual.n_vehicles == expected.n_vehicles
    np.testing.assert_array_equal(actual.alive, expected.alive)
    np.testing.assert_array_equal(actual.lane, expected.lane)
    np.testing.assert_array_equal(actual.pos, expected.pos)
    np.testing.assert_array_equal(actual.vel, expected.vel)
    np.testing.assert_array_equal(actual.length, expected.length)
    np.testing.assert_array_equal(actual.controlled, expected.controlled)
    np.testing.assert_array_equal(actual.changed_lane, expected.changed_lane)
    np.testing.assert_array_equal(actual.last_lane_delta, expected.last_lane_delta)


def test_backend_public_imports_and_reference_singleton():
    backend = get_reference_backend()
    assert StepBackend is not None
    assert isinstance(backend, ReferenceBackend)
    assert backend.name == "reference"
    assert callable(backend.step)


def test_reference_backend_one_step_equivalence():
    params = SimulationParams(num_lanes=3, road_length=100, vmax_default=5, vmax_controlled=6, p_lane_change=0.5)
    topology = RingTopology(num_lanes=3, length=100)
    state = make_uniform_random_state(num_lanes=3, road_length=100, density=0.35, seed=123)

    rng_ref = np.random.default_rng(456)
    rng_backend = np.random.default_rng(456)

    expected = step_reference(state, params, topology, rng_ref)
    actual = get_reference_backend().step(state, params, topology, rng_backend)

    assert_states_equal(actual, expected)
    validate_runtime_invariants(actual, params, topology)


def test_reference_backend_multistep_equivalence_across_densities():
    params = SimulationParams(num_lanes=3, road_length=80, vmax_default=5, vmax_controlled=6, p_lane_change=0.5)
    topology = RingTopology(num_lanes=3, length=80)
    backend = get_reference_backend()

    for density in [0.0, 0.05, 0.2, 0.5, 0.8, 1.0]:
        seed = int(1000 * density) + 11
        state_ref = make_uniform_random_state(num_lanes=3, road_length=80, density=density, seed=seed)
        state_backend = make_uniform_random_state(num_lanes=3, road_length=80, density=density, seed=seed)

        rng_ref = np.random.default_rng(seed + 1000)
        rng_backend = np.random.default_rng(seed + 1000)

        for _ in range(30):
            validate_runtime_invariants(state_ref, params, topology)
            validate_runtime_invariants(state_backend, params, topology)

            state_ref = step_reference(state_ref, params, topology, rng_ref)
            state_backend = backend.step(state_backend, params, topology, rng_backend)

            assert_states_equal(state_backend, state_ref)
            validate_runtime_invariants(state_backend, params, topology)


def test_reference_backend_multiseed_equivalence_smoke():
    params = SimulationParams(num_lanes=4, road_length=40)
    topology = RingTopology(num_lanes=4, length=40)
    backend = get_reference_backend()

    for seed in [0, 1, 2, 3, 4]:
        state_ref = make_uniform_random_state(num_lanes=4, road_length=40, density=0.35, seed=seed)
        state_backend = make_uniform_random_state(num_lanes=4, road_length=40, density=0.35, seed=seed)
        rng_ref = np.random.default_rng(seed + 1000)
        rng_backend = np.random.default_rng(seed + 1000)

        for _ in range(20):
            validate_runtime_invariants(state_ref, params, topology)
            validate_runtime_invariants(state_backend, params, topology)

            state_ref = step_reference(state_ref, params, topology, rng_ref)
            state_backend = backend.step(state_backend, params, topology, rng_backend)
            assert_states_equal(state_backend, state_ref)
            validate_runtime_invariants(state_backend, params, topology)


def test_reference_backend_full_occupancy_equivalence_no_lane_changes():
    params = SimulationParams(num_lanes=2, road_length=5)
    topology = RingTopology(num_lanes=2, length=5)
    backend = get_reference_backend()

    state_ref = make_uniform_random_state(num_lanes=2, road_length=5, density=1.0, seed=123)
    state_backend = make_uniform_random_state(num_lanes=2, road_length=5, density=1.0, seed=123)

    rng_ref = np.random.default_rng(456)
    rng_backend = np.random.default_rng(456)

    for _ in range(15):
        state_ref = step_reference(state_ref, params, topology, rng_ref)
        state_backend = backend.step(state_backend, params, topology, rng_backend)
        assert_states_equal(state_backend, state_ref)
        validate_runtime_invariants(state_backend, params, topology)
        assert not np.any(state_backend.changed_lane)
        assert np.all(state_backend.last_lane_delta == 0)


def test_reference_backend_equivalence_intentionally_ignores_bus_body_cells():
    params = SimulationParams(num_lanes=3, road_length=30)
    topology = RingTopology(num_lanes=3, length=30)
    backend = get_reference_backend()

    mix = VehicleMix(bus_fraction=1.0, bus_length=3)
    state_ref = make_uniform_random_state(num_lanes=3, road_length=30, density=0.4, seed=123, vehicle_mix=mix)
    state_backend = make_uniform_random_state(num_lanes=3, road_length=30, density=0.4, seed=123, vehicle_mix=mix)

    assert np.all(state_ref.length[state_ref.alive] == 3)
    assert np.all(state_backend.length[state_backend.alive] == 3)

    rng_ref = np.random.default_rng(456)
    rng_backend = np.random.default_rng(456)

    for _ in range(20):
        state_ref = step_reference(state_ref, params, topology, rng_ref)
        state_backend = backend.step(state_backend, params, topology, rng_backend)
        assert_states_equal(state_backend, state_ref)
        validate_runtime_invariants(state_backend, params, topology)
