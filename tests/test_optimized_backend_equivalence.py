from __future__ import annotations

import numpy as np
import pytest

from snfs_traffic.backends.optimized import LONG_NUMBA_AVAILABLE, get_optimized_backend
from snfs_traffic.core import SimulationParams, step_reference, validate_runtime_invariants
from snfs_traffic.core.types import VELOCITY_DTYPE
from snfs_traffic.scenarios import make_uniform_random_state
from snfs_traffic.topology import RingTopology

FIELDS = ("lane", "pos", "vel", "alive", "controlled", "changed_lane", "last_lane_delta")


def _assert_equal_fields(actual, expected) -> None:
    for field in FIELDS:
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


def test_optimized_backend_one_step_equivalence() -> None:
    params = SimulationParams(num_lanes=3, road_length=100, p_lane_change=0.5)
    topology = RingTopology(num_lanes=3, length=100)
    state = make_uniform_random_state(num_lanes=3, road_length=100, density=0.3, seed=3)
    rng_ref = np.random.default_rng(7)
    rng_opt = np.random.default_rng(7)

    expected = step_reference(state.copy(), params, topology, rng_ref)
    actual = get_optimized_backend().step(state.copy(), params, topology, rng_opt)
    _assert_equal_fields(actual, expected)
    assert float(rng_ref.random()) == float(rng_opt.random())


def test_optimized_backend_multistep_rollout_grid_equivalence() -> None:
    backend = get_optimized_backend()
    for num_lanes in (1, 2, 3, 4):
        for density in (0.0, 0.05, 0.2, 0.5, 1.0):
            for p_lane_change in (0.0, 0.5, 1.0):
                for seed in (0, 1, 2):
                    road_length = 60
                    params = SimulationParams(num_lanes=num_lanes, road_length=road_length, p_lane_change=p_lane_change)
                    topology = RingTopology(num_lanes=num_lanes, length=road_length)

                    state_ref = make_uniform_random_state(
                        num_lanes=num_lanes,
                        road_length=road_length,
                        density=density,
                        seed=seed,
                    )
                    state_opt = state_ref.copy()
                    rng_ref = np.random.default_rng(seed + 1000)
                    rng_opt = np.random.default_rng(seed + 1000)

                    for _ in range(8):
                        state_ref = step_reference(state_ref, params, topology, rng_ref)
                        state_opt = backend.step(state_opt, params, topology, rng_opt)
                        _assert_equal_fields(state_opt, state_ref)
                        validate_runtime_invariants(state_opt, params, topology)

                    assert float(rng_ref.random()) == float(rng_opt.random())


@pytest.mark.skipif(not LONG_NUMBA_AVAILABLE, reason="numba unavailable")
def test_optimized_backend_unit_lengths_use_numba_velocity_not_python_kernel(monkeypatch) -> None:
    params = SimulationParams(num_lanes=2, road_length=40, p_lane_change=0.4)
    topology = RingTopology(num_lanes=2, length=40)
    state = make_uniform_random_state(num_lanes=2, road_length=40, density=0.25, seed=22)
    rng_ref = np.random.default_rng(555)
    rng_opt = np.random.default_rng(555)

    expected = step_reference(state.copy(), params, topology, rng_ref)

    def fail_python_velocity(*args, **kwargs):
        raise AssertionError("optimized unit-length path must not call Python longitudinal velocity kernel")

    monkeypatch.setattr("snfs_traffic.backends.optimized.compute_longitudinal_velocities_kernel", fail_python_velocity)

    actual = get_optimized_backend().step(state.copy(), params, topology, rng_opt)
    _assert_equal_fields(actual, expected)
    assert float(rng_ref.random()) == float(rng_opt.random())


def test_optimized_backend_non_unit_length_falls_back_to_reference(monkeypatch) -> None:
    params = SimulationParams(num_lanes=1, road_length=30, p_lane_change=0.0)
    topology = RingTopology(num_lanes=1, length=30)
    state = make_uniform_random_state(num_lanes=1, road_length=30, density=0.05, seed=33)
    assert state.n_vehicles >= 1
    state.length[0] = 2
    rng_ref = np.random.default_rng(777)
    rng_opt = np.random.default_rng(777)

    expected = step_reference(state.copy(), params, topology, rng_ref)

    monkeypatch.setattr("snfs_traffic.backends.optimized.INDEX_NUMBA_AVAILABLE", True)
    monkeypatch.setattr("snfs_traffic.backends.optimized.LANE_NUMBA_AVAILABLE", True)
    monkeypatch.setattr("snfs_traffic.backends.optimized.LONG_NUMBA_AVAILABLE", True)
    numba_velocity_called = False

    def fail_numba_velocity(*args, **kwargs):
        nonlocal numba_velocity_called
        numba_velocity_called = True
        raise AssertionError("non-unit lengths must not call Numba longitudinal velocity")

    monkeypatch.setattr("snfs_traffic.backends.optimized.compute_longitudinal_velocities_numba", fail_numba_velocity)

    actual = get_optimized_backend().step(state.copy(), params, topology, rng_opt)
    assert not numba_velocity_called
    _assert_equal_fields(actual, expected)
    assert float(rng_ref.random()) == float(rng_opt.random())


def test_optimized_backend_preserves_reference_input_validation() -> None:
    params = SimulationParams(num_lanes=2, road_length=10, p_lane_change=0.2)
    bad_topology = RingTopology(num_lanes=3, length=10)
    state = make_uniform_random_state(num_lanes=2, road_length=10, density=0.2, seed=9)

    with pytest.raises(ValueError, match="topology.num_lanes must match params.num_lanes"):
        get_optimized_backend().step(state, params, bad_topology, np.random.default_rng(0))


def test_optimized_backend_rejects_invalid_overflow_velocity_postconditions(monkeypatch) -> None:
    params = SimulationParams(num_lanes=1, road_length=10, vmax_default=10, vmax_controlled=10, p_lane_change=0.0)
    topology = RingTopology(num_lanes=1, length=10)
    state = make_uniform_random_state(num_lanes=1, road_length=10, density=0.1, seed=41)

    monkeypatch.setattr("snfs_traffic.backends.optimized.INDEX_NUMBA_AVAILABLE", True)
    monkeypatch.setattr("snfs_traffic.backends.optimized.LANE_NUMBA_AVAILABLE", True)
    monkeypatch.setattr("snfs_traffic.backends.optimized.LONG_NUMBA_AVAILABLE", True)
    monkeypatch.setattr(
        "snfs_traffic.backends.optimized.build_index_and_neighbors_numba",
        lambda lane, pos, alive, num_lanes, road_length: (
            np.full((num_lanes, road_length), -1, dtype=np.int32),
            np.full((num_lanes, road_length), -1, dtype=np.int32),
            np.zeros(num_lanes, dtype=np.int32),
            np.zeros(lane.shape[0], dtype=np.int32),
            np.full(lane.shape[0], -1, dtype=np.int32),
            np.full(lane.shape[0], -1, dtype=np.int32),
            np.full(lane.shape[0], road_length - 1, dtype=np.int32),
            np.full(lane.shape[0], road_length - 1, dtype=np.int32),
        ),
    )
    monkeypatch.setattr("snfs_traffic.backends.optimized.collect_lane_change_proposals_numba", lambda **kwargs: np.zeros(state.n_vehicles, dtype=np.int8))
    monkeypatch.setattr("snfs_traffic.backends.optimized.resolve_lane_change_conflicts_kernel", lambda proposals, rng, **kwargs: proposals)
    monkeypatch.setattr(
        "snfs_traffic.backends.optimized.apply_lane_changes_kernel",
        lambda lane, changed_lane, last_lane_delta, accepted: (lane.copy(), changed_lane.copy(), last_lane_delta.copy()),
    )
    velocity_min = int(np.iinfo(VELOCITY_DTYPE).min)
    monkeypatch.setattr(
        "snfs_traffic.backends.optimized.compute_longitudinal_velocities_numba",
        lambda *args, **kwargs: np.full(state.n_vehicles, velocity_min, dtype=VELOCITY_DTYPE),
    )
    monkeypatch.setattr("snfs_traffic.backends.optimized.advance_positions_numba", lambda pos, vel, alive, road_length: pos.copy())

    with pytest.raises(ValueError, match="vel out of range"):
        get_optimized_backend().step(state, params, topology, np.random.default_rng(99))


def test_reference_and_optimized_equivalent_at_velocity_dtype_boundary_when_valid() -> None:
    velocity_max = int(np.iinfo(VELOCITY_DTYPE).max)
    boundary_vmax = velocity_max - 1
    params = SimulationParams(num_lanes=1, road_length=100, vmax_default=boundary_vmax, vmax_controlled=boundary_vmax, P2=0.0, p_lane_change=0.0)
    topology = RingTopology(num_lanes=1, length=100)
    state = make_uniform_random_state(num_lanes=1, road_length=100, density=0.1, seed=42)
    state.controlled[:] = False
    state.vel[:] = np.full(state.n_vehicles, boundary_vmax, dtype=VELOCITY_DTYPE)

    rng_ref = np.random.default_rng(1234)
    rng_opt = np.random.default_rng(1234)

    expected = step_reference(state.copy(), params, topology, rng_ref)
    actual = get_optimized_backend().step(state.copy(), params, topology, rng_opt)
    _assert_equal_fields(actual, expected)
    assert float(rng_ref.random()) == float(rng_opt.random())
