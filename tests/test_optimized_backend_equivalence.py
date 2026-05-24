from __future__ import annotations

import numpy as np

from snfs_traffic.backends.optimized import get_optimized_backend
from snfs_traffic.core import SimulationParams, step_reference, validate_runtime_invariants
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
