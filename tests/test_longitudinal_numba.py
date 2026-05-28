from __future__ import annotations

import numpy as np
import pytest

from snfs_traffic.core import SimulationParams, build_lane_order, build_occupancy
from snfs_traffic.core.longitudinal_kernels import advance_positions_kernel, compute_longitudinal_velocities_kernel
from snfs_traffic.core.longitudinal_numba import (
    NUMBA_AVAILABLE,
    advance_positions_numba,
    compute_longitudinal_velocities_numba,
    draw_longitudinal_randoms,
)
from snfs_traffic.core.state import TrafficState, empty_state
from snfs_traffic.core.types import BOOL_DTYPE, LANE_DTYPE, POSITION_DTYPE, VELOCITY_DTYPE
from snfs_traffic.scenarios import make_uniform_random_state


def test_longitudinal_numba_optional_import():
    assert isinstance(NUMBA_AVAILABLE, bool)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba unavailable")
def test_advance_positions_numba_equivalent():
    pos = np.array([0, 5, 9], dtype=np.int64)
    vel = np.array([1, 2, 0], dtype=np.int64)
    alive = np.array([True, True, False])
    assert np.array_equal(
        advance_positions_kernel(pos, vel, alive, road_length=10),
        advance_positions_numba(pos, vel, alive, road_length=10),
    )


class FakeRng:
    def __init__(self) -> None:
        self.draws: list[float] = []
        self.next_value = 0

    def random(self) -> float:
        value = self.next_value / 10.0
        self.next_value += 1
        self.draws.append(value)
        return value


def test_draw_longitudinal_randoms_consumes_scalar_draws_for_alive_only() -> None:
    alive = np.array([True, False, True, True, False], dtype=BOOL_DTYPE)
    rng = FakeRng()

    u_s, u_q, u_b = draw_longitudinal_randoms(alive, rng)

    assert len(rng.draws) == 3 * int(np.count_nonzero(alive))
    assert rng.draws == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    assert u_s.shape == alive.shape and u_q.shape == alive.shape and u_b.shape == alive.shape
    assert u_s.dtype == np.float64 and u_q.dtype == np.float64 and u_b.dtype == np.float64
    np.testing.assert_array_equal(u_s, np.array([0.0, 0.0, 0.3, 0.6, 0.0]))
    np.testing.assert_array_equal(u_q, np.array([0.1, 0.0, 0.4, 0.7, 0.0]))
    np.testing.assert_array_equal(u_b, np.array([0.2, 0.0, 0.5, 0.8, 0.0]))


def _state_from_arrays(
    lane: list[int],
    pos: list[int],
    vel: list[int],
    *,
    alive: list[bool] | None = None,
    controlled: list[bool] | None = None,
) -> TrafficState:
    state = empty_state(len(lane))
    state.lane = np.array(lane, dtype=LANE_DTYPE)
    state.pos = np.array(pos, dtype=POSITION_DTYPE)
    state.vel = np.array(vel, dtype=VELOCITY_DTYPE)
    state.length[:] = 1
    if alive is not None:
        state.alive = np.array(alive, dtype=BOOL_DTYPE)
    if controlled is not None:
        state.controlled = np.array(controlled, dtype=BOOL_DTYPE)
    return state


def _assert_numba_velocity_matches_reference(state: TrafficState, params: SimulationParams, *, seed: int) -> None:
    occupancy = build_occupancy(state, params)
    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    rng_ref = np.random.default_rng(seed)
    rng_numba = np.random.default_rng(seed)

    expected = compute_longitudinal_velocities_kernel(
        state.lane,
        state.pos,
        state.vel,
        state.length,
        state.alive,
        state.controlled,
        lane_order,
        lane_counts,
        lane_rank,
        road_length=params.road_length,
        vmax_default=params.vmax_default,
        vmax_controlled=params.vmax_controlled,
        G=params.G,
        q=params.q,
        r=params.r,
        S=params.S,
        P1=params.P1,
        P2=params.P2,
        P3=params.P3,
        P4=params.P4,
        rng=rng_ref,
    )
    u_s, u_q, u_b = draw_longitudinal_randoms(state.alive, rng_numba)
    actual = compute_longitudinal_velocities_numba(
        state.lane,
        state.pos,
        state.vel,
        state.alive,
        state.controlled,
        lane_order,
        lane_counts,
        lane_rank,
        u_s,
        u_q,
        u_b,
        road_length=params.road_length,
        vmax_default=params.vmax_default,
        vmax_controlled=params.vmax_controlled,
        G=params.G,
        q=params.q,
        r=params.r,
        S=params.S,
        P1=params.P1,
        P2=params.P2,
        P3=params.P3,
        P4=params.P4,
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == state.vel.dtype
    assert float(rng_ref.random()) == float(rng_numba.random())


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba unavailable")
@pytest.mark.parametrize("num_lanes", [1, 2, 3, 4])
@pytest.mark.parametrize("density", [0.0, 0.05, 0.2, 0.5, 1.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_compute_longitudinal_velocities_numba_random_grid(num_lanes: int, density: float, seed: int) -> None:
    road_length = 40
    params = SimulationParams(
        num_lanes=num_lanes,
        road_length=road_length,
        vmax_default=5,
        vmax_controlled=7,
        G=2,
        q=0.65,
        r=0.45,
        S=3,
        P1=0.2,
        P2=0.4,
        P3=0.6,
        P4=0.8,
    )
    state = make_uniform_random_state(num_lanes=num_lanes, road_length=road_length, density=density, seed=seed)
    if state.n_vehicles:
        rng = np.random.default_rng(seed + 77)
        state.vel[:] = rng.integers(0, params.vmax_default + 1, size=state.n_vehicles, dtype=VELOCITY_DTYPE)
        state.controlled[:] = rng.random(state.n_vehicles) < 0.25
    _assert_numba_velocity_matches_reference(state, params, seed=seed + 100)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba unavailable")
@pytest.mark.parametrize(
    ("name", "state", "params"),
    [
        (
            "single_vehicle_lane",
            _state_from_arrays([0], [3], [1]),
            SimulationParams(num_lanes=1, road_length=10, q=1.0, r=1.0, S=2),
        ),
        (
            "adjacent_vehicles",
            _state_from_arrays([0, 0], [2, 3], [2, 0]),
            SimulationParams(num_lanes=1, road_length=10, q=1.0, r=1.0, S=2),
        ),
        (
            "wraparound_leader",
            _state_from_arrays([0, 0], [8, 1], [2, 1]),
            SimulationParams(num_lanes=1, road_length=10, q=0.0, r=0.0),
        ),
        (
            "inactive_vehicle_unchanged",
            _state_from_arrays([0, 0, 0], [1, 5, 5], [1, 3, 4], alive=[True, False, True]),
            SimulationParams(num_lanes=1, road_length=10, q=1.0, r=1.0, S=3),
        ),
        (
            "controlled_vmax",
            _state_from_arrays([0], [4], [3], controlled=[True]),
            SimulationParams(num_lanes=1, road_length=20, vmax_default=3, vmax_controlled=5, q=0.0, r=0.0, P1=1.0),
        ),
        (
            "q_one",
            _state_from_arrays([0, 0, 0], [0, 4, 9], [3, 1, 2]),
            SimulationParams(num_lanes=1, road_length=12, q=1.0, r=1.0, S=3, P1=1.0, P2=1.0, P3=1.0, P4=1.0),
        ),
        (
            "q_zero",
            _state_from_arrays([0, 0, 0], [0, 4, 9], [3, 1, 2]),
            SimulationParams(num_lanes=1, road_length=12, q=0.0, r=1.0, S=3, P1=1.0, P2=1.0, P3=1.0, P4=1.0),
        ),
        (
            "r_s_lookahead",
            _state_from_arrays([0, 0, 0], [0, 2, 7], [4, 0, 0]),
            SimulationParams(num_lanes=1, road_length=12, q=1.0, r=1.0, S=3, P1=1.0, P2=1.0, P3=1.0, P4=1.0),
        ),
        (
            "p1_keep",
            _state_from_arrays([0, 0], [0, 8], [1, 0]),
            SimulationParams(num_lanes=1, road_length=20, G=2, P1=1.0, q=0.0, r=0.0),
        ),
        (
            "p1_brake",
            _state_from_arrays([0, 0], [0, 8], [1, 0]),
            SimulationParams(num_lanes=1, road_length=20, G=2, P1=0.0, q=0.0, r=0.0),
        ),
        (
            "p2_branch",
            _state_from_arrays([0, 0], [0, 5], [1, 3]),
            SimulationParams(num_lanes=1, road_length=20, G=10, P2=0.0, q=0.0, r=0.0),
        ),
        (
            "p3_branch",
            _state_from_arrays([0, 0], [0, 5], [2, 2]),
            SimulationParams(num_lanes=1, road_length=20, G=10, P3=0.0, q=0.0, r=0.0),
        ),
        (
            "p4_branch",
            _state_from_arrays([0, 0], [0, 5], [3, 1]),
            SimulationParams(num_lanes=1, road_length=20, G=10, P4=0.0, q=0.0, r=0.0),
        ),
        (
            "stopped_braking_guard",
            _state_from_arrays([0, 0], [0, 1], [0, 0]),
            SimulationParams(num_lanes=1, road_length=10, G=5, P3=0.0, q=0.0, r=0.0),
        ),
        (
            "collision_avoidance_propagates",
            _state_from_arrays([0, 0, 0], [0, 3, 7], [5, 3, 1]),
            SimulationParams(num_lanes=1, road_length=12, q=0.0, r=0.0, P1=1.0, P2=1.0, P3=1.0, P4=1.0),
        ),
    ],
)
def test_compute_longitudinal_velocities_numba_focused_cases(name: str, state: TrafficState, params: SimulationParams) -> None:
    del name
    _assert_numba_velocity_matches_reference(state, params, seed=123)
