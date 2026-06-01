from __future__ import annotations

import numpy as np
import pytest

from snfs_traffic.control import (
    LANE_STAY,
    ControlledVehicleAction,
    step_with_controlled_lateral_actions_optimized,
    step_with_controlled_lateral_actions_reference,
)
from snfs_traffic.core import SimulationParams, build_lane_order, build_occupancy, empty_state, step_longitudinal_reference
from snfs_traffic.core.longitudinal_kernels import compute_longitudinal_velocities_kernel
from snfs_traffic.core.longitudinal_numba import (
    NUMBA_AVAILABLE,
    compute_longitudinal_velocities_numba,
    draw_longitudinal_randoms,
)
from snfs_traffic.core.state import TrafficState
from snfs_traffic.topology import RingTopology


def _params(**overrides) -> SimulationParams:
    defaults = dict(
        num_lanes=1,
        road_length=20,
        vmax_default=5,
        vmax_controlled=5,
        G=15,
        q=0.0,
        r=0.0,
        S=1,
        P1=1.0,
        P2=1.0,
        P3=1.0,
        P4=1.0,
        p_lane_change=0.0,
    )
    defaults.update(overrides)
    return SimulationParams(**defaults)


def _state(*, pos: tuple[int, ...], vel: tuple[int, ...], controlled: tuple[bool, ...] | None = None) -> TrafficState:
    state = empty_state(len(pos))
    state.lane[:] = 0
    state.pos[:] = np.asarray(pos, dtype=state.pos.dtype)
    state.vel[:] = np.asarray(vel, dtype=state.vel.dtype)
    if controlled is not None:
        state.controlled[:] = np.asarray(controlled, dtype=state.controlled.dtype)
    return state


def _step_reference_velocity(state: TrafficState, params: SimulationParams) -> np.ndarray:
    topology = RingTopology(num_lanes=params.num_lanes, length=params.road_length)
    out = step_longitudinal_reference(state, params, topology, np.random.default_rng(123))
    return out.vel


def _reference_kernel_velocity(state: TrafficState, params: SimulationParams, *, seed: int = 123) -> np.ndarray:
    occupancy = build_occupancy(state, params)
    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    return compute_longitudinal_velocities_kernel(
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
        rng=np.random.default_rng(seed),
    )


@pytest.mark.parametrize(
    ("name", "params", "state", "expected_velocity"),
    [
        (
            "stopped_equal_speed_positive_gap",
            _params(road_length=20),
            _state(pos=(0, 3), vel=(0, 0)),
            1,
        ),
        (
            "moving_equal_speed_gap_below_G",
            _params(road_length=20),
            _state(pos=(0, 4), vel=(1, 1)),
            2,
        ),
        (
            "gap_equal_G",
            _params(road_length=40),
            _state(pos=(0, 16), vel=(2, 1)),
            3,
        ),
        (
            "strict_non_acceleration_gap_below_G_and_faster_than_leader",
            _params(road_length=30),
            _state(pos=(0, 10), vel=(3, 1)),
            3,
        ),
    ],
)
def test_reference_rule1_acceleration_regressions(
    name: str, params: SimulationParams, state: TrafficState, expected_velocity: int
) -> None:
    del name

    vel = _step_reference_velocity(state, params)

    assert int(vel[0]) == expected_velocity


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba unavailable")
@pytest.mark.parametrize(
    ("params", "state"),
    [
        (_params(road_length=20), _state(pos=(0, 3), vel=(0, 0))),
        (_params(road_length=20), _state(pos=(0, 4), vel=(1, 1))),
        (_params(road_length=40), _state(pos=(0, 16), vel=(2, 1))),
        (_params(road_length=30), _state(pos=(0, 10), vel=(3, 1))),
    ],
)
def test_rule1_reference_and_numba_velocities_match(params: SimulationParams, state: TrafficState) -> None:
    occupancy = build_occupancy(state, params)
    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    seed = 456

    expected = _reference_kernel_velocity(state, params, seed=seed)
    u_s, u_q, u_b = draw_longitudinal_randoms(state.alive, np.random.default_rng(seed))
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


def test_controlled_speed_none_uses_corrected_legacy_rule1_in_reference_path() -> None:
    params = _params(road_length=20)
    topology = RingTopology(num_lanes=params.num_lanes, length=params.road_length)
    state = _state(pos=(0, 4), vel=(1, 1), controlled=(True, True))

    out, result = step_with_controlled_lateral_actions_reference(
        state,
        params,
        topology,
        np.random.default_rng(789),
        {
            0: ControlledVehicleAction(LANE_STAY, None),
            1: ControlledVehicleAction(LANE_STAY, 0),
        },
    )

    assert int(out.vel[0]) == 2
    assert int(result.desired_velocity[0]) == 2
    assert int(out.vel[1]) == 1


def test_explicit_speed_delta_zero_and_plus_one_semantics_are_unchanged() -> None:
    params = _params(road_length=20)
    topology = RingTopology(num_lanes=params.num_lanes, length=params.road_length)
    state = _state(pos=(0, 3), vel=(0, 0), controlled=(True, False))

    keep, keep_result = step_with_controlled_lateral_actions_reference(
        state,
        params,
        topology,
        np.random.default_rng(321),
        {0: ControlledVehicleAction(LANE_STAY, 0)},
    )
    accelerate, accelerate_result = step_with_controlled_lateral_actions_reference(
        state,
        params,
        topology,
        np.random.default_rng(321),
        {0: ControlledVehicleAction(LANE_STAY, 1)},
    )

    assert int(keep.vel[0]) == 0
    assert int(keep_result.desired_velocity[0]) == 0
    assert int(accelerate.vel[0]) == 1
    assert int(accelerate_result.desired_velocity[0]) == 1


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba unavailable")
def test_controlled_speed_none_reference_and_optimized_paths_match() -> None:
    params = _params(road_length=20)
    topology = RingTopology(num_lanes=params.num_lanes, length=params.road_length)
    state = _state(pos=(0, 4), vel=(1, 1), controlled=(True, True))
    actions = {
        0: ControlledVehicleAction(LANE_STAY, None),
        1: ControlledVehicleAction(LANE_STAY, 0),
    }

    ref, ref_result = step_with_controlled_lateral_actions_reference(
        state, params, topology, np.random.default_rng(987), actions
    )
    opt, opt_result, used_reference = step_with_controlled_lateral_actions_optimized(
        state, params, topology, np.random.default_rng(987), actions
    )

    assert used_reference is False
    for name in ("lane", "pos", "vel", "alive", "controlled", "length"):
        np.testing.assert_array_equal(getattr(opt, name), getattr(ref, name), err_msg=name)
    np.testing.assert_array_equal(opt_result.desired_velocity, ref_result.desired_velocity)
    np.testing.assert_array_equal(opt_result.applied_velocity, ref_result.applied_velocity)
    np.testing.assert_array_equal(opt_result.speed_clipped_by_safety, ref_result.speed_clipped_by_safety)
    assert int(opt.vel[0]) == 2
