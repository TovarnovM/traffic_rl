from __future__ import annotations

import numpy as np
import pytest

from snfs_traffic.control import (
    LANE_LEFT,
    step_with_lateral_overrides_optimized,
    step_with_lateral_overrides_reference,
)
from snfs_traffic.core import SimulationParams, empty_state, step_reference
from snfs_traffic.core.longitudinal_numba import NUMBA_AVAILABLE as LONG_NUMBA_AVAILABLE
from snfs_traffic.core.lane_change_numba import NUMBA_AVAILABLE as LANE_NUMBA_AVAILABLE
from snfs_traffic.topology import RingTopology


def _params():
    return SimulationParams(
        num_lanes=2,
        road_length=50,
        p_lane_change=0.0,
        q=1.0,
        r=1.0,
        P1=1.0,
        P2=1.0,
        P3=1.0,
        P4=1.0,
    )


def _state():
    state = empty_state(3)
    state.lane[:] = np.asarray([1, 1, 0], dtype=state.lane.dtype)
    state.pos[:] = np.asarray([10, 35, 30], dtype=state.pos.dtype)
    state.vel[:] = np.asarray([2, 2, 2], dtype=state.vel.dtype)
    return state


def test_empty_override_is_exactly_native_reference_step():
    params = _params()
    topology = RingTopology(num_lanes=2, length=50)

    expected = step_reference(_state(), params, topology, np.random.default_rng(123))
    actual, result = step_with_lateral_overrides_reference(
        _state(), params, topology, np.random.default_rng(123), {}
    )

    for field in ("lane", "pos", "vel", "changed_lane", "last_lane_delta"):
        np.testing.assert_array_equal(
            getattr(actual, field), getattr(expected, field), err_msg=field
        )
    assert result.vehicle_id.size == 0


def test_safe_override_moves_uncontrolled_vehicle_and_reports_result():
    params = _params()
    topology = RingTopology(num_lanes=2, length=50)

    out, result = step_with_lateral_overrides_reference(
        _state(), params, topology, np.random.default_rng(9), {0: LANE_LEFT}
    )

    assert int(out.lane[0]) == 0
    assert bool(out.changed_lane[0])
    assert int(out.last_lane_delta[0]) == LANE_LEFT
    assert result.vehicle_id.tolist() == [0]
    assert result.valid.tolist() == [True]
    assert result.applied.tolist() == [True]


def test_unsafe_override_is_rejected():
    params = _params()
    topology = RingTopology(num_lanes=2, length=50)
    state = _state()
    state.pos[2] = state.pos[0]

    out, result = step_with_lateral_overrides_reference(
        state, params, topology, np.random.default_rng(9), {0: LANE_LEFT}
    )

    assert int(out.lane[0]) == 1
    assert result.valid.tolist() == [False]
    assert result.applied.tolist() == [False]
    assert result.rejection_reason == ("target_occupied",)


@pytest.mark.skipif(
    not (LONG_NUMBA_AVAILABLE and LANE_NUMBA_AVAILABLE), reason="Numba unavailable"
)
def test_reference_and_optimized_override_paths_match():
    params = _params()
    topology = RingTopology(num_lanes=2, length=50)
    actions = {0: LANE_LEFT}
    ref, ref_result = step_with_lateral_overrides_reference(
        _state(), params, topology, np.random.default_rng(44), actions
    )
    opt, opt_result, used_reference = step_with_lateral_overrides_optimized(
        _state(), params, topology, np.random.default_rng(44), actions
    )

    assert used_reference is False
    for field in ("lane", "pos", "vel", "changed_lane", "last_lane_delta"):
        np.testing.assert_array_equal(
            getattr(opt, field), getattr(ref, field), err_msg=field
        )
    np.testing.assert_array_equal(opt_result.applied, ref_result.applied)
