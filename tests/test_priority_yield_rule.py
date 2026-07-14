from __future__ import annotations

import numpy as np

from snfs_traffic.control import LateralOverrideResult
from snfs_traffic.core import PRIORITY_VEH_TYPE, SimulationParams, empty_state
from snfs_traffic.rules import PriorityYieldConfig, PriorityYieldController
from snfs_traffic.topology import RingTopology


def _world():
    params = SimulationParams(num_lanes=3, road_length=100, p_lane_change=0.0)
    topology = RingTopology(num_lanes=3, length=100)
    state = empty_state(2)
    state.lane[:] = 1
    state.pos[:] = np.asarray([20, 10], dtype=state.pos.dtype)
    state.vel[:] = np.asarray([3, 6], dtype=state.vel.dtype)
    state.veh_type[1] = PRIORITY_VEH_TYPE
    return state, params, topology


def test_rule_requests_lane_change_only_for_vehicle_with_pv_behind():
    state, params, topology = _world()
    controller = PriorityYieldController(PriorityYieldConfig(detection_distance=20))

    decision = controller.decide(state, params, topology)

    assert decision.detected_vehicle_ids == (0,)
    assert set(decision.overrides) == {0}
    assert decision.overrides[0] in {-1, 1}
    assert decision.blocked_vehicle_ids == ()


def test_rule_does_not_trigger_when_pv_is_outside_detection_range():
    state, params, topology = _world()
    state.pos[1] = 80
    controller = PriorityYieldController(PriorityYieldConfig(detection_distance=20))

    decision = controller.decide(state, params, topology)

    assert decision.detected_vehicle_ids == ()
    assert dict(decision.overrides) == {}


def test_rule_defers_to_native_model_when_no_adjacent_lane_is_safe():
    state, params, topology = _world()
    expanded = empty_state(4)
    expanded.lane[:] = np.asarray([1, 1, 0, 2], dtype=expanded.lane.dtype)
    expanded.pos[:] = np.asarray([20, 10, 20, 20], dtype=expanded.pos.dtype)
    expanded.vel[:] = np.asarray([3, 6, 3, 3], dtype=expanded.vel.dtype)
    expanded.veh_type[1] = PRIORITY_VEH_TYPE
    controller = PriorityYieldController(PriorityYieldConfig(detection_distance=20))

    decision = controller.decide(expanded, params, topology)

    assert decision.detected_vehicle_ids == (0,)
    assert decision.blocked_vehicle_ids == (0,)
    assert dict(decision.overrides) == {}


def test_matched_eligibility_limits_who_can_yield():
    state, params, topology = _world()
    controller = PriorityYieldController(eligible_vehicle_ids=[])

    decision = controller.decide(state, params, topology)

    assert dict(decision.overrides) == {}


def test_cooldown_suppresses_exact_number_of_future_decisions():
    state, params, topology = _world()
    controller = PriorityYieldController(
        PriorityYieldConfig(detection_distance=20, cooldown_steps=3)
    )
    first = controller.decide(state, params, topology)
    delta = first.overrides[0]
    controller.observe(
        LateralOverrideResult(
            vehicle_id=np.asarray([0], dtype=np.int32),
            requested_lane_delta=np.asarray([delta], dtype=np.int8),
            applied_lane_delta=np.asarray([delta], dtype=np.int8),
            valid=np.asarray([True], dtype=np.bool_),
            applied=np.asarray([True], dtype=np.bool_),
            rejection_reason=("",),
        )
    )

    suppressed = [controller.decide(state, params, topology) for _ in range(3)]
    available_again = controller.decide(state, params, topology)

    assert all(decision.cooldown_vehicle_ids == (0,) for decision in suppressed)
    assert all(dict(decision.overrides) == {} for decision in suppressed)
    assert set(available_again.overrides) == {0}
