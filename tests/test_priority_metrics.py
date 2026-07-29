from __future__ import annotations

import numpy as np
import pytest

from snfs_traffic.control import LateralOverrideResult
from snfs_traffic.core import PRIORITY_VEH_TYPE, SimulationParams, empty_state
from snfs_traffic.metrics import PriorityMetricsAccumulator, paired_bootstrap_interval
from snfs_traffic.rules import PriorityYieldDecision


def test_priority_metrics_compute_flow_speeds_laps_and_rule_counts():
    params = SimulationParams(
        num_lanes=1, road_length=10, vmax_default=5, vmax_controlled=6
    )
    state = empty_state(2)
    state.veh_type[0] = PRIORITY_VEH_TYPE
    state.vel[:] = np.asarray([5, 3], dtype=state.vel.dtype)
    state.changed_lane[1] = True
    state.last_lane_delta[1] = 1
    decision = PriorityYieldDecision({1: -1}, (1,), (), ())
    result = LateralOverrideResult(
        vehicle_id=np.asarray([1], dtype=np.int32),
        requested_lane_delta=np.asarray([-1], dtype=np.int8),
        applied_lane_delta=np.asarray([-1], dtype=np.int8),
        valid=np.asarray([True], dtype=np.bool_),
        applied=np.asarray([True], dtype=np.bool_),
        rejection_reason=("",),
    )
    metrics = PriorityMetricsAccumulator(params)

    metrics.observe(state, decision=decision, result=result)
    metrics.observe(state, decision=decision, result=result)
    summary = metrics.summary()

    assert summary.measurement_steps == 2
    assert summary.density == pytest.approx(0.2)
    assert summary.flow_all == pytest.approx(0.8)
    assert summary.flow_background == pytest.approx(0.3)
    assert summary.flow_priority == pytest.approx(0.5)
    assert summary.mean_speed_all == pytest.approx(4.0)
    assert summary.mean_speed_background == pytest.approx(3.0)
    assert summary.mean_speed_priority == pytest.approx(5.0)
    assert summary.worst_priority_mean_speed == pytest.approx(5.0)
    assert summary.mean_priority_time_loss == pytest.approx(1.0 / 6.0)
    assert summary.completed_priority_laps == 1
    assert summary.mean_priority_lap_time == pytest.approx(2.0)
    assert summary.rule_detections == 2
    assert summary.rule_requests == 2
    assert summary.rule_applied == 2
    assert summary.rule_rejected == 0
    assert summary.lane_change_rate_background == pytest.approx(1.0)


def test_metrics_refuse_zero_steps():
    metrics = PriorityMetricsAccumulator(SimulationParams(num_lanes=1, road_length=10))

    with pytest.raises(ValueError, match="zero"):
        metrics.summary()


def test_paired_bootstrap_interval_is_reproducible_and_contains_mean():
    values = np.asarray([1.0, 2.0, 3.0, 4.0])

    first = paired_bootstrap_interval(values, samples=500, seed=12)
    second = paired_bootstrap_interval(values, samples=500, seed=12)

    assert first == second
    assert first[0] <= float(np.mean(values)) <= first[1]
