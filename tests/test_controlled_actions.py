import numpy as np
import pytest

from snfs_traffic.control import (
    LANE_STAY,
    LaneActionBatch,
    compute_lateral_action_mask,
    normalize_lane_actions,
    step_with_controlled_lateral_actions_reference,
)
from snfs_traffic.core import SimulationParams, empty_state
from snfs_traffic.simulator import TrafficSimulator
from snfs_traffic.topology import RingTopology


def test_lane_action_batch_rejects_non_contiguous_and_dtype():
    with pytest.raises(ValueError):
        LaneActionBatch(np.array([1, 2], dtype=np.int64), np.array([0, 1], dtype=np.int8))
    with pytest.raises(ValueError):
        LaneActionBatch(np.array([1, 2], dtype=np.int32), np.array([0, 1], dtype=np.int16))
    with pytest.raises(ValueError):
        LaneActionBatch(np.array([1, 2, 3, 4], dtype=np.int32)[::2], np.array([0, 1], dtype=np.int8))


def test_lane_action_from_mapping_overflow_value_error():
    with pytest.raises(ValueError):
        LaneActionBatch.from_mapping({1: 128})


def test_normalize_lane_actions_validation_and_fill_stay():
    s = empty_state(3)
    s.controlled[:] = [True, True, False]
    s.alive[:] = [True, False, True]
    with pytest.raises(ValueError):
        normalize_lane_actions({9: 0}, s, require_all_controlled=True)
    with pytest.raises(ValueError):
        normalize_lane_actions({2: 0}, s, require_all_controlled=True)  # uncontrolled
    with pytest.raises(ValueError):
        normalize_lane_actions({1: 0}, s, require_all_controlled=True)  # non-alive
    with pytest.raises(ValueError):
        normalize_lane_actions({}, s, require_all_controlled=True)

    s2 = empty_state(2)
    s2.controlled[:] = [True, True]
    out = normalize_lane_actions({0: 1}, s2, require_all_controlled=False)
    np.testing.assert_array_equal(out.lane_delta, np.array([1, LANE_STAY], dtype=np.int8))


def test_action_mask_rejects_mismatched_topology_and_occupied_target_invalid():
    p = SimulationParams(num_lanes=2, road_length=10)
    s = empty_state(2)
    s.controlled[:] = [True, False]
    s.lane[:] = [0, 1]
    s.pos[:] = [2, 2]
    with pytest.raises(ValueError):
        compute_lateral_action_mask(s, p, RingTopology(num_lanes=3, length=10))
    _, mask = compute_lateral_action_mask(s, p, RingTopology(num_lanes=2, length=10))
    assert bool(mask[0, 2]) is False


def test_conflict_and_rejection_reasons_and_valid_apply():
    p = SimulationParams(num_lanes=3, road_length=10, p_lane_change=0.0)
    t = RingTopology(num_lanes=3, length=10)
    s = empty_state(3)
    s.controlled[:] = [True, True, False]
    s.lane[:] = [0, 2, 1]
    s.pos[:] = [4, 4, 8]
    _, res = step_with_controlled_lateral_actions_reference(s, p, t, np.random.default_rng(2), {0: +1, 1: -1})
    assert res.rejection_reason == ("controlled_conflict", "controlled_conflict")
    _, res2 = step_with_controlled_lateral_actions_reference(s, p, t, np.random.default_rng(3), {0: -1, 1: 0})
    assert res2.rejection_reason[0] == "out_of_bounds"

    s3 = empty_state(2)
    s3.controlled[:] = [True, False]
    s3.lane[:] = [0, 1]
    s3.pos[:] = [2, 2]
    _, res3 = step_with_controlled_lateral_actions_reference(s3, SimulationParams(num_lanes=2, road_length=10, p_lane_change=0.0), RingTopology(num_lanes=2, length=10), np.random.default_rng(4), {0: +1})
    assert res3.rejection_reason[0] == "target_occupied"

    s4 = empty_state(2)
    s4.controlled[:] = [True, False]
    s4.lane[:] = [0, 2]
    s4.pos[:] = [2, 6]
    n4, _ = step_with_controlled_lateral_actions_reference(s4, p, t, np.random.default_rng(5), {0: +1})
    assert bool(n4.changed_lane[0]) is True
    assert int(n4.last_lane_delta[0]) == 1


def test_facade_step_info_action_path_flags():
    p = SimulationParams(num_lanes=3, road_length=30)
    sim = TrafficSimulator(params=p)
    sim.reset()
    sim.step(None)
    assert sim.last_step_info.used_reference_action_path is False
    ids = sim.last_step_info.controlled_vehicle_ids
    action_map = {int(v): 0 for v in ids}
    sim.step(action_map)
    assert sim.last_step_info.used_reference_action_path is True


def test_controlled_reserved_body_cells_block_overlapping_uncontrolled_proposal(monkeypatch):
    p = SimulationParams(num_lanes=3, road_length=20, p_lane_change=1.0)
    t = RingTopology(num_lanes=3, length=20)
    s = empty_state(2)
    s.controlled[:] = [True, False]
    s.lane[:] = [0, 2]
    s.pos[:] = [5, 6]
    s.length[:] = [3, 3]

    def fake_uncontrolled_proposals(*args, **kwargs):
        return {(1, 6): [1]}

    monkeypatch.setattr("snfs_traffic.control.collect_lane_change_proposals_kernel", fake_uncontrolled_proposals)

    out, result = step_with_controlled_lateral_actions_reference(s, p, t, np.random.default_rng(0), {0: +1})

    assert bool(result.applied[0]) is True
    assert int(out.lane[0]) == 1
    assert int(out.lane[1]) == 2
