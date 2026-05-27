import numpy as np
import pytest

from snfs_traffic.control import (
    LaneActionBatch,
    compute_lateral_action_mask,
    step_with_controlled_lateral_actions_reference,
)
from snfs_traffic.core import SimulationParams, empty_state
from snfs_traffic.topology import RingTopology


def test_lane_action_batch_rejects_non_contiguous_and_dtype():
    with pytest.raises(ValueError):
        LaneActionBatch(np.array([1, 2], dtype=np.int64), np.array([0, 1], dtype=np.int8))
    with pytest.raises(ValueError):
        LaneActionBatch(np.array([1, 2], dtype=np.int32), np.array([0, 1], dtype=np.int16))
    arr = np.array([1, 2, 3, 4], dtype=np.int32)[::2]
    with pytest.raises(ValueError):
        LaneActionBatch(arr, np.array([0, 1], dtype=np.int8))


def test_lane_action_from_mapping_overflow_value_error():
    with pytest.raises(ValueError):
        LaneActionBatch.from_mapping({1: 128})


def test_action_mask_rejects_mismatched_topology():
    p = SimulationParams(num_lanes=2, road_length=10)
    s = empty_state(1)
    s.controlled[:] = True
    with pytest.raises(ValueError):
        compute_lateral_action_mask(s, p, RingTopology(num_lanes=3, length=10))


def test_occupied_target_invalid_in_mask_and_reasons():
    p = SimulationParams(num_lanes=2, road_length=10, p_lane_change=0.0)
    t = RingTopology(num_lanes=2, length=10)
    s = empty_state(2)
    s.controlled[:] = [True, False]
    s.lane[:] = [0, 1]
    s.pos[:] = [2, 2]
    vids, mask = compute_lateral_action_mask(s, p, t)
    assert vids.tolist() == [0]
    assert mask[0, 2] == False
    _, res = step_with_controlled_lateral_actions_reference(s, p, t, np.random.default_rng(1), {0: +1})
    assert res.rejection_reason[0] == "target_occupied"


def test_controlled_conflict_rejects_all_and_oob_reason():
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
