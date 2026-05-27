import numpy as np

from snfs_traffic.control import LaneActionBatch, LANE_STAY, compute_lateral_action_mask, normalize_lane_actions
from snfs_traffic.core import SimulationParams, empty_state
from snfs_traffic.topology import RingTopology


def test_lane_action_batch_order():
    b = LaneActionBatch.from_mapping({10: 1, 3: -1})
    np.testing.assert_array_equal(b.vehicle_id, np.array([10, 3], dtype=np.int32))


def test_normalize_fill_stay():
    s = empty_state(2)
    s.controlled[:] = True
    b = normalize_lane_actions({0: 1}, s, require_all_controlled=False)
    np.testing.assert_array_equal(b.lane_delta, np.array([1, LANE_STAY], dtype=np.int8))


def test_action_mask_stay_true():
    p = SimulationParams(num_lanes=2, road_length=10)
    s = empty_state(1)
    s.controlled[:] = True
    vids, mask = compute_lateral_action_mask(s, p, RingTopology(num_lanes=2, length=10))
    assert mask.shape == (1, 3)
    assert mask[0, 1]
