import numpy as np

from snfs_traffic.core import SimulationParams, empty_state
from snfs_traffic.observations import LOCAL_OBSERVATION_FEATURES, build_local_observations
from snfs_traffic.topology import RingTopology


def test_no_controlled_empty_batch():
    p = SimulationParams(num_lanes=3, road_length=30)
    s = empty_state(2)
    s.controlled[:] = False
    s.pos[:] = [0, 1]
    out = build_local_observations(s, p, RingTopology(num_lanes=p.num_lanes, length=p.road_length))
    assert out.vehicle_id.shape == (0,)
    assert out.obs.shape == (0, 21)
    assert out.action_mask.shape == (0, 3)


def test_dtype_and_feature_count():
    p = SimulationParams(num_lanes=3, road_length=30)
    s = empty_state(1)
    s.controlled[:] = True
    out = build_local_observations(s, p, RingTopology(num_lanes=p.num_lanes, length=p.road_length))
    assert out.obs.dtype == np.float32
    assert out.obs.shape[1] == len(LOCAL_OBSERVATION_FEATURES)
