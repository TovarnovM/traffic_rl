import numpy as np
import pytest

from snfs_traffic.core import SimulationParams, TrafficState
from snfs_traffic.topology import RingTopology
from snfs_traffic.visualization import RoadRenderConfig, RoadRenderer


def test_renderer_smoke():
    pytest.importorskip("pygame")
    params = SimulationParams(num_lanes=2, road_length=20)
    topo = RingTopology(num_lanes=2, length=20)

    def mk(pos0, lane0, pos1, lane1, ch1=False):
        return TrafficState(
            vehicle_id=np.array([10, 11], dtype=np.int32),
            lane=np.array([lane0, lane1], dtype=np.int16),
            pos=np.array([pos0, pos1], dtype=np.int32),
            vel=np.array([1, 2], dtype=np.int16),
            length=np.array([1, 1], dtype=np.int16),
            veh_type=np.array([0, 1], dtype=np.int16),
            behavior_id=np.array([0, 0], dtype=np.int16),
            alive=np.array([True, True], dtype=np.bool_),
            last_lane_delta=np.array([0, 1 if ch1 else 0], dtype=np.int8),
            changed_lane=np.array([False, ch1], dtype=np.bool_),
            controlled=np.array([False, True], dtype=np.bool_),
        )

    s0 = mk(19, 0, 2, 0, False)
    s1 = mk(1, 0, 4, 1, True)
    r = RoadRenderer(params=params, topology=topo, config=RoadRenderConfig(width=320, height=180, interpolation_frames=3))
    r.reset(s0)
    frames = r.render_step(s1, step=1)
    assert len(frames) == 3
    for f in frames:
        assert f.shape == (180, 320, 3)
        assert f.dtype == np.uint8


def test_renderer_smoke_with_vehicle_ids():
    pytest.importorskip("pygame")
    params = SimulationParams(num_lanes=2, road_length=20)
    topo = RingTopology(num_lanes=2, length=20)

    def mk(pos0, lane0, pos1, lane1):
        return TrafficState(
            vehicle_id=np.array([10, 11], dtype=np.int32),
            lane=np.array([lane0, lane1], dtype=np.int16),
            pos=np.array([pos0, pos1], dtype=np.int32),
            vel=np.array([1, 2], dtype=np.int16),
            length=np.array([1, 1], dtype=np.int16),
            veh_type=np.array([0, 1], dtype=np.int16),
            behavior_id=np.array([0, 0], dtype=np.int16),
            alive=np.array([True, True], dtype=np.bool_),
            last_lane_delta=np.array([0, 0], dtype=np.int8),
            changed_lane=np.array([False, False], dtype=np.bool_),
            controlled=np.array([False, True], dtype=np.bool_),
        )

    s0 = mk(19, 0, 2, 0)
    s1 = mk(1, 0, 4, 1)
    r = RoadRenderer(
        params=params,
        topology=topo,
        config=RoadRenderConfig(
            width=320,
            height=180,
            interpolation_frames=3,
            draw_vehicle_ids=True,
        ),
    )
    r.reset(s0)
    frames = r.render_step(s1, step=1)
    assert len(frames) == 3
    for f in frames:
        assert f.shape == (180, 320, 3)
        assert f.dtype == np.uint8
