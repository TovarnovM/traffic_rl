import numpy as np
import pytest

from snfs_traffic.core import TrafficState
from snfs_traffic.visualization.road import (
    VehicleRenderState,
    interpolate_snapshots,
    select_follow_vehicle_id,
    snapshot_from_state,
)


def _state(ids, alive, lane, pos, vel, length, veh_type, behavior_id, controlled, changed_lane):
    n = len(ids)
    return TrafficState(
        vehicle_id=np.array(ids, dtype=np.int32),
        lane=np.array(lane, dtype=np.int16),
        pos=np.array(pos, dtype=np.int32),
        vel=np.array(vel, dtype=np.int16),
        length=np.array(length, dtype=np.int16),
        veh_type=np.array(veh_type, dtype=np.int16),
        behavior_id=np.array(behavior_id, dtype=np.int16),
        alive=np.array(alive, dtype=np.bool_),
        last_lane_delta=np.zeros(n, dtype=np.int8),
        changed_lane=np.array(changed_lane, dtype=np.bool_),
        controlled=np.array(controlled, dtype=np.bool_),
    )


def test_snapshot_filters_and_keys_by_vehicle_id():
    s = _state([101, 305, 77], [True, False, True], [0, 1, 2], [3, 4, 5], [1, 2, 3], [1, 1, 2], [0, 1, 2], [9, 8, 7], [False, True, True], [False, True, True])
    snap = snapshot_from_state(s)
    assert list(snap.keys()) == [101, 77]
    assert snap[77].lane == 2.0 and snap[77].pos == 5.0 and snap[77].vel == 3
    assert snap[77].length == 2 and snap[77].veh_type == 2 and snap[77].behavior_id == 7
    assert snap[77].controlled is True and snap[77].changed_lane is True


def test_interpolate_forward_cyclic_and_linear_lane():
    prev = {1: VehicleRenderState(1, 58.0, 0.0, 1, 1, 0, 0, False, False)}
    curr = {1: VehicleRenderState(1, 2.0, 2.0, 1, 2, 1, 1, True, True)}
    out = interpolate_snapshots(prev, curr, road_length=60, alpha=0.5)
    assert out[1].pos == pytest.approx(0.0)
    assert out[1].lane == pytest.approx(1.0)


def test_interpolate_not_shortest_signed_and_new_removed():
    prev = {1: VehicleRenderState(1, 2.0, 0.0, 1, 1, 0, 0, False, False), 2: VehicleRenderState(2, 10.0, 1.0, 1, 1, 0, 0, False, False)}
    curr = {1: VehicleRenderState(1, 58.0, 0.0, 1, 1, 0, 0, False, False), 3: VehicleRenderState(3, 7.0, 1.0, 1, 1, 0, 0, False, False)}
    out = interpolate_snapshots(prev, curr, road_length=60, alpha=0.5)
    assert out[1].pos == pytest.approx(30.0)
    assert 2 not in out
    assert out[3].pos == pytest.approx(7.0)


def test_select_follow_vehicle_id_behaviors():
    snap = {10: VehicleRenderState(10, 0, 0, 1, 1, 0, 0, False, False), 11: VehicleRenderState(11, 0, 0, 1, 1, 0, 0, True, False)}
    assert select_follow_vehicle_id(snap, follow_vehicle_id=10, follow="first-controlled", fallback_to_first_alive=True) == 10
    with pytest.raises(ValueError):
        select_follow_vehicle_id(snap, follow_vehicle_id=999, follow="first-controlled", fallback_to_first_alive=True)
    assert select_follow_vehicle_id(snap, follow_vehicle_id=None, follow="first-controlled", fallback_to_first_alive=True) == 11
    snap2 = {20: VehicleRenderState(20, 0, 0, 1, 1, 0, 0, False, False)}
    assert select_follow_vehicle_id(snap2, follow_vehicle_id=None, follow="first-controlled", fallback_to_first_alive=True) == 20
    assert select_follow_vehicle_id(snap2, follow_vehicle_id=None, follow="first-alive", fallback_to_first_alive=False) == 20
    assert select_follow_vehicle_id({}, follow_vehicle_id=None, follow="first-alive", fallback_to_first_alive=False) is None
