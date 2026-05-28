import numpy as np

from snfs_traffic.core.indexing import build_body_occupancy, build_lane_order, build_occupancy, compute_neighbors
from snfs_traffic.core.lane_change_kernels import collect_lane_change_proposals_kernel, eligible_target_lanes_for_vehicle_kernel, resolve_lane_change_conflicts_kernel
from snfs_traffic.core.params import SimulationParams
from snfs_traffic.core.state import TrafficState
from snfs_traffic.topology import RingTopology


def _state(lane,pos,vel,length):
    n=len(lane)
    return TrafficState(np.arange(n,dtype=np.int32),np.array(lane,dtype=np.int16),np.array(pos,dtype=np.int32),np.array(vel,dtype=np.int16),np.array(length,dtype=np.int16),np.zeros(n,dtype=np.int16),np.zeros(n,dtype=np.int16),np.ones(n,dtype=np.bool_),np.zeros(n,dtype=np.int8),np.zeros(n,dtype=np.bool_),np.zeros(n,dtype=np.bool_))


def test_length3_cannot_change_into_occupied_target_body_cell():
    s=_state([1,0],[5,6],[2,0],[3,1])
    p=SimulationParams(num_lanes=2,road_length=20)
    occ=build_occupancy(s,p); body=build_body_occupancy(s,p); lo,lc,lr=build_lane_order(occ,n_vehicles=s.n_vehicles)
    fid,_,fg,_=compute_neighbors(s,lo,lc,lr,RingTopology(num_lanes=2,length=20))
    e=eligible_target_lanes_for_vehicle_kernel(s.lane,s.pos,s.vel,s.length,s.alive,s.controlled,body,lo,lc,fid,fg,vehicle_index=0,num_lanes=2,road_length=20,vmax_default=5,vmax_controlled=6)
    assert e==[]


def test_collect_proposals_uses_new_signature():
    s=_state([1,1],[5,10],[2,0],[3,1])
    p=SimulationParams(num_lanes=3,road_length=20,p_lane_change=1.0)
    occ=build_occupancy(s,p); body=build_body_occupancy(s,p); lo,lc,lr=build_lane_order(occ,n_vehicles=s.n_vehicles)
    fid,_,fg,_=compute_neighbors(s,lo,lc,lr,RingTopology(num_lanes=3,length=20))
    proposals=collect_lane_change_proposals_kernel(s.lane,s.pos,s.vel,s.length,s.alive,s.controlled,body,lo,lc,fid,fg,num_lanes=3,road_length=20,vmax_default=5,vmax_controlled=6,p_lane_change=1.0,rng=np.random.default_rng(0))
    assert isinstance(proposals,dict)


def test_conflict_resolution_rejects_overlapping_target_bodies_with_distinct_heads():
    pos = np.array([5, 6], dtype=np.int32)
    length = np.array([3, 3], dtype=np.int16)
    proposals = {(1, 5): [0], (1, 6): [1]}

    accepted = resolve_lane_change_conflicts_kernel(
        proposals,
        np.random.default_rng(0),
        pos=pos,
        length=length,
        road_length=20,
    )

    assert len(accepted) == 1
    assert set(accepted) in ({0}, {1})


def test_conflict_resolution_respects_reserved_controlled_body_cells():
    pos = np.array([6], dtype=np.int32)
    length = np.array([3], dtype=np.int16)
    proposals = {(1, 6): [0]}
    reserved = {(1, 5), (1, 6), (1, 7)}

    accepted = resolve_lane_change_conflicts_kernel(
        proposals,
        np.random.default_rng(0),
        pos=pos,
        length=length,
        road_length=20,
        reserved_body_cells=reserved,
    )

    assert accepted == {}
