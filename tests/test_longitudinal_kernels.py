import numpy as np

from snfs_traffic.core import SimulationParams, build_lane_order, build_occupancy
from snfs_traffic.core.longitudinal_kernels import compute_longitudinal_velocities_kernel
from snfs_traffic.core.state import TrafficState


class FakeRng:
    def __init__(self, vals): self.vals=list(vals); self.calls=0
    def random(self): v=self.vals[self.calls]; self.calls+=1; return v


def _state(lane,pos,vel,length=None,alive=None,controlled=None):
    n=len(lane)
    return TrafficState(
        vehicle_id=np.arange(n,dtype=np.int32), lane=np.array(lane,dtype=np.int16), pos=np.array(pos,dtype=np.int32),
        vel=np.array(vel,dtype=np.int16), length=np.array(length or [1]*n,dtype=np.int16), veh_type=np.zeros(n,dtype=np.int16),
        behavior_id=np.zeros(n,dtype=np.int16), alive=np.array(alive if alive is not None else [True]*n,dtype=np.bool_),
        last_lane_delta=np.zeros(n,dtype=np.int8), changed_lane=np.zeros(n,dtype=np.bool_), controlled=np.array(controlled if controlled is not None else [False]*n,dtype=np.bool_)
    )


def _run(s, **kwargs):
    rng = kwargs.pop("rng", np.random.default_rng(0))
    p = SimulationParams(num_lanes=kwargs.pop('num_lanes',1), road_length=kwargs.pop('road_length',20), **kwargs)
    occ = build_occupancy(s,p); lo,lc,lr = build_lane_order(occ,n_vehicles=s.n_vehicles)
    return compute_longitudinal_velocities_kernel(s.lane,s.pos,s.vel,s.length,s.alive,s.controlled,lo,lc,lr,road_length=p.road_length,vmax_default=p.vmax_default,vmax_controlled=p.vmax_controlled,G=p.G,q=p.q,r=p.r,S=p.S,P1=p.P1,P2=p.P2,P3=p.P3,P4=p.P4,rng=rng)


def test_rng_contract_three_draws_per_alive():
    s=_state([0,0,0],[0,5,10],[0,0,0],alive=[True,False,True])
    rng=FakeRng([0.1,0.2,0.3,0.4,0.5,0.6])
    _run(s,rng=rng)
    assert rng.calls==6


def test_q_affects_slow_to_start():
    s=_state([0,0],[0,4],[3,0])
    rng=FakeRng([0.9,0.0,0.9, 0.9,0.0,0.9])
    vq1=_run(s,q=1.0,r=0.0,P1=1.0,P2=1.0,P3=1.0,P4=1.0,rng=rng)
    rng=FakeRng([0.9,0.9,0.9, 0.9,0.9,0.9])
    vq0=_run(s,q=0.0,r=0.0,P1=1.0,P2=1.0,P3=1.0,P4=1.0,rng=rng)
    assert int(vq1[0]) <= int(vq0[0])


def test_p1_branch_active():
    s=_state([0],[0],[1])
    rng=FakeRng([0.9,0.9,0.0])
    vb=_run(s,P1=0.0,P2=1.0,P3=1.0,P4=1.0,rng=rng)
    rng=FakeRng([0.9,0.9,0.0])
    vn=_run(s,P1=1.0,P2=1.0,P3=1.0,P4=1.0,rng=rng)
    assert int(vb[0]) < int(vn[0])


def test_stopped_vehicle_not_accelerated_by_brake_rule():
    s=_state([0,0],[0,1],[0,0])
    rng=FakeRng([0.9,0.9,0.0, 0.9,0.9,0.0])
    # Keep rule3/current-gap clipping local to the immediately adjacent leader.
    out=_run(s,P1=1.0,P2=1.0,P3=0.0,P4=1.0,S=1,rng=rng)
    assert int(out[0]) == 0
