from __future__ import annotations
import numpy as np
from snfs_traffic.core.indexing import MISSING_INDEX

def target_lane_neighbors_at_pos_kernel(pos, lane_order, lane_counts, *, target_lane, candidate_pos, road_length):
    count = int(lane_counts[target_lane])
    if count == 0:
        return MISSING_INDEX, MISSING_INDEX, -1, -1
    front_idx = back_idx = MISSING_INDEX
    front_dist = back_dist = None
    for rank in range(count):
        idx = int(lane_order[target_lane, rank]); pos_j = int(pos[idx])
        df = int((pos_j - candidate_pos) % road_length); db = int((candidate_pos - pos_j) % road_length)
        if df > 0 and (front_dist is None or df < front_dist): front_dist, front_idx = df, idx
        if db > 0 and (back_dist is None or db < back_dist): back_dist, back_idx = db, idx
    if front_idx == MISSING_INDEX or back_idx == MISSING_INDEX:
        raise ValueError("target lane neighbor lookup failed")
    return front_idx, back_idx, int(front_dist - 1), int(back_dist - 1)

def eligible_target_lanes_for_vehicle_kernel(lane, pos, vel, length, alive, controlled, body_occupancy, lane_order, lane_counts, front_id, front_gap, *, vehicle_index, num_lanes, road_length, vmax_default, vmax_controlled):
    i=vehicle_index; lane_i=int(lane[i]); pos_i=int(pos[i]); v_i=int(vel[i]); l_i=int(length[i])
    vmax_i = int(vmax_controlled if controlled[i] else vmax_default)
    if int(front_id[i]) == MISSING_INDEX: g_pf=road_length-l_i; v_p_front=vmax_i
    else: g_pf=int(front_gap[i]); v_p_front=int(vel[int(front_id[i])])
    eligible=[]
    for lane_delta in (-1,1):
        target_lane=lane_i+lane_delta
        if target_lane<0 or target_lane>=num_lanes: continue
        if any(int(body_occupancy[target_lane,(pos_i+d)%road_length]) != MISSING_INDEX for d in range(l_i)): continue
        t_front, t_back, g_nf, g_nb = target_lane_neighbors_at_pos_kernel(pos,lane_order,lane_counts,target_lane=target_lane,candidate_pos=pos_i,road_length=road_length)
        if t_front == MISSING_INDEX:
            g_nf=road_length-l_i; v_n_front=vmax_i; safety=True
        else:
            v_n_front=int(vel[t_front]); v_n_back=int(vel[t_back]);
            g_nf = g_nf - l_i + 1
            g_nb = g_nb - int(length[t_back]) + 1
            safety = v_i > (v_n_back - g_nb)
        incentive = (g_nf + v_n_front > v_i) and (v_i >= g_pf + v_p_front)
        if incentive and safety: eligible.append(target_lane)
    return eligible

def collect_lane_change_proposals_kernel(lane,pos,vel,length,alive,controlled,body_occupancy,lane_order,lane_counts,front_id,front_gap,*,num_lanes,road_length,vmax_default,vmax_controlled,p_lane_change,rng):
    proposals={}
    for i in range(lane.shape[0]):
        if not alive[i]: continue
        eligible_targets = eligible_target_lanes_for_vehicle_kernel(lane,pos,vel,length,alive,controlled,body_occupancy,lane_order,lane_counts,front_id,front_gap,vehicle_index=i,num_lanes=num_lanes,road_length=road_length,vmax_default=vmax_default,vmax_controlled=vmax_controlled)
        if not eligible_targets: continue
        target_lane = eligible_targets[0] if len(eligible_targets)==1 else eligible_targets[int(rng.integers(0,len(eligible_targets)))]
        if float(rng.random()) >= p_lane_change: continue
        proposals.setdefault((target_lane,int(pos[i])),[]).append(i)
    return proposals

def resolve_lane_change_conflicts_kernel(proposals,rng):
    accepted={}
    for (target_lane,_), candidates in proposals.items():
        winner = candidates[0] if len(candidates)==1 else candidates[int(rng.integers(0,len(candidates)))]
        accepted[winner]=target_lane
    return accepted

def apply_lane_changes_kernel(lane,changed_lane,last_lane_delta,accepted):
    out_lane=lane.copy(); out_changed=np.zeros_like(changed_lane); out_delta=np.zeros_like(last_lane_delta)
    for i,target_lane in accepted.items():
        old=int(lane[i]); out_lane[i]=np.asarray(target_lane,dtype=out_lane.dtype); out_changed[i]=True; out_delta[i]=np.asarray(target_lane-old,dtype=out_delta.dtype)
    return out_lane,out_changed,out_delta
