from __future__ import annotations

import numpy as np

from snfs_traffic.core.indexing import MISSING_INDEX

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _collect_candidates_numba(
        lane, pos, vel, alive, controlled, occupancy, lane_order, lane_counts, front_id, front_gap,
        num_lanes, road_length, vmax_default, vmax_controlled
    ):
        n = lane.shape[0]
        lane_delta = np.zeros((n,), dtype=np.int8)
        eligible_count = np.zeros((n,), dtype=np.int8)
        for i in range(n):
            if not alive[i]:
                continue
            lane_i = int(lane[i]); pos_i = int(pos[i]); v_i = int(vel[i])
            vmax_i = int(vmax_controlled if controlled[i] else vmax_default)
            fid = int(front_id[i])
            if fid == MISSING_INDEX:
                g_pf = road_length - 1; v_p_front = vmax_i
            else:
                g_pf = int(front_gap[i]); v_p_front = int(vel[fid])
            left_ok = False; right_ok = False
            for lane_delta_i in (-1, 1):
                target_lane = lane_i + lane_delta_i
                if target_lane < 0 or target_lane >= num_lanes:
                    continue
                if int(occupancy[target_lane, pos_i]) != MISSING_INDEX:
                    continue
                count = int(lane_counts[target_lane])
                t_front = MISSING_INDEX; t_back = MISSING_INDEX
                front_dist = road_length + 1; back_dist = road_length + 1
                for rank in range(count):
                    idx = int(lane_order[target_lane, rank])
                    pos_j = int(pos[idx])
                    dist_f = int((pos_j - pos_i) % road_length)
                    dist_b = int((pos_i - pos_j) % road_length)
                    if dist_f > 0 and dist_f < front_dist:
                        front_dist = dist_f; t_front = idx
                    if dist_b > 0 and dist_b < back_dist:
                        back_dist = dist_b; t_back = idx
                if t_front == MISSING_INDEX:
                    g_nf = road_length - 1; v_n_front = vmax_i; safety = True
                else:
                    g_nf = front_dist - 1; g_nb = back_dist - 1
                    v_n_front = int(vel[t_front]); v_n_back = int(vel[t_back])
                    safety = v_i > (v_n_back - g_nb)
                incentive = (g_nf + v_n_front > v_i) and (v_i >= g_pf + v_p_front)
                if incentive and safety:
                    if lane_delta_i < 0:
                        left_ok = True
                    else:
                        right_ok = True
            c = 0
            if left_ok: c += 1
            if right_ok: c += 1
            eligible_count[i] = c
            if c == 1:
                lane_delta[i] = -1 if left_ok else 1
            elif c == 2:
                lane_delta[i] = 2
        return lane_delta, eligible_count
else:
    def _collect_candidates_numba(*args, **kwargs):
        raise ImportError("Numba is not installed")


def collect_lane_change_proposals_numba(*, lane, pos, vel, alive, controlled, occupancy, lane_order, lane_counts, front_id, front_gap,
                                        num_lanes, road_length, vmax_default, vmax_controlled, p_lane_change, rng):
    if not NUMBA_AVAILABLE:
        raise ImportError("Numba is not installed")
    lane_delta, eligible_count = _collect_candidates_numba(
        lane, pos, vel, alive, controlled, occupancy, lane_order, lane_counts, front_id, front_gap,
        num_lanes, road_length, vmax_default, vmax_controlled
    )
    proposals: dict[tuple[int, int], list[int]] = {}
    for i in range(lane.shape[0]):
        c = int(eligible_count[i])
        if c <= 0:
            continue
        d = int(lane_delta[i])
        if c == 2:
            d = -1 if int(rng.integers(0, 2)) == 0 else 1
        if float(rng.random()) >= p_lane_change:
            continue
        target_lane = int(lane[i]) + d
        key = (target_lane, int(pos[i]))
        proposals.setdefault(key, []).append(i)
    return proposals
