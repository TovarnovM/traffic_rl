"""Internal pure-array kernels for full Revised S-NFS longitudinal phase."""

from __future__ import annotations
import numpy as np
from snfs_traffic.core.indexing_kernels import compute_cumulative_forward_gap_kernel, compute_forward_empty_gap_kernel


def compute_longitudinal_velocities_kernel(
    lane: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    length: np.ndarray,
    alive: np.ndarray,
    controlled: np.ndarray,
    lane_order: np.ndarray,
    lane_counts: np.ndarray,
    lane_rank: np.ndarray,
    *,
    road_length: int,
    vmax_default: int,
    vmax_controlled: int,
    G: int,
    q: float,
    r: float,
    S: int,
    P1: float,
    P2: float,
    P3: float,
    P4: float,
    rng,
) -> np.ndarray:
    n = vel.shape[0]
    out = vel.astype(np.int64, copy=True)
    v0 = vel.astype(np.int64, copy=False)
    prev_pos = (pos.astype(np.int64) - v0) % int(road_length)
    v4 = v0.copy()
    gap1 = np.full((n,), -1, dtype=np.int64)
    front = np.full((n,), -1, dtype=np.int64)

    for i in range(n):
        if not alive[i]:
            continue
        li = int(lane[i]); count = int(lane_counts[li]); rank = int(lane_rank[i])
        vmax_i = int(vmax_controlled if controlled[i] else vmax_default)
        if count <= 1:
            leader = -1
            g = int(road_length - int(length[i]))
            leader_v = int(v0[i])
        else:
            leader = int(lane_order[li, (rank + 1) % count])
            g = int((int(pos[leader]) - int(pos[i]) - int(length[i])) % road_length)
            leader_v = int(v0[leader])
        front[i] = leader; gap1[i] = g

        u_s = float(rng.random()); u_q = float(rng.random()); u_b = float(rng.random())
        s_i = int(S if u_s < float(r) else 1)

        v1 = min(vmax_i, int(v0[i]) + 1) if (g > int(G) or int(v0[i]) < leader_v) else int(v0[i])

        prev_gap = compute_cumulative_forward_gap_kernel(i, s_i, prev_pos, length, lane_order, lane_counts, lane_rank, road_length=road_length)
        v2 = min(v1, int(prev_gap)) if u_q < float(q) else v1

        cur_gap = compute_cumulative_forward_gap_kernel(i, s_i, pos, length, lane_order, lane_counts, lane_rank, road_length=road_length)
        v3 = min(v2, int(cur_gap))

        if g > int(G):
            p_i = float(P1)
        elif int(v0[i]) < leader_v:
            p_i = float(P2)
        elif int(v0[i]) == leader_v:
            p_i = float(P3)
        else:
            p_i = float(P4)

        if u_b < (1.0 - p_i):
            v4_i = max(v3 - 1, 1) if v3 > 0 else 0
        else:
            v4_i = v3
        v4[i] = max(0, min(vmax_i, int(v4_i)))

    v_safe = v4.copy()
    for l in range(lane_order.shape[0]):
        count = int(lane_counts[l])
        if count <= 1:
            continue
        for _ in range(count):
            for rr in range(count - 1, -1, -1):
                i = int(lane_order[l, rr]); j = int(lane_order[l, (rr + 1) % count])
                if not alive[i]:
                    continue
                g = int((int(pos[j]) - int(pos[i]) - int(length[i])) % road_length)
                v_safe[i] = min(int(v_safe[i]), g + int(v_safe[j]))

    out[alive] = np.maximum(0, v_safe[alive])
    return out.astype(vel.dtype, copy=False)


def advance_positions_kernel(pos: np.ndarray, vel: np.ndarray, alive: np.ndarray, *, road_length: int) -> np.ndarray:
    out = pos.copy()
    for i in range(pos.shape[0]):
        if alive[i]:
            out[i] = (int(pos[i]) + int(vel[i])) % road_length
    return out
