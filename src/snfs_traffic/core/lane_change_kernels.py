from __future__ import annotations

import numpy as np

from snfs_traffic.core.indexing import MISSING_INDEX


def target_lane_neighbors_at_pos_kernel(
    pos: np.ndarray,
    lane_order: np.ndarray,
    lane_counts: np.ndarray,
    *,
    target_lane: int,
    candidate_pos: int,
    road_length: int,
) -> tuple[int, int, int, int]:
    count = int(lane_counts[target_lane])
    if count == 0:
        return MISSING_INDEX, MISSING_INDEX, -1, -1

    front_idx = MISSING_INDEX
    front_dist = None
    back_idx = MISSING_INDEX
    back_dist = None

    for rank in range(count):
        idx = int(lane_order[target_lane, rank])
        pos_j = int(pos[idx])

        dist_f = int((pos_j - candidate_pos) % road_length)
        dist_b = int((candidate_pos - pos_j) % road_length)

        if dist_f > 0 and (front_dist is None or dist_f < front_dist):
            front_dist = dist_f
            front_idx = idx

        if dist_b > 0 and (back_dist is None or dist_b < back_dist):
            back_dist = dist_b
            back_idx = idx

    if front_idx == MISSING_INDEX or back_idx == MISSING_INDEX:
        raise ValueError("target lane neighbor lookup failed")

    g_nf = int(front_dist - 1)
    g_nb = int(back_dist - 1)
    return front_idx, back_idx, g_nf, g_nb


def eligible_target_lanes_for_vehicle_kernel(
    lane: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    alive: np.ndarray,
    controlled: np.ndarray,
    occupancy: np.ndarray,
    lane_order: np.ndarray,
    lane_counts: np.ndarray,
    front_id: np.ndarray,
    front_gap: np.ndarray,
    *,
    vehicle_index: int,
    num_lanes: int,
    road_length: int,
    vmax_default: int,
    vmax_controlled: int,
) -> list[int]:
    i = vehicle_index
    lane_i = int(lane[i])
    pos_i = int(pos[i])
    v_i = int(vel[i])

    vmax_i = int(vmax_controlled if controlled[i] else vmax_default)

    if int(front_id[i]) == MISSING_INDEX:
        g_pf = road_length - 1
        v_p_front = vmax_i
    else:
        g_pf = int(front_gap[i])
        v_p_front = int(vel[int(front_id[i])])

    eligible: list[int] = []
    for lane_delta in (-1, 1):
        target_lane = lane_i + lane_delta

        if target_lane < 0 or target_lane >= num_lanes:
            continue

        if int(occupancy[target_lane, pos_i]) != MISSING_INDEX:
            continue

        t_front, t_back, g_nf, g_nb = target_lane_neighbors_at_pos_kernel(
            pos,
            lane_order,
            lane_counts,
            target_lane=target_lane,
            candidate_pos=pos_i,
            road_length=road_length,
        )

        if t_front == MISSING_INDEX:
            g_nf = road_length - 1
            v_n_front = vmax_i
            safety = True
        else:
            v_n_front = int(vel[t_front])
            v_n_back = int(vel[t_back])
            safety = v_i > (v_n_back - g_nb)

        incentive = (g_nf + v_n_front > v_i) and (v_i >= g_pf + v_p_front)

        if incentive and safety:
            eligible.append(target_lane)

    return eligible


def collect_lane_change_proposals_kernel(
    lane: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    alive: np.ndarray,
    controlled: np.ndarray,
    occupancy: np.ndarray,
    lane_order: np.ndarray,
    lane_counts: np.ndarray,
    front_id: np.ndarray,
    front_gap: np.ndarray,
    *,
    num_lanes: int,
    road_length: int,
    vmax_default: int,
    vmax_controlled: int,
    p_lane_change: float,
    rng,
) -> dict[tuple[int, int], list[int]]:
    proposals: dict[tuple[int, int], list[int]] = {}

    for i in range(lane.shape[0]):
        if not alive[i]:
            continue

        eligible_targets = eligible_target_lanes_for_vehicle_kernel(
            lane,
            pos,
            vel,
            alive,
            controlled,
            occupancy,
            lane_order,
            lane_counts,
            front_id,
            front_gap,
            vehicle_index=i,
            num_lanes=num_lanes,
            road_length=road_length,
            vmax_default=vmax_default,
            vmax_controlled=vmax_controlled,
        )

        if not eligible_targets:
            continue

        target_lane = eligible_targets[0]
        if len(eligible_targets) > 1:
            target_lane = eligible_targets[int(rng.integers(0, len(eligible_targets)))]

        if float(rng.random()) >= p_lane_change:
            continue

        key = (target_lane, int(pos[i]))
        proposals.setdefault(key, []).append(i)

    return proposals


def resolve_lane_change_conflicts_kernel(
    proposals: dict[tuple[int, int], list[int]],
    rng,
) -> dict[int, int]:
    accepted: dict[int, int] = {}
    for (target_lane, _pos), candidates in proposals.items():
        if len(candidates) == 1:
            winner = candidates[0]
        else:
            winner = candidates[int(rng.integers(0, len(candidates)))]
        accepted[winner] = target_lane
    return accepted


def apply_lane_changes_kernel(
    lane: np.ndarray,
    changed_lane: np.ndarray,
    last_lane_delta: np.ndarray,
    accepted: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_lane = lane.copy()
    out_changed_lane = np.zeros_like(changed_lane, dtype=changed_lane.dtype)
    out_last_lane_delta = np.zeros_like(last_lane_delta, dtype=last_lane_delta.dtype)

    for i, target_lane in accepted.items():
        old_lane = int(lane[i])
        out_lane[i] = np.asarray(target_lane, dtype=out_lane.dtype)
        out_changed_lane[i] = True
        out_last_lane_delta[i] = np.asarray(target_lane - old_lane, dtype=out_last_lane_delta.dtype)

    return out_lane, out_changed_lane, out_last_lane_delta
