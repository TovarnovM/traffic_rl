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
):
    count = int(lane_counts[target_lane])
    if count == 0:
        return MISSING_INDEX, MISSING_INDEX, -1, -1

    front_idx = MISSING_INDEX
    back_idx = MISSING_INDEX
    front_dist = None
    back_dist = None
    for rank in range(count):
        idx = int(lane_order[target_lane, rank])
        pos_j = int(pos[idx])
        dist_front = int((pos_j - candidate_pos) % road_length)
        dist_back = int((candidate_pos - pos_j) % road_length)
        if dist_front > 0 and (front_dist is None or dist_front < front_dist):
            front_dist = dist_front
            front_idx = idx
        if dist_back > 0 and (back_dist is None or dist_back < back_dist):
            back_dist = dist_back
            back_idx = idx

    if front_idx == MISSING_INDEX or back_idx == MISSING_INDEX:
        raise ValueError("target lane neighbor lookup failed")
    return front_idx, back_idx, int(front_dist - 1), int(back_dist - 1)


def _candidate_body_cells(target_lane: int, head_pos: int, vehicle_length: int, road_length: int) -> set[tuple[int, int]]:
    return {(target_lane, (head_pos + d) % road_length) for d in range(vehicle_length)}


def eligible_target_lanes_for_vehicle_kernel(
    lane: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    length: np.ndarray,
    alive: np.ndarray,
    controlled: np.ndarray,
    body_occupancy: np.ndarray,
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
):
    i = int(vehicle_index)
    lane_i = int(lane[i])
    pos_i = int(pos[i])
    vel_i = int(vel[i])
    length_i = int(length[i])
    vmax_i = int(vmax_controlled if controlled[i] else vmax_default)

    if int(front_id[i]) == MISSING_INDEX:
        present_front_gap = road_length - length_i
        present_front_velocity = vmax_i
    else:
        present_front = int(front_id[i])
        present_front_gap = int((int(pos[present_front]) - pos_i - length_i) % road_length)
        present_front_velocity = int(vel[present_front])

    eligible = []
    for lane_delta in (-1, 1):
        target_lane = lane_i + lane_delta
        if target_lane < 0 or target_lane >= num_lanes:
            continue

        body_blocked = False
        for d in range(length_i):
            target_cell = (pos_i + d) % road_length
            if int(body_occupancy[target_lane, target_cell]) != MISSING_INDEX:
                body_blocked = True
                break
        if body_blocked:
            continue

        target_front, target_back, target_front_gap, target_back_gap = target_lane_neighbors_at_pos_kernel(
            pos,
            lane_order,
            lane_counts,
            target_lane=target_lane,
            candidate_pos=pos_i,
            road_length=road_length,
        )
        if target_front == MISSING_INDEX:
            target_front_gap = road_length - length_i
            target_front_velocity = vmax_i
            safety = True
        else:
            target_front_velocity = int(vel[target_front])
            target_back_velocity = int(vel[target_back])
            # Empty cells from candidate body tail to target front head.
            target_front_gap = target_front_gap - length_i + 1
            # Empty cells from target back body tail to candidate head.
            target_back_gap = target_back_gap - int(length[target_back]) + 1
            safety = vel_i > (target_back_velocity - target_back_gap)

        incentive = (target_front_gap + target_front_velocity > vel_i) and (
            vel_i >= present_front_gap + present_front_velocity
        )
        if incentive and safety:
            eligible.append(target_lane)
    return eligible


def collect_lane_change_proposals_kernel(
    lane: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    length: np.ndarray,
    alive: np.ndarray,
    controlled: np.ndarray,
    body_occupancy: np.ndarray,
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
):
    proposals = {}
    for i in range(lane.shape[0]):
        if not alive[i]:
            continue
        eligible_targets = eligible_target_lanes_for_vehicle_kernel(
            lane,
            pos,
            vel,
            length,
            alive,
            controlled,
            body_occupancy,
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
        if len(eligible_targets) == 1:
            target_lane = eligible_targets[0]
        else:
            target_lane = eligible_targets[int(rng.integers(0, len(eligible_targets)))]
        if float(rng.random()) >= p_lane_change:
            continue
        proposals.setdefault((target_lane, int(pos[i])), []).append(i)
    return proposals


def resolve_lane_change_conflicts_kernel(
    proposals,
    rng,
    *,
    pos: np.ndarray | None = None,
    length: np.ndarray | None = None,
    road_length: int | None = None,
    reserved_body_cells: set[tuple[int, int]] | None = None,
):
    accepted = {}
    reserved = set() if reserved_body_cells is None else set(reserved_body_cells)

    for (target_lane, head_pos), candidates in proposals.items():
        if len(candidates) == 1:
            winner = int(candidates[0])
        else:
            winner = int(candidates[int(rng.integers(0, len(candidates)))])

        if pos is None or length is None or road_length is None:
            accepted[winner] = target_lane
            continue

        cells = _candidate_body_cells(target_lane, int(pos[winner]), int(length[winner]), int(road_length))
        if cells.isdisjoint(reserved):
            accepted[winner] = target_lane
            reserved.update(cells)

    return accepted


def apply_lane_changes_kernel(lane, changed_lane, last_lane_delta, accepted):
    out_lane = lane.copy()
    out_changed = np.zeros_like(changed_lane)
    out_delta = np.zeros_like(last_lane_delta)
    for i, target_lane in accepted.items():
        old_lane = int(lane[i])
        out_lane[i] = np.asarray(target_lane, dtype=out_lane.dtype)
        out_changed[i] = True
        out_delta[i] = np.asarray(target_lane - old_lane, dtype=out_delta.dtype)
    return out_lane, out_changed, out_delta
