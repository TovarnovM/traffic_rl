"""Internal pure-array indexing kernels.

These kernels are backend-neutral and operate only on NumPy arrays/scalars.
"""

from __future__ import annotations

import numpy as np

INDEX_DTYPE = np.int32
MISSING_INDEX = -1
MISSING_GAP = -1


def build_occupancy_kernel(
    lane: np.ndarray,
    pos: np.ndarray,
    alive: np.ndarray,
    *,
    num_lanes: int,
    road_length: int,
) -> np.ndarray:
    occupancy = np.full((num_lanes, road_length), MISSING_INDEX, dtype=INDEX_DTYPE)

    n_vehicles = int(lane.shape[0])
    for i in range(n_vehicles):
        if not alive[i]:
            continue
        lane_i = int(lane[i])
        pos_i = int(pos[i])
        if occupancy[lane_i, pos_i] != MISSING_INDEX:
            raise ValueError(f"duplicate occupied head cell at lane={lane_i}, pos={pos_i}")
        occupancy[lane_i, pos_i] = i

    return occupancy


def build_lane_order_kernel(occupancy: np.ndarray, *, n_vehicles: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_lanes, road_length = occupancy.shape
    lane_order = np.full((num_lanes, road_length), MISSING_INDEX, dtype=INDEX_DTYPE)
    lane_counts = np.zeros((num_lanes,), dtype=INDEX_DTYPE)
    lane_rank = np.full((n_vehicles,), MISSING_INDEX, dtype=INDEX_DTYPE)

    for lane in range(num_lanes):
        count = 0
        for pos in range(road_length):
            vehicle_idx = int(occupancy[lane, pos])
            if vehicle_idx == MISSING_INDEX:
                continue
            lane_order[lane, count] = vehicle_idx
            lane_rank[vehicle_idx] = count
            count += 1
        lane_counts[lane] = count

    return lane_order, lane_counts, lane_rank


def compute_neighbors_kernel(
    lane: np.ndarray,
    pos: np.ndarray,
    alive: np.ndarray,
    lane_order: np.ndarray,
    lane_counts: np.ndarray,
    lane_rank: np.ndarray,
    *,
    road_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_vehicles = int(lane.shape[0])
    front_id = np.full((n_vehicles,), MISSING_INDEX, dtype=INDEX_DTYPE)
    back_id = np.full((n_vehicles,), MISSING_INDEX, dtype=INDEX_DTYPE)
    front_gap = np.full((n_vehicles,), MISSING_GAP, dtype=INDEX_DTYPE)
    back_gap = np.full((n_vehicles,), MISSING_GAP, dtype=INDEX_DTYPE)

    for i in range(n_vehicles):
        if not alive[i]:
            continue

        lane_i = int(lane[i])
        rank = int(lane_rank[i])
        count = int(lane_counts[lane_i])

        if count <= 1:
            continue

        front_rank = (rank + 1) % count
        back_rank = (rank - 1) % count

        front_idx = int(lane_order[lane_i, front_rank])
        back_idx = int(lane_order[lane_i, back_rank])

        front_id[i] = front_idx
        back_id[i] = back_idx

        front_gap[i] = int((int(pos[front_idx]) - int(pos[i])) % road_length - 1)
        back_gap[i] = int((int(pos[i]) - int(pos[back_idx])) % road_length - 1)

    return front_id, back_id, front_gap, back_gap
