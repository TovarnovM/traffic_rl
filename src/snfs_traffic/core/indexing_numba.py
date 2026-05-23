"""Optional Numba-compiled indexing kernels.

This module is internal and intentionally separate from the public indexing API.
"""

from __future__ import annotations

import numpy as np

from snfs_traffic.core.indexing_kernels import INDEX_DTYPE, MISSING_GAP, MISSING_INDEX

try:
    from numba import njit
except ImportError:  # pragma: no cover - tested via behavior checks
    njit = None
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _build_occupancy_numba_impl(
        lane: np.ndarray,
        pos: np.ndarray,
        alive: np.ndarray,
        num_lanes: int,
        road_length: int,
    ) -> np.ndarray:
        occupancy = np.empty((num_lanes, road_length), dtype=INDEX_DTYPE)
        for lane_idx in range(num_lanes):
            for pos_idx in range(road_length):
                occupancy[lane_idx, pos_idx] = MISSING_INDEX

        n_vehicles = int(lane.shape[0])
        for i in range(n_vehicles):
            if not alive[i]:
                continue
            lane_i = int(lane[i])
            pos_i = int(pos[i])
            if occupancy[lane_i, pos_i] != MISSING_INDEX:
                raise ValueError("duplicate occupied head cell")
            occupancy[lane_i, pos_i] = i

        return occupancy


    @njit(cache=True)
    def _build_lane_order_numba_impl(occupancy: np.ndarray, n_vehicles: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_lanes, road_length = occupancy.shape
        lane_order = np.empty((num_lanes, road_length), dtype=INDEX_DTYPE)
        for lane_idx in range(num_lanes):
            for pos_idx in range(road_length):
                lane_order[lane_idx, pos_idx] = MISSING_INDEX

        lane_counts = np.zeros((num_lanes,), dtype=INDEX_DTYPE)
        lane_rank = np.empty((n_vehicles,), dtype=INDEX_DTYPE)
        for i in range(n_vehicles):
            lane_rank[i] = MISSING_INDEX

        for lane_idx in range(num_lanes):
            count = 0
            for pos_idx in range(road_length):
                vehicle_idx = int(occupancy[lane_idx, pos_idx])
                if vehicle_idx == MISSING_INDEX:
                    continue
                lane_order[lane_idx, count] = vehicle_idx
                lane_rank[vehicle_idx] = count
                count += 1
            lane_counts[lane_idx] = count

        return lane_order, lane_counts, lane_rank


    @njit(cache=True)
    def _compute_neighbors_numba_impl(
        lane: np.ndarray,
        pos: np.ndarray,
        alive: np.ndarray,
        lane_order: np.ndarray,
        lane_counts: np.ndarray,
        lane_rank: np.ndarray,
        road_length: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_vehicles = int(lane.shape[0])

        front_id = np.empty((n_vehicles,), dtype=INDEX_DTYPE)
        back_id = np.empty((n_vehicles,), dtype=INDEX_DTYPE)
        front_gap = np.empty((n_vehicles,), dtype=INDEX_DTYPE)
        back_gap = np.empty((n_vehicles,), dtype=INDEX_DTYPE)

        for i in range(n_vehicles):
            front_id[i] = MISSING_INDEX
            back_id[i] = MISSING_INDEX
            front_gap[i] = MISSING_GAP
            back_gap[i] = MISSING_GAP

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

else:

    def _build_occupancy_numba_impl(*args: object, **kwargs: object) -> np.ndarray:
        raise ImportError("Numba is not installed. Install optional dependency: pip install 'snfs-traffic[numba]'.")

    def _build_lane_order_numba_impl(*args: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise ImportError("Numba is not installed. Install optional dependency: pip install 'snfs-traffic[numba]'.")

    def _compute_neighbors_numba_impl(*args: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise ImportError("Numba is not installed. Install optional dependency: pip install 'snfs-traffic[numba]'.")


def _ensure_numba_available() -> None:
    if not NUMBA_AVAILABLE:
        raise ImportError("Numba is not installed. Install optional dependency: pip install 'snfs-traffic[numba]'.")


def build_occupancy_numba(
    lane: np.ndarray,
    pos: np.ndarray,
    alive: np.ndarray,
    *,
    num_lanes: int,
    road_length: int,
) -> np.ndarray:
    _ensure_numba_available()
    return _build_occupancy_numba_impl(lane, pos, alive, num_lanes, road_length)


def build_lane_order_numba(occupancy: np.ndarray, *, n_vehicles: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _ensure_numba_available()
    return _build_lane_order_numba_impl(occupancy, n_vehicles)


def compute_neighbors_numba(
    lane: np.ndarray,
    pos: np.ndarray,
    alive: np.ndarray,
    lane_order: np.ndarray,
    lane_counts: np.ndarray,
    lane_rank: np.ndarray,
    *,
    road_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _ensure_numba_available()
    return _compute_neighbors_numba_impl(lane, pos, alive, lane_order, lane_counts, lane_rank, road_length)
