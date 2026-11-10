"""Reference indexing for periodic ring traffic states.

This module keeps head-cell indexing for lane ordering and also exposes length-aware body occupancy:
`occupancy[lane, head_pos] = vehicle_index` for alive vehicles.
Body occupancy is validated via `build_body_occupancy`.

Neighbor helpers remain head-order based; length-aware body occupancy and gap helpers are provided in indexing_kernels.
"""

from __future__ import annotations

import numpy as np

from snfs_traffic.core.indexing_kernels import (
    INDEX_DTYPE,
    MISSING_GAP,
    MISSING_INDEX,
    build_lane_order_kernel,
    build_body_occupancy_kernel,
    build_occupancy_kernel,
    compute_cumulative_forward_gap_kernel,
    compute_forward_empty_gap_kernel,
    compute_neighbors_kernel,
)
from snfs_traffic.core.params import SimulationParams
from snfs_traffic.core.state import TrafficState, validate_state
from snfs_traffic.topology import RingTopology

def build_occupancy(state: TrafficState, params: SimulationParams) -> np.ndarray:
    validate_state(state, params)

    return build_occupancy_kernel(
        state.lane,
        state.pos,
        state.alive,
        num_lanes=params.num_lanes,
        road_length=params.road_length,
    )


def build_body_occupancy(state: TrafficState, params: SimulationParams) -> np.ndarray:
    validate_state(state, params)
    return build_body_occupancy_kernel(
        state.lane,
        state.pos,
        state.length,
        state.alive,
        num_lanes=params.num_lanes,
        road_length=params.road_length,
    )


def _validate_occupancy(occupancy: np.ndarray, *, n_vehicles: int) -> None:
    if not isinstance(occupancy, np.ndarray):
        raise ValueError("occupancy must be a numpy.ndarray")
    if occupancy.ndim != 2:
        raise ValueError("occupancy must be 2D")
    if occupancy.dtype != np.dtype(INDEX_DTYPE):
        raise ValueError(f"occupancy must have dtype {np.dtype(INDEX_DTYPE)}")
    if not occupancy.flags.c_contiguous:
        raise ValueError("occupancy must be C-contiguous")

    if np.any(occupancy < MISSING_INDEX):
        raise ValueError("occupancy contains value < -1")
    if np.any(occupancy >= n_vehicles):
        raise ValueError("occupancy contains value >= n_vehicles")

    used = occupancy[occupancy != MISSING_INDEX]
    if used.size != np.unique(used).size:
        raise ValueError("occupancy contains duplicate vehicle index")


def _validate_lane_order_inputs(occupancy: np.ndarray, *, n_vehicles: int) -> None:
    if isinstance(n_vehicles, bool) or not isinstance(n_vehicles, int):
        raise ValueError("n_vehicles must be an int")
    if n_vehicles < 0:
        raise ValueError("n_vehicles must be >= 0")
    _validate_occupancy(occupancy, n_vehicles=n_vehicles)


def build_lane_order(occupancy: np.ndarray, *, n_vehicles: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _validate_lane_order_inputs(occupancy, n_vehicles=n_vehicles)

    return build_lane_order_kernel(occupancy, n_vehicles=n_vehicles)


def compute_neighbors(
    state: TrafficState,
    lane_order: np.ndarray,
    lane_counts: np.ndarray,
    lane_rank: np.ndarray,
    topology: RingTopology,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(topology, RingTopology):
        raise ValueError("topology must be RingTopology")
    if topology.boundary != "periodic":
        raise ValueError("topology boundary must be periodic")

    if not isinstance(lane_order, np.ndarray) or lane_order.ndim != 2:
        raise ValueError("lane_order must be a 2D numpy.ndarray")
    if lane_order.dtype != np.dtype(INDEX_DTYPE):
        raise ValueError("lane_order must have dtype int32")
    if not lane_order.flags.c_contiguous:
        raise ValueError("lane_order must be C-contiguous")

    num_lanes, road_length = lane_order.shape

    if topology.num_lanes != num_lanes:
        raise ValueError("topology.num_lanes must match lane_order shape")
    if topology.length != road_length:
        raise ValueError("topology.length must match lane_order shape")

    if not isinstance(lane_counts, np.ndarray) or lane_counts.shape != (num_lanes,):
        raise ValueError("lane_counts must have shape (num_lanes,)")
    if lane_counts.dtype != np.dtype(INDEX_DTYPE):
        raise ValueError("lane_counts must have dtype int32")
    if not lane_counts.flags.c_contiguous:
        raise ValueError("lane_counts must be C-contiguous")

    if not isinstance(lane_rank, np.ndarray) or lane_rank.shape != (state.n_vehicles,):
        raise ValueError("lane_rank must have shape (state.n_vehicles,)")
    if lane_rank.dtype != np.dtype(INDEX_DTYPE):
        raise ValueError("lane_rank must have dtype int32")
    if not lane_rank.flags.c_contiguous:
        raise ValueError("lane_rank must be C-contiguous")

    for i in range(state.n_vehicles):
        if not state.alive[i]:
            continue

        lane = int(state.lane[i])
        rank = int(lane_rank[i])
        count = int(lane_counts[lane])

        if rank == MISSING_INDEX:
            raise ValueError(f"alive vehicle index {i} has missing lane_rank")
        if rank < 0 or rank >= count:
            raise ValueError(f"lane_rank for vehicle index {i} is inconsistent with lane_counts")

        if int(lane_order[lane, rank]) != i:
            raise ValueError(f"lane_order and lane_rank inconsistent for vehicle index {i}")

        if count > 1:
            front_rank = (rank + 1) % count
            back_rank = (rank - 1) % count
            front_idx = int(lane_order[lane, front_rank])
            back_idx = int(lane_order[lane, back_rank])
            if front_idx == MISSING_INDEX or back_idx == MISSING_INDEX:
                raise ValueError(f"lane_order missing neighbor index for lane {lane}")

    return compute_neighbors_kernel(
        state.lane,
        state.pos,
        state.alive,
        lane_order,
        lane_counts,
        lane_rank,
        road_length=topology.length,
    )
