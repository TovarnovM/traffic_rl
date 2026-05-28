"""Runtime invariant checks for reference traffic states."""

from __future__ import annotations

import numpy as np

from snfs_traffic.core.indexing import build_body_occupancy, build_lane_order, build_occupancy, compute_neighbors
from snfs_traffic.core.params import SimulationParams
from snfs_traffic.core.state import TrafficState, validate_state
from snfs_traffic.topology import RingTopology


def validate_runtime_invariants(
    state: TrafficState,
    params: SimulationParams,
    topology: RingTopology,
) -> None:
    """Validate runtime invariants including length-aware body occupancy."""

    validate_state(state, params)

    if not isinstance(topology, RingTopology):
        raise ValueError("topology must be RingTopology")
    if topology.num_lanes != params.num_lanes:
        raise ValueError("topology.num_lanes must match params.num_lanes")
    if topology.length != params.road_length:
        raise ValueError("topology.length must match params.road_length")

    occupancy = build_occupancy(state, params)
    body_occupancy = build_body_occupancy(state, params)
    occupied_count = int((occupancy >= 0).sum())
    alive_count = int(state.alive.sum())
    if occupied_count != alive_count:
        raise ValueError(
            "occupied head-cell count must equal alive count "
            f"(occupied={occupied_count}, alive={alive_count})"
        )

    alive = state.alive

    if np.any((state.lane[alive] < 0) | (state.lane[alive] >= params.num_lanes)):
        raise ValueError("alive vehicle lane out of range")

    if np.any((state.pos[alive] < 0) | (state.pos[alive] >= params.road_length)):
        raise ValueError("alive vehicle pos out of range")

    if np.any(state.vel[alive] < 0):
        raise ValueError("alive vehicle velocity must be >= 0")

    alive_controlled = alive & state.controlled
    alive_uncontrolled = alive & ~state.controlled
    if np.any(state.vel[alive_uncontrolled] > params.vmax_default):
        raise ValueError("uncontrolled alive vehicle velocity exceeds params.vmax_default")
    if np.any(state.vel[alive_controlled] > params.vmax_controlled):
        raise ValueError("controlled alive vehicle velocity exceeds params.vmax_controlled")

    if np.any(~np.isin(state.last_lane_delta, np.array([-1, 0, 1], dtype=state.last_lane_delta.dtype))):
        raise ValueError("last_lane_delta must be one of {-1, 0, +1}")

    expected_changed = state.last_lane_delta != 0
    if np.any(state.changed_lane[alive] != expected_changed[alive]):
        raise ValueError("alive vehicles must satisfy changed_lane == (last_lane_delta != 0)")

    inactive = ~alive
    if np.any(state.changed_lane[inactive]):
        raise ValueError("inactive vehicles must have changed_lane == False")
    if np.any(state.last_lane_delta[inactive] != 0):
        raise ValueError("inactive vehicles must have last_lane_delta == 0")

    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    front_index, back_index, front_gap, back_gap = compute_neighbors(
        state,
        lane_order,
        lane_counts,
        lane_rank,
        topology,
    )

    expected_shape = (state.n_vehicles,)
    if front_index.shape != expected_shape:
        raise ValueError("front_index has invalid shape")
    if front_gap.shape != expected_shape:
        raise ValueError("front_gap has invalid shape")
    if back_index.shape != expected_shape:
        raise ValueError("back_index has invalid shape")
    if back_gap.shape != expected_shape:
        raise ValueError("back_gap has invalid shape")
    alive_lengths = state.length[alive]
    if np.any(alive_lengths < 1):
        raise ValueError("alive vehicle length must be >= 1")
    if np.any(alive_lengths > params.road_length):
        raise ValueError("alive vehicle length must be <= road_length")
    body_count = int((body_occupancy >= 0).sum())
    expected_body = int(alive_lengths.astype(np.int64).sum())
    if body_count != expected_body:
        raise ValueError("occupied body-cell count must equal sum of alive lengths")
