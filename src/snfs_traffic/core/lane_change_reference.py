from __future__ import annotations

import numpy as np

from snfs_traffic.core.indexing import build_body_occupancy, build_lane_order, build_occupancy, compute_neighbors
from snfs_traffic.core.lane_change_kernels import (
    apply_lane_changes_kernel,
    collect_lane_change_proposals_kernel,
    resolve_lane_change_conflicts_kernel,
)
from snfs_traffic.core.params import SimulationParams
from snfs_traffic.core.state import TrafficState, validate_state
from snfs_traffic.topology import RingTopology


def _validate_topology(params: SimulationParams, topology: RingTopology) -> None:
    if not isinstance(topology, RingTopology):
        raise ValueError("topology must be RingTopology")
    if topology.boundary != "periodic":
        raise ValueError("topology.boundary must be periodic")
    if topology.num_lanes != params.num_lanes:
        raise ValueError("topology.num_lanes must match params.num_lanes")
    if topology.length != params.road_length:
        raise ValueError("topology.length must match params.road_length")


def _validate_rng(rng: np.random.Generator) -> None:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator")


def step_lane_change_reference(
    state: TrafficState,
    params: SimulationParams,
    topology: RingTopology,
    rng: np.random.Generator,
) -> TrafficState:
    validate_state(state, params)
    _validate_topology(params, topology)
    _validate_rng(rng)

    occupancy = build_occupancy(state, params)
    body_occupancy = build_body_occupancy(state, params)
    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    front_id, _back_id, front_gap, _back_gap = compute_neighbors(state, lane_order, lane_counts, lane_rank, topology)

    proposals = collect_lane_change_proposals_kernel(
        state.lane,
        state.pos,
        state.vel,
        state.length,
        state.alive,
        state.controlled,
        body_occupancy,
        lane_order,
        lane_counts,
        front_id,
        front_gap,
        num_lanes=params.num_lanes,
        road_length=params.road_length,
        vmax_default=params.vmax_default,
        vmax_controlled=params.vmax_controlled,
        p_lane_change=params.p_lane_change,
        rng=rng,
    )

    accepted = resolve_lane_change_conflicts_kernel(
        proposals,
        rng,
        pos=state.pos,
        length=state.length,
        road_length=params.road_length,
    )

    new_lane, new_changed_lane, new_last_lane_delta = apply_lane_changes_kernel(
        state.lane,
        state.changed_lane,
        state.last_lane_delta,
        accepted,
    )

    out = state.copy()
    out.lane = new_lane
    out.changed_lane = new_changed_lane
    out.last_lane_delta = new_last_lane_delta

    validate_state(out, params)
    build_occupancy(out, params)
    build_body_occupancy(out, params)
    return out
