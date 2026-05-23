"""Reference step semantics for periodic-ring simulation.

This module keeps the explicit reference longitudinal semantics and provides a
small full-step composition helper that applies lane-change then longitudinal
motion while preserving lane-change flags from the lane-change phase.
"""

from __future__ import annotations

import numpy as np

from snfs_traffic.core.indexing import build_lane_order, build_occupancy, compute_neighbors
from snfs_traffic.core.lane_change_reference import step_lane_change_reference
from snfs_traffic.core.longitudinal_kernels import advance_positions_kernel, compute_longitudinal_velocities_kernel
from snfs_traffic.core.params import SimulationParams
from snfs_traffic.core.state import TrafficState, validate_state
from snfs_traffic.topology import RingTopology


def _validate_topology(params: SimulationParams, topology: RingTopology) -> None:
    if not isinstance(topology, RingTopology):
        raise ValueError("topology must be RingTopology")
    if topology.boundary != "periodic":
        raise ValueError("topology.boundary must be 'periodic'")
    if topology.num_lanes != params.num_lanes:
        raise ValueError("topology.num_lanes must match params.num_lanes")
    if topology.length != params.road_length:
        raise ValueError("topology.length must match params.road_length")


def step_longitudinal_reference(
    state: TrafficState,
    params: SimulationParams,
    topology: RingTopology,
    rng: np.random.Generator,
) -> TrafficState:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator")

    validate_state(state, params)
    _validate_topology(params, topology)

    occupancy = build_occupancy(state, params)
    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    front_id, _, front_gap, _ = compute_neighbors(state, lane_order, lane_counts, lane_rank, topology)

    new_state = state.copy()
    new_state.changed_lane.fill(False)
    new_state.last_lane_delta.fill(0)

    new_state.vel = compute_longitudinal_velocities_kernel(
        state.vel,
        state.alive,
        state.controlled,
        front_id,
        front_gap,
        road_length=params.road_length,
        vmax_default=params.vmax_default,
        vmax_controlled=params.vmax_controlled,
        G=params.G,
        S=params.S,
        r=params.r,
        P2=params.P2,
        P3=params.P3,
        P4=params.P4,
        rng=rng,
    )
    new_state.pos = advance_positions_kernel(
        state.pos,
        new_state.vel,
        state.alive,
        road_length=params.road_length,
    )

    validate_state(new_state, params)
    build_occupancy(new_state, params)
    return new_state


def step_reference(
    state: TrafficState,
    params: SimulationParams,
    topology: RingTopology,
    rng: np.random.Generator,
) -> TrafficState:
    """Apply one full reference step: lane-change phase then longitudinal phase.

    Lane-change flags from the lane-change phase are restored after the
    longitudinal phase so callers can observe lateral movement that happened
    during this full step.
    """

    after_lane_change = step_lane_change_reference(state, params, topology, rng)
    changed_lane = after_lane_change.changed_lane.copy()
    last_lane_delta = after_lane_change.last_lane_delta.copy()

    out = step_longitudinal_reference(after_lane_change, params, topology, rng)
    out = out.copy()
    out.changed_lane = changed_lane
    out.last_lane_delta = last_lane_delta

    validate_state(out, params)
    build_occupancy(out, params)
    return out
