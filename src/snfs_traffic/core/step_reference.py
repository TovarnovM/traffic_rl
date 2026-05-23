"""Reference step semantics for periodic-ring simulation.

This module keeps the explicit reference longitudinal semantics and provides a
small full-step composition helper that applies lane-change then longitudinal
motion while preserving lane-change flags from the lane-change phase.
"""

from __future__ import annotations

import numpy as np

from snfs_traffic.core.indexing import build_lane_order, build_occupancy, compute_neighbors
from snfs_traffic.core.lane_change_reference import step_lane_change_reference
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


def _vehicle_vmax(state: TrafficState, params: SimulationParams, index: int) -> int:
    return int(params.vmax_controlled if state.controlled[index] else params.vmax_default)


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

    new_vel_i64 = new_state.vel.astype(np.int64, copy=True)

    for i in range(state.n_vehicles):
        if not state.alive[i]:
            continue

        old_v = int(state.vel[i])
        vmax_i = _vehicle_vmax(state, params, i)

        if int(front_id[i]) == -1:
            gap_i = int(params.road_length - 1)
            leader_v = vmax_i
        else:
            gap_i = int(front_gap[i])
            leader_v = int(state.vel[int(front_id[i])])

        v = min(old_v + 1, vmax_i)

        if int(front_id[i]) != -1 and gap_i <= params.G:
            anticipated_leader_motion = max(leader_v - int(params.S), 0)
            anticipated_gap = gap_i + anticipated_leader_motion
            v = min(v, anticipated_gap)

        if old_v == 0 and v > 0 and rng.random() < params.r:
            v = 0

        brake_prob = float(params.P4)
        if int(front_id[i]) != -1 and gap_i <= params.S:
            brake_prob = max(brake_prob, 1.0 - float(params.P2))
        elif int(front_id[i]) != -1 and gap_i <= params.G:
            brake_prob = max(brake_prob, 1.0 - float(params.P3))

        if v > 0 and rng.random() < brake_prob:
            v -= 1

        if int(front_id[i]) != -1:
            v = min(v, gap_i)
        v = max(0, min(v, vmax_i))

        new_vel_i64[i] = v

    new_state.vel = new_vel_i64.astype(state.vel.dtype, copy=False)

    for i in range(state.n_vehicles):
        if not state.alive[i]:
            continue
        new_state.pos[i] = topology.normalize_pos(int(state.pos[i]) + int(new_state.vel[i]))

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
