from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from snfs_traffic.core.backend import StepBackend
from snfs_traffic.core.indexing_numba import NUMBA_AVAILABLE as INDEX_NUMBA_AVAILABLE, build_index_and_neighbors_numba
from snfs_traffic.core.lane_change_kernels import apply_lane_changes_kernel, resolve_lane_change_conflicts_kernel
from snfs_traffic.core.lane_change_numba import NUMBA_AVAILABLE as LANE_NUMBA_AVAILABLE, collect_lane_change_proposals_numba
from snfs_traffic.core.longitudinal_kernels import compute_longitudinal_velocities_kernel
from snfs_traffic.core.longitudinal_numba import (
    NUMBA_AVAILABLE as LONG_NUMBA_AVAILABLE,
    advance_positions_numba,
    compute_longitudinal_velocities_numba,
    draw_longitudinal_randoms,
)
from snfs_traffic.core.params import SimulationParams
from snfs_traffic.core.state import TrafficState, validate_state
from snfs_traffic.core.indexing import build_occupancy
from snfs_traffic.core.step_reference import step_reference
from snfs_traffic.topology import RingTopology


def _validate_reference_inputs(
    state: TrafficState,
    params: SimulationParams,
    topology: RingTopology,
    rng: np.random.Generator,
) -> None:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator")

    validate_state(state, params)

    if not isinstance(topology, RingTopology):
        raise ValueError("topology must be RingTopology")
    if topology.boundary != "periodic":
        raise ValueError("topology.boundary must be 'periodic'")
    if topology.num_lanes != params.num_lanes:
        raise ValueError("topology.num_lanes must match params.num_lanes")
    if topology.length != params.road_length:
        raise ValueError("topology.length must match params.road_length")



@dataclass(frozen=True)
class OptimizedBackend:
    name: str = "optimized"

    def step(self, state: TrafficState, params: SimulationParams, topology: RingTopology, rng: np.random.Generator) -> TrafficState:
        _validate_reference_inputs(state, params, topology, rng)

        if not (INDEX_NUMBA_AVAILABLE and LANE_NUMBA_AVAILABLE and LONG_NUMBA_AVAILABLE):
            return step_reference(state, params, topology, rng)
        if np.any(state.length[state.alive] != 1):
            return step_reference(state, params, topology, rng)
        occupancy, lane_order, lane_counts, _, front_id, _, front_gap, _ = build_index_and_neighbors_numba(
            state.lane, state.pos, state.alive, num_lanes=params.num_lanes, road_length=params.road_length
        )
        proposals = collect_lane_change_proposals_numba(
            lane=state.lane, pos=state.pos, vel=state.vel, alive=state.alive, controlled=state.controlled,
            occupancy=occupancy, lane_order=lane_order, lane_counts=lane_counts, front_id=front_id, front_gap=front_gap,
            num_lanes=params.num_lanes, road_length=params.road_length, vmax_default=params.vmax_default,
            vmax_controlled=params.vmax_controlled, p_lane_change=params.p_lane_change, rng=rng
        )
        accepted = resolve_lane_change_conflicts_kernel(
            proposals, rng, pos=state.pos, length=state.length, road_length=params.road_length
        )
        new_lane, new_changed_lane, new_last_lane_delta = apply_lane_changes_kernel(state.lane, state.changed_lane, state.last_lane_delta, accepted)
        after_lane = state.copy(); after_lane.lane = new_lane; after_lane.changed_lane = new_changed_lane; after_lane.last_lane_delta = new_last_lane_delta
        _, lane_order2, lane_counts2, lane_rank2, _, _, _, _ = build_index_and_neighbors_numba(
            after_lane.lane, after_lane.pos, after_lane.alive, num_lanes=params.num_lanes, road_length=params.road_length
        )
        u_s, u_q, u_b = draw_longitudinal_randoms(after_lane.alive, rng)
        new_vel = compute_longitudinal_velocities_numba(
            after_lane.lane,
            after_lane.pos,
            after_lane.vel,
            after_lane.alive,
            after_lane.controlled,
            lane_order2,
            lane_counts2,
            lane_rank2,
            u_s,
            u_q,
            u_b,
            road_length=params.road_length,
            vmax_default=params.vmax_default,
            vmax_controlled=params.vmax_controlled,
            G=params.G,
            q=params.q,
            r=params.r,
            S=params.S,
            P1=params.P1,
            P2=params.P2,
            P3=params.P3,
            P4=params.P4,
        )
        new_pos = advance_positions_numba(after_lane.pos, new_vel, after_lane.alive, road_length=params.road_length)
        out = after_lane.copy(); out.vel = new_vel; out.pos = new_pos
        validate_state(out, params)
        build_occupancy(out, params)
        return out


def get_optimized_backend() -> StepBackend:
    return OptimizedBackend()
