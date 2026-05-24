from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from snfs_traffic.core.backend import StepBackend
from snfs_traffic.core.indexing import build_lane_order, build_occupancy, compute_neighbors
from snfs_traffic.core.lane_change_kernels import apply_lane_changes_kernel, resolve_lane_change_conflicts_kernel
from snfs_traffic.core.lane_change_numba import NUMBA_AVAILABLE as LANE_NUMBA_AVAILABLE, collect_lane_change_proposals_numba
from snfs_traffic.core.longitudinal_kernels import compute_longitudinal_velocities_kernel
from snfs_traffic.core.longitudinal_numba import NUMBA_AVAILABLE as LONG_NUMBA_AVAILABLE, advance_positions_numba
from snfs_traffic.core.params import SimulationParams
from snfs_traffic.core.state import TrafficState
from snfs_traffic.core.step_reference import step_reference
from snfs_traffic.topology import RingTopology


@dataclass(frozen=True)
class OptimizedBackend:
    name: str = "optimized"

    def step(self, state: TrafficState, params: SimulationParams, topology: RingTopology, rng: np.random.Generator) -> TrafficState:
        if not (LANE_NUMBA_AVAILABLE and LONG_NUMBA_AVAILABLE):
            return step_reference(state, params, topology, rng)
        occupancy = build_occupancy(state, params)
        lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
        front_id, _, front_gap, _ = compute_neighbors(state, lane_order, lane_counts, lane_rank, topology)
        proposals = collect_lane_change_proposals_numba(
            lane=state.lane, pos=state.pos, vel=state.vel, alive=state.alive, controlled=state.controlled,
            occupancy=occupancy, lane_order=lane_order, lane_counts=lane_counts, front_id=front_id, front_gap=front_gap,
            num_lanes=params.num_lanes, road_length=params.road_length, vmax_default=params.vmax_default,
            vmax_controlled=params.vmax_controlled, p_lane_change=params.p_lane_change, rng=rng
        )
        accepted = resolve_lane_change_conflicts_kernel(proposals, rng)
        new_lane, new_changed_lane, new_last_lane_delta = apply_lane_changes_kernel(state.lane, state.changed_lane, state.last_lane_delta, accepted)
        after_lane = state.copy(); after_lane.lane = new_lane; after_lane.changed_lane = new_changed_lane; after_lane.last_lane_delta = new_last_lane_delta
        occupancy2 = build_occupancy(after_lane, params)
        lane_order2, lane_counts2, lane_rank2 = build_lane_order(occupancy2, n_vehicles=after_lane.n_vehicles)
        front_id2, _, front_gap2, _ = compute_neighbors(after_lane, lane_order2, lane_counts2, lane_rank2, topology)
        new_vel = compute_longitudinal_velocities_kernel(after_lane.vel, after_lane.alive, after_lane.controlled, front_id2, front_gap2,
            road_length=params.road_length, vmax_default=params.vmax_default, vmax_controlled=params.vmax_controlled,
            G=params.G, S=params.S, r=params.r, P2=params.P2, P3=params.P3, P4=params.P4, rng=rng)
        new_pos = advance_positions_numba(after_lane.pos, new_vel, after_lane.alive, road_length=params.road_length)
        out = after_lane.copy(); out.vel = new_vel; out.pos = new_pos
        return out


def get_optimized_backend() -> StepBackend:
    return OptimizedBackend()
