from __future__ import annotations

import numpy as np

from snfs_traffic.core.indexing import MISSING_INDEX, build_lane_order, build_occupancy, compute_neighbors
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


def _vehicle_vmax(state: TrafficState, params: SimulationParams, i: int) -> int:
    return int(params.vmax_controlled if state.controlled[i] else params.vmax_default)


def _target_lane_neighbors_at_pos(
    state: TrafficState,
    lane_order: np.ndarray,
    lane_counts: np.ndarray,
    target_lane: int,
    candidate_pos: int,
    topology: RingTopology,
) -> tuple[int, int, int, int]:
    count = int(lane_counts[target_lane])
    if count == 0:
        return MISSING_INDEX, MISSING_INDEX, -1, -1

    front_idx = MISSING_INDEX
    front_dist = None
    back_idx = MISSING_INDEX
    back_dist = None

    for rank in range(count):
        idx = int(lane_order[target_lane, rank])
        pos = int(state.pos[idx])
        dist_f = int(topology.forward_distance(candidate_pos, pos))
        dist_b = int(topology.forward_distance(pos, candidate_pos))

        if dist_f > 0 and (front_dist is None or dist_f < front_dist):
            front_dist = dist_f
            front_idx = idx
        if dist_b > 0 and (back_dist is None or dist_b < back_dist):
            back_dist = dist_b
            back_idx = idx

    if front_idx == MISSING_INDEX or back_idx == MISSING_INDEX:
        raise ValueError("target lane neighbor lookup failed")

    g_nf = int(front_dist - 1)
    g_nb = int(back_dist - 1)
    return front_idx, back_idx, g_nf, g_nb


def _eligible_target_lanes(
    i: int,
    state: TrafficState,
    params: SimulationParams,
    topology: RingTopology,
    occupancy: np.ndarray,
    lane_order: np.ndarray,
    lane_counts: np.ndarray,
    front_id: np.ndarray,
    front_gap: np.ndarray,
) -> list[int]:
    lane_i = int(state.lane[i])
    pos_i = int(state.pos[i])
    v_i = int(state.vel[i])

    if int(front_id[i]) == MISSING_INDEX:
        g_pf = params.road_length - 1
        v_p_front = _vehicle_vmax(state, params, i)
    else:
        g_pf = int(front_gap[i])
        v_p_front = int(state.vel[int(front_id[i])])

    eligible: list[int] = []
    for lane_delta in (-1, 1):
        target_lane = lane_i + lane_delta
        if target_lane < 0 or target_lane >= params.num_lanes:
            continue
        if int(occupancy[target_lane, pos_i]) != MISSING_INDEX:
            continue

        t_front, t_back, g_nf, g_nb = _target_lane_neighbors_at_pos(
            state, lane_order, lane_counts, target_lane, pos_i, topology
        )

        if t_front == MISSING_INDEX:
            g_nf = params.road_length - 1
            v_n_front = _vehicle_vmax(state, params, i)
            safety = True
        else:
            v_n_front = int(state.vel[t_front])
            v_n_back = int(state.vel[t_back])
            safety = v_i > (v_n_back - g_nb)

        incentive = (g_nf + v_n_front > v_i) and (v_i >= g_pf + v_p_front)
        if incentive and safety:
            eligible.append(target_lane)

    return eligible


def _resolve_lane_change_conflicts(proposals: dict[tuple[int, int], list[int]], rng: np.random.Generator) -> dict[int, int]:
    accepted: dict[int, int] = {}
    for (target_lane, _pos), candidates in proposals.items():
        if len(candidates) == 1:
            winner = candidates[0]
        else:
            winner = candidates[int(rng.integers(0, len(candidates)))]
        accepted[winner] = target_lane
    return accepted


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
    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    front_id, _back_id, front_gap, _back_gap = compute_neighbors(state, lane_order, lane_counts, lane_rank, topology)

    proposals: dict[tuple[int, int], list[int]] = {}

    for i in range(state.n_vehicles):
        if not state.alive[i]:
            continue

        eligible_targets = _eligible_target_lanes(
            i, state, params, topology, occupancy, lane_order, lane_counts, front_id, front_gap
        )
        if not eligible_targets:
            continue

        target_lane = eligible_targets[0]
        if len(eligible_targets) > 1:
            target_lane = eligible_targets[int(rng.integers(0, len(eligible_targets)))]

        if float(rng.random()) >= params.p_lane_change:
            continue

        key = (target_lane, int(state.pos[i]))
        proposals.setdefault(key, []).append(i)

    accepted = _resolve_lane_change_conflicts(proposals, rng)

    out = state.copy()
    out.changed_lane.fill(False)
    out.last_lane_delta.fill(0)

    for i, target_lane in accepted.items():
        old_lane = int(state.lane[i])
        out.lane[i] = np.asarray(target_lane, dtype=out.lane.dtype)
        out.changed_lane[i] = True
        out.last_lane_delta[i] = np.asarray(target_lane - old_lane, dtype=out.last_lane_delta.dtype)

    validate_state(out, params)
    build_occupancy(out, params)
    return out
