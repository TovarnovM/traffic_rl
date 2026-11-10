from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from snfs_traffic.core import SimulationParams, TrafficState, high_speed_vehicle_mask
from snfs_traffic.core.indexing import MISSING_INDEX, build_body_occupancy, build_lane_order, build_occupancy, compute_neighbors
from snfs_traffic.core.indexing_kernels import compute_cumulative_forward_gap_kernel
from snfs_traffic.core.longitudinal_kernels import advance_positions_kernel
from snfs_traffic.core.longitudinal_numba import (
    NUMBA_AVAILABLE as LONG_NUMBA_AVAILABLE,
    advance_positions_numba,
    compute_longitudinal_velocities_numba,
    compute_longitudinal_velocities_controlled_speed_numba,
    draw_longitudinal_randoms,
)
from snfs_traffic.core.lane_change_kernels import (
    apply_lane_changes_kernel,
    collect_lane_change_proposals_kernel,
    resolve_lane_change_conflicts_kernel,
    target_lane_neighbors_at_pos_kernel,
)
from snfs_traffic.core.lane_change_numba import (
    NUMBA_AVAILABLE as LANE_NUMBA_AVAILABLE,
    collect_lane_change_proposals_numba,
)
from snfs_traffic.core.state import validate_state
from snfs_traffic.core.step_reference import step_longitudinal_reference
from snfs_traffic.topology import RingTopology

LANE_LEFT = -1
LANE_STAY = 0
LANE_RIGHT = 1
LANE_ACTION_VALUES = (LANE_LEFT, LANE_STAY, LANE_RIGHT)

@dataclass(frozen=True, slots=True)
class ControlledVehicleAction:
    lane_delta: int
    speed_delta: int | None = None


def _validate_action_delta(name: str, value: object, *, allow_none: bool) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in {{-1, 0, +1}}" + (" or None" if allow_none else ""))
    value_int = int(value)
    if value_int not in LANE_ACTION_VALUES:
        raise ValueError(f"{name} must be one of {{-1, 0, +1}}" + (" or None" if allow_none else ""))
    return value_int


def normalize_controlled_action(action: object) -> ControlledVehicleAction:
    if isinstance(action, ControlledVehicleAction):
        lane_delta = _validate_action_delta("lane_delta", action.lane_delta, allow_none=False)
        speed_delta = _validate_action_delta("speed_delta", action.speed_delta, allow_none=True)
        return ControlledVehicleAction(lane_delta=int(lane_delta), speed_delta=None if speed_delta is None else int(speed_delta))
    if isinstance(action, tuple):
        if len(action) != 2:
            raise ValueError("tuple controlled actions must be (lane_delta, speed_delta)")
        lane_delta = _validate_action_delta("lane_delta", action[0], allow_none=False)
        speed_delta = _validate_action_delta("speed_delta", action[1], allow_none=True)
        return ControlledVehicleAction(lane_delta=int(lane_delta), speed_delta=None if speed_delta is None else int(speed_delta))
    lane_delta = _validate_action_delta("lane_delta", action, allow_none=False)
    return ControlledVehicleAction(lane_delta=int(lane_delta), speed_delta=None)


@dataclass(frozen=True, slots=True)
class ControlledActionBatch:
    vehicle_id: np.ndarray
    lane_delta: np.ndarray
    speed_delta: np.ndarray
    speed_delta_provided: np.ndarray

    def __post_init__(self) -> None:
        if not all(isinstance(arr, np.ndarray) for arr in (self.vehicle_id, self.lane_delta, self.speed_delta, self.speed_delta_provided)):
            raise ValueError("controlled action batch fields must be numpy arrays")
        if self.vehicle_id.ndim != 1 or self.lane_delta.ndim != 1 or self.speed_delta.ndim != 1 or self.speed_delta_provided.ndim != 1:
            raise ValueError("controlled action batch fields must be 1D")
        if not (self.vehicle_id.shape == self.lane_delta.shape == self.speed_delta.shape == self.speed_delta_provided.shape):
            raise ValueError("controlled action batch fields must have same shape")
        if self.vehicle_id.dtype != np.int32:
            raise ValueError("vehicle_id dtype must be int32")
        if self.lane_delta.dtype != np.int8 or self.speed_delta.dtype != np.int8:
            raise ValueError("lane_delta and speed_delta dtypes must be int8")
        if self.speed_delta_provided.dtype != np.bool_:
            raise ValueError("speed_delta_provided dtype must be bool")
        if not all(arr.flags.c_contiguous for arr in (self.vehicle_id, self.lane_delta, self.speed_delta, self.speed_delta_provided)):
            raise ValueError("controlled action batch fields must be C-contiguous")
        if len(np.unique(self.vehicle_id)) != self.vehicle_id.size:
            raise ValueError("vehicle_id values must be unique")
        allowed = np.array(LANE_ACTION_VALUES, dtype=np.int8)
        if not np.all(np.isin(self.lane_delta, allowed)):
            raise ValueError("lane_delta values must be in {-1, 0, +1}")
        if not np.all(np.isin(self.speed_delta[self.speed_delta_provided], allowed)):
            raise ValueError("speed_delta values must be in {-1, 0, +1} when provided")

    @classmethod
    def from_mapping(cls, actions: Mapping[int, object]) -> "ControlledActionBatch":
        ids: list[int] = []
        lane_deltas: list[int] = []
        speed_deltas: list[int] = []
        speed_provided: list[bool] = []
        try:
            for vehicle_id, raw_action in actions.items():
                action = normalize_controlled_action(raw_action)
                ids.append(int(vehicle_id))
                lane_deltas.append(int(action.lane_delta))
                speed_deltas.append(0 if action.speed_delta is None else int(action.speed_delta))
                speed_provided.append(action.speed_delta is not None)
            vehicle_ids = np.asarray(ids, dtype=np.int32)
            lane = np.asarray(lane_deltas, dtype=np.int8)
            speed = np.asarray(speed_deltas, dtype=np.int8)
            provided = np.asarray(speed_provided, dtype=np.bool_)
        except (OverflowError, ValueError, TypeError) as exc:
            raise ValueError("failed to convert mapping into ControlledActionBatch") from exc
        return cls(np.ascontiguousarray(vehicle_ids), np.ascontiguousarray(lane), np.ascontiguousarray(speed), np.ascontiguousarray(provided))



@dataclass(frozen=True, slots=True)
class LaneActionBatch:
    vehicle_id: np.ndarray
    lane_delta: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.vehicle_id, np.ndarray) or not isinstance(self.lane_delta, np.ndarray):
            raise ValueError("vehicle_id and lane_delta must be numpy arrays")
        if self.vehicle_id.ndim != 1 or self.lane_delta.ndim != 1 or self.vehicle_id.shape != self.lane_delta.shape:
            raise ValueError("vehicle_id and lane_delta must be 1D with same shape")
        if self.vehicle_id.dtype != np.int32:
            raise ValueError("vehicle_id dtype must be int32")
        if self.lane_delta.dtype != np.int8:
            raise ValueError("lane_delta dtype must be int8")
        if not self.vehicle_id.flags.c_contiguous or not self.lane_delta.flags.c_contiguous:
            raise ValueError("vehicle_id and lane_delta must be C-contiguous")
        if len(np.unique(self.vehicle_id)) != self.vehicle_id.size:
            raise ValueError("vehicle_id values must be unique")
        if not np.all(np.isin(self.lane_delta, np.array(LANE_ACTION_VALUES, dtype=self.lane_delta.dtype))):
            raise ValueError("lane_delta values must be in {-1, 0, +1}")

    @classmethod
    def from_mapping(cls, actions: Mapping[int, int]) -> "LaneActionBatch":
        try:
            ids = np.asarray(list(actions.keys()), dtype=np.int32)
            deltas = np.asarray(list(actions.values()), dtype=np.int8)
        except (OverflowError, ValueError, TypeError) as exc:
            raise ValueError("failed to convert mapping into LaneActionBatch") from exc
        return cls(np.ascontiguousarray(ids), np.ascontiguousarray(deltas))


@dataclass(frozen=True, slots=True)
class ControlledActionResult:
    vehicle_id: np.ndarray
    requested_lane_delta: np.ndarray
    applied_lane_delta: np.ndarray
    valid: np.ndarray
    applied: np.ndarray
    rejection_reason: tuple[str, ...]
    requested_speed_delta: np.ndarray | None = None
    desired_velocity: np.ndarray | None = None
    applied_velocity: np.ndarray | None = None
    speed_clipped_by_safety: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class LateralOverrideResult:
    """Outcome of rule-based lateral requests for arbitrary alive vehicles."""

    vehicle_id: np.ndarray
    requested_lane_delta: np.ndarray
    applied_lane_delta: np.ndarray
    valid: np.ndarray
    applied: np.ndarray
    rejection_reason: tuple[str, ...]


def controlled_vehicle_ids(state: TrafficState) -> np.ndarray:
    return state.vehicle_id[state.alive & state.controlled].copy()


def normalize_controlled_actions(actions, state: TrafficState, *, require_all_controlled: bool) -> ControlledActionBatch:
    if isinstance(actions, Mapping):
        batch = ControlledActionBatch.from_mapping(actions)
    elif isinstance(actions, LaneActionBatch):
        batch = ControlledActionBatch(
            actions.vehicle_id.copy(),
            actions.lane_delta.copy(),
            np.zeros(actions.lane_delta.shape, dtype=np.int8),
            np.zeros(actions.lane_delta.shape, dtype=np.bool_),
        )
    elif isinstance(actions, ControlledActionBatch):
        batch = actions
    else:
        raise TypeError("actions must be Mapping[int, object], LaneActionBatch, or ControlledActionBatch")

    ids = controlled_vehicle_ids(state)
    alive_ids = set(int(v) for v in ids)
    if len(np.unique(batch.vehicle_id)) != batch.vehicle_id.size:
        raise ValueError("duplicate vehicle_id actions")
    for v in batch.vehicle_id:
        if int(v) not in alive_ids:
            raise ValueError(f"action vehicle_id {int(v)} is not an alive controlled vehicle")
    action_map = {
        int(v): (int(lane), int(speed), bool(provided))
        for v, lane, speed, provided in zip(batch.vehicle_id, batch.lane_delta, batch.speed_delta, batch.speed_delta_provided)
    }
    if require_all_controlled and len(action_map) != len(ids):
        raise ValueError("missing actions for alive controlled vehicles")
    out_lane = np.zeros(ids.shape[0], dtype=np.int8)
    out_speed = np.zeros(ids.shape[0], dtype=np.int8)
    out_provided = np.zeros(ids.shape[0], dtype=np.bool_)
    for i, vid in enumerate(ids):
        if int(vid) in action_map:
            lane_delta, speed_delta, provided = action_map[int(vid)]
            out_lane[i] = np.int8(lane_delta)
            out_speed[i] = np.int8(speed_delta)
            out_provided[i] = np.bool_(provided)
        elif require_all_controlled:
            raise ValueError("missing actions for alive controlled vehicles")
    return ControlledActionBatch(
        np.ascontiguousarray(ids.astype(state.vehicle_id.dtype)),
        np.ascontiguousarray(out_lane),
        np.ascontiguousarray(out_speed),
        np.ascontiguousarray(out_provided),
    )


def normalize_lane_actions(actions, state: TrafficState, *, require_all_controlled: bool) -> LaneActionBatch:
    normalized = normalize_controlled_actions(actions, state, require_all_controlled=require_all_controlled)
    return LaneActionBatch(normalized.vehicle_id.copy(), normalized.lane_delta.copy())


def _lateral_valid(state, lane_order, lane_counts, occupancy, body_occupancy, idx: int, target_lane: int, road_length: int):
    pos_i = int(state.pos[idx])
    vel_i = int(state.vel[idx])
    len_i = int(state.length[idx])
    for d in range(len_i):
        cell = (pos_i + d) % road_length
        if int(body_occupancy[target_lane, cell]) != MISSING_INDEX:
            return False, "target_occupied"
    if int(lane_counts[target_lane]) == 0:
        return True, ""
    t_front, t_back, _g_nf, g_nb = target_lane_neighbors_at_pos_kernel(
        state.pos, lane_order, lane_counts, target_lane=target_lane, candidate_pos=pos_i, road_length=road_length
    )
    if t_front == MISSING_INDEX:
        return True, ""
    g_nb_len = int(g_nb) - int(state.length[t_back]) + 1
    safe = vel_i > (int(state.vel[t_back]) - g_nb_len)
    return (bool(safe), "" if safe else "unsafe")


def compute_lateral_action_mask(state: TrafficState, params: SimulationParams, topology: RingTopology) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(topology, RingTopology):
        raise ValueError("topology must be RingTopology")
    if topology.boundary != "periodic":
        raise ValueError("topology.boundary must be periodic")
    if topology.num_lanes != params.num_lanes:
        raise ValueError("topology.num_lanes must match params.num_lanes")
    if topology.length != params.road_length:
        raise ValueError("topology.length must match params.road_length")
    validate_state(state, params)
    occupancy = build_occupancy(state, params)
    body_occupancy = build_body_occupancy(state, params)
    lane_order, lane_counts, _ = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    vids = controlled_vehicle_ids(state)
    mask = np.zeros((vids.shape[0], 3), dtype=np.bool_)
    for i, vid in enumerate(vids):
        idx = int(np.flatnonzero(state.vehicle_id == vid)[0])
        lane = int(state.lane[idx])
        mask[i, 1] = True
        for col, delta in [(0, -1), (2, 1)]:
            target = lane + delta
            if 0 <= target < params.num_lanes:
                ok, _ = _lateral_valid(state, lane_order, lane_counts, occupancy, body_occupancy, idx, target, params.road_length)
                mask[i, col] = ok
    return vids, mask



def _step_longitudinal_with_controlled_speed_actions_reference(
    state: TrafficState,
    params: SimulationParams,
    topology: RingTopology,
    rng: np.random.Generator,
    normalized: ControlledActionBatch,
) -> tuple[TrafficState, np.ndarray, np.ndarray, np.ndarray]:
    occupancy = build_occupancy(state, params)
    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)

    n = state.n_vehicles
    out_vel = state.vel.astype(np.int64, copy=True)
    v0 = state.vel.astype(np.int64, copy=False)
    prev_pos = (state.pos.astype(np.int64) - v0) % int(params.road_length)
    v_candidate = v0.copy()
    speed_action_by_idx = {}
    vid_to_idx = {int(v): i for i, v in enumerate(state.vehicle_id)}
    high_speed = high_speed_vehicle_mask(state)
    for i, vid in enumerate(normalized.vehicle_id):
        if bool(normalized.speed_delta_provided[i]):
            speed_action_by_idx[vid_to_idx[int(vid)]] = int(normalized.speed_delta[i])

    for i in range(n):
        if not bool(state.alive[i]):
            continue
        lane_i = int(state.lane[i])
        count = int(lane_counts[lane_i])
        rank = int(lane_rank[i])
        vmax_i = int(params.vmax_controlled if high_speed[i] else params.vmax_default)
        if count <= 1:
            g = int(params.road_length - int(state.length[i]))
            leader_v = int(v0[i])
        else:
            leader = int(lane_order[lane_i, (rank + 1) % count])
            g = int((int(state.pos[leader]) - int(state.pos[i]) - int(state.length[i])) % int(params.road_length))
            leader_v = int(v0[leader])

        # Preserve the reference kernel's random draw cadence for all alive
        # vehicles. Explicit controlled speed actions replace stochastic
        # longitudinal update semantics only for the acting controlled vehicle.
        u_s = float(rng.random())
        u_q = float(rng.random())
        u_b = float(rng.random())

        if bool(state.controlled[i]) and i in speed_action_by_idx:
            requested = int(v0[i]) + int(speed_action_by_idx[i])
            v_candidate[i] = max(0, min(int(params.vmax_controlled), requested))
            continue

        s_i = int(params.S if u_s < float(params.r) else 1)
        v1 = min(vmax_i, int(v0[i]) + 1) if (g >= int(params.G) or int(v0[i]) <= leader_v) else int(v0[i])
        prev_gap = compute_cumulative_forward_gap_kernel(i, s_i, prev_pos, state.length, lane_order, lane_counts, lane_rank, road_length=params.road_length)
        v2 = min(v1, int(prev_gap)) if u_q < float(params.q) else v1
        cur_gap = compute_cumulative_forward_gap_kernel(i, s_i, state.pos, state.length, lane_order, lane_counts, lane_rank, road_length=params.road_length)
        v3 = min(v2, int(cur_gap))

        if g >= int(params.G):
            p_i = float(params.P1)
        elif int(v0[i]) < leader_v:
            p_i = float(params.P2)
        elif int(v0[i]) == leader_v:
            p_i = float(params.P3)
        else:
            p_i = float(params.P4)

        if u_b < (1.0 - p_i):
            next_v = max(v3 - 1, 1) if v3 > 0 else 0
        else:
            next_v = v3
        v_candidate[i] = max(0, min(vmax_i, int(next_v)))

    v_safe = v_candidate.copy()
    for lane_i in range(lane_order.shape[0]):
        count = int(lane_counts[lane_i])
        if count <= 1:
            continue
        for _ in range(count):
            for rr in range(count - 1, -1, -1):
                i = int(lane_order[lane_i, rr])
                j = int(lane_order[lane_i, (rr + 1) % count])
                if not bool(state.alive[i]):
                    continue
                g = int((int(state.pos[j]) - int(state.pos[i]) - int(state.length[i])) % int(params.road_length))
                v_safe[i] = min(int(v_safe[i]), g + int(v_safe[j]))

    out_vel[state.alive] = np.maximum(0, v_safe[state.alive])
    new_state = state.copy()
    new_state.changed_lane.fill(False)
    new_state.last_lane_delta.fill(0)
    new_state.vel = out_vel.astype(state.vel.dtype, copy=False)
    new_state.pos = advance_positions_kernel(state.pos, new_state.vel, state.alive, road_length=params.road_length)

    validate_state(new_state, params)
    build_occupancy(new_state, params)
    build_body_occupancy(new_state, params)

    desired = np.zeros(normalized.vehicle_id.shape[0], dtype=state.vel.dtype)
    applied = np.zeros(normalized.vehicle_id.shape[0], dtype=state.vel.dtype)
    clipped = np.zeros(normalized.vehicle_id.shape[0], dtype=np.bool_)
    for row, vid in enumerate(normalized.vehicle_id):
        idx = vid_to_idx[int(vid)]
        desired[row] = np.asarray(v_candidate[idx], dtype=state.vel.dtype)
        applied[row] = new_state.vel[idx]
        clipped[row] = bool(normalized.speed_delta_provided[row] and int(v_candidate[idx]) > int(new_state.vel[idx]))

    return new_state, desired, applied, clipped


def _step_longitudinal_with_controlled_speed_actions_optimized(
    state: TrafficState,
    params: SimulationParams,
    topology: RingTopology,
    rng: np.random.Generator,
    normalized: ControlledActionBatch,
) -> tuple[TrafficState, np.ndarray, np.ndarray, np.ndarray, bool]:
    if not LONG_NUMBA_AVAILABLE or np.any(state.length[state.alive] != 1):
        out, desired, applied, clipped = _step_longitudinal_with_controlled_speed_actions_reference(
            state, params, topology, rng, normalized
        )
        return out, desired, applied, clipped, True

    occupancy = build_occupancy(state, params)
    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    n = state.n_vehicles
    speed_delta_by_idx = np.zeros(n, dtype=np.int8)
    has_speed_delta_by_idx = np.zeros(n, dtype=np.bool_)
    vid_to_idx = {int(v): i for i, v in enumerate(state.vehicle_id)}
    for row, vid in enumerate(normalized.vehicle_id):
        idx = vid_to_idx[int(vid)]
        if bool(normalized.speed_delta_provided[row]):
            speed_delta_by_idx[idx] = np.int8(normalized.speed_delta[row])
            has_speed_delta_by_idx[idx] = True

    u_s, u_q, u_b = draw_longitudinal_randoms(state.alive, rng)
    new_vel, desired_by_idx = compute_longitudinal_velocities_controlled_speed_numba(
        state.lane,
        state.pos,
        state.vel,
        state.alive,
        high_speed_vehicle_mask(state),
        has_speed_delta_by_idx,
        speed_delta_by_idx,
        lane_order,
        lane_counts,
        lane_rank,
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

    new_state = state.copy()
    new_state.changed_lane.fill(False)
    new_state.last_lane_delta.fill(0)
    new_state.vel = new_vel.astype(state.vel.dtype, copy=False)
    new_state.pos = advance_positions_numba(state.pos, new_state.vel, state.alive, road_length=params.road_length)

    validate_state(new_state, params)
    build_occupancy(new_state, params)
    build_body_occupancy(new_state, params)

    desired = np.zeros(normalized.vehicle_id.shape[0], dtype=state.vel.dtype)
    applied = np.zeros(normalized.vehicle_id.shape[0], dtype=state.vel.dtype)
    clipped = np.zeros(normalized.vehicle_id.shape[0], dtype=np.bool_)
    for row, vid in enumerate(normalized.vehicle_id):
        idx = vid_to_idx[int(vid)]
        desired[row] = np.asarray(desired_by_idx[idx], dtype=state.vel.dtype)
        applied[row] = new_state.vel[idx]
        clipped[row] = bool(normalized.speed_delta_provided[row] and int(desired_by_idx[idx]) > int(new_state.vel[idx]))

    return new_state, desired, applied, clipped, False


def _step_with_controlled_lateral_actions(
    state: TrafficState,
    params: SimulationParams,
    topology: RingTopology,
    rng: np.random.Generator,
    actions,
    *,
    require_all_controlled: bool,
    optimized_speed_longitudinal: bool,
):
    if not isinstance(topology, RingTopology):
        raise ValueError("topology must be RingTopology")
    if topology.boundary != "periodic":
        raise ValueError("topology.boundary must be periodic")
    if topology.num_lanes != params.num_lanes:
        raise ValueError("topology.num_lanes must match params.num_lanes")
    if topology.length != params.road_length:
        raise ValueError("topology.length must match params.road_length")
    validate_state(state, params)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    normalized = normalize_controlled_actions(actions, state, require_all_controlled=require_all_controlled)
    occupancy = build_occupancy(state, params)
    body_occupancy = build_body_occupancy(state, params)
    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    front_id, _back_id, front_gap, _ = compute_neighbors(state, lane_order, lane_counts, lane_rank, topology)

    vid_to_idx = {int(v): i for i, v in enumerate(state.vehicle_id)}
    proposals = {}
    valid = np.zeros(normalized.vehicle_id.shape[0], dtype=np.bool_)
    applied = np.zeros(normalized.vehicle_id.shape[0], dtype=np.bool_)
    applied_delta = np.zeros(normalized.vehicle_id.shape[0], dtype=np.int8)
    reasons = [""] * normalized.vehicle_id.shape[0]
    for i, vid in enumerate(normalized.vehicle_id):
        idx = vid_to_idx[int(vid)]
        req = int(normalized.lane_delta[i])
        if req == 0:
            valid[i] = True
            continue
        target = int(state.lane[idx]) + req
        if target < 0 or target >= params.num_lanes:
            reasons[i] = "out_of_bounds"
            continue
        ok, reason = _lateral_valid(state, lane_order, lane_counts, occupancy, body_occupancy, idx, target, params.road_length)
        if not ok:
            reasons[i] = reason
            continue
        valid[i] = True
        proposals.setdefault((target, int(state.pos[idx])), []).append((i, idx, target))

    accepted_controlled = {}
    reserved_targets = set()
    reserved_body_cells = set()
    for key, vals in proposals.items():
        if len(vals) > 1:
            for i, _, _ in vals:
                valid[i] = False
                reasons[i] = "controlled_conflict"
            continue

        i, idx, target = vals[0]
        candidate_cells = {
            (target, (int(state.pos[idx]) + d) % params.road_length)
            for d in range(int(state.length[idx]))
        }
        if not candidate_cells.isdisjoint(reserved_body_cells):
            valid[i] = False
            reasons[i] = "controlled_conflict"
            continue

        accepted_controlled[idx] = target
        reserved_targets.add(key)
        reserved_body_cells.update(candidate_cells)
        applied[i] = True
        applied_delta[i] = np.int8(target - int(state.lane[idx]))

    alive_uncontrolled = state.alive & ~state.controlled
    proposals_un = collect_lane_change_proposals_kernel(
        state.lane, state.pos, state.vel, state.length, alive_uncontrolled, high_speed_vehicle_mask(state), body_occupancy, lane_order, lane_counts,
        front_id, front_gap, num_lanes=params.num_lanes, road_length=params.road_length, vmax_default=params.vmax_default,
        vmax_controlled=params.vmax_controlled, p_lane_change=params.p_lane_change, rng=rng,
    )
    proposals_un = {k: v for k, v in proposals_un.items() if k not in reserved_targets}
    accepted_un = resolve_lane_change_conflicts_kernel(
        proposals_un,
        rng,
        pos=state.pos,
        length=state.length,
        road_length=params.road_length,
        reserved_body_cells=reserved_body_cells,
    )

    accepted_all = dict(accepted_un)
    accepted_all.update(accepted_controlled)
    new_lane, new_changed_lane, new_last_lane_delta = apply_lane_changes_kernel(state.lane, state.changed_lane, state.last_lane_delta, accepted_all)
    after_lane_change = state.copy()
    after_lane_change.lane = new_lane
    after_lane_change.changed_lane = new_changed_lane
    after_lane_change.last_lane_delta = new_last_lane_delta
    build_body_occupancy(after_lane_change, params)
    saved_changed = after_lane_change.changed_lane.copy()
    saved_delta = after_lane_change.last_lane_delta.copy()
    requested_speed_delta = None
    desired_velocity = None
    applied_velocity = None
    speed_clipped_by_safety = None
    used_reference_longitudinal = True
    if np.any(normalized.speed_delta_provided):
        if optimized_speed_longitudinal:
            out, desired_velocity, applied_velocity, speed_clipped_by_safety, used_reference_longitudinal = _step_longitudinal_with_controlled_speed_actions_optimized(
                after_lane_change, params, topology, rng, normalized
            )
        else:
            out, desired_velocity, applied_velocity, speed_clipped_by_safety = _step_longitudinal_with_controlled_speed_actions_reference(
                after_lane_change, params, topology, rng, normalized
            )
        requested_speed_delta = normalized.speed_delta.copy()
    else:
        out = step_longitudinal_reference(after_lane_change, params, topology, rng)
    out = out.copy()
    out.changed_lane = saved_changed
    out.last_lane_delta = saved_delta
    build_occupancy(out, params)

    result = ControlledActionResult(
        vehicle_id=normalized.vehicle_id.copy(),
        requested_lane_delta=normalized.lane_delta.copy(),
        applied_lane_delta=applied_delta,
        valid=valid,
        applied=applied,
        rejection_reason=tuple(reasons),
        requested_speed_delta=requested_speed_delta,
        desired_velocity=desired_velocity,
        applied_velocity=applied_velocity,
        speed_clipped_by_safety=speed_clipped_by_safety,
    )
    return out, result, used_reference_longitudinal


def step_with_controlled_lateral_actions_reference(state: TrafficState, params: SimulationParams, topology: RingTopology, rng: np.random.Generator, actions, *, require_all_controlled: bool = True):
    out, result, _ = _step_with_controlled_lateral_actions(
        state,
        params,
        topology,
        rng,
        actions,
        require_all_controlled=require_all_controlled,
        optimized_speed_longitudinal=False,
    )
    return out, result


def step_with_controlled_lateral_actions_optimized(state: TrafficState, params: SimulationParams, topology: RingTopology, rng: np.random.Generator, actions, *, require_all_controlled: bool = True):
    return _step_with_controlled_lateral_actions(
        state,
        params,
        topology,
        rng,
        actions,
        require_all_controlled=require_all_controlled,
        optimized_speed_longitudinal=True,
    )


def normalize_lateral_overrides(actions: Mapping[int, int], state: TrafficState) -> LaneActionBatch:
    """Normalize rule overrides keyed by any alive vehicle id."""

    if not isinstance(actions, Mapping):
        raise TypeError("lateral overrides must be a mapping of vehicle_id to lane_delta")
    alive_ids = {int(vehicle_id) for vehicle_id in state.vehicle_id[state.alive]}
    normalized: list[tuple[int, int]] = []
    for raw_vehicle_id, raw_delta in actions.items():
        if isinstance(raw_vehicle_id, (bool, np.bool_)) or not isinstance(raw_vehicle_id, (int, np.integer)):
            raise ValueError("lateral override vehicle ids must be integers")
        vehicle_id = int(raw_vehicle_id)
        if vehicle_id not in alive_ids:
            raise ValueError(f"lateral override vehicle_id {vehicle_id} is not alive")
        delta = _validate_action_delta("lane_delta", raw_delta, allow_none=False)
        normalized.append((vehicle_id, int(delta)))
    normalized.sort(key=lambda item: item[0])
    ids = np.ascontiguousarray(np.asarray([item[0] for item in normalized], dtype=np.int32))
    deltas = np.ascontiguousarray(np.asarray([item[1] for item in normalized], dtype=np.int8))
    return LaneActionBatch(ids, deltas)


def _step_longitudinal_after_override(
    state: TrafficState,
    params: SimulationParams,
    topology: RingTopology,
    rng: np.random.Generator,
    *,
    optimized: bool,
) -> tuple[TrafficState, bool]:
    if not optimized or not LONG_NUMBA_AVAILABLE or np.any(state.length[state.alive] != 1):
        return step_longitudinal_reference(state, params, topology, rng), True

    occupancy = build_occupancy(state, params)
    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    u_s, u_q, u_b = draw_longitudinal_randoms(state.alive, rng)
    new_vel = compute_longitudinal_velocities_numba(
        state.lane,
        state.pos,
        state.vel,
        state.alive,
        high_speed_vehicle_mask(state),
        lane_order,
        lane_counts,
        lane_rank,
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
    out = state.copy()
    out.changed_lane.fill(False)
    out.last_lane_delta.fill(0)
    out.vel = new_vel.astype(state.vel.dtype, copy=False)
    out.pos = advance_positions_numba(state.pos, out.vel, state.alive, road_length=params.road_length)
    validate_state(out, params)
    build_occupancy(out, params)
    build_body_occupancy(out, params)
    return out, False


def _step_with_lateral_overrides(
    state: TrafficState,
    params: SimulationParams,
    topology: RingTopology,
    rng: np.random.Generator,
    actions: Mapping[int, int],
    *,
    optimized: bool,
) -> tuple[TrafficState, LateralOverrideResult, bool]:
    """Overlay explicit requests on the native Revised S-NFS lateral phase."""

    if not isinstance(topology, RingTopology):
        raise ValueError("topology must be RingTopology")
    if topology.boundary != "periodic":
        raise ValueError("topology.boundary must be periodic")
    if topology.num_lanes != params.num_lanes or topology.length != params.road_length:
        raise ValueError("topology must match params")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    validate_state(state, params)
    normalized = normalize_lateral_overrides(actions, state)

    occupancy = build_occupancy(state, params)
    body_occupancy = build_body_occupancy(state, params)
    lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles)
    front_id, _back_id, front_gap, _back_gap = compute_neighbors(
        state, lane_order, lane_counts, lane_rank, topology
    )
    high_speed = high_speed_vehicle_mask(state)
    vid_to_idx = {int(vehicle_id): idx for idx, vehicle_id in enumerate(state.vehicle_id)}

    valid = np.zeros(normalized.vehicle_id.shape, dtype=np.bool_)
    applied = np.zeros(normalized.vehicle_id.shape, dtype=np.bool_)
    applied_delta = np.zeros(normalized.vehicle_id.shape, dtype=np.int8)
    reasons = [""] * normalized.vehicle_id.size
    explicit_proposals: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    native_alive = state.alive.copy()

    for row, (vehicle_id, requested_delta) in enumerate(
        zip(normalized.vehicle_id, normalized.lane_delta)
    ):
        idx = vid_to_idx[int(vehicle_id)]
        native_alive[idx] = False
        delta = int(requested_delta)
        if delta == LANE_STAY:
            valid[row] = True
            continue
        target_lane = int(state.lane[idx]) + delta
        if target_lane < 0 or target_lane >= params.num_lanes:
            reasons[row] = "out_of_bounds"
            continue
        ok, reason = _lateral_valid(
            state,
            lane_order,
            lane_counts,
            occupancy,
            body_occupancy,
            idx,
            target_lane,
            params.road_length,
        )
        if not ok:
            reasons[row] = reason
            continue
        valid[row] = True
        explicit_proposals.setdefault((target_lane, int(state.pos[idx])), []).append(
            (row, idx, target_lane)
        )

    accepted_explicit: dict[int, int] = {}
    reserved_targets: set[tuple[int, int]] = set()
    reserved_body_cells: set[tuple[int, int]] = set()
    for key, candidates in explicit_proposals.items():
        if len(candidates) > 1:
            for row, _idx, _target_lane in candidates:
                valid[row] = False
                reasons[row] = "override_conflict"
            continue
        row, idx, target_lane = candidates[0]
        candidate_cells = {
            (target_lane, (int(state.pos[idx]) + offset) % params.road_length)
            for offset in range(int(state.length[idx]))
        }
        if not candidate_cells.isdisjoint(reserved_body_cells):
            valid[row] = False
            reasons[row] = "override_conflict"
            continue
        accepted_explicit[idx] = target_lane
        reserved_targets.add(key)
        reserved_body_cells.update(candidate_cells)
        applied[row] = True
        applied_delta[row] = np.int8(target_lane - int(state.lane[idx]))

    unit_length = not np.any(state.length[state.alive] != 1)
    used_reference_lane = not (optimized and LANE_NUMBA_AVAILABLE and unit_length)
    if used_reference_lane:
        native_proposals = collect_lane_change_proposals_kernel(
            state.lane,
            state.pos,
            state.vel,
            state.length,
            native_alive,
            high_speed,
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
    else:
        native_proposals = collect_lane_change_proposals_numba(
            lane=state.lane,
            pos=state.pos,
            vel=state.vel,
            alive=native_alive,
            controlled=high_speed,
            occupancy=occupancy,
            lane_order=lane_order,
            lane_counts=lane_counts,
            front_id=front_id,
            front_gap=front_gap,
            num_lanes=params.num_lanes,
            road_length=params.road_length,
            vmax_default=params.vmax_default,
            vmax_controlled=params.vmax_controlled,
            p_lane_change=params.p_lane_change,
            rng=rng,
        )
    native_proposals = {
        key: candidates for key, candidates in native_proposals.items() if key not in reserved_targets
    }
    accepted_native = resolve_lane_change_conflicts_kernel(
        native_proposals,
        rng,
        pos=state.pos,
        length=state.length,
        road_length=params.road_length,
        reserved_body_cells=reserved_body_cells,
    )
    accepted = dict(accepted_native)
    accepted.update(accepted_explicit)
    new_lane, new_changed, new_delta = apply_lane_changes_kernel(
        state.lane, state.changed_lane, state.last_lane_delta, accepted
    )
    after_lane = state.copy()
    after_lane.lane = new_lane
    after_lane.changed_lane = new_changed
    after_lane.last_lane_delta = new_delta
    build_body_occupancy(after_lane, params)

    saved_changed = after_lane.changed_lane.copy()
    saved_delta = after_lane.last_lane_delta.copy()
    out, used_reference_longitudinal = _step_longitudinal_after_override(
        after_lane, params, topology, rng, optimized=optimized
    )
    out = out.copy()
    out.changed_lane = saved_changed
    out.last_lane_delta = saved_delta
    validate_state(out, params)
    build_occupancy(out, params)
    build_body_occupancy(out, params)

    result = LateralOverrideResult(
        vehicle_id=normalized.vehicle_id.copy(),
        requested_lane_delta=normalized.lane_delta.copy(),
        applied_lane_delta=applied_delta,
        valid=valid,
        applied=applied,
        rejection_reason=tuple(reasons),
    )
    return out, result, bool(used_reference_lane or used_reference_longitudinal)


def step_with_lateral_overrides_reference(
    state: TrafficState,
    params: SimulationParams,
    topology: RingTopology,
    rng: np.random.Generator,
    actions: Mapping[int, int],
) -> tuple[TrafficState, LateralOverrideResult]:
    out, result, _used_reference = _step_with_lateral_overrides(
        state, params, topology, rng, actions, optimized=False
    )
    return out, result


def step_with_lateral_overrides_optimized(
    state: TrafficState,
    params: SimulationParams,
    topology: RingTopology,
    rng: np.random.Generator,
    actions: Mapping[int, int],
) -> tuple[TrafficState, LateralOverrideResult, bool]:
    return _step_with_lateral_overrides(state, params, topology, rng, actions, optimized=True)
