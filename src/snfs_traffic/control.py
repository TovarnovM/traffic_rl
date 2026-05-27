from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from snfs_traffic.core import SimulationParams, TrafficState
from snfs_traffic.core.indexing import MISSING_INDEX, build_lane_order, build_occupancy, compute_neighbors
from snfs_traffic.core.lane_change_kernels import (
    apply_lane_changes_kernel,
    collect_lane_change_proposals_kernel,
    resolve_lane_change_conflicts_kernel,
    target_lane_neighbors_at_pos_kernel,
)
from snfs_traffic.core.state import validate_state
from snfs_traffic.core.step_reference import step_longitudinal_reference
from snfs_traffic.topology import RingTopology

LANE_LEFT = -1
LANE_STAY = 0
LANE_RIGHT = 1
LANE_ACTION_VALUES = (LANE_LEFT, LANE_STAY, LANE_RIGHT)


@dataclass(frozen=True, slots=True)
class LaneActionBatch:
    vehicle_id: np.ndarray
    lane_delta: np.ndarray

    def __post_init__(self) -> None:
        if self.vehicle_id.ndim != 1 or self.lane_delta.ndim != 1 or self.vehicle_id.shape != self.lane_delta.shape:
            raise ValueError("vehicle_id and lane_delta must be 1D with same shape")
        if len(np.unique(self.vehicle_id)) != self.vehicle_id.size:
            raise ValueError("vehicle_id values must be unique")
        if not np.all(np.isin(self.lane_delta, np.array(LANE_ACTION_VALUES, dtype=self.lane_delta.dtype))):
            raise ValueError("lane_delta values must be in {-1, 0, +1}")

    @classmethod
    def from_mapping(cls, actions: Mapping[int, int]) -> "LaneActionBatch":
        ids = np.asarray(list(actions.keys()), dtype=np.int32)
        deltas = np.asarray(list(actions.values()), dtype=np.int8)
        return cls(np.ascontiguousarray(ids), np.ascontiguousarray(deltas))


@dataclass(frozen=True, slots=True)
class ControlledActionResult:
    vehicle_id: np.ndarray
    requested_lane_delta: np.ndarray
    applied_lane_delta: np.ndarray
    valid: np.ndarray
    applied: np.ndarray
    rejection_reason: tuple[str, ...]


def controlled_vehicle_ids(state: TrafficState) -> np.ndarray:
    return state.vehicle_id[state.alive & state.controlled].copy()


def normalize_lane_actions(actions, state: TrafficState, *, require_all_controlled: bool) -> LaneActionBatch:
    if isinstance(actions, Mapping):
        batch = LaneActionBatch.from_mapping(actions)
    elif isinstance(actions, LaneActionBatch):
        batch = actions
    else:
        raise TypeError("actions must be Mapping[int, int] or LaneActionBatch")

    ids = controlled_vehicle_ids(state)
    alive_ids = set(int(v) for v in ids)
    if len(np.unique(batch.vehicle_id)) != batch.vehicle_id.size:
        raise ValueError("duplicate vehicle_id actions")
    for v in batch.vehicle_id:
        if int(v) not in alive_ids:
            raise ValueError(f"action vehicle_id {int(v)} is not an alive controlled vehicle")
    action_map = {int(v): int(d) for v, d in zip(batch.vehicle_id, batch.lane_delta)}
    if require_all_controlled and len(action_map) != len(ids):
        raise ValueError("missing actions for alive controlled vehicles")
    out_delta = np.zeros(ids.shape[0], dtype=np.int8)
    for i, vid in enumerate(ids):
        if int(vid) in action_map:
            out_delta[i] = np.int8(action_map[int(vid)])
        elif require_all_controlled:
            raise ValueError("missing actions for alive controlled vehicles")
    return LaneActionBatch(np.ascontiguousarray(ids.astype(state.vehicle_id.dtype)), np.ascontiguousarray(out_delta))


def _lateral_valid(state, lane_order, lane_counts, occupancy, idx: int, target_lane: int, road_length: int):
    pos_i = int(state.pos[idx])
    vel_i = int(state.vel[idx])
    if int(occupancy[target_lane, pos_i]) != MISSING_INDEX:
        return False, "target_occupied"
    if int(lane_counts[target_lane]) == 0:
        return True, ""
    t_front, t_back, _g_nf, g_nb = target_lane_neighbors_at_pos_kernel(
        state.pos, lane_order, lane_counts, target_lane=target_lane, candidate_pos=pos_i, road_length=road_length
    )
    if t_front == MISSING_INDEX:
        return True, ""
    safe = vel_i > (int(state.vel[t_back]) - int(g_nb))
    return (bool(safe), "" if safe else "unsafe")


def compute_lateral_action_mask(state: TrafficState, params: SimulationParams, topology: RingTopology) -> tuple[np.ndarray, np.ndarray]:
    validate_state(state, params)
    occupancy = build_occupancy(state, params)
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
                ok, _ = _lateral_valid(state, lane_order, lane_counts, occupancy, idx, target, params.road_length)
                mask[i, col] = ok
    return vids, mask


def step_with_controlled_lateral_actions_reference(state: TrafficState, params: SimulationParams, topology: RingTopology, rng: np.random.Generator, actions, *, require_all_controlled: bool = True):
    validate_state(state, params)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    normalized = normalize_lane_actions(actions, state, require_all_controlled=require_all_controlled)
    occupancy = build_occupancy(state, params)
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
        ok, reason = _lateral_valid(state, lane_order, lane_counts, occupancy, idx, target, params.road_length)
        if not ok:
            reasons[i] = reason
            continue
        valid[i] = True
        proposals.setdefault((target, int(state.pos[idx])), []).append((i, idx, target))

    accepted_controlled = {}
    reserved_targets = set()
    for key, vals in proposals.items():
        if len(vals) > 1:
            for i, _, _ in vals:
                valid[i] = False
                reasons[i] = "controlled_conflict"
        else:
            i, idx, target = vals[0]
            accepted_controlled[idx] = target
            reserved_targets.add(key)
            applied[i] = True
            applied_delta[i] = np.int8(target - int(state.lane[idx]))

    alive_uncontrolled = state.alive & ~state.controlled
    proposals_un = collect_lane_change_proposals_kernel(
        state.lane, state.pos, state.vel, alive_uncontrolled, state.controlled, occupancy, lane_order, lane_counts,
        front_id, front_gap, num_lanes=params.num_lanes, road_length=params.road_length, vmax_default=params.vmax_default,
        vmax_controlled=params.vmax_controlled, p_lane_change=params.p_lane_change, rng=rng,
    )
    proposals_un = {k: v for k, v in proposals_un.items() if k not in reserved_targets}
    accepted_un = resolve_lane_change_conflicts_kernel(proposals_un, rng)

    accepted_all = dict(accepted_un)
    accepted_all.update(accepted_controlled)
    new_lane, new_changed_lane, new_last_lane_delta = apply_lane_changes_kernel(state.lane, state.changed_lane, state.last_lane_delta, accepted_all)
    after_lane_change = state.copy()
    after_lane_change.lane = new_lane
    after_lane_change.changed_lane = new_changed_lane
    after_lane_change.last_lane_delta = new_last_lane_delta
    saved_changed = after_lane_change.changed_lane.copy()
    saved_delta = after_lane_change.last_lane_delta.copy()
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
    )
    return out, result
