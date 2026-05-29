from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from snfs_traffic.control import LANE_STAY, LaneActionBatch
from snfs_traffic.core import SimulationParams, TrafficState


@dataclass(frozen=True, slots=True)
class RewardConfig:
    speed_weight: float = 1.0
    lane_change_penalty: float = 0.05
    blocked_action_penalty: float = 0.10
    stopped_penalty: float = 0.0
    normalize_by_vmax: bool = True


def _vehicle_index(state: TrafficState, vehicle_id: int) -> int | None:
    matches = np.flatnonzero(state.vehicle_id == vehicle_id)
    if matches.size == 0:
        return None
    return int(matches[0])


def _requested_lane_delta(action: object, controlled_vehicle_id: int) -> int:
    if action is None:
        return LANE_STAY
    if isinstance(action, LaneActionBatch):
        matches = np.flatnonzero(action.vehicle_id == controlled_vehicle_id)
        if matches.size == 0:
            return LANE_STAY
        return int(action.lane_delta[int(matches[0])])
    if isinstance(action, Mapping):
        return int(action.get(int(controlled_vehicle_id), LANE_STAY))
    return LANE_STAY


def compute_controlled_reward(
    prev_state: TrafficState,
    next_state: TrafficState,
    *,
    controlled_vehicle_id: int,
    action: object,
    action_applied: bool | None,
    params: SimulationParams,
    config: RewardConfig,
) -> tuple[float, dict[str, float]]:
    """Compute deterministic MVP reward components for one controlled vehicle."""
    prev_idx = _vehicle_index(prev_state, int(controlled_vehicle_id))
    next_idx = _vehicle_index(next_state, int(controlled_vehicle_id))
    if next_idx is None or not bool(next_state.alive[next_idx]):
        next_vel = 0.0
        changed_lane = False
    else:
        next_vel = float(next_state.vel[next_idx])
        if prev_idx is not None:
            changed_lane = int(prev_state.lane[prev_idx]) != int(next_state.lane[next_idx])
        else:
            changed_lane = bool(next_state.changed_lane[next_idx])

    speed_basis = next_vel
    if config.normalize_by_vmax:
        vmax = max(float(params.vmax_controlled), 1.0)
        speed_basis = next_vel / vmax
    speed_reward = float(config.speed_weight) * speed_basis

    lane_change = -float(config.lane_change_penalty) if changed_lane else 0.0

    requested_delta = _requested_lane_delta(action, int(controlled_vehicle_id))
    if action_applied is None:
        action_applied = changed_lane if requested_delta != LANE_STAY else True
    blocked = requested_delta != LANE_STAY and not bool(action_applied)
    blocked_action = -float(config.blocked_action_penalty) if blocked else 0.0

    stopped = -float(config.stopped_penalty) if next_vel == 0.0 else 0.0
    total = float(speed_reward + lane_change + blocked_action + stopped)

    components = {
        "speed_reward": float(speed_reward),
        "lane_change_penalty": float(lane_change),
        "blocked_action_penalty": float(blocked_action),
        "stopped_penalty": float(stopped),
        "total": total,
    }
    if not np.isfinite(total) or any(not np.isfinite(v) for v in components.values()):
        raise ValueError("reward must be finite")
    return total, components
