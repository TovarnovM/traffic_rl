from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snfs_traffic.core import SimulationParams, TrafficState


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    max_steps: int = 1000
    terminate_on_no_alive_controlled: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError("max_steps must be an int >= 1")


def _controlled_alive(state: TrafficState, controlled_vehicle_id: int) -> bool:
    matches = np.flatnonzero(state.vehicle_id == int(controlled_vehicle_id))
    if matches.size == 0:
        return False
    idx = int(matches[0])
    return bool(state.alive[idx]) and bool(state.controlled[idx])


def compute_episode_flags(
    state: TrafficState,
    *,
    controlled_vehicle_id: int,
    step_index: int,
    config: EpisodeConfig,
) -> tuple[bool, bool]:
    if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
        raise ValueError("step_index must be an int >= 0")
    terminated = bool(config.terminate_on_no_alive_controlled and not _controlled_alive(state, controlled_vehicle_id))
    truncated = bool(step_index >= config.max_steps)
    return terminated, truncated


def build_reset_info(
    state: TrafficState,
    *,
    controlled_vehicle_id: int,
    step_index: int,
    backend_name: str,
    params: SimulationParams,
) -> dict[str, object]:
    return {
        "step_index": int(step_index),
        "controlled_vehicle_id": int(controlled_vehicle_id),
        "backend_name": str(backend_name),
        "num_vehicles": int(state.n_vehicles),
        "num_lanes": int(params.num_lanes),
        "road_length": int(params.road_length),
    }


def build_step_info(
    prev_state: TrafficState,
    next_state: TrafficState,
    *,
    controlled_vehicle_id: int,
    step_index: int,
    backend_name: str,
    reward_components: dict[str, float],
    terminated: bool,
    truncated: bool,
) -> dict[str, object]:
    alive = next_state.alive
    controlled_alive = alive & next_state.controlled
    mean_speed = float(np.mean(next_state.vel[alive])) if bool(np.any(alive)) else 0.0
    mean_controlled_speed = float(np.mean(next_state.vel[controlled_alive])) if bool(np.any(controlled_alive)) else 0.0
    num_lane_changes = int(np.sum(next_state.changed_lane[alive]))
    matches = np.flatnonzero((next_state.vehicle_id == int(controlled_vehicle_id)) & alive & next_state.controlled)
    num_controlled_lane_changes = int(bool(matches.size and next_state.changed_lane[int(matches[0])]))
    return {
        "step_index": int(step_index),
        "controlled_vehicle_id": int(controlled_vehicle_id),
        "backend_name": str(backend_name),
        "reward_components": {str(k): float(v) for k, v in reward_components.items()},
        "mean_speed": mean_speed,
        "mean_controlled_speed": mean_controlled_speed,
        "num_lane_changes": num_lane_changes,
        "num_controlled_lane_changes": num_controlled_lane_changes,
        "terminated_reason": "no_alive_controlled" if terminated else None,
        "truncated_reason": "max_steps" if truncated else None,
    }
