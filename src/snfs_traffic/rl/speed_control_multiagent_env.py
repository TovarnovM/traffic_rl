from __future__ import annotations

import numpy as np

try:
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised when optional extra is absent
    raise ImportError(
        "snfs_traffic.rl.speed_control_multiagent_env requires Gymnasium; install it with "
        "'python -m pip install -e \".[rl]\"' from this repository, or install 'gymnasium>=0.29'."
    ) from exc

from snfs_traffic.control import ControlledActionResult, ControlledVehicleAction, LANE_LEFT, LANE_RIGHT, LANE_STAY
from snfs_traffic.rl.multiagent_env import SnfsTrafficMultiAgentEnv

_LATERAL_ACTION_TO_DELTA = {0: LANE_STAY, 1: LANE_LEFT, 2: LANE_RIGHT}
_SPEED_ACTION_TO_DELTA = {0: -1, 1: 0, 2: 1}


class SnfsTrafficSpeedControlMultiAgentEnv(SnfsTrafficMultiAgentEnv):
    """Multi-agent environment with lateral lane requests and speed deltas."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.single_agent_action_space = spaces.MultiDiscrete([3, 3])
        self.single_agent_observation_space = spaces.Dict(
            {
                "obs": self.single_agent_observation_space["obs"],
                "action_mask": spaces.MultiBinary((2, 3)),
            }
        )
        self.action_space = self.single_agent_action_space
        self.observation_space = self.single_agent_observation_space

    def _convert_agent_action(self, agent_id: str, action: object) -> object:
        if not self.single_agent_action_space.contains(action):
            raise ValueError(
                f"invalid action {action!r} for agent {agent_id}; expected MultiDiscrete([3, 3]) "
                "action [lateral, speed] with lateral 0=keep, 1=left, 2=right and speed 0=brake, 1=keep, 2=accelerate"
            )
        action_array = np.asarray(action)
        lateral_action = int(action_array[0])
        speed_action = int(action_array[1])
        return ControlledVehicleAction(
            lane_delta=int(_LATERAL_ACTION_TO_DELTA[lateral_action]),
            speed_delta=int(_SPEED_ACTION_TO_DELTA[speed_action]),
        )

    def _build_observations(self) -> dict[str, dict[str, np.ndarray]]:
        observations = super()._build_observations()
        for agent_obs in observations.values():
            lateral_mask = np.asarray(agent_obs["action_mask"], dtype=np.int8)
            speed_mask = np.ones(3, dtype=np.int8)
            agent_obs["action_mask"] = np.stack([lateral_mask, speed_mask], axis=0)
        return observations

    def _add_agent_step_info(self, info: dict[str, object], controlled_vehicle_id: int) -> None:
        step_info = self._sim.last_step_info
        if step_info is None or not isinstance(step_info.action_result, ControlledActionResult):
            return
        result = step_info.action_result
        if result.requested_speed_delta is None:
            return
        matches = np.flatnonzero(result.vehicle_id == int(controlled_vehicle_id))
        if matches.size == 0:
            return
        row = int(matches[0])
        info["controlled_speed_delta"] = int(result.requested_speed_delta[row])
        if result.desired_velocity is not None:
            info["desired_velocity"] = int(result.desired_velocity[row])
        if result.applied_velocity is not None:
            info["applied_velocity"] = int(result.applied_velocity[row])
        if result.speed_clipped_by_safety is not None:
            info["speed_clipped_by_safety"] = bool(result.speed_clipped_by_safety[row])
