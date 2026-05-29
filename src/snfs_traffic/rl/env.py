from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised when optional extra is absent
    raise ImportError(
        "snfs_traffic.rl.env requires Gymnasium; install it with 'python -m pip install -e \".[rl]\"' "
        "from this repository, or install 'gymnasium>=0.29'."
    ) from exc

from snfs_traffic.control import LANE_LEFT, LANE_RIGHT, LANE_STAY, ControlledActionResult
from snfs_traffic.core import SimulationParams, TrafficState
from snfs_traffic.observations import LOCAL_OBSERVATION_FEATURES, LocalObservationConfig
from snfs_traffic.rl.episode import EpisodeConfig, build_reset_info, build_step_info, compute_episode_flags
from snfs_traffic.rl.rewards import RewardConfig, compute_controlled_reward
from snfs_traffic.scenarios import VehicleMix
from snfs_traffic.simulator import ScenarioConfig, TrafficSimulator

_ACTION_TO_DELTA = {0: LANE_STAY, 1: LANE_LEFT, 2: LANE_RIGHT}


class SnfsTrafficEnv(gym.Env):
    """Minimal Gymnasium wrapper for one controlled vehicle on a periodic ring road."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        num_lanes: int = 3,
        road_length: int = 100,
        density: float = 0.2,
        controlled_vehicle_id: int | None = None,
        backend: str = "reference",
        episode_config: EpisodeConfig | None = None,
        reward_config: RewardConfig | None = None,
        seed: int | None = None,
        scenario_seed: int | None = None,
    ) -> None:
        self.params = SimulationParams(num_lanes=num_lanes, road_length=road_length)
        self.episode_config = episode_config or EpisodeConfig()
        self.reward_config = reward_config or RewardConfig()
        self._density = density
        self._backend = backend
        self._initial_seed = seed
        self._scenario_seed = scenario_seed
        self._requested_controlled_vehicle_id = controlled_vehicle_id
        self._controlled_vehicle_id: int | None = None
        self._step_index = 0
        self._done = True
        self._obs_config = LocalObservationConfig(dtype=np.float32)
        self._sim = TrafficSimulator(
            params=self.params,
            backend=backend,
            rng_seed=0 if seed is None else seed,
            scenario=ScenarioConfig(density=density, seed=0 if scenario_seed is None else scenario_seed, vehicle_mix=VehicleMix()),
            require_all_controlled_actions=True,
        )
        self.action_space = spaces.Discrete(3)
        obs_len = len(LOCAL_OBSERVATION_FEATURES)
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(low=-1.0, high=1.0, shape=(obs_len,), dtype=np.float32),
                "action_mask": spaces.MultiBinary(3),
            }
        )

    @property
    def controlled_vehicle_id(self) -> int:
        if self._controlled_vehicle_id is None:
            raise RuntimeError("environment is not reset")
        return self._controlled_vehicle_id

    @property
    def backend_name(self) -> str:
        return self._sim.backend_name

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        env_seed = seed if seed is not None else self._initial_seed
        super().reset(seed=env_seed)
        options = options or {}
        scenario_seed = options.get("scenario_seed", self._scenario_seed)
        if scenario_seed is None:
            scenario_seed = seed if seed is not None else self._initial_seed
        rng_seed = options.get("rng_seed", seed if seed is not None else self._initial_seed)
        if scenario_seed is None:
            scenario_seed = int(self.np_random.integers(0, np.iinfo(np.int32).max))
        if rng_seed is None:
            rng_seed = int(self.np_random.integers(0, np.iinfo(np.int32).max))

        state = self._sim.reset(seed=int(scenario_seed), rng_seed=int(rng_seed))
        state = self._select_single_controlled(state)
        state = self._sim.reset(state=state, rng_seed=int(rng_seed))
        self._step_index = 0
        self._done = False
        obs = self._build_observation()
        info = build_reset_info(
            state,
            controlled_vehicle_id=self.controlled_vehicle_id,
            step_index=self._step_index,
            backend_name=self._sim.backend_name,
            params=self.params,
        )
        return obs, info

    def step(self, action):
        if self._done:
            raise RuntimeError("episode is done; call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}; expected Discrete(3) action 0=keep, 1=left, 2=right")

        action_int = int(action)
        lane_delta = int(_ACTION_TO_DELTA[action_int])
        vehicle_id = self.controlled_vehicle_id
        prev_state = self._sim.state
        next_state = self._sim.step({vehicle_id: lane_delta})
        self._step_index += 1
        action_applied = self._action_applied(vehicle_id)
        reward, components = self._compute_reward(
            prev_state,
            next_state,
            controlled_vehicle_id=vehicle_id,
            action={vehicle_id: lane_delta},
            action_applied=action_applied,
        )
        terminated, truncated = compute_episode_flags(
            next_state,
            controlled_vehicle_id=vehicle_id,
            step_index=self._step_index,
            config=self.episode_config,
        )
        self._done = bool(terminated or truncated)
        obs = self._build_observation()
        info = build_step_info(
            prev_state,
            next_state,
            controlled_vehicle_id=vehicle_id,
            step_index=self._step_index,
            backend_name=self._sim.last_step_info.backend_name if self._sim.last_step_info is not None else self._sim.backend_name,
            reward_components=components,
            terminated=terminated,
            truncated=truncated,
        )
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _compute_reward(
        self,
        prev_state: TrafficState,
        next_state: TrafficState,
        *,
        controlled_vehicle_id: int,
        action: object,
        action_applied: bool | None,
    ) -> tuple[float, dict[str, float]]:
        """Compute the controlled-vehicle reward for one environment step.

        Subclasses may override this protected hook to customize reward logic
        without copying :meth:`step`. The default implementation is deterministic,
        does not consume environment RNG, and delegates to the MVP reward helper.
        """
        return compute_controlled_reward(
            prev_state,
            next_state,
            controlled_vehicle_id=controlled_vehicle_id,
            action=action,
            action_applied=action_applied,
            params=self.params,
            config=self.reward_config,
        )

    def _build_observation(self) -> dict[str, np.ndarray]:
        """Build the current Gymnasium observation.

        Subclasses that override this hook should update ``observation_space`` to
        match the returned schema. ``_controlled_observation`` remains available
        as the default lower-level helper.
        """
        return self._controlled_observation()

    def _select_single_controlled(self, state: TrafficState) -> TrafficState:
        """Select and mark the single controlled vehicle for a reset state.

        This protected hook is the priority-vehicle selection extension point.
        The default honors a constructor-provided ``controlled_vehicle_id`` or
        otherwise draws one vehicle deterministically from the environment RNG.
        """
        if state.n_vehicles < 1:
            raise ValueError("SnfsTrafficEnv requires at least one vehicle; increase density or road capacity")
        selected = self._requested_controlled_vehicle_id
        if selected is None:
            selected = int(state.vehicle_id[int(self.np_random.integers(0, state.n_vehicles))])
        matches = np.flatnonzero(state.vehicle_id == int(selected))
        if matches.size == 0:
            raise ValueError(f"controlled_vehicle_id {selected} is not present in the reset state")
        state = state.copy()
        state.controlled[:] = False
        state.controlled[int(matches[0])] = True
        self._controlled_vehicle_id = int(selected)
        return state

    def _controlled_observation(self) -> dict[str, np.ndarray]:
        batch = self._sim.observe(self._obs_config)
        matches = np.flatnonzero(batch.vehicle_id == self.controlled_vehicle_id)
        if matches.size == 0:
            return {
                "obs": np.zeros(self.observation_space["obs"].shape, dtype=np.float32),
                "action_mask": np.zeros(3, dtype=np.int8),
            }
        row = int(matches[0])
        return {
            "obs": np.asarray(batch.obs[row], dtype=np.float32),
            "action_mask": np.asarray(batch.action_mask[row], dtype=np.int8),
        }

    def _action_applied(self, controlled_vehicle_id: int) -> bool | None:
        info = self._sim.last_step_info
        if info is None or not isinstance(info.action_result, ControlledActionResult):
            return None
        result = info.action_result
        matches = np.flatnonzero(result.vehicle_id == int(controlled_vehicle_id))
        if matches.size == 0:
            return None
        return bool(result.applied[int(matches[0])] or result.requested_lane_delta[int(matches[0])] == LANE_STAY)
