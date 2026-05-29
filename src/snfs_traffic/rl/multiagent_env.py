from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised when optional extra is absent
    raise ImportError(
        "snfs_traffic.rl.multiagent_env requires Gymnasium; install it with "
        "'python -m pip install -e \".[rl]\"' from this repository, or install 'gymnasium>=0.29'."
    ) from exc

from snfs_traffic.control import LANE_LEFT, LANE_RIGHT, LANE_STAY, ControlledActionResult
from snfs_traffic.core import SimulationParams, TrafficState
from snfs_traffic.observations import LOCAL_OBSERVATION_FEATURES, LocalObservationConfig
from snfs_traffic.rl.episode import EpisodeConfig, build_reset_info, build_step_info, compute_episode_flags
from snfs_traffic.rl.rewards import RewardConfig, compute_controlled_reward
from snfs_traffic.scenarios import VehicleMix
from snfs_traffic.simulator import ScenarioConfig, TrafficSimulator

_ACTION_TO_DELTA = {0: LANE_STAY, 1: LANE_LEFT, 2: LANE_RIGHT}


class SnfsTrafficMultiAgentEnv(gym.Env):
    """Ray-shaped synchronous multi-agent wrapper for controlled vehicles."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        num_lanes: int = 3,
        road_length: int = 100,
        density: float = 0.2,
        num_controlled: int = 2,
        controlled_vehicle_ids: Sequence[int] | None = None,
        backend: str = "reference",
        episode_config: EpisodeConfig | None = None,
        reward_config: RewardConfig | None = None,
        seed: int | None = None,
        scenario_seed: int | None = None,
    ) -> None:
        if isinstance(num_controlled, bool) or not isinstance(num_controlled, int) or num_controlled < 1:
            raise ValueError("num_controlled must be an int >= 1")
        requested_ids: tuple[int, ...] | None = None
        if controlled_vehicle_ids is not None:
            if isinstance(controlled_vehicle_ids, (str, bytes)):
                raise ValueError("controlled_vehicle_ids must be a non-empty iterable of unique integer vehicle ids")
            try:
                requested = tuple(controlled_vehicle_ids)
            except TypeError as exc:
                raise ValueError("controlled_vehicle_ids must be a non-empty iterable of unique integer vehicle ids") from exc
            if not requested:
                raise ValueError("controlled_vehicle_ids must be non-empty when provided")
            if any(isinstance(v, bool) or not isinstance(v, int) for v in requested):
                raise ValueError("controlled_vehicle_ids must contain integer vehicle ids")
            if len(set(requested)) != len(requested):
                raise ValueError("controlled_vehicle_ids must contain unique vehicle ids")
            requested_ids = tuple(int(v) for v in requested)

        self.params = SimulationParams(num_lanes=num_lanes, road_length=road_length)
        self.episode_config = episode_config or EpisodeConfig()
        self.reward_config = reward_config or RewardConfig()
        self._density = density
        self._backend = backend
        self._initial_seed = seed
        self._scenario_seed = scenario_seed
        self._num_controlled = int(num_controlled)
        self._requested_controlled_vehicle_ids = requested_ids
        self._selected_controlled_count = len(requested_ids) if requested_ids is not None else int(num_controlled)
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

        capacity = self.params.num_lanes * self.params.road_length
        self.possible_agents: list[str] = [f"vehicle_{vehicle_id}" for vehicle_id in range(capacity)]
        self.agents: list[str] = []
        self._agent_to_vehicle_id: dict[str, int] = {}
        self._vehicle_id_to_agent: dict[int, str] = {}
        self._controlled_vehicle_ids: tuple[int, ...] = ()

        self.single_agent_action_space = spaces.Discrete(3)
        obs_len = len(LOCAL_OBSERVATION_FEATURES)
        self.single_agent_observation_space = spaces.Dict(
            {
                "obs": spaces.Box(low=-1.0, high=1.0, shape=(obs_len,), dtype=np.float32),
                "action_mask": spaces.MultiBinary(3),
            }
        )
        self.action_space = self.single_agent_action_space
        self.observation_space = self.single_agent_observation_space
        self.observation_spaces = {}
        self.action_spaces = {}

    @property
    def backend_name(self) -> str:
        return self._sim.backend_name

    def get_observation_space(self, agent_id: str):
        self._validate_agent_id_format(agent_id)
        return self.single_agent_observation_space

    def get_action_space(self, agent_id: str):
        self._validate_agent_id_format(agent_id)
        return self.single_agent_action_space

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
        state = self._select_controlled(state)
        state = self._sim.reset(state=state, rng_seed=int(rng_seed))
        self._step_index = 0
        self._done = False
        observations = self._build_observations()
        infos = {}
        for agent_id in self.agents:
            vehicle_id = self._agent_to_vehicle_id[agent_id]
            info = build_reset_info(
                state,
                controlled_vehicle_id=vehicle_id,
                step_index=self._step_index,
                backend_name=self._sim.backend_name,
                params=self.params,
            )
            info["agent_id"] = agent_id
            infos[agent_id] = info
        return observations, infos

    def step(self, action_dict):
        if self._done:
            raise RuntimeError("episode is done; call reset() before step()")
        if not isinstance(action_dict, Mapping):
            raise ValueError("action_dict must be a mapping keyed by active agent id")

        expected_agents = set(self.agents)
        provided_agents = set(action_dict.keys())
        missing = sorted(expected_agents - provided_agents, key=repr)
        extra = sorted(provided_agents - expected_agents, key=repr)
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing actions for active agents {missing}")
            if extra:
                parts.append(f"unknown/inactive extra agent actions {extra}")
            raise ValueError("; ".join(parts))

        sim_actions: dict[int, int] = {}
        for agent_id in self.agents:
            action = action_dict[agent_id]
            if not self.single_agent_action_space.contains(action):
                raise ValueError(f"invalid action {action!r} for agent {agent_id}; expected Discrete(3) action 0=keep, 1=left, 2=right")
            action_int = int(action)
            sim_actions[self._agent_to_vehicle_id[agent_id]] = int(_ACTION_TO_DELTA[action_int])

        prev_agents = list(self.agents)
        prev_agent_to_vehicle_id = dict(self._agent_to_vehicle_id)
        prev_state = self._sim.state
        next_state = self._sim.step(sim_actions)
        self._step_index += 1

        rewards: dict[str, float] = {}
        terminateds: dict[str, bool] = {}
        truncateds: dict[str, bool] = {}
        infos: dict[str, dict[str, object]] = {}
        backend_name = self._sim.last_step_info.backend_name if self._sim.last_step_info is not None else self._sim.backend_name
        for agent_id in prev_agents:
            vehicle_id = prev_agent_to_vehicle_id[agent_id]
            action_applied = self._action_applied(vehicle_id)
            reward, components = self._compute_agent_reward(
                prev_state,
                next_state,
                agent_id=agent_id,
                controlled_vehicle_id=vehicle_id,
                action={vehicle_id: sim_actions[vehicle_id]},
                action_applied=action_applied,
            )
            reward = float(reward)
            components = {str(k): float(v) for k, v in components.items()}
            if not np.isfinite(reward) or any(not np.isfinite(v) for v in components.values()):
                raise ValueError("reward must be finite")
            terminated, truncated = compute_episode_flags(
                next_state,
                controlled_vehicle_id=vehicle_id,
                step_index=self._step_index,
                config=self.episode_config,
            )
            rewards[agent_id] = reward
            terminateds[agent_id] = bool(terminated)
            truncateds[agent_id] = bool(truncated)
            info = build_step_info(
                prev_state,
                next_state,
                controlled_vehicle_id=vehicle_id,
                step_index=self._step_index,
                backend_name=backend_name,
                reward_components=components,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            info["agent_id"] = agent_id
            infos[agent_id] = info

        next_active_vehicle_ids = self._active_controlled_vehicle_ids(next_state)
        global_truncated = bool(self._step_index >= self.episode_config.max_steps)
        global_terminated = bool(not global_truncated and len(next_active_vehicle_ids) == 0)
        terminateds["__all__"] = global_terminated
        truncateds["__all__"] = global_truncated

        if global_terminated or global_truncated:
            self.agents = []
            self._agent_to_vehicle_id = {}
            self._vehicle_id_to_agent = {}
        else:
            self._set_active_agents(next_active_vehicle_ids)
        self._done = bool(terminateds["__all__"] or truncateds["__all__"])
        observations = {} if self._done else self._build_observations()
        return observations, rewards, terminateds, truncateds, infos

    def _select_controlled(self, state: TrafficState) -> TrafficState:
        """Select and mark controlled vehicles for the reset state."""
        alive_ids = np.asarray(state.vehicle_id[state.alive], dtype=state.vehicle_id.dtype)
        if alive_ids.size < self._selected_controlled_count:
            raise ValueError(
                f"SnfsTrafficMultiAgentEnv requires {self._selected_controlled_count} alive vehicles; "
                "increase density or road capacity"
            )
        if self._requested_controlled_vehicle_ids is None:
            selected = tuple(int(v) for v in self.np_random.choice(alive_ids, size=self._num_controlled, replace=False))
        else:
            selected = self._requested_controlled_vehicle_ids
            alive_set = {int(v) for v in alive_ids}
            missing = [vehicle_id for vehicle_id in selected if int(vehicle_id) not in alive_set]
            if missing:
                raise ValueError(f"controlled_vehicle_ids {missing} are not present and alive in the reset state")

        selected = tuple(sorted(int(v) for v in selected))
        state = state.copy()
        state.controlled[:] = False
        for vehicle_id in selected:
            matches = np.flatnonzero(state.vehicle_id == int(vehicle_id))
            if matches.size == 0 or not bool(state.alive[int(matches[0])]):
                raise ValueError(f"controlled_vehicle_id {vehicle_id} is not present and alive in the reset state")
            state.controlled[int(matches[0])] = True
        self._controlled_vehicle_ids = selected
        self._set_active_agents(selected)
        return state

    def _build_observations(self) -> dict[str, dict[str, np.ndarray]]:
        """Build per-agent observations for currently active agents."""
        batch = self._sim.observe(self._obs_config)
        row_by_vehicle_id = {int(vehicle_id): i for i, vehicle_id in enumerate(batch.vehicle_id)}
        observations = {}
        obs_shape = self.single_agent_observation_space["obs"].shape
        for agent_id in self.agents:
            vehicle_id = self._agent_to_vehicle_id[agent_id]
            row = row_by_vehicle_id.get(vehicle_id)
            if row is None:
                obs = {
                    "obs": np.zeros(obs_shape, dtype=np.float32),
                    "action_mask": np.zeros(3, dtype=np.int8),
                }
            else:
                obs = {
                    "obs": np.asarray(batch.obs[row], dtype=np.float32),
                    "action_mask": np.asarray(batch.action_mask[row], dtype=np.int8),
                }
            observations[agent_id] = obs
        return observations

    def _compute_agent_reward(
        self,
        prev_state: TrafficState,
        next_state: TrafficState,
        *,
        agent_id: str,
        controlled_vehicle_id: int,
        action: object,
        action_applied: bool | None,
    ) -> tuple[float, dict[str, float]]:
        """Compute one acting agent's reward for an environment step."""
        return compute_controlled_reward(
            prev_state,
            next_state,
            controlled_vehicle_id=controlled_vehicle_id,
            action=action,
            action_applied=action_applied,
            params=self.params,
            config=self.reward_config,
        )

    def _action_applied(self, controlled_vehicle_id: int) -> bool | None:
        info = self._sim.last_step_info
        if info is None or not isinstance(info.action_result, ControlledActionResult):
            return None
        result = info.action_result
        matches = np.flatnonzero(result.vehicle_id == int(controlled_vehicle_id))
        if matches.size == 0:
            return None
        return bool(result.applied[int(matches[0])] or result.requested_lane_delta[int(matches[0])] == LANE_STAY)

    def _active_controlled_vehicle_ids(self, state: TrafficState) -> tuple[int, ...]:
        selected = set(self._controlled_vehicle_ids)
        active = [int(vehicle_id) for vehicle_id, alive, controlled in zip(state.vehicle_id, state.alive, state.controlled) if bool(alive) and bool(controlled) and int(vehicle_id) in selected]
        return tuple(sorted(active))

    def _set_active_agents(self, vehicle_ids: Sequence[int]) -> None:
        ordered = tuple(sorted(int(vehicle_id) for vehicle_id in vehicle_ids))
        self.agents = [f"vehicle_{vehicle_id}" for vehicle_id in ordered]
        self._agent_to_vehicle_id = {f"vehicle_{vehicle_id}": vehicle_id for vehicle_id in ordered}
        self._vehicle_id_to_agent = {vehicle_id: f"vehicle_{vehicle_id}" for vehicle_id in ordered}

    @staticmethod
    def _validate_agent_id_format(agent_id: str) -> int:
        if not isinstance(agent_id, str) or not agent_id.startswith("vehicle_"):
            raise ValueError("agent_id must have format 'vehicle_<vehicle_id>'")
        suffix = agent_id[len("vehicle_") :]
        if not suffix or not suffix.isdecimal():
            raise ValueError("agent_id must have format 'vehicle_<vehicle_id>'")
        return int(suffix)
