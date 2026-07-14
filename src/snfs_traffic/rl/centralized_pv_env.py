"""Single-agent Gymnasium environment coordinating AVs around one PV."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise ImportError(
        "snfs_traffic.rl.centralized_pv_env requires Gymnasium; install local "
        "training dependencies with 'python -m pip install -e \".[rl,ray,numba]\"'"
    ) from exc

from snfs_traffic.control import LateralOverrideResult
from snfs_traffic.core import SimulationParams
from snfs_traffic.observations import (
    PV_GRAPH_EDGE_FEATURES,
    PV_GRAPH_GLOBAL_FEATURES,
    PV_GRAPH_NEIGHBOR_RELATIONS,
    PV_GRAPH_NODE_FEATURES,
    PvGraphConfig,
    PvGraphObservation,
    build_pv_graph_observation,
)
from snfs_traffic.scenarios import (
    PriorityVehicleConfig,
    VehicleMix,
    assign_priority_vehicles,
    select_yielding_vehicle_ids,
)
from snfs_traffic.simulator import ScenarioConfig, TrafficSimulator


_ACTION_INDEX_TO_DELTA = np.asarray([-1, 0, 1], dtype=np.int8)


def _float_grid(
    raw_value: Any,
    *,
    fallback: float,
    name: str,
    lower: float,
    upper: float,
    lower_inclusive: bool,
    upper_inclusive: bool,
) -> tuple[float, ...]:
    """Parse one scalar or a sequence into a validated, de-duplicated grid."""

    if raw_value is None:
        values = [fallback]
    elif isinstance(raw_value, str):
        values = [part.strip() for part in raw_value.split(",") if part.strip()]
    elif np.isscalar(raw_value):
        values = [raw_value]
    else:
        values = list(raw_value)
    if not values:
        raise ValueError(f"{name} must contain at least one value")

    parsed: list[float] = []
    for raw in values:
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(f"{name} values must be finite numbers")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} values must be finite numbers") from exc
        lower_ok = value >= lower if lower_inclusive else value > lower
        upper_ok = value <= upper if upper_inclusive else value < upper
        if not np.isfinite(value) or not lower_ok or not upper_ok:
            interval = (
                f"{'[' if lower_inclusive else '('}{lower}, {upper}"
                f"{']' if upper_inclusive else ')'}"
            )
            raise ValueError(f"{name} values must be in {interval}")
        if value not in parsed:
            parsed.append(value)
    return tuple(parsed)


def _condition_key(density: float, av_fraction: float) -> str:
    def token(value: float) -> str:
        return f"{value:.3f}".replace(".", "p")

    return f"rho_{token(density)}_av_{token(av_fraction)}"


@dataclass(frozen=True, slots=True)
class PvSpeedRewardConfig:
    """PV-speed objective with small intervention regularizers."""

    applied_penalty: float = 0.001
    rejected_penalty: float = 0.002

    def __post_init__(self) -> None:
        for name, value in (
            ("applied_penalty", self.applied_penalty),
            ("rejected_penalty", self.rejected_penalty),
        ):
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise ValueError(f"{name} must be a finite number >= 0")
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be a finite number >= 0")


def compute_pv_speed_reward(
    *,
    pv_speed: int | float,
    pv_vmax: int | float,
    applied_count: int,
    rejected_count: int,
    config: PvSpeedRewardConfig,
) -> tuple[float, dict[str, float]]:
    """Return the deliberately minimal first-stage reward."""

    speed_reward = float(pv_speed) / max(float(pv_vmax), 1.0)
    applied_regularizer = -float(config.applied_penalty) * int(applied_count)
    rejected_regularizer = -float(config.rejected_penalty) * int(rejected_count)
    total = float(speed_reward + applied_regularizer + rejected_regularizer)
    components = {
        "pv_speed_reward": speed_reward,
        "applied_regularizer": applied_regularizer,
        "rejected_regularizer": rejected_regularizer,
        "applied_count": float(applied_count),
        "rejected_count": float(rejected_count),
        "total": total,
    }
    if not np.isfinite(total):
        raise ValueError("PV reward must be finite")
    return total, components


class CentralizedPvEnv(gym.Env):
    """One persistent coordinator issuing lateral commands to nearby AVs.

    Every active graph node receives an explicit action.  Action index 1 means
    a forced lateral stay for that step.  Vehicles absent from the graph are
    omitted from the override mapping and retain native Revised S-NFS rules.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        raw_config = dict(config or {})
        raw_config.update(kwargs)
        worker_index = int(getattr(config, "worker_index", 0))
        vector_index = int(getattr(config, "vector_index", 0))

        self.params = SimulationParams(
            num_lanes=int(raw_config.get("num_lanes", 4)),
            road_length=int(raw_config.get("road_length", 1000)),
            vmax_default=int(raw_config.get("vmax_hdv", 5)),
            vmax_controlled=int(raw_config.get("vmax_priority", 6)),
            G=int(raw_config.get("G", 15)),
            q=float(raw_config.get("q", 0.99)),
            r=float(raw_config.get("r", 0.99)),
            S=int(raw_config.get("S", 2)),
            P1=float(raw_config.get("P1", 0.999)),
            P2=float(raw_config.get("P2", 0.99)),
            P3=float(raw_config.get("P3", 0.98)),
            P4=float(raw_config.get("P4", 0.01)),
            p_lane_change=float(raw_config.get("p_lane_change", 0.5)),
        )
        self.graph_config = PvGraphConfig(
            front_distance=int(raw_config.get("front_distance", 30)),
            back_distance=int(raw_config.get("back_distance", 10)),
            sensor_distance=int(raw_config.get("sensor_distance", 60)),
            cooldown_steps=int(raw_config.get("cooldown_steps", 1)),
        )
        self.graph_config.validate_for(self.params)
        self.reward_config = PvSpeedRewardConfig(
            applied_penalty=float(raw_config.get("applied_penalty", 0.001)),
            rejected_penalty=float(raw_config.get("rejected_penalty", 0.002)),
        )
        scalar_density = float(raw_config.get("density", 0.20))
        scalar_av_fraction = float(raw_config.get("av_fraction", 0.95))
        self._density_values = _float_grid(
            raw_config.get("density_values"),
            fallback=scalar_density,
            name="density_values",
            lower=0.0,
            upper=1.0,
            lower_inclusive=False,
            upper_inclusive=False,
        )
        self._av_fraction_values = _float_grid(
            raw_config.get("av_fraction_values"),
            fallback=scalar_av_fraction,
            name="av_fraction_values",
            lower=0.0,
            upper=1.0,
            lower_inclusive=True,
            upper_inclusive=True,
        )
        self._condition_pairs = tuple(
            product(self._density_values, self._av_fraction_values)
        )
        self._condition_order: list[int] = []
        self._condition_cycle = 0
        self._density = self._density_values[0]
        self._av_fraction = self._av_fraction_values[0]
        self._warmup_steps = int(raw_config.get("warmup_steps", 1000))
        self._warmup_cache_enabled = bool(raw_config.get("warmup_cache", True))
        self._warmup_cache: dict[float, tuple[Any, int, int]] = {}
        self._episode_steps = int(raw_config.get("episode_steps", 1000))
        self._backend = str(raw_config.get("backend", "optimized"))
        self._validate = bool(raw_config.get("validate", False))
        base_seed = int(raw_config.get("seed", 0))
        self._initial_seed = base_seed + worker_index * 100_000 + vector_index * 1_000
        if self._warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self._episode_steps < 1:
            raise ValueError("episode_steps must be >= 1")

        self._sim = TrafficSimulator(
            params=self.params,
            backend=self._backend,
            rng_seed=self._initial_seed,
            scenario=ScenarioConfig(
                density=self._density,
                seed=self._initial_seed,
                vehicle_mix=VehicleMix(),
            ),
            validate=self._validate,
        )
        self._priority_vehicle_id: int | None = None
        self._cooldown_by_vehicle_id: dict[int, int] = {}
        self._current_graph: PvGraphObservation | None = None
        self._step_index = 0
        self._pv_speed_sum = 0.0
        self._episode_applied_count = 0
        self._episode_rejected_count = 0
        self._done = True
        self._seed_initialized = False

        max_nodes = self.graph_config.max_nodes(self.params)
        self.action_space = spaces.MultiDiscrete(
            np.full(max_nodes, 3, dtype=np.int64)
        )
        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(max_nodes, len(PV_GRAPH_NODE_FEATURES)),
                    dtype=np.float32,
                ),
                "node_mask": spaces.MultiBinary(max_nodes),
                "neighbor_index": spaces.Box(
                    low=0,
                    high=max_nodes - 1,
                    shape=(max_nodes, len(PV_GRAPH_NEIGHBOR_RELATIONS)),
                    dtype=np.int32,
                ),
                "neighbor_mask": spaces.MultiBinary(
                    (max_nodes, len(PV_GRAPH_NEIGHBOR_RELATIONS))
                ),
                "edge_features": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(
                        max_nodes,
                        len(PV_GRAPH_NEIGHBOR_RELATIONS),
                        len(PV_GRAPH_EDGE_FEATURES),
                    ),
                    dtype=np.float32,
                ),
                "action_mask": spaces.MultiBinary((max_nodes, 3)),
                "global_features": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(len(PV_GRAPH_GLOBAL_FEATURES),),
                    dtype=np.float32,
                ),
            }
        )

    @property
    def priority_vehicle_id(self) -> int:
        if self._priority_vehicle_id is None:
            raise RuntimeError("environment is not reset")
        return self._priority_vehicle_id

    @property
    def backend_name(self) -> str:
        return self._sim.backend_name

    def _next_seed(self) -> int:
        return int(self.np_random.integers(0, np.iinfo(np.int32).max))

    def _sample_condition(self) -> tuple[float, float]:
        """Visit every configured pair once before reshuffling the grid."""

        if not self._condition_order:
            self._condition_order = [
                int(index)
                for index in self.np_random.permutation(len(self._condition_pairs))
            ]
            self._condition_cycle += 1
        pair_index = self._condition_order.pop()
        density, av_fraction = self._condition_pairs[pair_index]
        return float(density), float(av_fraction)

    def _new_simulator(self, *, scenario_seed: int) -> TrafficSimulator:
        return TrafficSimulator(
            params=self.params,
            backend=self._backend,
            rng_seed=self._initial_seed,
            scenario=ScenarioConfig(
                density=self._density,
                seed=scenario_seed,
                vehicle_mix=VehicleMix(),
            ),
            validate=self._validate,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        env_seed = seed
        if env_seed is None and not self._seed_initialized:
            env_seed = self._initial_seed
        super().reset(seed=env_seed)
        if seed is not None:
            self._condition_order.clear()
            self._condition_cycle = 0
            self._warmup_cache.clear()
        self._seed_initialized = True
        options = options or {}
        def option_seed(name: str) -> int:
            return int(options[name]) if name in options else self._next_seed()

        scenario_seed = option_seed("scenario_seed")
        warmup_rng_seed = option_seed("warmup_rng_seed")
        assignment_seed = option_seed("assignment_seed")
        av_selection_seed = option_seed("av_selection_seed")
        measurement_rng_seed = option_seed("measurement_rng_seed")

        self._density, self._av_fraction = self._sample_condition()
        cached_warmup = self._warmup_cache.get(self._density)
        warmup_cache_hit = self._warmup_cache_enabled and cached_warmup is not None
        if warmup_cache_hit:
            state, scenario_seed, warmup_rng_seed = cached_warmup
            state = state.copy()
            self._sim = self._new_simulator(scenario_seed=scenario_seed)
        else:
            self._sim = self._new_simulator(scenario_seed=scenario_seed)
            state = self._sim.reset(
                scenario_seed=scenario_seed,
                rng_seed=warmup_rng_seed,
            )
            for _ in range(self._warmup_steps):
                state = self._sim.step()
            if self._warmup_cache_enabled:
                self._warmup_cache[self._density] = (
                    state.copy(),
                    scenario_seed,
                    warmup_rng_seed,
                )
        assignment = assign_priority_vehicles(
            state,
            PriorityVehicleConfig(count=1, placement="random", seed=assignment_seed),
            road_length=self.params.road_length,
        )
        state, av_vehicle_ids = select_yielding_vehicle_ids(
            assignment.state,
            fraction=self._av_fraction,
            seed=av_selection_seed,
            tag_as_av=True,
        )
        if np.any(state.controlled):
            raise RuntimeError("centralized PV environment must not mark AVs as controlled")
        self._priority_vehicle_id = int(assignment.vehicle_ids[0])
        state = self._sim.reset(state=state, rng_seed=measurement_rng_seed)
        self._cooldown_by_vehicle_id.clear()
        self._step_index = 0
        self._pv_speed_sum = 0.0
        self._episode_applied_count = 0
        self._episode_rejected_count = 0
        self._done = False
        observation = self._build_observation(state)
        info = {
            "step_index": 0,
            "priority_vehicle_id": self.priority_vehicle_id,
            "active_av_count": self._current_graph.active_count,
            "fleet_av_count": len(av_vehicle_ids),
            "density": self._density,
            "av_fraction": self._av_fraction,
            "condition_key": _condition_key(self._density, self._av_fraction),
            "condition_cycle": self._condition_cycle,
            "condition_count": len(self._condition_pairs),
            "warmup_steps": self._warmup_steps,
            "warmup_cache_hit": warmup_cache_hit,
            "warmup_cache_size": len(self._warmup_cache),
            "backend_name": self._sim.backend_name,
            "scenario_seed": scenario_seed,
            "warmup_rng_seed": warmup_rng_seed,
            "assignment_seed": assignment_seed,
            "av_selection_seed": av_selection_seed,
            "measurement_rng_seed": measurement_rng_seed,
        }
        return observation, info

    def _build_observation(self, state) -> dict[str, np.ndarray]:
        graph = build_pv_graph_observation(
            state,
            self.params,
            self._sim.topology,
            priority_vehicle_id=self.priority_vehicle_id,
            config=self.graph_config,
            cooldown_by_vehicle_id=self._cooldown_by_vehicle_id,
        )
        self._current_graph = graph
        return graph.as_dict()

    def _update_cooldowns(self, result: LateralOverrideResult | None) -> None:
        next_cooldowns = {
            vehicle_id: remaining - 1
            for vehicle_id, remaining in self._cooldown_by_vehicle_id.items()
            if remaining > 1
        }
        if result is not None and self.graph_config.cooldown_steps > 0:
            for vehicle_id, applied in zip(result.vehicle_id, result.applied):
                if bool(applied):
                    next_cooldowns[int(vehicle_id)] = self.graph_config.cooldown_steps
        self._cooldown_by_vehicle_id = next_cooldowns

    def step(self, action):
        if self._done:
            raise RuntimeError("episode is done; call reset() before step()")
        if self._current_graph is None:
            raise RuntimeError("environment has no current graph")
        action_array = np.asarray(action)
        if not self.action_space.contains(action_array):
            raise ValueError(
                f"invalid centralized action with shape {action_array.shape}; "
                f"expected {self.action_space}"
            )

        active_count = self._current_graph.active_count
        overrides: dict[int, int] = {}
        cooldown_rejections = 0
        for slot in range(active_count):
            vehicle_id = int(self._current_graph.vehicle_id[slot])
            action_index = int(action_array[slot])
            lane_delta = int(_ACTION_INDEX_TO_DELTA[action_index])
            if (
                lane_delta != 0
                and int(self._cooldown_by_vehicle_id.get(vehicle_id, 0)) > 0
            ):
                lane_delta = 0
                cooldown_rejections += 1
            overrides[vehicle_id] = lane_delta

        if overrides:
            next_state = self._sim.step_lateral_overrides(overrides)
            raw_result = self._sim.last_step_info.action_result
            if not isinstance(raw_result, LateralOverrideResult):
                raise RuntimeError("lateral override step returned no result")
            result: LateralOverrideResult | None = raw_result
        else:
            next_state = self._sim.step()
            result = None

        applied_count = 0
        rejected_count = cooldown_rejections
        if result is not None:
            requested_non_stay = result.requested_lane_delta != 0
            applied_count = int(np.sum(result.applied & requested_non_stay))
            rejected_count += int(np.sum(requested_non_stay & ~result.applied))
        self._update_cooldowns(result)
        self._step_index += 1

        pv_matches = np.flatnonzero(
            next_state.alive
            & (next_state.vehicle_id == int(self.priority_vehicle_id))
        )
        terminated = pv_matches.size != 1
        pv_speed = 0 if terminated else int(next_state.vel[int(pv_matches[0])])
        reward, reward_components = compute_pv_speed_reward(
            pv_speed=pv_speed,
            pv_vmax=self.params.vmax_controlled,
            applied_count=applied_count,
            rejected_count=rejected_count,
            config=self.reward_config,
        )
        self._pv_speed_sum += float(pv_speed)
        self._episode_applied_count += applied_count
        self._episode_rejected_count += rejected_count
        truncated = self._step_index >= self._episode_steps
        self._done = bool(terminated or truncated)
        if terminated:
            observation = {
                key: np.zeros(space.shape, dtype=space.dtype)
                for key, space in self.observation_space.spaces.items()
            }
            observation["action_mask"][:, 1] = 1
        else:
            observation = self._build_observation(next_state)
        info = {
            "step_index": self._step_index,
            "priority_vehicle_id": self.priority_vehicle_id,
            "pv_speed": pv_speed,
            "pv_speed_norm": pv_speed / max(self.params.vmax_controlled, 1),
            "episode_mean_pv_speed": self._pv_speed_sum / self._step_index,
            "active_av_count": active_count,
            "next_active_av_count": (
                0 if terminated else self._current_graph.active_count
            ),
            "applied_count": applied_count,
            "rejected_count": rejected_count,
            "episode_applied_count": self._episode_applied_count,
            "episode_rejected_count": self._episode_rejected_count,
            "density": self._density,
            "av_fraction": self._av_fraction,
            "condition_key": _condition_key(self._density, self._av_fraction),
            "reward_components": reward_components,
            "backend_name": (
                self._sim.last_step_info.backend_name
                if self._sim.last_step_info is not None
                else self._sim.backend_name
            ),
        }
        return observation, reward, bool(terminated), bool(truncated), info
