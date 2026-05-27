from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from snfs_traffic.backends import get_backend
from snfs_traffic.control import normalize_lane_actions, step_with_controlled_lateral_actions_reference, controlled_vehicle_ids
from snfs_traffic.core import SimulationParams, TrafficState, validate_runtime_invariants
from snfs_traffic.core.indexing import build_occupancy
from snfs_traffic.core.state import validate_state
from snfs_traffic.observations import LocalObservationBatch, LocalObservationConfig, build_local_observations
from snfs_traffic.scenarios import VehicleMix, make_uniform_random_state
from snfs_traffic.topology import RingTopology

@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    density: float = 0.20
    seed: int = 1
    vehicle_mix: VehicleMix = field(default_factory=VehicleMix)

@dataclass(frozen=True, slots=True)
class SimulatorStepInfo:
    step: int
    backend_name: str
    actions_supplied: bool
    used_reference_action_path: bool
    controlled_vehicle_ids: np.ndarray
    action_result: object | None = None

@dataclass(frozen=True, slots=True)
class RolloutSnapshot:
    step: int
    state: TrafficState
    info: SimulatorStepInfo | None

class TrafficSimulator:
    def __init__(self, *, params: SimulationParams, topology: RingTopology | None = None, backend: str = "auto", rng_seed: int = 123, scenario: ScenarioConfig | None = None, validate: bool = True, require_all_controlled_actions: bool = True) -> None:
        self._params = params
        self._topology = topology or RingTopology(num_lanes=params.num_lanes, length=params.road_length)
        if self._topology.num_lanes != params.num_lanes or self._topology.length != params.road_length:
            raise ValueError("topology must match params")
        self._backend = get_backend(backend)
        self._backend_name = self._backend.name
        self._rng_seed = rng_seed
        self._scenario = scenario or ScenarioConfig()
        self._validate = validate
        self._require_all = require_all_controlled_actions
        self._state = None
        self._rng = None
        self._step_count = 0
        self._last = None

    @property
    def params(self): return self._params
    @property
    def topology(self): return self._topology
    @property
    def backend(self): return self._backend
    @property
    def backend_name(self): return self._backend_name
    @property
    def step_count(self): return self._step_count
    @property
    def last_step_info(self): return self._last
    @property
    def state(self):
        if self._state is None: raise RuntimeError("simulator is not reset")
        return self._state.copy()

    def reset(self, *, seed: int | None = None, scenario_seed: int | None = None, rng_seed: int | None = None, state: TrafficState | None = None) -> TrafficState:
        scen_seed = scenario_seed if scenario_seed is not None else seed if seed is not None else self._scenario.seed
        step_seed = rng_seed if rng_seed is not None else seed if seed is not None else self._rng_seed
        self._rng = np.random.default_rng(step_seed)
        if state is not None:
            validate_state(state, self._params)
            self._state = state.copy()
            build_occupancy(self._state, self._params)
        else:
            self._state = make_uniform_random_state(num_lanes=self._params.num_lanes, road_length=self._params.road_length, density=self._scenario.density, seed=scen_seed, vehicle_mix=self._scenario.vehicle_mix)
        if self._validate:
            validate_runtime_invariants(self._state, self._params, self._topology)
        self._step_count = 0
        self._last = None
        return self._state.copy()

    def step(self, actions=None) -> TrafficState:
        if self._state is None or self._rng is None:
            raise RuntimeError("simulator is not reset")
        if actions is None:
            nxt = self._backend.step(self._state, self._params, self._topology, self._rng)
            info = SimulatorStepInfo(step=self._step_count + 1, backend_name=self._backend_name, actions_supplied=False, used_reference_action_path=False, controlled_vehicle_ids=controlled_vehicle_ids(self._state))
        else:
            nxt, result = step_with_controlled_lateral_actions_reference(self._state, self._params, self._topology, self._rng, actions, require_all_controlled=self._require_all)
            info = SimulatorStepInfo(step=self._step_count + 1, backend_name=f"{self._backend_name}+reference-action", actions_supplied=True, used_reference_action_path=True, controlled_vehicle_ids=controlled_vehicle_ids(self._state), action_result=result)
        if self._validate:
            validate_runtime_invariants(nxt, self._params, self._topology)
        self._state = nxt
        self._step_count += 1
        self._last = info
        return self._state.copy()

    def observe(self, config: LocalObservationConfig | None = None) -> LocalObservationBatch:
        if self._state is None:
            raise RuntimeError("simulator is not reset")
        return build_local_observations(self._state, self._params, self._topology, config)

    def snapshot(self) -> RolloutSnapshot:
        if self._state is None:
            raise RuntimeError("simulator is not reset")
        return RolloutSnapshot(step=self._step_count, state=self._state.copy(), info=self._last)

    def iter_rollout(self, *, steps: int, action_provider: Callable[[TrafficState, int], object | None] | None = None, include_initial: bool = False):
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be int >= 0")
        if include_initial:
            yield self.snapshot()
        for _ in range(steps):
            actions = action_provider(self._state.copy(), self._step_count) if action_provider is not None else None
            self.step(actions)
            yield self.snapshot()

    def rollout(self, *, steps: int, action_provider=None, include_initial: bool = False) -> list[RolloutSnapshot]:
        return list(self.iter_rollout(steps=steps, action_provider=action_provider, include_initial=include_initial))
