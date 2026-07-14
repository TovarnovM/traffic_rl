#!/usr/bin/env python3
"""Paired evaluation of a trained centralized PV Graph-PPO policy.

For every ``(density, run)`` this script creates one warmed HDV snapshot, tags
the same priority vehicle, selects the same AV fleet, and then replays three
measurement modes with the same simulator RNG seed:

* Baseline 1: HDV traffic without yielding;
* Baseline 2 matched: the deterministic yield rule controls the selected AVs;
* Graph-PPO: the restored policy controls those same AVs near the PV.

The RLlib checkpoint is restored once in the parent process.  Only the small
CPU model state is sent to evaluation workers, so repeated simulations can be
parallelized without starting one Ray instance (or allocating one GPU) per
worker.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from snfs_traffic.control import LateralOverrideResult
from snfs_traffic.core import SimulationParams
from snfs_traffic.metrics import PriorityMetricsAccumulator, paired_bootstrap_interval
from snfs_traffic.observations import (
    PV_GRAPH_EDGE_FEATURES,
    PV_GRAPH_GLOBAL_FEATURES,
    PV_GRAPH_NODE_FEATURES,
    PvGraphConfig,
    build_pv_graph_observation,
)
from snfs_traffic.rules import PriorityYieldConfig, PriorityYieldController
from snfs_traffic.scenarios import (
    PriorityVehicleConfig,
    VehicleMix,
    assign_priority_vehicles,
    select_yielding_vehicle_ids,
)
from snfs_traffic.simulator import ScenarioConfig, TrafficSimulator


BASELINE1 = "baseline1_hdv"
BASELINE2 = "baseline2_matched"
GRAPH_PPO = "graph_ppo"
ACTION_INDEX_TO_DELTA = np.asarray([-1, 0, 1], dtype=np.int8)

SUMMARY_METRICS = (
    "flow_all",
    "flow_background",
    "mean_speed_all",
    "mean_speed_background",
    "mean_speed_priority",
    "mean_priority_time_loss",
    "stopped_fraction_all",
    "stopped_fraction_priority",
    "lane_change_rate_all",
    "rule_requests",
    "rule_applied",
    "rule_rejected",
    "active_av_count_mean",
)

DEFAULT_ENV_CONFIG: dict[str, Any] = {
    "road_length": 1000,
    "num_lanes": 4,
    "density": 0.30,
    "av_fraction": 0.95,
    "front_distance": 30,
    "back_distance": 10,
    "sensor_distance": 60,
    "cooldown_steps": 5,
    "warmup_steps": 1000,
    "episode_steps": 1000,
    "backend": "optimized",
    "validate": False,
    "vmax_hdv": 5,
    "vmax_priority": 6,
    "G": 15,
    "q": 0.99,
    "r": 0.99,
    "S": 2,
    "P1": 0.999,
    "P2": 0.99,
    "P3": 0.98,
    "P4": 0.01,
    "p_lane_change": 0.5,
}


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    density: float
    density_index: int
    run_index: int
    config: dict[str, Any]


_POLICY_MODEL = None
_TORCH = None


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.fmean(values)) if values else 0.0


def _sample_std(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.stdev(values)) if len(values) >= 2 else 0.0


def _parse_densities(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(not 0.0 < value < 1.0 for value in values):
        raise ValueError("--densities must contain comma-separated values in (0, 1)")
    return values


def _checkpoint_marker(path: Path) -> bool:
    return any(
        (path / name).exists()
        for name in ("algorithm_state.pkl", "rllib_checkpoint.json", ".is_checkpoint")
    )


def _resolve_checkpoint(raw_path: Path) -> Path:
    path = raw_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint path does not exist: {path}")
    if path.is_file():
        path = path.parent
    if _checkpoint_marker(path):
        return path

    candidates = {
        marker.parent
        for pattern in ("algorithm_state.pkl", "rllib_checkpoint.json", ".is_checkpoint")
        for marker in path.rglob(pattern)
    }
    if not candidates:
        raise FileNotFoundError(
            f"no RLlib checkpoint found below {path}; pass the directory printed "
            "as checkpoint=... or final_checkpoint=... by the training script"
        )

    def rank(candidate: Path) -> tuple[int, float]:
        numbers = re.findall(r"\d+", candidate.name)
        sequence = int(numbers[-1]) if numbers else -1
        return sequence, candidate.stat().st_mtime

    return max(candidates, key=rank)


def _find_run_config(checkpoint: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"run config does not exist: {path}")
        return path
    for directory in (checkpoint, *checkpoint.parents):
        candidate = directory / "run_config.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "run_config.json was not found next to the checkpoint; pass it with "
        "--run-config so the model architecture can be reconstructed exactly"
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _evaluation_config(
    args: argparse.Namespace,
    run_config: Mapping[str, Any],
) -> dict[str, Any]:
    config = dict(DEFAULT_ENV_CONFIG)
    raw_saved_env = run_config.get("env_config", {})
    saved_env = dict(raw_saved_env) if isinstance(raw_saved_env, Mapping) else {}
    config.update(saved_env)
    for key in DEFAULT_ENV_CONFIG:
        if key in run_config and key not in saved_env:
            config[key] = run_config[key]

    overrides = {
        "road_length": args.road_length,
        "num_lanes": args.num_lanes,
        "av_fraction": args.av_fraction,
        "front_distance": args.front_distance,
        "back_distance": args.back_distance,
        "sensor_distance": args.sensor_distance,
        "cooldown_steps": args.cooldown_steps,
        "warmup_steps": args.warmup_steps,
        "backend": args.backend,
        "validate": args.validate,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    config["measure_steps"] = int(args.measure_steps)
    config["yield_distance"] = int(
        args.yield_distance
        if args.yield_distance is not None
        else config["front_distance"]
    )
    config["yield_cooldown_steps"] = int(
        args.yield_cooldown_steps
        if args.yield_cooldown_steps is not None
        else config["cooldown_steps"]
    )
    config["seed"] = int(args.seed)
    config["hidden_dim"] = int(
        args.hidden_dim
        if args.hidden_dim is not None
        else run_config.get("hidden_dim", 64)
    )
    config["message_layers"] = int(
        args.message_layers
        if args.message_layers is not None
        else run_config.get("message_layers", 2)
    )
    config["torch_threads"] = int(args.torch_threads)

    if not 0.0 <= float(config["av_fraction"]) <= 1.0:
        raise ValueError("av_fraction must be in [0, 1]")
    if int(config["warmup_steps"]) < 0 or int(config["measure_steps"]) < 1:
        raise ValueError("warmup_steps must be >= 0 and measure_steps must be >= 1")
    return config


def _make_params(config: Mapping[str, Any]) -> SimulationParams:
    return SimulationParams(
        num_lanes=int(config["num_lanes"]),
        road_length=int(config["road_length"]),
        vmax_default=int(config["vmax_hdv"]),
        vmax_controlled=int(config["vmax_priority"]),
        G=int(config["G"]),
        q=float(config["q"]),
        r=float(config["r"]),
        S=int(config["S"]),
        P1=float(config["P1"]),
        P2=float(config["P2"]),
        P3=float(config["P3"]),
        P4=float(config["P4"]),
        p_lane_change=float(config["p_lane_change"]),
    )


def _make_inference_model(
    state_dict: Mapping[str, np.ndarray], config: Mapping[str, Any]
):
    """Build the checkpoint-compatible GNN without importing RLlib.

    Evaluation workers need only deterministic forward passes.  Mirroring the
    small Torch module here avoids loading Ray/RLlib into every process; the
    exact action check in ``_restore_policy`` guards against architecture drift.
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    hidden_dim = int(config["hidden_dim"])
    message_layer_count = int(config["message_layers"])
    graph_config = PvGraphConfig(
        front_distance=int(config["front_distance"]),
        back_distance=int(config["back_distance"]),
        sensor_distance=int(config["sensor_distance"]),
        cooldown_steps=int(config["cooldown_steps"]),
    )
    max_nodes = graph_config.max_nodes(_make_params(config))

    class MessagePassingLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.message = nn.Sequential(
                nn.Linear(hidden_dim + len(PV_GRAPH_EDGE_FEATURES), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.update = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(
            self,
            node_embeddings,
            neighbor_index,
            neighbor_mask,
            edge_features,
            node_mask,
        ):
            batch_size, node_count, neighbor_count = neighbor_index.shape
            batch_index = torch.arange(
                batch_size, device=node_embeddings.device
            ).view(batch_size, 1, 1)
            batch_index = batch_index.expand(
                batch_size, node_count, neighbor_count
            )
            neighbors = node_embeddings[batch_index, neighbor_index]
            messages = self.message(torch.cat((neighbors, edge_features), dim=-1))
            weights = neighbor_mask.unsqueeze(-1)
            messages = messages * weights
            aggregated = messages.sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
            update = self.update(torch.cat((node_embeddings, aggregated), dim=-1))
            output = self.norm(node_embeddings + update)
            return functional.relu(output) * node_mask.unsqueeze(-1)

    class InferencePvGraphModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.max_nodes = max_nodes
            self.node_encoder = nn.Sequential(
                nn.Linear(len(PV_GRAPH_NODE_FEATURES), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.global_encoder = nn.Sequential(
                nn.Linear(len(PV_GRAPH_GLOBAL_FEATURES), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.message_layers = nn.ModuleList(
                MessagePassingLayer() for _ in range(message_layer_count)
            )
            self.actor_head = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3),
            )
            # It is unused for action selection, but retaining it makes strict
            # checkpoint loading catch any model mismatch.
            self.value_head = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, observation):
            node_features = observation["node_features"].float()
            node_mask = observation["node_mask"].float()
            neighbor_index = observation["neighbor_index"].long()
            neighbor_mask = observation["neighbor_mask"].float()
            edge_features = observation["edge_features"].float()
            action_mask = observation["action_mask"].bool()
            global_features = observation["global_features"].float()

            global_embedding = self.global_encoder(global_features)
            nodes = self.node_encoder(node_features)
            nodes = functional.relu(
                nodes + global_embedding.unsqueeze(1)
            ) * node_mask.unsqueeze(-1)
            for layer in self.message_layers:
                nodes = layer(
                    nodes,
                    neighbor_index,
                    neighbor_mask,
                    edge_features,
                    node_mask,
                )
            repeated_global = global_embedding.unsqueeze(1).expand(
                -1, self.max_nodes, -1
            )
            logits = self.actor_head(torch.cat((nodes, repeated_global), dim=-1))
            return logits.masked_fill(~action_mask, -1.0e9)

    model = InferencePvGraphModel()
    tensors = {key: torch.from_numpy(value) for key, value in state_dict.items()}
    model.load_state_dict(tensors, strict=True)
    model.eval()
    return model


def _worker_init(
    state_dict: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> None:
    global _POLICY_MODEL, _TORCH

    import torch

    threads = max(1, int(config.get("torch_threads", 1)))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _POLICY_MODEL = _make_inference_model(state_dict, config)
    _TORCH = torch


def _policy_action(observation: Mapping[str, np.ndarray]) -> np.ndarray:
    if _POLICY_MODEL is None or _TORCH is None:
        raise RuntimeError("policy model is not initialized in this worker")
    tensors = {
        key: _TORCH.from_numpy(np.asarray(value)).unsqueeze(0)
        for key, value in observation.items()
    }
    with _TORCH.inference_mode():
        logits = _POLICY_MODEL(tensors)
    logits = logits[0]
    return _TORCH.argmax(logits, dim=-1).cpu().numpy().astype(np.int64)


def _restore_policy(
    checkpoint: Path,
    run_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Restore once on CPU and verify extracted-model deterministic actions."""

    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.models import ModelCatalog
        from ray.tune.registry import register_env

        from snfs_traffic.rl.centralized_pv_env import CentralizedPvEnv
        from snfs_traffic.rl.graph_ppo import FactorizedGraphPPO, PvGraphTorchModel
    except ImportError as exc:
        raise RuntimeError(
            f"missing evaluation dependency: {exc}; install with "
            "python -m pip install -e \".[rl,ray,numba,viz]\""
        ) from exc

    research_dir = str(Path(__file__).resolve().parent)
    if research_dir not in sys.path:
        sys.path.insert(0, research_dir)
    import train_pv_graph_ppo as train_runner

    train_args = train_runner.build_parser().parse_args([])
    for key, value in run_config.items():
        if hasattr(train_args, key) and key not in {"env_config", "out_dir"}:
            setattr(train_args, key, value)
    train_args.workers = 0
    train_args.envs_per_worker = 1
    train_args.num_gpus = 0.0
    train_args.ray_local_mode = False
    train_args.smoke = False
    train_args.hidden_dim = int(config["hidden_dim"])
    train_args.message_layers = int(config["message_layers"])

    restore_env = dict(config)
    restore_env.update({"warmup_steps": 0, "episode_steps": 2})
    register_env(train_runner.ENV_NAME, lambda context: CentralizedPvEnv(context))
    ModelCatalog.register_custom_model(train_runner.MODEL_NAME, PvGraphTorchModel)
    algorithm_config = train_runner._classic_api_stack(PPOConfig())
    algorithm_config = algorithm_config.environment(
        env=train_runner.ENV_NAME,
        env_config=restore_env,
        disable_env_checking=True,
    )
    algorithm_config = algorithm_config.framework("torch")
    algorithm_config = train_runner._configure_workers(algorithm_config, train_args)
    algorithm_config = train_runner._configure_training(algorithm_config, train_args)
    if hasattr(algorithm_config, "resources"):
        algorithm_config = algorithm_config.resources(num_gpus=0.0)
    if hasattr(algorithm_config, "debugging"):
        algorithm_config = algorithm_config.debugging(
            seed=int(run_config.get("seed", 0)), log_level="ERROR"
        )

    ray.init(
        include_dashboard=False,
        num_cpus=1,
        num_gpus=0,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    algorithm = None
    smoke_env = None
    try:
        algorithm = FactorizedGraphPPO(config=algorithm_config)
        algorithm.restore(str(checkpoint))
        policy = algorithm.get_policy()
        state_dict = {
            key: value.detach().cpu().numpy().copy()
            for key, value in policy.model.state_dict().items()
        }

        smoke_config = dict(config)
        smoke_config.update({"warmup_steps": 2, "episode_steps": 2})
        smoke_env = CentralizedPvEnv(smoke_config)
        observation, info = smoke_env.reset(seed=71_417)
        action_output = algorithm.compute_single_action(observation, explore=False)
        # Algorithm.compute_single_action normally returns only the action,
        # while Policy.compute_single_action returns (action, state, info).
        if isinstance(action_output, tuple) and len(action_output) == 3:
            action_output = action_output[0]
        rllib_action = np.asarray(action_output, dtype=np.int64)
        extracted_model = _make_inference_model(state_dict, config)
        import torch

        tensors = {
            key: torch.from_numpy(np.asarray(value)).unsqueeze(0)
            for key, value in observation.items()
        }
        with torch.inference_mode():
            logits = extracted_model(tensors)
        extracted_action = (
            torch.argmax(logits[0], dim=-1)
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        if rllib_action.shape != extracted_action.shape or not np.array_equal(
            rllib_action, extracted_action
        ):
            raise RuntimeError(
                "restore smoke failed: RLlib and extracted CPU model actions differ"
            )
        smoke = {
            "checkpoint": str(checkpoint),
            "active_av_count": int(info["active_av_count"]),
            "action_shape": list(rllib_action.shape),
            "non_stay_actions": int(np.sum(rllib_action != 1)),
            "status": "passed",
        }
        return state_dict, smoke
    finally:
        if smoke_env is not None:
            smoke_env.close()
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()


def _warm_snapshot(
    config: Mapping[str, Any],
    params: SimulationParams,
    *,
    density: float,
    scenario_seed: int,
    warmup_rng_seed: int,
):
    sim = TrafficSimulator(
        params=params,
        backend=str(config["backend"]),
        rng_seed=warmup_rng_seed,
        scenario=ScenarioConfig(
            density=density,
            seed=scenario_seed,
            vehicle_mix=VehicleMix(),
        ),
        validate=bool(config["validate"]),
    )
    state = sim.reset(scenario_seed=scenario_seed, rng_seed=warmup_rng_seed)
    for _ in range(int(config["warmup_steps"])):
        state = sim.step()
    return state, sim.backend_name


def _measure_native_or_rule(
    config: Mapping[str, Any],
    params: SimulationParams,
    *,
    initial_state,
    priority_id: int,
    measurement_rng_seed: int,
    controller: PriorityYieldController | None,
) -> tuple[dict[str, Any], str]:
    sim = TrafficSimulator(
        params=params,
        backend=str(config["backend"]),
        rng_seed=measurement_rng_seed,
        validate=bool(config["validate"]),
    )
    state = sim.reset(state=initial_state, rng_seed=measurement_rng_seed)
    metrics = PriorityMetricsAccumulator(
        params, priority_vehicle_ids=(priority_id,)
    )
    backend_paths: set[str] = set()
    for _ in range(int(config["measure_steps"])):
        if controller is None:
            state = sim.step()
            metrics.observe(state)
        else:
            decision = controller.decide(state, params, sim.topology)
            result = None
            if decision.overrides:
                state = sim.step_lateral_overrides(decision.overrides)
                result = sim.last_step_info.action_result
                if not isinstance(result, LateralOverrideResult):
                    raise RuntimeError("yield rule returned no lateral result")
                controller.observe(result)
            else:
                state = sim.step()
            metrics.observe(state, decision=decision, result=result)
        if sim.last_step_info is not None:
            backend_paths.add(sim.last_step_info.backend_name)
    summary = metrics.summary().to_dict()
    summary["active_av_count_mean"] = 0.0
    summary["active_av_count_max"] = 0
    return summary, ";".join(sorted(backend_paths)) or sim.backend_name


def _measure_graph_ppo(
    config: Mapping[str, Any],
    params: SimulationParams,
    *,
    initial_state,
    priority_id: int,
    measurement_rng_seed: int,
) -> tuple[dict[str, Any], str]:
    sim = TrafficSimulator(
        params=params,
        backend=str(config["backend"]),
        rng_seed=measurement_rng_seed,
        validate=bool(config["validate"]),
    )
    state = sim.reset(state=initial_state, rng_seed=measurement_rng_seed)
    graph_config = PvGraphConfig(
        front_distance=int(config["front_distance"]),
        back_distance=int(config["back_distance"]),
        sensor_distance=int(config["sensor_distance"]),
        cooldown_steps=int(config["cooldown_steps"]),
    )
    metrics = PriorityMetricsAccumulator(
        params, priority_vehicle_ids=(priority_id,)
    )
    cooldowns: dict[int, int] = {}
    backend_paths: set[str] = set()
    active_counts: list[int] = []
    non_stay_requests = 0
    applied_non_stay = 0
    rejected_non_stay = 0
    cooldown_suppressed = 0
    left_requests = 0
    right_requests = 0

    for _ in range(int(config["measure_steps"])):
        graph = build_pv_graph_observation(
            state,
            params,
            sim.topology,
            priority_vehicle_id=priority_id,
            config=graph_config,
            cooldown_by_vehicle_id=cooldowns,
        )
        active_counts.append(graph.active_count)
        action = _policy_action(graph.as_dict())
        overrides: dict[int, int] = {}
        for slot in range(graph.active_count):
            vehicle_id = int(graph.vehicle_id[slot])
            lane_delta = int(ACTION_INDEX_TO_DELTA[int(action[slot])])
            if lane_delta != 0 and cooldowns.get(vehicle_id, 0) > 0:
                lane_delta = 0
                cooldown_suppressed += 1
            overrides[vehicle_id] = lane_delta
            left_requests += int(lane_delta == -1)
            right_requests += int(lane_delta == 1)

        result = None
        if overrides:
            state = sim.step_lateral_overrides(overrides)
            result = sim.last_step_info.action_result
            if not isinstance(result, LateralOverrideResult):
                raise RuntimeError("Graph-PPO override step returned no lateral result")
            requested = result.requested_lane_delta != 0
            non_stay_requests += int(np.sum(requested))
            applied_non_stay += int(np.sum(result.applied & requested))
            rejected_non_stay += int(np.sum(~result.applied & requested))
        else:
            state = sim.step()

        next_cooldowns = {
            vehicle_id: remaining - 1
            for vehicle_id, remaining in cooldowns.items()
            if remaining > 1
        }
        if result is not None and graph_config.cooldown_steps > 0:
            for vehicle_id, applied in zip(result.vehicle_id, result.applied):
                if bool(applied):
                    next_cooldowns[int(vehicle_id)] = graph_config.cooldown_steps
        cooldowns = next_cooldowns
        metrics.observe(state)
        if sim.last_step_info is not None:
            backend_paths.add(sim.last_step_info.backend_name)

    summary = metrics.summary().to_dict()
    summary.update(
        {
            "rule_requests": non_stay_requests,
            "rule_applied": applied_non_stay,
            "rule_rejected": rejected_non_stay + cooldown_suppressed,
            "rule_cooldown_suppressed": cooldown_suppressed,
            "active_av_count_mean": _mean(active_counts),
            "active_av_count_max": max(active_counts, default=0),
            "policy_left_requests": left_requests,
            "policy_right_requests": right_requests,
            "policy_stay_actions": int(sum(active_counts))
            - left_requests
            - right_requests,
        }
    )
    return summary, ";".join(sorted(backend_paths)) or sim.backend_name


def run_replication(spec: WorkerSpec) -> list[dict[str, Any]]:
    config = spec.config
    params = _make_params(config)
    base_seed = int(
        config["seed"] + spec.density_index * 1_000_000 + spec.run_index * 10_000
    )
    scenario_seed = base_seed + 11
    warmup_rng_seed = base_seed + 97
    measurement_rng_seed = base_seed + 193
    assignment_seed = base_seed + 1_000
    av_selection_seed = base_seed + 2_000

    warm_state, warmup_backend = _warm_snapshot(
        config,
        params,
        density=spec.density,
        scenario_seed=scenario_seed,
        warmup_rng_seed=warmup_rng_seed,
    )
    assignment = assign_priority_vehicles(
        warm_state,
        PriorityVehicleConfig(count=1, placement="random", seed=assignment_seed),
        road_length=params.road_length,
    )
    priority_id = int(assignment.vehicle_ids[0])
    av_state, av_ids = select_yielding_vehicle_ids(
        assignment.state,
        fraction=float(config["av_fraction"]),
        seed=av_selection_seed,
        tag_as_av=True,
    )
    effective_density = float(np.sum(warm_state.alive)) / (
        params.num_lanes * params.road_length
    )
    mode_specs = (
        (BASELINE1, assignment.state, None),
        (
            BASELINE2,
            av_state,
            PriorityYieldController(
                PriorityYieldConfig(
                    detection_distance=int(config["yield_distance"]),
                    cooldown_steps=int(config["yield_cooldown_steps"]),
                ),
                eligible_vehicle_ids=av_ids,
            ),
        ),
        (GRAPH_PPO, av_state, None),
    )
    rows: list[dict[str, Any]] = []
    for mode, initial_state, controller in mode_specs:
        if mode == GRAPH_PPO:
            summary, measurement_backend = _measure_graph_ppo(
                config,
                params,
                initial_state=initial_state,
                priority_id=priority_id,
                measurement_rng_seed=measurement_rng_seed,
            )
        else:
            summary, measurement_backend = _measure_native_or_rule(
                config,
                params,
                initial_state=initial_state,
                priority_id=priority_id,
                measurement_rng_seed=measurement_rng_seed,
                controller=controller,
            )
        row: dict[str, Any] = {
            "requested_density": float(spec.density),
            "effective_density": effective_density,
            "density_index": spec.density_index,
            "run_index": spec.run_index,
            "mode": mode,
            "priority_vehicle_id": priority_id,
            "av_fraction": float(config["av_fraction"]),
            "av_vehicle_count": len(av_ids),
            "scenario_seed": scenario_seed,
            "warmup_rng_seed": warmup_rng_seed,
            "assignment_seed": assignment_seed,
            "av_selection_seed": av_selection_seed,
            "measurement_rng_seed": measurement_rng_seed,
            "backend_requested": config["backend"],
            "warmup_backend": warmup_backend,
            "measurement_backend": measurement_backend,
            "num_lanes": params.num_lanes,
            "road_length": params.road_length,
            "warmup_steps": int(config["warmup_steps"]),
            "measure_steps": int(config["measure_steps"]),
        }
        row.update(summary)
        rows.append(row)
    return rows


def _grouped_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = float(row["requested_density"]), str(row["mode"])
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (density, mode), group in sorted(groups.items()):
        result: dict[str, Any] = {
            "requested_density": density,
            "effective_density_mean": _mean(
                float(row["effective_density"]) for row in group
            ),
            "mode": mode,
            "runs": len(group),
            "av_fraction": float(group[0]["av_fraction"]),
        }
        for metric in SUMMARY_METRICS:
            values = [float(row.get(metric, 0.0)) for row in group]
            std = _sample_std(values)
            sem = std / math.sqrt(len(values))
            result[f"{metric}_mean"] = _mean(values)
            result[f"{metric}_std"] = std
            result[f"{metric}_sem"] = sem
            result[f"{metric}_ci95_low"] = _mean(values) - 1.96 * sem
            result[f"{metric}_ci95_high"] = _mean(values) + 1.96 * sem
        output.append(result)
    return output


def _paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (int(row["density_index"]), int(row["run_index"]), str(row["mode"])): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    run_keys = sorted(
        {
            (int(row["density_index"]), int(row["run_index"]))
            for row in rows
            if row["mode"] == GRAPH_PPO
        }
    )
    for density_index, run_index in run_keys:
        treatment = indexed[(density_index, run_index, GRAPH_PPO)]
        for comparison_mode in (BASELINE1, BASELINE2):
            comparison = indexed[(density_index, run_index, comparison_mode)]
            background_flow = float(comparison["flow_background"])
            pv_speed = float(comparison["mean_speed_priority"])
            output.append(
                {
                    "requested_density": float(treatment["requested_density"]),
                    "effective_density": float(treatment["effective_density"]),
                    "density_index": density_index,
                    "run_index": run_index,
                    "comparison_mode": comparison_mode,
                    "pv_speed_graph_ppo": float(treatment["mean_speed_priority"]),
                    "pv_speed_comparison": pv_speed,
                    "pv_speed_delta": float(treatment["mean_speed_priority"])
                    - pv_speed,
                    "pv_speed_ratio": (
                        float(treatment["mean_speed_priority"]) / pv_speed
                        if pv_speed > 0.0
                        else None
                    ),
                    "priority_time_loss_delta": float(
                        treatment["mean_priority_time_loss"]
                    )
                    - float(comparison["mean_priority_time_loss"]),
                    "background_flow_delta": float(treatment["flow_background"])
                    - background_flow,
                    "background_flow_ratio": (
                        float(treatment["flow_background"]) / background_flow
                        if background_flow > 0.0
                        else None
                    ),
                    "all_speed_delta": float(treatment["mean_speed_all"])
                    - float(comparison["mean_speed_all"]),
                    "priority_stopped_delta": float(
                        treatment["stopped_fraction_priority"]
                    )
                    - float(comparison["stopped_fraction_priority"]),
                }
            )
    return output


def _paired_summary(
    rows: list[dict[str, Any]],
    *,
    confidence: float,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = float(row["requested_density"]), str(row["comparison_mode"])
        groups.setdefault(key, []).append(row)
    metrics = (
        "pv_speed_delta",
        "priority_time_loss_delta",
        "background_flow_delta",
        "background_flow_ratio",
        "all_speed_delta",
        "priority_stopped_delta",
    )
    output: list[dict[str, Any]] = []
    for group_index, ((density, comparison), group) in enumerate(
        sorted(groups.items())
    ):
        result: dict[str, Any] = {
            "requested_density": density,
            "comparison_mode": comparison,
            "runs": len(group),
            "pv_speed_win_rate": _mean(
                float(row["pv_speed_delta"] > 0.0) for row in group
            ),
        }
        for metric_index, metric in enumerate(metrics):
            values = np.asarray(
                [float(row[metric]) for row in group if row[metric] is not None],
                dtype=np.float64,
            )
            low, high = paired_bootstrap_interval(
                values,
                confidence=confidence,
                samples=bootstrap_samples,
                seed=seed + group_index * 100 + metric_index,
            )
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            )
            result[f"{metric}_ci_low"] = low
            result[f"{metric}_ci_high"] = high
        result["graph_ppo_speed_advantage"] = bool(
            result["pv_speed_delta_ci_low"] > 0.0
        )
        output.append(result)
    return output


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fields})


def _write_plot(
    path: Path,
    summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib is unavailable; PNG plot was skipped", flush=True)
        return False

    colors = {BASELINE1: "#6b7280", BASELINE2: "#2563eb", GRAPH_PPO: "#dc2626"}
    labels = {BASELINE1: "Baseline 1 (HDV)", BASELINE2: "Baseline 2 (matched)", GRAPH_PPO: "Graph-PPO"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for mode in (BASELINE1, BASELINE2, GRAPH_PPO):
        values = sorted(
            (row for row in summary if row["mode"] == mode),
            key=lambda row: float(row["requested_density"]),
        )
        x = [float(row["requested_density"]) for row in values]
        y = [float(row["mean_speed_priority_mean"]) for row in values]
        error = [1.96 * float(row["mean_speed_priority_sem"]) for row in values]
        axes[0].errorbar(
            x, y, yerr=error, marker="o", capsize=3,
            color=colors[mode], label=labels[mode],
        )
    axes[0].set_title("Priority-vehicle speed")
    axes[0].set_xlabel("Density")
    axes[0].set_ylabel("Mean speed [cells/step]")
    axes[0].legend(fontsize=8)

    for comparison in (BASELINE1, BASELINE2):
        values = sorted(
            (row for row in paired if row["comparison_mode"] == comparison),
            key=lambda row: float(row["requested_density"]),
        )
        x = [float(row["requested_density"]) for row in values]
        y = [float(row["pv_speed_delta_mean"]) for row in values]
        low = [float(row["pv_speed_delta_ci_low"]) for row in values]
        high = [float(row["pv_speed_delta_ci_high"]) for row in values]
        yerr = [[mean - lo for mean, lo in zip(y, low)], [hi - mean for mean, hi in zip(y, high)]]
        axes[1].errorbar(
            x, y, yerr=yerr, marker="o", capsize=3,
            color=colors[comparison], label=f"vs {labels[comparison]}",
        )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("Paired Graph-PPO speed gain")
    axes[1].set_xlabel("Density")
    axes[1].set_ylabel("Delta speed [cells/step]")
    axes[1].legend(fontsize=8)

    for comparison in (BASELINE1, BASELINE2):
        values = sorted(
            (row for row in paired if row["comparison_mode"] == comparison),
            key=lambda row: float(row["requested_density"]),
        )
        x = [float(row["requested_density"]) for row in values]
        y = [float(row["background_flow_ratio_mean"]) for row in values]
        low = [float(row["background_flow_ratio_ci_low"]) for row in values]
        high = [float(row["background_flow_ratio_ci_high"]) for row in values]
        yerr = [[mean - lo for mean, lo in zip(y, low)], [hi - mean for mean, hi in zip(y, high)]]
        axes[2].errorbar(
            x, y, yerr=yerr, marker="o", capsize=3,
            color=colors[comparison], label=f"vs {labels[comparison]}",
        )
    axes[2].axhline(1.0, color="black", linewidth=1)
    axes[2].set_title("Graph-PPO background-flow ratio")
    axes[2].set_xlabel("Density")
    axes[2].set_ylabel("Flow ratio")
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired CPU evaluation of a trained centralized PV Graph-PPO policy"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--densities", default="0.30")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=0, help="0 = auto")
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--road-length", type=int)
    parser.add_argument("--num-lanes", type=int)
    parser.add_argument("--av-fraction", type=float)
    parser.add_argument("--front-distance", type=int)
    parser.add_argument("--back-distance", type=int)
    parser.add_argument("--sensor-distance", type=int)
    parser.add_argument("--cooldown-steps", type=int)
    parser.add_argument("--yield-distance", type=int)
    parser.add_argument("--yield-cooldown-steps", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--message-layers", type=int)
    parser.add_argument(
        "--backend", choices=("auto", "reference", "optimized")
    )
    parser.add_argument(
        "--validate", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--plot", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/rl/pv_graph_ppo/evaluation_rho_030"),
    )
    parser.add_argument("--prefix", default="pv_graph_ppo_eval")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="restore plus one tiny paired reference-backend replication",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.runs < 1 or args.workers < 0 or args.chunksize < 1:
        raise SystemExit("runs/chunksize must be >= 1 and workers must be >= 0")
    if args.torch_threads < 1 or args.bootstrap_samples < 1:
        raise SystemExit("torch-threads and bootstrap-samples must be >= 1")
    if not 0.0 < args.confidence < 1.0:
        raise SystemExit("confidence must be in (0, 1)")

    densities = _parse_densities(args.densities)
    checkpoint = _resolve_checkpoint(args.checkpoint)
    run_config_path = _find_run_config(checkpoint, args.run_config)
    run_config = _load_json(run_config_path)
    config = _evaluation_config(args, run_config)
    if args.smoke:
        densities = densities[:1]
        args.runs = 1
        args.workers = 1
        smoke_road_length = max(
            int(config["front_distance"]) + int(config["back_distance"]) + 1,
            min(int(config["road_length"]), 100),
        )
        config.update(
            {
                # front/back distances determine max_nodes and therefore must
                # remain identical to the checkpoint architecture.
                "road_length": smoke_road_length,
                "sensor_distance": min(int(config["sensor_distance"]), 20),
                "warmup_steps": 5,
                "measure_steps": 32,
                "backend": "reference",
                "validate": True,
            }
        )

    print(f"checkpoint={checkpoint}", flush=True)
    print(f"run_config={run_config_path}", flush=True)
    print("restore_smoke=starting", flush=True)
    state_dict, restore_smoke = _restore_policy(
        checkpoint, run_config, config
    )
    print(
        "restore_smoke=passed "
        f"active_avs={restore_smoke['active_av_count']} "
        f"action_shape={restore_smoke['action_shape']}",
        flush=True,
    )

    tasks = [
        WorkerSpec(density, density_index, run_index, config)
        for density_index, density in enumerate(densities)
        for run_index in range(args.runs)
    ]
    workers = args.workers or min(
        len(tasks), max(1, min(20, (os.cpu_count() or 2) - 1))
    )
    workers = min(workers, len(tasks))
    print(
        f"evaluation=starting tasks={len(tasks)} workers={workers} "
        f"modes=3 measure_steps={config['measure_steps']}",
        flush=True,
    )
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    if workers == 1:
        _worker_init(state_dict, config)
        iterator = map(run_replication, tasks)
        for completed, result in enumerate(iterator, start=1):
            rows.extend(result)
            print(
                f"[{completed}/{len(tasks)}] elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    else:
        context = mp.get_context("spawn")
        with context.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(state_dict, config),
        ) as pool:
            iterator = pool.imap_unordered(
                run_replication, tasks, chunksize=args.chunksize
            )
            for completed, result in enumerate(iterator, start=1):
                rows.extend(result)
                print(
                    f"[{completed}/{len(tasks)}] "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    rows.sort(
        key=lambda row: (
            int(row["density_index"]),
            int(row["run_index"]),
            (BASELINE1, BASELINE2, GRAPH_PPO).index(str(row["mode"])),
        )
    )

    summary = _grouped_summary(rows)
    paired_rows = _paired_rows(rows)
    paired_summary = _paired_summary(
        paired_rows,
        confidence=args.confidence,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + 9_001,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = args.out_dir / f"{args.prefix}_runs.csv"
    summary_path = args.out_dir / f"{args.prefix}_summary.csv"
    paired_path = args.out_dir / f"{args.prefix}_paired.csv"
    paired_summary_path = args.out_dir / f"{args.prefix}_paired_summary.csv"
    plot_path = args.out_dir / f"{args.prefix}.png"
    metadata_path = args.out_dir / f"{args.prefix}_metadata.json"
    _write_csv(runs_path, rows)
    _write_csv(summary_path, summary)
    _write_csv(paired_path, paired_rows)
    _write_csv(paired_summary_path, paired_summary)
    plot_written = bool(args.plot) and _write_plot(
        plot_path, summary, paired_summary
    )
    metadata = {
        "checkpoint": str(checkpoint),
        "run_config": str(run_config_path),
        "restore_smoke": restore_smoke,
        "densities": densities,
        "runs": args.runs,
        "workers": workers,
        "config": config,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {
            "runs_csv": str(runs_path),
            "summary_csv": str(summary_path),
            "paired_csv": str(paired_path),
            "paired_summary_csv": str(paired_summary_path),
            "plot_png": str(plot_path) if plot_written else None,
        },
    }
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
    for result in paired_summary:
        print(
            f"rho={result['requested_density']:.3f} "
            f"Graph-PPO vs {result['comparison_mode']}: "
            f"delta_pv_speed={result['pv_speed_delta_mean']:.4f} "
            f"CI=[{result['pv_speed_delta_ci_low']:.4f}, "
            f"{result['pv_speed_delta_ci_high']:.4f}] "
            f"win_rate={result['pv_speed_win_rate']:.2%}",
            flush=True,
        )
    print(f"runs_csv={runs_path}", flush=True)
    print(f"paired_summary_csv={paired_summary_path}", flush=True)
    if plot_written:
        print(f"plot={plot_path}", flush=True)
    print(f"metadata={metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
