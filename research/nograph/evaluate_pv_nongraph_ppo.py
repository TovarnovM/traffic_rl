#!/usr/bin/env python3
"""Paired deterministic evaluation for PV NodeMLP and GridCNN PPO policies."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from snfs_traffic.control import LateralOverrideResult
from snfs_traffic.metrics import paired_bootstrap_interval
from snfs_traffic.observations import (
    PV_GRAPH_GLOBAL_FEATURES,
    PV_GRAPH_NODE_FEATURES,
    PvGraphConfig,
    build_pv_graph_observation,
)
from snfs_traffic.rules import PriorityYieldConfig, PriorityYieldController
from snfs_traffic.scenarios import (
    PriorityVehicleConfig,
    assign_priority_vehicles,
    select_yielding_vehicle_ids,
)
from snfs_traffic.simulator import TrafficSimulator


RESEARCH_DIR = str(Path(__file__).resolve().parent)
if RESEARCH_DIR not in sys.path:
    sys.path.insert(0, RESEARCH_DIR)
import evaluate_pv_graph_ppo as graph_eval
import train_pv_graph_ppo as graph_train_runner
import train_pv_nongraph_ppo as train_runner


BASELINE1 = graph_eval.BASELINE1
BASELINE2 = graph_eval.BASELINE2
POLICY_NOOP = "policy_noop"
LEFT_IF_SAFE = graph_eval.LEFT_IF_SAFE
NON_GRAPH_PPO = "nongraph_ppo"
MEASUREMENT_MODES = (
    BASELINE1,
    BASELINE2,
    POLICY_NOOP,
    LEFT_IF_SAFE,
    NON_GRAPH_PPO,
)
ACTION_INDEX_TO_DELTA = graph_eval.ACTION_INDEX_TO_DELTA

_POLICY_MODEL = None
_TORCH = None


def _saved_model_preset(run_config: Mapping[str, Any]) -> str:
    preset = run_config.get("model_preset")
    model_spec = run_config.get("model_spec")
    if preset is None and isinstance(model_spec, Mapping):
        preset = model_spec.get("preset")
    if not isinstance(preset, str) or preset not in train_runner.MODEL_PRESETS:
        raise ValueError(
            "run_config.json does not contain a supported model_preset; "
            "the checkpoint cannot be reconstructed safely"
        )
    return preset


def _evaluation_config(
    args: argparse.Namespace, run_config: Mapping[str, Any]
) -> dict[str, Any]:
    # Reuse the exact Graph-PPO simulator/default resolution.  These two
    # attributes are graph-model-only but expected by that helper.
    args.hidden_dim = None
    args.message_layers = None
    config = graph_eval._evaluation_config(args, run_config)
    config["model_preset"] = _saved_model_preset(run_config)
    config["torch_threads"] = int(args.torch_threads)
    return config


def _make_inference_model(
    state_dict: Mapping[str, np.ndarray], config: Mapping[str, Any]
):
    import torch

    from snfs_traffic.rl.nongraph_ppo import build_non_graph_core

    graph_config = PvGraphConfig(
        front_distance=int(config["front_distance"]),
        back_distance=int(config["back_distance"]),
        sensor_distance=int(config["sensor_distance"]),
        cooldown_steps=int(config["cooldown_steps"]),
    )
    max_nodes = graph_config.max_nodes(graph_eval._make_params(config))
    model = build_non_graph_core(
        preset=str(config["model_preset"]),
        node_dim=len(PV_GRAPH_NODE_FEATURES),
        global_dim=len(PV_GRAPH_GLOBAL_FEATURES),
        max_nodes=max_nodes,
        num_lanes=int(config["num_lanes"]),
        front_distance=int(config["front_distance"]),
        back_distance=int(config["back_distance"]),
    )
    tensors = {key: torch.from_numpy(value) for key, value in state_dict.items()}
    model.load_state_dict(tensors, strict=True)
    model.eval()
    return model


def _worker_init(
    state_dict: Mapping[str, np.ndarray], config: Mapping[str, Any]
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
        logits, _value, _node_mask = _POLICY_MODEL(tensors)
    return _TORCH.argmax(logits[0], dim=-1).cpu().numpy().astype(np.int64)


def _controller_action(graph, mode: str) -> np.ndarray:
    if mode == NON_GRAPH_PPO:
        return _policy_action(graph.as_dict())
    action = np.ones(graph.node_mask.shape, dtype=np.int64)
    if mode == POLICY_NOOP:
        return action
    if mode == LEFT_IF_SAFE:
        active = graph.node_mask.astype(bool)
        action[active & graph.action_mask[:, 0].astype(bool)] = 0
        return action
    raise ValueError(f"unknown non-graph controller mode: {mode}")


def _restore_policy(
    checkpoint: Path,
    run_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Restore once on CPU and verify extracted-core deterministic actions."""

    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.models import ModelCatalog
        from ray.tune.registry import register_env

        from snfs_traffic.rl.centralized_pv_env import CentralizedPvEnv
        from snfs_traffic.rl.nongraph_ppo import (
            FactorizedGraphPPO,
            PvNonGraphTorchModel,
            count_trainable_parameters,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"missing evaluation dependency: {exc}; install with "
            "python -m pip install -e \".[rl,ray,numba,viz]\""
        ) from exc

    train_args = train_runner.build_parser().parse_args([])
    for key, value in run_config.items():
        if hasattr(train_args, key) and key not in {"env_config", "out_dir"}:
            setattr(train_args, key, value)
    train_args.workers = 0
    train_args.envs_per_worker = 1
    train_args.num_gpus = 0.0
    train_args.ray_local_mode = False
    train_args.smoke = False
    train_args.model_preset = str(config["model_preset"])
    train_args.num_lanes = int(config["num_lanes"])
    train_args.front_distance = int(config["front_distance"])
    train_args.back_distance = int(config["back_distance"])

    restore_env = dict(config)
    restore_env.update({"warmup_steps": 0, "episode_steps": 2})
    register_env(train_runner.ENV_NAME, lambda context: CentralizedPvEnv(context))
    ModelCatalog.register_custom_model(
        train_runner.MODEL_NAME, PvNonGraphTorchModel
    )
    algorithm_config = graph_train_runner._classic_api_stack(PPOConfig())
    algorithm_config = algorithm_config.environment(
        env=train_runner.ENV_NAME,
        env_config=restore_env,
        disable_env_checking=True,
    )
    algorithm_config = algorithm_config.framework("torch")
    algorithm_config = graph_train_runner._configure_workers(
        algorithm_config, train_args
    )
    algorithm_config = train_runner._configure_training(
        algorithm_config, train_args
    )
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
        policy_model = algorithm.get_policy().model
        state_dict = {
            key: value.detach().cpu().numpy().copy()
            for key, value in policy_model.core.state_dict().items()
        }

        smoke_config = dict(config)
        smoke_config.update({"warmup_steps": 2, "episode_steps": 2})
        smoke_env = CentralizedPvEnv(smoke_config)
        observation, info = smoke_env.reset(seed=71_417)
        action_output = algorithm.compute_single_action(observation, explore=False)
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
            logits, _value, _node_mask = extracted_model(tensors)
        extracted_action = (
            torch.argmax(logits[0], dim=-1).cpu().numpy().astype(np.int64)
        )
        if rllib_action.shape != extracted_action.shape or not np.array_equal(
            rllib_action, extracted_action
        ):
            raise RuntimeError(
                "restore smoke failed: RLlib and extracted CPU model actions differ"
            )
        smoke = {
            "checkpoint": str(checkpoint),
            "model_preset": str(config["model_preset"]),
            "model_parameter_count": count_trainable_parameters(policy_model),
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


def _measure_policy(
    config: Mapping[str, Any],
    params,
    *,
    initial_state,
    priority_id: int,
    measurement_rng_seed: int,
    action_mode: str,
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
    metrics = graph_eval.PriorityMetricsAccumulator(
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
        action = _controller_action(graph, action_mode)
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
                raise RuntimeError("non-graph override step returned no lateral result")
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
            "active_av_count_mean": graph_eval._mean(active_counts),
            "active_av_count_max": max(active_counts, default=0),
            "policy_left_requests": left_requests,
            "policy_right_requests": right_requests,
            "policy_stay_actions": int(sum(active_counts))
            - left_requests
            - right_requests,
        }
    )
    return summary, ";".join(sorted(backend_paths)) or sim.backend_name


def run_replication(spec: graph_eval.WorkerSpec) -> list[dict[str, Any]]:
    config = spec.config
    params = graph_eval._make_params(config)
    base_seed = int(
        config["seed"] + spec.density_index * 1_000_000 + spec.run_index * 10_000
    )
    scenario_seed = base_seed + 11
    warmup_rng_seed = base_seed + 97
    measurement_rng_seed = base_seed + 193
    assignment_seed = base_seed + 1_000
    av_selection_seed = base_seed + 2_000

    warm_state, warmup_backend = graph_eval._warm_snapshot(
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
        (POLICY_NOOP, av_state, None),
        (LEFT_IF_SAFE, av_state, None),
        (NON_GRAPH_PPO, av_state, None),
    )
    rows: list[dict[str, Any]] = []
    for mode, initial_state, controller in mode_specs:
        if mode in (POLICY_NOOP, LEFT_IF_SAFE, NON_GRAPH_PPO):
            summary, measurement_backend = _measure_policy(
                config,
                params,
                initial_state=initial_state,
                priority_id=priority_id,
                measurement_rng_seed=measurement_rng_seed,
                action_mode=mode,
            )
        else:
            summary, measurement_backend = graph_eval._measure_native_or_rule(
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
            "model_preset": str(config["model_preset"]),
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
            if row["mode"] == NON_GRAPH_PPO
        }
    )
    for density_index, run_index in run_keys:
        treatment = indexed[(density_index, run_index, NON_GRAPH_PPO)]
        for comparison_mode in (BASELINE1, BASELINE2, POLICY_NOOP, LEFT_IF_SAFE):
            comparison = indexed[(density_index, run_index, comparison_mode)]
            background_flow = float(comparison["flow_background"])
            pv_speed = float(comparison["mean_speed_priority"])
            output.append(
                {
                    "requested_density": float(treatment["requested_density"]),
                    "effective_density": float(treatment["effective_density"]),
                    "density_index": density_index,
                    "run_index": run_index,
                    "model_preset": treatment["model_preset"],
                    "comparison_mode": comparison_mode,
                    "pv_speed_nongraph_ppo": float(
                        treatment["mean_speed_priority"]
                    ),
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
            "model_preset": group[0]["model_preset"],
            "comparison_mode": comparison,
            "runs": len(group),
            "pv_speed_win_rate": graph_eval._mean(
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
        result["nongraph_ppo_speed_advantage"] = bool(
            result["pv_speed_delta_ci_low"] > 0.0
        )
        output.append(result)
    return output


def _write_plot(
    path: Path,
    summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    *,
    model_preset: str,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib is unavailable; PNG plot was skipped", flush=True)
        return False

    colors = {
        BASELINE1: "#6b7280",
        BASELINE2: "#2563eb",
        POLICY_NOOP: "#7c3aed",
        LEFT_IF_SAFE: "#059669",
        NON_GRAPH_PPO: "#dc2626",
    }
    labels = {
        BASELINE1: "Baseline 1 (HDV)",
        BASELINE2: "Baseline 2 (matched)",
        POLICY_NOOP: "Policy noop",
        LEFT_IF_SAFE: "Left-if-safe",
        NON_GRAPH_PPO: model_preset,
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for mode in MEASUREMENT_MODES:
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

    for comparison in (BASELINE1, BASELINE2, POLICY_NOOP, LEFT_IF_SAFE):
        values = sorted(
            (row for row in paired if row["comparison_mode"] == comparison),
            key=lambda row: float(row["requested_density"]),
        )
        x = [float(row["requested_density"]) for row in values]
        y = [float(row["pv_speed_delta_mean"]) for row in values]
        low = [float(row["pv_speed_delta_ci_low"]) for row in values]
        high = [float(row["pv_speed_delta_ci_high"]) for row in values]
        yerr = [
            [mean - lo for mean, lo in zip(y, low)],
            [hi - mean for mean, hi in zip(y, high)],
        ]
        axes[1].errorbar(
            x, y, yerr=yerr, marker="o", capsize=3,
            color=colors[comparison], label=f"vs {labels[comparison]}",
        )
        ratio = [float(row["background_flow_ratio_mean"]) for row in values]
        ratio_low = [float(row["background_flow_ratio_ci_low"]) for row in values]
        ratio_high = [float(row["background_flow_ratio_ci_high"]) for row in values]
        ratio_err = [
            [mean - lo for mean, lo in zip(ratio, ratio_low)],
            [hi - mean for mean, hi in zip(ratio, ratio_high)],
        ]
        axes[2].errorbar(
            x, ratio, yerr=ratio_err, marker="o", capsize=3,
            color=colors[comparison], label=f"vs {labels[comparison]}",
        )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title(f"Paired {model_preset} speed gain")
    axes[1].set_xlabel("Density")
    axes[1].set_ylabel("Delta speed [cells/step]")
    axes[1].legend(fontsize=8)
    axes[2].axhline(1.0, color="black", linewidth=1)
    axes[2].set_title(f"{model_preset} background-flow ratio")
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
        description="Paired CPU evaluation of a trained PV non-graph PPO policy"
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
    parser.add_argument("--backend", choices=("auto", "reference", "optimized"))
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
        default=Path("artifacts/rl/pv_nongraph_ppo/evaluation_rho_030"),
    )
    parser.add_argument("--prefix", default="pv_nongraph_ppo_eval")
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

    densities = graph_eval._parse_densities(args.densities)
    checkpoint = graph_eval._resolve_checkpoint(args.checkpoint)
    run_config_path = graph_eval._find_run_config(checkpoint, args.run_config)
    run_config = graph_eval._load_json(run_config_path)
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
    print(f"model_preset={config['model_preset']}", flush=True)
    print("restore_smoke=starting", flush=True)
    state_dict, restore_smoke = _restore_policy(checkpoint, run_config, config)
    print(
        "restore_smoke=passed "
        f"active_avs={restore_smoke['active_av_count']} "
        f"action_shape={restore_smoke['action_shape']}",
        flush=True,
    )

    tasks = [
        graph_eval.WorkerSpec(density, density_index, run_index, config)
        for density_index, density in enumerate(densities)
        for run_index in range(args.runs)
    ]
    workers = args.workers or min(
        len(tasks), max(1, min(20, (os.cpu_count() or 2) - 1))
    )
    workers = min(workers, len(tasks))
    print(
        f"evaluation=starting tasks={len(tasks)} workers={workers} "
        f"modes={len(MEASUREMENT_MODES)} measure_steps={config['measure_steps']}",
        flush=True,
    )
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    if workers == 1:
        _worker_init(state_dict, config)
        for completed, result in enumerate(map(run_replication, tasks), start=1):
            rows.extend(result)
            print(
                f"[{completed}/{len(tasks)}] "
                f"elapsed={time.perf_counter() - started:.1f}s",
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
            MEASUREMENT_MODES.index(str(row["mode"])),
        )
    )

    summary = graph_eval._grouped_summary(rows)
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
    graph_eval._write_csv(runs_path, rows)
    graph_eval._write_csv(summary_path, summary)
    graph_eval._write_csv(paired_path, paired_rows)
    graph_eval._write_csv(paired_summary_path, paired_summary)
    plot_written = bool(args.plot) and _write_plot(
        plot_path,
        summary,
        paired_summary,
        model_preset=str(config["model_preset"]),
    )
    metadata = {
        "checkpoint": str(checkpoint),
        "run_config": str(run_config_path),
        "model_preset": str(config["model_preset"]),
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
            f"{config['model_preset']} vs {result['comparison_mode']}: "
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
