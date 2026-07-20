#!/usr/bin/env python3
"""Joint paired evaluation of Graph-PPO, NodeMLP-PPO, and GridCNN-PPO.

Each worker task owns one (density, run) pair. It warms the traffic state once,
assigns one PV once, and then evaluates all requested AV fractions. The HDV
baseline is simulated once per task; Baseline2, noop, and left-if-safe are
simulated once per AV fraction and shared by all three learned policies.

RLlib checkpoints are restored sequentially in the parent process. Workers
receive only small CPU state dictionaries and use deterministic argmax actions.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

RESEARCH_DIR = Path(__file__).resolve().parent
NOGRAPH_DIR = RESEARCH_DIR / "nograph"
for directory in (RESEARCH_DIR, NOGRAPH_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import evaluate_pv_graph_ppo as graph_eval
import evaluate_pv_nongraph_ppo as nongraph_eval

from snfs_traffic.control import LateralOverrideResult
from snfs_traffic.metrics import paired_bootstrap_interval
from snfs_traffic.observations import PvGraphConfig, build_pv_graph_observation
from snfs_traffic.rules import PriorityYieldConfig, PriorityYieldController
from snfs_traffic.scenarios import (
    PriorityVehicleConfig,
    assign_priority_vehicles,
    select_yielding_vehicle_ids,
)
from snfs_traffic.simulator import TrafficSimulator


BASELINE1 = graph_eval.BASELINE1
BASELINE2 = graph_eval.BASELINE2
NOOP = "noop"
LEFT_IF_SAFE = graph_eval.LEFT_IF_SAFE
GRAPH_PPO = graph_eval.GRAPH_PPO
NODE_MLP_PPO = "node_mlp_ppo"
GRID_CNN_PPO = "grid_cnn_ppo"

CONTROL_MODES = (BASELINE1, BASELINE2, NOOP, LEFT_IF_SAFE)
LEARNED_MODES = (GRAPH_PPO, NODE_MLP_PPO, GRID_CNN_PPO)
MEASUREMENT_MODES = CONTROL_MODES + LEARNED_MODES
ACTION_INDEX_TO_DELTA = graph_eval.ACTION_INDEX_TO_DELTA

DEFAULT_DENSITIES = (
    "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,"
    "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90"
)
DEFAULT_AV_FRACTIONS = "0.90,0.95,1.00"

SUMMARY_METRICS = graph_eval.SUMMARY_METRICS + (
    "completed_priority_laps",
    "mean_priority_lap_time",
)

# The sign is always treatment - comparison.
PAIRINGS = (
    (BASELINE2, BASELINE1),
    (NOOP, BASELINE1),
    (NOOP, BASELINE2),
    (LEFT_IF_SAFE, BASELINE1),
    (LEFT_IF_SAFE, BASELINE2),
    (LEFT_IF_SAFE, NOOP),
) + tuple(
    (learned, control)
    for learned in LEARNED_MODES
    for control in CONTROL_MODES
) + (
    (GRAPH_PPO, NODE_MLP_PPO),
    (GRAPH_PPO, GRID_CNN_PPO),
    (NODE_MLP_PPO, GRID_CNN_PPO),
)


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    density: float
    density_index: int
    run_index: int
    av_fractions: tuple[float, ...]
    config: dict[str, Any]


_POLICY_MODELS: dict[str, Any] = {}
_TORCH = None


def _parse_av_fractions(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    values = list(dict.fromkeys(values))
    if not values or any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("--av-fractions must contain comma-separated values in [0, 1]")
    return values


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, (float, np.floating)) or isinstance(
        right, (float, np.floating)
    ):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def _validate_compatible_configs(
    graph_config: Mapping[str, Any],
    node_config: Mapping[str, Any],
    cnn_config: Mapping[str, Any],
) -> None:
    keys = (
        "road_length",
        "num_lanes",
        "front_distance",
        "back_distance",
        "sensor_distance",
        "cooldown_steps",
        "vmax_hdv",
        "vmax_priority",
        "G",
        "q",
        "r",
        "S",
        "P1",
        "P2",
        "P3",
        "P4",
        "p_lane_change",
    )
    for label, candidate in (("NodeMLP", node_config), ("GridCNN", cnn_config)):
        mismatches = [
            key
            for key in keys
            if key not in candidate
            or key not in graph_config
            or not _same_value(candidate[key], graph_config[key])
        ]
        if mismatches:
            details = ", ".join(
                f"{key}: graph={graph_config.get(key)!r}, "
                f"{label.lower()}={candidate.get(key)!r}"
                for key in mismatches
            )
            raise ValueError(
                f"{label} checkpoint is not evaluation-compatible with Graph-PPO: "
                f"{details}"
            )


def _load_checkpoint_inputs(
    *,
    checkpoint_arg: Path,
    run_config_arg: Path | None,
    args: argparse.Namespace,
    av_fraction: float,
    policy_kind: str,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    checkpoint = graph_eval._resolve_checkpoint(checkpoint_arg)
    run_config_path = graph_eval._find_run_config(checkpoint, run_config_arg)
    run_config = graph_eval._load_json(run_config_path)
    eval_args = argparse.Namespace(**vars(args))
    eval_args.av_fraction = av_fraction
    if policy_kind == "graph":
        config = graph_eval._evaluation_config(eval_args, run_config)
    else:
        config = nongraph_eval._evaluation_config(eval_args, run_config)
    return checkpoint, run_config_path, run_config, config


def _worker_init(
    state_dicts: Mapping[str, Mapping[str, np.ndarray]],
    model_configs: Mapping[str, Mapping[str, Any]],
    torch_threads: int,
) -> None:
    global _POLICY_MODELS, _TORCH

    import torch

    torch.set_num_threads(max(1, int(torch_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    _POLICY_MODELS = {
        GRAPH_PPO: graph_eval._make_inference_model(
            state_dicts[GRAPH_PPO], model_configs[GRAPH_PPO]
        ),
        NODE_MLP_PPO: nongraph_eval._make_inference_model(
            state_dicts[NODE_MLP_PPO], model_configs[NODE_MLP_PPO]
        ),
        GRID_CNN_PPO: nongraph_eval._make_inference_model(
            state_dicts[GRID_CNN_PPO], model_configs[GRID_CNN_PPO]
        ),
    }
    for model in _POLICY_MODELS.values():
        model.eval()
    _TORCH = torch


def _policy_action(
    observation: Mapping[str, np.ndarray],
    mode: str,
) -> np.ndarray:
    if _TORCH is None or mode not in _POLICY_MODELS:
        raise RuntimeError("evaluation policy models are not initialized")
    tensors = {
        key: _TORCH.from_numpy(np.asarray(value)).unsqueeze(0)
        for key, value in observation.items()
    }
    with _TORCH.inference_mode():
        output = _POLICY_MODELS[mode](tensors)
    logits = output if mode == GRAPH_PPO else output[0]
    return _TORCH.argmax(logits[0], dim=-1).cpu().numpy().astype(np.int64)


def _controller_action(graph, mode: str) -> np.ndarray:
    if mode in LEARNED_MODES:
        return _policy_action(graph.as_dict(), mode)
    action = np.ones(graph.node_mask.shape, dtype=np.int64)
    if mode == NOOP:
        return action
    if mode == LEFT_IF_SAFE:
        active = graph.node_mask.astype(bool)
        action[active & graph.action_mask[:, 0].astype(bool)] = 0
        return action
    raise ValueError(f"unknown controller mode: {mode}")


def _measure_controller(
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
                raise RuntimeError(
                    f"{action_mode} override step returned no lateral result"
                )
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


def _model_preset(mode: str, config: Mapping[str, Any]) -> str:
    if mode == GRAPH_PPO:
        return "edge-graph"
    if mode == NODE_MLP_PPO:
        return str(config["node_mlp_model_preset"])
    if mode == GRID_CNN_PPO:
        return str(config["grid_cnn_model_preset"])
    return ""


def _row(
    *,
    spec: WorkerSpec,
    config: Mapping[str, Any],
    params,
    mode: str,
    summary: Mapping[str, Any],
    effective_density: float,
    av_fraction: float,
    av_vehicle_count: int,
    priority_id: int,
    scenario_seed: int,
    warmup_rng_seed: int,
    assignment_seed: int,
    av_selection_seed: int,
    measurement_rng_seed: int,
    warmup_backend: str,
    measurement_backend: str,
    baseline1_reused: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "requested_density": float(spec.density),
        "effective_density": effective_density,
        "density_index": spec.density_index,
        "run_index": spec.run_index,
        "mode": mode,
        "mode_category": "learned" if mode in LEARNED_MODES else "control",
        "model_preset": _model_preset(mode, config),
        "priority_vehicle_id": priority_id,
        "av_fraction": av_fraction,
        "av_vehicle_count": av_vehicle_count,
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
        "baseline1_reused_across_av_fractions": baseline1_reused,
    }
    row.update(summary)
    return row


def run_replication(spec: WorkerSpec) -> list[dict[str, Any]]:
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
    effective_density = float(np.sum(warm_state.alive)) / (
        params.num_lanes * params.road_length
    )

    baseline1_summary, baseline1_backend = graph_eval._measure_native_or_rule(
        config,
        params,
        initial_state=assignment.state,
        priority_id=priority_id,
        measurement_rng_seed=measurement_rng_seed,
        controller=None,
    )

    rows: list[dict[str, Any]] = []
    for av_fraction in spec.av_fractions:
        av_state, av_ids = select_yielding_vehicle_ids(
            assignment.state,
            fraction=av_fraction,
            seed=av_selection_seed,
            tag_as_av=True,
        )
        rows.append(
            _row(
                spec=spec,
                config=config,
                params=params,
                mode=BASELINE1,
                summary=baseline1_summary,
                effective_density=effective_density,
                av_fraction=av_fraction,
                av_vehicle_count=len(av_ids),
                priority_id=priority_id,
                scenario_seed=scenario_seed,
                warmup_rng_seed=warmup_rng_seed,
                assignment_seed=assignment_seed,
                av_selection_seed=av_selection_seed,
                measurement_rng_seed=measurement_rng_seed,
                warmup_backend=warmup_backend,
                measurement_backend=baseline1_backend,
                baseline1_reused=len(spec.av_fractions) > 1,
            )
        )

        yield_controller = PriorityYieldController(
            PriorityYieldConfig(
                detection_distance=int(config["yield_distance"]),
                cooldown_steps=int(config["yield_cooldown_steps"]),
            ),
            eligible_vehicle_ids=av_ids,
        )
        baseline2_summary, baseline2_backend = graph_eval._measure_native_or_rule(
            config,
            params,
            initial_state=av_state,
            priority_id=priority_id,
            measurement_rng_seed=measurement_rng_seed,
            controller=yield_controller,
        )
        rows.append(
            _row(
                spec=spec,
                config=config,
                params=params,
                mode=BASELINE2,
                summary=baseline2_summary,
                effective_density=effective_density,
                av_fraction=av_fraction,
                av_vehicle_count=len(av_ids),
                priority_id=priority_id,
                scenario_seed=scenario_seed,
                warmup_rng_seed=warmup_rng_seed,
                assignment_seed=assignment_seed,
                av_selection_seed=av_selection_seed,
                measurement_rng_seed=measurement_rng_seed,
                warmup_backend=warmup_backend,
                measurement_backend=baseline2_backend,
                baseline1_reused=False,
            )
        )

        for mode in (NOOP, LEFT_IF_SAFE) + LEARNED_MODES:
            summary, measurement_backend = _measure_controller(
                config,
                params,
                initial_state=av_state,
                priority_id=priority_id,
                measurement_rng_seed=measurement_rng_seed,
                action_mode=mode,
            )
            rows.append(
                _row(
                    spec=spec,
                    config=config,
                    params=params,
                    mode=mode,
                    summary=summary,
                    effective_density=effective_density,
                    av_fraction=av_fraction,
                    av_vehicle_count=len(av_ids),
                    priority_id=priority_id,
                    scenario_seed=scenario_seed,
                    warmup_rng_seed=warmup_rng_seed,
                    assignment_seed=assignment_seed,
                    av_selection_seed=av_selection_seed,
                    measurement_rng_seed=measurement_rng_seed,
                    warmup_backend=warmup_backend,
                    measurement_backend=measurement_backend,
                    baseline1_reused=False,
                )
            )
    return rows


def _finite_values(
    group: Iterable[Mapping[str, Any]],
    metric: str,
) -> list[float]:
    values: list[float] = []
    for row in group:
        value = row.get(metric)
        if value is None or value == "":
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    return values


def _grouped_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            float(row["requested_density"]),
            float(row["av_fraction"]),
            str(row["mode"]),
        )
        groups.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    mode_rank = {mode: index for index, mode in enumerate(MEASUREMENT_MODES)}
    for (density, av_fraction, mode), group in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            mode_rank[item[0][2]],
        ),
    ):
        result: dict[str, Any] = {
            "requested_density": density,
            "effective_density_mean": graph_eval._mean(
                float(row["effective_density"]) for row in group
            ),
            "av_fraction": av_fraction,
            "mode": mode,
            "model_preset": group[0].get("model_preset", ""),
            "runs": len(group),
        }
        for metric in SUMMARY_METRICS:
            values = _finite_values(group, metric)
            result[f"{metric}_valid_runs"] = len(values)
            if not values:
                result[f"{metric}_mean"] = None
                result[f"{metric}_std"] = None
                result[f"{metric}_sem"] = None
                result[f"{metric}_ci95_low"] = None
                result[f"{metric}_ci95_high"] = None
                continue
            mean = graph_eval._mean(values)
            std = graph_eval._sample_std(values)
            sem = std / math.sqrt(len(values))
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
            result[f"{metric}_sem"] = sem
            result[f"{metric}_ci95_low"] = mean - 1.96 * sem
            result[f"{metric}_ci95_high"] = mean + 1.96 * sem
        output.append(result)
    return output


def _paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (
            int(row["density_index"]),
            int(row["run_index"]),
            float(row["av_fraction"]),
            str(row["mode"]),
        ): row
        for row in rows
    }
    run_keys = sorted(
        {
            (
                int(row["density_index"]),
                int(row["run_index"]),
                float(row["av_fraction"]),
            )
            for row in rows
        }
    )
    output: list[dict[str, Any]] = []
    for density_index, run_index, av_fraction in run_keys:
        for treatment_mode, comparison_mode in PAIRINGS:
            treatment = indexed[
                (density_index, run_index, av_fraction, treatment_mode)
            ]
            comparison = indexed[
                (density_index, run_index, av_fraction, comparison_mode)
            ]
            pv_speed = float(comparison["mean_speed_priority"])
            background_flow = float(comparison["flow_background"])
            treatment_lap = treatment.get("mean_priority_lap_time")
            comparison_lap = comparison.get("mean_priority_lap_time")
            output.append(
                {
                    "requested_density": float(treatment["requested_density"]),
                    "effective_density": float(treatment["effective_density"]),
                    "density_index": density_index,
                    "run_index": run_index,
                    "av_fraction": av_fraction,
                    "treatment_mode": treatment_mode,
                    "comparison_mode": comparison_mode,
                    "pv_speed_treatment": float(
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
                    "completed_priority_laps_delta": float(
                        treatment["completed_priority_laps"]
                    )
                    - float(comparison["completed_priority_laps"]),
                    "priority_lap_time_delta": (
                        float(treatment_lap) - float(comparison_lap)
                        if treatment_lap is not None
                        and comparison_lap is not None
                        else None
                    ),
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
    groups: dict[
        tuple[float, float, str, str],
        list[dict[str, Any]],
    ] = {}
    for row in rows:
        key = (
            float(row["requested_density"]),
            float(row["av_fraction"]),
            str(row["treatment_mode"]),
            str(row["comparison_mode"]),
        )
        groups.setdefault(key, []).append(row)

    metrics = (
        "pv_speed_delta",
        "pv_speed_ratio",
        "priority_time_loss_delta",
        "background_flow_delta",
        "background_flow_ratio",
        "all_speed_delta",
        "priority_stopped_delta",
        "completed_priority_laps_delta",
        "priority_lap_time_delta",
    )
    output: list[dict[str, Any]] = []
    for group_index, (
        (density, av_fraction, treatment_mode, comparison_mode),
        group,
    ) in enumerate(sorted(groups.items())):
        result: dict[str, Any] = {
            "requested_density": density,
            "av_fraction": av_fraction,
            "treatment_mode": treatment_mode,
            "comparison_mode": comparison_mode,
            "runs": len(group),
            "pv_speed_win_rate": graph_eval._mean(
                float(row["pv_speed_delta"] > 0.0) for row in group
            ),
            "pv_speed_tie_rate": graph_eval._mean(
                float(row["pv_speed_delta"] == 0.0) for row in group
            ),
        }
        for metric_index, metric in enumerate(metrics):
            values = np.asarray(_finite_values(group, metric), dtype=np.float64)
            result[f"{metric}_valid_runs"] = int(values.size)
            if values.size == 0:
                result[f"{metric}_mean"] = None
                result[f"{metric}_std"] = None
                result[f"{metric}_ci_low"] = None
                result[f"{metric}_ci_high"] = None
                continue
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
        result["pv_speed_advantage"] = bool(
            result["pv_speed_delta_ci_low"] is not None
            and result["pv_speed_delta_ci_low"] > 0.0
        )
        output.append(result)
    return output


def _write_plots(
    out_dir: Path,
    prefix: str,
    summary: list[dict[str, Any]],
    paired_summary: list[dict[str, Any]],
    av_fractions: list[float],
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib is unavailable; plots were skipped", flush=True)
        return []

    colors = {
        BASELINE1: "#6b7280",
        BASELINE2: "#2563eb",
        NOOP: "#7c3aed",
        LEFT_IF_SAFE: "#059669",
        GRAPH_PPO: "#dc2626",
        NODE_MLP_PPO: "#ea580c",
        GRID_CNN_PPO: "#0891b2",
    }
    labels = {
        BASELINE1: "Baseline 1 (HDV)",
        BASELINE2: "Baseline 2",
        NOOP: "Noop",
        LEFT_IF_SAFE: "Left-if-safe",
        GRAPH_PPO: "Graph-PPO",
        NODE_MLP_PPO: "NodeMLP-PPO",
        GRID_CNN_PPO: "GridCNN-PPO",
    }
    metrics = (
        ("mean_speed_priority", "PV mean speed", "cells/step"),
        ("stopped_fraction_priority", "PV stopped fraction", "fraction"),
        ("mean_priority_lap_time", "PV mean lap time", "steps"),
        ("flow_background", "Background flow", "veh-cell/(cell step)"),
    )
    fig, axes = plt.subplots(
        len(av_fractions),
        len(metrics),
        figsize=(20, max(4.2, 4.0 * len(av_fractions))),
        squeeze=False,
    )
    for row_index, av_fraction in enumerate(av_fractions):
        for col_index, (metric, title, ylabel) in enumerate(metrics):
            axis = axes[row_index][col_index]
            for mode in MEASUREMENT_MODES:
                values = sorted(
                    (
                        row
                        for row in summary
                        if row["mode"] == mode
                        and math.isclose(
                            float(row["av_fraction"]),
                            av_fraction,
                            abs_tol=1e-12,
                        )
                        and row.get(f"{metric}_mean") is not None
                    ),
                    key=lambda row: float(row["requested_density"]),
                )
                if not values:
                    continue
                x = np.asarray(
                    [float(row["requested_density"]) for row in values]
                )
                y = np.asarray([float(row[f"{metric}_mean"]) for row in values])
                low = np.asarray(
                    [float(row[f"{metric}_ci95_low"]) for row in values]
                )
                high = np.asarray(
                    [float(row[f"{metric}_ci95_high"]) for row in values]
                )
                axis.plot(
                    x,
                    y,
                    marker="o",
                    markersize=3,
                    linewidth=1.5,
                    color=colors[mode],
                    label=labels[mode],
                )
                axis.fill_between(x, low, high, color=colors[mode], alpha=0.08)
            axis.set_title(f"{title}; AV={av_fraction:.2f}")
            axis.set_xlabel("Density")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
    axes[0][0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    overview_path = out_dir / f"{prefix}_overview.png"
    fig.savefig(overview_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(
        1,
        len(av_fractions),
        figsize=(6 * len(av_fractions), 4.5),
        squeeze=False,
    )
    for index, av_fraction in enumerate(av_fractions):
        axis = axes[0][index]
        for mode in (NOOP, LEFT_IF_SAFE) + LEARNED_MODES:
            values = sorted(
                (
                    row
                    for row in paired_summary
                    if row["treatment_mode"] == mode
                    and row["comparison_mode"] == BASELINE2
                    and math.isclose(
                        float(row["av_fraction"]),
                        av_fraction,
                        abs_tol=1e-12,
                    )
                ),
                key=lambda row: float(row["requested_density"]),
            )
            if not values:
                continue
            x = np.asarray([float(row["requested_density"]) for row in values])
            y = np.asarray([float(row["pv_speed_delta_mean"]) for row in values])
            low = np.asarray(
                [float(row["pv_speed_delta_ci_low"]) for row in values]
            )
            high = np.asarray(
                [float(row["pv_speed_delta_ci_high"]) for row in values]
            )
            axis.plot(
                x,
                y,
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=colors[mode],
                label=labels[mode],
            )
            axis.fill_between(x, low, high, color=colors[mode], alpha=0.10)
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_title(f"PV speed gain vs Baseline 2; AV={av_fraction:.2f}")
        axis.set_xlabel("Density")
        axis.set_ylabel("Delta speed [cells/step]")
        axis.grid(alpha=0.25)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    delta_path = out_dir / f"{prefix}_delta_vs_baseline2.png"
    fig.savefig(delta_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [overview_path, delta_path]


def _configure_process_threads(torch_threads: int) -> None:
    value = str(max(1, int(torch_threads)))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[name] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Joint paired multiprocessing evaluation of Graph-PPO, NodeMLP-PPO, "
            "GridCNN-PPO, and four shared control modes"
        )
    )
    parser.add_argument("--graph-checkpoint", type=Path, required=True)
    parser.add_argument("--node-mlp-checkpoint", type=Path, required=True)
    parser.add_argument("--grid-cnn-checkpoint", type=Path, required=True)
    parser.add_argument("--graph-run-config", type=Path)
    parser.add_argument("--node-mlp-run-config", type=Path)
    parser.add_argument("--grid-cnn-run-config", type=Path)
    parser.add_argument("--densities", default=DEFAULT_DENSITIES)
    parser.add_argument("--av-fractions", default=DEFAULT_AV_FRACTIONS)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=510_001)
    parser.add_argument("--workers", type=int, default=0, help="0 = auto")
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--road-length", type=int)
    parser.add_argument("--num-lanes", type=int)
    parser.add_argument("--front-distance", type=int)
    parser.add_argument("--back-distance", type=int)
    parser.add_argument("--sensor-distance", type=int)
    parser.add_argument("--cooldown-steps", type=int, default=1)
    parser.add_argument("--yield-distance", type=int)
    parser.add_argument("--yield-cooldown-steps", type=int, default=1)
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
        default=Path("artifacts/rl/pv_architecture_comparison"),
    )
    parser.add_argument("--prefix", default="pv_architecture_comparison")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="restore all checkpoints and run one short reference-backend task",
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

    try:
        densities = graph_eval._parse_densities(args.densities)
        av_fractions = _parse_av_fractions(args.av_fractions)
        graph_inputs = _load_checkpoint_inputs(
            checkpoint_arg=args.graph_checkpoint,
            run_config_arg=args.graph_run_config,
            args=args,
            av_fraction=av_fractions[0],
            policy_kind="graph",
        )
        node_inputs = _load_checkpoint_inputs(
            checkpoint_arg=args.node_mlp_checkpoint,
            run_config_arg=args.node_mlp_run_config,
            args=args,
            av_fraction=av_fractions[0],
            policy_kind="nongraph",
        )
        cnn_inputs = _load_checkpoint_inputs(
            checkpoint_arg=args.grid_cnn_checkpoint,
            run_config_arg=args.grid_cnn_run_config,
            args=args,
            av_fraction=av_fractions[0],
            policy_kind="nongraph",
        )
        _validate_compatible_configs(
            graph_inputs[3], node_inputs[3], cnn_inputs[3]
        )
        if node_inputs[3]["model_preset"] != "node-mlp-matched":
            raise ValueError(
                "the NodeMLP checkpoint must use model_preset=node-mlp-matched"
            )
        if not str(cnn_inputs[3]["model_preset"]).startswith("grid-cnn-"):
            raise ValueError("the GridCNN checkpoint must use a grid-cnn preset")
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    config = dict(graph_inputs[3])
    config["node_mlp_model_preset"] = str(node_inputs[3]["model_preset"])
    config["grid_cnn_model_preset"] = str(cnn_inputs[3]["model_preset"])
    config["torch_threads"] = int(args.torch_threads)

    if args.smoke:
        densities = densities[:1]
        av_fractions = av_fractions[:1]
        args.runs = 1
        args.workers = 1
        config.update(
            {
                "warmup_steps": 5,
                "measure_steps": 32,
                "backend": "reference",
                "validate": True,
            }
        )
        for inputs in (graph_inputs, node_inputs, cnn_inputs):
            inputs[3].update(config)

    _configure_process_threads(args.torch_threads)

    restore_specs = (
        (GRAPH_PPO, graph_inputs, graph_eval._restore_policy),
        (NODE_MLP_PPO, node_inputs, nongraph_eval._restore_policy),
        (GRID_CNN_PPO, cnn_inputs, nongraph_eval._restore_policy),
    )
    state_dicts: dict[str, Mapping[str, np.ndarray]] = {}
    model_configs: dict[str, Mapping[str, Any]] = {}
    restore_smokes: dict[str, Mapping[str, Any]] = {}
    for mode, inputs, restore in restore_specs:
        checkpoint, run_config_path, run_config, model_config = inputs
        print(f"{mode}_checkpoint={checkpoint}", flush=True)
        print(f"{mode}_run_config={run_config_path}", flush=True)
        print(f"{mode}_restore_smoke=starting", flush=True)
        state_dict, smoke = restore(checkpoint, run_config, model_config)
        state_dicts[mode] = state_dict
        model_configs[mode] = model_config
        restore_smokes[mode] = smoke
        print(
            f"{mode}_restore_smoke=passed "
            f"active_avs={smoke['active_av_count']} "
            f"action_shape={smoke['action_shape']}",
            flush=True,
        )

    tasks = [
        WorkerSpec(
            density=density,
            density_index=density_index,
            run_index=run_index,
            av_fractions=tuple(av_fractions),
            config=config,
        )
        for density_index, density in enumerate(densities)
        for run_index in range(args.runs)
    ]
    workers = args.workers or min(
        len(tasks), max(1, min(32, (os.cpu_count() or 2) - 1))
    )
    workers = min(workers, len(tasks))
    trajectories_per_task = 1 + len(av_fractions) * (
        len(MEASUREMENT_MODES) - 1
    )
    print(
        f"evaluation=starting tasks={len(tasks)} workers={workers} "
        f"av_fractions={len(av_fractions)} modes={len(MEASUREMENT_MODES)} "
        f"simulated_trajectories={len(tasks) * trajectories_per_task} "
        f"measure_steps={config['measure_steps']}",
        flush=True,
    )

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    if workers == 1:
        _worker_init(state_dicts, model_configs, args.torch_threads)
        iterator = map(run_replication, tasks)
        for completed, result in enumerate(iterator, start=1):
            rows.extend(result)
            elapsed = time.perf_counter() - started
            eta = elapsed / completed * (len(tasks) - completed)
            print(
                f"[{completed}/{len(tasks)}] elapsed={elapsed:.1f}s "
                f"eta={eta:.1f}s",
                flush=True,
            )
    else:
        context = mp.get_context("spawn")
        with context.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(state_dicts, model_configs, args.torch_threads),
        ) as pool:
            iterator = pool.imap_unordered(
                run_replication, tasks, chunksize=args.chunksize
            )
            for completed, result in enumerate(iterator, start=1):
                rows.extend(result)
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * (len(tasks) - completed)
                print(
                    f"[{completed}/{len(tasks)}] elapsed={elapsed:.1f}s "
                    f"eta={eta:.1f}s",
                    flush=True,
                )

    mode_rank = {mode: index for index, mode in enumerate(MEASUREMENT_MODES)}
    av_rank = {value: index for index, value in enumerate(av_fractions)}
    rows.sort(
        key=lambda row: (
            int(row["density_index"]),
            av_rank[float(row["av_fraction"])],
            int(row["run_index"]),
            mode_rank[str(row["mode"])],
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
    metadata_path = args.out_dir / f"{args.prefix}_metadata.json"

    graph_eval._write_csv(runs_path, rows)
    graph_eval._write_csv(summary_path, summary)
    graph_eval._write_csv(paired_path, paired_rows)
    graph_eval._write_csv(paired_summary_path, paired_summary)
    plot_paths = (
        _write_plots(
            args.out_dir,
            args.prefix,
            summary,
            paired_summary,
            av_fractions,
        )
        if args.plot
        else []
    )

    metadata = {
        "checkpoints": {
            GRAPH_PPO: str(graph_inputs[0]),
            NODE_MLP_PPO: str(node_inputs[0]),
            GRID_CNN_PPO: str(cnn_inputs[0]),
        },
        "run_configs": {
            GRAPH_PPO: str(graph_inputs[1]),
            NODE_MLP_PPO: str(node_inputs[1]),
            GRID_CNN_PPO: str(cnn_inputs[1]),
        },
        "restore_smokes": restore_smokes,
        "densities": densities,
        "av_fractions": av_fractions,
        "runs": args.runs,
        "seed": args.seed,
        "workers": workers,
        "chunksize": args.chunksize,
        "torch_threads": args.torch_threads,
        "confidence": args.confidence,
        "bootstrap_samples": args.bootstrap_samples,
        "modes": list(MEASUREMENT_MODES),
        "pairings": [
            {"treatment_mode": treatment, "comparison_mode": comparison}
            for treatment, comparison in PAIRINGS
        ],
        "task_count": len(tasks),
        "simulated_trajectories": len(tasks) * trajectories_per_task,
        "runs_csv_rows": len(rows),
        "paired_csv_rows": len(paired_rows),
        "baseline1_reuse": (
            "Baseline1 is simulated once per (density, run) and copied into "
            "each AV-fraction group; warmup and PV assignment are also shared."
        ),
        "config": config,
        "plots": [str(path) for path in plot_paths],
        "elapsed_seconds": time.perf_counter() - started,
    }
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)

    print(f"runs_csv={runs_path}", flush=True)
    print(f"summary_csv={summary_path}", flush=True)
    print(f"paired_csv={paired_path}", flush=True)
    print(f"paired_summary_csv={paired_summary_path}", flush=True)
    for path in plot_paths:
        print(f"plot={path}", flush=True)
    print(f"metadata={metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
