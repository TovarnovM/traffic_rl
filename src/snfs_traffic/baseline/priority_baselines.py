"""Paired Baseline-1/Baseline-2 experiments for priority vehicles.

The runner warms one HDV-only snapshot per ``(density, run)`` and reuses that
snapshot and the same measurement RNG seed for every controller mode.  This is
the common-random-numbers design used for paired treatment comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from snfs_traffic.control import LateralOverrideResult
from snfs_traffic.core import SimulationParams
from snfs_traffic.metrics import PriorityMetricsAccumulator, paired_bootstrap_interval
from snfs_traffic.rules import PriorityYieldConfig, PriorityYieldController
from snfs_traffic.scenarios import (
    PriorityVehicleConfig,
    VehicleMix,
    assign_priority_vehicles,
    select_yielding_vehicle_ids,
)
from snfs_traffic.simulator import ScenarioConfig, TrafficSimulator


BASELINE1 = "baseline1_hdv"
BASELINE2_ALL = "baseline2_all_hdv_yield"
BASELINE2_MATCHED_PREFIX = "baseline2_matched_"

SUMMARY_METRICS = (
    "flow_all",
    "flow_background",
    "mean_speed_all",
    "mean_speed_background",
    "mean_speed_priority",
    "worst_priority_mean_speed",
    "mean_priority_time_loss",
    "stopped_fraction_all",
    "stopped_fraction_priority",
    "lane_change_rate_all",
    "lane_change_rate_priority",
    "rule_requests",
    "rule_applied",
)


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    density: float
    density_index: int
    run_index: int
    args_dict: dict[str, object]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.fmean(values)) if values else 0.0


def _sample_std(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.stdev(values)) if len(values) >= 2 else 0.0


def parse_float_list(raw: str, *, name: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def parse_densities(args: argparse.Namespace) -> list[float]:
    values = (
        parse_float_list(args.densities, name="--densities")
        if args.densities
        else np.linspace(
            args.density_min, args.density_max, args.density_count
        ).tolist()
    )
    if any(not 0.0 < value < 1.0 for value in values):
        raise ValueError("densities must be in (0, 1)")
    return values


def parse_matched_fractions(raw: str) -> list[float]:
    values = parse_float_list(raw, name="--matched-fractions")
    if any(not 0.0 < value <= 1.0 for value in values):
        raise ValueError("matched fractions must be in (0, 1]")
    return values


def parse_priority_scenarios(raw: str) -> list[str]:
    aliases = {
        "one": "one",
        "1": "one",
        "two-g": "two-g",
        "2-g": "two-g",
        "two-2g": "two-2g",
        "2-2g": "two-2g",
    }
    values: list[str] = []
    for part in raw.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError("priority scenarios must use one, two-g, or two-2g")
        normalized = aliases[key]
        if normalized not in values:
            values.append(normalized)
    if not values:
        raise ValueError("--priority-scenarios must not be empty")
    return values


def make_params(args: argparse.Namespace) -> SimulationParams:
    return SimulationParams(
        num_lanes=args.num_lanes,
        road_length=args.road_length,
        vmax_default=args.vmax_hdv,
        vmax_controlled=args.vmax_priority,
        G=args.G,
        q=args.q,
        r=args.r,
        S=args.S,
        P1=args.P1,
        P2=args.P2,
        P3=args.P3,
        P4=args.P4,
        p_lane_change=args.p_lane_change,
    )


def priority_config(name: str, *, G: int, seed: int) -> PriorityVehicleConfig:
    if name == "one":
        return PriorityVehicleConfig(
            count=1, placement="random", target_gap=G, seed=seed
        )
    if name == "two-g":
        return PriorityVehicleConfig(
            count=2, placement="convoy", target_gap=G, seed=seed
        )
    if name == "two-2g":
        return PriorityVehicleConfig(
            count=2, placement="convoy", target_gap=2 * G, seed=seed
        )
    raise ValueError(f"unknown priority scenario {name!r}")


def _warm_snapshot(
    args: argparse.Namespace,
    params: SimulationParams,
    *,
    density: float,
    scenario_seed: int,
    warmup_rng_seed: int,
):
    sim = TrafficSimulator(
        params=params,
        backend=args.backend,
        rng_seed=warmup_rng_seed,
        scenario=ScenarioConfig(
            density=density,
            seed=scenario_seed,
            vehicle_mix=VehicleMix(),
        ),
        validate=not args.no_validate,
    )
    state = sim.reset(scenario_seed=scenario_seed, rng_seed=warmup_rng_seed)
    for _ in range(args.warmup_steps):
        state = sim.step()
    return state, sim.backend_name


def _measure_mode(
    args: argparse.Namespace,
    params: SimulationParams,
    *,
    initial_state,
    priority_ids: tuple[int, ...],
    measurement_rng_seed: int,
    controller: PriorityYieldController | None,
) -> tuple[dict[str, object], str]:
    sim = TrafficSimulator(
        params=params,
        backend=args.backend,
        rng_seed=measurement_rng_seed,
        validate=not args.no_validate,
    )
    state = sim.reset(state=initial_state, rng_seed=measurement_rng_seed)
    metrics = PriorityMetricsAccumulator(params, priority_vehicle_ids=priority_ids)
    backend_paths: set[str] = set()
    for _ in range(args.measure_steps):
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
                    raise RuntimeError(
                        "lateral override step did not return LateralOverrideResult"
                    )
                controller.observe(result)
            else:
                # Preserve the optimized native path and RNG cadence whenever
                # the rule has no intervention to apply.
                state = sim.step()
            metrics.observe(state, decision=decision, result=result)
        if sim.last_step_info is not None:
            backend_paths.add(sim.last_step_info.backend_name)
    summary = metrics.summary().to_dict()
    actual_backend = ";".join(sorted(backend_paths)) or sim.backend_name
    return summary, actual_backend


def run_density_replication(spec: WorkerSpec) -> list[dict[str, object]]:
    args = argparse.Namespace(**spec.args_dict)
    params = make_params(args)
    matched_fractions = parse_matched_fractions(args.matched_fractions)
    priority_scenarios = parse_priority_scenarios(args.priority_scenarios)
    base = int(args.seed + spec.density_index * 1_000_000 + spec.run_index * 10_000)
    scenario_seed = base + 11
    warmup_rng_seed = base + 97
    measurement_rng_seed = base + 193
    warm_state, warmup_backend = _warm_snapshot(
        args,
        params,
        density=spec.density,
        scenario_seed=scenario_seed,
        warmup_rng_seed=warmup_rng_seed,
    )
    capacity = params.num_lanes * params.road_length
    effective_density = int(np.sum(warm_state.alive)) / capacity
    rows: list[dict[str, object]] = []

    for scenario_index, scenario_name in enumerate(priority_scenarios):
        assignment_seed = base + 1_000 + scenario_index * 101
        assignment = assign_priority_vehicles(
            warm_state,
            priority_config(scenario_name, G=params.G, seed=assignment_seed),
            road_length=params.road_length,
        )
        base_state = assignment.state
        modes: list[tuple[str, float, object, tuple[int, ...] | None, int | None]] = [
            (BASELINE1, 0.0, base_state, None, None),
            (BASELINE2_ALL, 1.0, base_state, None, None),
        ]
        matched_selection_seed = base + 2_000 + scenario_index * 101
        for fraction in matched_fractions:
            matched_state, eligible_ids = select_yielding_vehicle_ids(
                base_state,
                fraction=fraction,
                seed=matched_selection_seed,
            )
            label = f"{BASELINE2_MATCHED_PREFIX}{fraction:.3f}"
            modes.append(
                (
                    label,
                    float(fraction),
                    matched_state,
                    eligible_ids,
                    matched_selection_seed,
                )
            )

        for (
            mode,
            controller_fraction,
            mode_state,
            eligible_ids,
            selection_seed,
        ) in modes:
            controller = None
            if mode != BASELINE1:
                controller = PriorityYieldController(
                    PriorityYieldConfig(
                        detection_distance=args.detection_distance,
                        cooldown_steps=args.cooldown_steps,
                    ),
                    eligible_vehicle_ids=eligible_ids,
                )
            summary, measurement_backend = _measure_mode(
                args,
                params,
                initial_state=mode_state,
                priority_ids=assignment.vehicle_ids,
                measurement_rng_seed=measurement_rng_seed,
                controller=controller,
            )
            if mode == BASELINE1:
                controller_vehicle_ids = ""
                controller_vehicle_count = 0
            elif mode == BASELINE2_ALL:
                controller_vehicle_ids = "all_non_priority"
                controller_vehicle_count = int(np.sum(mode_state.alive)) - len(
                    assignment.vehicle_ids
                )
            else:
                assert eligible_ids is not None
                controller_vehicle_ids = ",".join(str(value) for value in eligible_ids)
                controller_vehicle_count = len(eligible_ids)
            row: dict[str, object] = {
                "requested_density": float(spec.density),
                "effective_density": float(effective_density),
                "density_index": spec.density_index,
                "run_index": spec.run_index,
                "priority_scenario": scenario_name,
                "priority_vehicle_ids": ",".join(
                    str(value) for value in assignment.vehicle_ids
                ),
                "priority_lane": assignment.lane,
                "priority_actual_gaps": ",".join(
                    str(value) for value in assignment.actual_gaps
                ),
                "mode": mode,
                "controller_fraction": controller_fraction,
                "controller_vehicle_count": controller_vehicle_count,
                "controller_vehicle_ids": controller_vehicle_ids,
                "scenario_seed": scenario_seed,
                "warmup_rng_seed": warmup_rng_seed,
                "measurement_rng_seed": measurement_rng_seed,
                "assignment_seed": assignment_seed,
                "controller_selection_seed": selection_seed,
                "backend_requested": args.backend,
                "warmup_backend": warmup_backend,
                "measurement_backend": measurement_backend,
                "num_lanes": params.num_lanes,
                "road_length": params.road_length,
                "vmax_hdv": params.vmax_default,
                "vmax_priority": params.vmax_controlled,
                "warmup_steps": args.warmup_steps,
            }
            row.update(summary)
            rows.append(row)
    return rows


def worker_entry(spec: WorkerSpec) -> list[dict[str, object]]:
    return run_density_replication(spec)


def grouped_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[float, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            float(row["requested_density"]),
            str(row["priority_scenario"]),
            str(row["mode"]),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for (density, scenario, mode), group in sorted(groups.items()):
        result: dict[str, object] = {
            "requested_density": density,
            "effective_density_mean": _mean(
                float(row["effective_density"]) for row in group
            ),
            "priority_scenario": scenario,
            "mode": mode,
            "runs": len(group),
            "controller_fraction": float(group[0]["controller_fraction"]),
        }
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in group]
            result[f"{metric}_mean"] = _mean(values)
            result[f"{metric}_std"] = _sample_std(values)
            result[f"{metric}_sem"] = _sample_std(values) / math.sqrt(len(values))
        output.append(result)
    return output


def paired_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    baselines = {
        (
            int(row["density_index"]),
            int(row["run_index"]),
            str(row["priority_scenario"]),
        ): row
        for row in rows
        if row["mode"] == BASELINE1
    }
    output: list[dict[str, object]] = []
    for treatment in rows:
        if treatment["mode"] == BASELINE1:
            continue
        key = (
            int(treatment["density_index"]),
            int(treatment["run_index"]),
            str(treatment["priority_scenario"]),
        )
        baseline = baselines[key]
        baseline_background_flow = float(baseline["flow_background"])
        background_ratio = (
            float(treatment["flow_background"]) / baseline_background_flow
            if baseline_background_flow > 0.0
            else None
        )
        baseline_lap = baseline["mean_priority_lap_time"]
        treatment_lap = treatment["mean_priority_lap_time"]
        output.append(
            {
                "requested_density": float(treatment["requested_density"]),
                "effective_density": float(treatment["effective_density"]),
                "density_index": int(treatment["density_index"]),
                "run_index": int(treatment["run_index"]),
                "priority_scenario": str(treatment["priority_scenario"]),
                "treatment_mode": str(treatment["mode"]),
                "controller_fraction": float(treatment["controller_fraction"]),
                "priority_speed_delta": float(treatment["mean_speed_priority"])
                - float(baseline["mean_speed_priority"]),
                "worst_priority_speed_delta": float(
                    treatment["worst_priority_mean_speed"]
                )
                - float(baseline["worst_priority_mean_speed"]),
                "priority_time_loss_delta": float(treatment["mean_priority_time_loss"])
                - float(baseline["mean_priority_time_loss"]),
                "background_flow_ratio": background_ratio,
                "background_flow_loss": None
                if background_ratio is None
                else 1.0 - background_ratio,
                "priority_lap_time_delta": (
                    None
                    if baseline_lap is None or treatment_lap is None
                    else float(treatment_lap) - float(baseline_lap)
                ),
            }
        )
    return output


def paired_summary(
    rows: list[dict[str, object]],
    *,
    bootstrap_samples: int,
    seed: int,
    max_background_flow_loss: float,
    min_effective_pairs: int,
) -> list[dict[str, object]]:
    groups: dict[tuple[float, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            float(row["requested_density"]),
            str(row["priority_scenario"]),
            str(row["treatment_mode"]),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for group_index, ((density, scenario, mode), group) in enumerate(
        sorted(groups.items())
    ):
        pv_delta = np.asarray([float(row["priority_speed_delta"]) for row in group])
        worst_delta = np.asarray(
            [float(row["worst_priority_speed_delta"]) for row in group]
        )
        pv_low, pv_high = paired_bootstrap_interval(
            pv_delta, samples=bootstrap_samples, seed=seed + group_index * 2
        )
        worst_low, worst_high = paired_bootstrap_interval(
            worst_delta, samples=bootstrap_samples, seed=seed + group_index * 2 + 1
        )
        background_ratios = [
            float(row["background_flow_ratio"])
            for row in group
            if row["background_flow_ratio"] is not None
        ]
        mean_background_ratio = _mean(background_ratios) if background_ratios else None
        sufficient_pairs = len(group) >= min_effective_pairs
        pv_effective = pv_low > 0.0
        background_acceptable = (
            mean_background_ratio is not None
            and mean_background_ratio >= 1.0 - max_background_flow_loss
        )
        output.append(
            {
                "requested_density": density,
                "priority_scenario": scenario,
                "treatment_mode": mode,
                "controller_fraction": float(group[0]["controller_fraction"]),
                "pairs": len(group),
                "priority_speed_delta_mean": float(np.mean(pv_delta)),
                "priority_speed_delta_ci95_low": pv_low,
                "priority_speed_delta_ci95_high": pv_high,
                "worst_priority_speed_delta_mean": float(np.mean(worst_delta)),
                "worst_priority_speed_delta_ci95_low": worst_low,
                "worst_priority_speed_delta_ci95_high": worst_high,
                "background_flow_ratio_mean": mean_background_ratio,
                "background_flow_loss_mean": (
                    None
                    if mean_background_ratio is None
                    else 1.0 - mean_background_ratio
                ),
                "sufficient_pairs": sufficient_pairs,
                "pv_improvement_significant": pv_effective,
                "background_flow_acceptable": background_acceptable,
                "effective_region": bool(
                    sufficient_pairs and pv_effective and background_acceptable
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True, ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_results(
    path: Path, summary: list[dict[str, object]], paired: list[dict[str, object]]
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"plot skipped: matplotlib import failed: {exc}")
        return
    if not summary:
        return
    scenarios = sorted({str(row["priority_scenario"]) for row in summary})
    all_modes = {str(row["mode"]) for row in summary}
    mode_order = [mode for mode in (BASELINE1, BASELINE2_ALL) if mode in all_modes]
    mode_order.extend(sorted(mode for mode in all_modes if mode not in mode_order))
    color_map = {mode: f"C{index % 10}" for index, mode in enumerate(mode_order)}
    fig, axes = plt.subplots(
        len(scenarios), 4, figsize=(19, 4.2 * len(scenarios)), dpi=140
    )
    axes = np.asarray(axes, dtype=object).reshape(len(scenarios), 4)
    for scenario_index, scenario in enumerate(scenarios):
        scenario_rows = [row for row in summary if row["priority_scenario"] == scenario]
        modes = [
            mode
            for mode in mode_order
            if any(row["mode"] == mode for row in scenario_rows)
        ]
        for mode in modes:
            mode_rows = sorted(
                (row for row in scenario_rows if row["mode"] == mode),
                key=lambda row: float(row["effective_density_mean"]),
            )
            x = [float(row["effective_density_mean"]) for row in mode_rows]
            axes[scenario_index, 0].plot(
                x,
                [float(row["flow_all_mean"]) for row in mode_rows],
                marker="o",
                label=mode,
                color=color_map[mode],
            )
            axes[scenario_index, 1].plot(
                x,
                [float(row["mean_speed_all_mean"]) for row in mode_rows],
                marker="o",
                label=mode,
                color=color_map[mode],
            )
            axes[scenario_index, 2].plot(
                x,
                [float(row["mean_speed_priority_mean"]) for row in mode_rows],
                marker="o",
                label=mode,
                color=color_map[mode],
            )
        paired_scenario = [
            row for row in paired if row["priority_scenario"] == scenario
        ]
        for mode in sorted({str(row["treatment_mode"]) for row in paired_scenario}):
            mode_rows = sorted(
                (row for row in paired_scenario if row["treatment_mode"] == mode),
                key=lambda row: float(row["requested_density"]),
            )
            axes[scenario_index, 3].plot(
                [float(row["requested_density"]) for row in mode_rows],
                [float(row["background_flow_ratio_mean"]) for row in mode_rows],
                marker="o",
                label=mode,
                color=color_map[mode],
            )
        axes[scenario_index, 0].set_ylabel(f"{scenario}\nflow all")
        axes[scenario_index, 1].set_ylabel("all-traffic mean speed")
        axes[scenario_index, 2].set_ylabel("PV mean speed")
        axes[scenario_index, 3].set_ylabel("background flow B2 / B1")
        axes[scenario_index, 3].axhline(
            0.95, color="black", linestyle="--", linewidth=1
        )
        for axis in axes[scenario_index]:
            axis.set_xlabel("density")
            axis.grid(True, alpha=0.3)
        axes[scenario_index, 0].legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--road-length", type=int, default=1000)
    parser.add_argument("--num-lanes", type=int, default=4)
    parser.add_argument("--vmax-hdv", type=int, default=5)
    parser.add_argument("--vmax-priority", type=int, default=6)
    parser.add_argument(
        "--backend", choices=["auto", "reference", "optimized"], default="auto"
    )
    parser.add_argument("--densities", default="")
    parser.add_argument("--density-min", type=float, default=0.02)
    parser.add_argument("--density-max", type=float, default=0.90)
    parser.add_argument("--density-count", type=int, default=30)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--measure-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--priority-scenarios", default="one,two-g,two-2g")
    parser.add_argument("--matched-fractions", default="0.05,0.10,0.20")
    parser.add_argument("--detection-distance", type=int, default=30)
    parser.add_argument("--cooldown-steps", type=int, default=5)
    parser.add_argument("--max-background-flow-loss", type=float, default=0.05)
    parser.add_argument("--min-effective-pairs", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("priority_baseline_results")
    )
    parser.add_argument("--prefix", default="priority_baselines")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--G", type=int, default=15)
    parser.add_argument("--q", type=float, default=0.99)
    parser.add_argument("--r", type=float, default=0.99)
    parser.add_argument("--S", type=int, default=2)
    parser.add_argument("--P1", type=float, default=0.999)
    parser.add_argument("--P2", type=float, default=0.99)
    parser.add_argument("--P3", type=float, default=0.98)
    parser.add_argument("--P4", type=float, default=0.01)
    parser.add_argument("--p-lane-change", type=float, default=0.5)
    args = parser.parse_args(argv)
    for name in (
        "road_length",
        "num_lanes",
        "runs",
        "measure_steps",
        "density_count",
        "chunksize",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.warmup_steps < 0 or args.workers < 0:
        raise ValueError("warmup steps and workers must be >= 0")
    if not 0.0 <= args.max_background_flow_loss < 1.0:
        raise ValueError("--max-background-flow-loss must be in [0, 1)")
    if args.min_effective_pairs < 1 or args.bootstrap_samples < 1:
        raise ValueError("effective pairs and bootstrap samples must be >= 1")
    parse_densities(args)
    parse_matched_fractions(args.matched_fractions)
    parse_priority_scenarios(args.priority_scenarios)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    densities = parse_densities(args)
    specs = [
        WorkerSpec(float(density), density_index, run_index, vars(args).copy())
        for density_index, density in enumerate(densities)
        for run_index in range(args.runs)
    ]
    workers = (os.cpu_count() or 1) if args.workers == 0 else args.workers
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    if workers == 1:
        for index, spec in enumerate(specs, start=1):
            result = worker_entry(spec)
            rows.extend(result)
            print(
                f"[{index}/{len(specs)}] density={spec.density:.4f} run={spec.run_index + 1}"
            )
    else:
        with mp.Pool(workers) as pool:
            iterator = pool.imap_unordered(
                worker_entry, specs, chunksize=args.chunksize
            )
            for index, result in enumerate(iterator, start=1):
                rows.extend(result)
                print(f"[{index}/{len(specs)}] replication complete")
    rows.sort(
        key=lambda row: (
            int(row["density_index"]),
            int(row["run_index"]),
            str(row["priority_scenario"]),
            str(row["mode"]),
        )
    )
    summary = grouped_summary(rows)
    paired = paired_rows(rows)
    paired_aggregate = paired_summary(
        paired,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + 999_983,
        max_background_flow_loss=args.max_background_flow_loss,
        min_effective_pairs=args.min_effective_pairs,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "runs": args.out_dir / f"{args.prefix}_runs.csv",
        "summary": args.out_dir / f"{args.prefix}_summary.csv",
        "paired": args.out_dir / f"{args.prefix}_paired.csv",
        "paired_summary": args.out_dir / f"{args.prefix}_paired_summary.csv",
        "json": args.out_dir / f"{args.prefix}.json",
        "plot": args.out_dir / f"{args.prefix}.png",
    }
    write_csv(paths["runs"], rows)
    write_csv(paths["summary"], summary)
    write_csv(paths["paired"], paired)
    write_csv(paths["paired_summary"], paired_aggregate)
    write_json(
        paths["json"],
        {
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "runs": rows,
            "summary": summary,
            "paired": paired,
            "paired_summary": paired_aggregate,
        },
    )
    if not args.no_plot:
        plot_results(paths["plot"], summary, paired_aggregate)
    print(f"Done in {time.perf_counter() - started:.2f}s")
    for label, path in paths.items():
        if label != "plot" or not args.no_plot:
            print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
