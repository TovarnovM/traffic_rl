from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from snfs_traffic.core import SimulationParams
from snfs_traffic.scenarios import VehicleMix
from snfs_traffic.simulator import ScenarioConfig, TrafficSimulator




@dataclass(frozen=True)
class RunSpec:
    density: float
    density_index: int
    run_index: int
    args_dict: dict[str, object]


@dataclass(frozen=True)
class RunResult:
    requested_density: float
    effective_density: float
    density_index: int
    run_index: int
    scenario_seed: int
    rng_seed: int
    backend_requested: str
    backend_actual: str
    num_lanes: int
    road_length: int
    vmax: int
    n_vehicles: int
    warmup_steps: int
    measure_steps: int
    mean_speed: float
    velocity_sum_mean: float
    flow: float
    mean_stopped_fraction: float


@dataclass(frozen=True)
class SummaryResult:
    requested_density: float
    effective_density_mean: float
    effective_density_std: float
    runs: int
    n_vehicles_mean: float
    mean_speed_mean: float
    mean_speed_std: float
    velocity_sum_mean: float
    velocity_sum_std: float
    flow_mean: float
    flow_std: float
    flow_sem: float
    stopped_fraction_mean: float
    stopped_fraction_std: float


def parse_density_values(args: argparse.Namespace) -> list[float]:
    if args.densities:
        values = [float(part.strip()) for part in args.densities.split(",") if part.strip()]
    else:
        if args.density_count < 1:
            raise ValueError("--density-count must be >= 1")
        values = np.linspace(args.density_min, args.density_max, args.density_count).tolist()

    if not values:
        raise ValueError("no densities provided")
    for density in values:
        if not (0.0 <= density <= 1.0):
            raise ValueError(f"density must be in [0, 1], got {density}")
    return values


def make_params(args: argparse.Namespace) -> SimulationParams:
    return SimulationParams(
        num_lanes=args.num_lanes,
        road_length=args.road_length,
        vmax_default=args.vmax,
        vmax_controlled=args.vmax,
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


def measure_state(state, capacity: int) -> tuple[float, float, float, float]:
    alive = state.alive
    n_alive = int(np.count_nonzero(alive))
    if n_alive == 0:
        return 0.0, 0.0, 0.0, 0.0

    vel = state.vel[alive].astype(np.float64, copy=False)
    velocity_sum = float(np.sum(vel))
    mean_speed = velocity_sum / float(n_alive)

    # Fundamental-diagram flow requested here:
    #   q = sum_i(v_i) / (num_lanes * road_length)
    # where v_i is the velocity of vehicle i in cells/step.
    # Units: vehicles per road cell per step.
    flow = velocity_sum / float(capacity)
    stopped_fraction = float(np.count_nonzero(vel == 0.0)) / float(n_alive)
    return mean_speed, velocity_sum, flow, stopped_fraction


def run_one(
    *,
    params: SimulationParams,
    density: float,
    density_index: int,
    run_index: int,
    args: argparse.Namespace,
) -> RunResult:
    scenario_seed = int(args.seed + density_index * 100_000 + run_index * 1_000 + 11)
    rng_seed = int(args.seed + density_index * 100_000 + run_index * 1_000 + 97)

    sim = TrafficSimulator(
        params=params,
        backend=args.backend,
        rng_seed=rng_seed,
        scenario=ScenarioConfig(
            density=density,
            seed=scenario_seed,
            # HDV-only: no AV, no controlled vehicles, no buses; all length == 1.
            vehicle_mix=VehicleMix(
                av_fraction=0.0,
                controlled_fraction=0.0,
                bus_fraction=0.0,
            ),
        ),
        validate=not args.no_validate,
        require_all_controlled_actions=True,
    )

    state = sim.reset(scenario_seed=scenario_seed, rng_seed=rng_seed)
    if bool(np.any(state.controlled[state.alive])):
        raise RuntimeError("HDV-only scenario unexpectedly contains controlled vehicles")
    if bool(np.any(state.length[state.alive] != 1)):
        raise RuntimeError("HDV-only scenario unexpectedly contains non-unit vehicle lengths")

    capacity = params.num_lanes * params.road_length
    n_vehicles = int(np.count_nonzero(state.alive))
    effective_density = float(n_vehicles) / float(capacity)

    for _ in range(args.warmup_steps):
        state = sim.step()

    speed_values: list[float] = []
    velocity_sum_values: list[float] = []
    flow_values: list[float] = []
    stopped_values: list[float] = []
    for _ in range(args.measure_steps):
        state = sim.step()
        mean_speed, velocity_sum, flow, stopped_fraction = measure_state(
            state,
            capacity=capacity,
        )
        speed_values.append(mean_speed)
        velocity_sum_values.append(velocity_sum)
        flow_values.append(flow)
        stopped_values.append(stopped_fraction)

    mean_speed = float(statistics.fmean(speed_values)) if speed_values else 0.0
    velocity_sum_mean = float(statistics.fmean(velocity_sum_values)) if velocity_sum_values else 0.0
    flow = float(statistics.fmean(flow_values)) if flow_values else 0.0
    stopped_fraction = float(statistics.fmean(stopped_values)) if stopped_values else 0.0

    return RunResult(
        requested_density=float(density),
        effective_density=effective_density,
        density_index=density_index,
        run_index=run_index,
        scenario_seed=scenario_seed,
        rng_seed=rng_seed,
        backend_requested=args.backend,
        backend_actual=sim.backend_name,
        num_lanes=params.num_lanes,
        road_length=params.road_length,
        vmax=args.vmax,
        n_vehicles=n_vehicles,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        mean_speed=mean_speed,
        velocity_sum_mean=velocity_sum_mean,
        flow=flow,
        mean_stopped_fraction=stopped_fraction,
    )



def run_one_worker(spec: RunSpec) -> RunResult:
    # Worker entry point for multiprocessing. Keep it top-level so it is
    # picklable with the spawn start method as well as fork.
    args = argparse.Namespace(**spec.args_dict)
    params = make_params(args)
    return run_one(
        params=params,
        density=spec.density,
        density_index=spec.density_index,
        run_index=spec.run_index,
        args=args,
    )


def print_run_progress(case_idx: int, total_cases: int, result: RunResult) -> None:
    print(
        f"[{case_idx:4d}/{total_cases}] "
        f"rho_req={result.requested_density:.4f} "
        f"rho_eff={result.effective_density:.4f} "
        f"run={result.run_index + 1} "
        f"v_mean={result.mean_speed:.4f} "
        f"sum_v={result.velocity_sum_mean:.2f} "
        f"q={result.flow:.6f} "
        f"stopped={result.mean_stopped_fraction:.3f} "
        f"backend={result.backend_actual}"
    )


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.fmean(values)) if values else 0.0


def sample_std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def summarize(results: list[RunResult]) -> list[SummaryResult]:
    by_density: dict[float, list[RunResult]] = {}
    for result in results:
        by_density.setdefault(result.requested_density, []).append(result)

    summaries: list[SummaryResult] = []
    for density in sorted(by_density):
        rows = by_density[density]
        flow_std = sample_std(row.flow for row in rows)
        velocity_sum_std = sample_std(row.velocity_sum_mean for row in rows)
        summaries.append(
            SummaryResult(
                requested_density=float(density),
                effective_density_mean=mean(row.effective_density for row in rows),
                effective_density_std=sample_std(row.effective_density for row in rows),
                runs=len(rows),
                n_vehicles_mean=mean(row.n_vehicles for row in rows),
                mean_speed_mean=mean(row.mean_speed for row in rows),
                mean_speed_std=sample_std(row.mean_speed for row in rows),
                velocity_sum_mean=mean(row.velocity_sum_mean for row in rows),
                velocity_sum_std=velocity_sum_std,
                flow_mean=mean(row.flow for row in rows),
                flow_std=flow_std,
                flow_sem=flow_std / math.sqrt(len(rows)) if rows else 0.0,
                stopped_fraction_mean=mean(row.mean_stopped_fraction for row in rows),
                stopped_fraction_std=sample_std(row.mean_stopped_fraction for row in rows),
            )
        )
    return summaries


def write_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(path: Path, *, args: argparse.Namespace, summaries: list[SummaryResult], runs: list[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "summary": [asdict(row) for row in summaries],
        "runs": [asdict(row) for row in runs],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_results(path: Path, summaries: list[SummaryResult], title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional local install
        print(f"plot skipped: matplotlib import failed: {exc}")
        return

    if not summaries:
        return

    x = np.asarray([row.effective_density_mean for row in summaries], dtype=np.float64)
    y = np.asarray([row.flow_mean for row in summaries], dtype=np.float64)
    yerr = np.asarray([row.flow_sem for row in summaries], dtype=np.float64)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=140)
    ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=1.2)
    ax.set_xlabel("Density, vehicles / cell / lane")
    ax.set_ylabel("Flow = sum(v_i) / (lanes * road_length)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fundamental diagram q(rho) for an HDV-only Revised S-NFS "
            "ring-road simulation. Flow is computed directly as "
            "sum_i(v_i) / (num_lanes * road_length). Each point is averaged "
            "over several runs after a warm-up interval."
        )
    )
    parser.add_argument("--road-length", type=int, default=1000, help="ring length in cells")
    parser.add_argument("--num-lanes", type=int, default=4, help="number of lanes")
    parser.add_argument("--vmax", type=int, default=5, help="HDV max speed in cells/step")
    parser.add_argument("--backend", choices=["auto", "reference", "optimized"], default="auto")

    density_group = parser.add_argument_group("density sweep")
    density_group.add_argument("--densities", type=str, default="", help="comma-separated density list, overrides min/max/count")
    density_group.add_argument("--density-min", type=float, default=0.02)
    density_group.add_argument("--density-max", type=float, default=0.90)
    density_group.add_argument("--density-count", type=int, default=30)

    parser.add_argument("--runs", type=int, default=5, help="independent simulations per density point")
    parser.add_argument("--workers", type=int, default=1, help="multiprocessing workers; 1 disables multiprocessing; 0 uses os.cpu_count()")
    parser.add_argument("--chunksize", type=int, default=1, help="Pool.imap_unordered chunksize for run tasks")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="M stabilization steps discarded at start")
    parser.add_argument("--measure-steps", type=int, default=2000, help="N measured steps per run")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", type=Path, default=Path("fundamental_diagram_results"))
    parser.add_argument("--prefix", type=str, default="hdv_fd")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-validate", action="store_true", help="disable runtime invariant validation for faster long sweeps")

    params_group = parser.add_argument_group("Revised S-NFS parameters")
    params_group.add_argument("--G", type=int, default=15)
    params_group.add_argument("--q", type=float, default=0.99)
    params_group.add_argument("--r", type=float, default=0.99)
    params_group.add_argument("--S", type=int, default=2)
    params_group.add_argument("--P1", type=float, default=0.999)
    params_group.add_argument("--P2", type=float, default=0.99)
    params_group.add_argument("--P3", type=float, default=0.98)
    params_group.add_argument("--P4", type=float, default=0.01)
    params_group.add_argument("--p-lane-change", type=float, default=0.5)

    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.workers < 0:
        raise ValueError("--workers must be >= 0")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be >= 1")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if args.measure_steps < 1:
        raise ValueError("--measure-steps must be >= 1")
    if not args.prefix:
        raise ValueError("--prefix must be non-empty")
    return args


def main() -> None:
    args = parse_args()
    densities = parse_density_values(args)
    params = make_params(args)

    started = time.perf_counter()

    # One independent task = one (density, run_index) simulation.
    # Keep args as a plain dict so RunSpec is easy to pickle.
    args_dict = vars(args).copy()
    run_specs = [
        RunSpec(
            density=float(density),
            density_index=density_index,
            run_index=run_index,
            args_dict=args_dict,
        )
        for density_index, density in enumerate(densities)
        for run_index in range(args.runs)
    ]

    total_cases = len(run_specs)
    workers = int(args.workers)
    if workers == 0:
        workers = os.cpu_count() or 1

    run_results: list[RunResult] = []
    if workers == 1:
        for case_idx, spec in enumerate(run_specs, start=1):
            result = run_one_worker(spec)
            run_results.append(result)
            print_run_progress(case_idx, total_cases, result)
    else:
        print(
            f"Running {total_cases} independent simulations with "
            f"multiprocessing Pool(workers={workers}, chunksize={args.chunksize})"
        )
        with mp.Pool(processes=workers) as pool:
            iterator = pool.imap_unordered(
                run_one_worker,
                run_specs,
                chunksize=int(args.chunksize),
            )
            for case_idx, result in enumerate(iterator, start=1):
                run_results.append(result)
                print_run_progress(case_idx, total_cases, result)

    # imap_unordered returns tasks as they finish; sort before aggregation/output
    # so CSV/JSON are deterministic.
    run_results.sort(key=lambda row: (row.density_index, row.run_index))

    summaries = summarize(run_results)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.out_dir / f"{args.prefix}_summary.csv"
    runs_csv = args.out_dir / f"{args.prefix}_runs.csv"
    json_path = args.out_dir / f"{args.prefix}.json"
    png_path = args.out_dir / f"{args.prefix}.png"

    write_csv(summary_csv, summaries)
    write_csv(runs_csv, run_results)
    write_json(json_path, args=args, summaries=summaries, runs=run_results)

    if not args.no_plot:
        title = (
            f"HDV fundamental diagram: lanes={args.num_lanes}, "
            f"L={args.road_length}, vmax={args.vmax}, "
            f"runs={args.runs}, warmup={args.warmup_steps}, N={args.measure_steps}"
        )
        plot_results(png_path, summaries, title)

    elapsed = time.perf_counter() - started
    print("\nDone")
    print(f"  elapsed:      {elapsed:.2f} s")
    print(f"  workers:      {workers}")
    print(f"  summary_csv:  {summary_csv}")
    print(f"  runs_csv:     {runs_csv}")
    print(f"  json:         {json_path}")
    if not args.no_plot:
        print(f"  plot:         {png_path}")


if __name__ == "__main__":
    main()



"""

PYTHONPATH=src python research/fundamental_diagram_hdv_flow_mp.py \
  --road-length 1000 \
  --num-lanes 1 \
  --vmax 5 \
  --density-min 0.02 \
  --density-max 0.90 \
  --density-count 30 \
  --runs 10 \
  --warmup-steps 1000 \
  --measure-steps 2000 \
  --backend auto \
  --workers 24 \
  --chunksize 1 \
  --out-dir tmp/fd_results \
  --prefix hdv_1lane_vmax5

PYTHONPATH=src python research/fundamental_diagram_hdv_flow_mp.py \
  --road-length 1000 \
  --num-lanes 3 \
  --vmax 5 \
  --density-min 0.02 \
  --density-max 0.95 \
  --density-count 30 \
  --runs 20 \
  --warmup-steps 1000 \
  --measure-steps 2000 \
  --backend auto \
  --workers 24 \
  --chunksize 1 \
  --out-dir tmp/fd_results \
  --prefix hdv_3lane_vmax5


"""