from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from snfs_traffic.core import SimulationParams, build_lane_order, build_occupancy, compute_neighbors
from snfs_traffic.core.indexing_kernels import (
    build_lane_order_kernel,
    build_occupancy_kernel,
    compute_neighbors_kernel,
)
from snfs_traffic.core.indexing_numba import (
    NUMBA_AVAILABLE,
    build_lane_order_numba,
    build_occupancy_numba,
    compute_neighbors_numba,
)
from snfs_traffic.scenarios import make_uniform_random_state
from snfs_traffic.topology import RingTopology

SCHEMA_VERSION = "snfs_indexing_benchmark_v1"


@dataclass(frozen=True)
class BenchmarkConfig:
    out_json: Path
    out_md: Path
    quick: bool
    repeat: int
    warmup: int
    sizes: str
    include_numba: bool


def _size_preset(name: str) -> dict[str, object]:
    presets = {
        "small": {"lane_length_cases": [(2, 200)], "densities": [0.05, 0.2, 0.5], "seeds": [0, 1], "repeat": 3, "warmup": 1},
        "default": {"lane_length_cases": [(2, 500), (4, 1000), (8, 2000)], "densities": [0.05, 0.2, 0.5, 0.8], "seeds": [0, 1, 2]},
        "stress": {"lane_length_cases": [(4, 5000), (8, 10000)], "densities": [0.05, 0.2, 0.5, 0.8], "seeds": [0, 1]},
    }
    if name not in presets:
        raise ValueError(f"unknown sizes preset: {name}")
    return presets[name]


def time_callable(fn, *, repeat: int, warmup: int) -> dict[str, object]:
    for _ in range(warmup):
        fn()
    samples_ns: list[int] = []
    for _ in range(repeat):
        start = perf_counter_ns()
        fn()
        end = perf_counter_ns()
        samples_ns.append(end - start)

    return summarize_samples(samples_ns, repeat=repeat, warmup=warmup)


def summarize_samples(samples_ns: list[int], *, repeat: int, warmup: int) -> dict[str, object]:
    min_ns = min(samples_ns)
    median_ns = statistics.median(samples_ns)
    mean_ns = statistics.fmean(samples_ns)
    max_ns = max(samples_ns)
    return {
        "repeat": repeat,
        "warmup": warmup,
        "samples_ns": samples_ns,
        "min_ns": min_ns,
        "median_ns": median_ns,
        "mean_ns": mean_ns,
        "max_ns": max_ns,
        "min_ms": min_ns / 1e6,
        "median_ms": median_ns / 1e6,
        "mean_ms": mean_ns / 1e6,
        "max_ms": max_ns / 1e6,
    }


def _add_throughput(m: dict[str, object], *, n_vehicles: int, n_cells: int) -> None:
    median_ns = float(m["median_ns"])
    if median_ns <= 0:
        m["vehicles_per_second"] = None
        m["cells_per_second"] = None
        return
    seconds = median_ns / 1e9
    m["vehicles_per_second"] = n_vehicles / seconds
    m["cells_per_second"] = n_cells / seconds


def _speedup(a_ns: float | int | None, b_ns: float | int | None) -> float | None:
    if a_ns is None or b_ns is None or float(a_ns) <= 0:
        return None
    return float(b_ns) / float(a_ns)


def _assert_equal_tuple(a: tuple[np.ndarray, ...], b: tuple[np.ndarray, ...], name: str) -> None:
    if len(a) != len(b):
        raise AssertionError(f"{name} length mismatch")
    for idx, (x, y) in enumerate(zip(a, b, strict=True)):
        if not np.array_equal(x, y):
            raise AssertionError(f"{name} mismatch at tuple index {idx}")


def run_benchmark(config: BenchmarkConfig) -> dict[str, object]:
    preset = _size_preset("small" if config.quick else config.sizes)
    repeat = int(preset.get("repeat", config.repeat)) if config.quick else config.repeat
    warmup = int(preset.get("warmup", config.warmup)) if config.quick else config.warmup

    include_numba = config.include_numba and NUMBA_AVAILABLE
    numba_version = None
    if NUMBA_AVAILABLE:
        import numba  # type: ignore

        numba_version = numba.__version__

    cases: list[dict[str, object]] = []
    first_call_notes: list[str] = []

    for num_lanes, road_length in preset["lane_length_cases"]:
        for density in preset["densities"]:
            for seed in preset["seeds"]:
                params = SimulationParams(num_lanes=num_lanes, road_length=road_length)
                topology = RingTopology(num_lanes=num_lanes, length=road_length)
                state = make_uniform_random_state(num_lanes=num_lanes, road_length=road_length, density=density, seed=seed)
                case_id = f"L{num_lanes}_R{road_length}_D{density:.2f}_S{seed}"

                occ_public = build_occupancy(state, params)
                lo_public = build_lane_order(occ_public, n_vehicles=state.n_vehicles)
                nb_public = compute_neighbors(state, *lo_public, topology)

                occ_kernel = build_occupancy_kernel(state.lane, state.pos, state.alive, num_lanes=params.num_lanes, road_length=params.road_length)
                lo_kernel = build_lane_order_kernel(occ_kernel, n_vehicles=state.n_vehicles)
                nb_kernel = compute_neighbors_kernel(state.lane, state.pos, state.alive, *lo_kernel, road_length=topology.length)

                if not np.array_equal(occ_public, occ_kernel):
                    raise AssertionError(f"occupancy mismatch for {case_id}")
                _assert_equal_tuple(lo_public, lo_kernel, f"lane-order {case_id}")
                _assert_equal_tuple(nb_public, nb_kernel, f"neighbors {case_id}")

                measurements: dict[str, dict[str, object]] = {}

                m_public = time_callable(lambda: compute_neighbors(state, *build_lane_order(build_occupancy(state, params), n_vehicles=state.n_vehicles), topology), repeat=repeat, warmup=warmup)
                m_kernel = time_callable(lambda: compute_neighbors_kernel(state.lane, state.pos, state.alive, *build_lane_order_kernel(build_occupancy_kernel(state.lane, state.pos, state.alive, num_lanes=params.num_lanes, road_length=params.road_length), n_vehicles=state.n_vehicles), road_length=topology.length), repeat=repeat, warmup=warmup)
                _add_throughput(m_public, n_vehicles=state.n_vehicles, n_cells=params.num_lanes * params.road_length)
                _add_throughput(m_kernel, n_vehicles=state.n_vehicles, n_cells=params.num_lanes * params.road_length)
                measurements["public_indexing_pipeline"] = m_public
                measurements["kernel_indexing_pipeline"] = m_kernel

                m_k_occ = time_callable(lambda: build_occupancy_kernel(state.lane, state.pos, state.alive, num_lanes=params.num_lanes, road_length=params.road_length), repeat=repeat, warmup=warmup)
                m_k_lo = time_callable(lambda: build_lane_order_kernel(occ_kernel, n_vehicles=state.n_vehicles), repeat=repeat, warmup=warmup)
                m_k_nb = time_callable(lambda: compute_neighbors_kernel(state.lane, state.pos, state.alive, *lo_kernel, road_length=topology.length), repeat=repeat, warmup=warmup)
                measurements["kernel_build_occupancy"] = m_k_occ
                measurements["kernel_build_lane_order"] = m_k_lo
                measurements["kernel_compute_neighbors"] = m_k_nb

                if include_numba:
                    compiled_already = bool(build_occupancy_numba.__name__)  # marker only
                    first_occ_start = perf_counter_ns(); occ_numba = build_occupancy_numba(state.lane, state.pos, state.alive, num_lanes=params.num_lanes, road_length=params.road_length); first_occ_ns = perf_counter_ns() - first_occ_start
                    first_lo_start = perf_counter_ns(); lo_numba = build_lane_order_numba(occ_numba, n_vehicles=state.n_vehicles); first_lo_ns = perf_counter_ns() - first_lo_start
                    first_nb_start = perf_counter_ns(); nb_numba = compute_neighbors_numba(state.lane, state.pos, state.alive, *lo_numba, road_length=topology.length); first_nb_ns = perf_counter_ns() - first_nb_start
                    first_pipe_ns = first_occ_ns + first_lo_ns + first_nb_ns

                    if not np.array_equal(occ_numba, occ_public):
                        raise AssertionError(f"numba occupancy mismatch for {case_id}")
                    _assert_equal_tuple(lo_numba, lo_public, f"numba lane-order {case_id}")
                    _assert_equal_tuple(nb_numba, nb_public, f"numba neighbors {case_id}")

                    measurements["numba_build_occupancy_first_call_compile_inclusive"] = summarize_samples([first_occ_ns], repeat=1, warmup=0)
                    measurements["numba_build_lane_order_first_call_compile_inclusive"] = summarize_samples([first_lo_ns], repeat=1, warmup=0)
                    measurements["numba_compute_neighbors_first_call_compile_inclusive"] = summarize_samples([first_nb_ns], repeat=1, warmup=0)
                    measurements["numba_indexing_pipeline_first_call_compile_inclusive"] = summarize_samples([first_pipe_ns], repeat=1, warmup=0)
                    if compiled_already:
                        first_call_notes.append("First-call compile-inclusive measurements may be cache/warm affected within this process.")

                    m_n_pipe = time_callable(lambda: compute_neighbors_numba(state.lane, state.pos, state.alive, *build_lane_order_numba(build_occupancy_numba(state.lane, state.pos, state.alive, num_lanes=params.num_lanes, road_length=params.road_length), n_vehicles=state.n_vehicles), road_length=topology.length), repeat=repeat, warmup=warmup)
                    m_n_occ = time_callable(lambda: build_occupancy_numba(state.lane, state.pos, state.alive, num_lanes=params.num_lanes, road_length=params.road_length), repeat=repeat, warmup=warmup)
                    m_n_lo = time_callable(lambda: build_lane_order_numba(occ_numba, n_vehicles=state.n_vehicles), repeat=repeat, warmup=warmup)
                    m_n_nb = time_callable(lambda: compute_neighbors_numba(state.lane, state.pos, state.alive, *lo_numba, road_length=topology.length), repeat=repeat, warmup=warmup)
                    _add_throughput(m_n_pipe, n_vehicles=state.n_vehicles, n_cells=params.num_lanes * params.road_length)
                    measurements["numba_indexing_pipeline_warmed"] = m_n_pipe
                    measurements["numba_build_occupancy_warmed"] = m_n_occ
                    measurements["numba_build_lane_order_warmed"] = m_n_lo
                    measurements["numba_compute_neighbors_warmed"] = m_n_nb

                k_ns = measurements["kernel_indexing_pipeline"]["median_ns"]
                p_ns = measurements["public_indexing_pipeline"]["median_ns"]
                measurements["kernel_indexing_pipeline"]["speedup_vs_public_median"] = _speedup(k_ns, p_ns)
                measurements["public_indexing_pipeline"]["speedup_vs_kernel_median"] = _speedup(p_ns, k_ns)
                if "numba_indexing_pipeline_warmed" in measurements:
                    n_ns = measurements["numba_indexing_pipeline_warmed"]["median_ns"]
                    measurements["numba_indexing_pipeline_warmed"]["speedup_vs_kernel_median"] = _speedup(n_ns, k_ns)
                    measurements["numba_indexing_pipeline_warmed"]["speedup_vs_public_median"] = _speedup(n_ns, p_ns)

                cases.append({
                    "case_id": case_id,
                    "num_lanes": num_lanes,
                    "road_length": road_length,
                    "density": density,
                    "seed": seed,
                    "n_vehicles": int(state.n_vehicles),
                    "alive_count": int(np.count_nonzero(state.alive)),
                    "occupied_cells": int(np.count_nonzero(occ_public >= 0)),
                    "measurements": measurements,
                })

    notes = [
        "Benchmark numbers are environment-dependent.",
        "Codex Cloud/CI numbers are indicative only.",
        "No simulator behavior is changed by this benchmark.",
    ]
    notes.extend(sorted(set(first_call_notes)))

    results = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "numba_available": bool(NUMBA_AVAILABLE),
        "numba_version": numba_version,
        "quick": config.quick,
        "repeat": repeat,
        "warmup": warmup,
        "sizes_preset": "small" if config.quick else config.sizes,
        "notes": notes,
        "cases": cases,
    }
    return results


def _format_report(data: dict[str, object]) -> str:
    lines = ["# S-NFS indexing benchmark report", "", "## Environment", ""]
    lines.extend([
        f"- Python: `{data['python_version']}`",
        f"- Platform: `{data['platform']}`",
        f"- Processor: `{data['processor']}`",
        f"- NumPy: `{data['numpy_version']}`",
        f"- Numba available: `{data['numba_available']}` (version: `{data.get('numba_version')}`)",
        f"- repeat/warmup: `{data['repeat']}` / `{data['warmup']}`",
        f"- sizes preset: `{data['sizes_preset']}`",
        "",
        "These numbers are environment-dependent. Codex Cloud/CI timings are indicative only and should not be treated as final production performance measurements.",
        "",
        "## Summary table",
        "",
        "| case_id | num_lanes | road_length | density | n_vehicles | public_median_ms | kernel_median_ms | numba_warmed_median_ms | kernel_vs_public_speedup | numba_vs_kernel_speedup | numba_vs_public_speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for case in data["cases"]:
        m = case["measurements"]
        pub = m["public_indexing_pipeline"]["median_ms"]
        ker = m["kernel_indexing_pipeline"]["median_ms"]
        kn = m["kernel_indexing_pipeline"].get("speedup_vs_public_median")
        nw = m.get("numba_indexing_pipeline_warmed", {})
        nms = nw.get("median_ms", "n/a")
        nvk = nw.get("speedup_vs_kernel_median", "n/a")
        nvp = nw.get("speedup_vs_public_median", "n/a")
        lines.append(f"| {case['case_id']} | {case['num_lanes']} | {case['road_length']} | {case['density']:.2f} | {case['n_vehicles']} | {pub:.4f} | {ker:.4f} | {nms if isinstance(nms, str) else format(nms, '.4f')} | {kn:.3f} | {nvk if isinstance(nvk, str) else format(nvk, '.3f')} | {nvp if isinstance(nvp, str) else format(nvp, '.3f')} |")

    lines.extend(["", "## Individual phase table", "", "| case_id | phase | kernel_median_ms | numba_warmed_median_ms | numba_vs_kernel_speedup |", "|---|---|---:|---:|---:|"])
    phase_map = [("build_occupancy", "kernel_build_occupancy", "numba_build_occupancy_warmed"), ("build_lane_order", "kernel_build_lane_order", "numba_build_lane_order_warmed"), ("compute_neighbors", "kernel_compute_neighbors", "numba_compute_neighbors_warmed")]
    for case in data["cases"]:
        m = case["measurements"]
        for label, kp, npk in phase_map:
            km = m[kp]["median_ms"]
            if npk in m:
                nm = m[npk]["median_ms"]
                sp = _speedup(m[npk]["median_ns"], m[kp]["median_ns"])
                lines.append(f"| {case['case_id']} | {label} | {km:.4f} | {nm:.4f} | {sp:.3f} |")
            else:
                lines.append(f"| {case['case_id']} | {label} | {km:.4f} | n/a | n/a |")

    lines.extend(["", "## First-call compile-inclusive timings", "", "These are measured once before warmed Numba timing loops in this process."])
    for case in data["cases"]:
        m = case["measurements"]
        if "numba_indexing_pipeline_first_call_compile_inclusive" not in m:
            continue
        lines.append(f"- {case['case_id']}: pipeline first-call compile-inclusive median ms = {m['numba_indexing_pipeline_first_call_compile_inclusive']['median_ms']:.4f}")

    lines.extend(["", "## Recommendation", ""])
    rec = _recommendation(data)
    lines.append(rec)
    return "\n".join(lines) + "\n"


def _recommendation(data: dict[str, object]) -> str:
    if not data["numba_available"]:
        return "Numba is unavailable in this run, so no Numba integration decision can be made from these results. Conservatively proceed with splitting longitudinal into pure-array kernels first."
    speedups = []
    for case in data["cases"]:
        if case["num_lanes"] < 4 or case["road_length"] < 1000:
            continue
        nw = case["measurements"].get("numba_indexing_pipeline_warmed")
        if nw is None:
            continue
        speedups.append(float(nw.get("speedup_vs_kernel_median") or 0.0))
    if speedups and all(s >= 1.5 for s in speedups):
        return "Warmed Numba indexing appears consistently >=1.5x faster on medium/large cases in this run. Consider a future explicit optimized backend path that uses Numba indexing while keeping ReferenceBackend/step_reference unchanged. Reconfirm on target hardware before deciding."
    return "Warmed Numba indexing speedups are small/inconsistent or not clearly decisive here. Conservatively, do not wire Numba indexing yet; split longitudinal phase into pure-array kernels and benchmark full-step phase costs again on target hardware."


def parse_args() -> BenchmarkConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--out-json", type=Path, default=Path("benchmark_indexing_results.json"))
    p.add_argument("--out-md", type=Path, default=Path("benchmark_indexing_results.md"))
    p.add_argument("--quick", action="store_true")
    p.add_argument("--repeat", type=int, default=7)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--sizes", choices=["small", "default", "stress"], default="default")
    p.add_argument("--include-numba", dest="include_numba", action="store_true", default=True)
    p.add_argument("--no-include-numba", dest="include_numba", action="store_false")
    a = p.parse_args()
    return BenchmarkConfig(out_json=a.out_json, out_md=a.out_md, quick=a.quick, repeat=a.repeat, warmup=a.warmup, sizes=a.sizes, include_numba=a.include_numba)


def main() -> int:
    config = parse_args()
    data = run_benchmark(config)
    config.out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    config.out_md.write_text(_format_report(data), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
