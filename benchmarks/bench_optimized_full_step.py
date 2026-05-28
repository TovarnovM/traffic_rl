from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from bench_full_step_phases import get_cases
from snfs_traffic.backends import get_backend
from snfs_traffic.core import SimulationParams, step_reference
from snfs_traffic.core.indexing_numba import NUMBA_AVAILABLE as INDEX_NUMBA_AVAILABLE
from snfs_traffic.core.lane_change_numba import NUMBA_AVAILABLE as LANE_NUMBA_AVAILABLE
from snfs_traffic.core.longitudinal_numba import NUMBA_AVAILABLE as LONG_NUMBA_AVAILABLE
from snfs_traffic.scenarios import make_uniform_random_state
from snfs_traffic.topology import RingTopology

NUMBA_AVAILABLE = INDEX_NUMBA_AVAILABLE and LANE_NUMBA_AVAILABLE and LONG_NUMBA_AVAILABLE

FIELDS = ("lane", "pos", "vel", "alive", "controlled", "changed_lane", "last_lane_delta")
PROFILE_COMPONENTS = (
    "pre_lane_change_index_neighbors_fused",
    "lane_change_proposals",
    "conflict_resolution",
    "apply_lane_changes",
    "post_lane_change_index_neighbors_fused",
    "longitudinal_random_draws",
    "longitudinal_velocity_numba",
    "position_advance",
    "state_copy_orchestration",
)


def _parse_requested_case_names(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _validate_requested_cases(parser: argparse.ArgumentParser, requested: list[str], available: list[str]) -> None:
    unknown = [name for name in requested if name not in available]
    if unknown:
        parser.error(
            "unknown case name(s): "
            + ", ".join(unknown)
            + ". valid case names: "
            + ", ".join(available)
        )


def _run_rollout(case, *, kind: str):
    params = SimulationParams(num_lanes=case.num_lanes, road_length=case.road_length, p_lane_change=case.p_lane_change)
    topology = RingTopology(num_lanes=case.num_lanes, length=case.road_length)
    state = make_uniform_random_state(
        num_lanes=case.num_lanes,
        road_length=case.road_length,
        density=case.density,
        seed=case.seed,
    )
    backend = get_backend("optimized")
    rng = np.random.default_rng(case.seed + 999)

    t0 = perf_counter_ns()
    for _ in range(case.steps):
        if kind == "reference":
            state = step_reference(state, params, topology, rng)
        else:
            state = backend.step(state, params, topology, rng)
    elapsed_ns = perf_counter_ns() - t0
    return elapsed_ns / case.steps, state, float(rng.random())


def _run_optimized_profiled_rollout(case) -> tuple[dict[str, float], object]:
    from snfs_traffic.core.indexing_numba import build_index_and_neighbors_numba
    from snfs_traffic.core.lane_change_kernels import apply_lane_changes_kernel, resolve_lane_change_conflicts_kernel
    from snfs_traffic.core.lane_change_numba import collect_lane_change_proposals_numba
    from snfs_traffic.core.longitudinal_numba import (
        advance_positions_numba,
        compute_longitudinal_velocities_numba,
        draw_longitudinal_randoms,
    )

    params = SimulationParams(num_lanes=case.num_lanes, road_length=case.road_length, p_lane_change=case.p_lane_change)
    topology = RingTopology(num_lanes=case.num_lanes, length=case.road_length)
    state = make_uniform_random_state(
        num_lanes=case.num_lanes,
        road_length=case.road_length,
        density=case.density,
        seed=case.seed,
    )
    rng = np.random.default_rng(case.seed + 999)
    totals_ns = {name: 0 for name in PROFILE_COMPONENTS}

    for _ in range(case.steps):
        t = perf_counter_ns()
        occupancy, lane_order, lane_counts, _, front_id, _, front_gap, _ = build_index_and_neighbors_numba(
            state.lane, state.pos, state.alive, num_lanes=params.num_lanes, road_length=params.road_length
        )
        totals_ns["pre_lane_change_index_neighbors_fused"] += perf_counter_ns() - t

        t = perf_counter_ns()
        proposals = collect_lane_change_proposals_numba(
            lane=state.lane, pos=state.pos, vel=state.vel, alive=state.alive, controlled=state.controlled,
            occupancy=occupancy, lane_order=lane_order, lane_counts=lane_counts, front_id=front_id, front_gap=front_gap,
            num_lanes=params.num_lanes, road_length=params.road_length, vmax_default=params.vmax_default,
            vmax_controlled=params.vmax_controlled, p_lane_change=params.p_lane_change, rng=rng
        )
        totals_ns["lane_change_proposals"] += perf_counter_ns() - t

        t = perf_counter_ns()
        accepted = resolve_lane_change_conflicts_kernel(proposals, rng, pos=state.pos, length=state.length, road_length=params.road_length)
        totals_ns["conflict_resolution"] += perf_counter_ns() - t

        t = perf_counter_ns()
        new_lane, new_changed_lane, new_last_lane_delta = apply_lane_changes_kernel(state.lane, state.changed_lane, state.last_lane_delta, accepted)
        totals_ns["apply_lane_changes"] += perf_counter_ns() - t

        t = perf_counter_ns()
        after_lane = state.copy()
        after_lane.lane = new_lane
        after_lane.changed_lane = new_changed_lane
        after_lane.last_lane_delta = new_last_lane_delta
        totals_ns["state_copy_orchestration"] += perf_counter_ns() - t

        t = perf_counter_ns()
        _, lane_order2, lane_counts2, lane_rank2, _, _, _, _ = build_index_and_neighbors_numba(
            after_lane.lane, after_lane.pos, after_lane.alive, num_lanes=params.num_lanes, road_length=params.road_length
        )
        totals_ns["post_lane_change_index_neighbors_fused"] += perf_counter_ns() - t

        t = perf_counter_ns()
        u_s, u_q, u_b = draw_longitudinal_randoms(after_lane.alive, rng)
        totals_ns["longitudinal_random_draws"] += perf_counter_ns() - t

        t = perf_counter_ns()
        new_vel = compute_longitudinal_velocities_numba(
            after_lane.lane, after_lane.pos, after_lane.vel, after_lane.alive, after_lane.controlled,
            lane_order2, lane_counts2, lane_rank2, u_s, u_q, u_b,
            road_length=params.road_length, vmax_default=params.vmax_default, vmax_controlled=params.vmax_controlled,
            G=params.G, q=params.q, r=params.r, S=params.S, P1=params.P1, P2=params.P2, P3=params.P3, P4=params.P4
        )
        totals_ns["longitudinal_velocity_numba"] += perf_counter_ns() - t

        t = perf_counter_ns()
        new_pos = advance_positions_numba(after_lane.pos, new_vel, after_lane.alive, road_length=params.road_length)
        totals_ns["position_advance"] += perf_counter_ns() - t

        t = perf_counter_ns()
        out = after_lane.copy()
        out.vel = new_vel
        out.pos = new_pos
        state = out
        totals_ns["state_copy_orchestration"] += perf_counter_ns() - t

    profile_ms_per_step = {k: (v / case.steps) / 1e6 for k, v in totals_ns.items()}
    return profile_ms_per_step, state


def _precompile_optimized_if_available() -> None:
    if not NUMBA_AVAILABLE:
        return
    warm = get_cases("smoke", seed=0, steps=1)[0]
    _run_rollout(warm, kind="optimized")


def _case_note() -> str:
    if not NUMBA_AVAILABLE:
        return "optimized backend falls back to reference (Numba unavailable)"
    return "optimized backend active"


def _recommend(cases: list[dict]) -> str:
    if not cases:
        return "keep backend experimental"
    if not all(c["equivalence"]["state_equal"] and c["equivalence"]["rng_next_draw_equal"] for c in cases):
        return "keep backend experimental"

    rep = [c for c in cases if c["name"] in {"medium_moderate", "medium_dense", "wide_moderate"}]
    if not rep:
        return "keep OptimizedBackend supported but do not merge indexing fast path"
    if all(c["speedup_x"] >= 1.10 for c in rep):
        return "keep OptimizedBackend supported and merge indexing fast path"
    if any(c["speedup_x"] > 1.0 for c in rep):
        return "keep OptimizedBackend supported but do not merge indexing fast path"
    return "keep backend experimental"


def _format_markdown(data: dict) -> str:
    lines = [
        "# Optimized full-step benchmark",
        "",
        f"Timestamp: {data['created_at_utc']}",
        "",
        "## Environment",
        "",
        f"- Python: {data['python']}",
        f"- Platform: {data['platform']}",
        f"- NumPy: {data['numpy_version']}",
        f"- Numba available: {data['numba_available']}",
        "",
        "## Comparison",
        "",
        "| case | vehicles | steps | reference mean ms/step | optimized mean ms/step | speedup x | equivalence | top optimized component | notes |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for case in data["cases"]:
        eq = "yes" if (case["equivalence"]["state_equal"] and case["equivalence"]["rng_next_draw_equal"]) else "no"
        lines.append(
            f"| {case['name']} | {case['n_vehicles']} | {case['steps']} | {case['reference']['mean_ms_per_step']:.4f} | "
            f"{case['optimized']['mean_ms_per_step']:.4f} | {case['speedup_x']:.3f} | {eq} | {case['optimized_profile']['top_component']} | {case['notes']} |"
        )
    lines += ["", "## Recommendation", "", data["recommendation"], ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["smoke", "standard"], default="smoke")
    parser.add_argument("--backend", choices=["reference", "optimized", "both"], default="both")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cases", type=str, default="")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    cases = get_cases(args.preset, args.seed, args.steps)
    if args.cases:
        requested = _parse_requested_case_names(args.cases)
        if not requested:
            parser.error("--cases was provided but no non-empty case names were given")
        available_case_names = [c.name for c in cases]
        _validate_requested_cases(parser, requested, available_case_names)
        requested_set = set(requested)
        cases = [c for c in cases if c.name in requested_set]
        if not cases:
            parser.error("--cases filter selected zero cases")

    _precompile_optimized_if_available()

    results: list[dict] = []
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for case in cases:
            for _ in range(args.warmups):
                if args.backend in {"reference", "both"}:
                    _run_rollout(case, kind="reference")
                if args.backend in {"optimized", "both"}:
                    _run_rollout(case, kind="optimized")

            reference_repeat_ns = []
            optimized_repeat_ns = []
            last_reference_state = None
            last_optimized_state = None
            last_reference_rng = None
            last_optimized_rng = None

            for _ in range(args.repeats):
                if args.backend in {"reference", "both"}:
                    rt, rs, rr = _run_rollout(case, kind="reference")
                    reference_repeat_ns.append(rt)
                    last_reference_state = rs
                    last_reference_rng = rr
                if args.backend in {"optimized", "both"}:
                    ot, os, orng = _run_rollout(case, kind="optimized")
                    optimized_repeat_ns.append(ot)
                    last_optimized_state = os
                    last_optimized_rng = orng

            if args.backend == "reference":
                optimized_repeat_ns = list(reference_repeat_ns)
                last_optimized_state = last_reference_state
                last_optimized_rng = last_reference_rng
            elif args.backend == "optimized":
                reference_repeat_ns = list(optimized_repeat_ns)
                last_reference_state = last_optimized_state
                last_reference_rng = last_optimized_rng

            state_equal = True
            for field in FIELDS:
                if not np.array_equal(getattr(last_reference_state, field), getattr(last_optimized_state, field)):
                    state_equal = False
                    break
            rng_equal = bool(last_reference_rng == last_optimized_rng)

            ref_mean = statistics.fmean(reference_repeat_ns)
            ref_median = statistics.median(reference_repeat_ns)
            opt_mean = statistics.fmean(optimized_repeat_ns)
            opt_median = statistics.median(optimized_repeat_ns)
            speedup = ref_mean / opt_mean if opt_mean > 0 else 0.0

            results.append(
                {
                    **asdict(case),
                    "n_vehicles": int(case.num_lanes * case.road_length * case.density),
                    "repeats": args.repeats,
                    "warmups": args.warmups,
                    "reference": {
                        "per_repeat_ms_per_step": [ns / 1e6 for ns in reference_repeat_ns],
                        "mean_ms_per_step": ref_mean / 1e6,
                        "median_ms_per_step": ref_median / 1e6,
                    },
                    "optimized": {
                        "per_repeat_ms_per_step": [ns / 1e6 for ns in optimized_repeat_ns],
                        "mean_ms_per_step": opt_mean / 1e6,
                        "median_ms_per_step": opt_median / 1e6,
                    },
                    "optimized_profile": {
                        "component_ms_per_step": {},
                        "top_component": "n/a",
                    },
                    "speedup_x": speedup,
                    "equivalence": {
                        "state_equal": state_equal,
                        "rng_next_draw_equal": rng_equal,
                    },
                    "backend_status": {
                        "optimized_requested": args.backend in {"optimized", "both"},
                        "optimized_available": NUMBA_AVAILABLE,
                        "fallback_to_reference": not NUMBA_AVAILABLE,
                    },
                    "notes": _case_note(),
                }
            )
            if NUMBA_AVAILABLE and args.backend in {"optimized", "both"}:
                prof, prof_state = _run_optimized_profiled_rollout(case)
                top_component = max(prof.items(), key=lambda kv: kv[1])[0]
                results[-1]["optimized_profile"] = {
                    "component_ms_per_step": prof,
                    "top_component": top_component,
                }
                if args.backend == "optimized":
                    state_equal_profile = True
                    for field in FIELDS:
                        if not np.array_equal(getattr(last_optimized_state, field), getattr(prof_state, field)):
                            state_equal_profile = False
                            break
                    results[-1]["optimized_profile"]["matches_optimized_state"] = state_equal_profile
    finally:
        if gc_enabled:
            gc.enable()

    data = {
        "benchmark": "snfs_optimized_full_step",
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "numba_available": NUMBA_AVAILABLE,
        "preset": args.preset,
        "seed": args.seed,
        "steps": args.steps,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "backend": args.backend,
        "cases": results,
    }
    data["recommendation"] = _recommend(results)

    if args.out_json is not None:
        args.out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if args.out_md is not None:
        args.out_md.write_text(_format_markdown(data), encoding="utf-8")

    print("Benchmark=snfs_optimized_full_step")


if __name__ == "__main__":
    main()
