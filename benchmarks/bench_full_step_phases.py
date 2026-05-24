from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from snfs_traffic.core import (
    SimulationParams,
    build_lane_order,
    build_occupancy,
    compute_neighbors,
    step_longitudinal_reference,
    step_reference,
    validate_state,
)
from snfs_traffic.core.lane_change_kernels import (
    apply_lane_changes_kernel,
    collect_lane_change_proposals_kernel,
    resolve_lane_change_conflicts_kernel,
)
from snfs_traffic.scenarios import make_uniform_random_state
from snfs_traffic.topology import RingTopology

PHASE_KEYS = [
    "validate_state_input",
    "build_occupancy_pre_lane_change",
    "build_lane_order_pre_lane_change",
    "compute_neighbors_pre_lane_change",
    "lane_change_collect_proposals",
    "lane_change_resolve_conflicts",
    "lane_change_apply",
    "validation_and_final_checks",
    "longitudinal_phase",
    "full_step_reference_blackbox",
]

@dataclass(frozen=True)
class BenchCase:
    name: str
    num_lanes: int
    road_length: int
    density: float
    p_lane_change: float
    steps: int
    seed: int


def get_cases(preset: str, seed: int, steps: int | None) -> list[BenchCase]:
    smoke = [
        BenchCase("smoke_low_density", 2, 100, 0.10, 0.5, 10, seed),
        BenchCase("smoke_dense", 3, 100, 0.50, 0.5, 10, seed + 1),
    ]
    standard = [
        BenchCase("small_sparse", 2, 200, 0.05, 0.5, 100, seed),
        BenchCase("medium_moderate", 3, 1000, 0.20, 0.5, 100, seed + 1),
        BenchCase("medium_dense", 3, 1000, 0.50, 0.5, 100, seed + 2),
        BenchCase("wide_moderate", 5, 1000, 0.20, 0.5, 100, seed + 3),
        BenchCase("large_moderate", 4, 5000, 0.20, 0.5, 100, seed + 4),
        BenchCase("no_lane_change_attempts", 3, 1000, 0.20, 0.0, 100, seed + 5),
        BenchCase("always_lane_change_when_eligible", 3, 1000, 0.20, 1.0, 100, seed + 6),
    ]
    out = smoke if preset == "smoke" else standard
    if steps is not None:
        out = [replace(c, steps=steps) for c in out]
    return out


def phased_step(state, params, topology, rng):
    phase_ns = {k: 0 for k in PHASE_KEYS}
    t0 = perf_counter_ns(); validate_state(state, params); phase_ns["validate_state_input"] += perf_counter_ns() - t0
    t0 = perf_counter_ns(); occupancy = build_occupancy(state, params); phase_ns["build_occupancy_pre_lane_change"] += perf_counter_ns() - t0
    t0 = perf_counter_ns(); lane_order, lane_counts, lane_rank = build_lane_order(occupancy, n_vehicles=state.n_vehicles); phase_ns["build_lane_order_pre_lane_change"] += perf_counter_ns() - t0
    t0 = perf_counter_ns(); front_id, _, front_gap, _ = compute_neighbors(state, lane_order, lane_counts, lane_rank, topology); phase_ns["compute_neighbors_pre_lane_change"] += perf_counter_ns() - t0
    t0 = perf_counter_ns(); proposals = collect_lane_change_proposals_kernel(state.lane, state.pos, state.vel, state.alive, state.controlled, occupancy, lane_order, lane_counts, front_id, front_gap, num_lanes=params.num_lanes, road_length=params.road_length, vmax_default=params.vmax_default, vmax_controlled=params.vmax_controlled, p_lane_change=params.p_lane_change, rng=rng); phase_ns["lane_change_collect_proposals"] += perf_counter_ns() - t0
    t0 = perf_counter_ns(); accepted = resolve_lane_change_conflicts_kernel(proposals, rng); phase_ns["lane_change_resolve_conflicts"] += perf_counter_ns() - t0
    t0 = perf_counter_ns(); new_lane, new_changed_lane, new_last_lane_delta = apply_lane_changes_kernel(state.lane, state.changed_lane, state.last_lane_delta, accepted); phase_ns["lane_change_apply"] += perf_counter_ns() - t0

    after_lane = state.copy(); after_lane.lane = new_lane; after_lane.changed_lane = new_changed_lane; after_lane.last_lane_delta = new_last_lane_delta
    t0 = perf_counter_ns(); validate_state(after_lane, params); build_occupancy(after_lane, params); phase_ns["validation_and_final_checks"] += perf_counter_ns() - t0
    changed = after_lane.changed_lane.copy(); delta = after_lane.last_lane_delta.copy()
    t0 = perf_counter_ns(); out = step_longitudinal_reference(after_lane, params, topology, rng); phase_ns["longitudinal_phase"] += perf_counter_ns() - t0
    out = out.copy(); out.changed_lane = changed; out.last_lane_delta = delta
    t0 = perf_counter_ns(); validate_state(out, params); build_occupancy(out, params); phase_ns["validation_and_final_checks"] += perf_counter_ns() - t0
    return out, phase_ns


def assert_phased_equivalent(case: BenchCase):
    params = SimulationParams(num_lanes=case.num_lanes, road_length=case.road_length, p_lane_change=case.p_lane_change)
    topology = RingTopology(num_lanes=case.num_lanes, length=case.road_length)
    state = make_uniform_random_state(num_lanes=case.num_lanes, road_length=case.road_length, density=case.density, seed=case.seed)
    r1 = np.random.default_rng(case.seed + 12345)
    r2 = np.random.default_rng(case.seed + 12345)
    a = step_reference(state.copy(), params, topology, r1)
    b, _ = phased_step(state.copy(), params, topology, r2)
    for f in ("lane", "pos", "vel", "alive", "controlled", "changed_lane", "last_lane_delta"):
        if not np.array_equal(getattr(a, f), getattr(b, f)):
            raise AssertionError(f"phased mismatch field={f} case={case.name}")
    if float(r1.random()) != float(r2.random()):
        raise AssertionError(f"rng mismatch after phased step for case={case.name}")


def run_case(case: BenchCase, repeats: int, warmups: int):
    params = SimulationParams(num_lanes=case.num_lanes, road_length=case.road_length, p_lane_change=case.p_lane_change)
    topology = RingTopology(num_lanes=case.num_lanes, length=case.road_length)
    base_state = make_uniform_random_state(num_lanes=case.num_lanes, road_length=case.road_length, density=case.density, seed=case.seed)

    def do_rollout(count_steps: int, *, collect_phase: bool):
        s = base_state.copy()
        rng = np.random.default_rng(case.seed + 999)
        totals = {k: 0 for k in PHASE_KEYS}
        t_start = perf_counter_ns()
        for _ in range(count_steps):
            if collect_phase:
                s, phase_ns = phased_step(s, params, topology, rng)
                for k, v in phase_ns.items(): totals[k] += v
            else:
                t0 = perf_counter_ns(); s = step_reference(s, params, topology, rng); totals["full_step_reference_blackbox"] += perf_counter_ns() - t0
        total_ns = perf_counter_ns() - t_start
        return total_ns, totals

    for _ in range(warmups):
        do_rollout(case.steps, collect_phase=True)
        do_rollout(case.steps, collect_phase=False)

    repeats_data = []
    gc_enabled = gc.isenabled(); gc.disable()
    try:
        for i in range(repeats):
            total_ns, phase_totals = do_rollout(case.steps, collect_phase=True)
            roll_ns, full_totals = do_rollout(case.steps, collect_phase=False)
            phase_totals["full_step_reference_blackbox"] = full_totals["full_step_reference_blackbox"]
            repeats_data.append({"repeat_index": i, "total_ns": total_ns, "rollout_reference_blackbox_ns": roll_ns, "per_step_ns": total_ns / case.steps, "phase_ns": phase_totals})
    finally:
        if gc_enabled: gc.enable()

    per_step = [r["per_step_ns"] for r in repeats_data]
    phase_mean = {k: statistics.fmean([r["phase_ns"][k] / case.steps for r in repeats_data]) for k in PHASE_KEYS}
    phased_total = sum(phase_mean[k] for k in PHASE_KEYS if k != "full_step_reference_blackbox")
    pct = {k: (100.0 * v / phased_total if phased_total > 0 else 0.0) for k, v in phase_mean.items() if k != "full_step_reference_blackbox"}
    top = sorted(pct.items(), key=lambda kv: kv[1], reverse=True)[:3]
    summary = {"per_step_ns_mean": statistics.fmean(per_step), "per_step_ns_median": statistics.median(per_step), "phase_per_step_ns_mean": phase_mean, "phase_percent_mean": pct, "top_phases_by_time": top}
    return {**asdict(case), "n_vehicles": int(base_state.n_vehicles), "repeats": repeats_data, "summary": summary}


def recommend(preset: str, cases: list[dict]):
    agg = {}
    for c in cases:
        for k, p in c["summary"]["phase_percent_mean"].items(): agg[k] = agg.get(k, 0.0) + p
    if not agg:
        return {"top_bottlenecks": [], "suggested_next_task": "No data."}
    top = sorted(((k, v / len(cases)) for k, v in agg.items()), key=lambda kv: kv[1], reverse=True)
    names = [t[0] for t in top[:3]]
    t0, p0 = top[0]
    if preset == "smoke":
        msg = "Run standard preset before final optimization choice; smoke data is local and limited."
    elif t0 in {"build_occupancy_pre_lane_change", "build_lane_order_pre_lane_change", "compute_neighbors_pre_lane_change"}:
        msg = "Task 17 — optimize indexing/neighbor computation; indexing dominates phased full-step cost."
    elif t0 == "lane_change_collect_proposals":
        msg = "Task 17 — implement optional Numba lane-change proposal/conflict kernels; proposal collection dominates."
    elif t0 == "longitudinal_phase":
        msg = "Task 17 — implement optional Numba longitudinal kernel; longitudinal phase dominates."
    elif t0 == "validation_and_final_checks":
        msg = "Task 17 — design optimized full-step backend separating validation-heavy reference checks from fast path."
    else:
        msg = "Task 17 — optimize the measured dominant full-step bottleneck identified by Task 16."
    return {"top_bottlenecks": names, "suggested_next_task": msg, "top_phase": t0, "top_phase_percent_mean": p0}


def format_markdown(data: dict) -> str:
    lines = ["# SNFS full-step phase benchmark", "", f"Timestamp: {data['created_at_utc']}", "", "## Environment", "", f"- Python: {data['python']}", f"- Platform: {data['platform']}", f"- NumPy: {data['numpy_version']}", "", "## Benchmark config", "", f"- Preset: {data['preset']}", f"- Seed: {data['seed']}", f"- Warmups: {data['warmups']}", f"- Repeats: {data['repeats']}", f"- Default steps: {data['default_steps']}", "", "## Case summary", "", "| case | lanes | road_length | density | vehicles | p_lane_change | steps | mean step ms | median step ms | top bottleneck |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|" ]
    for c in data["cases"]:
        top = c["summary"]["top_phases_by_time"][0][0] if c["summary"]["top_phases_by_time"] else "n/a"
        lines.append(f"| {c['name']} | {c['num_lanes']} | {c['road_length']} | {c['density']:.2f} | {c['n_vehicles']} | {c['p_lane_change']:.2f} | {c['steps']} | {c['summary']['per_step_ns_mean']/1e6:.4f} | {c['summary']['per_step_ns_median']/1e6:.4f} | {top} |")
    for c in data["cases"]:
        lines += ["", f"## Phase breakdown — {c['name']}", "", "| phase | mean ns/step | mean ms/step | percent of phased step |", "|---|---:|---:|---:|"]
        for k in PHASE_KEYS:
            if k == "full_step_reference_blackbox":
                continue
            ns = c["summary"]["phase_per_step_ns_mean"][k]
            pc = c["summary"]["phase_percent_mean"][k]
            lines.append(f"| {k} | {ns:.1f} | {ns/1e6:.6f} | {pc:.2f}% |")
    lines += ["", "## Recommended next task", "", data["recommendation"]["suggested_next_task"], "", "_These benchmark results are local-environment measurements._"]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["smoke", "standard"], default="smoke")
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-md", type=Path, default=None)
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--warmups", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cases", type=str, default="")
    args = p.parse_args()

    defaults = {"smoke": (2, 1, 10), "standard": (5, 2, 100)}
    d_rep, d_warm, d_steps = defaults[args.preset]
    repeats = args.repeats if args.repeats is not None else d_rep
    warmups = args.warmups if args.warmups is not None else d_warm
    cases = get_cases(args.preset, args.seed, args.steps if args.steps is not None else d_steps)
    if args.cases:
        wanted = {x.strip() for x in args.cases.split(",") if x.strip()}
        cases = [c for c in cases if c.name in wanted]

    for c in cases[: min(2, len(cases))]:
        assert_phased_equivalent(c)

    case_results = [run_case(c, repeats=repeats, warmups=warmups) for c in cases]
    data = {
        "benchmark": "snfs_full_step_phase_costs",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "preset": args.preset,
        "seed": args.seed,
        "warmups": warmups,
        "repeats": repeats,
        "default_steps": d_steps,
        "cases": case_results,
    }
    data["recommendation"] = recommend(args.preset, case_results)

    print(f"Benchmark={data['benchmark']} preset={args.preset} cases={len(case_results)} repeats={repeats} warmups={warmups}")
    for c in case_results:
        top = c["summary"]["top_phases_by_time"][0]
        print(f"- {c['name']}: mean_step_ms={c['summary']['per_step_ns_mean']/1e6:.4f}, top_phase={top[0]} ({top[1]:.2f}%)")
    print(f"Recommendation: {data['recommendation']['suggested_next_task']}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(format_markdown(data), encoding="utf-8")


if __name__ == "__main__":
    main()
