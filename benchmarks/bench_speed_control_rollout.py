from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from snfs_traffic.backends import optimized_backend_available

STATE_FIELDS = ("lane", "pos", "vel", "alive", "controlled", "length")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark speed-control multi-agent rollout throughput.")
    parser.add_argument("--road-length", type=int, default=1000)
    parser.add_argument("--num-lanes", type=int, default=4)
    parser.add_argument("--density", type=float, default=0.20)
    parser.add_argument("--num-controlled", type=int, default=50)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--action-mode", choices=("cycle", "keep", "accelerate", "brake", "mixed"), default="cycle")
    parser.add_argument("--out-dir", default="benchmark_results")
    return parser.parse_args()


def _actual_num_controlled(args: argparse.Namespace) -> int:
    num_vehicles = int(np.floor(args.road_length * args.num_lanes * args.density))
    if num_vehicles < 1:
        raise ValueError("scenario must contain at least one vehicle; increase density or road capacity")
    return max(1, min(int(args.num_controlled), num_vehicles))


def _vehicle_id(agent_id: str) -> int:
    return int(agent_id.rsplit("_", 1)[1])


def _speed_action(vehicle_id: int, step: int, mode: str) -> int:
    if mode == "keep":
        return 1
    if mode == "accelerate":
        return 2
    if mode == "brake":
        return 0
    if mode == "mixed":
        return (2 * step + vehicle_id) % 3
    return (step + vehicle_id) % 3


def _actions(env, step: int, mode: str) -> dict[str, np.ndarray]:
    return {
        agent_id: np.asarray([0, _speed_action(_vehicle_id(agent_id), step, mode)], dtype=np.int64)
        for agent_id in env.agents
    }


def _make_env(args: argparse.Namespace, backend: str, num_controlled: int):
    from snfs_traffic.rl.episode import EpisodeConfig
    from snfs_traffic.rl.speed_control_multiagent_env import SnfsTrafficSpeedControlMultiAgentEnv

    return SnfsTrafficSpeedControlMultiAgentEnv(
        num_lanes=args.num_lanes,
        road_length=args.road_length,
        density=args.density,
        num_controlled=num_controlled,
        backend=backend,
        seed=args.seed,
        scenario_seed=args.seed,
        episode_config=EpisodeConfig(max_steps=args.warmup_steps + args.steps + 5),
    )


def _state_snapshot(env) -> dict[str, np.ndarray]:
    state = env._sim.state
    return {field: getattr(state, field).copy() for field in STATE_FIELDS}


def _run_rollout(args: argparse.Namespace, backend: str, num_controlled: int, *, timed: bool) -> dict[str, object]:
    env = _make_env(args, backend, num_controlled)
    env.reset(seed=args.seed)
    for step in range(args.warmup_steps):
        env.step(_actions(env, step, args.action_mode))

    cumulative_rewards = {agent_id: 0.0 for agent_id in env.agents}
    step_ms: list[float] = []
    agent_steps = 0
    start = perf_counter_ns()
    for offset in range(args.steps):
        step = args.warmup_steps + offset
        actions = _actions(env, step, args.action_mode)
        t0 = perf_counter_ns()
        _obs, rewards, terminateds, truncateds, _infos = env.step(actions)
        elapsed_step = perf_counter_ns() - t0
        step_ms.append(elapsed_step / 1_000_000.0)
        agent_steps += len(actions)
        for agent_id, reward in rewards.items():
            cumulative_rewards[agent_id] = cumulative_rewards.get(agent_id, 0.0) + float(reward)
        if terminateds["__all__"] or truncateds["__all__"]:
            break
    elapsed = (perf_counter_ns() - start) / 1_000_000_000.0
    completed_steps = len(step_ms)
    result = {
        "backend": backend,
        "reported_backend": env.backend_name,
        "road_length": args.road_length,
        "num_lanes": args.num_lanes,
        "density": args.density,
        "num_vehicles": int(env._sim.state.n_vehicles),
        "num_controlled": int(num_controlled),
        "steps": int(completed_steps),
        "warmup_steps": int(args.warmup_steps),
        "elapsed_seconds": float(elapsed),
        "steps_per_second": float(completed_steps / elapsed) if elapsed > 0 else 0.0,
        "agent_steps_per_second": float(agent_steps / elapsed) if elapsed > 0 else 0.0,
        "mean_step_ms": float(statistics.fmean(step_ms)) if step_ms else 0.0,
        "p50_step_ms": float(statistics.median(step_ms)) if step_ms else 0.0,
        "p95_step_ms": float(np.percentile(np.asarray(step_ms, dtype=np.float64), 95)) if step_ms else 0.0,
        "cumulative_rewards": cumulative_rewards,
        "final_state": _state_snapshot(env),
    }
    if not timed:
        result["step_ms"] = step_ms
    return result


def _assert_equivalent(reference: dict[str, object], optimized: dict[str, object]) -> None:
    ref_state = reference["final_state"]
    opt_state = optimized["final_state"]
    for field in STATE_FIELDS:
        np.testing.assert_array_equal(ref_state[field], opt_state[field], err_msg=field)
    assert reference["cumulative_rewards"] == optimized["cumulative_rewards"]


def _jsonable(result: dict[str, object]) -> dict[str, object]:
    out = dict(result)
    out.pop("final_state", None)
    return out


def _print_table(results: list[dict[str, object]], speedup: float | None) -> None:
    headers = ("backend", "vehicles", "controlled", "steps", "elapsed_s", "steps/s", "agent_steps/s", "mean_ms", "p50_ms", "p95_ms")
    print(" | ".join(headers))
    print(" | ".join("-" * len(h) for h in headers))
    for row in results:
        print(
            f"{row['backend']} | {row['num_vehicles']} | {row['num_controlled']} | {row['steps']} | "
            f"{row['elapsed_seconds']:.6f} | {row['steps_per_second']:.3f} | "
            f"{row['agent_steps_per_second']:.3f} | {row['mean_step_ms']:.3f} | "
            f"{row['p50_step_ms']:.3f} | {row['p95_step_ms']:.3f}"
        )
    if speedup is None:
        print("optimized speedup: unavailable (optimized backend dependencies are not installed)")
    else:
        print(f"optimized speedup: {speedup:.3f}x")


def main() -> int:
    args = _parse_args()
    num_controlled = _actual_num_controlled(args)
    if num_controlled != args.num_controlled:
        print(f"requested num_controlled={args.num_controlled}; using available num_controlled={num_controlled}")

    results: list[dict[str, object]] = []
    reference = _run_rollout(args, "reference", num_controlled, timed=True)
    results.append(reference)

    speedup = None
    optimized_available = optimized_backend_available()
    if optimized_available:
        ref_check = _run_rollout(args, "reference", num_controlled, timed=False)
        opt_check = _run_rollout(args, "optimized", num_controlled, timed=False)
        _assert_equivalent(ref_check, opt_check)
        optimized = _run_rollout(args, "optimized", num_controlled, timed=True)
        results.append(optimized)
        speedup = float(optimized["steps_per_second"] / reference["steps_per_second"]) if reference["steps_per_second"] else 0.0
    else:
        print("optimized backend unavailable; timing reference only and recording optimized skip")

    _print_table(results, speedup)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"speed_control_rollout_{timestamp}.json"
    payload = {
        "scenario": vars(args) | {"actual_num_controlled": num_controlled},
        "optimized_available": optimized_available,
        "equivalence_checked": optimized_available,
        "speedup": speedup,
        "results": [_jsonable(row) for row in results],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote JSON report: {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ImportError as exc:
        print(f"speed-control benchmark requires optional RL dependencies: {exc}", file=sys.stderr)
        raise SystemExit(2)
