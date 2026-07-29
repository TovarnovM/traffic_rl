#!/usr/bin/env python3
"""Select a Graph or non-graph PPO checkpoint and run held-out evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


RESEARCH_DIR = Path(__file__).resolve().parent
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))
import evaluate_pv_graph_ppo as graph_eval
import evaluate_pv_nongraph_ppo as nongraph_eval


def _parse_float_list(raw: str, *, name: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"{name} must not be empty")
    return list(dict.fromkeys(values))


def _parse_int_set(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    values = {int(part.strip()) for part in raw.split(",") if part.strip()}
    if not values or any(value < 1 for value in values):
        raise ValueError("--checkpoint-iterations must contain positive integers")
    return values


def _checkpoint_iteration(path: Path) -> int | None:
    match = re.search(r"_iter_(\d+)", path.name)
    return int(match.group(1)) if match else None


def _is_final_checkpoint(path: Path) -> bool:
    return path.name.startswith("final_") or "final" in path.name.lower()


def _discover_checkpoints(
    root: Path,
    *,
    iterations: set[int] | None,
    include_final: bool,
) -> list[Path]:
    resolved_root = root.expanduser().resolve()
    if not resolved_root.exists():
        raise FileNotFoundError(f"checkpoint root does not exist: {resolved_root}")
    candidates = {
        marker.parent
        for pattern in ("algorithm_state.pkl", "rllib_checkpoint.json", ".is_checkpoint")
        for marker in resolved_root.rglob(pattern)
        if "policies" not in marker.relative_to(resolved_root).parts
    }
    selected: list[Path] = []
    for candidate in candidates:
        iteration = _checkpoint_iteration(candidate)
        is_final = _is_final_checkpoint(candidate)
        if iterations is None:
            if include_final or not is_final:
                selected.append(candidate)
        elif iteration in iterations or (include_final and is_final):
            selected.append(candidate)
    selected = sorted(
        set(selected),
        key=lambda path: (
            _checkpoint_iteration(path) is None,
            _checkpoint_iteration(path) or 0,
            path.name,
        ),
    )
    if not selected:
        requested = sorted(iterations) if iterations is not None else "all"
        raise FileNotFoundError(
            f"no requested checkpoints below {resolved_root}; iterations={requested} "
            f"include_final={include_final}"
        )
    return selected


def _checkpoint_tag(path: Path, index: int) -> str:
    iteration = _checkpoint_iteration(path)
    if iteration is not None:
        return f"iter_{iteration:06d}"
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", path.name).strip("_").lower()
    return cleaned or f"checkpoint_{index:02d}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _selection_result(
    *,
    checkpoint: Path,
    tag: str,
    summary_path: Path,
    paired_summary_path: Path,
    learned_mode: str,
    model_label: str,
) -> dict[str, Any]:
    policy_rows = [
        row for row in _read_csv(summary_path)
        if row.get("mode") == learned_mode
    ]
    if len(policy_rows) != 1:
        raise ValueError(
            f"expected one {learned_mode} row in {summary_path}, "
            f"found {len(policy_rows)}"
        )
    baseline_rows = [
        row for row in _read_csv(paired_summary_path)
        if row.get("comparison_mode") == nongraph_eval.BASELINE2
    ]
    if len(baseline_rows) != 1:
        raise ValueError(
            f"expected one Baseline2 paired row in {paired_summary_path}, "
            f"found {len(baseline_rows)}"
        )
    policy = policy_rows[0]
    paired = baseline_rows[0]
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_tag": tag,
        "checkpoint_iteration": _checkpoint_iteration(checkpoint),
        "model_preset": (
            policy.get("model_preset")
            or paired.get("model_preset")
            or model_label
        ),
        "runs": int(policy["runs"]),
        "pv_speed_mean": float(policy["mean_speed_priority_mean"]),
        "pv_speed_std": float(policy["mean_speed_priority_std"]),
        "pv_speed_ci95_low": float(policy["mean_speed_priority_ci95_low"]),
        "pv_speed_ci95_high": float(policy["mean_speed_priority_ci95_high"]),
        "baseline2_delta_mean": float(paired["pv_speed_delta_mean"]),
        "baseline2_delta_ci_low": float(paired["pv_speed_delta_ci_low"]),
        "baseline2_delta_ci_high": float(paired["pv_speed_delta_ci_high"]),
        "baseline2_win_rate": float(paired["pv_speed_win_rate"]),
        "summary_csv": str(summary_path),
        "paired_summary_csv": str(paired_summary_path),
    }


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write an empty leaderboard")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run(command: list[str], *, dry_run: bool) -> None:
    print("command=" + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def _evaluation_command(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    densities: str,
    av_fraction: float,
    runs: int,
    seed: int,
    out_dir: Path,
    prefix: str,
) -> list[str]:
    evaluator = (
        "evaluate_pv_graph_ppo.py"
        if args.policy_kind == "graph"
        else "evaluate_pv_nongraph_ppo.py"
    )
    command = [
        sys.executable,
        str(RESEARCH_DIR / evaluator),
        "--checkpoint",
        str(checkpoint),
        "--densities",
        densities,
        "--av-fraction",
        str(av_fraction),
        "--runs",
        str(runs),
        "--seed",
        str(seed),
        "--workers",
        str(args.workers),
        "--chunksize",
        str(args.chunksize),
        "--torch-threads",
        str(args.torch_threads),
        "--measure-steps",
        str(args.measure_steps),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--out-dir",
        str(out_dir),
        "--prefix",
        prefix,
        "--plot" if args.selection_plots or "selection" not in out_dir.parts else "--no-plot",
    ]
    if args.warmup_steps is not None:
        command.extend(("--warmup-steps", str(args.warmup_steps)))
    if args.backend is not None:
        command.extend(("--backend", args.backend))
    if args.yield_cooldown_steps is not None:
        command.extend(
            ("--yield-cooldown-steps", str(args.yield_cooldown_steps))
        )
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic checkpoint selection followed by a held-out density "
            "and AV-fraction matrix"
        )
    )
    parser.add_argument(
        "--policy-kind",
        choices=("graph", "nongraph"),
        default="nongraph",
        help="select the existing Edge Graph-PPO or a new non-graph policy",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoints", type=Path, nargs="+")
    source.add_argument("--checkpoint-root", type=Path)
    parser.add_argument(
        "--checkpoint-iterations",
        help="comma-separated periodic iterations when using --checkpoint-root",
    )
    parser.add_argument(
        "--include-final", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--selection-density", type=float, default=0.30)
    parser.add_argument("--selection-av-fraction", type=float, default=1.00)
    parser.add_argument("--selection-runs", type=int, default=20)
    parser.add_argument("--selection-seed", type=int, default=300_001)
    parser.add_argument("--full-densities", default="0.10,0.20,0.30,0.40")
    parser.add_argument("--full-av-fractions", default="0.90,0.95,1.00")
    parser.add_argument("--full-runs", type=int, default=20)
    parser.add_argument("--full-seed", type=int, default=400_001)
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--yield-cooldown-steps", type=int)
    parser.add_argument(
        "--backend", choices=("auto", "reference", "optimized")
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--selection-plots", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selection_runs < 1 or args.full_runs < 1:
        raise SystemExit("selection-runs and full-runs must be >= 1")
    if args.workers < 1 or args.chunksize < 1 or args.torch_threads < 1:
        raise SystemExit("workers/chunksize/torch-threads must be >= 1")
    if args.selection_seed == args.full_seed and not args.skip_full:
        raise SystemExit(
            "selection-seed and full-seed must differ to avoid checkpoint-selection bias"
        )
    try:
        iterations = _parse_int_set(args.checkpoint_iterations)
        full_av_fractions = _parse_float_list(
            args.full_av_fractions, name="--full-av-fractions"
        )
        graph_eval._parse_densities(args.full_densities)
        if args.checkpoints is not None:
            checkpoints = [
                graph_eval._resolve_checkpoint(path) for path in args.checkpoints
            ]
        else:
            checkpoints = _discover_checkpoints(
                args.checkpoint_root,
                iterations=iterations,
                include_final=args.include_final,
            )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    args.out_dir.mkdir(parents=True, exist_ok=True)
    learned_mode = (
        graph_eval.GRAPH_PPO
        if args.policy_kind == "graph"
        else nongraph_eval.NON_GRAPH_PPO
    )
    model_label = (
        "edge-graph-ppo" if args.policy_kind == "graph" else "non-graph-ppo"
    )
    leaderboard: list[dict[str, Any]] = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        tag = _checkpoint_tag(checkpoint, index)
        output = args.out_dir / "selection" / tag
        prefix = f"selection_{tag}"
        command = _evaluation_command(
            args,
            checkpoint=checkpoint,
            densities=f"{args.selection_density:.12g}",
            av_fraction=args.selection_av_fraction,
            runs=args.selection_runs,
            seed=args.selection_seed,
            out_dir=output,
            prefix=prefix,
        )
        _run(command, dry_run=args.dry_run)
        if not args.dry_run:
            leaderboard.append(
                _selection_result(
                    checkpoint=checkpoint,
                    tag=tag,
                    summary_path=output / f"{prefix}_summary.csv",
                    paired_summary_path=output / f"{prefix}_paired_summary.csv",
                    learned_mode=learned_mode,
                    model_label=model_label,
                )
            )
    if args.dry_run:
        print("dry_run=selection commands only; winner-dependent full matrix skipped")
        return 0

    leaderboard.sort(
        key=lambda row: (
            float(row["pv_speed_mean"]),
            float(row["baseline2_delta_ci_low"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(leaderboard, start=1):
        row["rank"] = rank
        row["selected"] = rank == 1
    leaderboard_path = args.out_dir / "checkpoint_leaderboard.csv"
    _write_csv(leaderboard_path, leaderboard)
    winner = leaderboard[0]
    winner_path = Path(str(winner["checkpoint"]))
    result_json = args.out_dir / "checkpoint_selection.json"
    with result_json.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "selection_density": args.selection_density,
                "selection_av_fraction": args.selection_av_fraction,
                "selection_runs": args.selection_runs,
                "selection_seed": args.selection_seed,
                "policy_kind": args.policy_kind,
                "winner": winner,
                "leaderboard": leaderboard,
            },
            stream,
            indent=2,
            sort_keys=True,
        )
    print(
        f"best_checkpoint={winner_path} pv_speed={winner['pv_speed_mean']:.6f}",
        flush=True,
    )

    if not args.skip_full:
        for av_fraction in full_av_fractions:
            tag = f"av_{av_fraction:.2f}".replace(".", "p")
            output = args.out_dir / "full" / tag
            prefix = f"full_{tag}"
            command = _evaluation_command(
                args,
                checkpoint=winner_path,
                densities=args.full_densities,
                av_fraction=av_fraction,
                runs=args.full_runs,
                seed=args.full_seed,
                out_dir=output,
                prefix=prefix,
            )
            _run(command, dry_run=False)
    print(f"leaderboard_csv={leaderboard_path}", flush=True)
    print(f"selection_json={result_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
