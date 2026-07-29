#!/usr/bin/env python3
"""Compare learned PV policies from paired raw evaluation CSV files."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from snfs_traffic.metrics import paired_bootstrap_interval


RESEARCH_DIR = str(Path(__file__).resolve().parent)
if RESEARCH_DIR not in sys.path:
    sys.path.insert(0, RESEARCH_DIR)
import evaluate_pv_graph_ppo as graph_eval


CONTROL_MODES = {
    "baseline1_hdv",
    "baseline2_matched",
    "graph_noop",
    "policy_noop",
    "left_if_safe",
}
PAIR_KEY_FIELDS = (
    "requested_density",
    "av_fraction",
    "run_index",
    "priority_vehicle_id",
    "scenario_seed",
    "warmup_rng_seed",
    "assignment_seed",
    "av_selection_seed",
    "measurement_rng_seed",
)


def _parse_input(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--input must use LABEL=/path/to/runs.csv")
    label, path_text = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("architecture label must not be empty")
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"evaluation runs CSV does not exist: {path}")
    return label, path


def _read_policy_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    learned_modes = {
        str(row.get("mode", "")) for row in rows
        if str(row.get("mode", "")) not in CONTROL_MODES
    }
    if len(learned_modes) != 1:
        raise ValueError(
            f"expected exactly one learned policy mode in {path}, "
            f"found {sorted(learned_modes)}"
        )
    learned_mode = next(iter(learned_modes))
    selected = [row for row in rows if row.get("mode") == learned_mode]
    if not selected:
        raise ValueError(f"no learned-policy rows in {path}")
    return selected


def _pair_key(row: dict[str, str]) -> tuple[Any, ...]:
    try:
        return (
            float(row["requested_density"]),
            float(row["av_fraction"]),
            int(row["run_index"]),
            int(row["priority_vehicle_id"]),
            int(row["scenario_seed"]),
            int(row["warmup_rng_seed"]),
            int(row["assignment_seed"]),
            int(row["av_selection_seed"]),
            int(row["measurement_rng_seed"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"evaluation row lacks a valid paired key: required={PAIR_KEY_FIELDS}"
        ) from exc


def _load_inputs(raw_inputs: list[str]) -> tuple[list[str], dict[str, dict]]:
    labels: list[str] = []
    indexed: dict[str, dict[tuple[Any, ...], dict[str, str]]] = {}
    for raw in raw_inputs:
        label, path = _parse_input(raw)
        if label not in indexed:
            labels.append(label)
            indexed[label] = {}
        for row in _read_policy_rows(path):
            key = _pair_key(row)
            if key in indexed[label]:
                raise ValueError(f"duplicate paired row for {label}: key={key}")
            indexed[label][key] = row
    if len(labels) < 2:
        raise ValueError("at least two distinct architecture labels are required")
    reference_keys = set(indexed[labels[0]])
    for label in labels[1:]:
        keys = set(indexed[label])
        if keys != reference_keys:
            missing = len(reference_keys - keys)
            extra = len(keys - reference_keys)
            raise ValueError(
                f"paired seeds do not align for {label}: missing={missing}, extra={extra}"
            )
    return labels, indexed


def _architecture_summary(
    labels: list[str], indexed: dict[str, dict]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for label in labels:
        groups: dict[tuple[float, float], list[dict[str, str]]] = {}
        for key, row in indexed[label].items():
            groups.setdefault((float(key[0]), float(key[1])), []).append(row)
        for (density, av_fraction), rows in sorted(groups.items()):
            pv_speeds = [float(row["mean_speed_priority"]) for row in rows]
            background_flows = [float(row["flow_background"]) for row in rows]
            pv_std = graph_eval._sample_std(pv_speeds)
            flow_std = graph_eval._sample_std(background_flows)
            pv_sem = pv_std / math.sqrt(len(rows))
            flow_sem = flow_std / math.sqrt(len(rows))
            output.append(
                {
                    "architecture": label,
                    "requested_density": density,
                    "av_fraction": av_fraction,
                    "runs": len(rows),
                    "pv_speed_mean": graph_eval._mean(pv_speeds),
                    "pv_speed_std": pv_std,
                    "pv_speed_sem": pv_sem,
                    "pv_speed_ci95_low": graph_eval._mean(pv_speeds) - 1.96 * pv_sem,
                    "pv_speed_ci95_high": graph_eval._mean(pv_speeds) + 1.96 * pv_sem,
                    "background_flow_mean": graph_eval._mean(background_flows),
                    "background_flow_std": flow_std,
                    "background_flow_sem": flow_sem,
                    "background_flow_ci95_low": graph_eval._mean(background_flows)
                    - 1.96 * flow_sem,
                    "background_flow_ci95_high": graph_eval._mean(background_flows)
                    + 1.96 * flow_sem,
                }
            )
    return output


def _pairwise_rows(
    labels: list[str], indexed: dict[str, dict]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted(indexed[labels[0]])
    for architecture_a, architecture_b in itertools.combinations(labels, 2):
        for key in keys:
            first = indexed[architecture_a][key]
            second = indexed[architecture_b][key]
            first_flow = float(first["flow_background"])
            second_flow = float(second["flow_background"])
            output.append(
                {
                    "architecture_a": architecture_a,
                    "architecture_b": architecture_b,
                    "requested_density": float(key[0]),
                    "av_fraction": float(key[1]),
                    "run_index": int(key[2]),
                    "priority_vehicle_id": int(key[3]),
                    "scenario_seed": int(key[4]),
                    "warmup_rng_seed": int(key[5]),
                    "assignment_seed": int(key[6]),
                    "av_selection_seed": int(key[7]),
                    "measurement_rng_seed": int(key[8]),
                    "pv_speed_a": float(first["mean_speed_priority"]),
                    "pv_speed_b": float(second["mean_speed_priority"]),
                    "pv_speed_delta_a_minus_b": float(
                        first["mean_speed_priority"]
                    )
                    - float(second["mean_speed_priority"]),
                    "background_flow_a": first_flow,
                    "background_flow_b": second_flow,
                    "background_flow_delta_a_minus_b": first_flow - second_flow,
                    "background_flow_ratio_a_over_b": (
                        first_flow / second_flow if second_flow > 0.0 else None
                    ),
                    "lane_change_rate_delta_a_minus_b": float(
                        first["lane_change_rate_all"]
                    )
                    - float(second["lane_change_rate_all"]),
                }
            )
    return output


def _pairwise_summary(
    rows: list[dict[str, Any]],
    *,
    confidence: float,
    bootstrap_samples: int,
    seed: int,
    equivalence_margin: float | None,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["architecture_a"]),
            str(row["architecture_b"]),
            float(row["requested_density"]),
            float(row["av_fraction"]),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for group_index, (key, group) in enumerate(sorted(groups.items())):
        architecture_a, architecture_b, density, av_fraction = key
        pv_delta = np.asarray(
            [float(row["pv_speed_delta_a_minus_b"]) for row in group],
            dtype=np.float64,
        )
        flow_delta = np.asarray(
            [float(row["background_flow_delta_a_minus_b"]) for row in group],
            dtype=np.float64,
        )
        pv_low, pv_high = paired_bootstrap_interval(
            pv_delta,
            confidence=confidence,
            samples=bootstrap_samples,
            seed=seed + group_index * 10,
        )
        flow_low, flow_high = paired_bootstrap_interval(
            flow_delta,
            confidence=confidence,
            samples=bootstrap_samples,
            seed=seed + group_index * 10 + 1,
        )
        result: dict[str, Any] = {
            "architecture_a": architecture_a,
            "architecture_b": architecture_b,
            "requested_density": density,
            "av_fraction": av_fraction,
            "runs": len(group),
            "pv_speed_delta_a_minus_b_mean": float(np.mean(pv_delta)),
            "pv_speed_delta_a_minus_b_std": (
                float(np.std(pv_delta, ddof=1)) if pv_delta.size > 1 else 0.0
            ),
            "pv_speed_delta_ci_low": pv_low,
            "pv_speed_delta_ci_high": pv_high,
            "pv_speed_a_win_rate": float(np.mean(pv_delta > 0.0)),
            "background_flow_delta_a_minus_b_mean": float(np.mean(flow_delta)),
            "background_flow_delta_ci_low": flow_low,
            "background_flow_delta_ci_high": flow_high,
        }
        if equivalence_margin is not None:
            result["equivalence_margin"] = equivalence_margin
            result["pv_speed_equivalent_within_margin"] = bool(
                pv_low >= -equivalence_margin and pv_high <= equivalence_margin
            )
        output.append(result)
    return output


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(
    path: Path,
    architecture_summary: list[dict[str, Any]],
    pairwise_summary: list[dict[str, Any]],
    *,
    labels: list[str],
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib is unavailable; comparison plot was skipped")
        return False

    av_fractions = sorted(
        {float(row["av_fraction"]) for row in architecture_summary}
    )
    figure, axes = plt.subplots(
        2,
        len(av_fractions),
        figsize=(5.2 * len(av_fractions), 8.2),
        squeeze=False,
        sharex="col",
    )
    colors = {
        label: plt.get_cmap("tab10")(index % 10)
        for index, label in enumerate(labels)
    }
    reference = labels[0]
    for column, av_fraction in enumerate(av_fractions):
        speed_axis = axes[0][column]
        delta_axis = axes[1][column]
        for label in labels:
            values = sorted(
                (
                    row for row in architecture_summary
                    if row["architecture"] == label
                    and float(row["av_fraction"]) == av_fraction
                ),
                key=lambda row: float(row["requested_density"]),
            )
            x = [float(row["requested_density"]) for row in values]
            y = [float(row["pv_speed_mean"]) for row in values]
            error = [1.96 * float(row["pv_speed_sem"]) for row in values]
            speed_axis.errorbar(
                x,
                y,
                yerr=error,
                marker="o",
                capsize=3,
                color=colors[label],
                label=label,
            )
        speed_axis.set_title(f"AV fraction = {av_fraction:.2f}")
        speed_axis.set_ylabel("PV speed [cells/step]")
        speed_axis.grid(alpha=0.25)

        for comparison in labels[1:]:
            values = sorted(
                (
                    row for row in pairwise_summary
                    if row["architecture_a"] == reference
                    and row["architecture_b"] == comparison
                    and float(row["av_fraction"]) == av_fraction
                ),
                key=lambda row: float(row["requested_density"]),
            )
            x = [float(row["requested_density"]) for row in values]
            y = [
                float(row["pv_speed_delta_a_minus_b_mean"])
                for row in values
            ]
            low = [float(row["pv_speed_delta_ci_low"]) for row in values]
            high = [float(row["pv_speed_delta_ci_high"]) for row in values]
            error = [
                [mean - lo for mean, lo in zip(y, low)],
                [hi - mean for mean, hi in zip(y, high)],
            ]
            delta_axis.errorbar(
                x,
                y,
                yerr=error,
                marker="o",
                capsize=3,
                color=colors[comparison],
                label=f"{reference} - {comparison}",
            )
        delta_axis.axhline(0.0, color="black", linewidth=1)
        delta_axis.set_xlabel("Density")
        delta_axis.set_ylabel("Paired PV-speed delta")
        delta_axis.grid(alpha=0.25)
    axes[0][0].legend(fontsize=8)
    if len(labels) > 1:
        axes[1][0].legend(fontsize=8)
    figure.suptitle("PV policy architecture comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict paired comparison of Graph, NodeMLP, and GridCNN policies"
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help=(
            "LABEL=/path/to/*_runs.csv; repeat for every AV-fraction file and "
            "architecture, reusing LABEL across files"
        ),
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=900_001)
    parser.add_argument(
        "--equivalence-margin",
        type=float,
        help="optional predeclared absolute PV-speed equivalence margin",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="pv_architecture_comparison")
    parser.add_argument(
        "--plot", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.confidence < 1.0:
        raise SystemExit("--confidence must be in (0, 1)")
    if args.bootstrap_samples < 1:
        raise SystemExit("--bootstrap-samples must be >= 1")
    if args.equivalence_margin is not None and args.equivalence_margin <= 0.0:
        raise SystemExit("--equivalence-margin must be > 0")
    try:
        labels, indexed = _load_inputs(args.input)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    architecture_summary = _architecture_summary(labels, indexed)
    pairwise_rows = _pairwise_rows(labels, indexed)
    pairwise_summary = _pairwise_summary(
        pairwise_rows,
        confidence=args.confidence,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        equivalence_margin=args.equivalence_margin,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    architecture_path = args.out_dir / f"{args.prefix}_architectures.csv"
    paired_path = args.out_dir / f"{args.prefix}_paired_runs.csv"
    paired_summary_path = args.out_dir / f"{args.prefix}_paired_summary.csv"
    plot_path = args.out_dir / f"{args.prefix}.png"
    metadata_path = args.out_dir / f"{args.prefix}_metadata.json"
    _write_csv(architecture_path, architecture_summary)
    _write_csv(paired_path, pairwise_rows)
    _write_csv(paired_summary_path, pairwise_summary)
    plot_written = bool(args.plot) and _write_plot(
        plot_path,
        architecture_summary,
        pairwise_summary,
        labels=labels,
    )
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "labels": labels,
                "paired_rows_per_architecture": len(indexed[labels[0]]),
                "confidence": args.confidence,
                "bootstrap_samples": args.bootstrap_samples,
                "seed": args.seed,
                "equivalence_margin": args.equivalence_margin,
                "inputs": args.input,
                "outputs": {
                    "architectures_csv": str(architecture_path),
                    "paired_runs_csv": str(paired_path),
                    "paired_summary_csv": str(paired_summary_path),
                    "plot_png": str(plot_path) if plot_written else None,
                },
            },
            stream,
            indent=2,
            sort_keys=True,
        )
    print(f"architectures={','.join(labels)}", flush=True)
    print(f"architectures_csv={architecture_path}", flush=True)
    print(f"paired_summary_csv={paired_summary_path}", flush=True)
    if plot_written:
        print(f"plot={plot_path}", flush=True)
    print(f"metadata={metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
