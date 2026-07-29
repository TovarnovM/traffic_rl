from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


def _load(name: str):
    research = Path(__file__).resolve().parents[1] / "research"
    if str(research) not in sys.path:
        sys.path.insert(0, str(research))
    path = research / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_runs(path: Path, *, mode: str, offset: float = 0.0) -> None:
    fields = [
        "requested_density",
        "av_fraction",
        "run_index",
        "priority_vehicle_id",
        "scenario_seed",
        "warmup_rng_seed",
        "assignment_seed",
        "av_selection_seed",
        "measurement_rng_seed",
        "mode",
        "mean_speed_priority",
        "flow_background",
        "lane_change_rate_all",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in range(3):
            writer.writerow(
                {
                    "requested_density": 0.30,
                    "av_fraction": 1.0,
                    "run_index": run,
                    "priority_vehicle_id": 900 + run,
                    "scenario_seed": 100 + run,
                    "warmup_rng_seed": 200 + run,
                    "assignment_seed": 300 + run,
                    "av_selection_seed": 400 + run,
                    "measurement_rng_seed": 500 + run,
                    "mode": mode,
                    "mean_speed_priority": 4.0 + run * 0.1 + offset,
                    "flow_background": 0.3 + run * 0.01,
                    "lane_change_rate_all": 0.02,
                }
            )


def test_checkpoint_discovery_selects_requested_iterations_and_final(tmp_path):
    selector = _load("select_pv_nongraph_checkpoint")
    periodic = tmp_path / "periodic_iter_000825_steps_000001"
    ignored = tmp_path / "periodic_iter_000850_steps_000002"
    final = tmp_path / "final_iter_001221_steps_000003"
    for directory in (periodic, ignored, final):
        directory.mkdir()
        (directory / "rllib_checkpoint.json").write_text("{}", encoding="utf-8")

    selected = selector._discover_checkpoints(
        tmp_path, iterations={825}, include_final=True
    )

    assert selected == [periodic, final]


def test_architecture_comparison_requires_identical_paired_seeds(tmp_path):
    compare = _load("compare_pv_architectures")
    graph = tmp_path / "graph.csv"
    cnn = tmp_path / "cnn.csv"
    _write_runs(graph, mode="graph_ppo")
    _write_runs(cnn, mode="nongraph_ppo", offset=-0.1)

    labels, indexed = compare._load_inputs(
        [f"graph={graph}", f"cnn={cnn}"]
    )
    paired = compare._pairwise_rows(labels, indexed)
    summary = compare._pairwise_summary(
        paired,
        confidence=0.95,
        bootstrap_samples=100,
        seed=7,
        equivalence_margin=0.15,
    )

    assert labels == ["graph", "cnn"]
    assert len(paired) == 3
    assert summary[0]["pv_speed_delta_a_minus_b_mean"] == pytest.approx(0.1)
    assert summary[0]["pv_speed_equivalent_within_margin"] is True

    with cnn.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows.pop()
    with cnn.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="paired seeds do not align"):
        compare._load_inputs([f"graph={graph}", f"cnn={cnn}"])


def test_selection_and_full_seeds_must_differ(tmp_path):
    selector = _load("select_pv_nongraph_checkpoint")
    checkpoint = tmp_path / "final_iter_1_steps_1"
    checkpoint.mkdir()
    (checkpoint / "rllib_checkpoint.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="must differ"):
        selector.main(
            [
                "--checkpoints",
                str(checkpoint),
                "--selection-seed",
                "10",
                "--full-seed",
                "10",
                "--out-dir",
                str(tmp_path / "out"),
                "--dry-run",
            ]
        )
