from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_runner():
    path = (
        Path(__file__).resolve().parents[1]
        / "research"
        / "priority_baselines.py"
    )
    spec = importlib.util.spec_from_file_location(
        "priority_baselines_test_module", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_paired_baseline_runner_smoke_uses_shared_snapshot_and_seed():
    runner = _load_runner()
    args = runner.parse_args(
        [
            "--road-length",
            "40",
            "--num-lanes",
            "4",
            "--densities",
            "0.10",
            "--runs",
            "1",
            "--warmup-steps",
            "1",
            "--measure-steps",
            "3",
            "--priority-scenarios",
            "one",
            "--matched-fractions",
            "0.25",
            "--backend",
            "reference",
            "--bootstrap-samples",
            "20",
            "--no-plot",
        ]
    )
    spec = runner.WorkerSpec(0.10, 0, 0, vars(args).copy())

    rows = runner.run_density_replication(spec)

    assert {row["mode"] for row in rows} == {
        runner.BASELINE1,
        runner.BASELINE2_ALL,
        f"{runner.BASELINE2_MATCHED_PREFIX}0.250",
    }
    assert len({row["scenario_seed"] for row in rows}) == 1
    assert len({row["measurement_rng_seed"] for row in rows}) == 1
    assert len({row["priority_vehicle_ids"] for row in rows}) == 1
    assert all(row["measurement_steps"] == 3 for row in rows)
    baseline = next(row for row in rows if row["mode"] == runner.BASELINE1)
    assert baseline["rule_requests"] == 0

    paired = runner.paired_rows(rows)
    aggregate = runner.paired_summary(
        paired,
        bootstrap_samples=20,
        seed=5,
        max_background_flow_loss=0.05,
        min_effective_pairs=1,
    )
    assert len(paired) == 2
    assert len(aggregate) == 2
