from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_full_step_phases.py"
    spec = importlib.util.spec_from_file_location("bench_full_step_phases", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    base = str(Path(__file__).resolve().parents[1] / "src")
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = base if not current else f"{base}:{current}"
    return env


def test_import_smoke():
    assert _load_module() is not None


def test_case_construction_smoke():
    mod = _load_module()
    cases = mod.get_cases("smoke", seed=0, steps=10)
    assert len(cases) >= 1
    for case in cases:
        assert case.num_lanes > 0
        assert case.road_length > 0
        assert case.steps > 0
        assert 0.0 <= case.density <= 1.0


def test_correctness_guardrail_tiny_case():
    mod = _load_module()
    case = mod.BenchCase("tiny", 2, 20, 0.2, 0.5, 2, 123)
    mod.assert_phased_equivalent(case)


def test_cli_smoke(tmp_path):
    out_json = tmp_path / "full_step_phase.json"
    out_md = tmp_path / "full_step_phase.md"
    cmd = [sys.executable, "benchmarks/bench_full_step_phases.py", "--preset", "smoke", "--repeats", "1", "--warmups", "0", "--steps", "2", "--out-json", str(out_json), "--out-md", str(out_md)]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True, env=_cli_env())
    assert "Benchmark=snfs_full_step_phase_costs" in completed.stdout
    assert out_json.exists() and out_md.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    for key in ("benchmark", "schema_version", "cases", "recommendation", "python", "platform", "numpy_version"):
        assert key in data
    assert data["cases"][0]["summary"]["per_step_ns_mean"] >= 0
    assert "orchestration_and_copy_overhead" in data["cases"][0]["repeats"][0]["phase_ns"]
    assert "orchestration_and_copy_overhead" in data["cases"][0]["summary"]["phase_per_step_ns_mean"]
    assert data["cases"][0]["summary"]["phase_per_step_ns_mean"]["orchestration_and_copy_overhead"] >= 0
    md = out_md.read_text(encoding="utf-8")
    assert "## Case summary" in md
    assert "## Recommended next task" in md
    assert "orchestration_and_copy_overhead" in md


def test_cli_unknown_case_fails_with_valid_names(tmp_path):
    out_json = tmp_path / "unused.json"
    cmd = [sys.executable, "benchmarks/bench_full_step_phases.py", "--preset", "standard", "--cases", "small,medium,dense", "--out-json", str(out_json)]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True, env=_cli_env())
    assert completed.returncode != 0
    combined = f"{completed.stdout}\n{completed.stderr}"
    assert "unknown case name(s)" in combined
    assert "small_sparse" in combined
    assert "medium_moderate" in combined
    assert "medium_dense" in combined


def test_cli_valid_subset_single_case(tmp_path):
    out_json = tmp_path / "one_case.json"
    out_md = tmp_path / "one_case.md"
    cmd = [
        sys.executable,
        "benchmarks/bench_full_step_phases.py",
        "--preset",
        "smoke",
        "--cases",
        "smoke_low_density",
        "--repeats",
        "1",
        "--warmups",
        "0",
        "--steps",
        "2",
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, env=_cli_env())
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(data["cases"]) == 1
    assert data["cases"][0]["name"] == "smoke_low_density"
