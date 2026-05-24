from __future__ import annotations

import importlib.util
import json
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
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert "Benchmark=snfs_full_step_phase_costs" in completed.stdout
    assert out_json.exists() and out_md.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    for key in ("benchmark", "schema_version", "cases", "recommendation", "python", "platform", "numpy_version"):
        assert key in data
    assert data["cases"][0]["summary"]["per_step_ns_mean"] >= 0
    md = out_md.read_text(encoding="utf-8")
    assert "## Case summary" in md
    assert "## Recommended next task" in md
