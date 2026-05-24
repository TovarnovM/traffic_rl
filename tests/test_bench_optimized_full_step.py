from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    base = str(Path(__file__).resolve().parents[1] / "src")
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = base if not current else f"{base}:{current}"
    return env


def test_bench_optimized_cli_smoke(tmp_path) -> None:
    out_json = tmp_path / "o.json"
    out_md = tmp_path / "o.md"
    subprocess.run(
        [
            sys.executable,
            "benchmarks/bench_optimized_full_step.py",
            "--preset",
            "smoke",
            "--backend",
            "both",
            "--repeats",
            "1",
            "--warmups",
            "1",
            "--steps",
            "2",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_cli_env(),
    )
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["benchmark"] == "snfs_optimized_full_step"
    assert "numba_available" in data
    assert "recommendation" in data
    assert len(data["cases"]) >= 1
    case = data["cases"][0]
    assert "reference" in case and "optimized" in case
    assert "per_repeat_ms_per_step" in case["reference"]
    assert "equivalence" in case and "state_equal" in case["equivalence"]
    assert "speedup_x" in case
    assert "| case | vehicles | steps |" in out_md.read_text(encoding="utf-8")


def test_bench_optimized_unknown_case_fails(tmp_path) -> None:
    out_json = tmp_path / "o.json"
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/bench_optimized_full_step.py",
            "--preset",
            "standard",
            "--cases",
            "small,medium,dense",
            "--out-json",
            str(out_json),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_cli_env(),
    )
    assert completed.returncode != 0
    combined = f"{completed.stdout}\n{completed.stderr}"
    assert "unknown case name(s)" in combined
