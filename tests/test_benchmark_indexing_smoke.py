from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_benchmark_module():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_indexing.py"
    spec = importlib.util.spec_from_file_location("benchmark_indexing", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_indexing_quick_smoke(tmp_path):
    mod = _load_benchmark_module()
    out_json = tmp_path / "bench.json"
    out_md = tmp_path / "bench.md"
    config = mod.BenchmarkConfig(
        out_json=out_json,
        out_md=out_md,
        quick=True,
        repeat=1,
        warmup=0,
        sizes="small",
        include_numba=False,
    )
    data = mod.run_benchmark(config)
    out_json.write_text(json.dumps(data), encoding="utf-8")
    out_md.write_text(mod._format_report(data), encoding="utf-8")

    assert out_json.exists()
    assert out_md.exists()

    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "snfs_indexing_benchmark_v1"
    assert loaded["cases"]

    measurements = loaded["cases"][0]["measurements"]
    assert "public_indexing_pipeline" in measurements
    assert "kernel_indexing_pipeline" in measurements
