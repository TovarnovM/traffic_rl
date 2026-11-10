"""Strict task/seed/execution/source-identity fixtures; no simulator runtime."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from physica_mc import schema

BASE_SEEDS = {
    "scenario": 101,
    "warmup": 202,
    "measurement": 303,
    "pv_assignment": 404,
    "av_selection": 505,
}


def config(**overrides):
    value = dict(schema.DEFAULT_CONFIG)
    value.update(overrides)
    return value


def scientific_config(**overrides):
    value = config(
        density=0.1,
        av_fraction=0.9,
        warmup_steps=1000,
        measure_steps=2000,
        backend="optimized",
    )
    value.update(overrides)
    return value


def test_canonical_json_is_strict_and_deterministic():
    assert schema.canonical_json({"b": 1, "a": [True, None, "é"]}) == (
        '{"a":[true,null,"é"],"b":1}'.encode("utf-8")
    )
    assert schema.digest({"a": 1}) == hashlib.sha256(b'{"a":1}').hexdigest()
    for value in (
        float("nan"),
        float("inf"),
        (1, 2),
        {1: "x"},
        b"bytes",
        np.int64(1),
    ):
        with pytest.raises(ValueError):
            schema.canonical_json(value)
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError):
        schema.canonical_json(cyclic)


def test_file_sha256(tmp_path):
    path = tmp_path / "blob"
    path.write_bytes(b"abc")
    assert schema.file_sha256(path) == hashlib.sha256(b"abc").hexdigest()


def test_seed_recipe_and_default_task_identity():
    expected = {
        "scenario": 1_103_020_012,
        "warmup": 1_103_020_098,
        "measurement": 1_103_020_194,
        "pv_assignment": 1_103_021_001,
        "av_selection": 1_103_022_001,
    }
    assert schema.seed_record("pilot", 3, 2) == expected
    task = schema.make_task("B1", density=0.2, density_index=3, replicate=2)
    assert task["science"]["seeds"] == expected
    assert task["science"]["density_index"] == 3
    assert task["task_id"] == schema.digest(task["science"])
    assert schema.validate_task(task) is None
    with pytest.raises(ValueError):
        schema.seed_record("bogus", 0, 0)
    with pytest.raises(ValueError):
        schema.seed_record("pilot", -1, 0)
    with pytest.raises(ValueError):
        schema.seed_record("pilot", 0, -1)


def test_make_task_accepts_explicit_seeds_and_rejects_bad_inputs():
    task = schema.make_task("B1", density=0.2, seeds=BASE_SEEDS)
    assert task["science"]["seeds"] == BASE_SEEDS
    assert schema.validate_task(task) is None

    with pytest.raises(ValueError):
        schema.make_task("bogus", seeds=BASE_SEEDS)
    with pytest.raises(ValueError):
        schema.make_task(
            "B1", density=0.2, config={**config(), "bogus": 1}, seeds=BASE_SEEDS
        )
    with pytest.raises(ValueError):
        schema.make_task(
            "B1", density=0.2, density_index=0, seeds=BASE_SEEDS
        )
    with pytest.raises(ValueError):
        schema.make_task("B1", density=0.011, seeds=BASE_SEEDS)
    with pytest.raises(ValueError):
        schema.make_task("B1", density=float("nan"), seeds=BASE_SEEDS)
    with pytest.raises(ValueError):
        schema.make_task("B1", density=0.2, seeds={"scenario": 1})
    with pytest.raises(ValueError):
        schema.make_task("B1", density=0.2, model_id="unexpected", seeds=BASE_SEEDS)


def test_task_validation_rejects_unknown_fields_and_digest_mismatch():
    task = schema.make_task("B1", density=0.2, seeds=BASE_SEEDS)
    broken = json.loads(json.dumps(task))
    broken["extra"] = 1
    with pytest.raises(ValueError):
        schema.validate_task(broken)

    broken = json.loads(json.dumps(task))
    broken["science"]["extra"] = 1
    with pytest.raises(ValueError):
        schema.validate_task(broken)

    broken = json.loads(json.dumps(task))
    broken["science"]["replicate"] = 1
    with pytest.raises(ValueError):
        schema.validate_task(broken)

    broken = json.loads(json.dumps(task))
    broken["task_id"] = "0" * 64
    with pytest.raises(ValueError):
        schema.validate_task(broken)


def test_scientific_purpose_is_checked_after_make_task():
    task = schema.make_task(
        "B1",
        density=0.1,
        av_fraction=0.9,
        config=scientific_config(backend="reference"),
        seeds=BASE_SEEDS,
        purpose="scientific",
    )
    with pytest.raises(ValueError, match="optimized backend"):
        schema.validate_task(task)

    task = schema.make_task(
        "B1",
        density=0.1,
        av_fraction=0.9,
        config=scientific_config(warmup_steps=1, measure_steps=2),
        seeds=BASE_SEEDS,
        purpose="scientific",
    )
    with pytest.raises(ValueError):
        schema.validate_task(task)


def test_learned_and_fixed_model_provenance_are_strict():
    with pytest.raises(ValueError):
        schema.make_task("graph", density=0.2, seeds=BASE_SEEDS)
    with pytest.raises(ValueError):
        schema.make_task(
            "graph",
            density=0.2,
            seeds=BASE_SEEDS,
            model_id="",
            training_line="line",
            candidate_id="candidate",
            env_steps=1,
        )
    with pytest.raises(ValueError):
        schema.make_task(
            "B1",
            density=0.2,
            seeds=BASE_SEEDS,
            training_line="line",
        )
    learned = schema.make_task(
        "graph",
        density=0.2,
        seeds=BASE_SEEDS,
        model_id="a" * 64,
        training_line="line",
        candidate_id="candidate-1",
        env_steps=100,
    )
    assert schema.validate_task(learned) is None


def test_pairing_key_excludes_policy_and_av_fraction_but_not_traffic():
    base = schema.make_task("B1", density=0.2, seeds=BASE_SEEDS)
    h1 = schema.make_task("H1", density=0.2, seeds=BASE_SEEDS)
    av = schema.make_task("B1", density=0.2, av_fraction=0.1, seeds=BASE_SEEDS)
    assert schema.pairing_key(base) == schema.pairing_key(h1)
    assert schema.pairing_key(base) == schema.pairing_key(av)
    assert len(schema.pairing_key(base)) == 64

    assert schema.pairing_key(base) != schema.pairing_key(
        schema.make_task("B1", density=0.1, seeds=BASE_SEEDS)
    )
    assert schema.pairing_key(base) != schema.pairing_key(
        schema.make_task("B1", density=0.2, replicate=1, seeds=BASE_SEEDS)
    )
    assert schema.pairing_key(base) != schema.pairing_key(
        schema.make_task("B1", density=0.2, split="validation", seeds=BASE_SEEDS)
    )
    transfer = schema.make_task(
        "B1", density=0.2, config=config(road_length=2000), seeds=BASE_SEEDS
    )
    assert schema.pairing_key(base) != schema.pairing_key(transfer)


def test_execution_identity_excludes_bundle_transport_path():
    task = schema.make_task("B1", density=0.2, seeds=BASE_SEEDS)
    default = schema.make_execution(task)
    assert default["profile"] == {"device": "cpu", "threads": 1, "observer": "full12"}
    assert default["implementation"] == schema.source_identity()
    assert default["task_id"] == task["task_id"]
    schema._validate_execution(default, task)

    transport_a = schema.make_execution(task, model_bundle="/transport/a")
    transport_b = schema.make_execution(task, model_bundle="/transport/b")
    assert transport_a["execution_id"] == transport_b["execution_id"]
    assert transport_a["execution_id"] == default["execution_id"]

    mechanism = schema.make_execution(task, observer="mechanism")
    device = schema.make_execution(task, device="cuda:0", threads=2)
    assert mechanism["execution_id"] != default["execution_id"]
    assert device["execution_id"] != default["execution_id"]
    with pytest.raises(ValueError):
        schema.make_execution(task, observer="bogus")
    with pytest.raises(ValueError):
        schema.make_execution(task, threads=0)
    with pytest.raises(ValueError):
        schema.make_execution(task, device="")


def test_source_identity_is_stable_relative_and_relocation_independent(tmp_path):
    first = schema.source_identity()
    second = schema.source_identity()
    assert first == second
    assert len(first["source_digest"]) == 64
    assert set(first["packages"]) == {"physica_mc", "snfs_traffic"}
    for package in first["packages"].values():
        assert package["files"]
        assert all(not Path(name).is_absolute() for name in package["files"])
        assert ".." not in "".join(package["files"])

    source_root = Path(schema.__file__).resolve().parent
    relocated_root = tmp_path / "relocated"
    shutil.copytree(
        source_root,
        relocated_root / "physica_mc",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    shutil.copytree(
        source_root.parent / "snfs_traffic",
        relocated_root / "snfs_traffic",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    spec = importlib.util.spec_from_file_location(
        "relocated_physica_schema", relocated_root / "physica_mc" / "schema.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    relocated = module.source_identity()
    assert relocated["packages"]["physica_mc"]["digest"] == first["packages"]["physica_mc"]["digest"]
    assert relocated["source_digest"] == first["source_digest"]
