"""Canonical task, execution and source identity schema v1.

This module is deliberately dependency-light: standard library only.  It is
imported by the storage/Reader path, so it must never import the simulator,
Torch or Ray.  The scientific identity is the hash of the canonical task
``science`` object; the execution identity adds the actual implementation and
the execution profile, but not the transport location of a model bundle.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

SCHEMA_VERSION = 1
PROTOCOL = "physica-a-e2-native-eq9-v1"

SPLITS = ("pilot", "validation", "fresh_test")
PURPOSES = ("technical", "scientific")
METHODS = (
    "B1",
    "B2",
    "noop",
    "left_if_safe_original",
    "H1",
    "H2",
    "H4",
    "H8",
    "graph",
    "node",
    "grid",
)
LEARNED_METHODS = frozenset(("graph", "node", "grid"))
FIXED_METHODS = frozenset(METHODS) - LEARNED_METHODS
OBSERVER_PROFILES = ("off", "full12", "mechanism")

SEED_ROLES = (
    "scenario",
    "warmup",
    "measurement",
    "pv_assignment",
    "av_selection",
)
ROLE_OFFSETS = {
    "scenario": 11,
    "warmup": 97,
    "measurement": 193,
    "pv_assignment": 1000,
    "av_selection": 2000,
}
SPLIT_BASES = {
    "pilot": 1_100_000_001,
    "validation": 310_000_001,
    "fresh_test": 710_000_001,
}
DENSITY_STEP_INDEX = 1_000_000
REPLICATE_STEP_INDEX = 10_000
MAX_DENSITY_INDEX = 1_000_000
# 0..99 keeps replicate*10000 + role offset below the density step (1e6),
# so different canonical density indices cannot alias one another.
MAX_REPLICATE = 99
MAX_TASK_REPLICATE = 1_000_000
MAX_SEED = (1 << 63) - 1

DEFAULT_CONFIG = {
    "road_length": 1000,
    "num_lanes": 4,
    "density": 0.2,
    "av_fraction": 0.95,
    "front_distance": 30,
    "back_distance": 10,
    "sensor_distance": 60,
    "cooldown_steps": 1,
    "warmup_steps": 1000,
    "measure_steps": 2000,
    "backend": "optimized",
    "validate": False,
    "vmax_hdv": 5,
    "vmax_priority": 6,
    "G": 15,
    "q": 0.99,
    "r": 0.99,
    "S": 2,
    "P1": 0.999,
    "P2": 0.99,
    "P3": 0.98,
    "P4": 0.01,
    "p_lane_change": 0.5,
    "yield_distance": 30,
    "yield_cooldown_steps": 1,
}
CONFIG_KEYS = tuple(DEFAULT_CONFIG)

DENSITY_GRID = tuple(round(0.05 * (index + 1), 2) for index in range(18))

# Scientific v1 fixes every scientific default except the explicit transfer
# road length and the two scenario parameters (density/AV fraction).
SCIENTIFIC_FIXED = {
    key: value
    for key, value in DEFAULT_CONFIG.items()
    if key not in ("road_length", "density", "av_fraction")
}

_INT_FIELDS = {
    "road_length": (109, 4000),
    "num_lanes": (1, 8),
    "front_distance": (0, 1024),
    "back_distance": (0, 1024),
    "sensor_distance": (1, 1024),
    "cooldown_steps": (1, 1),
    "warmup_steps": (0, 2000),
    "measure_steps": (1, 2000),
    "vmax_hdv": (1, 5),
    "vmax_priority": (1, 6),
    "G": (0, 255),
    "S": (1, 255),
    "yield_distance": (1, 1000),
    "yield_cooldown_steps": (0, 1000),
}
_FLOAT_FIELDS = {
    "density": (0.0, 0.9),
    "av_fraction": (0.0, 1.0),
    "q": (0.0, 1.0),
    "r": (0.0, 1.0),
    "P1": (0.0, 1.0),
    "P2": (0.0, 1.0),
    "P3": (0.0, 1.0),
    "P4": (0.0, 1.0),
    "p_lane_change": (0.0, 1.0),
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_PACKAGE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TEXT_SAFE = re.compile(r"^[^\x00-\x1f\x7f]+$")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(value, name: str) -> float:
    if not _is_number(value):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _integer(value, name: str, low: int, high: int) -> int:
    if not _is_int(value):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _text(value, name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{name} must be a nonempty string of at most {maximum} characters")
    if not _TEXT_SAFE.match(value):
        raise ValueError(f"{name} contains a control character")
    return value


def _hex64(value, name: str) -> str:
    if not isinstance(value, str) or not _HEX64.match(value):
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")
    return value


def _check_json(value, *, depth: int = 0, seen: set | None = None) -> None:
    """Reject objects that are not finite, canonical JSON-native values."""

    if seen is None:
        seen = set()
    if depth > 64:
        raise ValueError("JSON value is nested too deeply")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return
    if isinstance(value, str):
        return
    if isinstance(value, (list, dict)):
        identity = id(value)
        if identity in seen:
            raise ValueError("JSON value contains a cycle")
        seen.add(identity)
        if isinstance(value, list):
            for item in value:
                _check_json(item, depth=depth + 1, seen=seen)
        else:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError("JSON object keys must be strings")
                _check_json(item, depth=depth + 1, seen=seen)
        seen.remove(identity)
        return
    raise ValueError(f"value of type {type(value).__name__} is not JSON-native")


def canonical_json(value) -> bytes:
    """Return canonical UTF-8 JSON (sorted keys, ``,``/``:`` separators)."""

    _check_json(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError(f"value cannot be encoded as canonical JSON: {exc}") from exc


def digest(value) -> str:
    """SHA256 of the canonical UTF-8 JSON encoding of ``value``."""

    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path) -> str:
    digest_value = hashlib.sha256()
    with open(Path(path).resolve(), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest_value.update(block)
    return digest_value.hexdigest()


def _package_root(name: str) -> Path:
    if name == "physica_mc":
        return Path(__file__).resolve().parent
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        spec = None
    locations = list(spec.submodule_search_locations or ()) if spec is not None else []
    if locations:
        return Path(locations[0]).resolve()
    sibling = Path(__file__).resolve().parent.parent / name
    if sibling.is_dir():
        return sibling.resolve()
    raise RuntimeError(f"owned package {name!r} is not installed or importable")


def _package_files(root: Path) -> dict[str, str]:
    stem = root.resolve()
    if not stem.is_dir():
        raise RuntimeError(f"owned package directory is missing: {stem}")
    files: dict[str, str] = {}
    for path in sorted(stem.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise RuntimeError(f"owned package contains a symlink: {path}")
        if path.is_dir():
            continue
        if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(stem).as_posix()
        files[relative] = file_sha256(path)
    if not files:
        raise RuntimeError(f"owned package {stem.name!r} contains no files")
    return files


def source_identity() -> dict:
    """Hash the installed owned package files, independent of absolute paths.

    Only file contents and package-relative names enter the digest, so moving
    the installed tree or running from another working directory does not
    change the identity.
    """

    packages = {}
    for name in ("physica_mc", "snfs_traffic"):
        files = _package_files(_package_root(name))
        packages[name] = {
            "digest": digest({"name": name, "files": files}),
            "files": files,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "packages": packages,
        "source_digest": digest(
            {"schema_version": SCHEMA_VERSION, "packages": packages}
        ),
    }


def canonical_density_index(density: float) -> int:
    value = _number(density, "density")
    for index, grid_density in enumerate(DENSITY_GRID):
        if value == grid_density:
            return index
    raise ValueError(
        "non-grid density requires an explicit density_index and seed manifest"
    )


def seed_record(split: str, density_index: int, replicate: int) -> dict:
    """Return the five independent protocol seeds for one pairing block."""

    if not isinstance(split, str) or split not in SPLIT_BASES:
        raise ValueError(f"split must be one of {SPLITS}")
    density_index = _integer(
        density_index, "density_index", 0, len(DENSITY_GRID) - 1
    )
    replicate = _integer(replicate, "replicate", 0, MAX_REPLICATE)
    base = SPLIT_BASES[split]
    seeds = {
        role: base
        + density_index * DENSITY_STEP_INDEX
        + replicate * REPLICATE_STEP_INDEX
        + ROLE_OFFSETS[role]
        for role in SEED_ROLES
    }
    for role, value in seeds.items():
        if value < 0 or value > MAX_SEED:
            raise ValueError(f"derived seed {role} is outside the supported range")
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("derived seed roles collide")
    return seeds


def _validate_config(config, purpose: str, *, purpose_rules: bool) -> None:
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    if set(config) != set(CONFIG_KEYS):
        missing = sorted(set(CONFIG_KEYS) - set(config))
        unknown = sorted(set(config) - set(CONFIG_KEYS))
        raise ValueError(f"config keys mismatch (missing={missing}, unknown={unknown})")

    for key, (low, high) in _INT_FIELDS.items():
        _integer(config[key], f"config.{key}", low, high)
    for key, (low, high) in _FLOAT_FIELDS.items():
        value = config[key]
        if not isinstance(value, float):
            raise ValueError(f"config.{key} must be a JSON float")
        if not math.isfinite(value):
            raise ValueError(f"config.{key} must be finite")
        if not low <= value <= high:
            raise ValueError(f"config.{key} must be in [{low}, {high}]")
    if config["density"] <= 0.0:
        raise ValueError("config.density must be > 0 for a closed-ring episode")
    if not isinstance(config["validate"], bool):
        raise ValueError("config.validate must be boolean")
    if config["backend"] not in ("optimized", "reference"):
        raise ValueError("config.backend must be 'optimized' or 'reference'")
    if config["front_distance"] + config["back_distance"] < 1:
        raise ValueError("front_distance + back_distance must be >= 1")
    if config["front_distance"] + config["back_distance"] >= config["road_length"]:
        raise ValueError("front_distance + back_distance must be < road_length")
    if config["sensor_distance"] >= config["road_length"]:
        raise ValueError("sensor_distance must be < road_length")
    if config["cooldown_steps"] != 1:
        raise ValueError("protocol v1 requires the common cooldown K=1")

    if purpose_rules and purpose == "scientific":
        if config["backend"] != "optimized":
            raise ValueError(
                "scientific purpose requires the optimized backend (no silent fallback)"
            )
        for key, expected in SCIENTIFIC_FIXED.items():
            if config[key] != expected:
                raise ValueError(
                    f"scientific config.{key}={config[key]!r} differs from v1 value {expected!r}"
                )
        if config["road_length"] not in (1000, 2000):
            raise ValueError("scientific v1 supports road_length 1000 (main) or 2000 (transfer)")
        if config["warmup_steps"] != 1000 or config["measure_steps"] != 2000:
            raise ValueError(
                "scientific purpose requires warmup_steps=1000 and measure_steps=2000"
            )


def _validate_seeds(seeds) -> None:
    if not isinstance(seeds, dict):
        raise ValueError("seeds must be a mapping")
    if tuple(seeds) != SEED_ROLES:
        if set(seeds) != set(SEED_ROLES):
            raise ValueError("seeds must contain exactly the five protocol roles")
    for role in SEED_ROLES:
        _integer(seeds[role], f"seeds.{role}", 0, MAX_SEED)
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("seed roles must not collide")


def _validate_model_fields(science, method: str) -> None:
    fields = {
        name: science[name]
        for name in ("model_id", "training_line", "candidate_id", "env_steps")
    }
    if method in LEARNED_METHODS:
        for name in ("model_id", "training_line", "candidate_id"):
            _text(fields[name], f"science.{name}")
        _integer(fields["env_steps"], "science.env_steps", 1, 1_000_000_000)
    else:
        for name, value in fields.items():
            if value is not None:
                raise ValueError(
                    f"fixed controller {method} requires null science.{name}"
                )


def _validate_task(task, *, purpose_rules: bool) -> None:
    if not isinstance(task, dict):
        raise ValueError("task must be a mapping")
    if set(task) != {"schema_version", "task_id", "science"}:
        raise ValueError("task has unknown or missing keys")
    if not _is_int(task["schema_version"]) or task["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported task schema_version")
    _hex64(task["task_id"], "task.task_id")
    science = task["science"]
    if not isinstance(science, dict):
        raise ValueError("task.science must be a mapping")
    expected_keys = {
        "protocol",
        "purpose",
        "split",
        "method",
        "density_index",
        "replicate",
        "config",
        "seeds",
        "model_id",
        "training_line",
        "candidate_id",
        "env_steps",
    }
    if set(science) != expected_keys:
        raise ValueError("task.science has unknown or missing keys")
    if science["protocol"] != PROTOCOL:
        raise ValueError(f"unsupported protocol {science['protocol']!r}")
    if science["purpose"] not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}")
    if science["split"] not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if science["method"] not in METHODS:
        raise ValueError(f"unsupported method {science['method']!r}")
    density_index = _integer(
        science["density_index"], "science.density_index", 0, MAX_DENSITY_INDEX
    )
    _integer(science["replicate"], "science.replicate", 0, MAX_TASK_REPLICATE)
    config = science["config"]
    _validate_config(config, science["purpose"], purpose_rules=purpose_rules)
    _validate_seeds(science["seeds"])
    _validate_model_fields(science, science["method"])

    if config["density"] in DENSITY_GRID:
        expected_index = DENSITY_GRID.index(config["density"])
        if density_index != expected_index:
            raise ValueError(
                f"canonical density {config['density']} requires density_index={expected_index}, "
                f"got {density_index}"
            )
    if digest(science) != task["task_id"]:
        raise ValueError("task_id does not match the canonical science digest")


def validate_task(task) -> None:
    """Validate one complete task; never silently coerce scientific values."""

    _validate_task(task, purpose_rules=True)


def make_task(
    method,
    *,
    density=0.2,
    av_fraction=0.95,
    density_index=None,
    replicate=0,
    split="pilot",
    config=None,
    seeds=None,
    model_id=None,
    training_line=None,
    candidate_id=None,
    env_steps=None,
    purpose="technical",
):
    """Build a canonical task object.  Purpose-specific rules run in validate."""

    if method not in METHODS:
        raise ValueError(f"unsupported method {method!r}")
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    density = _number(density, "density")
    av_fraction = _number(av_fraction, "av_fraction")
    replicate = _integer(replicate, "replicate", 0, MAX_TASK_REPLICATE)
    if config is None:
        overrides = {}
    elif isinstance(config, dict):
        overrides = dict(config)
    else:
        raise ValueError("config must be a mapping or None")
    unknown = sorted(set(overrides) - set(CONFIG_KEYS))
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    if "density" in overrides and float(overrides["density"]) != density:
        raise ValueError("config.density disagrees with the density argument")
    if "av_fraction" in overrides and float(overrides["av_fraction"]) != av_fraction:
        raise ValueError("config.av_fraction disagrees with the av_fraction argument")
    merged = dict(DEFAULT_CONFIG)
    merged.update(overrides)
    merged["density"] = density
    merged["av_fraction"] = av_fraction

    if density_index is None:
        density_index = canonical_density_index(density)
    else:
        density_index = _integer(
            density_index, "density_index", 0, MAX_DENSITY_INDEX
        )
        if density in DENSITY_GRID:
            expected_index = DENSITY_GRID.index(density)
            if density_index != expected_index:
                raise ValueError(
                    f"canonical density {density} requires density_index={expected_index}"
                )

    if seeds is None:
        if density not in DENSITY_GRID:
            raise ValueError(
                "non-grid density requires an explicit seed manifest"
            )
        seeds = seed_record(split, density_index, replicate)
    elif not isinstance(seeds, dict):
        raise ValueError("seeds must be a mapping")
    else:
        seeds = dict(seeds)

    science = {
        "protocol": PROTOCOL,
        "purpose": purpose,
        "split": split,
        "method": method,
        "density_index": density_index,
        "replicate": replicate,
        "config": merged,
        "seeds": seeds,
        "model_id": model_id,
        "training_line": training_line,
        "candidate_id": candidate_id,
        "env_steps": env_steps,
    }
    task = {
        "schema_version": SCHEMA_VERSION,
        "task_id": digest(science),
        "science": science,
    }
    _validate_task(task, purpose_rules=False)
    return task


def pairing_key(task) -> str:
    """Stable pairing-block digest, excluding policy and AV fraction."""

    validate_task(task)
    science = task["science"]
    config = science["config"]
    traffic_keys = (
        "road_length",
        "num_lanes",
        "density",
        "warmup_steps",
        "measure_steps",
        "vmax_hdv",
        "vmax_priority",
        "G",
        "q",
        "r",
        "S",
        "P1",
        "P2",
        "P3",
        "P4",
        "p_lane_change",
    )
    block = {
        "schema_version": SCHEMA_VERSION,
        "protocol": science["protocol"],
        "split": science["split"],
        "density_index": science["density_index"],
        "replicate": science["replicate"],
        "traffic": {key: config[key] for key in traffic_keys},
    }
    return digest(block)


def _validate_implementation(implementation) -> None:
    """Validate an actual source-identity record without reading the source."""

    if isinstance(implementation, str):
        _hex64(implementation, "execution.implementation")
        return
    if not isinstance(implementation, dict):
        raise ValueError("execution.implementation must be a source identity mapping")
    if set(implementation) != {"schema_version", "packages", "source_digest"}:
        raise ValueError("execution.implementation has unknown or missing keys")
    if not _is_int(implementation["schema_version"]) or implementation["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported implementation schema_version")
    _hex64(implementation["source_digest"], "implementation.source_digest")
    packages = implementation["packages"]
    if not isinstance(packages, dict) or not 1 <= len(packages) <= 16:
        raise ValueError("implementation.packages must contain the owned packages")
    for name, package in packages.items():
        if not isinstance(name, str) or not _PACKAGE_NAME.match(name):
            raise ValueError("invalid implementation package name")
        if not isinstance(package, dict) or set(package) != {"digest", "files"}:
            raise ValueError(f"invalid package identity for {name}")
        _hex64(package["digest"], f"package[{name}].digest")
        files = package["files"]
        if not isinstance(files, dict) or not files or len(files) > 100_000:
            raise ValueError(f"invalid package file inventory for {name}")
        for relative, value in files.items():
            if not isinstance(relative, str) or not relative:
                raise ValueError("invalid package-relative file name")
            if relative.startswith("/") or "\\" in relative or "\x00" in relative:
                raise ValueError("package file name must be a safe relative POSIX path")
            parts = relative.split("/")
            if any(part in ("", ".", "..") for part in parts):
                raise ValueError("package file name must be a safe relative POSIX path")
            _hex64(value, f"package[{name}].files[{relative}]")
        if digest({"name": name, "files": files}) != package["digest"]:
            raise ValueError(f"package identity digest mismatch for {name}")
    if digest(
        {"schema_version": SCHEMA_VERSION, "packages": packages}
    ) != implementation["source_digest"]:
        raise ValueError("source identity digest mismatch")


def _identity_profile(profile) -> dict:
    if not isinstance(profile, dict):
        raise ValueError("execution.profile must be a mapping")
    unknown = sorted(set(profile) - {"device", "threads", "observer", "model_bundle"})
    if unknown:
        raise ValueError(f"unknown execution profile keys: {unknown}")
    for name in ("device", "threads", "observer"):
        if name not in profile:
            raise ValueError(f"execution.profile is missing {name}")
    device = _text(profile["device"], "execution.profile.device", maximum=64)
    threads = _integer(profile["threads"], "execution.profile.threads", 1, 1024)
    observer = profile["observer"]
    if observer not in OBSERVER_PROFILES:
        raise ValueError(f"observer must be one of {OBSERVER_PROFILES}")
    bundle = profile.get("model_bundle")
    if bundle is not None:
        _text(bundle, "execution.profile.model_bundle", maximum=4096)
    return {"device": device, "threads": threads, "observer": observer}


def _validate_execution(execution, task) -> None:
    if not isinstance(execution, dict):
        raise ValueError("execution must be a mapping")
    required_keys = {
        "schema_version",
        "execution_id",
        "task_id",
        "implementation",
        "profile",
    }
    if not required_keys.issubset(execution):
        raise ValueError("execution has missing keys")
    unknown = sorted(set(execution) - required_keys - {"model_bundle"})
    if unknown:
        raise ValueError(f"execution has unknown keys: {unknown}")
    if not _is_int(execution["schema_version"]) or execution["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported execution schema_version")
    _hex64(execution["execution_id"], "execution.execution_id")
    if execution["task_id"] != task["task_id"]:
        raise ValueError("execution.task_id does not match the task")
    _validate_implementation(execution["implementation"])
    profile = execution["profile"]
    identity_profile = _identity_profile(profile)
    bundle = execution.get("model_bundle")
    if bundle is not None:
        _text(bundle, "execution.model_bundle", maximum=4096)
    profile_bundle = profile.get("model_bundle")
    if profile_bundle is not None and bundle is not None and profile_bundle != bundle:
        raise ValueError("conflicting model_bundle transport values")
    body = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task["task_id"],
        "implementation": execution["implementation"],
        "profile": identity_profile,
    }
    if digest(body) != execution["execution_id"]:
        raise ValueError("execution_id does not match the execution identity")


def make_execution(
    task,
    *,
    device="cpu",
    threads=1,
    observer="full12",
    model_bundle=None,
):
    """Build an execution record; the bundle transport path is not identity."""

    validate_task(task)
    profile = _identity_profile(
        {"device": device, "threads": threads, "observer": observer}
    )
    if model_bundle is not None:
        _text(model_bundle, "model_bundle", maximum=4096)
    implementation = source_identity()
    body = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task["task_id"],
        "implementation": implementation,
        "profile": profile,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "execution_id": digest(body),
        "task_id": task["task_id"],
        "implementation": implementation,
        "profile": profile,
        "model_bundle": model_bundle,
    }
