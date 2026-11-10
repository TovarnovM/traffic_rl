"""Immutable episode publication and an independent NumPy-only Reader.

Layout: ``output_root/task_id/attempt_id/{arrays.npz,completion.json}``.
``arrays.npz`` is published first and ``completion.json`` last; the marker is
an exclusive hard-link, so an existing attempt is never overwritten.  The
Reader does not import the simulator, Torch, Ray or the Observer producer
helpers: it recomputes the observer reductions and the control/PV invariants
independently from the stored arrays.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from . import schema

COMPLETE_SCHEMA_VERSION = 1
COMPLETE_STATUS = "complete"
SIDECAR_NAME = "arrays.npz"
MARKER_NAME = "completion.json"

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_RESULT_JSON_BYTES = 12 * 1024 * 1024
MAX_HEADER_BYTES = 10000
MAX_ARRAYS = 128
MAX_CONTROL_COLUMNS = 128
MAX_MECHANISM_COLUMNS = 64
MAX_DECODED_BYTES = 1 << 30
MAX_ARCHIVE_BYTES = MAX_DECODED_BYTES + (16 * 1024 * 1024)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ARRAY_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BACKEND_NAME = re.compile(r"^[A-Za-z0-9_.:+-]{1,128}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

WINDOW_IDS = tuple(
    f"{kind}_{shape}_{m}"
    for kind in ("fixed", "speed")
    for m in (1, 2, 3)
    for shape in ("sym", "front")
)
PRIMARY_WINDOW = "fixed_front_2"
REGIONS = ["rear", "centre", "front", "full"]
GROUPS = ["HDV", "AV", "background"]
REJECTION_REASONS = (
    "out_of_bounds",
    "target_occupied",
    "unsafe",
    "override_conflict",
    "other",
)
CONTROL_NAMES = (
    "step",
    "active_av",
    "selected_left",
    "selected_right",
    "selected_stay",
    "explicit_stay_commands",
    "available_left",
    "available_right",
    "eligible_nonstay",
    "common_throttled",
    "private_throttled",
    "private_suppressed_left",
    "common_suppressed",
    "requested_nonstay",
    "applied_nonstay",
    "rejected_nonstay",
    "legal_nonstay_requests",
    "applied_legal_nonstay_requests",
    "front_background",
    "centre_background",
    "rear_background",
    "front_hdv",
    "centre_hdv",
    "rear_hdv",
    "front_av",
    "centre_av",
    "rear_av",
    "requested_front",
    "requested_centre",
    "requested_rear",
    "applied_front",
    "applied_centre",
    "applied_rear",
)
HEURISTIC_C = {"H1": 1, "H2": 2, "H4": 4, "H8": 8}
MECHANISM_DTYPES = {
    "pre_pv_leader_gap": np.dtype(np.int32),
    "pre_pv_leader_type": np.dtype(np.int8),
    "pre_pv_leader_speed": np.dtype(np.int8),
    "pre_pv_leader_stopped": np.dtype(np.int8),
    "pre_pv_native_left": np.dtype(np.int8),
    "pre_pv_native_right": np.dtype(np.int8),
    "pre_pv_native_left_reason": np.dtype(np.int8),
    "pre_pv_native_right_reason": np.dtype(np.int8),
    "measurement_step": np.dtype(np.int64),
    "pre_pv_lane": np.dtype(np.int16),
    "pre_pv_native_eligible_left": np.dtype(np.int8),
    "pre_pv_native_eligible_right": np.dtype(np.int8),
    "pre_pv_native_eligible_count": np.dtype(np.int8),
    "native_phase_pv_native_eligible_left": np.dtype(np.int8),
    "native_phase_pv_native_eligible_right": np.dtype(np.int8),
    "native_phase_pv_native_eligible_count": np.dtype(np.int8),
    "long_pv_lane": np.dtype(np.int16),
    "long_pv_length": np.dtype(np.int32),
    "long_pv_speed_before": np.dtype(np.int8),
    "long_pv_speed_after": np.dtype(np.int8),
    "long_pv_leader_gap": np.dtype(np.int32),
    "long_pv_leader_type": np.dtype(np.int8),
    "long_pv_leader_speed": np.dtype(np.int8),
    "long_pv_leader_final_speed": np.dtype(np.int8),
    "long_pv_acceleration_gate": np.dtype(np.int8),
    "long_pv_v1": np.dtype(np.int8),
    "long_pv_prev_gap_s1": np.dtype(np.int64),
    "long_pv_prev_gap_sS": np.dtype(np.int64),
    "long_pv_cur_gap_s1": np.dtype(np.int64),
    "long_pv_cur_gap_sS": np.dtype(np.int64),
    "long_pv_eq9_slack": np.dtype(np.int32),
    "long_pv_eq9_tight": np.dtype(np.int8),
}
MECHANISM_UNAVAILABLE = (
    "native_stochastic_lane_change_draw",
    "native_direction_tie_break",
    "native_conflict_resolution_outcome",
    "longitudinal_s_q_brake_draws",
)

RESULT_KEYS = (
    "schema_version",
    "protocol",
    "task_id",
    "execution_id",
    "purpose",
    "split",
    "method",
    "architecture",
    "density_index",
    "replicate",
    "config",
    "seeds",
    "priority_id",
    "initial_av_ids",
    "av_assignment_applied",
    "requested_density",
    "requested_background_av_fraction",
    "realized_density",
    "realized_background_av_fraction",
    "initial_vehicle_count",
    "background_vehicle_count",
    "av_count",
    "background_av_count",
    "measurement_steps",
    "warmup_steps_completed",
    "steps_completed",
    "warm_state_sha256",
    "pv_assignment_state_sha256",
    "av_assignment_state_sha256",
    "initial_state_sha256",
    "final_state_sha256",
    "initial_rng_sha256",
    "final_rng_sha256",
    "trace_enabled",
    "trace_sha256",
    "model",
    "observer_profile",
    "observer_summary",
    "metrics",
    "control",
    "control_columns",
    "control_rejection_columns",
    "mechanism_notes",
    "mechanism_columns",
    "backend_requested",
    "backend_paths",
    "device",
    "timings",
    "resources",
    "deadline_utc",
)

METRIC_KEYS = (
    "measurement_steps",
    "vehicle_count",
    "priority_vehicle_count",
    "density",
    "flow_all",
    "flow_background",
    "flow_priority",
    "mean_speed_all",
    "mean_speed_background",
    "mean_speed_priority",
    "worst_priority_mean_speed",
    "mean_priority_time_loss",
    "worst_priority_time_loss",
    "stopped_fraction_all",
    "stopped_fraction_background",
    "stopped_fraction_priority",
    "lane_change_rate_all",
    "lane_change_rate_background",
    "lane_change_rate_priority",
    "completed_priority_laps",
    "mean_priority_lap_time",
    "rule_detections",
    "rule_requests",
    "rule_applied",
    "rule_rejected",
    "rule_blocked_no_safe_lane",
    "rule_cooldown_suppressed",
    "per_priority_vehicle",
)

TIMING_KEYS = (
    "warmup",
    "pv_assignment",
    "av_assignment",
    "model_load",
    "observation",
    "inference",
    "mask",
    "mechanism",
    "simulator",
    "observer",
    "metrics",
    "setup",
    "measurement",
    "total",
)


# ---------------------------------------------------------------------------
# Small strict helpers
# ---------------------------------------------------------------------------


def _fail(message: str) -> None:
    raise ValueError(message)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _integer(value, name: str, low: int | None = None, high: int | None = None) -> int:
    if not _is_int(value):
        _fail(f"{name} must be an integer")
    if low is not None and value < low:
        _fail(f"{name} must be >= {low}")
    if high is not None and value > high:
        _fail(f"{name} must be <= {high}")
    return value


def _number(value, name: str, *, low=None, high=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        _fail(f"{name} must be finite")
    if low is not None and value < low:
        _fail(f"{name} must be >= {low}")
    if high is not None and value > high:
        _fail(f"{name} must be <= {high}")
    return value


def _require_mapping(value, name: str) -> dict:
    if not isinstance(value, dict):
        _fail(f"{name} must be a mapping")
    return value


def _require_keys(value: dict, names, name: str) -> None:
    missing = [key for key in names if key not in value]
    if missing:
        _fail(f"{name} is missing keys: {missing}")


def _hex64(value, name: str) -> str:
    if not isinstance(value, str) or not _HEX64.match(value):
        _fail(f"{name} must be a lowercase SHA256 hex digest")
    return value


def _safe_name(value, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.match(value):
        _fail(f"{name} is not a safe path component")
    return value


def _close(actual, expected, name: str, *, rel_tol: float = 1e-12, abs_tol: float = 1e-12) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        _fail(f"{name} must be numeric")
    if not math.isfinite(actual) or not math.isclose(
        float(actual), float(expected), rel_tol=rel_tol, abs_tol=abs_tol
    ):
        _fail(f"{name} is inconsistent with the stored primary data")


def _nbytes(shape, dtype) -> int:
    count = 1
    for dimension in shape:
        count *= int(dimension)
    return count * int(np.dtype(dtype).itemsize)


def _numeric_dtype(value, name: str) -> np.dtype:
    if not isinstance(value, str) or not value:
        _fail(f"{name} dtype must be a string")
    try:
        dtype = np.dtype(value)
    except TypeError as exc:
        raise ValueError(f"{name} has an unsupported dtype") from exc
    if dtype.hasobject or dtype.kind not in "iuf":
        _fail(f"{name} has an unsupported dtype {dtype}")
    if dtype.itemsize not in (1, 2, 4, 8):
        _fail(f"{name} has an unsupported dtype width")
    return dtype


# ---------------------------------------------------------------------------
# Controlled JSON
# ---------------------------------------------------------------------------


def _json_pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            _fail("duplicate JSON key")
        result[key] = value
    return result


def _json_constant(value):
    _fail(f"invalid JSON constant {value!r}")


def _json_float(text):
    value = float(text)
    if not math.isfinite(value):
        _fail("JSON number is not finite")
    return value


def _strict_json(raw: bytes):
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("completion JSON is not valid UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_json_pairs,
            parse_constant=_json_constant,
            parse_float=_json_float,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"completion JSON is invalid: {exc}") from exc


def _read_regular(path: Path, limit: int, name: str) -> bytes:
    try:
        descriptor = os.open(
            os.fspath(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
    except OSError as exc:
        raise ValueError(f"{name} is not a safe readable regular file") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            _fail(f"{name} is not a regular file")
        if info.st_size > limit:
            _fail(f"{name} exceeds the declared size limit")
        blocks = []
        total = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > limit:
                _fail(f"{name} exceeds the declared size limit")
            blocks.append(block)
        return b"".join(blocks)
    finally:
        os.close(descriptor)


def _resolve_marker(path) -> tuple[Path, Path]:
    try:
        supplied = Path(os.fspath(path))
    except TypeError as exc:
        raise ValueError("episode path is not a filesystem path") from exc
    if os.path.islink(supplied):
        _fail("episode path must not be a symlink")
    if supplied.is_dir():
        attempt_dir = supplied
        marker = attempt_dir / MARKER_NAME
    elif supplied.name == MARKER_NAME:
        attempt_dir = supplied.parent
        marker = supplied
    else:
        _fail("episode path must be an attempt directory or completion.json")
    attempt_dir = Path(os.path.abspath(attempt_dir))
    marker = attempt_dir / MARKER_NAME
    task_dir = attempt_dir.parent
    if os.path.islink(attempt_dir) or os.path.islink(task_dir) or os.path.islink(marker):
        _fail("episode layout must not contain symlinks")
    return marker, attempt_dir


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(os.fspath(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    try:
        os.mkdir(os.fspath(path))
    except FileExistsError:
        pass
    if os.path.islink(path) or not path.is_dir():
        _fail(f"unsafe or non-directory publication path: {path.name}")


def _install_file(temporary: str, destination: Path, name: str) -> None:
    try:
        os.link(temporary, os.fspath(destination))
    except FileExistsError as exc:
        raise ValueError(f"{name} already exists; immutable attempts are not overwritten") from exc
    except OSError as exc:
        raise ValueError(f"cannot publish {name}: {exc}") from exc
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    _fsync_directory(destination.parent)


def _write_temp(directory: Path, data: bytes, name: str) -> str:
    descriptor, temporary = tempfile.mkstemp(
        dir=os.fspath(directory), prefix=".tmp-", suffix=".part"
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return temporary


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


def _validate_av_ids(ids, name: str) -> tuple[int, ...]:
    if not isinstance(ids, list):
        _fail(f"{name} must be a JSON list")
    values = []
    for value in ids:
        values.append(_integer(value, name, 0))
    if len(set(values)) != len(values):
        _fail(f"{name} must contain unique physical IDs")
    return tuple(values)


def _validate_model(model, task, method: str) -> None:
    science = task["science"]
    if method in schema.LEARNED_METHODS:
        if not isinstance(model, dict):
            _fail("learned method result.model must be a mapping")
        _require_keys(
            model,
            (
                "model_id",
                "training_line",
                "candidate_id",
                "env_steps",
                "architecture",
                "bundle_identity",
                "bundle_config_sha256",
            ),
            "result.model",
        )
        for name in ("model_id", "training_line", "candidate_id", "env_steps"):
            if model[name] != science[name]:
                _fail(f"result.model.{name} does not match the task")
        if model["architecture"] != method:
            _fail("result.model.architecture does not match the method")
        if model["bundle_identity"] != science["model_id"]:
            _fail("result.model.bundle_identity does not match the task model_id")
        _hex64(model["bundle_config_sha256"], "result.model.bundle_config_sha256")
        # The current kernel no longer reports bundle_weights_sha256, but an
        # older/extra provenance field is accepted and checked when present.
        if "bundle_weights_sha256" in model:
            _hex64(model["bundle_weights_sha256"], "result.model.bundle_weights_sha256")
    elif model is not None:
        _fail("fixed controllers must not report a model bundle")


def _validate_mechanism_notes(notes) -> None:
    notes = _require_mapping(notes, "result.mechanism_notes")
    _require_keys(
        notes,
        ("pre_pv_leader", "pre_pv_native_left_right", "unavailable"),
        "result.mechanism_notes",
    )
    if not isinstance(notes["pre_pv_leader"], str) or not notes["pre_pv_leader"]:
        _fail("mechanism_notes.pre_pv_leader must be a nonempty string")
    if (
        not isinstance(notes["pre_pv_native_left_right"], str)
        or not notes["pre_pv_native_left_right"]
    ):
        _fail("mechanism_notes.pre_pv_native_left_right must be a nonempty string")
    for name in ("native_phase", "longitudinal"):
        if not isinstance(notes.get(name), str) or not notes[name]:
            _fail(f"mechanism_notes.{name} must be a nonempty string")
    unavailable = notes["unavailable"]
    if not isinstance(unavailable, list):
        _fail("mechanism_notes.unavailable must be a list")
    for name in unavailable:
        if not isinstance(name, str) or not name:
            _fail("mechanism_notes.unavailable entries must be nonempty strings")
    if len(set(unavailable)) != len(unavailable):
        _fail("mechanism_notes.unavailable entries must be unique")
    missing = set(MECHANISM_UNAVAILABLE) - set(unavailable)
    if missing:
        _fail(
            "mechanism_notes.unavailable is missing: " + ", ".join(sorted(missing))
        )


def _validate_metrics_header(metrics, result, task) -> None:
    metrics = _require_mapping(metrics, "result.metrics")
    _require_keys(metrics, METRIC_KEYS, "result.metrics")
    _integer(metrics["measurement_steps"], "metrics.measurement_steps", 0)
    _integer(metrics["vehicle_count"], "metrics.vehicle_count", 1)
    _integer(metrics["priority_vehicle_count"], "metrics.priority_vehicle_count", 1, 1)
    _integer(metrics["completed_priority_laps"], "metrics.completed_priority_laps", 0)
    for key in (
        "rule_detections",
        "rule_requests",
        "rule_applied",
        "rule_rejected",
        "rule_blocked_no_safe_lane",
        "rule_cooldown_suppressed",
    ):
        _integer(metrics[key], f"metrics.{key}", 0)
    for key in (
        "density",
        "flow_all",
        "flow_background",
        "flow_priority",
        "mean_speed_all",
        "mean_speed_background",
        "mean_speed_priority",
        "worst_priority_mean_speed",
        "mean_priority_time_loss",
        "worst_priority_time_loss",
        "stopped_fraction_all",
        "stopped_fraction_background",
        "stopped_fraction_priority",
        "lane_change_rate_all",
        "lane_change_rate_background",
        "lane_change_rate_priority",
    ):
        _number(metrics[key], f"metrics.{key}")
    for key in ("density", "stopped_fraction_all", "stopped_fraction_background",
                "stopped_fraction_priority"):
        _number(metrics[key], f"metrics.{key}", low=0.0, high=1.0)
    for key in ("mean_speed_all", "mean_speed_background", "mean_speed_priority",
                "worst_priority_mean_speed"):
        _number(metrics[key], f"metrics.{key}", low=0.0, high=6.0)
    for key in ("flow_all", "flow_background", "flow_priority",
                "lane_change_rate_all", "lane_change_rate_background",
                "lane_change_rate_priority"):
        _number(metrics[key], f"metrics.{key}", low=0.0)
    for key in ("lane_change_rate_all", "lane_change_rate_background",
                "lane_change_rate_priority"):
        if metrics[key] > 1.0:
            _fail(f"metrics.{key} cannot exceed one lane change per vehicle-step")
    if metrics["mean_priority_lap_time"] is not None:
        _number(metrics["mean_priority_lap_time"], "metrics.mean_priority_lap_time", low=0.0)
    if metrics["measurement_steps"] != result["steps_completed"]:
        _fail("metrics.measurement_steps disagrees with steps_completed")
    if metrics["vehicle_count"] != result["initial_vehicle_count"]:
        _fail("metrics.vehicle_count disagrees with the actual population")
    if not math.isclose(
        metrics["density"], result["realized_density"], rel_tol=1e-12, abs_tol=1e-12
    ):
        _fail("metrics.density disagrees with realized_density")
    if not math.isclose(
        metrics["flow_all"],
        metrics["flow_background"] + metrics["flow_priority"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        _fail("metrics.flow_all must equal flow_background + flow_priority")
    vmax = max(1, int(result["config"]["vmax_priority"]))
    expected_loss = 1.0 - metrics["mean_speed_priority"] / vmax
    _close(metrics["mean_priority_time_loss"], expected_loss, "metrics.mean_priority_time_loss")
    _close(
        metrics["worst_priority_mean_speed"],
        metrics["mean_speed_priority"],
        "metrics.worst_priority_mean_speed",
    )
    _close(
        metrics["worst_priority_time_loss"],
        metrics["mean_priority_time_loss"],
        "metrics.worst_priority_time_loss",
    )

    per = _require_mapping(
        metrics["per_priority_vehicle"], "metrics.per_priority_vehicle"
    )
    if set(per) != {str(result["priority_id"])}:
        _fail("per_priority_vehicle must describe exactly the priority vehicle")
    pv = _require_mapping(per[str(result["priority_id"])], "per_priority_vehicle entry")
    for key in ("mean_speed", "time_loss", "distance", "completed_laps", "mean_lap_time"):
        if key not in pv:
            _fail(f"per_priority_vehicle is missing {key}")
    _number(pv["mean_speed"], "per_priority_vehicle.mean_speed", low=0.0, high=6.0)
    _number(pv["time_loss"], "per_priority_vehicle.time_loss", low=0.0, high=1.0)
    _number(pv["distance"], "per_priority_vehicle.distance", low=0.0)
    _integer(pv["completed_laps"], "per_priority_vehicle.completed_laps", 0)
    if pv["mean_lap_time"] is not None:
        _number(pv["mean_lap_time"], "per_priority_vehicle.mean_lap_time", low=0.0)
    _close(pv["mean_speed"], metrics["mean_speed_priority"], "per_priority_vehicle.mean_speed")
    _close(pv["time_loss"], metrics["mean_priority_time_loss"], "per_priority_vehicle.time_loss")
    if pv["completed_laps"] != metrics["completed_priority_laps"]:
        _fail("per_priority_vehicle.completed_laps disagrees with completed_priority_laps")
    if (
        (pv["mean_lap_time"] is None) != (metrics["mean_priority_lap_time"] is None)
    ):
        _fail("completed lap time availability is inconsistent")
    if pv["mean_lap_time"] is not None:
        _close(
            pv["mean_lap_time"],
            metrics["mean_priority_lap_time"],
            "per_priority_vehicle.mean_lap_time",
        )


def _validate_control_columns(columns):
    columns = _require_mapping(columns, "result.control_columns")
    if len(columns) > MAX_CONTROL_COLUMNS:
        _fail("too many control columns")
    if set(columns) != set(CONTROL_NAMES):
        _fail("v1 requires exactly all 33 mandatory control columns")
    indexed = []
    dtypes = set()
    for name, description in columns.items():
        if not isinstance(name, str) or not _ARRAY_NAME.match(name):
            _fail("control column names must be safe identifiers")
        if not isinstance(description, dict):
            _fail(f"control column {name} must have a descriptor")
        if set(description) != {"index", "dtype", "description"}:
            _fail(f"control column {name} has an invalid descriptor")
        index = _integer(description["index"], f"control column {name}.index", 0)
        dtype = _numeric_dtype(description["dtype"], f"control column {name}")
        if not isinstance(description["description"], str):
            _fail(f"control column {name} description must be a string")
        dtypes.add(dtype.str)
        indexed.append((index, name))
    if [index for index, _name in sorted(indexed)] != list(range(len(indexed))):
        _fail("control column indices must be exactly 0..C-1")
    if len(dtypes) != 1:
        _fail("all control columns must share one numeric dtype")
    names = [name for _index, name in sorted(indexed)]
    if len(set(names)) != len(names):
        _fail("control column names must be unique")
    dtype = np.dtype(dtypes.pop())
    if tuple(names) != CONTROL_NAMES or dtype != np.dtype(np.int64):
        _fail("v1 control columns must have the documented order and int64 dtype")
    return names, dtype


def _validate_rejection_columns(columns):
    columns = _require_mapping(columns, "result.control_rejection_columns")
    if len(columns) > 32:
        _fail("too many rejection columns")
    indexed = []
    for name, description in columns.items():
        if not isinstance(name, str) or not name:
            _fail("rejection column names must be nonempty strings")
        if not isinstance(description, dict):
            _fail(f"rejection column {name} must have a descriptor")
        if set(description) != {"index", "description"}:
            _fail(f"rejection column {name} has an invalid descriptor")
        index = _integer(description["index"], f"rejection column {name}.index", 0)
        if not isinstance(description["description"], str):
            _fail(f"rejection column {name} description must be a string")
        indexed.append((index, name))
    if [index for index, _name in sorted(indexed)] != list(range(len(indexed))):
        _fail("rejection column indices must be exactly 0..R-1")
    names = [name for _index, name in sorted(indexed)]
    if len(set(names)) != len(names):
        _fail("rejection column names must be unique")
    if tuple(names) != REJECTION_REASONS:
        _fail("v1 requires the five mandatory rejection columns in order")
    return names


def _validate_mechanism_columns(columns):
    columns = _require_mapping(columns, "result.mechanism_columns")
    if len(columns) > MAX_MECHANISM_COLUMNS:
        _fail("too many mechanism columns")
    if set(MECHANISM_DTYPES) - set(columns):
        _fail("v1 is missing mandatory mechanism columns")
    specs = {}
    for name, description in columns.items():
        if not isinstance(name, str) or not _ARRAY_NAME.match(name):
            _fail("mechanism column names must be safe identifiers")
        if not isinstance(description, dict):
            _fail(f"mechanism column {name} must have a descriptor")
        if set(description) != {"dtype", "description"}:
            _fail(f"mechanism column {name} has an invalid descriptor")
        dtype = _numeric_dtype(description["dtype"], f"mechanism column {name}")
        if not isinstance(description["description"], str) or not description["description"].strip():
            _fail(f"mechanism column {name} description must be a nonempty string")
        if name in ("control_trace", "control_rejection"):
            _fail(f"mechanism column {name} collides with a kernel array")
        if name in MECHANISM_DTYPES and dtype != MECHANISM_DTYPES[name]:
            _fail(
                f"mechanism column {name} must use dtype {MECHANISM_DTYPES[name]}"
            )
        specs[name] = dtype
    return specs


def _validate_control(control, result, task) -> None:
    control = _require_mapping(control, "result.control")
    required = (
        "interface",
        "steps",
        "active_av_decisions",
        "selected_left",
        "selected_right",
        "selected_stay",
        "explicit_stay_commands",
        "available_left_decisions",
        "available_right_decisions",
        "eligible_nonstay_decisions",
        "common_throttled_active_decisions",
        "private_throttled_active_decisions",
        "private_suppressed_left_opportunities",
        "common_suppressed_nonstop_requests",
        "common_cooldown_steps",
        "private_cooldown_steps",
        "requested_nonstay",
        "applied_nonstay",
        "rejected_nonstay",
        "emitted_nonstay",
        "legal_nonstay_requests",
        "applied_legal_nonstay_requests",
        "request_opportunity_use",
        "applied_opportunity_use",
        "opportunity_use_status",
        "applied_per_step",
        "applied_per_active_decision",
        "applied_per_request",
        "rejected_per_request",
        "rejection_reasons",
        "rejection_reason_names",
        "requested_front",
        "requested_centre",
        "requested_rear",
        "applied_front",
        "applied_centre",
        "applied_rear",
    )
    _require_keys(control, required, "result.control")
    method = result["method"]
    expected_interface = (
        "native" if method == "B1" else "rule" if method == "B2" else "av_action"
    )
    if control["interface"] != expected_interface:
        _fail("control.interface does not match the method")
    steps = result["steps_completed"]
    if _integer(control["steps"], "control.steps", 0) != steps:
        _fail("control.steps disagrees with steps_completed")
    _integer(control["active_av_decisions"], "control.active_av_decisions", 0)
    applicable = expected_interface == "av_action"
    for key in (
        "selected_left",
        "selected_right",
        "selected_stay",
        "explicit_stay_commands",
        "available_left_decisions",
        "available_right_decisions",
        "eligible_nonstay_decisions",
        "common_throttled_active_decisions",
        "common_suppressed_nonstop_requests",
        "legal_nonstay_requests",
        "applied_legal_nonstay_requests",
    ):
        value = control[key]
        if applicable:
            _integer(value, f"control.{key}", 0)
        elif value is not None:
            _fail(f"control.{key} must be null for a non-AV interface")
    private_applicable = method in HEURISTIC_C
    for key in (
        "private_throttled_active_decisions",
        "private_suppressed_left_opportunities",
    ):
        value = control[key]
        if private_applicable:
            _integer(value, f"control.{key}", 0)
        elif value is not None:
            _fail(f"control.{key} must be null for this method")
    if _integer(control["common_cooldown_steps"], "control.common_cooldown_steps", 0) != int(
        result["config"]["cooldown_steps"]
    ):
        _fail("control.common_cooldown_steps disagrees with the task")
    expected_private = HEURISTIC_C.get(method)
    if control["private_cooldown_steps"] != expected_private:
        _fail("control.private_cooldown_steps disagrees with the method")

    requested = _integer(control["requested_nonstay"], "control.requested_nonstay", 0)
    applied = _integer(control["applied_nonstay"], "control.applied_nonstay", 0)
    rejected = _integer(control["rejected_nonstay"], "control.rejected_nonstay", 0)
    emitted = _integer(control["emitted_nonstay"], "control.emitted_nonstay", 0)
    if requested != emitted:
        _fail("control.emitted_nonstay must equal requested_nonstay")
    if requested != applied + rejected:
        _fail("control conservation failed: emitted_nonstay != applied_nonstay + rejected_nonstay")
    if applied > requested:
        _fail("control.applied_nonstay exceeds requested_nonstay")

    status = control["opportunity_use_status"]
    if status not in ("defined", "undefined_zero_eligible", "not_applicable"):
        _fail("unsupported control.opportunity_use_status")
    if not applicable:
        if status != "not_applicable":
            _fail("non-AV methods must use opportunity_use_status='not_applicable'")
        if control["request_opportunity_use"] is not None or control["applied_opportunity_use"] is not None:
            _fail("opportunity-use ratios must be null for non-AV methods")
    else:
        eligible = control["eligible_nonstay_decisions"]
        legal = control["legal_nonstay_requests"]
        applied_legal = control["applied_legal_nonstay_requests"]
        if eligible > control["available_left_decisions"] + control["available_right_decisions"]:
            _fail("eligible_nonstay_decisions exceeds available left/right opportunities")
        if legal > eligible:
            _fail("legal_nonstay_requests exceeds eligible_nonstay_decisions")
        if applied_legal > legal:
            _fail("applied_legal_nonstay_requests exceeds legal_nonstay_requests")
        if control["selected_left"] + control["selected_right"] + control["selected_stay"] != control["active_av_decisions"]:
            _fail("selected action counts do not sum to active_av_decisions")
        if control["selected_stay"] > control["explicit_stay_commands"]:
            _fail("explicit_stay_commands cannot be smaller than selected_stay")
        if control["common_throttled_active_decisions"] > control["active_av_decisions"]:
            _fail("common_throttled_active_decisions exceeds active decisions")
        if private_applicable and control["private_throttled_active_decisions"] > control["active_av_decisions"]:
            _fail("private_throttled_active_decisions exceeds active decisions")
        if eligible == 0:
            if status != "undefined_zero_eligible":
                _fail("zero eligible decisions must be marked undefined_zero_eligible")
            if control["request_opportunity_use"] is not None or control["applied_opportunity_use"] is not None:
                _fail("zero-eligible opportunity-use ratios must be null")
        else:
            if status != "defined":
                _fail("positive eligible decisions must be marked defined")
            _close(
                control["request_opportunity_use"],
                legal / eligible,
                "control.request_opportunity_use",
            )
            _close(
                control["applied_opportunity_use"],
                applied_legal / eligible,
                "control.applied_opportunity_use",
            )
    if control["active_av_decisions"] > result["background_vehicle_count"] * steps:
        _fail("active_av_decisions exceeds the available background exposure")

    _close(control["applied_per_step"], applied / steps, "control.applied_per_step")
    if control["active_av_decisions"] == 0:
        if control["applied_per_active_decision"] is not None:
            _fail("applied_per_active_decision must be null at zero denominator")
    else:
        _close(
            control["applied_per_active_decision"],
            applied / control["active_av_decisions"],
            "control.applied_per_active_decision",
        )
    if requested == 0:
        if control["applied_per_request"] is not None or control["rejected_per_request"] is not None:
            _fail("request ratios must be null at zero requests")
    else:
        _close(control["applied_per_request"], applied / requested, "control.applied_per_request")
        _close(control["rejected_per_request"], rejected / requested, "control.rejected_per_request")

    reasons = _require_mapping(control["rejection_reasons"], "control.rejection_reasons")
    for name, value in reasons.items():
        if not isinstance(name, str) or not name:
            _fail("rejection_reasons keys must be nonempty strings")
        _integer(value, f"control.rejection_reasons[{name}]", 0)
    names = control["rejection_reason_names"]
    if not isinstance(names, list) or not 1 <= len(names) <= 32:
        _fail("control.rejection_reason_names must be a nonempty bounded list")
    if len(set(names)) != len(names):
        _fail("control.rejection_reason_names must be unique")
    for name in names:
        if not isinstance(name, str) or not name:
            _fail("rejection reason names must be nonempty strings")
    if sum(reasons.values()) != rejected:
        _fail("rejection_reasons totals do not equal rejected_nonstay")

    zones_requested = (
        control["requested_front"] + control["requested_centre"] + control["requested_rear"]
    )
    zones_applied = (
        control["applied_front"] + control["applied_centre"] + control["applied_rear"]
    )
    for key in ("requested_front", "requested_centre", "requested_rear",
                "applied_front", "applied_centre", "applied_rear"):
        _integer(control[key], f"control.{key}", 0)
    if zones_requested != requested:
        _fail("requested zone totals do not equal requested_nonstay")
    if zones_applied != applied:
        _fail("applied zone totals do not equal applied_nonstay")


def _validate_observer_summary_header(summary, result, task, execution) -> None:
    summary = _require_mapping(summary, "result.observer_summary")
    required = (
        "schema_version",
        "profile",
        "measurement_steps",
        "road_length",
        "num_lanes",
        "pv_id",
        "av_ids",
        "window_ids",
        "primary_window",
        "regions",
        "lanes",
        "groups",
        "windows",
        "global_background",
        "global_all",
        "pv",
        "speed_floor_steps",
        "speed_floor_fraction",
        "spatial_cell_time_per_bin_lane",
        "spatial_extent",
        "spatial_extent_covers_all_windows",
        "realized_vehicle_count",
        "realized_density",
        "realized_background_av_fraction",
    )
    _require_keys(summary, required, "result.observer_summary")
    if not _is_int(summary["schema_version"]) or summary["schema_version"] != 1:
        _fail("unsupported observer summary schema_version")
    if summary["profile"] != result["observer_profile"]:
        _fail("observer summary profile disagrees with observer_profile")
    config = result["config"]
    if summary["measurement_steps"] != result["steps_completed"]:
        _fail("observer summary measurement_steps disagrees with the episode")
    if summary["road_length"] != config["road_length"]:
        _fail("observer summary road_length disagrees with the task")
    if summary["num_lanes"] != config["num_lanes"]:
        _fail("observer summary num_lanes disagrees with the task")
    if summary["pv_id"] != result["priority_id"]:
        _fail("observer summary pv_id disagrees with priority_id")
    av_ids = _validate_av_ids(summary["av_ids"], "observer_summary.av_ids")
    if tuple(sorted(av_ids)) != av_ids:
        _fail("observer summary av_ids must be sorted")
    if set(av_ids) != set(result["initial_av_ids"]):
        _fail("observer summary av_ids disagree with initial_av_ids")
    if summary["window_ids"] != list(WINDOW_IDS):
        _fail("observer summary does not contain the canonical 12 windows")
    if summary["primary_window"] != PRIMARY_WINDOW:
        _fail("observer summary primary_window is not fixed_front_2")
    if summary["regions"] != REGIONS:
        _fail("observer summary regions are not canonical")
    lanes = list(range(int(config["num_lanes"]))) + ["all"]
    if summary["lanes"] != lanes:
        _fail("observer summary lanes are not canonical")
    if summary["groups"] != GROUPS:
        _fail("observer summary groups are not canonical")
    if not isinstance(summary["windows"], dict):
        _fail("observer summary windows must be a mapping")
    if not isinstance(summary["global_background"], dict):
        _fail("observer summary global_background must be a mapping")
    if not isinstance(summary["global_all"], dict):
        _fail("observer summary global_all must be a mapping")
    if not isinstance(summary["pv"], dict):
        _fail("observer summary pv must be a mapping")
    _integer(summary["speed_floor_steps"], "observer_summary.speed_floor_steps", 0)
    _number(summary["speed_floor_fraction"], "observer_summary.speed_floor_fraction", low=0.0, high=1.0)
    if summary["spatial_cell_time_per_bin_lane"] != result["steps_completed"]:
        _fail("spatial cell-time exposure disagrees with the measurement horizon")
    if summary["spatial_extent"] != [-36, 54]:
        _fail("observer summary must use the corrected spatial extent -36..54")
    if summary["spatial_extent_covers_all_windows"] is not True:
        _fail("observer summary must declare complete spatial coverage")
    if summary["realized_vehicle_count"] != result["initial_vehicle_count"]:
        _fail("observer realized_vehicle_count disagrees with the result")
    _close(summary["realized_density"], result["realized_density"], "observer_summary.realized_density")
    fraction = summary["realized_background_av_fraction"]
    expected_fraction = result["realized_background_av_fraction"]
    if (fraction is None) != (expected_fraction is None):
        _fail("realized_background_av_fraction availability disagrees")
    if fraction is not None:
        _close(fraction, expected_fraction, "observer_summary.realized_background_av_fraction")


def _validate_result(result, task, execution) -> None:
    result = _require_mapping(result, "result")
    _require_keys(result, RESULT_KEYS, "result")
    encoded = schema.canonical_json(result)
    if len(encoded) > MAX_RESULT_JSON_BYTES:
        _fail("result JSON exceeds the declared size limit")
    science = task["science"]
    config = science["config"]
    if not _is_int(result["schema_version"]) or result["schema_version"] != schema.SCHEMA_VERSION:
        _fail("unsupported result schema_version")
    if result["protocol"] != schema.PROTOCOL:
        _fail("result protocol does not match the task")
    if result["task_id"] != task["task_id"]:
        _fail("result.task_id does not match the task")
    if result["execution_id"] != execution["execution_id"]:
        _fail("result.execution_id does not match the execution")
    if result["purpose"] != science["purpose"]:
        _fail("result.purpose does not match the task")
    if result["split"] != science["split"]:
        _fail("result.split does not match the task")
    if result["method"] != science["method"]:
        _fail("result.method does not match the task")
    if result["architecture"] != (
        science["method"] if science["method"] in schema.LEARNED_METHODS else None
    ):
        _fail("result.architecture does not match the method")
    for name in ("model_id", "training_line", "candidate_id", "env_steps"):
        if name in result and result[name] != science[name]:
            _fail(f"result.{name} does not match the immutable task")
    if result["density_index"] != science["density_index"]:
        _fail("result.density_index does not match the task")
    if result["replicate"] != science["replicate"]:
        _fail("result.replicate does not match the task")
    if schema.canonical_json(result["config"]) != schema.canonical_json(config):
        _fail("result.config does not match the immutable task config")
    if schema.canonical_json(result["seeds"]) != schema.canonical_json(science["seeds"]):
        _fail("result.seeds do not match the immutable task seeds")
    _integer(result["priority_id"], "result.priority_id", 0)
    av_ids = _validate_av_ids(result["initial_av_ids"], "result.initial_av_ids")
    if result["priority_id"] in av_ids:
        _fail("the priority vehicle must not be an AV")
    if not isinstance(result["av_assignment_applied"], bool):
        _fail("result.av_assignment_applied must be boolean")
    if result["av_assignment_applied"] != (result["method"] != "B1"):
        _fail("av_assignment_applied disagrees with the method")
    _close(result["requested_density"], config["density"], "result.requested_density")
    _close(
        result["requested_background_av_fraction"],
        config["av_fraction"],
        "result.requested_background_av_fraction",
    )
    count = _integer(result["initial_vehicle_count"], "result.initial_vehicle_count", 1)
    background = _integer(result["background_vehicle_count"], "result.background_vehicle_count", 0)
    if background != count - 1:
        _fail("background_vehicle_count must equal initial_vehicle_count - 1")
    av_count = _integer(result["av_count"], "result.av_count", 0)
    if av_count != _integer(result["background_av_count"], "result.background_av_count", 0):
        _fail("av_count and background_av_count must agree")
    if av_count != len(av_ids):
        _fail("av_count disagrees with initial_av_ids")
    if av_count > background:
        _fail("av_count exceeds background_vehicle_count")
    if result["method"] == "B1" and av_count != 0:
        _fail("B1 must not assign AVs")
    capacity = int(config["num_lanes"]) * int(config["road_length"])
    _close(
        result["realized_density"],
        count / capacity,
        "result.realized_density",
    )
    if background == 0:
        if result["realized_background_av_fraction"] is not None:
            _fail("realized_background_av_fraction must be null without background")
    else:
        _close(
            result["realized_background_av_fraction"],
            av_count / background,
            "result.realized_background_av_fraction",
        )
        if result["method"] == "B1" and result["realized_background_av_fraction"] != 0.0:
            _fail("B1 must have zero realized AV fraction")
    steps = _integer(result["steps_completed"], "result.steps_completed", 1)
    if _integer(result["measurement_steps"], "result.measurement_steps", 1) != config["measure_steps"]:
        _fail("result.measurement_steps disagrees with the task")
    if steps != config["measure_steps"]:
        _fail("a complete result must contain the full measurement horizon")
    if (
        _integer(result["warmup_steps_completed"], "result.warmup_steps_completed", 0)
        != config["warmup_steps"]
    ):
        _fail("warmup_steps_completed disagrees with the task")
    _hex64(result["warm_state_sha256"], "result.warm_state_sha256")
    _hex64(result["pv_assignment_state_sha256"], "result.pv_assignment_state_sha256")
    if result["method"] == "B1":
        if result["av_assignment_state_sha256"] is not None:
            _fail("B1 av_assignment_state_sha256 must be null")
    else:
        _hex64(result["av_assignment_state_sha256"], "result.av_assignment_state_sha256")
    for key in (
        "initial_state_sha256",
        "final_state_sha256",
        "initial_rng_sha256",
        "final_rng_sha256",
    ):
        _hex64(result[key], f"result.{key}")
    if not isinstance(result["trace_enabled"], bool):
        _fail("result.trace_enabled must be boolean")
    if result["trace_enabled"]:
        _hex64(result["trace_sha256"], "result.trace_sha256")
    elif result["trace_sha256"] is not None:
        _fail("trace_sha256 must be null when tracing is disabled")
    _validate_model(result["model"], task, result["method"])
    profile = execution["profile"]["observer"]
    if result["observer_profile"] != profile:
        _fail("result.observer_profile disagrees with the execution profile")
    if profile == "off":
        if result["observer_summary"] is not None:
            _fail("observer_summary must be null for profile off")
    else:
        _validate_observer_summary_header(
            result["observer_summary"], result, task, execution
        )
    _validate_metrics_header(result["metrics"], result, task)
    _validate_control(result["control"], result, task)
    _validate_control_columns(result["control_columns"])
    if "control_rejection_columns" in result:
        rejection_names = _validate_rejection_columns(
            result["control_rejection_columns"]
        )
        if rejection_names != result["control"]["rejection_reason_names"]:
            _fail("control_rejection_columns disagree with rejection_reason_names")
    _validate_mechanism_columns(result["mechanism_columns"])
    if "mechanism_notes" in result:
        _validate_mechanism_notes(result["mechanism_notes"])

    if result["backend_requested"] != config["backend"]:
        _fail("backend_requested disagrees with the task config")
    paths = result["backend_paths"]
    if not isinstance(paths, list) or not 1 <= len(paths) <= 16:
        _fail("backend_paths must be a nonempty bounded list")
    for path in paths:
        if not isinstance(path, str) or not _BACKEND_NAME.match(path):
            _fail("backend_paths contains an unsafe backend name")
        if config["backend"] == "optimized" and "reference" in path:
            _fail("optimized execution records a reference fallback")
        if config["backend"] == "reference" and "optimized" in path:
            _fail("reference execution records an optimized backend")
    device = _require_mapping(result["device"], "result.device")
    if not isinstance(device.get("requested"), str) or not device["requested"]:
        _fail("result.device.requested must be a nonempty string")
    if not isinstance(device.get("torch_imported"), bool):
        _fail("result.device.torch_imported must be boolean")
    if not isinstance(device.get("type"), str) or not device["type"]:
        _fail("result.device.type must be a nonempty string")
    timings = _require_mapping(result["timings"], "result.timings")
    _require_keys(timings, TIMING_KEYS, "result.timings")
    for key in TIMING_KEYS:
        _number(timings[key], f"timings.{key}", low=0.0)
    resources = _require_mapping(result["resources"], "result.resources")
    _require_keys(
        resources,
        ("cpu_seconds", "wall_seconds", "process_peak_rss_kib"),
        "result.resources",
    )
    _number(resources["cpu_seconds"], "resources.cpu_seconds", low=0.0)
    _number(resources["wall_seconds"], "resources.wall_seconds", low=0.0)
    _integer(resources["process_peak_rss_kib"], "resources.process_peak_rss_kib", 0)
    if result["deadline_utc"] is not None:
        if not isinstance(result["deadline_utc"], str) or len(result["deadline_utc"]) > 64:
            _fail("deadline_utc must be null or a short ISO8601 string")


# ---------------------------------------------------------------------------
# Expected arrays and archive handling
# ---------------------------------------------------------------------------


def _observer_specs(steps: int, lanes: int, profile: str) -> dict:
    if profile == "off":
        return {}
    specs = {
        "window_hist": ((steps, 12, 3, lanes, 2, 6), np.dtype(np.uint16)),
        "initial_window_hist": ((12, 3, lanes, 2, 6), np.dtype(np.uint16)),
        "global_hist": ((steps, lanes, 2, 6), np.dtype(np.uint16)),
        "initial_global_hist": ((lanes, 2, 6), np.dtype(np.uint16)),
        "window_bounds": ((steps + 1, 12, 2), np.dtype(np.int16)),
        "window_transitions": ((steps, 12, 3, lanes, 2, 4), np.dtype(np.uint16)),
        "global_lane_changes": ((steps, lanes, 2), np.dtype(np.uint16)),
        "step": ((steps + 1,), np.dtype(np.int64)),
        "pv_speed": ((steps + 1,), np.dtype(np.uint8)),
        "pv_lane": ((steps + 1,), np.dtype(np.uint8)),
        "pv_pos": ((steps + 1,), np.dtype(np.int64)),
        "pv_path": ((steps + 1,), np.dtype(np.int64)),
        "spatial_hist": ((91, lanes, 2, 6), np.dtype(np.uint64)),
        "spatial_edges": ((92,), np.dtype(np.int16)),
    }
    if profile == "mechanism":
        specs.update(
            {
                "local_occupancy": ((steps, 41, lanes), np.dtype(np.uint8)),
                "local_speed": ((steps, 41, lanes), np.dtype(np.int8)),
                "local_type": ((steps, 41, lanes), np.dtype(np.int8)),
            }
        )
    return specs


def _expected_specs(task, execution, result) -> tuple[dict, dict]:
    steps = int(result["steps_completed"])
    lanes = int(result["config"]["num_lanes"])
    profile = execution["profile"]["observer"]
    required = dict(_observer_specs(steps, lanes, profile))
    control_names, control_dtype = _validate_control_columns(result["control_columns"])
    if control_names:
        required["control_trace"] = ((steps, len(control_names)), control_dtype)
    mechanism = _validate_mechanism_columns(result["mechanism_columns"])
    for name, dtype in mechanism.items():
        if name in required:
            _fail(f"mechanism column {name} collides with another array")
        required[name] = ((steps,), dtype)
    optional = {}
    names = result["control"]["rejection_reason_names"]
    required["control_rejection"] = ((steps, len(names)), np.dtype(np.int64))
    total = sum(_nbytes(shape, dtype) for shape, dtype in required.values())
    if total > MAX_DECODED_BYTES:
        _fail("declared arrays exceed the hard decoded-byte ceiling")
    return required, optional


def _max_archive_size(required, optional) -> int:
    total = sum(_nbytes(shape, dtype) for shape, dtype in required.values())
    total += sum(_nbytes(shape, dtype) for shape, dtype in optional.values())
    return min(MAX_ARCHIVE_BYTES, total + 4 * 1024 * 1024)


def _validate_arrays(arrays, required, optional, task, result) -> None:
    if not isinstance(arrays, dict):
        _fail("arrays must be a mapping of names to NumPy arrays")
    if len(arrays) > MAX_ARRAYS:
        _fail("too many arrays")
    missing = [name for name in required if name not in arrays]
    if missing:
        _fail(f"missing required arrays: {sorted(missing)}")
    allowed = set(required) | set(optional)
    unknown = sorted(set(arrays) - allowed)
    if unknown:
        _fail(f"undeclared arrays: {unknown}")
    if "control_rejection" in arrays and "control_rejection_columns" not in result:
        _fail("control_rejection is present but its declared columns are missing")
    if "control_rejection_columns" in result and "control_rejection" not in arrays:
        _fail("control_rejection_columns are declared but control_rejection is missing")
    if result["control"]["rejected_nonstay"] > 0 and "control_rejection" not in arrays:
        _fail("rejected requests require a control_rejection array")
    for name, array in arrays.items():
        if not _ARRAY_NAME.match(name):
            _fail(f"unsafe array name {name!r}")
        if not isinstance(array, np.ndarray):
            _fail(f"array {name} is not a NumPy array")
        shape, dtype = required.get(name, optional.get(name))
        if array.dtype != dtype:
            _fail(f"array {name} must have dtype {dtype}, got {array.dtype}")
        if tuple(array.shape) != tuple(shape):
            _fail(f"array {name} must have shape {shape}, got {array.shape}")
        if array.dtype.hasobject or array.dtype.kind not in "iuf":
            _fail(f"array {name} has an unsupported dtype")
        if not array.flags.c_contiguous:
            _fail(f"array {name} must be C-contiguous")
    total = sum(int(array.nbytes) for array in arrays.values())
    if total > MAX_DECODED_BYTES:
        _fail("arrays exceed the hard decoded-byte ceiling")


def _validate_sidecar(sidecar, required, optional, sizes: dict) -> set:
    sidecar = _require_mapping(sidecar, "completion.sidecar")
    if set(sidecar) != {"path", "size", "sha256", "arrays"}:
        _fail("completion.sidecar has unknown or missing keys")
    if sidecar["path"] != SIDECAR_NAME:
        _fail("completion.sidecar.path must be exactly 'arrays.npz'")
    _integer(sidecar["size"], "sidecar.size", 0)
    _hex64(sidecar["sha256"], "sidecar.sha256")
    descriptors = _require_mapping(sidecar["arrays"], "sidecar.arrays")
    allowed = set(required) | set(optional)
    if not set(required).issubset(set(descriptors)):
        _fail("sidecar is missing required arrays")
    unknown = sorted(set(descriptors) - allowed)
    if unknown:
        _fail(f"sidecar declares undeclared arrays: {unknown}")
    total = 0
    for name, descriptor in descriptors.items():
        if name not in sizes:
            _fail(f"sidecar declares an unknown array {name}")
        descriptor = _require_mapping(descriptor, f"sidecar.arrays[{name}]")
        if set(descriptor) != {"dtype", "shape", "nbytes"}:
            _fail(f"sidecar array {name} has an invalid descriptor")
        shape, dtype = sizes[name]
        if not isinstance(descriptor["shape"], list) or any(
            not _is_int(dimension) or dimension < 0 for dimension in descriptor["shape"]
        ):
            _fail(f"sidecar array {name} has an invalid shape")
        if list(descriptor["shape"]) != list(shape):
            _fail(f"sidecar array {name} shape disagrees with the declared schema")
        if _numeric_dtype(descriptor["dtype"], f"sidecar array {name}") != dtype:
            _fail(f"sidecar array {name} dtype disagrees with the declared schema")
        nbytes = _integer(descriptor["nbytes"], f"sidecar array {name}.nbytes", 0)
        if nbytes != _nbytes(shape, dtype):
            _fail(f"sidecar array {name} byte count is inconsistent")
        total += nbytes
    if total > MAX_DECODED_BYTES:
        _fail("sidecar decoded arrays exceed the hard byte ceiling")
    return set(descriptors)


def _load_npz(path: Path, specs: dict, max_file: int) -> dict:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError("arrays sidecar is missing") from exc
    if size > max_file:
        _fail("arrays sidecar exceeds the declared archive size limit")
    try:
        archive = zipfile.ZipFile(os.fspath(path))
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("arrays sidecar is not a valid NPZ archive") from exc
    arrays = {}
    with archive:
        members = archive.infolist()
        if len(members) != len(specs):
            _fail("archive member count disagrees with the declared arrays")
        seen = set()
        for member in members:
            filename = member.filename
            if not filename.endswith(".npy") or "/" in filename or "\\" in filename:
                _fail("archive contains an unsafe member path")
            name = filename[:-4]
            if name in seen:
                _fail("archive contains duplicate members")
            seen.add(name)
            if name not in specs:
                _fail(f"archive contains an undeclared member {name!r}")
            if member.flag_bits & 1:
                _fail("encrypted archive members are not supported")
            if member.compress_type != zipfile.ZIP_STORED:
                _fail("only uncompressed NPZ members are supported")
            shape, dtype = specs[name]
            expected_bytes = _nbytes(shape, dtype)
            if member.file_size < 0 or member.file_size > expected_bytes + MAX_HEADER_BYTES:
                _fail(f"archive member {name} has an impossible size")
            with archive.open(member) as stream:
                try:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        header = np.lib.format.read_array_header_1_0(
                            stream, max_header_size=MAX_HEADER_BYTES
                        )
                    elif version == (2, 0):
                        header = np.lib.format.read_array_header_2_0(
                            stream, max_header_size=MAX_HEADER_BYTES
                        )
                    else:
                        _fail("unsupported NPY header version")
                    member_shape, fortran, member_dtype = header
                except ValueError:
                    raise
                except Exception as exc:
                    raise ValueError(f"archive member {name} has an invalid NPY header") from exc
                if fortran:
                    _fail(f"archive member {name} must be C-order")
                if member_dtype.hasobject or member_dtype.kind not in "iuf":
                    _fail(f"archive member {name} has an unsupported dtype")
                if tuple(member_shape) != tuple(shape) or member_dtype != dtype:
                    _fail(f"archive member {name} header disagrees with the declared schema")
                payload = _nbytes(member_shape, member_dtype)
                if stream.tell() + payload != member.file_size:
                    _fail(f"archive member {name} byte length is inconsistent")
        if seen != set(specs):
            _fail("archive member set disagrees with the declared arrays")
        if sum(member.file_size for member in members) > MAX_DECODED_BYTES + len(members) * MAX_HEADER_BYTES:
            _fail("archive expands beyond the hard decoded-byte ceiling")
        try:
            with np.load(os.fspath(path), allow_pickle=False) as loaded:
                if set(loaded.files) != set(specs):
                    _fail("NPZ file names disagree with the declared arrays")
                for name, (shape, dtype) in specs.items():
                    array = loaded[name]
                    if array.dtype != dtype or tuple(array.shape) != tuple(shape):
                        _fail(f"loaded array {name} disagrees with its header")
                    if not array.flags.c_contiguous:
                        array = np.ascontiguousarray(array)
                    if array.dtype.kind == "f" and not bool(np.all(np.isfinite(array))):
                        _fail(f"array {name} contains non-finite values")
                    arrays[name] = array.copy()
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"cannot read NPZ arrays: {exc}") from exc
    return arrays


def _write_npz(path: Path, arrays: dict) -> None:
    with zipfile.ZipFile(
        os.fspath(path), "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            with archive.open(info, "w") as stream:
                np.lib.format.write_array(
                    stream,
                    np.ascontiguousarray(arrays[name]),
                    allow_pickle=False,
                )
    descriptor = os.open(os.fspath(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _same_arrays(left: dict, right: dict) -> bool:
    if set(left) != set(right):
        return False
    for name in left:
        if left[name].dtype != right[name].dtype:
            return False
        if tuple(left[name].shape) != tuple(right[name].shape):
            return False
        if not np.array_equal(left[name], right[name]):
            return False
    return True


# ---------------------------------------------------------------------------
# Independent numerical validation
# ---------------------------------------------------------------------------


def _ratio(num, den):
    num, den = np.broadcast_arrays(np.asarray(num), np.asarray(den))
    output = np.empty(num.shape, dtype=object)
    output.fill(None)
    defined = den != 0
    output[defined] = num[defined].astype(np.float64) / den[defined]
    return output.tolist()


def _reduce_metrics(hist, area, steps: int, *, deficit: bool) -> dict:
    n = hist.sum(axis=-1, dtype=np.uint64)
    speeds = np.arange(hist.shape[-1], dtype=np.uint64)
    s = (hist * speeds).sum(axis=-1, dtype=np.uint64)
    z = hist[..., 0].astype(np.uint64)
    area = np.broadcast_to(np.asarray(area, dtype=np.uint64), n.shape)
    totals = {
        "N": n.sum(axis=0, dtype=np.uint64),
        "S": s.sum(axis=0, dtype=np.uint64),
        "Z": z.sum(axis=0, dtype=np.uint64),
        "A": area.sum(axis=0, dtype=np.uint64),
    }
    empty = (n == 0).sum(axis=0, dtype=np.uint64)
    N, S, Z, A = (totals[key] for key in ("N", "S", "Z", "A"))
    output = {key: value.tolist() for key, value in totals.items()}
    output.update(
        vehicle_time=N.tolist(),
        cell_time=A.tolist(),
        defined_steps=(steps - empty).tolist(),
        empty_steps=empty.tolist(),
        pooled_speed=_ratio(S, N),
        pooled_stopped_fraction=_ratio(Z, N),
        cell_time_density=_ratio(N, A),
        cell_time_flow=_ratio(S, A),
        time_mean_density=_ratio((n / area).sum(axis=0), steps),
        time_mean_flow=_ratio((s / area).sum(axis=0), steps),
        empty_fraction=_ratio(empty, steps),
    )
    if deficit:
        deficit_sum = 5 * N.astype(np.int64) - S.astype(np.int64)
        output.update(
            speed_deficit_sum=deficit_sum.tolist(),
            speed_deficit=_ratio(deficit_sum, N),
            normalized_local_delay=_ratio(deficit_sum, 5 * N),
        )
    return output


def _compare_saved(saved, expected, path: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(saved, dict):
            _fail(f"{path} must be a mapping")
        for key, value in expected.items():
            if key not in saved:
                _fail(f"{path}.{key} is missing")
            _compare_saved(saved[key], value, f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(saved, list) or len(saved) != len(expected):
            _fail(f"{path} must be a list of length {len(expected)}")
        for index, value in enumerate(expected):
            _compare_saved(saved[index], value, f"{path}[{index}]")
        return
    if expected is None:
        if saved is not None:
            _fail(f"{path} must be null")
        return
    if isinstance(expected, bool):
        if saved is not expected:
            _fail(f"{path} disagrees with the independent recomputation")
        return
    if isinstance(expected, int):
        if not _is_int(saved) or saved != expected:
            _fail(f"{path} disagrees with the independent recomputation")
        return
    if isinstance(expected, float):
        if isinstance(saved, bool) or not isinstance(saved, (int, float)):
            _fail(f"{path} must be numeric")
        if not math.isfinite(saved) or not math.isclose(
            float(saved), expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            _fail(f"{path} disagrees with the independent recomputation")
        return
    if saved != expected:
        _fail(f"{path} disagrees with the independent recomputation")


def _expected_bounds(pv_speed: np.ndarray) -> np.ndarray:
    rows = len(pv_speed)
    output = np.empty((rows, 12, 2), dtype=np.int16)
    for index, speed in enumerate(pv_speed.tolist()):
        values = []
        for d in (5, max(int(speed), 1)):
            for scale in (1, 2, 3):
                for low, high in (
                    (-2 * scale * d, 2 * scale * d),
                    (-scale * d, 3 * scale * d),
                ):
                    values.append((low, high))
        output[index] = values
    return output


def _pv_summary(arrays: dict, road_length: int) -> dict:
    steps = len(arrays["pv_speed"]) - 1
    speed = arrays["pv_speed"][1:]
    distance = int(arrays["pv_path"][-1]) - int(arrays["pv_path"][0])
    cumulative = arrays["pv_path"] - arrays["pv_path"][0]
    crossings = (np.flatnonzero(np.diff(cumulative // road_length)) + 1).tolist()
    lap_times = np.diff([0] + crossings).tolist()
    stopped = int(np.count_nonzero(speed == 0))
    lane_changes = int(np.count_nonzero(np.diff(arrays["pv_lane"].astype(np.int16))))
    return dict(
        mean_speed=distance / steps,
        distance=distance,
        normalized_time_loss=1 - distance / (6 * steps),
        stopped_steps=stopped,
        stopped_fraction=stopped / steps,
        moving_steps=steps - stopped,
        moving_mean_speed=distance / (steps - stopped) if steps > stopped else None,
        lane_changes=lane_changes,
        lane_change_rate=lane_changes / steps,
        completed_laps=len(crossings),
        lap_completion_steps=crossings,
        completed_lap_times=lap_times,
        mean_completed_lap_time=sum(lap_times) / len(lap_times) if lap_times else None,
        remaining_distance=distance % road_length,
        no_completed_laps=not crossings,
    )


def _recompute_observer_summary(arrays: dict, task, result) -> dict:
    config = result["config"]
    steps = int(result["steps_completed"])
    lanes = int(config["num_lanes"])
    road = int(config["road_length"])
    bounds = arrays["window_bounds"].astype(np.int64)
    low = bounds[1:, :, 0]
    high = bounds[1:, :, 1]
    widths = np.stack(
        (-low, np.ones_like(low), high, high - low + 1), axis=-1
    ).astype(np.uint64)
    lane_factor = np.asarray([1] * lanes + [lanes], dtype=np.uint64)
    window_area = widths[..., None, None] * lane_factor[None, None, None, :, None]

    hist = arrays["window_hist"].astype(np.uint64)
    scoped = np.concatenate(
        (hist, hist.sum(axis=2, keepdims=True, dtype=np.uint64)), axis=2
    )
    scoped = np.concatenate(
        (scoped, scoped.sum(axis=-3, keepdims=True, dtype=np.uint64)), axis=-3
    )
    scoped = np.concatenate(
        (scoped, scoped.sum(axis=-2, keepdims=True, dtype=np.uint64)), axis=-2
    )
    windows = _reduce_metrics(scoped, window_area, steps, deficit=True)

    global_hist = arrays["global_hist"].astype(np.uint64)
    global_scoped = np.concatenate(
        (global_hist, global_hist.sum(axis=-2, keepdims=True, dtype=np.uint64)),
        axis=-2,
    )
    global_scoped = np.concatenate(
        (global_scoped, global_scoped.sum(axis=-3, keepdims=True, dtype=np.uint64)),
        axis=-3,
    )
    global_area = road * lane_factor[None, :, None]
    global_metrics = _reduce_metrics(global_scoped, global_area, steps, deficit=True)
    changes = arrays["global_lane_changes"].sum(axis=0, dtype=np.uint64)
    changes = np.concatenate((changes, changes.sum(axis=0, keepdims=True)), axis=0)
    changes = np.concatenate((changes, changes.sum(axis=1, keepdims=True)), axis=1)
    global_metrics["lane_changes"] = changes.tolist()
    exposure = np.asarray(global_metrics["N"], dtype=np.uint64)
    global_metrics["lane_change_rate"] = [
        [int(c) / int(n) if n else None for c, n in zip(change_row, exposure_row)]
        for change_row, exposure_row in zip(changes, exposure)
    ]

    speed = arrays["pv_speed"][1:]
    all_hist = np.zeros((steps, 7), dtype=np.uint64)
    all_hist[:, :6] = arrays["global_hist"].sum(axis=(1, 2), dtype=np.uint64)
    all_hist[np.arange(steps), speed] += 1
    all_metrics = _reduce_metrics(
        all_hist,
        np.full(steps, lanes * road, dtype=np.uint64),
        steps,
        deficit=False,
    )
    pv = _pv_summary(arrays, road)
    all_changes = int(changes[-1, -1]) + pv["lane_changes"]
    all_metrics["lane_changes"] = all_changes
    all_metrics["lane_change_rate"] = all_changes / all_metrics["N"]

    initial_background = int(arrays["initial_global_hist"].sum())
    count = initial_background + 1
    av_count = int(arrays["initial_global_hist"][:, 1, :].sum())
    stopped = int(pv["stopped_steps"])
    return dict(
        schema_version=1,
        profile=result["observer_profile"],
        measurement_steps=steps,
        road_length=road,
        num_lanes=lanes,
        pv_id=result["priority_id"],
        av_ids=sorted(result["initial_av_ids"]),
        window_ids=list(WINDOW_IDS),
        primary_window=PRIMARY_WINDOW,
        regions=list(REGIONS),
        lanes=list(range(lanes)) + ["all"],
        groups=list(GROUPS),
        windows=windows,
        global_background=global_metrics,
        global_all=all_metrics,
        pv=pv,
        speed_floor_steps=stopped,
        speed_floor_fraction=stopped / steps,
        spatial_cell_time_per_bin_lane=steps,
        spatial_extent=[-36, 54],
        spatial_extent_covers_all_windows=True,
        realized_vehicle_count=count,
        realized_density=count / (lanes * road),
        realized_background_av_fraction=(
            av_count / initial_background if initial_background else None
        ),
    )


def _validate_observer_arrays(arrays: dict, result) -> None:
    config = result["config"]
    steps = int(result["steps_completed"])
    lanes = int(config["num_lanes"])
    road = int(config["road_length"])
    background = int(result["background_vehicle_count"])
    count = int(result["initial_vehicle_count"])

    pv_speed = arrays["pv_speed"]
    pv_lane = arrays["pv_lane"]
    pv_pos = arrays["pv_pos"]
    pv_path = arrays["pv_path"]
    if np.any(pv_speed > 6):
        _fail("pv_speed exceeds the priority-vehicle speed limit")
    if np.any(pv_lane >= lanes):
        _fail("pv_lane is outside the configured lanes")
    if np.any(pv_pos < 0) or np.any(pv_pos >= road):
        _fail("pv_pos is outside the road")
    if not np.array_equal(arrays["step"], np.arange(steps + 1, dtype=np.int64)):
        _fail("observer step array is not the exact measurement index")
    expected_bounds = _expected_bounds(pv_speed)
    if not np.array_equal(arrays["window_bounds"], expected_bounds):
        _fail("window_bounds do not match the canonical 12-window geometry")
    if pv_path[0] != pv_pos[0]:
        _fail("pv_path must start at the initial modulo position")
    if not np.array_equal(np.diff(pv_path), pv_speed[1:].astype(np.int64)):
        _fail("pv_path increments disagree with pv_speed")
    for index in range(steps):
        if int(pv_pos[index + 1]) != (int(pv_pos[index]) + int(pv_speed[index + 1])) % road:
            _fail("pv_pos does not follow modular velocity movement")
    if np.any(np.abs(np.diff(pv_lane.astype(np.int16))) > 1):
        _fail("PV lane changes exceed one lane per step")

    global_hist = arrays["global_hist"]
    global_counts = global_hist.sum(axis=(1, 2, 3), dtype=np.uint64)
    if np.any(global_counts != background):
        _fail("global_hist must contain every background vehicle at every step")
    initial_total = int(arrays["initial_global_hist"].sum())
    if initial_total != background:
        _fail("initial_global_hist disagrees with background_vehicle_count")
    initial_groups = arrays["initial_global_hist"].sum(axis=(0, 2))
    av_initial = int(initial_groups[1])
    if av_initial != result["av_count"]:
        _fail("initial AV histogram count disagrees with av_count")
    if int(initial_groups[0]) != background - result["av_count"]:
        _fail("initial HDV histogram count disagrees with the background population")
    step_groups = global_hist.sum(axis=(1, 3))
    if not np.array_equal(step_groups, np.broadcast_to(initial_groups, step_groups.shape)):
        _fail("HDV/AV group membership changed during the closed-ring measurement")

    window_hist = arrays["window_hist"]
    initial_window_hist = arrays["initial_window_hist"]
    all_bounds = arrays["window_bounds"].astype(np.int64)
    step_bounds = all_bounds[1:]
    # One vehicle per cell, so a single-frame count is bounded by the region
    # width (rear/centre/front), not by the number of frames T.
    step_widths = np.stack(
        (
            -step_bounds[..., 0],
            np.ones(step_bounds.shape[:2], dtype=np.int64),
            step_bounds[..., 1],
        ),
        axis=-1,
    )
    initial_widths = np.stack(
        (-all_bounds[0, :, 0], np.ones(12, dtype=np.int64), all_bounds[0, :, 1]),
        axis=-1,
    )
    window_totals = window_hist.sum(axis=(4, 5), dtype=np.uint64)
    if np.any(window_totals > step_widths[..., None]):
        _fail("window_hist exceeds the region/lane cell capacity")
    window_group_counts = window_hist.sum(axis=-1, dtype=np.uint64)
    if np.any(window_group_counts > step_widths[..., None, None]):
        _fail("window_hist group count exceeds the region/lane cell capacity")
    if np.any(window_group_counts > initial_groups[None, None, None, None, :]):
        _fail("a window histogram exceeds the initial group membership")
    if np.any(
        window_hist.sum(axis=(2, 3, 4, 5), dtype=np.uint64)
        > global_counts[:, None]
    ):
        _fail("a window histogram contains more vehicles than the global background")
    window_groups = window_hist.sum(axis=(2, 3, 5), dtype=np.uint64)
    if np.any(window_groups > initial_groups[None, None, :]):
        _fail("a window histogram exceeds the initial group membership")

    initial_totals = initial_window_hist.sum(axis=(3, 4), dtype=np.uint64)
    if np.any(initial_totals > initial_widths[..., None]):
        _fail("initial_window_hist exceeds the region/lane cell capacity")
    initial_group_counts = initial_window_hist.sum(axis=-1, dtype=np.uint64)
    if np.any(initial_group_counts > initial_widths[..., None, None]):
        _fail("initial_window_hist group count exceeds the region/lane cell capacity")
    if np.any(initial_group_counts > initial_groups[None, None, None, :]):
        _fail("an initial window histogram exceeds the initial group membership")
    initial_per_window = initial_window_hist.sum(
        axis=(1, 2, 3, 4), dtype=np.uint64
    )
    if np.any(initial_per_window > initial_total):
        _fail("an initial window histogram exceeds the initial background")

    pv_lane_post = arrays["pv_lane"][1:].astype(np.int64)
    for step in range(steps):
        if np.any(window_hist[step, :, 1, int(pv_lane_post[step])]):
            _fail("the current PV cell must not contain background centre counts")
    initial_centre_pv = initial_window_hist[:, 1, int(arrays["pv_lane"][0])]
    if np.any(initial_centre_pv):
        _fail("the initial PV cell must not contain background centre counts")

    if not np.all(window_hist[:, :, 1] == window_hist[:, 0:1, 1]):
        _fail("the centre histogram must be identical across all 12 windows")
    if not np.all(initial_window_hist[:, 1] == initial_window_hist[0:1, 1]):
        _fail("the initial centre histogram must be identical across all windows")

    pv_speed = arrays["pv_speed"]
    for step in range(steps):
        if int(pv_speed[step + 1]) == 5:
            if not np.array_equal(
                window_hist[step, 0:6], window_hist[step, 6:12]
            ):
                _fail("fixed and adaptive window histograms differ at PV speed 5")
    if int(pv_speed[0]) == 5:
        if not np.array_equal(initial_window_hist[0:6], initial_window_hist[6:12]):
            _fail("initial fixed and adaptive histograms differ at PV speed 5")

    transitions = arrays["window_transitions"]
    if np.any(transitions[..., 0] > transitions[..., 2]):
        _fail("stop_to_move transitions must also be accelerations")
    if np.any(transitions[..., 1] > transitions[..., 3]):
        _fail("move_to_stop transitions must also be decelerations")
    current_total = window_hist.sum(axis=-1, dtype=np.uint64)
    current_moving = window_hist[..., 1:].sum(axis=-1, dtype=np.uint64)
    current_stopped = window_hist[..., 0].astype(np.uint64)
    current_speed_five = window_hist[..., 5].astype(np.uint64)
    transition_bounds = (
        current_moving,   # stop_to_move: current speed > 0
        current_stopped,  # move_to_stop: current speed == 0
        current_moving,   # acceleration: current speed > 0
        current_total - current_speed_five,  # deceleration: current speed 0..4
    )
    for category, bound in enumerate(transition_bounds):
        if np.any(transitions[..., category] > bound):
            _fail("a transition category exceeds its current stopped/moving population")

    lane_changes = arrays["global_lane_changes"]
    if np.any(lane_changes > global_hist.sum(axis=-1, dtype=np.uint64)):
        _fail("global_lane_changes exceeds the destination lane/group population")
    if np.any(lane_changes.sum(axis=(1, 2), dtype=np.uint64) > global_counts):
        _fail("global_lane_changes exceeds the background population per step")

    if not np.array_equal(
        arrays["spatial_edges"], np.arange(-36, 56, dtype=np.int16)
    ):
        _fail("spatial_edges must be exactly -36..55")
    spatial = arrays["spatial_hist"]
    spatial_bin_lane = spatial.sum(axis=(2, 3), dtype=np.uint64)
    if np.any(spatial_bin_lane > steps):
        _fail("spatial_hist exceeds one background cell per bin/lane per step")
    if np.any(spatial.sum(axis=(1, 2, 3), dtype=np.uint64) > lanes * steps):
        _fail("spatial_hist exceeds the lane-time exposure of one bin")
    if int(spatial.sum()) > steps * background:
        _fail("spatial_hist exceeds the total background exposure")
    pv_frames_by_lane = np.bincount(pv_lane_post, minlength=lanes).astype(np.uint64)
    if np.any(
        spatial[36].sum(axis=(1, 2), dtype=np.uint64)
        > steps - pv_frames_by_lane
    ):
        _fail("spatial_hist contains PV's excluded centre cell")

    # Fixed windows have static bounds, so their accumulated histograms can be
    # compared with the matching aligned spatial-bin aggregate.  Adaptive
    # windows depend on PV speed and cannot be reconstructed from the
    # time-accumulated spatial profile, so they are intentionally not compared.
    for window in range(6):
        low = int(step_bounds[0, window, 0])
        high = int(step_bounds[0, window, 1])
        if np.any(step_bounds[:, window, 0] != low) or np.any(
            step_bounds[:, window, 1] != high
        ):
            _fail("fixed window bounds changed during measurement")
        if low < -36 or high > 54:
            _fail("fixed window lies outside the spatial extent")
        spatial_window = spatial[low + 36:high + 37].sum(axis=0, dtype=np.uint64)
        histogram_window = window_hist[:, window].sum(axis=(0, 1), dtype=np.uint64)
        if not np.array_equal(spatial_window, histogram_window):
            _fail("fixed window histograms disagree with the spatial aggregate")

    if int(arrays["pv_path"][-1]) - int(arrays["pv_path"][0]) < 0:
        _fail("pv_path is not monotone")
    if int(arrays["pv_path"][0]) + sum(arrays["pv_speed"][1:].tolist()) != int(
        arrays["pv_path"][-1]
    ):
        _fail("pv_path does not accumulate the measured PV velocity")
    if count != background + 1:
        _fail("population bookkeeping is inconsistent")


def _validate_mechanism_arrays(arrays: dict, result) -> None:
    _validate_mechanism_columns(result["mechanism_columns"])
    steps = int(result["steps_completed"])
    measurement_step = arrays["measurement_step"]
    if not np.array_equal(
        measurement_step, np.arange(steps, dtype=np.int64)
    ):
        _fail("measurement_step must be exactly 0..T-1")
    gap = arrays["pre_pv_leader_gap"]
    leader_type = arrays["pre_pv_leader_type"]
    leader_speed = arrays["pre_pv_leader_speed"]
    leader_stopped = arrays["pre_pv_leader_stopped"]
    if np.any(gap < -1) or np.any(gap >= int(result["config"]["road_length"])):
        _fail("pre_pv_leader_gap is outside the supported range")
    if np.any((leader_type < -1) | (leader_type > 3)):
        _fail("pre_pv_leader_type is outside the supported range")
    if np.any((leader_speed < -1) | (leader_speed > 5)):
        _fail("pre_pv_leader_speed is outside the supported range")
    if np.any((leader_stopped != -1) & (leader_stopped != 0) & (leader_stopped != 1)):
        _fail("pre_pv_leader_stopped must be -1 (no leader), 0 or 1")
    missing = gap == -1
    if np.any(
        missing
        & ((leader_type != -1) | (leader_speed != -1) | (leader_stopped != -1))
    ):
        _fail("missing-leader mechanism rows are inconsistent")
    present = ~missing
    if np.any(present & ((leader_type < 0) | (leader_speed < 0))):
        _fail("present-leader mechanism rows are inconsistent")
    if np.any(
        present & (leader_stopped != (leader_speed == 0).astype(np.int8))
    ):
        _fail("pre_pv_leader_stopped disagrees with the observed leader speed")
    for side in ("left", "right"):
        available = arrays[f"pre_pv_native_{side}"]
        reason = arrays[f"pre_pv_native_{side}_reason"]
        if np.any((available != -1) & (available != 0) & (available != 1)):
            _fail(f"pre_pv_native_{side} must be -1, 0 or 1")
        if np.any((reason < 0) | (reason > 3)):
            _fail(f"pre_pv_native_{side}_reason is outside the supported range")
        expected = np.where(
            reason == 3, -1, (reason == 0).astype(np.int8)
        ).astype(np.int8)
        if np.any(available != expected):
            _fail(f"pre_pv_native_{side} disagrees with its reason code")

    config = result["config"]
    road, lanes, S = int(config["road_length"]), int(config["num_lanes"]), int(config["S"])
    pre_lane = arrays["pre_pv_lane"].astype(np.int64)
    lane = arrays["long_pv_lane"].astype(np.int64)
    if np.any((pre_lane < 0) | (pre_lane >= lanes) | (lane < 0) | (lane >= lanes)):
        _fail("diagnostic PV lanes are out of range")
    if np.any(np.abs(lane - pre_lane) > 1) or not np.array_equal(lane[:-1], pre_lane[1:]):
        _fail("diagnostic PV lane transitions are inconsistent")
    for phase in ("pre", "native_phase"):
        left = arrays[f"{phase}_pv_native_eligible_left"]
        right = arrays[f"{phase}_pv_native_eligible_right"]
        count = arrays[f"{phase}_pv_native_eligible_count"]
        if np.any((left < 0) | (left > 1) | (right < 0) | (right > 1)):
            _fail("native eligibility must be boolean 0/1")
        if np.any(count != left + right):
            _fail("native eligible count must equal left+right")
        if np.any((pre_lane == 0) & (left != 0)) or np.any((pre_lane == lanes - 1) & (right != 0)):
            _fail("missing target lane cannot be native eligible")
    for side in ("left", "right", "count"):
        if not np.array_equal(arrays[f"pre_pv_native_eligible_{side}"], arrays[f"native_phase_pv_native_eligible_{side}"]):
            _fail("native collector phase must use pre-action geometry")
    for side, boundary in (("left", pre_lane == 0), ("right", pre_lane == lanes - 1)):
        safety = arrays[f"pre_pv_native_{side}"]
        if np.any((safety == -1) != boundary):
            _fail("safety missing-lane sentinel disagrees with PV lane")
        if np.any((arrays[f"pre_pv_native_eligible_{side}"] == 1) & (safety != 1)):
            _fail("native eligibility requires the safety/accessibility gate")

    # Promote narrow stored integers before all arithmetic. These are checks
    # on observed candidate bounds, never guesses about the stochastic draws.
    def col(suffix):
        return arrays[f"long_pv_{suffix}"].astype(np.int64)

    length, v0, final = col("length"), col("speed_before"), col("speed_after")
    if np.any(length != 1):
        _fail("v1 episode PV length must be one cell")
    vmax = int(config["vmax_priority"])
    if np.any((v0 < 0) | (v0 > vmax) | (final < 0) | (final > vmax)):
        _fail("longitudinal PV speeds are out of range")
    if not np.array_equal(final[:-1], v0[1:]):
        _fail("longitudinal PV speed transitions are inconsistent")
    g, typ, speed = col("leader_gap"), col("leader_type"), col("leader_speed")
    leader_final = col("leader_final_speed")
    missing = g == -1
    if np.any((g < -1) | (g >= road)):
        _fail("longitudinal leader gap is out of range")
    if np.any(missing & ((typ != -1) | (speed != -1) | (leader_final != -1))):
        _fail("longitudinal missing-leader sentinels are inconsistent")
    if np.any(~missing & ((typ < 0) | (typ > 2) | (speed < 0) | (speed > 5) | (leader_final < 0) | (leader_final > 5))):
        _fail("longitudinal leader type/speeds are out of range")
    effective_gap = np.where(missing, road - length, g)
    expected_gate = (effective_gap >= int(config["G"])) | (v0 <= np.where(missing, v0, speed))
    if np.any(col("acceleration_gate") != expected_gate):
        _fail("longitudinal acceleration gate is inconsistent")
    v1 = np.where(expected_gate, np.minimum(vmax, v0 + 1), v0)
    if np.any(col("v1") != v1) or np.any(final > v1):
        _fail("longitudinal v1/final speed bound is inconsistent")
    for phase in ("prev", "cur"):
        one, many = col(f"{phase}_gap_s1"), col(f"{phase}_gap_sS")
        if np.any((one < 0) | (one > road - 1) | (many < one) | (many > S * (road - 1))):
            _fail("cumulative longitudinal bounds are out of range")
        if S == 1 and np.any(many != one):
            _fail("S=1 cumulative bounds must agree")
        if np.any(missing & ((one != road - length) | (many != S * (road - length)))):
            _fail("singleton cumulative bounds are inconsistent")
    if np.any(col("cur_gap_s1") != effective_gap):
        _fail("current s=1 bound disagrees with leader gap")
    expected_previous = np.where(missing, road - length, (g - speed + v0) % road)
    if np.any(col("prev_gap_s1") != expected_previous):
        _fail("previous s=1 bound disagrees with pre-step positions/speeds")
    if np.any(final > np.minimum(v1, col("cur_gap_sS"))):
        _fail("final speed exceeds both possible current-gap bounds")
    slack = np.where(missing, -1, g + leader_final - final)
    if np.any(~missing & (slack < 0)) or np.any(col("eq9_slack") != slack):
        _fail("Eq9 slack is inconsistent")
    if np.any(col("eq9_tight") != np.where(missing, -1, (slack == 0).astype(np.int64))):
        _fail("Eq9 tightness is inconsistent")
    if steps:
        _close(result["metrics"]["mean_speed_priority"], float(final.mean()), "diagnostic PV mean speed")
        _close(result["metrics"]["stopped_fraction_priority"], float(np.mean(final == 0)), "diagnostic PV stopped fraction")
        _close(result["metrics"]["lane_change_rate_priority"], float(np.mean(lane != pre_lane)), "diagnostic PV lane change rate")
    if result["observer_profile"] != "off":
        for actual, expected in ((pre_lane, arrays["pv_lane"][:-1]), (lane, arrays["pv_lane"][1:]),
                                 (v0, arrays["pv_speed"][:-1]), (final, arrays["pv_speed"][1:])):
            if not np.array_equal(actual, expected):
                _fail("phase diagnostics disagree with observer PV trace")


def _validate_mechanism_maps(arrays: dict, result) -> None:
    if result["observer_profile"] != "mechanism":
        return
    steps = int(result["steps_completed"])
    occupancy = arrays["local_occupancy"]
    speed = arrays["local_speed"]
    vehicle_type = arrays["local_type"]
    pv_lane = arrays["pv_lane"]
    window_hist = arrays["window_hist"]
    for step in range(steps):
        occ = occupancy[step]
        spd = speed[step]
        typ = vehicle_type[step]
        if np.any((typ != -2) & (typ != -1) & (typ != 0) & (typ != 1)):
            _fail("local_type contains an unsupported value")
        if np.any((spd < -2) | (spd > 5)):
            _fail("local_speed contains an unsupported value")
        if np.any((occ != 0) & (occ != 1)):
            _fail("local_occupancy must be binary")
        pv_cell = typ == -2
        if int(np.count_nonzero(pv_cell)) != 1:
            _fail("mechanism maps must contain exactly one PV cell")
        lane = int(pv_lane[step + 1])
        if not bool(pv_cell[10, lane]):
            _fail("the PV sentinel is not at longitudinal offset zero / pv_lane")
        if occ[10, lane] != 0 or spd[10, lane] != -2:
            _fail("the PV cell must be unoccupied and use the -2 speed sentinel")
        empty = typ == -1
        if np.any(occ[empty] != 0) or np.any(spd[empty] != -1):
            _fail("empty mechanism cells must use occupancy 0 and speed -1")
        background = (typ == 0) | (typ == 1)
        if np.any(occ[background] != 1) or np.any((spd[background] < 0) | (spd[background] > 5)):
            _fail("occupied mechanism cells must have a background speed 0..5")
        expected_occupied = int(window_hist[step, 3].sum())
        if int(occ.sum()) != expected_occupied:
            _fail("mechanism occupancy disagrees with fixed_front_2 background counts")
        hist = window_hist[step, 3].sum(axis=(0, 1), dtype=np.uint64)
        for group in (0, 1):
            actual = int(np.count_nonzero(typ == group))
            if actual != int(hist[group].sum()):
                _fail("mechanism group counts disagree with fixed_front_2")
            for bin_speed in range(6):
                actual_speed = int(
                    np.count_nonzero((typ == group) & (spd == bin_speed))
                )
                if actual_speed != int(hist[group, bin_speed]):
                    _fail("mechanism speed counts disagree with fixed_front_2")


def _validate_control_arrays(arrays: dict, result) -> None:
    control = result["control"]
    steps = int(result["steps_completed"])
    trace = arrays.get("control_trace")
    names, dtype = _validate_control_columns(result["control_columns"])
    index = {}
    if names:
        if trace is None:
            _fail("control_columns are declared but control_trace is missing")
        index = {name: position for position, name in enumerate(names)}
        required_semantics = {
            "step",
            "active_av",
            "requested_nonstay",
            "applied_nonstay",
            "rejected_nonstay",
        }
        if not required_semantics.issubset(index):
            _fail(
                "control_trace must declare the documented step/active/request/applied/rejected columns"
            )
        expected = np.arange(steps, dtype=np.int64)
        if np.any(trace[:, index["step"]] != expected):
            _fail("control_trace step column is not 0..T-1")
        if np.any(trace[:, index["active_av"]] < 0):
            _fail("control_trace active_av must be nonnegative")
        if np.any(trace[:, index["active_av"]] > result["background_vehicle_count"]):
            _fail("control_trace active_av exceeds the background population")
        for name in ("requested_nonstay", "applied_nonstay", "rejected_nonstay"):
            if np.any(trace[:, index[name]] < 0):
                _fail(f"control_trace {name} must be nonnegative")
        per_step_requested = trace[:, index["requested_nonstay"]]
        per_step_applied = trace[:, index["applied_nonstay"]]
        per_step_rejected = trace[:, index["rejected_nonstay"]]
        if np.any(per_step_requested != per_step_applied + per_step_rejected):
            _fail("control_trace per-step conservation failed")
        if int(per_step_requested.sum()) != control["requested_nonstay"]:
            _fail("control totals disagree with control_trace requests")
        if int(per_step_applied.sum()) != control["applied_nonstay"]:
            _fail("control totals disagree with control_trace applied")
        if int(per_step_rejected.sum()) != control["rejected_nonstay"]:
            _fail("control totals disagree with control_trace rejections")

        for prefix, total in (("requested", "requested_nonstay"), ("applied", "applied_nonstay")):
            columns = [f"{prefix}_front", f"{prefix}_centre", f"{prefix}_rear"]
            if all(name in index for name in columns):
                for name in columns:
                    if np.any(trace[:, index[name]] < 0):
                        _fail(f"control_trace {name} must be nonnegative")
                if np.any(
                    trace[:, index[columns[0]]]
                    + trace[:, index[columns[1]]]
                    + trace[:, index[columns[2]]]
                    != trace[:, index[total]]
                ):
                    _fail(f"control_trace {prefix} zones do not partition their total")

        zone_columns = (
            "front_background",
            "centre_background",
            "rear_background",
            "front_hdv",
            "centre_hdv",
            "rear_hdv",
            "front_av",
            "centre_av",
            "rear_av",
        )
        if all(name in index for name in zone_columns):
            for row in trace:
                if int(row[index["front_background"]]) + int(row[index["centre_background"]]) + int(row[index["rear_background"]]) != result["background_vehicle_count"]:
                    _fail("control_trace background zones do not partition the population")
                for zone in ("front", "centre", "rear"):
                    if int(row[index[f"{zone}_hdv"]]) + int(row[index[f"{zone}_av"]]) != int(row[index[f"{zone}_background"]]):
                        _fail("control_trace HDV+AV zones do not equal background")
        if control["interface"] != "av_action":
            for name in (
                "selected_left",
                "selected_right",
                "selected_stay",
                "explicit_stay_commands",
                "available_left",
                "available_right",
                "eligible_nonstay",
                "common_throttled",
                "common_suppressed",
                "legal_nonstay_requests",
                "applied_legal_nonstay_requests",
                "private_throttled",
                "private_suppressed_left",
            ):
                if name in index and np.any(trace[:, index[name]] != -1):
                    _fail(f"control_trace {name} must be -1 for a non-AV interface")
        else:
            if np.any(
                trace[:, index["selected_left"]]
                + trace[:, index["selected_right"]]
                + trace[:, index["selected_stay"]]
                != trace[:, index["active_av"]]
            ):
                _fail("control_trace selected actions do not sum to active AVs")
            if np.any(trace[:, index["explicit_stay_commands"]] < trace[:, index["selected_stay"]]):
                _fail("control_trace explicit stays are smaller than selected stays")
            if np.any(trace[:, index["eligible_nonstay"]] > trace[:, index["available_left"]] + trace[:, index["available_right"]]):
                _fail("control_trace eligible decisions exceed available directions")
            if np.any(trace[:, index["legal_nonstay_requests"]] > trace[:, index["eligible_nonstay"]]):
                _fail("control_trace legal requests exceed eligible decisions")
            if np.any(trace[:, index["applied_legal_nonstay_requests"]] > trace[:, index["legal_nonstay_requests"]]):
                _fail("control_trace applied legal requests exceed legal requests")
            if np.any(trace[:, index["common_throttled"]] > trace[:, index["active_av"]]):
                _fail("control_trace common throttling exceeds active decisions")
            if "private_throttled" in index and np.any(
                trace[:, index["private_throttled"]] > trace[:, index["active_av"]]
            ):
                _fail("control_trace private throttling exceeds active decisions")
            for name, total in (
                ("selected_left", "selected_left"),
                ("selected_right", "selected_right"),
                ("selected_stay", "selected_stay"),
                ("explicit_stay_commands", "explicit_stay_commands"),
                ("available_left", "available_left_decisions"),
                ("available_right", "available_right_decisions"),
                ("eligible_nonstay", "eligible_nonstay_decisions"),
                ("common_throttled", "common_throttled_active_decisions"),
                ("common_suppressed", "common_suppressed_nonstop_requests"),
                ("legal_nonstay_requests", "legal_nonstay_requests"),
                ("applied_legal_nonstay_requests", "applied_legal_nonstay_requests"),
            ):
                if total is not None and control[total] is not None:
                    if int(trace[:, index[name]].sum()) != control[total]:
                        _fail(f"control_trace {name} disagrees with the control total")
            if control["private_throttled_active_decisions"] is not None:
                if "private_throttled" not in index:
                    _fail("control_trace is missing the declared private throttle column")
                if int(trace[:, index["private_throttled"]].sum()) != control["private_throttled_active_decisions"]:
                    _fail("control_trace private throttling disagrees with the control total")
            if control["private_suppressed_left_opportunities"] is not None:
                if "private_suppressed_left" not in index:
                    _fail("control_trace is missing the declared private suppression column")
                if int(trace[:, index["private_suppressed_left"]].sum()) != control["private_suppressed_left_opportunities"]:
                    _fail("control_trace private suppression disagrees with the control total")
            if result["method"] not in HEURISTIC_C:
                for name in ("private_throttled", "private_suppressed_left"):
                    if name in index and np.any(trace[:, index[name]] != -1):
                        _fail(f"control_trace {name} must be -1 for a method without private H_C")
        if int(trace[:, index["active_av"]].sum()) != control["active_av_decisions"]:
            _fail("control_trace active AV decisions disagree with the control total")

    rejection = arrays.get("control_rejection")
    if rejection is not None:
        names = control["rejection_reason_names"]
        if rejection.shape[1] != len(names):
            _fail("control_rejection shape disagrees with rejection_reason_names")
        if np.any(rejection < 0):
            _fail("control_rejection contains negative counts")
        if trace is not None and "rejected_nonstay" in index:
            if np.any(rejection.sum(axis=1) != trace[:, index["rejected_nonstay"]]):
                _fail("control_rejection rows do not equal rejected_nonstay")
        if int(rejection.sum()) != control["rejected_nonstay"]:
            _fail("control_rejection total disagrees with rejected_nonstay")
        reasons = control["rejection_reasons"]
        for position, name in enumerate(names):
            expected = int(reasons.get(name, 0))
            actual = int(rejection[:, position].sum())
            if name == "other":
                if actual < expected:
                    _fail("control_rejection 'other' column is smaller than its reported total")
            elif actual != expected:
                _fail(f"control_rejection column {name} disagrees with rejection_reasons")


def _validate_metrics_arrays(metrics, result, arrays, summary) -> None:
    steps = int(result["steps_completed"])
    config = result["config"]
    lanes = int(config["num_lanes"])
    road = int(config["road_length"])
    capacity = lanes * road
    background = int(result["background_vehicle_count"])
    count = int(result["initial_vehicle_count"])
    global_hist = arrays["global_hist"].astype(np.uint64)
    speeds = np.arange(6, dtype=np.uint64)
    speed_total = int((global_hist * speeds).sum())
    stopped_total = int(global_hist[..., 0].sum())
    background_changes = int(arrays["global_lane_changes"].sum())
    pv_speed = arrays["pv_speed"][1:]
    pv_speed_total = int(pv_speed.sum())
    pv_stopped = int(np.count_nonzero(pv_speed == 0))
    pv_changes = int(np.count_nonzero(np.diff(arrays["pv_lane"].astype(np.int16))))
    pv = summary["pv"]

    _close(metrics["mean_speed_priority"], pv["mean_speed"], "metrics.mean_speed_priority")
    _close(
        metrics["stopped_fraction_priority"],
        pv["stopped_fraction"],
        "metrics.stopped_fraction_priority",
    )
    _close(
        metrics["lane_change_rate_priority"],
        pv["lane_change_rate"],
        "metrics.lane_change_rate_priority",
    )
    if metrics["completed_priority_laps"] != pv["completed_laps"]:
        _fail("completed_priority_laps disagrees with the stored PV trace")
    if (metrics["mean_priority_lap_time"] is None) != (
        pv["mean_completed_lap_time"] is None
    ):
        _fail("mean_priority_lap_time availability disagrees with the stored PV trace")
    if metrics["mean_priority_lap_time"] is not None:
        _close(
            metrics["mean_priority_lap_time"],
            pv["mean_completed_lap_time"],
            "metrics.mean_priority_lap_time",
        )
    per = metrics["per_priority_vehicle"][str(result["priority_id"])]
    if int(per["completed_laps"]) != pv["completed_laps"]:
        _fail("per-priority completed laps disagree with the stored PV trace")
    _close(per["distance"], pv["distance"], "per_priority_vehicle.distance")
    _close(per["mean_speed"], pv["mean_speed"], "per_priority_vehicle.mean_speed")

    _close(
        metrics["flow_background"],
        speed_total / (steps * capacity),
        "metrics.flow_background",
    )
    _close(
        metrics["flow_priority"],
        pv_speed_total / (steps * capacity),
        "metrics.flow_priority",
    )
    _close(
        metrics["flow_all"],
        (speed_total + pv_speed_total) / (steps * capacity),
        "metrics.flow_all",
    )
    _close(
        metrics["mean_speed_all"],
        (speed_total + pv_speed_total) / (steps * count),
        "metrics.mean_speed_all",
    )
    if background:
        _close(
            metrics["mean_speed_background"],
            speed_total / (steps * background),
            "metrics.mean_speed_background",
        )
        _close(
            metrics["stopped_fraction_background"],
            stopped_total / (steps * background),
            "metrics.stopped_fraction_background",
        )
        _close(
            metrics["lane_change_rate_background"],
            background_changes / (steps * background),
            "metrics.lane_change_rate_background",
        )
    else:
        if metrics["mean_speed_background"] != 0.0:
            _fail("mean_speed_background must be zero without background")
        if metrics["stopped_fraction_background"] != 0.0:
            _fail("stopped_fraction_background must be zero without background")
        if metrics["lane_change_rate_background"] != 0.0:
            _fail("lane_change_rate_background must be zero without background")
    _close(
        metrics["stopped_fraction_all"],
        (stopped_total + pv_stopped) / (steps * count),
        "metrics.stopped_fraction_all",
    )
    _close(
        metrics["lane_change_rate_all"],
        (background_changes + pv_changes) / (steps * count),
        "metrics.lane_change_rate_all",
    )
    _close(metrics["density"], count / capacity, "metrics.density")
    if metrics["measurement_steps"] != steps:
        _fail("metrics.measurement_steps disagrees with the episode")
    if metrics["vehicle_count"] != count:
        _fail("metrics.vehicle_count disagrees with the episode population")


def _validate_numeric(arrays: dict, task, execution, result) -> None:
    profile = execution["profile"]["observer"]
    if profile != "off":
        _validate_observer_arrays(arrays, result)
        _validate_mechanism_maps(arrays, result)
        expected_summary = _recompute_observer_summary(arrays, task, result)
        _compare_saved(
            result["observer_summary"], expected_summary, "result.observer_summary"
        )
        _validate_metrics_arrays(
            result["metrics"], result, arrays, result["observer_summary"]
        )
    _validate_control_arrays(arrays, result)
    _validate_mechanism_arrays(arrays, result)


# ---------------------------------------------------------------------------
# Publication and reading
# ---------------------------------------------------------------------------


def _resolve_expected_task(expected_task, task) -> None:
    if expected_task is None:
        return
    if isinstance(expected_task, str):
        _hex64(expected_task, "expected_task")
        if expected_task != task["task_id"]:
            _fail("expected_task does not match the stored task")
        return
    if not isinstance(expected_task, dict):
        _fail("expected_task must be a task mapping or a task_id")
    schema.validate_task(expected_task)
    if expected_task["task_id"] != task["task_id"]:
        _fail("expected_task does not match the stored task")


def _resolve_expected_execution(expected_execution, execution) -> None:
    if expected_execution is None:
        return
    if isinstance(expected_execution, str):
        _hex64(expected_execution, "expected_execution")
        if expected_execution != execution["execution_id"]:
            _fail("expected_execution does not match the stored execution")
        return
    if not isinstance(expected_execution, dict):
        _fail("expected_execution must be an execution mapping or an execution_id")
    # The transport model_bundle path is deliberately excluded from the
    # execution identity and is therefore never compared here.
    if expected_execution.get("execution_id") != execution["execution_id"]:
        _fail("expected_execution does not match the stored execution")


def _check_required_profile(required_profile, execution, result) -> None:
    if required_profile is None:
        return
    rank = {"off": 0, "full12": 1, "mechanism": 2}
    if not isinstance(required_profile, str) or required_profile not in rank:
        _fail("required_profile must be off, full12 or mechanism")
    actual = execution["profile"]["observer"]
    if rank[actual] < rank[required_profile]:
        _fail(f"stored profile {actual!r} cannot satisfy required_profile {required_profile!r}")
    if actual != result["observer_profile"]:
        _fail("stored observer profile disagrees with the result")


def _validate_marker(payload, attempt_dir: Path) -> None:
    if not isinstance(payload, dict):
        _fail("completion marker must be a JSON object")
    if set(payload) != {
        "schema_version",
        "status",
        "task",
        "execution",
        "attempt_id",
        "result",
        "sidecar",
    }:
        _fail("completion marker has unknown or missing keys")
    if not _is_int(payload["schema_version"]) or payload["schema_version"] != COMPLETE_SCHEMA_VERSION:
        _fail("unsupported completion schema_version")
    if payload["status"] != COMPLETE_STATUS:
        _fail("only status='complete' markers are accepted")
    attempt_id = _safe_name(payload["attempt_id"], "completion.attempt_id")
    if attempt_id != attempt_dir.name:
        _fail("attempt directory name disagrees with the marker")
    task = payload["task"]
    if not isinstance(task, dict):
        _fail("completion.task must be a mapping")
    if attempt_dir.parent.name != task.get("task_id"):
        _fail("task directory name disagrees with the marker task_id")
    schema.validate_task(task)
    execution = payload["execution"]
    schema._validate_execution(execution, task)  # structural; no live source read


def read_episode(
    path,
    *,
    expected_task=None,
    expected_execution=None,
    required_profile=None,
):
    """Read and independently validate one immutable completed episode."""

    marker_path, attempt_dir = _resolve_marker(path)
    raw = _read_regular(marker_path, MAX_JSON_BYTES, "completion marker")
    payload = _strict_json(raw)
    _validate_marker(payload, attempt_dir)
    task = payload["task"]
    execution = payload["execution"]
    result = payload["result"]
    _resolve_expected_task(expected_task, task)
    _resolve_expected_execution(expected_execution, execution)
    _validate_result(result, task, execution)
    _check_required_profile(required_profile, execution, result)

    sidecar = payload["sidecar"]
    sidecar_path = attempt_dir / SIDECAR_NAME
    if os.path.islink(sidecar_path):
        _fail("arrays sidecar must not be a symlink")
    try:
        info = os.lstat(sidecar_path)
    except OSError as exc:
        raise ValueError("arrays sidecar is missing") from exc
    if not stat.S_ISREG(info.st_mode):
        _fail("arrays sidecar is not a regular file")
    required, optional = _expected_specs(task, execution, result)
    sizes = {name: spec for name, spec in required.items()}
    sizes.update(optional)
    declared = _validate_sidecar(sidecar, required, optional, sizes)
    max_file = _max_archive_size(required, optional)
    if int(sidecar["size"]) != info.st_size:
        _fail("sidecar size disagrees with the actual arrays file")
    if info.st_size > max_file:
        _fail("arrays file exceeds the declared archive size limit")
    if schema.file_sha256(sidecar_path) != sidecar["sha256"]:
        _fail("arrays sidecar SHA256 mismatch")
    arrays = _load_npz(sidecar_path, {name: sizes[name] for name in declared}, max_file)
    _validate_arrays(arrays, required, optional, task, result)
    _validate_numeric(arrays, task, execution, result)
    return payload, arrays


def _sidecar_for(path: Path, arrays: dict) -> dict:
    return {
        "path": SIDECAR_NAME,
        "size": path.stat().st_size,
        "sha256": schema.file_sha256(path),
        "arrays": {
            name: {
                "dtype": arrays[name].dtype.str,
                "shape": list(arrays[name].shape),
                "nbytes": int(arrays[name].nbytes),
            }
            for name in sorted(arrays)
        },
    }


def _read_orphan_arrays(path: Path, task, execution, result, required, optional) -> dict:
    if os.path.islink(path):
        _fail("arrays sidecar must not be a symlink")
    max_file = _max_archive_size(required, optional)
    arrays = _load_npz(path, {**required, **optional}, max_file)
    _validate_arrays(arrays, required, optional, task, result)
    _validate_numeric(arrays, task, execution, result)
    return arrays


def publish_episode(output_root, task, execution, attempt_id, result, arrays) -> Path:
    """Publish one completed episode atomically and return the marker path."""

    schema.validate_task(task)
    schema._validate_execution(execution, task)
    attempt = _safe_name(attempt_id, "attempt_id")
    result = _require_mapping(result, "result")
    _validate_result(result, task, execution)
    required, optional = _expected_specs(task, execution, result)
    _validate_arrays(arrays, required, optional, task, result)
    _validate_numeric(arrays, task, execution, result)

    try:
        root = Path(os.fspath(output_root))
    except TypeError as exc:
        raise ValueError("output_root is not a filesystem path") from exc
    root = Path(os.path.abspath(root))
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    if not root.is_dir():
        _fail("output_root must be a directory")
    task_dir = root / task["task_id"]
    _ensure_directory(task_dir)
    attempt_dir = task_dir / attempt
    _ensure_directory(attempt_dir)

    marker_path = attempt_dir / MARKER_NAME
    arrays_path = attempt_dir / SIDECAR_NAME
    if os.path.islink(marker_path) or os.path.islink(arrays_path):
        _fail("attempt files must not be symlinks")

    if marker_path.exists():
        payload, existing = read_episode(
            marker_path, expected_task=task, expected_execution=execution
        )
        if schema.canonical_json(payload["result"]) != schema.canonical_json(result):
            _fail("attempt already contains a different result; refusing to overwrite")
        if not _same_arrays(existing, arrays):
            _fail("attempt already contains different arrays; refusing to overwrite")
        return marker_path

    if arrays_path.exists():
        existing = _read_orphan_arrays(
            arrays_path, task, execution, result, required, optional
        )
        if not _same_arrays(existing, arrays):
            _fail("orphan arrays sidecar conflicts with this publication")
        sidecar = _sidecar_for(arrays_path, existing)
    else:
        descriptor, temporary = tempfile.mkstemp(
            dir=os.fspath(attempt_dir), prefix=".arrays-", suffix=".npz"
        )
        os.close(descriptor)
        try:
            _write_npz(Path(temporary), arrays)
            written = (
                {**required, **optional}
                if set(optional).issubset(arrays)
                else dict(required)
            )
            _load_npz(
                Path(temporary), written, _max_archive_size(required, optional)
            )
            os.link(temporary, os.fspath(arrays_path))
        except FileExistsError as exc:
            raise ValueError(
                "arrays sidecar appeared concurrently; refusing to overwrite"
            ) from exc
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        _fsync_directory(attempt_dir)
        sidecar = _sidecar_for(arrays_path, arrays)

    if len(sidecar["arrays"]) > MAX_ARRAYS:
        _fail("too many arrays to publish")
    marker = {
        "schema_version": COMPLETE_SCHEMA_VERSION,
        "status": COMPLETE_STATUS,
        "task": task,
        "execution": execution,
        "attempt_id": attempt,
        "result": result,
        "sidecar": sidecar,
    }
    encoded = schema.canonical_json(marker)
    if len(encoded) > MAX_JSON_BYTES:
        _fail("completion marker exceeds the declared size limit")
    temporary = _write_temp(attempt_dir, encoded, "completion marker")
    _install_file(temporary, marker_path, "completion marker")
    payload, _loaded = read_episode(
        marker_path,
        expected_task=task,
        expected_execution=execution,
        required_profile=(
            execution["profile"]["observer"]
            if execution["profile"]["observer"] in ("full12", "mechanism")
            else None
        ),
    )
    return marker_path
