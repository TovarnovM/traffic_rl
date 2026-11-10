"""Storage/Reader fixtures built from manual WindowObserver snapshots.

No simulator is imported.  Tiny technical horizons are explicit fixtures only.
"""

from types import SimpleNamespace
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from physica_mc import schema, storage
from physica_mc.observer import WindowObserver

SEEDS = {
    "scenario": 101,
    "warmup": 202,
    "measurement": 303,
    "pv_assignment": 404,
    "av_selection": 505,
}
ROAD_LENGTH = 200
LANES = 2
STEPS = 2
PV_ID = 10

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
CONTROL_INDEX = {name: index for index, name in enumerate(CONTROL_NAMES)}

MECHANISM = tuple(storage.MECHANISM_DTYPES.items())


def task_config(density=0.01):
    value = dict(schema.DEFAULT_CONFIG)
    value.update(
        road_length=ROAD_LENGTH,
        num_lanes=LANES,
        density=density,
        av_fraction=0.0,
        warmup_steps=0,
        measure_steps=STEPS,
        backend="reference",
        validate=False,
    )
    return value


def make_task(density=0.01):
    return schema.make_task(
        "B1",
        density=density,
        av_fraction=0.0,
        density_index=0,
        config=task_config(density),
        seeds=SEEDS,
        purpose="technical",
    )


def state(ids, lane, pos, vel, changed=None):
    return SimpleNamespace(
        vehicle_id=np.asarray(ids, dtype=np.int64),
        lane=np.asarray(lane, dtype=np.int16),
        pos=np.asarray(pos, dtype=np.int64),
        vel=np.asarray(vel, dtype=np.int16),
        alive=np.ones(len(ids), dtype=bool),
        changed_lane=(
            np.zeros(len(ids), dtype=bool)
            if changed is None
            else np.asarray(changed, dtype=bool)
        ),
    )


def snapshots():
    initial = state((PV_ID, 20, 30, 40), (0, 1, 0, 1), (0, 50, 100, 150), (2, 1, 0, 3))
    first = state(
        (PV_ID, 20, 30, 40),
        (0, 1, 0, 1),
        (3, 51, 101, 153),
        (3, 1, 1, 3),
        (False, False, False, False),
    )
    second = state(
        (PV_ID, 20, 30, 40),
        (0, 1, 0, 1),
        (7, 52, 102, 156),
        (4, 1, 1, 3),
        (False, False, False, False),
    )
    return initial, first, second


def dense_snapshots():
    # Five stopped followers and one moving leader; singleton PV keeps speed 1.
    # These admit real s=1/brake outcomes, unlike a stopped singleton PV.
    ids = (PV_ID, 20, 21, 22, 23, 24, 25)
    lanes = (0, 1, 1, 1, 1, 1, 1)
    speeds = (1, 0, 0, 0, 0, 0, 1)
    initial = state(ids, lanes, (100, 94, 95, 96, 97, 98, 99), speeds)
    first = state(ids, lanes, (101, 94, 95, 96, 97, 98, 100), speeds)
    second = state(ids, lanes, (102, 94, 95, 96, 97, 98, 101), speeds)
    return initial, first, second


def speed_five_snapshots():
    # At PV speed 5 the fixed and adaptive windows coincide exactly.
    initial = state((PV_ID, 20), (0, 1), (100, 95), (5, 1))
    first = state((PV_ID, 20), (0, 1), (105, 96), (5, 1), (False, False))
    second = state((PV_ID, 20), (0, 1), (110, 97), (5, 1), (False, False))
    return initial, first, second


def observe(profile, snapshot_fn=snapshots):
    initial, first, second = snapshot_fn()
    observer = WindowObserver(
        initial,
        pv_id=PV_ID,
        av_ids=(),
        road_length=ROAD_LENGTH,
        num_lanes=LANES,
        horizon=STEPS,
        profile=profile,
    )
    observer.observe(first)
    observer.observe(second)
    return observer.finish()


def control_columns():
    return {
        name: {"index": index, "dtype": "int64", "description": "technical fixture"}
        for index, name in enumerate(CONTROL_NAMES)
    }


def mechanism_columns():
    return {
        name: {"dtype": np.dtype(dtype).name, "description": "technical fixture"}
        for name, dtype in MECHANISM
    }


def control_trace(background):
    trace = np.zeros((STEPS, len(CONTROL_NAMES)), dtype=np.int64)
    trace[:, CONTROL_INDEX["step"]] = np.arange(STEPS, dtype=np.int64)
    trace[:, CONTROL_INDEX["active_av"]] = 0
    for name in (
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
        "legal_nonstay_requests",
        "applied_legal_nonstay_requests",
    ):
        trace[:, CONTROL_INDEX[name]] = -1
    trace[:, CONTROL_INDEX["front_background"]] = background
    trace[:, CONTROL_INDEX["centre_background"]] = 0
    trace[:, CONTROL_INDEX["rear_background"]] = 0
    trace[:, CONTROL_INDEX["front_hdv"]] = background
    trace[:, CONTROL_INDEX["front_av"]] = 0
    return trace


def rejection_trace():
    return np.zeros((STEPS, len(storage.REJECTION_REASONS)), dtype=np.int64)


def mechanism_arrays(snapshot_fn=snapshots):
    # Independent hand-fixture arithmetic only, no physics/diagnostics imports.
    frames = snapshot_fn()
    arrays = {name: np.zeros(STEPS, dtype=dtype) for name, dtype in MECHANISM}
    arrays["measurement_step"][:] = np.arange(STEPS)
    for t, (before, after) in enumerate(zip(frames, frames[1:])):
        lane, pos, v0 = int(before.lane[0]), int(before.pos[0]), int(before.vel[0])
        members = np.flatnonzero(before.lane == lane)
        others = members[members != 0]
        leader = min(others, key=lambda j: (int(before.pos[j]) - pos) % ROAD_LENGTH) if len(others) else None
        gap = (int(before.pos[leader]) - pos - 1) % ROAD_LENGTH if leader is not None else -1
        speed = int(before.vel[leader]) if leader is not None else -1
        final_leader = int(after.vel[leader]) if leader is not None else -1
        values = {
            "pre_pv_lane": lane,
            "pre_pv_leader_gap": gap,
            "pre_pv_leader_type": 0 if leader is not None else -1,
            "pre_pv_leader_speed": speed,
            "pre_pv_leader_stopped": int(speed == 0) if leader is not None else -1,
            "pre_pv_native_left": -1,
            "pre_pv_native_left_reason": 3,
            "pre_pv_native_right": 0 if snapshot_fn is dense_snapshots else 1,
            "pre_pv_native_right_reason": 2 if snapshot_fn is dense_snapshots else 0,
            "long_pv_lane": lane,
            "long_pv_length": 1,
            "long_pv_speed_before": v0,
            "long_pv_speed_after": int(after.vel[0]),
            "long_pv_leader_gap": gap,
            "long_pv_leader_type": 0 if leader is not None else -1,
            "long_pv_leader_speed": speed,
            "long_pv_leader_final_speed": final_leader,
            "long_pv_acceleration_gate": 1,
            "long_pv_v1": min(6, v0 + 1),
            "long_pv_cur_gap_s1": gap if leader is not None else ROAD_LENGTH - 1,
            "long_pv_prev_gap_s1": (gap - speed + v0) % ROAD_LENGTH if leader is not None else ROAD_LENGTH - 1,
            # Each fixture PV lane has either one or two unit-length vehicles.
            "long_pv_cur_gap_sS": ROAD_LENGTH - 2 if leader is not None else 2 * (ROAD_LENGTH - 1),
            "long_pv_prev_gap_sS": ROAD_LENGTH - 2 if leader is not None else 2 * (ROAD_LENGTH - 1),
            "long_pv_eq9_slack": gap + final_leader - int(after.vel[0]) if leader is not None else -1,
            "long_pv_eq9_tight": 0 if leader is not None else -1,
        }
        for name, value in values.items():
            arrays[name][t] = value
    return arrays


def kernel_arrays(background, snapshot_fn=snapshots):
    arrays = {
        "control_trace": control_trace(background),
        "control_rejection": rejection_trace(),
    }
    arrays.update(mechanism_arrays(snapshot_fn))
    for t, frame in enumerate(snapshot_fn()[:-1]):
        distances = (frame.pos - int(frame.pos[0]) + ROAD_LENGTH // 2) % ROAD_LENGTH - ROAD_LENGTH // 2
        for zone, mask in (("front", distances > 0), ("centre", distances == 0), ("rear", distances < 0)):
            mask[0] = False
            for group in ("background", "hdv"):
                arrays["control_trace"][t, CONTROL_INDEX[f"{zone}_{group}"]] = int(mask.sum())
    return arrays


def control_summary():
    return {
        "interface": "native",
        "steps": STEPS,
        "active_av_decisions": 0,
        "selected_left": None,
        "selected_right": None,
        "selected_stay": None,
        "explicit_stay_commands": None,
        "available_left_decisions": None,
        "available_right_decisions": None,
        "eligible_nonstay_decisions": None,
        "common_throttled_active_decisions": None,
        "private_throttled_active_decisions": None,
        "private_suppressed_left_opportunities": None,
        "common_suppressed_nonstop_requests": None,
        "common_cooldown_steps": 1,
        "private_cooldown_steps": None,
        "requested_nonstay": 0,
        "applied_nonstay": 0,
        "rejected_nonstay": 0,
        "emitted_nonstay": 0,
        "legal_nonstay_requests": None,
        "applied_legal_nonstay_requests": None,
        "request_opportunity_use": None,
        "applied_opportunity_use": None,
        "opportunity_use_status": "not_applicable",
        "applied_per_step": 0.0,
        "applied_per_active_decision": None,
        "applied_per_request": None,
        "rejected_per_request": None,
        "rejection_reasons": {},
        "rejection_reason_names": list(storage.REJECTION_REASONS),
        "requested_front": 0,
        "requested_centre": 0,
        "requested_rear": 0,
        "applied_front": 0,
        "applied_centre": 0,
        "applied_rear": 0,
    }


def metrics_from_arrays(arrays, summary, config, count):
    background = count - 1
    capacity = int(config["num_lanes"]) * int(config["road_length"])
    vmax = max(1, int(config["vmax_priority"]))
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
    metrics = {
        "measurement_steps": STEPS,
        "vehicle_count": count,
        "priority_vehicle_count": 1,
        "density": count / capacity,
        "flow_all": (speed_total + pv_speed_total) / (STEPS * capacity),
        "flow_background": speed_total / (STEPS * capacity),
        "flow_priority": pv_speed_total / (STEPS * capacity),
        "mean_speed_all": (speed_total + pv_speed_total) / (STEPS * count),
        "mean_speed_background": speed_total / (STEPS * background),
        "mean_speed_priority": pv["mean_speed"],
        "worst_priority_mean_speed": pv["mean_speed"],
        "mean_priority_time_loss": 1.0 - pv["mean_speed"] / vmax,
        "worst_priority_time_loss": 1.0 - pv["mean_speed"] / vmax,
        "stopped_fraction_all": (stopped_total + pv_stopped) / (STEPS * count),
        "stopped_fraction_background": stopped_total / (STEPS * background),
        "stopped_fraction_priority": pv["stopped_fraction"],
        "lane_change_rate_all": (background_changes + pv_changes) / (STEPS * count),
        "lane_change_rate_background": background_changes / (STEPS * background),
        "lane_change_rate_priority": pv_changes / STEPS,
        "completed_priority_laps": pv["completed_laps"],
        "mean_priority_lap_time": pv["mean_completed_lap_time"],
        "rule_detections": 0,
        "rule_requests": 0,
        "rule_applied": 0,
        "rule_rejected": 0,
        "rule_blocked_no_safe_lane": 0,
        "rule_cooldown_suppressed": 0,
        "per_priority_vehicle": {
            str(PV_ID): {
                "mean_speed": pv["mean_speed"],
                "time_loss": 1.0 - pv["mean_speed"] / vmax,
                "distance": pv["distance"],
                "completed_laps": pv["completed_laps"],
                "mean_lap_time": pv["mean_completed_lap_time"],
            }
        },
    }
    return metrics


def token(name):
    return hashlib.sha256(name.encode("ascii")).hexdigest()


def build(profile, *, snapshot_fn=snapshots, density=0.01, off_background=3):
    task = make_task(density)
    execution = schema.make_execution(task, observer=profile)
    config = task["science"]["config"]
    if profile == "off":
        summary = None
        observer_arrays = {}
        background = off_background
        count = background + 1
        fixture_summary, fixture_arrays = observe("full12", snapshot_fn)
        metrics = metrics_from_arrays(fixture_arrays, fixture_summary, config, count)
    else:
        summary, observer_arrays = observe(profile, snapshot_fn)
        background = int(observer_arrays["initial_global_hist"].sum())
        count = background + 1
        arrays_with_observer = dict(observer_arrays)
        metrics = metrics_from_arrays(
            {**arrays_with_observer, **kernel_arrays(background, snapshot_fn)},
            summary,
            config,
            count,
        )
    arrays = {**observer_arrays, **kernel_arrays(background, snapshot_fn)}
    result = {
        "schema_version": schema.SCHEMA_VERSION,
        "protocol": schema.PROTOCOL,
        "task_id": task["task_id"],
        "execution_id": execution["execution_id"],
        "purpose": "technical",
        "split": "pilot",
        "method": "B1",
        "architecture": None,
        "model_id": None,
        "training_line": None,
        "candidate_id": None,
        "env_steps": None,
        "density_index": task["science"]["density_index"],
        "replicate": task["science"]["replicate"],
        "config": dict(config),
        "seeds": dict(task["science"]["seeds"]),
        "priority_id": PV_ID,
        "initial_av_ids": [],
        "av_assignment_applied": False,
        "requested_density": config["density"],
        "requested_background_av_fraction": config["av_fraction"],
        "realized_density": count / (int(config["num_lanes"]) * int(config["road_length"])),
        "realized_background_av_fraction": 0.0,
        "initial_vehicle_count": count,
        "background_vehicle_count": background,
        "av_count": 0,
        "background_av_count": 0,
        "measurement_steps": STEPS,
        "warmup_steps_completed": 0,
        "steps_completed": STEPS,
        "warm_state_sha256": token("warm"),
        "pv_assignment_state_sha256": token("pv"),
        "av_assignment_state_sha256": None,
        "initial_state_sha256": token("initial"),
        "final_state_sha256": token("final"),
        "initial_rng_sha256": token("initial-rng"),
        "final_rng_sha256": token("final-rng"),
        "trace_enabled": False,
        "trace_sha256": None,
        "model": None,
        "observer_profile": profile,
        "observer_summary": summary,
        "metrics": metrics,
        "control": control_summary(),
        "control_columns": control_columns(),
        "control_rejection_columns": {
            name: {"index": index, "description": "technical fixture"}
            for index, name in enumerate(storage.REJECTION_REASONS)
        },
        "mechanism_columns": mechanism_columns(),
        "mechanism_notes": {
            "pre_pv_leader": "technical fixture raw leader; -1 sentinels mean no leader",
            "pre_pv_native_left_right": "technical fixture safety/accessibility gate only, not a native proposal",
            "native_phase": "original pre-step geometry, before reserved-target filtering",
            "longitudinal": "post-lateral lanes/pre-step positions and velocities; no stochastic claims",
            "unavailable": list(storage.MECHANISM_UNAVAILABLE),
        },
        "backend_requested": "reference",
        "backend_paths": ["reference"],
        "device": {"requested": "cpu", "torch_imported": False, "type": "cpu"},
        "timings": {name: 0.0 for name in storage.TIMING_KEYS},
        "resources": {
            "cpu_seconds": 0.0,
            "wall_seconds": 0.0,
            "process_peak_rss_kib": 1,
        },
        "deadline_utc": None,
    }
    return task, execution, result, arrays


def write_attempt(root, task, execution, attempt, result, arrays, mutate=None):
    attempt_dir = Path(root) / task["task_id"] / attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    path = attempt_dir / "arrays.npz"
    np.savez(path, **{name: arrays[name] for name in sorted(arrays)})
    sidecar = {
        "path": "arrays.npz",
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
    payload = {
        "schema_version": storage.COMPLETE_SCHEMA_VERSION,
        "status": storage.COMPLETE_STATUS,
        "task": task,
        "execution": execution,
        "attempt_id": attempt,
        "result": result,
        "sidecar": sidecar,
    }
    if mutate is not None:
        mutate(payload)
    (attempt_dir / "completion.json").write_bytes(schema.canonical_json(payload))
    return attempt_dir / "completion.json"


def test_publication_roundtrip_and_layout(tmp_path):
    task, execution, result, arrays = build("full12")
    marker = storage.publish_episode(tmp_path, task, execution, "attempt-1", result, arrays)
    assert marker == tmp_path / task["task_id"] / "attempt-1" / "completion.json"
    assert marker.is_file()
    assert (marker.parent / "arrays.npz").is_file()
    payload, loaded = storage.read_episode(marker)
    assert payload["status"] == "complete"
    assert payload["schema_version"] == 1
    assert payload["task"] == task
    assert payload["execution"] == execution
    assert payload["attempt_id"] == "attempt-1"
    assert payload["sidecar"]["path"] == "arrays.npz"
    assert set(loaded) == set(arrays)
    for name in arrays:
        np.testing.assert_array_equal(loaded[name], arrays[name])

    payload2, loaded2 = storage.read_episode(
        marker.parent,
        expected_task=task,
        expected_execution=execution,
        required_profile="full12",
    )
    assert payload2 == payload
    for name in loaded:
        np.testing.assert_array_equal(loaded2[name], loaded[name])


def test_same_attempt_is_idempotent_and_conflicts_are_rejected(tmp_path):
    task, execution, result, arrays = build("full12")
    marker = storage.publish_episode(tmp_path, task, execution, "attempt-1", result, arrays)
    marker_bytes = marker.read_bytes()
    arrays_bytes = (marker.parent / "arrays.npz").read_bytes()
    second = storage.publish_episode(tmp_path, task, execution, "attempt-1", result, arrays)
    assert second == marker
    assert marker.read_bytes() == marker_bytes
    assert (marker.parent / "arrays.npz").read_bytes() == arrays_bytes

    changed_result = json.loads(json.dumps(result))
    changed_result["timings"]["total"] = 1.0
    with pytest.raises(ValueError):
        storage.publish_episode(
            tmp_path, task, execution, "attempt-1", changed_result, arrays
        )
    assert marker.read_bytes() == marker_bytes

    changed_arrays = {name: value.copy() for name, value in arrays.items()}
    changed_arrays["pre_pv_native_left"][0] = 0
    changed_arrays["pre_pv_native_left_reason"][0] = 1
    with pytest.raises(ValueError):
        storage.publish_episode(
            tmp_path, task, execution, "attempt-1", result, changed_arrays
        )
    assert (marker.parent / "arrays.npz").read_bytes() == arrays_bytes


def test_wrong_assignment_and_digest_are_rejected(tmp_path):
    task, execution, result, arrays = build("full12")
    marker = storage.publish_episode(tmp_path, task, execution, "attempt-1", result, arrays)
    other_task = schema.make_task("B1", density=0.1, seeds=SEEDS)
    with pytest.raises(ValueError):
        storage.read_episode(marker, expected_task=other_task)

    def break_execution(payload):
        payload["execution"]["execution_id"] = "0" * 64

    bad = write_attempt(tmp_path, task, execution, "attempt-exec", result, arrays, break_execution)
    with pytest.raises(ValueError):
        storage.read_episode(bad)

    def break_task(payload):
        payload["task"]["science"]["replicate"] = 1

    bad = write_attempt(tmp_path, task, execution, "attempt-task", result, arrays, break_task)
    with pytest.raises(ValueError):
        storage.read_episode(bad)

    def break_execution_task(payload):
        payload["execution"]["task_id"] = "f" * 64

    bad = write_attempt(
        tmp_path, task, execution, "attempt-exec-task", result, arrays, break_execution_task
    )
    with pytest.raises(ValueError):
        storage.read_episode(bad)

    def break_result(payload):
        payload["result"]["task_id"] = "f" * 64

    bad = write_attempt(
        tmp_path, task, execution, "attempt-result", result, arrays, break_result
    )
    with pytest.raises(ValueError):
        storage.read_episode(bad)


def test_truncated_archive_and_bad_sidecar_hashes(tmp_path):
    task, execution, result, arrays = build("full12")
    marker = storage.publish_episode(tmp_path, task, execution, "attempt-1", result, arrays)
    path = marker.parent / "arrays.npz"
    data = path.read_bytes()
    path.write_bytes(data[:-10])
    with pytest.raises(ValueError):
        storage.read_episode(marker)

    def bad_hash(payload):
        payload["sidecar"]["sha256"] = "0" * 64

    bad = write_attempt(tmp_path, task, execution, "attempt-hash", result, arrays, bad_hash)
    with pytest.raises(ValueError):
        storage.read_episode(bad)

    def bad_size(payload):
        payload["sidecar"]["size"] += 1

    bad = write_attempt(tmp_path, task, execution, "attempt-size", result, arrays, bad_size)
    with pytest.raises(ValueError):
        storage.read_episode(bad)


def test_dtype_shape_and_numeric_corruption_after_rehash(tmp_path):
    task, execution, result, arrays = build("full12")

    bad_shape = {name: value.copy() for name, value in arrays.items()}
    bad_shape["global_hist"] = bad_shape["global_hist"][:-1]
    bad = write_attempt(tmp_path, task, execution, "attempt-shape", result, bad_shape)
    with pytest.raises(ValueError):
        storage.read_episode(bad)

    bad_dtype = {name: value.copy() for name, value in arrays.items()}
    bad_dtype["global_hist"] = bad_dtype["global_hist"].astype(np.int32)
    bad = write_attempt(tmp_path, task, execution, "attempt-dtype", result, bad_dtype)
    with pytest.raises(ValueError):
        storage.read_episode(bad)

    # Corrupt a histogram count, then rehash the archive and sidecar.  Only a
    # numerical recomputation can reject this.
    bad_numeric = {name: value.copy() for name, value in arrays.items()}
    window_hist = bad_numeric["window_hist"]
    sums = window_hist.sum(axis=(2, 3, 4, 5))
    background = result["background_vehicle_count"]
    candidates = np.argwhere(sums < background)
    step, window = (int(value) for value in candidates[0])
    zeros = np.argwhere(window_hist[step, window][:, :, 0, :] == 0)
    region, lane, speed = (int(value) for value in zeros[0])
    window_hist[step, window][region, lane, 0, speed] = 1
    bad = write_attempt(tmp_path, task, execution, "attempt-numeric", result, bad_numeric)
    with pytest.raises(ValueError):
        storage.read_episode(bad)


def test_dense_short_fixture_accepts_legitimate_counts_above_t(tmp_path):
    task, execution, result, arrays = build(
        "full12", snapshot_fn=dense_snapshots, density=0.0175
    )
    # window 0 = fixed_sym_1, region 0 = rear, lane 1; five cars share one
    # frame/speed/group bin, so a per-frame histogram count is 5 > T=2.
    counts = arrays["window_hist"][:, 0, 0, 1, 0, 0]
    assert int(counts.max()) == 5
    assert int(counts.max()) > STEPS
    marker = storage.publish_episode(
        tmp_path, task, execution, "dense-attempt", result, arrays
    )
    payload, loaded = storage.read_episode(
        marker, expected_task=task, expected_execution=execution,
        required_profile="full12",
    )
    assert payload["result"]["background_vehicle_count"] == 6
    np.testing.assert_array_equal(
        loaded["window_hist"], arrays["window_hist"]
    )


def test_pv_speed_five_fixed_and_adaptive_histograms_are_identical(tmp_path):
    task, execution, result, arrays = build(
        "full12", snapshot_fn=speed_five_snapshots, density=0.005
    )
    np.testing.assert_array_equal(
        arrays["window_hist"][:, 0:6], arrays["window_hist"][:, 6:12]
    )
    np.testing.assert_array_equal(
        arrays["initial_window_hist"][0:6],
        arrays["initial_window_hist"][6:12],
    )
    marker = storage.publish_episode(
        tmp_path, task, execution, "v5-attempt", result, arrays
    )
    payload, loaded = storage.read_episode(marker)
    assert payload["result"]["priority_id"] == PV_ID
    np.testing.assert_array_equal(loaded["window_hist"], arrays["window_hist"])


def test_capacity_and_lane_change_corruption_rehashed(tmp_path):
    task, execution, result, arrays = build(
        "full12", snapshot_fn=dense_snapshots, density=0.0175
    )

    bad_capacity = {name: value.copy() for name, value in arrays.items()}
    bad_capacity["window_hist"][0, 0, 0, 1, 0, 0] = 11  # fixed_sym_1 rear width 10
    bad = write_attempt(
        tmp_path, task, execution, "capacity", result, bad_capacity
    )
    with pytest.raises(ValueError):
        storage.read_episode(bad)

    bad_lane = {name: value.copy() for name, value in arrays.items()}
    bad_lane["global_lane_changes"][0, 1, 0] = (
        int(bad_lane["global_hist"][0, 1, 0].sum()) + 1
    )
    bad = write_attempt(tmp_path, task, execution, "lane-change", result, bad_lane)
    with pytest.raises(ValueError):
        storage.read_episode(bad)


def test_spatial_aggregate_and_per_bin_corruption_rehashed(tmp_path):
    task, execution, result, arrays = build(
        "full12", snapshot_fn=dense_snapshots, density=0.0175
    )

    # Offset -5 is spatial bin index 31 (edges start at -36).  One vehicle can
    # occupy a bin/lane only once per frame.
    bad_bin = {name: value.copy() for name, value in arrays.items()}
    bad_bin["spatial_hist"][31, 1, 0, 0] = STEPS + 1
    bad = write_attempt(tmp_path, task, execution, "spatial-bin", result, bad_bin)
    with pytest.raises(ValueError):
        storage.read_episode(bad)

    # Fixed-window histograms are reconstructible from the aligned spatial
    # bins; a one-count mismatch must be rejected even after rehashing.
    bad_aggregate = {name: value.copy() for name, value in arrays.items()}
    bad_aggregate["spatial_hist"][31, 1, 0, 0] -= 1
    bad = write_attempt(
        tmp_path, task, execution, "spatial-aggregate", result, bad_aggregate
    )
    with pytest.raises(ValueError):
        storage.read_episode(bad)


def test_pv_exclusion_group_and_centre_identity_corruption_rehashed(tmp_path):
    task, execution, result, arrays = build(
        "full12", snapshot_fn=speed_five_snapshots, density=0.005
    )

    bad_group = {name: value.copy() for name, value in arrays.items()}
    bad_group["window_hist"][1, 0, 0, 1, 1, 0] = 1  # AV group has zero members
    bad = write_attempt(tmp_path, task, execution, "group", result, bad_group)
    with pytest.raises(ValueError):
        storage.read_episode(bad)

    bad_centre = {name: value.copy() for name, value in arrays.items()}
    bad_centre["window_hist"][1, 0, 1, 0, 0, 0] = 1  # centre/PV-lane cell
    bad = write_attempt(tmp_path, task, execution, "centre-pv", result, bad_centre)
    with pytest.raises(ValueError):
        storage.read_episode(bad)

    bad_identity = {name: value.copy() for name, value in arrays.items()}
    bad_identity["window_hist"][1, 1, 1, 1, 0, 0] = 1  # differs from window 0
    bad = write_attempt(
        tmp_path, task, execution, "centre-identity", result, bad_identity
    )
    with pytest.raises(ValueError):
        storage.read_episode(bad)


def test_pv_speed_five_histogram_corruption_rehashed(tmp_path):
    task, execution, result, arrays = build(
        "full12", snapshot_fn=speed_five_snapshots, density=0.005
    )
    # window 6 is speed_sym_1; at PV speed 5 its front region must equal the
    # fixed_sym_1 front region.  Frame 1 has one background vehicle globally
    # and none in this window, so the added count still fits all capacity and
    # total-population bounds and only the fixed/adaptive equality rejects it.
    bad_arrays = {name: value.copy() for name, value in arrays.items()}
    bad_arrays["window_hist"][1, 6, 2, 1, 0, 1] = 1
    bad = write_attempt(tmp_path, task, execution, "v5-corrupt", result, bad_arrays)
    with pytest.raises(ValueError):
        storage.read_episode(bad)


def test_deceleration_excludes_current_speed_five(tmp_path):
    task, execution, result, arrays = build(
        "full12", snapshot_fn=dense_snapshots, density=0.0175
    )
    bad = {name: value.copy() for name, value in arrays.items()}
    # Five stopped followers plus one moving leader: move one follower to
    # speed 5, so at most five of these six vehicles can decelerate.
    bad["window_hist"][0, 0, 0, 1, 0, 0] = 4
    bad["window_hist"][0, 0, 0, 1, 0, 5] = 1
    bad["window_transitions"][0, 0, 0, 1, 0, 3] = 6
    marker = write_attempt(
        tmp_path, task, execution, "deceleration", result, bad
    )
    with pytest.raises(ValueError):
        storage.read_episode(marker)


def test_mechanism_sentinels_and_dtypes_are_rehashed(tmp_path):
    task, execution, result, arrays = build("full12", snapshot_fn=speed_five_snapshots, density=0.005)

    bad_stopped = {name: value.copy() for name, value in arrays.items()}
    bad_stopped["pre_pv_leader_stopped"][0] = 0  # gap[0] == -1 means no leader
    bad = write_attempt(
        tmp_path, task, execution, "sentinel-stopped", result, bad_stopped
    )
    with pytest.raises(ValueError):
        storage.read_episode(bad)

    bad_native = {name: value.copy() for name, value in arrays.items()}
    bad_native["pre_pv_native_left"][0] = 0  # reason 3 = lane_missing => -1
    bad = write_attempt(
        tmp_path, task, execution, "sentinel-native", result, bad_native
    )
    with pytest.raises(ValueError):
        storage.read_episode(bad)

    bad_native = {name: value.copy() for name, value in arrays.items()}
    bad_native["pre_pv_native_right"][0] = -1  # reason 0 = pass => 1
    bad = write_attempt(
        tmp_path, task, execution, "sentinel-native-pass", result, bad_native
    )
    with pytest.raises(ValueError):
        storage.read_episode(bad)

    # The documented mechanism dtypes are now signed int8 for the gate/stopped
    # arrays.  A declaration that says uint8 must be rejected, not silently
    # accepted because the values happen to be nonnegative.
    bad_result = json.loads(json.dumps(result))
    bad_result["mechanism_columns"]["pre_pv_native_left"]["dtype"] = "uint8"
    bad_arrays = {name: value.copy() for name, value in arrays.items()}
    bad_arrays["pre_pv_native_left"] = bad_arrays["pre_pv_native_left"].astype(
        np.uint8
    )
    bad = write_attempt(
        tmp_path, task, execution, "mechanism-dtype", bad_result, bad_arrays
    )
    with pytest.raises(ValueError):
        storage.read_episode(bad)


def test_mechanism_notes_must_name_unavailable_evidence(tmp_path):
    task, execution, result, arrays = build("full12")
    bad_result = json.loads(json.dumps(result))
    bad_result["mechanism_notes"]["unavailable"] = ["native_incentive_gate"]
    bad = write_attempt(tmp_path, task, execution, "notes", bad_result, arrays)
    with pytest.raises(ValueError):
        storage.read_episode(bad)


def test_model_provenance_bindings_and_extra_fields(tmp_path):
    task = schema.make_task(
        "graph",
        density=0.2,
        seeds=SEEDS,
        model_id="a" * 64,
        training_line="line-1",
        candidate_id="candidate-1",
        env_steps=100,
    )
    model = {
        "model_id": "a" * 64,
        "training_line": "line-1",
        "candidate_id": "candidate-1",
        "env_steps": 100,
        "architecture": "graph",
        "bundle_identity": "a" * 64,
        "bundle_config_sha256": token("bundle-config"),
        # The current kernel no longer emits weights_sha256; old/extra
        # provenance is accepted and checked when present.
        "bundle_weights_sha256": token("bundle-weights"),
        "extra_provenance": {"note": "legitimate extra metadata"},
    }
    storage._validate_model(model, task, "graph")

    missing_line = dict(model)
    del missing_line["training_line"]
    with pytest.raises(ValueError):
        storage._validate_model(missing_line, task, "graph")

    bad_steps = dict(model)
    bad_steps["env_steps"] = 99
    with pytest.raises(ValueError):
        storage._validate_model(bad_steps, task, "graph")

    bad_hash = dict(model)
    bad_hash["bundle_weights_sha256"] = "not-a-digest"
    with pytest.raises(ValueError):
        storage._validate_model(bad_hash, task, "graph")

    # Top-level duplicate provenance must also bind to the immutable task.
    fixture_task, execution, result, arrays = build("full12")
    bad_result = json.loads(json.dumps(result))
    bad_result["model_id"] = "b" * 64
    bad = write_attempt(
        tmp_path, fixture_task, execution, "top-model-id", bad_result, arrays
    )
    with pytest.raises(ValueError):
        storage.read_episode(bad)


def test_control_and_scalar_corruption_are_recomputed(tmp_path):
    task, execution, result, arrays = build("full12")

    bad_array = {name: value.copy() for name, value in arrays.items()}
    bad_array["control_trace"][1, CONTROL_INDEX["requested_nonstay"]] += 1
    bad = write_attempt(tmp_path, task, execution, "attempt-control", result, bad_array)
    with pytest.raises(ValueError):
        storage.read_episode(bad)

    bad_result = json.loads(json.dumps(result))
    bad_result["metrics"]["mean_speed_background"] += 0.1
    bad = write_attempt(tmp_path, task, execution, "attempt-metric", bad_result, arrays)
    with pytest.raises(ValueError):
        storage.read_episode(bad)


def test_profile_hierarchy_and_observer_map_requirements(tmp_path):
    full_task, full_execution, full_result, full_arrays = build("full12")
    off_task, off_execution, off_result, off_arrays = build("off")
    mechanism_task, mechanism_execution, mechanism_result, mechanism_arrays_ = build(
        "mechanism"
    )

    full_marker = storage.publish_episode(
        tmp_path, full_task, full_execution, "attempt-full", full_result, full_arrays
    )
    off_marker = storage.publish_episode(
        tmp_path, off_task, off_execution, "attempt-off", off_result, off_arrays
    )
    mechanism_marker = storage.publish_episode(
        tmp_path,
        mechanism_task,
        mechanism_execution,
        "attempt-mechanism",
        mechanism_result,
        mechanism_arrays_,
    )

    storage.read_episode(full_marker, required_profile="full12")
    storage.read_episode(full_marker, required_profile="off")
    with pytest.raises(ValueError):
        storage.read_episode(full_marker, required_profile="mechanism")

    storage.read_episode(off_marker, required_profile="off")
    with pytest.raises(ValueError):
        storage.read_episode(off_marker, required_profile="full12")

    storage.read_episode(mechanism_marker, required_profile="full12")
    storage.read_episode(mechanism_marker, required_profile="mechanism")
    with pytest.raises(ValueError):
        storage.read_episode(mechanism_marker, required_profile="bogus")


def test_orphan_is_not_complete_and_publication_retry_completes_it(tmp_path):
    task, execution, result, arrays = build("full12")
    marker = storage.publish_episode(tmp_path, task, execution, "attempt-1", result, arrays)
    arrays_path = marker.parent / "arrays.npz"
    original = arrays_path.read_bytes()
    marker.unlink()
    with pytest.raises(ValueError):
        storage.read_episode(marker.parent)
    retried = storage.publish_episode(tmp_path, task, execution, "attempt-1", result, arrays)
    assert retried == marker
    assert arrays_path.read_bytes() == original
    storage.read_episode(retried, expected_task=task, expected_execution=execution)


def test_conflicting_orphan_arrays_are_rejected_without_overwrite(tmp_path):
    task, execution, result, arrays = build("full12")
    marker = storage.publish_episode(tmp_path, task, execution, "attempt-1", result, arrays)
    arrays_path = marker.parent / "arrays.npz"
    marker.unlink()
    changed = {name: value.copy() for name, value in arrays.items()}
    changed["pre_pv_native_left"][0] = 0
    changed["pre_pv_native_left_reason"][0] = 1
    np.savez(arrays_path, **{name: changed[name] for name in sorted(changed)})
    with pytest.raises(ValueError):
        storage.publish_episode(tmp_path, task, execution, "attempt-1", result, arrays)
    assert not marker.exists()


def test_unsafe_paths_symlinks_and_archive_members_are_rejected(tmp_path):
    task, execution, result, arrays = build("full12")
    for attempt in ("../escape", "a/b", ".", ""):
        with pytest.raises(ValueError):
            storage.publish_episode(
                tmp_path, task, execution, attempt, result, arrays
            )

    marker = storage.publish_episode(tmp_path, task, execution, "attempt-1", result, arrays)
    arrays_path = marker.parent / "arrays.npz"
    target = tmp_path / "outside.npz"
    target.write_bytes(arrays_path.read_bytes())
    arrays_path.unlink()
    os.symlink(target, arrays_path)
    with pytest.raises(ValueError):
        storage.read_episode(marker)

    def traversal(payload):
        payload["sidecar"]["path"] = "../../arrays.npz"

    bad = write_attempt(tmp_path, task, execution, "attempt-path", result, arrays, traversal)
    with pytest.raises(ValueError):
        storage.read_episode(bad)

    bad_marker = write_attempt(tmp_path, task, execution, "attempt-extra", result, arrays)
    extra_path = bad_marker.parent / "arrays.npz"
    np.savez(extra_path, **{name: arrays[name] for name in sorted(arrays)}, evil=np.zeros(1))
    payload = json.loads(bad_marker.read_text())
    payload["sidecar"]["size"] = extra_path.stat().st_size
    payload["sidecar"]["sha256"] = schema.file_sha256(extra_path)
    bad_marker.write_bytes(schema.canonical_json(payload))
    with pytest.raises(ValueError):
        storage.read_episode(bad_marker)

    wrong_dir = marker.parent.parent / "wrong-name"
    wrong_dir.mkdir()
    (wrong_dir / "arrays.npz").write_bytes((marker.parent / "arrays.npz").read_bytes())
    (wrong_dir / "completion.json").write_bytes(marker.read_bytes())
    with pytest.raises(ValueError):
        storage.read_episode(wrong_dir / "completion.json")

    def mismatched_attempt(payload):
        payload["attempt_id"] = "attempt-1"

    bad = write_attempt(
        tmp_path, task, execution, "attempt-mismatch", result, arrays, mismatched_attempt
    )
    with pytest.raises(ValueError):
        storage.read_episode(bad)


def test_partial_marker_status_is_rejected(tmp_path):
    task, execution, result, arrays = build("full12")

    def partial(payload):
        payload["status"] = "partial"

    bad = write_attempt(tmp_path, task, execution, "attempt-partial", result, arrays, partial)
    with pytest.raises(ValueError):
        storage.read_episode(bad)


def test_expected_execution_ignores_bundle_transport_path(tmp_path):
    task, _execution, result, arrays = build("full12")
    execution_a = schema.make_execution(
        task, observer="full12", model_bundle="/transport/a"
    )
    execution_b = schema.make_execution(
        task, observer="full12", model_bundle="/transport/b"
    )
    assert execution_a["execution_id"] == execution_b["execution_id"]
    result["execution_id"] = execution_a["execution_id"]
    marker = storage.publish_episode(
        tmp_path, task, execution_a, "attempt-1", result, arrays
    )
    payload, _loaded = storage.read_episode(
        marker, expected_execution=execution_b
    )
    assert payload["execution"]["model_bundle"] == "/transport/a"


def test_relocated_publication_is_byte_identical(tmp_path):
    task, execution, result, arrays = build("full12")
    marker_a = storage.publish_episode(
        tmp_path / "root-a", task, execution, "attempt-1", result, arrays
    )
    marker_b = storage.publish_episode(
        tmp_path / "root-b", task, execution, "attempt-1", result, arrays
    )
    payload_a, _ = storage.read_episode(marker_a)
    payload_b, _ = storage.read_episode(marker_b)
    assert payload_a["task"] == payload_b["task"]
    assert payload_a["execution"] == payload_b["execution"]
    assert payload_a["result"] == payload_b["result"]
    assert payload_a["sidecar"] == payload_b["sidecar"]
    assert schema.file_sha256(marker_a.parent / "arrays.npz") == schema.file_sha256(
        marker_b.parent / "arrays.npz"
    )


@pytest.mark.parametrize("profile", ["off", "full12", "mechanism"])
@pytest.mark.parametrize("kind", ["control", "rejection", "mechanism", "notes"])
def test_rehashed_missing_mandatory_v1_fields_are_rejected(tmp_path, profile, kind):
    task, execution, result, arrays = build(profile)
    result = json.loads(json.dumps(result))
    arrays = {name: value.copy() for name, value in arrays.items()}
    if kind == "control":
        # A plausible partial table with contiguous indices must still fail.
        del result["control_columns"]["applied_rear"]
        arrays["control_trace"] = arrays["control_trace"][:, :-1].copy()
    elif kind == "rejection":
        del result["control_rejection_columns"]
        del arrays["control_rejection"]
    elif kind == "mechanism":
        del result["mechanism_columns"]["pre_pv_native_eligible_count"]
        del arrays["pre_pv_native_eligible_count"]
    else:
        del result["mechanism_notes"]
    marker = write_attempt(tmp_path, task, execution, f"missing-{kind}", result, arrays)
    with pytest.raises(ValueError):
        storage.read_episode(marker)


@pytest.mark.parametrize("name,value", [
    ("pre_pv_native_eligible_count", 1),
    ("native_phase_pv_native_eligible_right", 1),
    ("pre_pv_native_eligible_left", 1),
    ("long_pv_acceleration_gate", 0),
    ("long_pv_v1", 2),
    ("long_pv_cur_gap_s1", 98),
    ("long_pv_prev_gap_s1", 100),
    ("long_pv_cur_gap_sS", 500),
    ("long_pv_prev_gap_sS", -1),
    ("long_pv_leader_type", -1),
    ("long_pv_leader_final_speed", -1),
    ("long_pv_eq9_slack", 0),
    ("long_pv_eq9_tight", 1),
    ("long_pv_speed_after", 6),
])
def test_rehashed_native_and_longitudinal_inconsistencies(tmp_path, name, value):
    task, execution, result, arrays = build("full12")
    arrays[name][0] = value
    marker = write_attempt(tmp_path, task, execution, "bad-bound", result, arrays)
    with pytest.raises(ValueError):
        storage.read_episode(marker)


def test_rehashed_singleton_bounds_and_sentinels(tmp_path):
    task, execution, result, arrays = build("full12", snapshot_fn=speed_five_snapshots, density=0.005)
    for name, value in (("long_pv_leader_speed", 0), ("long_pv_eq9_tight", 0),
                        ("long_pv_prev_gap_sS", 397), ("long_pv_cur_gap_sS", 397)):
        bad = {key: array.copy() for key, array in arrays.items()}
        bad[name][0] = value
        marker = write_attempt(tmp_path, task, execution, name, result, bad)
        with pytest.raises(ValueError):
            storage.read_episode(marker)

