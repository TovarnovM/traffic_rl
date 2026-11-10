"""Independent hand-array tests for offline episode diagnostics.

No simulator, Torch, Ray, Docker or cluster runtime is used.  Fixtures are
built here from first principles so expected event boundaries, censoring,
covariance and sufficient statistics do not depend on the producer helpers.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from physica_mc import postprocess, schema

FIXTURE = Path(__file__).resolve().parents[1] / "configs" / "technical_analysis.json"


# ---------------------------------------------------------------------------
# Hand fixtures
# ---------------------------------------------------------------------------


def bounds_for_speed(speed):
    """Observer v1 canonical window bounds, written independently here."""

    return np.asarray(
        [
            pair
            for d in (5, max(int(speed), 1))
            for m in (1, 2, 3)
            for pair in ((-2 * m * d, 2 * m * d), (-m * d, 3 * m * d))
        ],
        dtype=np.int16,
    )


def make_arrays(
    steps,
    pv_speed,
    *,
    lanes=2,
    road_length=200,
    initial_pos=100,
    window_hist=None,
    initial_window_hist=None,
    window_transitions=None,
    global_hist=None,
    global_lane_changes=None,
    local_occupancy=None,
):
    pv_speed = np.asarray(pv_speed, dtype=np.uint8)
    if pv_speed.shape != (steps + 1,):
        raise AssertionError("fixture pv_speed must have length T+1")
    increments = np.r_[0, pv_speed[1:].astype(np.int64)]
    path = initial_pos + np.cumsum(increments)
    pos = path % road_length
    bounds = np.stack([bounds_for_speed(int(value)) for value in pv_speed], axis=0)
    arrays = {
        "window_hist": np.zeros((steps, 12, 3, lanes, 2, 6), dtype=np.uint16),
        "initial_window_hist": np.zeros((12, 3, lanes, 2, 6), dtype=np.uint16),
        "global_hist": np.zeros((steps, lanes, 2, 6), dtype=np.uint16),
        "initial_global_hist": np.zeros((lanes, 2, 6), dtype=np.uint16),
        "window_bounds": bounds,
        "window_transitions": np.zeros((steps, 12, 3, lanes, 2, 4), dtype=np.uint16),
        "global_lane_changes": np.zeros((steps, lanes, 2), dtype=np.uint16),
        "step": np.arange(steps + 1, dtype=np.int64),
        "pv_speed": pv_speed,
        "pv_lane": np.zeros(steps + 1, dtype=np.uint8),
        "pv_pos": pos.astype(np.int64),
        "pv_path": path.astype(np.int64),
        "spatial_hist": np.zeros((91, lanes, 2, 6), dtype=np.uint64),
        "spatial_edges": np.arange(-36, 56, dtype=np.int16),
        "local_occupancy": np.zeros((steps, 41, lanes), dtype=np.uint8),
    }
    supplied = {
        "window_hist": window_hist,
        "initial_window_hist": initial_window_hist,
        "window_transitions": window_transitions,
        "global_hist": global_hist,
        "global_lane_changes": global_lane_changes,
        "local_occupancy": local_occupancy,
    }
    for name, value in supplied.items():
        if value is not None:
            value = np.asarray(value)
            if value.shape != arrays[name].shape:
                raise AssertionError(
                    f"fixture {name} shape {value.shape} != {arrays[name].shape}"
                )
            arrays[name] = value.astype(arrays[name].dtype)
    return arrays


def make_metrics(**overrides):
    metrics = {
        "measurement_steps": 1,
        "vehicle_count": 1,
        "priority_vehicle_count": 1,
        "density": 0.0,
        "flow_all": 0.0,
        "flow_background": 0.0,
        "flow_priority": 0.0,
        "mean_speed_all": 0.0,
        "mean_speed_background": 0.0,
        "mean_speed_priority": 0.0,
        "worst_priority_mean_speed": 0.0,
        "mean_priority_time_loss": 0.0,
        "worst_priority_time_loss": 0.0,
        "stopped_fraction_all": 0.0,
        "stopped_fraction_background": 0.0,
        "stopped_fraction_priority": 0.0,
        "lane_change_rate_all": 0.0,
        "lane_change_rate_background": 0.0,
        "lane_change_rate_priority": 0.0,
        "completed_priority_laps": 0,
        "mean_priority_lap_time": None,
        "rule_detections": 0,
        "rule_requests": 0,
        "rule_applied": 0,
        "rule_rejected": 0,
        "rule_blocked_no_safe_lane": 0,
        "rule_cooldown_suppressed": 0,
    }
    metrics.update(overrides)
    return metrics


def make_payload(
    steps,
    lanes,
    *,
    task_id="a" * 64,
    execution_id="b" * 64,
    attempt_id="attempt-1",
    method="B1",
    split="pilot",
    density_index=3,
    replicate=1,
    priority_id=10,
    observer_profile="mechanism",
    purpose="technical",
    training_line=None,
    metrics=None,
    control=None,
    road_length=200,
):
    config = {
        "road_length": road_length,
        "num_lanes": lanes,
        "density": 0.2,
        "av_fraction": 0.95,
        "measure_steps": steps,
        "warmup_steps": 1000,
        "cooldown_steps": 1,
        "vmax_hdv": 5,
        "vmax_priority": 6,
    }
    result = {
        "schema_version": 1,
        "protocol": "physica-a-e2-native-eq9-v1",
        "task_id": task_id,
        "execution_id": execution_id,
        "purpose": purpose,
        "split": split,
        "method": method,
        "architecture": None,
        "density_index": density_index,
        "replicate": replicate,
        "config": config,
        "priority_id": priority_id,
        "observer_profile": observer_profile,
        "steps_completed": steps,
        "measurement_steps": steps,
        "warmup_steps_completed": 1000,
        "metrics": metrics or make_metrics(measurement_steps=steps),
        "control": control or {},
    }
    task = {
        "schema_version": 1,
        "task_id": task_id,
        "science": {
            "protocol": "physica-a-e2-native-eq9-v1",
            "purpose": purpose,
            "split": split,
            "method": method,
            "density_index": density_index,
            "replicate": replicate,
            "config": config,
            "training_line": training_line,
            "model_id": None,
            "candidate_id": None,
        },
    }
    execution = {
        "schema_version": 1,
        "execution_id": execution_id,
        "task_id": task_id,
        "profile": {"device": "cpu", "threads": 1, "observer": observer_profile},
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "attempt_id": attempt_id,
        "task": task,
        "execution": execution,
        "result": result,
        "sidecar": {"path": "arrays.npz", "size": 123, "sha256": "c" * 64},
    }


def profile(**overrides):
    value = {
        "schema_version": 1,
        "profile_id": "hand-test-technical",
        "technical_only": True,
        "events": [
            {"id": "pv_stop", "condition": "pv_speed_eq_0", "min_duration": 2},
            {"id": "pv_low_progress", "condition": "pv_speed_le_1", "min_duration": 2},
            {
                "id": "primary_front_stagnation",
                "condition": "nonempty_all_stopped",
                "window": "fixed_front_2",
                "region": "front",
                "min_duration": 2,
            },
        ],
        "spatial": {
            "measure": "background_occupancy_connected_covariance",
            "window": "fixed_front_2",
            "lags": list(range(11)),
            "centering": "per_step_per_lane_full_window_mean",
            "normalization": "none",
            "pair_domain": "within_window_no_wrap",
        },
        "notes": "hand test fixture",
    }
    value.update(overrides)
    return value


def analyze(steps, pv_speed, *, arrays=None, payload=None, profile_value=None, **kwargs):
    if arrays is None:
        arrays = make_arrays(steps, pv_speed, **kwargs)
    if payload is None:
        payload = make_payload(
            steps,
            arrays["window_hist"].shape[3],
            road_length=kwargs.get("road_length", 200),
        )
    analysis = postprocess.analyze_episode(
        payload, arrays, profile_value if profile_value is not None else profile()
    )
    return analysis, arrays, payload


def find_stream(analysis, condition):
    for stream in analysis["events"]["streams"]:
        if stream["condition"] == condition:
            return stream
    raise AssertionError(f"missing event stream {condition}")


def find_summary(analysis, window_id, region, lane, group):
    for record in analysis["windows"]["summaries"]:
        if (
            record["window_id"] == window_id
            and record["region"] == region
            and record["lane"] == lane
            and record["group"] == group
        ):
            return record
    raise AssertionError("missing window summary scope")


def set_counts(hist, window, region, lane, group, speed, counts):
    hist[:, window, region, lane, group, speed] = np.asarray(counts, dtype=hist.dtype)


def test_technical_fixture_loads_but_is_refused_as_scientific():
    loaded = postprocess.load_profile(FIXTURE)
    assert loaded["purpose"] == "technical"
    assert loaded["technical_only"] is True
    assert loaded["spatial"]["lags"] == list(range(11))
    assert {event["condition"] for event in loaded["events"]} == {
        "pv_speed_eq_0",
        "pv_speed_le_1",
        "nonempty_all_stopped",
    }
    with pytest.raises(ValueError, match="scientific"):
        postprocess.load_profile(FIXTURE, require_scientific=True)
    with pytest.raises(ValueError, match="scientific"):
        postprocess.require_scientific_profile(FIXTURE)
    scientific = copy.deepcopy(loaded)
    scientific["technical_only"] = False
    scientific["purpose"] = "scientific"
    scientific["profile_id"] = "explicit-scientific-profile"
    normalized = postprocess.require_scientific_profile(scientific)
    assert normalized["purpose"] == "scientific"
    assert normalized["technical_only"] is False


def test_profile_requires_explicit_conditions_durations_and_spatial_choices():
    base = profile()
    postprocess.validate_profile(base)
    cases = []
    missing_duration = copy.deepcopy(base)
    del missing_duration["events"][0]["min_duration"]
    cases.append(missing_duration)
    zero_duration = copy.deepcopy(base)
    zero_duration["events"][0]["min_duration"] = 0
    cases.append(zero_duration)
    missing_event = copy.deepcopy(base)
    missing_event["events"] = missing_event["events"][:2]
    cases.append(missing_event)
    duplicate_condition = copy.deepcopy(base)
    duplicate_condition["events"][1]["condition"] = "pv_speed_eq_0"
    cases.append(duplicate_condition)
    bad_window = copy.deepcopy(base)
    bad_window["events"][2]["window"] = "not_a_window"
    cases.append(bad_window)
    bad_lags = copy.deepcopy(base)
    bad_lags["spatial"]["lags"] = [0, 2, 1]
    cases.append(bad_lags)
    no_zero_lag = copy.deepcopy(base)
    no_zero_lag["spatial"]["lags"] = [1, 2]
    cases.append(no_zero_lag)
    normalised = copy.deepcopy(base)
    normalised["spatial"]["normalization"] = "pearson"
    cases.append(normalised)
    missing_spatial = copy.deepcopy(base)
    del missing_spatial["spatial"]["normalization"]
    cases.append(missing_spatial)
    no_spatial_choice = copy.deepcopy(base)
    del no_spatial_choice["spatial"]
    cases.append(no_spatial_choice)
    unknown = copy.deepcopy(base)
    unknown["extra_option"] = 1
    cases.append(unknown)
    for case in cases:
        with pytest.raises(ValueError):
            postprocess.validate_profile(case)
    implicit_scientific = copy.deepcopy(base)
    implicit_scientific["technical_only"] = False
    with pytest.raises(ValueError, match="purpose"):
        postprocess.validate_profile(implicit_scientific)
    contradictory = copy.deepcopy(base)
    contradictory["purpose"] = "scientific"
    with pytest.raises(ValueError, match="technical_only"):
        postprocess.validate_profile(contradictory)


def test_explicit_spatial_null_roundtrips_full12_without_mechanism_maps(
    tmp_path, monkeypatch
):
    windows_profile = profile(spatial=None)
    normalized = postprocess.validate_profile(windows_profile)
    assert normalized["spatial"] is None
    steps = 2
    arrays = make_arrays(steps, [2, 2, 2])
    arrays.pop("local_occupancy")
    payload = make_payload(
        steps, arrays["window_hist"].shape[3], observer_profile="full12"
    )
    analysis = postprocess.analyze_episode(payload, arrays, windows_profile)
    assert analysis["spatial_status"] == "not_requested"
    assert analysis["spatial_covariance"] is None
    assert len(analysis["windows"]["state_frequencies"]) == 12
    assert analysis["timing_convention"]["spatial_covariance"] == (
        "not requested by the analysis profile"
    )
    # An explicitly requested covariance block still needs mechanism maps.
    with pytest.raises(ValueError, match="local_occupancy|mechanism"):
        postprocess.analyze_episode(payload, arrays, profile())

    attempt, _marker = _write_fake_episode(tmp_path)
    calls = []

    def fake_read(path, **kwargs):
        calls.append(kwargs)
        return payload, arrays

    monkeypatch.setattr(postprocess.storage, "read_episode", fake_read)
    target = postprocess.write_analysis(attempt, windows_profile, tmp_path / "out")
    assert calls[0]["required_profile"] == "full12"
    written = json.loads(target.read_bytes())
    assert written["spatial_status"] == "not_requested"
    assert written["spatial_covariance"] is None
    assert written["lineage"]["input"]["required_reader_profile"] == "full12"
    assert written["lineage"]["input"]["observer_profile"] == "full12"


def test_pv_stop_and_low_progress_boundaries_confirmation_restart_and_terminal():
    steps = 6
    speeds = [2, 0, 0, 1, 2, 0, 0]
    analysis, _, _ = analyze(steps, speeds)
    stop = find_stream(analysis, "pv_speed_eq_0")
    low = find_stream(analysis, "pv_speed_le_1")
    assert stop["confirmed_count"] == 2
    assert low["confirmed_count"] == 2

    first, terminal = stop["events"]
    assert first["onset_step"] == 1
    assert first["observed_start"] == 1
    assert first["confirmation_step"] == 2
    assert first["observed_end"] == 2
    assert first["end_step"] == 2
    assert first["duration_steps"] == 2
    assert first["left_censored"] is False
    assert first["right_censored"] is False
    assert first["exit_reason"] == "pv_restart"
    assert first["restart_observed"] is True
    assert first["restart_step"] == 3
    assert first["exit_observation"]["pv_speed"] == 1

    assert terminal["onset_step"] == 5
    assert terminal["observed_start"] == 5
    assert terminal["confirmation_step"] == 6
    assert terminal["observed_end"] == 6
    assert terminal["end_step"] is None
    assert terminal["duration_steps"] is None
    assert terminal["observed_duration_steps"] == 2
    assert terminal["right_censored"] is True
    assert terminal["exit_reason"] == "right_censored_horizon"
    assert terminal["restart_observed"] is False
    assert terminal["restart_step"] is None

    low_first, low_terminal = low["events"]
    assert low_first["observed_start"] == 1
    assert low_first["confirmation_step"] == 2
    assert low_first["observed_end"] == 3
    assert low_first["duration_steps"] == 3
    assert low_first["restart_observed"] is True
    assert low_first["restart_step"] == 4
    assert low_terminal["right_censored"] is True
    assert low_terminal["restart_observed"] is False
    assert low["descriptive_duration"]["mask"] == "fully_observed_only"
    assert low["descriptive_duration"]["n"] == 1


def test_event_left_censoring_uses_separate_t0_context():
    steps = 4
    analysis, _, _ = analyze(steps, [0, 0, 0, 2, 0])
    stop = find_stream(analysis, "pv_speed_eq_0")
    assert stop["confirmed_count"] == 1
    event = stop["events"][0]
    assert event["left_censored"] is True
    assert event["onset_step"] is None
    assert event["observed_start"] == 1
    assert event["confirmation_step"] == 2
    assert event["observed_end"] == 2
    assert event["end_step"] == 2
    assert event["duration_steps"] is None
    assert event["observed_duration_steps"] == 2
    assert event["right_censored"] is False
    assert event["restart_observed"] is True
    assert stop["unconfirmed_terminal_runs"] == 1

    # t=0 context is not itself a measured sample: if condition is already
    # false at t=1 there is no measured event.
    quiet, _, _ = analyze(2, [0, 2, 2])
    assert find_stream(quiet, "pv_speed_eq_0")["confirmed_count"] == 0

    # A terminal stop is confirmed and right-censored, never given a restart.
    terminal, _, _ = analyze(5, [2, 0, 0, 2, 0, 0])
    last = find_stream(terminal, "pv_speed_eq_0")["events"][-1]
    assert last["observed_start"] == 4
    assert last["observed_end"] == 5
    assert last["right_censored"] is True
    assert last["exit_reason"] == "right_censored_horizon"
    assert last["restart_observed"] is False


def test_window_all_stopped_event_exits_on_empty_window_not_pv_restart():
    steps = 6
    lanes = 2
    hist = np.zeros((steps, 12, 3, lanes, 2, 6), dtype=np.uint16)
    entries = [(2, 0), (2, 0), (0, 0), (1, 0), (1, 1), (0, 0)]
    for index, (count, speed) in enumerate(entries):
        if count:
            hist[index, 3, 2, 0, 0, speed] = count
    analysis, _, _ = analyze(
        steps, [0, 0, 0, 0, 0, 0, 0], arrays=make_arrays(steps, [0] * (steps + 1), window_hist=hist)
    )
    stream = find_stream(analysis, "nonempty_all_stopped")
    assert stream["confirmed_count"] == 1
    event = stream["events"][0]
    assert event["window"] == "fixed_front_2"
    assert event["region"] == "front"
    assert event["lane"] == "all"
    assert event["observed_start"] == 1
    assert event["observed_end"] == 2
    assert event["duration_steps"] == 2
    assert event["exit_reason"] == "window_empty"
    assert event["empty_window_exit"] is True
    assert event["restart_observed"] is False
    assert event["exit_observation"]["N"] == 0
    assert stream["unconfirmed_runs"] == 1
    assert stream["unconfirmed_terminal_runs"] == 0


def test_window_all_stopped_exit_reasons_and_left_censoring():
    lanes = 2
    # Resumed movement: the window remains nonempty but some vehicle moves.
    steps = 4
    hist = np.zeros((steps, 12, 3, lanes, 2, 6), dtype=np.uint16)
    for index, (count, speed) in enumerate([(2, 0), (2, 0), (1, 1), (0, 0)]):
        if count:
            hist[index, 3, 2, 0, 0, speed] = count
    analysis, _, _ = analyze(
        steps,
        [0, 0, 0, 1, 0],
        arrays=make_arrays(steps, [0, 0, 0, 1, 0], window_hist=hist),
    )
    event = find_stream(analysis, "nonempty_all_stopped")["events"][0]
    assert event["exit_reason"] == "vehicles_resumed"
    assert event["empty_window_exit"] is False
    assert event["restart_observed"] is True
    assert event["restart_step"] == 3

    # Left-censored local event: t=0 context was already nonempty/all-stopped.
    initial = np.zeros((12, 3, lanes, 2, 6), dtype=np.uint16)
    initial[3, 2, 0, 0, 0] = 2
    hist2 = np.zeros((3, 12, 3, lanes, 2, 6), dtype=np.uint16)
    hist2[0, 3, 2, 0, 0, 0] = 2
    hist2[1, 3, 2, 0, 0, 0] = 2
    analysis2, _, _ = analyze(
        3,
        [0, 0, 0, 0],
        arrays=make_arrays(
            3,
            [0, 0, 0, 0],
            window_hist=hist2,
            initial_window_hist=initial,
        ),
    )
    event2 = find_stream(analysis2, "nonempty_all_stopped")["events"][0]
    assert event2["left_censored"] is True
    assert event2["onset_step"] is None
    assert event2["duration_steps"] is None
    assert event2["exit_reason"] == "window_empty"
    assert event2["restart_observed"] is False


def test_empty_window_is_not_an_event_and_does_not_create_pv_restart():
    steps = 2
    lanes = 2
    hist = np.zeros((steps, 12, 3, lanes, 2, 6), dtype=np.uint16)
    analysis, _, _ = analyze(
        steps,
        [0, 0, 0],
        arrays=make_arrays(steps, [0, 0, 0], window_hist=hist),
    )
    stream = find_stream(analysis, "nonempty_all_stopped")
    assert stream["confirmed_count"] == 0
    assert stream["events"] == []
    # PV stop events are independent; an empty local window does not fabricate
    # or suppress them.
    assert find_stream(analysis, "pv_speed_eq_0")["confirmed_count"] == 1


def test_spatial_covariance_matches_independent_loop_and_has_no_periodic_pairs():
    steps = 3
    lanes = 2
    occupancy = np.zeros((steps, 41, lanes), dtype=np.uint8)
    occupancy[:, 0, 0] = 1
    occupancy[:, 0:10, 1] = 1
    analysis, _, _ = analyze(
        steps,
        [1, 1, 1, 1],
        arrays=make_arrays(steps, [1, 1, 1, 1], local_occupancy=occupancy),
    )
    spatial = analysis["spatial_covariance"]
    lags = list(range(11))
    expected = np.zeros((steps, lanes, len(lags)), dtype=np.float64)
    for t in range(steps):
        for lane in range(lanes):
            values = occupancy[t, :, lane].astype(np.float64)
            centered = values - values.mean()
            for position, lag in enumerate(lags):
                products = (
                    centered * centered
                    if lag == 0
                    else centered[: 41 - lag] * centered[lag:]
                )
                expected[t, lane, position] = products.mean()
    overall = expected.mean(axis=(0, 1))
    per_lane = expected.mean(axis=0)
    np.testing.assert_allclose(spatial["covariance"], overall, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(spatial["lag0_variance"], overall[0], rtol=1e-12)
    assert spatial["pair_counts"] == [steps * lanes * (41 - lag) for lag in lags]
    assert spatial["covariance0"] == pytest.approx(overall[0], rel=1e-12, abs=1e-15)
    assert spatial["is_correlation"] is False
    assert spatial["defined"] is True
    for lane in range(lanes):
        lane_record = spatial["by_lane"][lane]
        np.testing.assert_allclose(
            lane_record["covariance"], per_lane[lane], rtol=1e-12, atol=1e-15
        )
        assert lane_record["pair_counts"] == [steps * (41 - lag) for lag in lags]
        assert lane_record["lag0_variance"] == pytest.approx(
            per_lane[lane][0], rel=1e-12
        )

    # Prove that the implementation did not wrap the window end.
    values = occupancy[0, :, 1].astype(np.float64)
    centered = values - values.mean()
    wrapped = np.mean([centered[i] * centered[(i + 10) % 41] for i in range(41)])
    assert abs(wrapped - expected[0, 1, 10]) > 1e-6
    assert spatial["covariance"][10] == pytest.approx(
        expected[:, :, 10].mean(), rel=1e-12, abs=1e-15
    )


def test_spatial_covariance_zero_variance_remains_defined_and_is_not_correlation():
    steps = 2
    lanes = 2
    occupancy = np.ones((steps, 41, lanes), dtype=np.uint8)
    analysis, _, _ = analyze(
        steps,
        [2, 2, 2],
        arrays=make_arrays(steps, [2, 2, 2], local_occupancy=occupancy),
    )
    spatial = analysis["spatial_covariance"]
    assert spatial["lag0_variance"] == 0.0
    assert spatial["zero_variance_frames"] == steps * lanes
    assert spatial["zero_variance_fraction"] == 1.0
    assert spatial["covariance"] == [0.0] * len(spatial["lags"])
    assert spatial["covariance0"] == 0.0
    assert spatial["defined"] is True
    assert spatial["is_correlation"] is False
    assert spatial["measure_kind"] == "covariance_not_pearson_correlation"


def test_spatial_measure_rejects_missing_maps_and_never_uses_spatial_hist():
    steps = 2
    arrays = make_arrays(steps, [0, 0, 0])
    arrays.pop("local_occupancy")
    arrays["spatial_hist"] = np.ones_like(arrays["spatial_hist"])
    payload = make_payload(steps, arrays["window_hist"].shape[3])
    with pytest.raises(ValueError, match="local_occupancy|mechanism"):
        postprocess.analyze_episode(payload, arrays, profile())


def test_window_summaries_use_exact_raw_sufficient_statistics_and_none():
    steps = 2
    lanes = 2
    hist = np.zeros((steps, 12, 3, lanes, 2, 6), dtype=np.uint16)
    # Primary window front region: one HDV speed 2 and one AV speed 4 in step 1.
    hist[0, 3, 2, 0, 0, 2] = 1
    hist[0, 3, 2, 1, 1, 4] = 1
    transitions = np.zeros((steps, 12, 3, lanes, 2, 4), dtype=np.uint16)
    transitions[0, 3, 2, 0, 0, 0] = 1  # stop_to_move
    transitions[0, 3, 2, 1, 1, 2] = 2  # acceleration
    analysis, _, _ = analyze(
        steps,
        [0, 0, 0],
        arrays=make_arrays(
            steps,
            [0, 0, 0],
            window_hist=hist,
            window_transitions=transitions,
        ),
    )
    full = find_summary(analysis, "fixed_front_2", "full", "all", "background")
    assert (full["N"], full["S"], full["Z"], full["A"]) == (2, 6, 0, 164)
    assert full["defined_steps"] == 1
    assert full["empty_steps"] == 1
    assert full["empty_fraction"] == 0.5
    assert full["pooled_speed"] == 3.0
    assert full["pooled_stopped_fraction"] == 0.0
    assert full["cell_time_density"] == pytest.approx(2 / 164)
    assert full["cell_time_flow"] == pytest.approx(6 / 164)
    assert full["time_mean_density"] == pytest.approx(1 / 82)
    assert full["time_mean_flow"] == pytest.approx(3 / 82)

    front = find_summary(analysis, "fixed_front_2", "front", "all", "background")
    assert (front["N"], front["S"], front["Z"], front["A"]) == (2, 6, 0, 120)

    empty = find_summary(analysis, "fixed_front_2", "rear", 0, "HDV")
    assert (empty["N"], empty["S"], empty["Z"], empty["A"]) == (0, 0, 0, 20)
    assert empty["pooled_speed"] is None
    assert empty["pooled_stopped_fraction"] is None
    assert empty["speed_deficit"] is None
    assert empty["normalized_local_delay"] is None
    assert empty["cell_time_density"] == 0.0
    assert empty["cell_time_flow"] == 0.0
    assert empty["empty_fraction"] == 1.0
    assert empty["defined_steps"] == 0

    transition = None
    for record in analysis["windows"]["transitions"]:
        if (
            record["window_id"] == "fixed_front_2"
            and record["region"] == "front"
            and record["lane"] == "all"
            and record["group"] == "background"
        ):
            transition = record
            break
    assert transition is not None
    assert transition["counts"] == [1, 0, 2, 0]
    assert transition["exposure_N"] == 2
    assert transition["rates"] == [0.5, 0.0, 1.0, 0.0]


def test_state_frequencies_are_exact_joint_n_s_z_a_counts():
    steps = 3
    lanes = 1
    hist = np.zeros((steps, 12, 3, lanes, 2, 6), dtype=np.uint16)
    hist[0, 3, 2, 0, 0, 0] = 1
    hist[1, 3, 2, 0, 0, 0] = 1
    hist[2, 3, 2, 0, 0, 1] = 2
    analysis, _, _ = analyze(
        steps,
        [0, 0, 0, 0],
        arrays=make_arrays(steps, [0, 0, 0, 0], window_hist=hist, lanes=lanes),
    )
    primary = None
    for record in analysis["windows"]["state_frequencies"]:
        if record["window_id"] == "fixed_front_2":
            primary = record
            break
    assert primary is not None
    assert primary["steps"] == 3
    states = {(state["N"], state["S"], state["Z"], state["A"]): state for state in primary["states"]}
    assert set(states) == {(1, 0, 1, 41), (2, 2, 0, 41)}
    assert states[(1, 0, 1, 41)]["count"] == 2
    assert states[(1, 0, 1, 41)]["frequency"] == pytest.approx(2 / 3)
    assert states[(2, 2, 0, 41)]["count"] == 1
    assert states[(2, 2, 0, 41)]["frequency"] == pytest.approx(1 / 3)
    assert sum(state["count"] for state in primary["states"]) == steps


def test_window_geometry_records_actual_bounds_widths_and_floor_steps():
    steps = 3
    analysis, _, _ = analyze(steps, [0, 0, 1, 2])
    geometry = {
        record["window_id"]: record for record in analysis["windows"]["geometry"]
    }
    primary = geometry["fixed_front_2"]
    assert primary["low_min"] == -10
    assert primary["high_max"] == 30
    assert primary["width_values"] == [{"width": 41, "steps": steps}]
    speed_front = geometry["speed_front_2"]
    assert speed_front["floor_applies"] is True
    assert speed_front["pv_speed_below_1_steps"] == 1
    assert speed_front["pv_speed_eq_1_steps"] == 1
    widths = {item["width"]: item["steps"] for item in speed_front["width_values"]}
    assert widths == {9: 2, 17: 1}


def test_analysis_is_pure_marks_one_episode_and_does_not_mutate_inputs():
    steps = 3
    arrays = make_arrays(steps, [0, 1, 2, 0])
    payload = make_payload(steps, arrays["window_hist"].shape[3])
    profile_value = profile()
    arrays_before = {name: value.copy() for name, value in arrays.items()}
    payload_before = schema.canonical_json(payload)
    profile_before = schema.canonical_json(profile_value)
    analysis = postprocess.analyze_episode(payload, arrays, profile_value)
    for name, value in arrays_before.items():
        np.testing.assert_array_equal(arrays[name], value)
    assert schema.canonical_json(payload) == payload_before
    assert schema.canonical_json(profile_value) == profile_before
    assert analysis["scientific_unit"] == "episode"
    assert analysis["independence"]["independent_episodes"] == 1
    assert analysis["events"]["summary"]["scientific_unit"] == "episode"
    assert len(analysis["windows"]["summaries"]) == 12 * 4 * 3 * 3
    assert analysis["spatial_covariance"]["time_accumulated_spatial_hist_used"] is False
    assert analysis["spatial_status"] == "computed"
    assert analysis["analysis_scope"] == "technical"


# ---------------------------------------------------------------------------
# Immutable publication
# ---------------------------------------------------------------------------


def _write_fake_episode(tmp_path):
    attempt = tmp_path / "input" / "attempt-1"
    attempt.mkdir(parents=True)
    marker = attempt / "completion.json"
    marker.write_bytes(b'{"fake":"marker"}\n')
    (attempt / "arrays.npz").write_bytes(b"fake arrays sidecar\n")
    return attempt, marker


def _fixture_payload_arrays():
    steps = 2
    arrays = make_arrays(steps, [2, 2, 2])
    payload = make_payload(steps, arrays["window_hist"].shape[3])
    return payload, arrays


def test_write_analysis_uses_reader_and_publishes_immutable_hashed_derivative(
    tmp_path, monkeypatch
):
    attempt, marker = _write_fake_episode(tmp_path)
    payload, arrays = _fixture_payload_arrays()
    calls = []

    def fake_read_episode(path, **kwargs):
        calls.append((Path(path), kwargs))
        return payload, arrays

    monkeypatch.setattr(postprocess.storage, "read_episode", fake_read_episode)
    output = tmp_path / "derivatives"
    target = postprocess.write_analysis(attempt, FIXTURE, output)
    assert calls[0][1]["required_profile"] == "mechanism"
    assert target.exists()
    assert target.name == "analysis.json"
    encoded = target.read_bytes()
    analysis = json.loads(encoded)
    assert target.parent.name == analysis["analysis_id"]
    assert target.parent.parent.name == "analysis"
    assert target.parents[2].name == "attempt-1"
    normalized = postprocess.load_profile(FIXTURE)
    assert analysis["lineage"]["input"]["marker_sha256"] == schema.file_sha256(marker)
    assert analysis["lineage"]["input"]["arrays_sha256"] == "c" * 64
    assert analysis["lineage"]["input"]["required_reader_profile"] == "mechanism"
    assert analysis["profile_sha256"] == schema.digest(normalized)
    assert len(analysis["analysis_id"]) == 64
    assert len(analysis["lineage"]["code"]["source_digest"]) == 64
    assert analysis["scientific_unit"] == "episode"
    assert analysis["events"]["summary"]["confirmed_events"] == 0

    same = postprocess.write_analysis(attempt, FIXTURE, output)
    assert same == target
    assert target.read_bytes() == encoded

    target.write_bytes(b"{}")
    with pytest.raises(ValueError, match="different content"):
        postprocess.write_analysis(attempt, FIXTURE, output)
    assert target.read_bytes() == b"{}"


def test_write_analysis_refuses_a_changed_input_marker(tmp_path, monkeypatch):
    attempt, marker = _write_fake_episode(tmp_path)
    payload, arrays = _fixture_payload_arrays()

    def changing_read(path, **kwargs):
        Path(path).write_bytes(b"changed during analysis")
        return payload, arrays

    monkeypatch.setattr(postprocess.storage, "read_episode", changing_read)
    with pytest.raises(ValueError, match="changed"):
        postprocess.write_analysis(attempt, FIXTURE, tmp_path / "derivatives")


def test_global_lane_changes_use_matching_background_exposure():
    hist = np.zeros((2, 2, 2, 6), dtype=np.uint16)
    hist[:, 0, 0, 2] = 2
    hist[:, 1, 1, 3] = 3
    changes = np.zeros((2, 2, 2), dtype=np.uint16)
    changes[0, 0, 0] = 1
    changes[1, 1, 1] = 2
    analysis, _, _ = analyze(2, [0, 0, 0], arrays=make_arrays(
        2, [0, 0, 0], global_hist=hist, global_lane_changes=changes,
    ))
    records = {(row["lane"], row["group"]): row
               for row in analysis["global_lane_changes"]}
    for key, count, exposure in [((0, "HDV"), 1, 4), ((1, "AV"), 2, 6),
                                 (("all", "background"), 3, 10)]:
        assert records[key]["changes"] == count
        assert records[key]["exposure_N"] == exposure
        assert records[key]["rate"] == pytest.approx(count / exposure)
    assert records[(0, "AV")]["exposure_N"] == 0
    assert records[(0, "AV")]["rate"] is None


# W5 fixtures intentionally do not import the kernel/diagnostics producer.
def mechanism_profile():
    value = json.loads((FIXTURE.parent / "technical_mechanism.json").read_text())
    value["spatial"] = None
    return value


def mechanism_fixture():
    speed = [6, 0, 0, 1, 0, 0]
    arrays = make_arrays(5, speed)
    payload = make_payload(5, 2, method="B2", observer_profile="full12")
    for phase in ("pre", "long"):
        arrays[f"{phase}_pv_leader_gap"] = np.array([-1, 0, 2, 0, 0], dtype=np.int32)
        for suffix in ("type", "speed"):
            arrays[f"{phase}_pv_leader_{suffix}"] = np.array([-1, 0, 0, 0, 0], dtype=np.int8)
    arrays["pre_pv_leader_stopped"] = np.array([-1, 1, 1, 1, 1], dtype=np.int8)
    arrays["long_pv_leader_final_speed"] = np.array([-1, 0, 0, 0, 0], dtype=np.int8)
    arrays["long_pv_eq9_slack"] = np.array([-1, 0, 1, 0, 0], dtype=np.int32)
    arrays["long_pv_eq9_tight"] = np.array([-1, 1, 0, 1, 1], dtype=np.int8)
    arrays["long_pv_acceleration_gate"] = np.array([1, 0, 1, 0, 1], dtype=np.int8)
    arrays["long_pv_v1"] = np.ones(5, dtype=np.int8)
    for phase in ("pre", "native_phase"):
        for side in ("left", "count"):
            arrays[f"{phase}_pv_native_eligible_{side}"] = np.array([0, 0, 1, 0, 0], dtype=np.int8)
        arrays[f"{phase}_pv_native_eligible_right"] = np.zeros(5, dtype=np.int8)
    for side in ("left", "right"):
        arrays[f"pre_pv_native_{side}"] = np.zeros(5, dtype=np.int8)
        arrays[f"pre_pv_native_{side}_reason"] = np.ones(5, dtype=np.int8)
    for phase in ("prev", "cur"):
        for bound in ("s1", "sS"):
            arrays[f"long_pv_{phase}_gap_{bound}"] = np.arange(5, dtype=np.int64)
    names = ("step active_av selected_left selected_right selected_stay explicit_stay_commands "
             "available_left available_right eligible_nonstay common_throttled private_throttled "
             "private_suppressed_left common_suppressed requested_nonstay applied_nonstay rejected_nonstay "
             "legal_nonstay_requests applied_legal_nonstay_requests front_background centre_background "
             "rear_background front_hdv centre_hdv rear_hdv front_av centre_av rear_av "
             "requested_front requested_centre requested_rear applied_front applied_centre applied_rear").split()
    arrays["control_trace"] = np.zeros((5, 33), dtype=np.int64)
    arrays["control_trace"][:, 0] = np.arange(5)
    arrays["control_trace"][:, names.index("requested_front")] = [0, 1, 0, 2, 1]
    payload["result"]["control_columns"] = {
        name: {"index": index, "dtype": "int64"} for index, name in enumerate(names)
    }
    wi = postprocess.WINDOW_IDS.index(postprocess.PRIMARY_WINDOW)
    arrays["initial_window_hist"][wi, 2, 0, 0, 0] = 1
    arrays["window_hist"][-1, wi, 2, 0, 0, 0] = 9
    return payload, arrays


def test_mechanism_frequency_contexts_sentinels_and_immutability():
    payload, arrays = mechanism_fixture()
    chosen = mechanism_profile()
    before = {name: value.copy() for name, value in arrays.items()}
    metadata = schema.canonical_json(payload)
    profile_bytes = schema.canonical_json(chosen)
    analysis = postprocess.analyze_episode(payload, arrays, chosen)
    tables = {table["name"]: table for table in analysis["mechanism"]["tables"]}
    gap = tables["long_pv_leader_gap"]
    assert gap["contexts"] == 5 and gap["denominator"] == 4
    assert gap["sentinel"] == "no_leader" and gap["sentinel_count"] == 1
    assert gap["states"] == [{"value": 0, "count": 3, "frequency": .75},
                             {"value": 2, "count": 1, "frequency": .25}]
    assert [row["count"] for row in tables["native_phase_pv_native_eligible_count"]["states"]] == [4, 1, 0]
    assert [row["count"] for row in tables["native_phase_pv_native_direction_set"]["states"]] == [4, 1, 0, 0]
    for name in ("active_av", "available_left", "eligible_nonstay"):
        table = tables["control_" + name]
        assert table["denominator"] == 0 and table["mean"] is None
        assert table["sentinel_count"] == 5 and table["states"] == []
    assert tables["control_requested_front"]["mean"] == .8
    # Exactly decision contexts 0..4: initial count 1 included, final count 9 excluded.
    assert tables["primary_pre_front_HDV_count"]["states"] == [
        {"value": 0, "count": 4, "frequency": .8}, {"value": 1, "count": 1, "frequency": .2}]
    assert analysis["spatial_status"] == "not_requested"
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], before[name])
    assert schema.canonical_json(payload) == metadata
    assert schema.canonical_json(chosen) == profile_bytes
    assert analysis["profile_sha256"] == schema.digest(postprocess.validate_profile(chosen))


@pytest.mark.parametrize("name", ["long_pv_leader_gap", "native_phase_pv_native_eligible_count", "control_trace"])
def test_requested_mechanism_missing_columns_fail(name):
    payload, arrays = mechanism_fixture()
    del arrays[name]
    with pytest.raises(ValueError, match=name):
        postprocess.analyze_episode(payload, arrays, mechanism_profile())


def test_blocking_exact_conjunction_and_unknown_initial_context():
    chosen = mechanism_profile()
    chosen["mechanism"] = chosen["temporal"] = chosen["event_alignment"] = None
    arrays = make_arrays(6, [6, 0, 0, 0, 1, 0, 0])
    arrays["long_pv_leader_gap"] = np.array([0, -1, 0, 0, 1, 0], dtype=np.int32)
    arrays["native_phase_pv_native_eligible_count"] = np.array([0, 0, 1, 0, 0, 0], dtype=np.int8)
    analysis = postprocess.analyze_episode(make_payload(6, 2), arrays, chosen)
    block = analysis["operational_blocking"]
    assert block["label"] == "operational_descriptive"
    assert [event["observed_start"] for event in block["events"]] == [1, 6]
    first, last = block["events"]
    assert first["left_censored"] and first["onset_step"] is None
    assert first["exit_reason"] == "condition_ended" and not first["restart_observed"]
    assert last["right_censored"] and last["restart_step"] is None
    chosen["events"][-1]["min_duration"] = 2
    analysis = postprocess.analyze_episode(make_payload(6, 2), arrays, chosen)
    assert analysis["operational_blocking"]["confirmed_count"] == 0
    chosen["events"][-1]["logic"] = "pv_speed_eq_0"
    with pytest.raises(ValueError, match="blocking logic"):
        postprocess.validate_profile(chosen)


def test_stop_intervals_respect_threshold_and_boundary_censoring():
    # First run is left-censored, t=4 is an unconfirmed singleton, last run is terminal.
    analysis, _, _ = analyze(8, [0, 0, 0, 2, 0, 2, 2, 0, 0])
    intervals = analysis["events"]["pv_stop_intervals"]
    assert intervals["observed_interval_count"] == 1
    assert intervals["observed_restart_count"] == 1
    assert intervals["intervals"] == [{"previous_observed_end": 2, "restart_step": 3,
                                         "next_onset_step": 7, "observed_gap_steps": 4,
                                         "onset_to_onset_steps": None}]
    assert intervals["mean_observed_gap"] == 4
    assert intervals["min_duration"] == 2
    for name in ("mechanism", "temporal", "event_alignment", "operational_blocking"):
        assert analysis[name] is None and analysis[name + "_status"] == "not_requested"


@pytest.mark.parametrize("normalization", ["none", "lag0_variance"])
def test_temporal_no_wrap_pair_counts_constant_and_zero_pair_na(normalization):
    temporal = mechanism_profile()["temporal"]
    temporal["normalization"] = normalization
    chosen = profile(spatial=None, temporal=temporal)
    analysis, _, _ = analyze(3, [6, 2, 2, 2], profile_value=chosen)
    records = analysis["temporal"]["records"]
    assert [row["pair_count"] for row in records] == [3, 2, 1, 0, 0]
    assert [row["covariance"] for row in records] == [0., 0., 0., None, None]
    expected = [0., 0., 0., None, None] if normalization == "none" else [None] * 5
    assert [row["value"] for row in records] == expected
    analysis, _, _ = analyze(3, [6, 0, 1, 2], profile_value=chosen)
    assert [row["covariance"] for row in analysis["temporal"]["records"]] == [2 / 3, 0., -1., None, None]


@pytest.mark.parametrize("anchor,left,expected_counts", [
    ("onset_step", "exclude", [1, 1, 0]),
    ("observed_start", "include_observed_anchor", [1, 2, 1]),
    ("confirmation_step", "include_observed_anchor", [2, 2, 1]),
])
def test_event_alignment_boundaries_anchor_and_episode_n(anchor, left, expected_counts):
    aligned = mechanism_profile()["event_alignment"]
    aligned.update(anchor=anchor, left_censored=left, offsets=[-1, 0, 2])
    # Stops [1,2] left-censored and [5,6] right-censored; no t=0 sample.
    analysis, _, _ = analyze(6, [0, 0, 0, 2, 2, 0, 0],
                            profile_value=profile(spatial=None, event_alignment=aligned))
    block = analysis["event_alignment"]
    rows = [row for row in block["records"] if row["signal"] == "pv_speed"]
    assert [row["valid_count"] for row in rows] == expected_counts
    assert all(row["mean"] is None for row in rows if row["valid_count"] == 0)
    assert analysis["independence"]["independent_episodes"] == 1
    assert block["inferential_unit"] == "episode"
    assert {row["signal"] for row in block["records"]} == {
        "pv_speed", "primary_front_N", "primary_front_S", "primary_front_Z"}


def test_extension_choices_are_explicit_and_hash_sensitive():
    base = mechanism_profile()
    original = postprocess.validate_profile(base)
    for key, field in (("temporal", "pair_domain"), ("event_alignment", "left_censored")):
        bad = copy.deepcopy(base)
        del bad[key][field]
        with pytest.raises(ValueError):
            postprocess.validate_profile(bad)
    changed = copy.deepcopy(base)
    changed["event_alignment"]["offsets"] = [-1, 0, 1]
    assert schema.digest(postprocess.validate_profile(changed)) != schema.digest(original)
    with pytest.raises(ValueError, match="technical-only"):
        postprocess.validate_profile(base, require_scientific=True)


def test_mechanism_rejects_dtype_and_phase_contract_drift():
    payload, arrays = mechanism_fixture()
    arrays["long_pv_leader_gap"] = arrays["long_pv_leader_gap"].astype(np.int64)
    with pytest.raises(ValueError, match="dtype int32"):
        postprocess.analyze_episode(payload, arrays, mechanism_profile())
    payload, arrays = mechanism_fixture()
    arrays["native_phase_pv_native_eligible_left"][0] = 1
    arrays["native_phase_pv_native_eligible_count"][0] = 1
    with pytest.raises(ValueError, match="original pre-step geometry"):
        postprocess.analyze_episode(payload, arrays, mechanism_profile())


def test_all_no_leader_is_not_zero_gap_and_common_active_zero_is_observed():
    payload, arrays = mechanism_fixture()
    payload["result"]["method"] = "Graph"
    for name in ("pre_pv_leader_gap", "pre_pv_leader_type", "pre_pv_leader_speed", "pre_pv_leader_stopped",
                 "long_pv_leader_gap", "long_pv_leader_type", "long_pv_leader_speed",
                 "long_pv_leader_final_speed", "long_pv_eq9_slack", "long_pv_eq9_tight"):
        arrays[name][:] = -1
    analysis = postprocess.analyze_episode(payload, arrays, mechanism_profile())
    tables = {table["name"]: table for table in analysis["mechanism"]["tables"]}
    gap = tables["long_pv_leader_gap"]
    assert gap["denominator"] == 0 and gap["sentinel_count"] == 5
    assert gap["states"] == [] and gap["mean"] is None
    assert all(row["frequency"] is None for row in tables["long_pv_eq9_tight"]["states"])
    assert tables["control_active_av"]["denominator"] == 5
    assert tables["control_active_av"]["mean"] == 0.
    assert analysis["operational_blocking"]["confirmed_count"] == 0


def test_alignment_uses_measured_primary_front_sufficient_signals():
    arrays = make_arrays(4, [6, 2, 0, 0, 2])
    wi = postprocess.WINDOW_IDS.index(postprocess.PRIMARY_WINDOW)
    for index in range(4):
        arrays["window_hist"][index, wi, 2, 1, 0, 2] = index + 1
        arrays["window_hist"][index, wi, 0, 0, 1, 0] = 7  # rear excluded
    arrays["initial_window_hist"][wi, 2, 0, 0, 0] = 9  # context excluded
    aligned = mechanism_profile()["event_alignment"]
    aligned["offsets"] = [-1, 0, 1, 2]
    result = postprocess.analyze_episode(make_payload(4, 2), arrays,
                                        profile(spatial=None, event_alignment=aligned))
    records = result["event_alignment"]["records"]
    for signal, expected in (("pv_speed", [2., 0., 0., 2.]),
                             ("primary_front_N", [1., 2., 3., 4.]),
                             ("primary_front_S", [2., 4., 6., 8.]),
                             ("primary_front_Z", [0., 0., 0., 0.])):
        rows = [row for row in records if row["signal"] == signal]
        assert [row["mean"] for row in rows] == expected
        assert [row["valid_count"] for row in rows] == [1, 1, 1, 1]
