"""Streaming, coverage, sparse frequencies, plots and illustration selection.

Tests use hand-built analysis documents: no simulator, model, Torch, Ray,
Docker or cluster runtime.  Plot tests are tiny fixtures and save only from the
recorded structures passed in.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from physica_mc import reporting, schema

WINDOW_IDS = (
    "fixed_sym_1",
    "fixed_front_1",
    "fixed_sym_2",
    "fixed_front_2",
    "fixed_sym_3",
    "fixed_front_3",
    "speed_sym_1",
    "speed_front_1",
    "speed_sym_2",
    "speed_front_2",
    "speed_sym_3",
    "speed_front_3",
)
REGIONS = ("rear", "centre", "front", "full")
GROUPS = ("HDV", "AV", "background")


OFFICIAL_NAMES = (
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
    "rule_detections",
    "rule_requests",
    "rule_applied",
    "rule_rejected",
    "rule_blocked_no_safe_lane",
    "rule_cooldown_suppressed",
)
CONTROL_NAMES = (
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
    "requested_nonstay",
    "applied_nonstay",
    "rejected_nonstay",
    "emitted_nonstay",
    "legal_nonstay_requests",
    "applied_legal_nonstay_requests",
    "request_opportunity_use",
    "applied_opportunity_use",
    "applied_per_step",
    "applied_per_active_decision",
    "applied_per_request",
    "rejected_per_request",
)


def _stream(condition, event_id, *, count=2, left=1, right=1, restarts=1, window=None):
    stream = {
        "event_id": event_id,
        "condition": condition,
        "min_duration": 2,
        "confirmed_count": count,
        "left_censored_count": left,
        "right_censored_count": right,
        "fully_observed_count": max(count - left - right, 0),
        "observed_restart_count": restarts,
        "unconfirmed_runs": 0,
        "unconfirmed_terminal_runs": 0,
        "mean_observed_duration": 2.5,
        "descriptive_duration": {
            "mask": "fully_observed_only",
            "n": max(count - left - right, 0),
            "mean": 2.0 if count - left - right > 0 else None,
        },
        "events": [],
    }
    if condition == "nonempty_all_stopped":
        stream.update(
            window=window or "fixed_front_2",
            region="front",
            lane="all",
        )
    else:
        stream["speed_threshold"] = 0 if condition == "pv_speed_eq_0" else 1
    return stream


def make_analysis(
    task_id,
    attempt_id,
    *,
    steps=2,
    method="B1",
    split="pilot",
    training_line=None,
    density_index=1,
    replicate=0,
    metric=1.0,
    lap_time=None,
    states=None,
    profile_sha="a" * 64,
    profile_id="hand-profile",
    technical_only=True,
    analysis_scope="technical",
    event_counts=(2, 1, 1),
    spatial_status="computed",
):
    states = states or [(1, 0, 1, 41, steps)]
    if sum(state[4] for state in states) != steps:
        raise AssertionError("fixture states must sum to steps")
    analysis_id = hashlib.sha256(f"{task_id}:{attempt_id}".encode()).hexdigest()
    streams = [
        _stream("pv_speed_eq_0", "pv_stop", count=event_counts[0]),
        _stream("pv_speed_le_1", "pv_low_progress", count=event_counts[1]),
        _stream(
            "nonempty_all_stopped",
            "primary_front_stagnation",
            count=event_counts[2],
        ),
    ]
    summaries = []
    for window_id in WINDOW_IDS:
        for region in REGIONS:
            for lane in (0, "all"):
                for group in GROUPS:
                    summaries.append(
                        {
                            "window_id": window_id,
                            "region": region,
                            "lane": lane,
                            "group": group,
                            "N": 2 if group == "background" else 1,
                            "S": 6 if group == "background" else 3,
                            "Z": 0,
                            "A": 41 * (2 if lane == "all" else 1) * steps,
                            "defined_steps": steps,
                            "empty_steps": 0,
                            "empty_fraction": 0.0,
                            "pooled_speed": 3.0,
                            "pooled_stopped_fraction": 0.0,
                            "time_mean_density": 0.1,
                            "time_mean_flow": 0.3,
                            "cell_time_density": 0.05,
                            "cell_time_flow": 0.15,
                            "speed_deficit": 2.0,
                            "normalized_local_delay": 0.4,
                        }
                    )
    frequency_records = [
        {
            "window_id": window_id,
            "region": "full",
            "lane": "all",
            "group": "background",
            "steps": steps,
            "states": [
                {"N": n, "S": s, "Z": z, "A": a, "count": count, "frequency": count / steps}
                for n, s, z, a, count in states
            ],
        }
        for window_id in WINDOW_IDS
    ]
    trace = {
        "step": list(range(steps + 1)),
        "speed": [0] + [1] * steps,
        "lane": [0] * (steps + 1),
        "pos": list(range(steps + 1)),
        "path": list(range(100, 101 + steps)),
        "cumulative_distance": list(range(steps + 1)),
        "includes_t0_context": True,
    }
    metrics = {name: 0.0 for name in OFFICIAL_NAMES}
    metrics.update(
        vehicle_count=10,
        priority_vehicle_count=1,
        density=0.2,
        flow_all=1.0,
        flow_background=0.9,
        flow_priority=0.1,
        mean_speed_all=2.0,
        mean_speed_background=2.0,
        mean_speed_priority=metric,
        worst_priority_mean_speed=metric,
        mean_priority_time_loss=1 - metric / 6,
        worst_priority_time_loss=1 - metric / 6,
        stopped_fraction_all=0.0,
        stopped_fraction_background=0.0,
        stopped_fraction_priority=0.0,
        lane_change_rate_all=0.0,
        lane_change_rate_background=0.0,
        lane_change_rate_priority=0.0,
        completed_priority_laps=0,
        mean_priority_lap_time=lap_time,
        rule_detections=0,
        rule_requests=0,
        rule_applied=0,
        rule_rejected=0,
        rule_blocked_no_safe_lane=0,
        rule_cooldown_suppressed=0,
    )
    control = {name: 0 for name in CONTROL_NAMES}
    control.update(
        active_av_decisions=0,
        request_opportunity_use=None,
        applied_opportunity_use=None,
        applied_per_active_decision=None,
        applied_per_request=None,
        rejected_per_request=None,
        opportunity_use_status="not_applicable",
    )
    purpose = "technical" if technical_only else "scientific"
    if spatial_status == "not_requested":
        spatial_covariance = None
        reader_profile = "full12"
    else:
        spatial_covariance = {
            "measure": "background_occupancy_connected_covariance",
            "window": "fixed_front_2",
            "cells": 41,
            "lanes": 2,
            "steps": steps,
            "lags": [0, 1, 2],
            "pair_counts": [steps * 2 * (41 - lag) for lag in (0, 1, 2)],
            "lag0_variance": 0.25,
            "zero_variance_frames": 0,
            "zero_variance_total_frames": steps * 2,
            "zero_variance_fraction": 0.0,
            "covariance": [0.5, 0.25, 0.0],
            "covariance0": 0.5,
            "defined": True,
            "is_correlation": False,
        }
        reader_profile = "mechanism"
    return {
        "schema_version": 1,
        "analysis_kind": "episode_diagnostics_v1",
        "scientific_unit": "episode",
        "independence": {"independent_episodes": 1},
        "profile": {"profile_id": profile_id, "technical_only": technical_only,
                    "purpose": purpose},
        "profile_id": profile_id,
        "profile_sha256": profile_sha,
        "technical_only": technical_only,
        "analysis_scope": analysis_scope,
        "analysis_id": analysis_id,
        "episode": {
            "task_id": task_id,
            "execution_id": "e" * 64,
            "attempt_id": attempt_id,
            "purpose": purpose,
            "split": split,
            "method": method,
            "training_line": training_line,
            "model_id": None,
            "candidate_id": None,
            "density_index": density_index,
            "replicate": replicate,
            "priority_id": 7,
            "observer_profile": "full12" if reader_profile == "full12" else "mechanism",
            "measurement_steps": steps,
            "num_lanes": 2,
            "road_length": 200,
            "config": {
                "road_length": 200,
                "num_lanes": 2,
                "density": 0.2,
                "av_fraction": 0.95,
                "measure_steps": steps,
                "warmup_steps": 1000,
                "cooldown_steps": 1,
                "vmax_hdv": 5,
                "vmax_priority": 6,
            },
        },
        "windows": {
            "ids": list(WINDOW_IDS),
            "primary": "fixed_front_2",
            "regions": list(REGIONS),
            "lanes": [0, "all"],
            "groups": list(GROUPS),
            "summaries": summaries,
            "transitions": [],
            "state_frequencies": frequency_records,
            "geometry": [],
        },
        "global_background": [],
        "global_lane_changes": [],
        "pv": {
            "distance": steps,
            "mean_speed": 1.0,
            "normalized_time_loss": 1 - 1 / 6,
            "stopped_steps": 0,
            "stopped_fraction": 0.0,
            "moving_steps": steps,
            "moving_mean_speed": 1.0,
            "speed_counts": [0] + [steps] + [0] * 5,
            "lane_changes": 0,
            "lane_change_rate": 0.0,
            "completed_laps": 0,
            "lap_completion_steps": [],
            "completed_lap_times": [],
            "mean_completed_lap_time": None,
            "remaining_distance": steps,
            "no_completed_laps": True,
            "initial": {"speed": 0, "lane": 0, "pos": 100},
            "trace": trace,
        },
        "events": {
            "streams": streams,
            "summary": {
                "stream_count": 3,
                "confirmed_events": sum(event_counts),
                "left_censored_events": 3,
                "right_censored_events": 3,
                "fully_observed_events": 0,
                "observed_restarts": 3,
                "unconfirmed_runs": 0,
                "scientific_unit": "episode",
            },
        },
        "spatial_status": spatial_status,
        "spatial_covariance": spatial_covariance,
        "official": {"metrics": metrics, "control": control},
        "lineage": {
            "schema_version": 1,
            "analysis_id": analysis_id,
            "input": {
                "marker_sha256": "c" * 64,
                "arrays_sha256": "d" * 64,
                "task_id": task_id,
                "attempt_id": attempt_id,
                "execution_id": "e" * 64,
                "observer_profile": "full12" if reader_profile == "full12" else "mechanism",
                "required_reader_profile": reader_profile,
            },
            "profile": {
                "profile_id": profile_id,
                "profile_sha256": profile_sha,
                "purpose": purpose,
                "technical_only": technical_only,
            },
            "code": {"source_digest": "f" * 64},
        },
    }


def write_analysis(path, analysis):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(schema.canonical_json(analysis))
    return path


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_episode_scalar_row_includes_lineage_events_primary_and_spatial_lags():
    analysis = make_analysis("a" * 64, "attempt-1", metric=2.5, lap_time=None)
    row = reporting.episode_scalar_row(analysis, source_sha256="9" * 64)
    assert row["source_sha256"] == "9" * 64
    assert row["task_id"] == "a" * 64
    assert row["attempt_id"] == "attempt-1"
    assert row["profile_sha256"] == "a" * 64
    assert row["code_source_digest"] == "f" * 64
    assert row["input_marker_sha256"] == "c" * 64
    assert row["input_arrays_sha256"] == "d" * 64
    assert row["mean_speed_priority"] == 2.5
    assert row["pv_stop_events"] == 2
    assert row["pv_low_progress_events"] == 1
    assert row["window_all_stopped_events"] == 1
    assert row["window_all_stopped_window"] == "fixed_front_2"
    assert row["primary_pooled_speed"] == 3.0
    assert row["spatial_covariance_lag0"] == 0.5
    assert row["spatial_status"] == "computed"
    assert row["spatial_covariance_lag2"] == 0.0
    assert row["spatial_pair_count_lag0"] == 2 * 2 * 41
    assert row["opportunity_use_status"] == "not_applicable"
    assert reporting.scalar_columns(row) == tuple(row)


def test_full12_only_analysis_exports_without_spatial_lag_columns(tmp_path):
    analysis = make_analysis(
        "j" * 64, "attempt-1", spatial_status="not_requested"
    )
    path = write_analysis(tmp_path / "full12.json", analysis)
    output = tmp_path / "tables"
    result = reporting.stream_episode_tables([path], output)
    assert result["counts"]["accepted"] == 1
    episodes = read_csv(output / reporting.EPISODES_CSV)
    assert len(episodes) == 1
    assert episodes[0]["spatial_status"] == "not_requested"
    assert episodes[0]["observer_profile"] == "full12"
    assert not any(
        column.startswith("spatial_covariance_lag") for column in episodes[0]
    )
    summary = read_csv(output / reporting.POINT_SUMMARY_CSV)
    assert not any(
        row["metric"].startswith("spatial_covariance_lag") for row in summary
    )
    frequencies = read_csv(output / reporting.FREQUENCIES_CSV)
    assert {row["window_id"] for row in frequencies} == set(WINDOW_IDS)


def test_streaming_tables_coverage_na_and_zero_absent_frequency(tmp_path):
    stem = "a" * 64
    first = make_analysis(
        stem,
        "attempt-1",
        steps=2,
        metric=1.0,
        lap_time=None,
        states=[(1, 0, 1, 41, 2)],
        replicate=0,
    )
    second = make_analysis(
        "b" * 64,
        "attempt-1",
        steps=4,
        metric=3.0,
        lap_time=5.0,
        states=[(2, 2, 0, 82, 4)],
        replicate=1,
    )
    first_path = write_analysis(tmp_path / "a.json", first)
    second_path = write_analysis(tmp_path / "b.json", second)
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{not valid json", encoding="utf-8")
    point = ("pilot", "B1", None, 1)
    output = tmp_path / "tables"
    result = reporting.stream_episode_tables(
        [first_path, second_path, invalid_path],
        output,
        planned={point: 3},
    )
    manifest = json.loads((output / reporting.TABLES_MANIFEST).read_text())
    assert manifest["counts"]["sources"] == 3
    assert manifest["counts"]["accepted"] == 2
    assert manifest["counts"]["invalid"] == 1
    assert manifest["counts"]["points"] == 1
    assert manifest["counts"]["points_missing_planned"] == 1
    assert "not computed" in manifest["estimates"]["ci"]

    episodes = read_csv(output / reporting.EPISODES_CSV)
    assert len(episodes) == 2
    assert [row["mean_speed_priority"] for row in episodes] == ["1.0", "3.0"]
    jsonl = (output / reporting.EPISODES_JSONL).read_text().splitlines()
    assert len(jsonl) == 2

    coverage = read_csv(output / reporting.COVERAGE_CSV)
    assert len(coverage) == 1
    assert coverage[0]["planned_n"] == "3"
    assert coverage[0]["accepted_n"] == "2"
    assert coverage[0]["missing_planned"] == "1"
    assert coverage[0]["coverage_fraction"] == "0.6666666666666666"

    summary = read_csv(output / reporting.POINT_SUMMARY_CSV)
    speed = [row for row in summary if row["metric"] == "mean_speed_priority"]
    assert len(speed) == 1
    assert speed[0]["n_defined"] == "2"
    assert speed[0]["n_undefined"] == "0"
    assert float(speed[0]["mean"]) == pytest.approx(2.0)
    assert float(speed[0]["sample_sd"]) == pytest.approx(math.sqrt(2.0))
    lap = [row for row in summary if row["metric"] == "mean_priority_lap_time"]
    assert len(lap) == 1
    assert lap[0]["n_defined"] == "1"
    assert lap[0]["n_undefined"] == "1"
    assert lap[0]["sample_sd"] == ""
    assert float(lap[0]["mean"]) == pytest.approx(5.0)

    frequencies = read_csv(output / reporting.FREQUENCIES_CSV)
    first_state = [
        row
        for row in frequencies
        if row["window_id"] == "fixed_front_2"
        and row["N"] == "1"
        and row["S"] == "0"
        and row["Z"] == "1"
    ]
    assert len(first_state) == 1
    assert first_state[0]["n_episodes"] == "2"
    assert float(first_state[0]["mean_frequency"]) == pytest.approx(0.5)
    assert float(first_state[0]["sample_sd_frequency"]) == pytest.approx(
        math.sqrt(0.5)
    )
    absent_state = [
        row
        for row in frequencies
        if row["window_id"] == "fixed_front_2"
        and row["N"] == "2"
        and row["S"] == "2"
    ]
    assert len(absent_state) == 1
    assert float(absent_state[0]["mean_frequency"]) == pytest.approx(0.5)

    index = [
        json.loads(line)
        for line in (output / reporting.SOURCE_INDEX_JSONL).read_text().splitlines()
    ]
    statuses = [row["status"] for row in index]
    assert statuses.count("accepted") == 2
    assert statuses.count("invalid") == 1
    assert all(len(row["sha256"]) == 64 for row in index)
    assert result["files"]["episodes_csv"].endswith(reporting.EPISODES_CSV)


def test_single_accepted_episode_gives_none_sample_sd(tmp_path):
    analysis = make_analysis("c" * 64, "attempt-1", steps=2, metric=1.0)
    path = write_analysis(tmp_path / "one.json", analysis)
    output = tmp_path / "tables"
    reporting.stream_episode_tables([path], output)
    summary = read_csv(output / reporting.POINT_SUMMARY_CSV)
    row = [item for item in summary if item["metric"] == "mean_speed_priority"][0]
    assert row["n_defined"] == "1"
    assert row["sample_sd"] == ""
    frequencies = read_csv(output / reporting.FREQUENCIES_CSV)
    assert frequencies
    assert all(row["n_episodes"] == "1" for row in frequencies)
    assert all(row["sample_sd_frequency"] == "" for row in frequencies)
    coverage = read_csv(output / reporting.COVERAGE_CSV)
    assert coverage[0]["planned_n"] == ""
    assert coverage[0]["missing_planned"] == ""


def test_ambiguous_duplicates_are_rejected_and_canonical_mapping_counts_once(tmp_path):
    task = "d" * 64
    first = make_analysis(task, "attempt-1", steps=2, metric=1.0, replicate=0)
    second = make_analysis(task, "attempt-2", steps=2, metric=2.0, replicate=0)
    first_path = write_analysis(tmp_path / "a1.json", first)
    second_path = write_analysis(tmp_path / "a2.json", second)

    with pytest.raises(ValueError, match="ambiguous duplicate"):
        reporting.stream_episode_tables(
            [first_path, second_path], tmp_path / "rejected"
        )
    assert not (tmp_path / "rejected" / reporting.EPISODES_CSV).exists()

    output = tmp_path / "canonical"
    result = reporting.stream_episode_tables(
        [first_path, second_path],
        output,
        canonical_ids={task: "attempt-2"},
    )
    assert result["counts"]["accepted"] == 1
    assert result["counts"]["skipped_noncanonical"] == 1
    rows = read_csv(output / reporting.EPISODES_CSV)
    assert len(rows) == 1
    assert rows[0]["mean_speed_priority"] == "2.0"
    index = [
        json.loads(line)
        for line in (output / reporting.SOURCE_INDEX_JSONL).read_text().splitlines()
    ]
    statuses = [row["status"] for row in index]
    assert statuses == ["skipped_noncanonical_retry", "accepted"]

    # Re-reading the same canonical source must not increase n.
    duplicate_output = tmp_path / "duplicate"
    duplicate = reporting.stream_episode_tables(
        [second_path, second_path],
        duplicate_output,
        canonical_ids={task: "attempt-2"},
    )
    assert duplicate["counts"]["accepted"] == 1
    assert duplicate["counts"]["duplicate_attempts"] == 1
    rows = read_csv(duplicate_output / reporting.EPISODES_CSV)
    assert len(rows) == 1


def test_streaming_refuses_profile_mixing_and_technical_scientific_claims(tmp_path):
    first = make_analysis("e" * 64, "attempt-1", profile_sha="a" * 64)
    second = make_analysis("f" * 64, "attempt-1", profile_sha="b" * 64)
    paths = [
        write_analysis(tmp_path / "p1.json", first),
        write_analysis(tmp_path / "p2.json", second),
    ]
    with pytest.raises(ValueError, match="profiles cannot be mixed"):
        reporting.stream_episode_tables(paths, tmp_path / "mixed")
    with pytest.raises(ValueError, match="technical-only"):
        reporting.stream_episode_tables(
            [paths[0]], tmp_path / "scientific", require_scientific=True
        )
    scientific = make_analysis(
        "a" * 64,
        "attempt-1",
        technical_only=False,
        analysis_scope="scientific",
        profile_sha="c" * 64,
    )
    scientific_path = write_analysis(tmp_path / "scientific.json", scientific)
    result = reporting.stream_episode_tables(
        [scientific_path], tmp_path / "scientific-ok", require_scientific=True
    )
    assert result["counts"]["accepted"] == 1


def test_frequency_accumulator_sparse_zero_absent_and_small_n():
    first = make_analysis(
        "g" * 64,
        "attempt-1",
        steps=2,
        states=[(1, 0, 1, 41, 2)],
    )
    second = make_analysis(
        "h" * 64,
        "attempt-1",
        steps=4,
        states=[(2, 2, 0, 82, 4)],
    )
    accumulator = reporting.frequency_accumulator()
    reporting.add_episode_frequencies(accumulator, first)
    rows = list(reporting.frequency_rows(accumulator))
    primary = [
        row
        for row in rows
        if row["window_id"] == "fixed_front_2" and row["N"] == 1
    ]
    assert len(primary) == 1
    assert primary[0]["n_episodes"] == 1
    assert primary[0]["mean_frequency"] == pytest.approx(1.0)
    assert primary[0]["sample_sd_frequency"] is None

    reporting.add_episode_frequencies(accumulator, second)
    rows = list(reporting.frequency_rows(accumulator))
    by_state = {
        (row["window_id"], row["N"], row["S"], row["Z"]): row
        for row in rows
    }
    present = by_state[("fixed_front_2", 1, 0, 1)]
    absent_in_second = by_state[("fixed_front_2", 2, 2, 0)]
    assert present["n_episodes"] == 2
    # Episodes have different horizons: each has frequency 1, the other episode
    # contributes an implicit zero for the absent state.
    assert present["mean_frequency"] == pytest.approx(0.5)
    assert present["sample_sd_frequency"] == pytest.approx(math.sqrt(0.5))
    assert absent_in_second["mean_frequency"] == pytest.approx(0.5)
    assert absent_in_second["sample_sd_frequency"] == pytest.approx(math.sqrt(0.5))


def test_select_illustration_point_uses_median_then_smallest_seed():
    records = [
        {"training_line": "L1", "seed": 5, "environment": "low", "paired_pv_effect": -1.0},
        {"training_line": "L1", "seed": 9, "environment": "high", "paired_pv_effect": 2.0},
        {"training_line": "L1", "seed": 3, "environment": "near_median", "paired_pv_effect": 0.5},
        {"training_line": "L2", "seed": 1, "environment": "other_line", "paired_pv_effect": 0.5},
    ]
    selected = reporting.select_illustration_point(records, training_line="L1")
    assert selected["environment"] == "near_median"
    assert selected["seed"] == 3
    assert selected["median_paired_pv_effect"] == pytest.approx(0.5)
    assert "median" in selected["selection_rule"]
    assert "no best-trajectory" in selected["selection_rule"]

    tied = [
        {"training_line": "L1", "seed": 9, "environment": "a", "paired_pv_effect": 0.4},
        {"training_line": "L1", "seed": 3, "environment": "b", "paired_pv_effect": 0.6},
    ]
    selected_tie = reporting.select_illustration_point(tied, training_line="L1")
    assert selected_tie["environment"] == "b"
    assert selected_tie["seed"] == 3

    with pytest.raises(ValueError, match="fixed nonempty"):
        reporting.select_illustration_point(records, training_line="")
    with pytest.raises(ValueError, match="no paired"):
        reporting.select_illustration_point(records, training_line="missing")
    ambiguous = [
        {"training_line": "L1", "seed": 3, "environment": "a", "paired_pv_effect": 0.4},
        {"training_line": "L1", "seed": 3, "environment": "b", "paired_pv_effect": 0.6},
    ]
    with pytest.raises(ValueError, match="ambiguous"):
        reporting.select_illustration_point(ambiguous, training_line="L1")


def test_plots_save_tiny_fixtures_with_lineage(tmp_path):
    pytest.importorskip("matplotlib")
    analysis = make_analysis("i" * 64, "attempt-1", steps=3)
    arrays = {
        "local_occupancy": np.zeros((3, 41, 2), dtype=np.uint8),
    }
    arrays["local_occupancy"][:, 10, 0] = 1
    arrays["local_occupancy"][:, 11, 1] = 1

    pv_path = reporting.plot_pv_trace(analysis, tmp_path / "pv.png")
    window_path = reporting.plot_window_comparison(analysis, tmp_path / "windows.png")
    mechanism_path = reporting.plot_mechanism_space_time(
        analysis, arrays, tmp_path / "mechanism.png", lane=0
    )
    for path in (pv_path, window_path, mechanism_path):
        assert path.exists()
        assert path.stat().st_size > 0
        lineage = json.loads(
            path.with_name(path.name + ".lineage.json").read_text()
        )
        assert lineage["scientific_unit"] == "episode"
        assert lineage["source"]["input"]["marker_sha256"] == "c" * 64
        assert lineage["profile_sha256"] == "a" * 64
        assert lineage["source"]["code"]["source_digest"] == "f" * 64
        assert lineage["rule"]
        assert len(lineage["data_sha256"]) == 64
    assert "no trajectory or seed selection" in json.loads(
        pv_path.with_name(pv_path.name + ".lineage.json").read_text()
    )["rule"]
    assert "no best-window" in json.loads(
        window_path.with_name(window_path.name + ".lineage.json").read_text()
    )["rule"]


@pytest.mark.parametrize("changed_field, value", [
    ("method", "B2"), ("density_index", 2), ("training_line", "line-2"),
])
def test_frequencies_are_grouped_like_point_summary(tmp_path, changed_field, value):
    first = make_analysis("point-a-1", "attempt-1", states=[(1, 0, 1, 41, 2)])
    other = make_analysis("point-b-1", "attempt-1", states=[(3, 3, 0, 41, 2)],
                          **{changed_field: value})
    repeat = make_analysis("point-a-2", "attempt-1", states=[(2, 2, 0, 41, 2)])
    paths = [write_analysis(tmp_path / f"{i}.json", item)
             for i, item in enumerate((first, other, repeat))]
    output = tmp_path / "tables"
    reporting.stream_episode_tables(paths, output)
    rows = [row for row in read_csv(output / reporting.FREQUENCIES_CSV)
            if row["window_id"] == "fixed_front_2"]
    assert len(rows) == 3
    for row in rows:
        is_other = row["N"] == "3"
        assert (row[changed_field] == str(value)) == is_other
        assert int(row["n_episodes"]) == (1 if is_other else 2)
        assert float(row["mean_frequency"]) == (1.0 if is_other else 0.5)
        if is_other:
            assert row["sample_sd_frequency"] == ""
        else:
            assert float(row["sample_sd_frequency"]) == pytest.approx(math.sqrt(0.5))
    summary_keys = {tuple(row[field] for field in reporting.POINT_FIELDS)
                    for row in read_csv(output / reporting.POINT_SUMMARY_CSV)}
    assert {tuple(row[field] for field in reporting.POINT_FIELDS) for row in rows} == summary_keys
    manifest = json.loads((output / reporting.TABLES_MANIFEST).read_text())
    assert manifest["point_fields"] == list(reporting.POINT_FIELDS)
    assert "same point" in manifest["estimates"]["NA"]


def test_streaming_releases_payloads_and_each_points_sparse_accumulator(tmp_path, monkeypatch):
    import weakref

    class Tracked(dict):
        pass

    payload_refs = []
    accumulator_refs = []
    parse = reporting._parse_analysis
    new_accumulator = reporting.frequency_accumulator

    def tracked_parse(raw):
        assert all(ref() is None for ref in payload_refs)
        value = Tracked(parse(raw))
        payload_refs.append(weakref.ref(value))
        return value

    def tracked_accumulator():
        assert all(ref() is None for ref in payload_refs)
        assert all(ref() is None for ref in accumulator_refs)
        value = Tracked(new_accumulator())
        accumulator_refs.append(weakref.ref(value))
        return value

    monkeypatch.setattr(reporting, "_parse_analysis", tracked_parse)
    monkeypatch.setattr(reporting, "frequency_accumulator", tracked_accumulator)

    def sources():
        for repeat in range(2):
            for point in range(8):
                analysis = make_analysis(f"{point}-{repeat}", "attempt-1",
                                         density_index=point,
                                         states=[(point + 1, 0, point + 1, 41, 2)])
                yield write_analysis(tmp_path / f"{point}-{repeat}.json", analysis)

    output = tmp_path / "tables"
    reporting.stream_episode_tables(sources(), output)
    assert len(payload_refs) == 16
    assert len(accumulator_refs) == 8
    assert all(ref() is None for ref in accumulator_refs)
    assert all(row["n_episodes"] == "2"
               for row in read_csv(output / reporting.FREQUENCIES_CSV))
    assert not list(output.glob(".tables-*"))


def test_w5_exports_preserve_episode_tables_point_separation_and_na(tmp_path):
    sources = []
    for index, (line, density, value) in enumerate(((1, 1, 2.), (1, 1, 4.), (2, 1, 8.), (1, 2, 16.))):
        analysis = make_analysis(f"w5-{index}", "attempt-1", method="Graph", training_line=line,
                                 density_index=density)
        analysis["mechanism_status"] = "computed"
        analysis["mechanism"] = {"tables": [{"name": "long_pv_leader_gap", "mean": value,
            "denominator": 2, "sentinel_count": 0, "states": [{"value": int(value), "count": 2, "frequency": 1.}]}]}
        analysis["temporal_status"] = "computed"
        analysis["temporal"] = {"lag0_variance": 0., "records": [
            {"lag": 0, "pair_count": 2, "covariance": 0., "value": 0.},
            {"lag": 5, "pair_count": 0, "covariance": None, "value": None}]}
        analysis["event_alignment_status"] = "computed"
        analysis["event_alignment"] = {"included_anchors": 20, "records": [
            {"signal": "pv_speed", "offset": 0, "valid_count": 20, "mean": value}]}
        analysis["events"]["pv_stop_intervals"] = {"observed_interval_count": 1, "mean_observed_gap": value}
        sources.append(write_analysis(tmp_path / f"w5-{index}.json", analysis))
    output = tmp_path / "tables"
    result = reporting.stream_episode_tables(sources, output)
    documents = [json.loads(line) for line in (output / reporting.MECHANISM_JSONL).read_text().splitlines()]
    assert len(documents) == 4
    assert [(doc["training_line"], doc["density_index"]) for doc in documents] == [(1, 1), (1, 1), (2, 1), (1, 2)]
    assert [doc["mechanism"]["tables"][0]["states"][0]["value"] for doc in documents] == [2, 4, 8, 16]
    assert all(doc["scientific_unit"] == "episode" for doc in documents)
    rows = read_csv(output / reporting.POINT_SUMMARY_CSV)
    aligned = [row for row in rows if row["metric"] == "aligned_pv_speed_offset0_mean"]
    point = next(row for row in aligned if row["training_line"] == "1" and row["density_index"] == "1")
    assert float(point["mean"]) == 3. and point["n_defined"] == "2"
    assert len(aligned) == 3  # no events-as-n and no line/density pooling
    undefined = [row for row in rows if row["metric"] == "temporal_pv_speed_lag5_value"]
    assert all(row["n_defined"] == "0" for row in undefined)
    assert result["files"]["mechanism_tables_jsonl"].endswith(reporting.MECHANISM_JSONL)
    manifest = json.loads((output / reporting.TABLES_MANIFEST).read_text())
    assert manifest["tables"]["mechanism_tables_jsonl"]["rows"] == 4
    assert "not computed" in manifest["estimates"]["ci"]


def test_reporting_rejects_missing_requested_w5_and_dropped_point_identity():
    analysis = make_analysis("w5", "attempt-1")
    for name in ("mechanism", "temporal", "event_alignment"):
        analysis["profile"][name] = {"requested": True}
        with pytest.raises(ValueError, match=f"requested {name}"):
            reporting.episode_scalar_row(analysis)
        del analysis["profile"][name]
    for fields in (("method",), ("split", "method", "density_index")):
        with pytest.raises(ValueError, match="must retain"):
            reporting.point_key(analysis, fields)
