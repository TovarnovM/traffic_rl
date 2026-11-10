"""Bounded offline reporting from published episode diagnostics.

This module streams one accepted analysis at a time and never holds campaign
payloads or CSV text in memory.  It reports means and sample standard
deviations only; Monte Carlo confidence intervals and crossed bootstrap belong
to the statistics module and are deliberately not computed from one episode or
here.  Missing/undefined observations stay ``None`` and are counted explicitly.

Matplotlib is an optional, lazily imported reporting dependency.  Every chart
uses the Agg backend and is saved from already-recorded data only; no model,
simulator or runtime inference is imported.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from . import postprocess, schema

TABLES_SCHEMA_VERSION = 1
TABLES_KIND = "episode_tables_v1"

EPISODES_CSV = "episodes.csv"
EPISODES_JSONL = "episodes.jsonl"
SOURCE_INDEX_JSONL = "source_index.jsonl"
POINT_SUMMARY_CSV = "point_summary.csv"
COVERAGE_CSV = "coverage.csv"
FREQUENCIES_CSV = "window_state_frequencies.csv"
MECHANISM_JSONL = "mechanism_tables.jsonl"
TABLES_MANIFEST = "tables_manifest.json"

POINT_FIELDS = ("split", "method", "training_line", "density_index")
_ALLOWED_POINT_FIELDS = frozenset(
    ("split", "method", "training_line", "density_index", "model_id", "candidate_id")
)

_OFFICIAL_METRICS = (
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
)

_CONTROL_SCALARS = (
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

_EVENT_PREFIXES = (
    ("pv_speed_eq_0", "pv_stop"),
    ("pv_speed_le_1", "pv_low_progress"),
    ("nonempty_all_stopped", "window_all_stopped"),
    ("operational_pv_blocking", "pv_blocking"),
)

_PRIMARY_METRICS = (
    "N",
    "S",
    "Z",
    "A",
    "defined_steps",
    "empty_steps",
    "empty_fraction",
    "pooled_speed",
    "pooled_stopped_fraction",
    "time_mean_density",
    "time_mean_flow",
    "cell_time_density",
    "cell_time_flow",
    "speed_deficit",
    "normalized_local_delay",
)

_PV_METRICS = (
    "distance",
    "mean_speed",
    "normalized_time_loss",
    "stopped_steps",
    "stopped_fraction",
    "moving_steps",
    "moving_mean_speed",
    "lane_changes",
    "lane_change_rate",
    "completed_laps",
    "mean_completed_lap_time",
    "remaining_distance",
)

# Identity, provenance and categorical fields never enter per-point means.
_NON_METRIC = frozenset(
    {
        "analysis_schema_version",
        "schema_version",
        "analysis_kind",
        "analysis_id",
        "source_sha256",
        "task_id",
        "execution_id",
        "attempt_id",
        "split",
        "method",
        "training_line",
        "model_id",
        "candidate_id",
        "observer_profile",
        "profile_id",
        "profile_sha256",
        "technical_only",
        "analysis_scope",
        "purpose",
        "priority_id",
        "code_source_digest",
        "input_marker_sha256",
        "input_arrays_sha256",
        "density_index",
        "replicate",
        "measurement_steps",
        "opportunity_use_status",
        "spatial_status",
        "mechanism_status",
        "temporal_status",
        "event_alignment_status",
        "operational_blocking_status",
        "window_all_stopped_event_id",
        "window_all_stopped_window",
        "window_all_stopped_region",
        "window_all_stopped_lane",
    }
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _is_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _reject_constant(token: str):
    raise ValueError(f"non-finite JSON constant {token!r} is not allowed")


def _parse_analysis(raw: bytes) -> dict:
    try:
        value = json.loads(raw, parse_constant=_reject_constant)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError(f"analysis JSON is invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("analysis JSON must be an object")
    try:
        schema.canonical_json(value)
    except ValueError as exc:
        raise ValueError(f"analysis JSON is not finite canonical JSON: {exc}") from exc
    return value


def _identity(analysis: dict) -> dict:
    if not isinstance(analysis, dict):
        raise ValueError("analysis must be a mapping")
    if analysis.get("analysis_kind") != postprocess.ANALYSIS_KIND:
        raise ValueError("unsupported analysis_kind")
    episode = analysis.get("episode")
    if not isinstance(episode, dict):
        raise ValueError("analysis is missing episode identity")
    task_id = episode.get("task_id")
    attempt_id = episode.get("attempt_id")
    profile_sha = analysis.get("profile_sha256")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("analysis episode.task_id must be a nonempty string")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("analysis episode.attempt_id must be a nonempty string")
    if (
        not isinstance(profile_sha, str)
        or len(profile_sha) != 64
        or any(ch not in "0123456789abcdef" for ch in profile_sha)
    ):
        raise ValueError("analysis profile_sha256 must be a lowercase SHA256")
    analysis_id = analysis.get("analysis_id")
    if analysis_id is not None and not isinstance(analysis_id, str):
        raise ValueError("analysis_id must be a string when present")
    for name in ("windows", "pv", "events", "official", "lineage"):
        if not isinstance(analysis.get(name), dict):
            raise ValueError(f"analysis is missing the {name} section")
    if not isinstance(analysis["events"].get("streams"), list):
        raise ValueError("analysis events.streams must be a list")
    if not isinstance(analysis["windows"].get("summaries"), list):
        raise ValueError("analysis windows.summaries must be a list")
    spatial_status = analysis.get("spatial_status")
    spatial = analysis.get("spatial_covariance")
    if spatial_status is None:
        spatial_status = "computed" if isinstance(spatial, dict) else "not_requested"
    if spatial_status == "not_requested":
        if spatial is not None:
            raise ValueError(
                "not_requested spatial analysis must have null spatial_covariance"
            )
    elif spatial_status == "computed":
        if not isinstance(spatial, dict):
            raise ValueError(
                "computed spatial analysis must have a spatial_covariance mapping"
            )
        if not isinstance(spatial.get("lags"), list):
            raise ValueError("analysis spatial_covariance.lags must be a list")
        if not isinstance(spatial.get("covariance"), list) or len(
            spatial["covariance"]
        ) != len(spatial["lags"]):
            raise ValueError("analysis spatial covariance must match its lags")
        if not isinstance(spatial.get("pair_counts"), list) or len(
            spatial["pair_counts"]
        ) != len(spatial["lags"]):
            raise ValueError("analysis spatial pair counts must match its lags")
    else:
        raise ValueError("analysis spatial_status must be computed or not_requested")
    official = analysis["official"]
    if not isinstance(official.get("metrics"), dict) or not isinstance(
        official.get("control"), dict
    ):
        raise ValueError("analysis official metrics/control must be mappings")
    lineage = analysis["lineage"]
    if not isinstance(lineage.get("input"), dict) or not isinstance(
        lineage.get("code"), dict
    ):
        raise ValueError("analysis lineage must contain input and code mappings")
    return {
        "task_id": task_id,
        "attempt_id": attempt_id,
        "profile_sha256": profile_sha,
        "analysis_id": analysis_id,
        "profile_id": analysis.get("profile_id"),
        "technical_only": analysis.get("technical_only"),
        "analysis_scope": analysis.get("analysis_scope"),
        "spatial_status": spatial_status,
    }


def _safe_columns(values, name: str) -> tuple:
    names = tuple(values)
    if not names or len(set(names)) != len(names):
        raise ValueError(f"{name} must be a nonempty unique sequence")
    if not all(isinstance(item, str) and item for item in names):
        raise ValueError(f"{name} entries must be nonempty strings")
    return names


# ---------------------------------------------------------------------------
# One-episode scalar row
# ---------------------------------------------------------------------------


def _window_record(analysis: dict, window_id: str, region: str, lane, group: str):
    windows = analysis.get("windows")
    if not isinstance(windows, dict):
        return None
    records = windows.get("summaries")
    if not isinstance(records, list):
        return None
    for record in records:
        if (
            isinstance(record, dict)
            and record.get("window_id") == window_id
            and record.get("region") == region
            and record.get("lane") == lane
            and record.get("group") == group
        ):
            return record
    return None


def _event_streams(analysis: dict) -> dict:
    events = analysis.get("events")
    if not isinstance(events, dict):
        return {}
    streams = events.get("streams")
    if not isinstance(streams, list):
        return {}
    by_condition = {}
    for stream in streams:
        if isinstance(stream, dict) and isinstance(stream.get("condition"), str):
            by_condition[stream["condition"]] = stream
    return by_condition


def episode_scalar_row(analysis: dict, *, source_sha256: str | None = None) -> dict:
    """Flatten one episode analysis into a deterministic scalar row.

    The row contains official episode scalars, control/opportunity-use scalars,
    diagnostic PV/event/window/spatial scalars and provenance.  It never
    contains raw arrays, frame tables or frequency lists.
    """

    identity = _identity(analysis)
    episode = analysis["episode"]
    official = analysis.get("official")
    if not isinstance(official, dict):
        official = {}
    metrics = official.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    control = official.get("control")
    if not isinstance(control, dict):
        control = {}
    lineage = analysis.get("lineage")
    if not isinstance(lineage, dict):
        lineage = {}
    input_lineage = lineage.get("input")
    if not isinstance(input_lineage, dict):
        input_lineage = {}
    code_lineage = lineage.get("code")
    if not isinstance(code_lineage, dict):
        code_lineage = {}

    row = {}
    row["analysis_schema_version"] = analysis.get("schema_version")
    row["analysis_kind"] = analysis.get("analysis_kind")
    row["analysis_id"] = identity["analysis_id"]
    row["source_sha256"] = source_sha256
    row["task_id"] = identity["task_id"]
    row["execution_id"] = episode.get("execution_id")
    row["attempt_id"] = identity["attempt_id"]
    row["split"] = episode.get("split")
    row["method"] = episode.get("method")
    row["training_line"] = episode.get("training_line")
    row["model_id"] = episode.get("model_id")
    row["candidate_id"] = episode.get("candidate_id")
    row["density_index"] = episode.get("density_index")
    row["replicate"] = episode.get("replicate")
    row["priority_id"] = episode.get("priority_id")
    row["observer_profile"] = episode.get("observer_profile")
    row["measurement_steps"] = episode.get("measurement_steps")
    row["purpose"] = episode.get("purpose")
    row["profile_id"] = analysis.get("profile_id")
    row["profile_sha256"] = analysis.get("profile_sha256")
    row["technical_only"] = analysis.get("technical_only")
    row["analysis_scope"] = analysis.get("analysis_scope")
    row["code_source_digest"] = code_lineage.get("source_digest")
    row["input_marker_sha256"] = input_lineage.get("marker_sha256")
    row["input_arrays_sha256"] = input_lineage.get("arrays_sha256")

    for name in _OFFICIAL_METRICS:
        row[name] = metrics.get(name)
    for name in _CONTROL_SCALARS:
        row[name] = control.get(name)
    row["opportunity_use_status"] = control.get("opportunity_use_status")

    pv = analysis.get("pv")
    if not isinstance(pv, dict):
        pv = {}
    for name in _PV_METRICS:
        row[f"pv_{name}"] = pv.get(name)

    streams = _event_streams(analysis)
    total_confirmed = 0
    total_left = 0
    total_right = 0
    total_restarts = 0
    for condition, prefix in _EVENT_PREFIXES:
        stream = streams.get(condition)
        if not isinstance(stream, dict):
            stream = {}
        duration = stream.get("descriptive_duration")
        if not isinstance(duration, dict):
            duration = {}
        row[f"{prefix}_events"] = stream.get("confirmed_count")
        row[f"{prefix}_left_censored"] = stream.get("left_censored_count")
        row[f"{prefix}_right_censored"] = stream.get("right_censored_count")
        row[f"{prefix}_restarts"] = stream.get("observed_restart_count")
        row[f"{prefix}_unconfirmed_runs"] = stream.get("unconfirmed_runs")
        row[f"{prefix}_mean_observed_duration"] = stream.get("mean_observed_duration")
        row[f"{prefix}_fully_observed"] = duration.get("n")
        row[f"{prefix}_mean_fully_observed_duration"] = duration.get("mean")
        if condition == "nonempty_all_stopped":
            row[f"{prefix}_event_id"] = stream.get("event_id")
            row[f"{prefix}_window"] = stream.get("window")
            row[f"{prefix}_region"] = stream.get("region")
            row[f"{prefix}_lane"] = stream.get("lane")
        for key in ("confirmed_count", "left_censored_count", "right_censored_count",
                    "observed_restart_count"):
            value = stream.get(key)
            if _is_int(value):
                if key == "confirmed_count":
                    total_confirmed += int(value)
                elif key == "left_censored_count":
                    total_left += int(value)
                elif key == "right_censored_count":
                    total_right += int(value)
                else:
                    total_restarts += int(value)
    row["event_confirmed_total"] = total_confirmed
    row["event_left_censored_total"] = total_left
    row["event_right_censored_total"] = total_right
    row["event_restarts_total"] = total_restarts

    primary = _window_record(
        analysis, postprocess.PRIMARY_WINDOW, "full", "all", "background"
    )
    if not isinstance(primary, dict):
        primary = {}
    for name in _PRIMARY_METRICS:
        row[f"primary_{name}"] = primary.get(name)

    spatial_status = identity["spatial_status"]
    row["spatial_status"] = spatial_status
    if spatial_status == "computed":
        spatial = analysis["spatial_covariance"]
        row["spatial_lag0_variance"] = spatial.get("lag0_variance")
        row["spatial_zero_variance_frames"] = spatial.get("zero_variance_frames")
        row["spatial_zero_variance_fraction"] = spatial.get("zero_variance_fraction")
        lags = spatial.get("lags")
        covariance = spatial.get("covariance")
        pair_counts = spatial.get("pair_counts")
        if isinstance(lags, list) and isinstance(covariance, list):
            for lag, value in zip(lags, covariance):
                row[f"spatial_covariance_lag{int(lag)}"] = value
        if isinstance(lags, list) and isinstance(pair_counts, list):
            for lag, value in zip(lags, pair_counts):
                row[f"spatial_pair_count_lag{int(lag)}"] = value
    _extension_scalars(analysis, row)
    return row


def _extension_scalars(analysis, row):
    profile = analysis.get("profile", {})
    for name in ("mechanism", "temporal", "event_alignment"):
        block = analysis.get(name)
        status = analysis.get(name + "_status", "not_requested")
        requested = profile.get(name) is not None
        if requested and (status != "computed" or not isinstance(block, dict)):
            raise ValueError(f"requested {name} is missing")
        if status not in ("computed", "not_requested") or (status == "not_requested" and block is not None):
            raise ValueError(f"inconsistent {name} status")
        if status == "computed" and not isinstance(block, dict):
            raise ValueError(f"computed {name} must be a mapping")
        row[name + "_status"] = status
    blocking = _event_streams(analysis).get("operational_pv_blocking")
    if any(event.get("condition") == "operational_pv_blocking" for event in profile.get("events", [])) and blocking is None:
        raise ValueError("requested blocking stream is missing")
    row["operational_blocking_status"] = "computed" if blocking is not None else "not_requested"
    intervals = analysis.get("events", {}).get("pv_stop_intervals", {})
    row["pv_stop_observed_intervals"] = intervals.get("observed_interval_count")
    row["pv_stop_mean_observed_gap"] = intervals.get("mean_observed_gap")
    mechanism = analysis.get("mechanism")
    if mechanism is not None:
        for table in mechanism["tables"]:
            prefix = "mechanism_" + table["name"]
            row[prefix + "_mean"] = table["mean"]
            row[prefix + "_denominator"] = table["denominator"]
            row[prefix + "_sentinel_count"] = table["sentinel_count"]
    temporal = analysis.get("temporal")
    if temporal is not None:
        row["temporal_lag0_variance"] = temporal["lag0_variance"]
        for record in temporal["records"]:
            prefix = f"temporal_pv_speed_lag{record['lag']}"
            for name in ("pair_count", "covariance", "value"):
                row[prefix + "_" + name] = record[name]
    aligned = analysis.get("event_alignment")
    if aligned is not None:
        row["aligned_included_anchors"] = aligned["included_anchors"]
        for record in aligned["records"]:
            prefix = f"aligned_{record['signal']}_offset{record['offset']}"
            row[prefix + "_mean"] = record["mean"]
            row[prefix + "_valid_count"] = record["valid_count"]


def scalar_columns(row: dict) -> tuple[str, ...]:
    """Return the stable column order implied by one scalar row."""

    return tuple(row)


# ---------------------------------------------------------------------------
# Point grouping, planned coverage and Welford statistics
# ---------------------------------------------------------------------------


def _validate_point_fields(point_fields) -> tuple:
    fields = _safe_columns(point_fields, "point_fields")
    unknown = sorted(set(fields) - _ALLOWED_POINT_FIELDS)
    if unknown:
        raise ValueError(f"unsupported point_fields {unknown}")
    if not set(POINT_FIELDS) <= set(fields):
        raise ValueError("point_fields must retain split, method, training_line and density_index")
    return fields


def point_key(analysis: dict, point_fields=POINT_FIELDS) -> tuple:
    """Return the explicit point identity for one episode analysis."""

    fields = _validate_point_fields(point_fields)
    episode = analysis.get("episode")
    if not isinstance(episode, dict):
        raise ValueError("analysis is missing episode identity")
    return tuple(episode.get(name) for name in fields)


def _hashable_point_key(values, fields) -> tuple:
    values = tuple(values)
    if len(values) != len(fields):
        raise ValueError(f"point key must contain {len(fields)} values")
    for value in values:
        if value is not None and not isinstance(value, (str, int, float)):
            raise ValueError("point key values must be scalar JSON-like values")
        if isinstance(value, bool):
            raise ValueError("point key values must not be booleans")
    return values


def _normalise_planned(planned, fields) -> dict:
    points: dict[tuple, int] = {}
    if planned is None:
        return points
    if isinstance(planned, dict):
        items = planned.items()
    else:
        items = []
        for index, item in enumerate(planned):
            if not isinstance(item, dict):
                raise ValueError(f"planned[{index}] must be a mapping")
            try:
                values = [item[name] for name in fields]
            except KeyError as exc:
                raise ValueError(
                    f"planned[{index}] is missing point field {exc.args[0]!r}"
                ) from exc
            if "planned_n" not in item:
                raise ValueError(f"planned[{index}] is missing planned_n")
            items.append((tuple(values), item["planned_n"]))
    for key, value in items:
        if not isinstance(key, tuple):
            if len(fields) == 1:
                key = (key,)
            else:
                raise ValueError("planned mapping keys must be point tuples")
        key = _hashable_point_key(key, fields)
        if not _is_int(value) or int(value) < 0:
            raise ValueError("planned_n must be a nonnegative integer")
        if key in points:
            raise ValueError(f"duplicate planned point {key}")
        points[key] = int(value)
    return points


def _new_point(planned_n=None) -> dict:
    return {"planned_n": planned_n, "accepted_n": 0, "metrics": {}}


def _accumulate_metrics(point: dict, metric_columns, row: dict) -> None:
    for metric in metric_columns:
        value = _finite_number(row.get(metric))
        if value is None:
            continue
        state = point["metrics"].get(metric)
        if state is None:
            state = [0, 0.0, 0.0]
            point["metrics"][metric] = state
        count, mean, m2 = state
        count += 1
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)
        state[0] = count
        state[1] = mean
        state[2] = m2


def _mean_sd(state):
    if state is None or state[0] == 0:
        return 0, None, None
    count, mean, m2 = state
    if count >= 2:
        variance = m2 / (count - 1)
        if variance < 0:
            variance = 0.0
        sd = math.sqrt(variance)
    else:
        sd = None
    return int(count), float(mean), sd


def _sort_scalar(value):
    if value is None:
        return (0, 0, "")
    if isinstance(value, bool):
        return (1, int(value), "")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (2, float(value), "")
    return (3, 0, str(value))


def _point_sort_key(key):
    return tuple(_sort_scalar(value) for value in key)


# ---------------------------------------------------------------------------
# Sparse state-frequency accumulation (absent states contribute zero)
# ---------------------------------------------------------------------------


def frequency_accumulator() -> dict:
    """Create a sparse accumulator for per-window joint (N,S,Z,A) states."""

    return {"episodes": 0, "scopes": {}}


def _validate_frequency_records(analysis: dict) -> list[dict]:
    windows = analysis.get("windows")
    if not isinstance(windows, dict):
        raise ValueError("analysis is missing windows")
    records = windows.get("state_frequencies")
    if not isinstance(records, list):
        raise ValueError("analysis is missing window state_frequencies")
    episode = analysis.get("episode")
    if not isinstance(episode, dict):
        raise ValueError("analysis is missing episode identity")
    steps = episode.get("measurement_steps")
    if not _is_int(steps) or int(steps) < 1:
        raise ValueError("analysis measurement_steps is invalid")
    steps = int(steps)
    seen_windows = set()
    validated = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("state frequency record must be a mapping")
        window_id = record.get("window_id")
        if window_id not in postprocess.WINDOW_IDS:
            raise ValueError("state frequency record has an unknown window_id")
        if window_id in seen_windows:
            raise ValueError("duplicate state frequency scope")
        seen_windows.add(window_id)
        if record.get("steps") != steps:
            raise ValueError("state frequency steps disagree with the episode")
        region = record.get("region")
        lane = record.get("lane")
        group = record.get("group")
        if region != "full":
            raise ValueError("state frequency records must use the full region")
        if lane != "all":
            raise ValueError("state frequency records must use all lanes")
        if group != "background":
            raise ValueError("state frequency records must use background")
        states = record.get("states")
        if not isinstance(states, list):
            raise ValueError("state frequency record has no states list")
        total = 0
        cleaned = []
        for state in states:
            if not isinstance(state, dict):
                raise ValueError("state frequency state must be a mapping")
            fields = []
            for name in ("N", "S", "Z", "A", "count"):
                value = state.get(name)
                if not _is_int(value) or int(value) < 0:
                    raise ValueError(f"state frequency {name} must be a nonnegative integer")
                fields.append(int(value))
            n, s, z, a, count = fields
            if count < 1:
                raise ValueError("state frequency count must be positive")
            if z > n:
                raise ValueError("stopped count cannot exceed N")
            if s > 5 * n:
                raise ValueError("speed sum exceeds the background vmax bound")
            total += count
            cleaned.append((n, s, z, a, count))
        if total != steps:
            raise ValueError("state frequencies do not sum to the horizon")
        validated.append(
            {
                "window_id": window_id,
                "region": region,
                "lane": lane,
                "group": group,
                "steps": steps,
                "states": cleaned,
            }
        )
    if seen_windows != set(postprocess.WINDOW_IDS):
        raise ValueError("analysis must contain all 12 canonical window frequency scopes")
    return validated


def add_episode_frequencies(accumulator: dict, analysis: dict) -> None:
    """Add one episode's exact frequency table; absent states stay zero."""

    if not isinstance(accumulator, dict) or "scopes" not in accumulator:
        raise ValueError("invalid frequency accumulator")
    validated = _validate_frequency_records(analysis)
    for record in validated:
        key = (
            record["window_id"],
            record["region"],
            record["lane"],
            record["group"],
        )
        scope = accumulator["scopes"].setdefault(key, {})
        steps = record["steps"]
        for n, s, z, a, count in record["states"]:
            frequency = count / steps
            state = (n, s, z, a)
            entry = scope.get(state)
            if entry is None:
                entry = [0.0, 0.0]
                scope[state] = entry
            entry[0] += frequency
            entry[1] += frequency * frequency
    accumulator["episodes"] += 1


def frequency_rows(accumulator: dict):
    """Yield sparse aggregate frequency rows; absent states are implicit zeros."""

    if not isinstance(accumulator, dict):
        raise ValueError("invalid frequency accumulator")
    episodes = accumulator.get("episodes", 0)
    scopes = accumulator.get("scopes", {})
    window_order = {name: index for index, name in enumerate(postprocess.WINDOW_IDS)}
    region_order = {name: index for index, name in enumerate(postprocess.REGIONS)}
    group_order = {name: index for index, name in enumerate(postprocess.GROUPS)}

    def scope_key(item):
        window_id, region, lane, group = item
        lane_key = (0, int(lane)) if _is_int(lane) else (1, 0)
        return (
            window_order[window_id],
            region_order[region],
            lane_key,
            group_order[group],
        )

    for key in sorted(scopes, key=scope_key):
        window_id, region, lane, group = key
        states = scopes[key]
        for state in sorted(states):
            total, squares = states[state]
            n = episodes
            if n <= 0:
                mean = None
                sd = None
            else:
                mean = total / n
                if n >= 2:
                    variance = (squares - total * total / n) / (n - 1)
                    sd = math.sqrt(max(variance, 0.0))
                else:
                    sd = None
            yield {
                "window_id": window_id,
                "region": region,
                "lane": lane,
                "group": group,
                "N": state[0],
                "S": state[1],
                "Z": state[2],
                "A": state[3],
                "sum_frequency": total,
                "sum_sq_frequency": squares,
                "n_episodes": n,
                "mean_frequency": mean,
                "sample_sd_frequency": sd,
                "estimate": "mean_sample_sd_not_ci",
            }


# ---------------------------------------------------------------------------
# Streaming tables
# ---------------------------------------------------------------------------


def _prepare_output(output_dir) -> Path:
    try:
        root = Path(os.path.abspath(os.fspath(output_dir)))
    except TypeError as exc:
        raise ValueError("output_dir is not a filesystem path") from exc
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("output_dir must be a directory")
    return root


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return value
    return str(value)


def _write_json_line(handle, value) -> None:
    handle.write(json.dumps(value, sort_keys=True, allow_nan=False))
    handle.write("\n")


def _install_tables(staging: Path, output: Path, names) -> None:
    for name in names:
        os.replace(staging / name, output / name)


def _canonical_mapping(canonical_ids) -> dict | None:
    if canonical_ids is None:
        return None
    if not isinstance(canonical_ids, dict):
        raise ValueError("canonical_ids must be a mapping task_id -> attempt_id")
    mapping = {}
    for task_id, attempt_id in canonical_ids.items():
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("canonical_ids keys must be nonempty task_id strings")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("canonical_ids values must be nonempty attempt_id strings")
        mapping[task_id] = attempt_id
    return mapping


def stream_episode_tables(
    analysis_paths,
    output_dir,
    *,
    canonical_ids=None,
    planned=None,
    point_fields=POINT_FIELDS,
    require_scientific: bool = False,
) -> dict:
    """Stream accepted episode analyses into bounded machine-readable tables.

    One analysis file is read, validated, converted to a scalar row and then
    released before the next source. Frequency records are spooled to disk;
    only compact offsets and scalar statistics persist across points. Sparse
    frequency sums are retained for one point at a time.  Ambiguous duplicate scientific IDs
    are rejected unless ``canonical_ids`` explicitly chooses one attempt per
    ``task_id``; retries can never increase ``n``.
    """

    fields = _validate_point_fields(point_fields)
    canonical = _canonical_mapping(canonical_ids)
    planned_points = _normalise_planned(planned, fields)
    points = {key: _new_point(value) for key, value in planned_points.items()}
    accepted_ids: set[str] = set()
    counters = {
        "sources": 0,
        "accepted": 0,
        "invalid": 0,
        "skipped_noncanonical": 0,
        "duplicate_attempts": 0,
    }
    frequency_offsets = {}
    expected_columns = None
    metric_columns = None
    campaign_meta = None
    expected_profile = None

    output = _prepare_output(output_dir)
    staging = Path(tempfile.mkdtemp(dir=output, prefix=".tables-"))
    handles = {}
    frequency_spool = tempfile.TemporaryFile(mode="w+b")
    try:
        for name in (
            EPISODES_CSV,
            EPISODES_JSONL,
            SOURCE_INDEX_JSONL,
            POINT_SUMMARY_CSV,
            COVERAGE_CSV,
            FREQUENCIES_CSV,
            MECHANISM_JSONL,
        ):
            handles[name] = open(
                staging / name, "w", encoding="utf-8", newline=""
            )
        csv_writer = None
        row_counts = {name: 0 for name in handles}

        for source in analysis_paths:
            counters["sources"] += 1
            try:
                path = Path(os.fspath(source))
            except TypeError:
                path = None
            sha = None
            index = {
                "source_path": os.path.abspath(os.fspath(source)) if path else str(source),
                "sha256": None,
                "status": "invalid",
                "reason": None,
                "task_id": None,
                "attempt_id": None,
                "analysis_id": None,
                "profile_sha256": None,
            }
            # Stage 1: read/parse/identity failures are recorded coverage, not
            # campaign-fatal: one bad source must not hide the rest.
            try:
                if path is None:
                    raise ValueError("source path is not a filesystem path")
                raw = path.read_bytes()
                sha = hashlib.sha256(raw).hexdigest()
                index["sha256"] = sha
                analysis = _parse_analysis(raw)
                identity = _identity(analysis)
                index.update(
                    sha256=sha,
                    task_id=identity["task_id"],
                    attempt_id=identity["attempt_id"],
                    analysis_id=identity["analysis_id"],
                    profile_sha256=identity["profile_sha256"],
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                counters["invalid"] += 1
                index["reason"] = str(exc)[:500]
                _write_json_line(handles[SOURCE_INDEX_JSONL], index)
                row_counts[SOURCE_INDEX_JSONL] += 1
                continue

            # Stage 2: explicit policy gates.  These refuse the whole table
            # instead of silently lowering n or mixing incompatible analyses.
            if require_scientific and identity["analysis_scope"] != "scientific":
                raise ValueError(
                    "technical-only analysis refused for scientific-campaign "
                    "reporting; supply an explicit scientific analysis profile"
                )
            task_id = identity["task_id"]
            attempt_id = identity["attempt_id"]
            if canonical is not None and task_id in canonical:
                if attempt_id != canonical[task_id]:
                    counters["skipped_noncanonical"] += 1
                    index["status"] = "skipped_noncanonical_retry"
                    index["reason"] = (
                        "canonical_ids selects a different attempt for this task_id"
                    )
                    _write_json_line(handles[SOURCE_INDEX_JSONL], index)
                    row_counts[SOURCE_INDEX_JSONL] += 1
                    continue
            if task_id in accepted_ids:
                if canonical is not None and task_id in canonical:
                    counters["duplicate_attempts"] += 1
                    index["status"] = "skipped_duplicate_attempt"
                    index["reason"] = "the canonical attempt was already counted"
                    _write_json_line(handles[SOURCE_INDEX_JSONL], index)
                    row_counts[SOURCE_INDEX_JSONL] += 1
                    continue
                raise ValueError(
                    f"ambiguous duplicate scientific id {task_id}; provide "
                    "explicit canonical_ids mapping"
                )

            # Stage 3: scalar/frequency extraction.  A structurally malformed
            # analysis with an accepted profile is coverage, not a new n.
            try:
                row = episode_scalar_row(analysis, source_sha256=sha)
                _validate_frequency_records(analysis)
            except (KeyError, TypeError, ValueError) as exc:
                counters["invalid"] += 1
                index["reason"] = str(exc)[:500]
                _write_json_line(handles[SOURCE_INDEX_JSONL], index)
                row_counts[SOURCE_INDEX_JSONL] += 1
                continue

            if expected_profile is None:
                expected_profile = identity["profile_sha256"]
            elif identity["profile_sha256"] != expected_profile:
                raise ValueError(
                    "analysis profiles cannot be mixed in one campaign table"
                )

            if expected_columns is None:
                expected_columns = scalar_columns(row)
                metric_columns = [
                    column
                    for column in expected_columns
                    if column not in _NON_METRIC
                ]
                csv_writer = csv.writer(handles[EPISODES_CSV])
                csv_writer.writerow(expected_columns)
            elif tuple(row) != expected_columns:
                raise ValueError(
                    "scalar row columns changed between analyses; profiles and "
                    "analysis schema must not be mixed"
                )
            csv_writer.writerow([_csv_value(row[column]) for column in expected_columns])
            row_counts[EPISODES_CSV] += 1
            _write_json_line(handles[EPISODES_JSONL], row)
            row_counts[EPISODES_JSONL] += 1

            if campaign_meta is None:
                lineage = analysis.get("lineage")
                if not isinstance(lineage, dict):
                    lineage = {}
                profile_lineage = lineage.get("profile")
                if not isinstance(profile_lineage, dict):
                    profile_lineage = {}
                code_lineage = lineage.get("code")
                if not isinstance(code_lineage, dict):
                    code_lineage = {}
                campaign_meta = {
                    "profile_id": analysis.get("profile_id"),
                    "profile_sha256": analysis.get("profile_sha256"),
                    "technical_only": analysis.get("technical_only"),
                    "analysis_scope": analysis.get("analysis_scope"),
                    "purpose": profile_lineage.get("purpose"),
                    "code_digest": code_lineage.get("source_digest"),
                }
            accepted_ids.add(task_id)
            counters["accepted"] += 1
            key = point_key(analysis, fields)
            point = points.get(key)
            if point is None:
                point = _new_point(None)
                points[key] = point
            point["accepted_n"] += 1
            _accumulate_metrics(point, metric_columns, row)
            frequency_offsets.setdefault(key, []).append(frequency_spool.tell())
            frequency_spool.write(schema.canonical_json({
                "episode": {"measurement_steps": analysis["episode"]["measurement_steps"]},
                "windows": {"state_frequencies": analysis["windows"]["state_frequencies"]},
            }) + b"\n")
            # One episode document per line; distributions are never pooled
            # across points or retained in a campaign-wide sparse accumulator.
            _write_json_line(handles[MECHANISM_JSONL], {
                **{field: row.get(field) for field in fields},
                "task_id": task_id, "attempt_id": row["attempt_id"],
                "profile_sha256": row["profile_sha256"], "source_sha256": sha,
                "scientific_unit": "episode",
                **{name + "_status": row[name + "_status"] for name in ("mechanism", "temporal", "event_alignment", "operational_blocking")},
                **{name: analysis.get(name) for name in ("mechanism", "temporal", "event_alignment", "operational_blocking")},
                "pv_stop_intervals": analysis.get("events", {}).get("pv_stop_intervals"),
            })
            row_counts[MECHANISM_JSONL] += 1
            index["status"] = "accepted"
            _write_json_line(handles[SOURCE_INDEX_JSONL], index)
            row_counts[SOURCE_INDEX_JSONL] += 1
            del analysis, raw

        _write_point_tables(
            handles[POINT_SUMMARY_CSV],
            handles[COVERAGE_CSV],
            points,
            metric_columns or (),
            fields,
        )
        row_counts[POINT_SUMMARY_CSV] = len(points) * len(metric_columns or ())
        row_counts[COVERAGE_CSV] = len(points)
        frequency_count = 0
        frequency_writer = csv.writer(handles[FREQUENCIES_CSV])
        frequency_columns = (
            "window_id",
            "region",
            "lane",
            "group",
            "N",
            "S",
            "Z",
            "A",
            "sum_frequency",
            "sum_sq_frequency",
            "n_episodes",
            "mean_frequency",
            "sample_sd_frequency",
            "estimate",
        )
        frequency_writer.writerow(fields + frequency_columns)
        for key in sorted(frequency_offsets, key=_point_sort_key):
            frequency = frequency_accumulator()
            for offset in frequency_offsets[key]:
                frequency_spool.seek(offset)
                add_episode_frequencies(frequency, json.loads(frequency_spool.readline()))
            for row in frequency_rows(frequency):
                frequency_writer.writerow(
                    [_csv_value(value) for value in key]
                    + [_csv_value(row[name]) for name in frequency_columns]
                )
                frequency_count += 1
            del frequency
        row_counts[FREQUENCIES_CSV] = frequency_count

        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        handles.clear()

        table_specs = {
            "episodes_csv": EPISODES_CSV,
            "episodes_jsonl": EPISODES_JSONL,
            "source_index_jsonl": SOURCE_INDEX_JSONL,
            "point_summary_csv": POINT_SUMMARY_CSV,
            "coverage_csv": COVERAGE_CSV,
            "window_state_frequencies_csv": FREQUENCIES_CSV,
            "mechanism_tables_jsonl": MECHANISM_JSONL,
        }
        tables = {}
        for name, filename in table_specs.items():
            tables[name] = {
                "file": filename,
                "sha256": schema.file_sha256(staging / filename),
                "rows": int(row_counts[filename]),
            }

        profile_id = None
        profile_sha = None
        technical_only = None
        analysis_scope = None
        code_digest = None
        purpose = None
        if isinstance(campaign_meta, dict):
            profile_id = campaign_meta["profile_id"]
            profile_sha = campaign_meta["profile_sha256"]
            technical_only = campaign_meta["technical_only"]
            analysis_scope = campaign_meta["analysis_scope"]
            purpose = campaign_meta["purpose"]
            code_digest = campaign_meta["code_digest"]
        planned_with_data = sum(
            1 for point in points.values() if point["accepted_n"] > 0
        )
        points_missing = sum(
            1
            for point in points.values()
            if point["planned_n"] is not None
            and point["accepted_n"] < point["planned_n"]
        )
        manifest = {
            "schema_version": TABLES_SCHEMA_VERSION,
            "kind": TABLES_KIND,
            "analysis_kind": postprocess.ANALYSIS_KIND,
            "scientific_unit": "episode",
            "point_fields": list(fields),
            "profile": {
                "profile_id": profile_id,
                "profile_sha256": profile_sha,
                "technical_only": technical_only,
                "analysis_scope": analysis_scope,
                "purpose": purpose,
            },
            "code_source_digest": code_digest,
            "counts": {
                **counters,
                "points": len(points),
                "points_with_data": planned_with_data,
                "points_missing_planned": points_missing,
                "frequency_scope_states": frequency_count,
            },
            "tables": tables,
            "estimates": {
                "mean": "arithmetic mean over accepted defined episodes",
                "sample_sd": (
                    "sample standard deviation (ddof=1); null when n_defined<2"
                ),
                "ci": (
                    "not computed; Monte Carlo confidence intervals and crossed "
                    "bootstrap are owned by the statistics module"
                ),
                "NA": (
                    "undefined values are null and counted as n_undefined; absent "
                    "frequency states contribute an implicit zero only within "
                    "the same point's accepted episodes"
                ),
            },
            "canonical_ids_provided": canonical is not None,
            "planned_points_provided": planned is not None,
        }
        manifest_path = staging / TABLES_MANIFEST
        manifest_path.write_bytes(schema.canonical_json(manifest))
        names = list(table_specs.values()) + [TABLES_MANIFEST]
        _install_tables(staging, output, names)
        return {
            "output_dir": str(output),
            "files": {name: str(output / filename) for name, filename in table_specs.items()},
            "manifest": str(output / TABLES_MANIFEST),
            "counts": manifest["counts"],
        }
    finally:
        frequency_spool.close()
        for handle in handles.values():
            try:
                handle.close()
            except Exception:
                pass
        shutil.rmtree(staging, ignore_errors=True)


def _write_point_tables(
    summary_handle,
    coverage_handle,
    points: dict,
    metric_columns,
    fields,
) -> None:
    summary_columns = (
        list(fields)
        + [
            "planned_n",
            "accepted_n",
            "missing_planned",
            "coverage_fraction",
            "metric",
            "n_defined",
            "n_undefined",
            "mean",
            "sample_sd",
            "estimate",
        ]
    )
    coverage_columns = (
        list(fields)
        + [
            "planned_n",
            "accepted_n",
            "missing_planned",
            "excess_accepted",
            "coverage_fraction",
        ]
    )
    coverage_writer = csv.writer(coverage_handle)
    coverage_writer.writerow(coverage_columns)
    for key in sorted(points, key=_point_sort_key):
        point = points[key]
        planned_n = point["planned_n"]
        accepted_n = point["accepted_n"]
        if planned_n is None:
            missing = None
            excess = None
            coverage = None
        else:
            missing = max(planned_n - accepted_n, 0)
            excess = max(accepted_n - planned_n, 0)
            coverage = accepted_n / planned_n if planned_n > 0 else None
        coverage_writer.writerow(
            [_csv_value(value) for value in key]
            + [
                _csv_value(planned_n),
                accepted_n,
                _csv_value(missing),
                _csv_value(excess),
                _csv_value(coverage),
            ]
        )

    summary_writer = csv.writer(summary_handle)
    summary_writer.writerow(summary_columns)
    for key in sorted(points, key=_point_sort_key):
        point = points[key]
        planned_n = point["planned_n"]
        accepted_n = point["accepted_n"]
        if planned_n is None:
            missing = None
            coverage = None
        else:
            missing = max(planned_n - accepted_n, 0)
            coverage = accepted_n / planned_n if planned_n > 0 else None
        for metric in metric_columns:
            state = point["metrics"].get(metric)
            n_defined, mean, sd = _mean_sd(state)
            summary_writer.writerow(
                [_csv_value(value) for value in key]
                + [
                    _csv_value(planned_n),
                    accepted_n,
                    _csv_value(missing),
                    _csv_value(coverage),
                    metric,
                    n_defined,
                    accepted_n - n_defined,
                    _csv_value(mean),
                    _csv_value(sd),
                    "mean_sample_sd_not_ci",
                ]
            )


# ---------------------------------------------------------------------------
# Optional Matplotlib plotting (lazy Agg; recorded data only)
# ---------------------------------------------------------------------------


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "matplotlib is required for plots; install the 'report' extra"
        ) from exc
    return plt


def _plot_lineage(analysis: dict, plot: str, rule: str, data_sha256: str, **extra) -> dict:
    lineage = analysis.get("lineage")
    if not isinstance(lineage, dict):
        lineage = {}
    profile = lineage.get("profile")
    if not isinstance(profile, dict):
        profile = {}
    return {
        "schema_version": 1,
        "plot": plot,
        "rule": rule,
        "data_sha256": data_sha256,
        "profile_id": analysis.get("profile_id"),
        "profile_sha256": analysis.get("profile_sha256"),
        "technical_only": analysis.get("technical_only"),
        "analysis_scope": analysis.get("analysis_scope"),
        "scientific_unit": "episode",
        "source": {
            "input": lineage.get("input"),
            "profile": profile,
            "code": lineage.get("code"),
        },
        "analysis_id": analysis.get("analysis_id"),
        **extra,
    }


def _write_plot(path: Path, analysis: dict, lineage: dict) -> Path:
    path = Path(path)
    if path.suffix.lower() not in (".png", ".pdf", ".svg"):
        raise ValueError("plot path must end in .png, .pdf or .svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = path.with_name(path.name + ".lineage.json")
    descriptor, temporary = tempfile.mkstemp(
        dir=os.fspath(path.parent), prefix=".lineage-", suffix=".json"
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(schema.canonical_json(lineage))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, sidecar)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return sidecar


def plot_pv_trace(analysis: dict, path, *, title=None) -> Path:
    """Plot the recorded PV speed, lane and cumulative distance trace."""

    pv = analysis.get("pv")
    trace = pv.get("trace") if isinstance(pv, dict) else None
    if not isinstance(trace, dict):
        raise ValueError("analysis has no recorded PV trace")
    for name in ("step", "speed", "lane", "cumulative_distance"):
        if not isinstance(trace.get(name), list):
            raise ValueError(f"PV trace is missing {name}")
    plt = _pyplot()
    path = Path(path)
    steps = np.asarray(trace["step"], dtype=np.int64)
    speed = np.asarray(trace["speed"], dtype=np.float64)
    lane = np.asarray(trace["lane"], dtype=np.float64)
    distance = np.asarray(trace["cumulative_distance"], dtype=np.float64)
    figure, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=True)
    axes[0].plot(steps, speed, color="tab:blue", linewidth=1.2)
    axes[0].set_ylabel("PV speed")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(steps, distance, color="tab:green", linewidth=1.2)
    axes[1].set_ylabel("cumulative distance")
    axes[1].grid(True, alpha=0.25)
    axes[2].plot(steps, lane, color="tab:orange", drawstyle="steps-post")
    axes[2].set_ylabel("PV lane")
    axes[2].set_xlabel("measurement-relative step (t=0 context included)")
    axes[2].grid(True, alpha=0.25)
    episode = analysis.get("episode")
    task_label = ""
    if isinstance(episode, dict) and isinstance(episode.get("task_id"), str):
        task_label = episode["task_id"][:12]
    figure.suptitle(
        title
        or (
            "PV trace "
            f"{task_label} "
            f"profile={analysis.get('profile_id')}"
        )
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    lineage = _plot_lineage(
        analysis,
        "pv_trace",
        "all recorded PV trace steps; no trajectory or seed selection",
        schema.digest(trace),
        trace_steps=len(steps),
    )
    _write_plot(path, analysis, lineage)
    return path


def plot_window_comparison(analysis: dict, path, *, title=None) -> Path:
    """Plot all 12 canonical windows for background/all-lane regions."""

    windows = analysis.get("windows")
    if not isinstance(windows, dict) or not isinstance(windows.get("summaries"), list):
        raise ValueError("analysis has no window summaries")
    plt = _pyplot()
    path = Path(path)
    figure, axes = plt.subplots(2, 1, figsize=(10.0, 6.5), sharex=True)
    x = np.arange(len(postprocess.WINDOW_IDS))
    series = {}
    for region in postprocess.REGIONS:
        speeds = []
        stopped = []
        for window_id in postprocess.WINDOW_IDS:
            record = _window_record(analysis, window_id, region, "all", "background")
            value = record.get("pooled_speed") if isinstance(record, dict) else None
            stopped_value = (
                record.get("pooled_stopped_fraction")
                if isinstance(record, dict)
                else None
            )
            speeds.append(np.nan if value is None else float(value))
            stopped.append(np.nan if stopped_value is None else float(stopped_value))
        axes[0].plot(x, speeds, marker="o", linewidth=1.0, label=region)
        axes[1].plot(x, stopped, marker="o", linewidth=1.0, label=region)
        series[region] = {
            "pooled_speed": [
                None if math.isnan(value) else value for value in speeds
            ],
            "stopped": [
                None if math.isnan(value) else value for value in stopped
            ],
        }
    axes[0].set_ylabel("pooled background speed")
    axes[1].set_ylabel("pooled stopped fraction")
    axes[1].set_xlabel("canonical window (12-window axis, never reordered)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(postprocess.WINDOW_IDS, rotation=60, ha="right", fontsize=8)
    axes[0].grid(True, alpha=0.25)
    axes[1].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, ncol=4)
    figure.suptitle(
        title
        or (
            "12-window background comparison "
            f"profile={analysis.get('profile_id')}"
        )
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    lineage = _plot_lineage(
        analysis,
        "window_comparison",
        "all 12 canonical windows; no best-window selection or reordering",
        schema.digest(series),
        windows=list(postprocess.WINDOW_IDS),
    )
    _write_plot(path, analysis, lineage)
    return path


def plot_mechanism_space_time(analysis: dict, arrays: dict, path, *, lane=None, title=None) -> Path:
    """Plot the recorded primary-window occupancy map from saved arrays.

    ``arrays`` must come from the same recorded mechanism episode (for example
    ``storage.read_episode``).  No simulator, model or observer is invoked.
    """

    occupancy = arrays.get("local_occupancy") if isinstance(arrays, dict) else None
    if not isinstance(occupancy, np.ndarray):
        raise ValueError("recorded mechanism arrays have no local_occupancy")
    if occupancy.ndim != 3 or occupancy.shape[1] != postprocess.SPATIAL_CELLS:
        raise ValueError("local_occupancy must be (T, 41, M)")
    steps, cells, lanes = occupancy.shape
    if lane is None:
        panel = occupancy.sum(axis=2).astype(np.float64)
        label = "occupied lanes (0..M)"
    else:
        if not _is_int(lane) or not 0 <= int(lane) < lanes:
            raise ValueError("lane must be inside the saved lane axis")
        panel = occupancy[:, :, int(lane)].astype(np.float64)
        label = f"lane {int(lane)} occupancy"
    plt = _pyplot()
    path = Path(path)
    figure, axis = plt.subplots(1, 1, figsize=(9.0, 4.5))
    image = axis.imshow(
        panel.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=(0.5, steps + 0.5, -10.5, 30.5),
        cmap="viridis",
    )
    axis.set_xlabel("measured step t=1..T")
    axis.set_ylabel("signed longitudinal offset (primary fixed_front_2)")
    axis.set_title(
        title
        or (
            f"mechanism space-time {label} "
            f"profile={analysis.get('profile_id')}"
        )
    )
    figure.colorbar(image, ax=axis, label=label)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    lineage = _plot_lineage(
        analysis,
        "mechanism_space_time",
        "recorded local_occupancy from the mechanism profile; no simulation",
        hashlib.sha256(np.ascontiguousarray(panel).tobytes()).hexdigest(),
        steps=steps,
        cells=cells,
        lanes=lanes,
        lane=lane,
        panel=label,
    )
    _write_plot(path, analysis, lineage)
    return path


# ---------------------------------------------------------------------------
# Illustration selection (fixed training line, paired road records)
# ---------------------------------------------------------------------------


def select_illustration_point(paired_records, *, training_line) -> dict:
    """Select the paired environment closest to the median paired PV effect.

    ``paired_records`` are small paired-road summary records.  Each must carry
    ``training_line``, ``seed``, ``environment`` and ``paired_pv_effect``.  The
    fixed training line is mandatory.  Ties on absolute distance to the median
    are broken by the smallest seed; identical seeds are ambiguous and rejected.
    This is a pre-declared illustration rule, never a best-trajectory choice.
    """

    if not isinstance(training_line, str) or not training_line:
        raise ValueError("training_line must be a fixed nonempty string")
    candidates = []
    for index, record in enumerate(paired_records):
        if not isinstance(record, dict):
            raise ValueError(f"paired_records[{index}] must be a mapping")
        if record.get("training_line") != training_line:
            continue
        seed = record.get("seed")
        environment = record.get("environment")
        effect = record.get("paired_pv_effect")
        if not _is_int(seed):
            raise ValueError(f"paired_records[{index}].seed must be an integer")
        if not isinstance(environment, str) or not environment:
            raise ValueError(f"paired_records[{index}].environment must be a string")
        if _finite_number(effect) is None:
            raise ValueError(f"paired_records[{index}].paired_pv_effect must be finite")
        candidates.append(
            {
                "seed": int(seed),
                "environment": environment,
                "paired_pv_effect": float(effect),
                "method": record.get("method"),
                "index": index,
            }
        )
    if not candidates:
        raise ValueError(
            f"no paired road records for fixed training_line {training_line!r}"
        )
    effects = np.asarray([item["paired_pv_effect"] for item in candidates], dtype=np.float64)
    median = float(np.median(effects))
    for item in candidates:
        item["distance"] = abs(item["paired_pv_effect"] - median)
    best_distance = min(item["distance"] for item in candidates)
    closest = [
        item for item in candidates if item["distance"] == best_distance
    ]
    smallest_seed = min(item["seed"] for item in closest)
    finalists = [item for item in closest if item["seed"] == smallest_seed]
    if len(finalists) != 1:
        raise ValueError(
            "illustration tie is ambiguous: equal median distance and equal seed"
        )
    best = finalists[0]
    return {
        "training_line": training_line,
        "environment": best["environment"],
        "seed": best["seed"],
        "method": best["method"],
        "paired_pv_effect": best["paired_pv_effect"],
        "median_paired_pv_effect": median,
        "absolute_distance_to_median": best["distance"],
        "candidates": len(candidates),
        "selection_rule": (
            "fixed training_line; environment closest to the median paired PV "
            "effect; exact ties broken by the smallest seed; no best-trajectory "
            "or best-seed selection"
        ),
    }
