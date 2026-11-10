"""Offline, NumPy-only episode diagnostics and immutable derivative publication.

This module is deliberately independent of the simulator, Torch, Ray and the
observer producer helpers.  It consumes one already-published episode
(``payload, arrays`` from :func:`physica_mc.storage.read_episode`) and an
explicit analysis profile.  It never simulates, never mutates its inputs and
never turns frames, windows, events or spatial pairs into independent
scientific replicates: the scientific unit remains one episode.

The technical fixture ``configs/technical_analysis.json`` is a frozen
laboratory fixture, not an approved scientific campaign profile.  A profile
must state its own event conditions, minimum durations and spatial choices;
there are no implicit jam3/stop5 thresholds.  An explicit ``spatial: null``
declines the optional mechanism-map covariance tier (base ``full12`` is
sufficient), while a spatial mapping requires those maps.  A profile marked
``technical_only`` is refused for scientific-campaign analysis unless the
caller supplies an explicit ``purpose='scientific'`` profile.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np

from . import schema, storage

ANALYSIS_SCHEMA_VERSION = 1
ANALYSIS_KIND = "episode_diagnostics_v1"

# Canonical observer order; changing it changes the schema, not this module.
WINDOW_IDS = tuple(
    f"{kind}_{shape}_{m}"
    for kind in ("fixed", "speed")
    for m in (1, 2, 3)
    for shape in ("sym", "front")
)
PRIMARY_WINDOW = "fixed_front_2"
REGIONS = ("rear", "centre", "front", "full")
GROUPS = ("HDV", "AV", "background")
SPATIAL_CELLS = 41
SPATIAL_OFFSETS = (-10, 30)

# Hard ceiling for one derivative document.  The declared raw schema already
# bounds T, M and the arrays; this prevents a pathological state-frequency or
# event table from becoming an unbounded export.
MAX_ANALYSIS_BYTES = 64 * 1024 * 1024

PROFILE_SCHEMA_VERSION = 1
_REQUIRED_CONDITIONS = frozenset(
    ("pv_speed_eq_0", "pv_speed_le_1", "nonempty_all_stopped")
)
_SPATIAL_MEASURE = "background_occupancy_connected_covariance"
_SPATIAL_CENTERING = "per_step_per_lane_full_window_mean"
_SPATIAL_NORMALIZATION = "none"
_SPATIAL_PAIR_DOMAIN = "within_window_no_wrap"

_PROFILE_KEYS = {
    "schema_version",
    "profile_id",
    "technical_only",
    "purpose",
    "events",
    "spatial",
    "notes",
    "mechanism",
    "temporal",
    "event_alignment",
}
_BLOCKING = "operational_pv_blocking"
_BLOCKING_LOGIC = {
    "label": "operational_descriptive",
    "logic": "post_pv_speed_eq_0 AND long_pv_leader_gap_eq_0 AND native_phase_pv_native_eligible_count_eq_0",
    "phases": ["post_step_measurement", "after_all_lateral_before_movement", "native_collector_original_pre_step_geometry"],
}
_EVENT_KEYS = {"id", "condition", "min_duration", "window", "region", "lane", "label", "logic", "phases"}
_MECHANISM = {"measure": "integer_frequencies_v1", "contexts": "decision_steps_0_to_T_minus_1", "sentinels": "separate_no_leader_and_not_applicable"}
_SIGNALS = ("pv_speed", "primary_front_N", "primary_front_S", "primary_front_Z")
_CONTROL_NAMES = (
    "step active_av selected_left selected_right selected_stay explicit_stay_commands "
    "available_left available_right eligible_nonstay common_throttled private_throttled "
    "private_suppressed_left common_suppressed requested_nonstay applied_nonstay "
    "rejected_nonstay legal_nonstay_requests applied_legal_nonstay_requests "
    "front_background centre_background rear_background front_hdv centre_hdv rear_hdv "
    "front_av centre_av rear_av requested_front requested_centre requested_rear "
    "applied_front applied_centre applied_rear"
).split()
_SPATIAL_KEYS = {
    "measure",
    "window",
    "lags",
    "centering",
    "normalization",
    "pair_domain",
}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Small strict helpers
# ---------------------------------------------------------------------------


def _is_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _positive_int(value, name: str) -> int:
    if not _is_int(value):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_int(value, name: str) -> int:
    if not _is_int(value):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _mapping(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_keys(value: dict, names, name: str) -> None:
    missing = sorted(set(names) - set(value))
    if missing:
        raise ValueError(f"{name} is missing keys {missing}")


def _reject_json_constant(token: str):
    raise ValueError(f"non-finite JSON constant {token!r} is not allowed")


def _strict_json(raw: bytes, name: str):
    try:
        value = json.loads(raw, parse_constant=_reject_json_constant)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError(f"{name} is not valid JSON: {exc}") from exc
    try:
        schema.canonical_json(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not finite canonical JSON: {exc}") from exc
    return value


def _safe_name(value, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.match(value):
        raise ValueError(f"{name} is not a safe stable identifier")
    return value


# ---------------------------------------------------------------------------
# Analysis profile
# ---------------------------------------------------------------------------


def _validate_event(event, index: int) -> dict:
    event = _mapping(event, f"profile.events[{index}]")
    unknown = sorted(set(event) - _EVENT_KEYS)
    if unknown:
        raise ValueError(f"profile.events[{index}] has unknown keys {unknown}")
    _require_keys(event, ("id", "condition", "min_duration"), f"profile.events[{index}]")
    event_id = _text(event["id"], f"profile.events[{index}].id")
    condition = event["condition"]
    if not isinstance(condition, str):
        raise ValueError(f"profile.events[{index}].condition must be a string")
    if condition not in _REQUIRED_CONDITIONS | {_BLOCKING}:
        raise ValueError(
            f"profile.events[{index}].condition {condition!r} is not one of "
            f"{sorted(_REQUIRED_CONDITIONS)}"
        )
    duration = _positive_int(
        event["min_duration"], f"profile.events[{index}].min_duration"
    )
    common = {
        "id": event_id,
        "condition": condition,
        "min_duration": duration,
    }
    if condition == _BLOCKING:
        if set(event) != {"id", "condition", "min_duration", *_BLOCKING_LOGIC}:
            raise ValueError("blocking requires explicit label, logic and phases only")
        for key, expected in _BLOCKING_LOGIC.items():
            if event[key] != expected:
                raise ValueError(f"blocking {key} must be {expected!r}")
        return {**common, **copy.deepcopy(_BLOCKING_LOGIC)}
    if condition in ("pv_speed_eq_0", "pv_speed_le_1"):
        extra = sorted(set(event) - {"id", "condition", "min_duration"})
        if extra:
            raise ValueError(
                f"profile.events[{index}] is a PV event and must not carry {extra}"
            )
        return common
    # nonempty_all_stopped
    if set(event) - {"id", "condition", "min_duration", "window", "region", "lane"}:
        raise ValueError("window event must not carry blocking choices")
    _require_keys(event, ("window", "region"), f"profile.events[{index}]")
    window = event["window"]
    if window not in WINDOW_IDS:
        raise ValueError(f"profile.events[{index}].window is not a canonical window")
    region = event["region"]
    if region not in REGIONS:
        raise ValueError(f"profile.events[{index}].region is not rear/centre/front/full")
    lane = event.get("lane", "all")
    if lane != "all":
        lane = _nonnegative_int(lane, f"profile.events[{index}].lane")
    common.update(window=window, region=region, lane=lane)
    return common


def _validate_spatial(spatial) -> dict:
    spatial = _mapping(spatial, "profile.spatial")
    unknown = sorted(set(spatial) - _SPATIAL_KEYS)
    if unknown:
        raise ValueError(f"profile.spatial has unknown keys {unknown}")
    _require_keys(spatial, sorted(_SPATIAL_KEYS), "profile.spatial")
    if spatial["measure"] != _SPATIAL_MEASURE:
        raise ValueError(
            f"profile.spatial.measure must be {_SPATIAL_MEASURE!r}; "
            "time-accumulated spatial_hist is not a substitute"
        )
    if spatial["window"] != PRIMARY_WINDOW:
        raise ValueError(
            f"profile.spatial.window must be {PRIMARY_WINDOW!r}: mechanism maps "
            "cover exactly the primary 41 cells"
        )
    if spatial["centering"] != _SPATIAL_CENTERING:
        raise ValueError(f"profile.spatial.centering must be {_SPATIAL_CENTERING!r}")
    if spatial["normalization"] != _SPATIAL_NORMALIZATION:
        raise ValueError(
            f"profile.spatial.normalization must be {_SPATIAL_NORMALIZATION!r}; "
            "the measure is covariance, not Pearson correlation"
        )
    if spatial["pair_domain"] != _SPATIAL_PAIR_DOMAIN:
        raise ValueError(
            f"profile.spatial.pair_domain must be {_SPATIAL_PAIR_DOMAIN!r}"
        )
    lags = spatial["lags"]
    if not isinstance(lags, (list, tuple)):
        raise ValueError("profile.spatial.lags must be a list of integers")
    normalized = []
    for index, lag in enumerate(lags):
        lag = _nonnegative_int(lag, f"profile.spatial.lags[{index}]")
        if lag > SPATIAL_CELLS - 1:
            raise ValueError("profile.spatial.lags must be <= 40")
        normalized.append(lag)
    if not normalized:
        raise ValueError("profile.spatial.lags must not be empty")
    if normalized[0] != 0:
        raise ValueError("profile.spatial.lags must start at 0 for lag-zero variance")
    if any(right <= left for left, right in zip(normalized, normalized[1:])):
        raise ValueError("profile.spatial.lags must be strictly increasing")
    return {
        "measure": spatial["measure"],
        "window": spatial["window"],
        "lags": normalized,
        "centering": spatial["centering"],
        "normalization": spatial["normalization"],
        "pair_domain": spatial["pair_domain"],
    }


def validate_profile(profile, *, require_scientific: bool = False) -> dict:
    """Validate and normalize one explicit analysis profile.

    ``purpose`` is mandatory for a non-technical profile.  A profile with
    ``technical_only=True`` always has ``purpose='technical'`` (an omitted
    purpose is filled with that value for the frozen fixture).  Passing
    ``require_scientific=True`` is the explicit refusal point for using the
    technical fixture as a scientific-campaign analysis.
    """

    profile = _mapping(profile, "analysis profile")
    unknown = sorted(set(profile) - _PROFILE_KEYS)
    if unknown:
        raise ValueError(f"analysis profile has unknown keys {unknown}")
    value = copy.deepcopy(profile)
    if value.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"analysis profile schema_version must be {PROFILE_SCHEMA_VERSION}"
        )
    value["schema_version"] = PROFILE_SCHEMA_VERSION
    value["profile_id"] = _text(value.get("profile_id"), "profile.profile_id")
    if not isinstance(value.get("technical_only"), bool):
        raise ValueError("profile.technical_only must be boolean")
    technical_only = bool(value["technical_only"])
    purpose = value.get("purpose")
    if purpose is None:
        if technical_only:
            purpose = "technical"
        else:
            raise ValueError(
                "a non-technical analysis profile must explicitly set "
                "purpose='scientific'"
            )
    if purpose not in ("technical", "scientific"):
        raise ValueError("profile.purpose must be technical or scientific")
    if technical_only and purpose != "technical":
        raise ValueError(
            "profile.technical_only=true cannot be used as a scientific profile"
        )
    if not technical_only and purpose != "scientific":
        raise ValueError(
            "profile.technical_only=false requires purpose='scientific'"
        )
    value["purpose"] = purpose
    if require_scientific and purpose != "scientific":
        raise ValueError(
            "technical-only analysis profile refused for scientific-campaign "
            "analysis; supply an explicit purpose='scientific' profile"
        )

    events = value.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("profile.events must be a nonempty list")
    normalized_events = []
    identifiers = set()
    conditions = set()
    for index, event in enumerate(events):
        normalized_event = _validate_event(event, index)
        if normalized_event["id"] in identifiers:
            raise ValueError(f"duplicate event id {normalized_event['id']!r}")
        if normalized_event["condition"] in conditions:
            raise ValueError(
                f"duplicate event condition {normalized_event['condition']!r}; "
                "reporting keys one canonical stream per condition"
            )
        identifiers.add(normalized_event["id"])
        conditions.add(normalized_event["condition"])
        normalized_events.append(normalized_event)
    if not _REQUIRED_CONDITIONS <= conditions:
        raise ValueError(
            "profile.events must define one each of "
            f"{sorted(_REQUIRED_CONDITIONS)}; got {sorted(conditions)}"
        )
    value["events"] = normalized_events
    # ``spatial`` is mandatory as an explicit choice.  ``null`` means the
    # profile deliberately declines the optional mechanism-map covariance tier;
    # it is not a silently dropped requested covariance block.
    if "spatial" not in value:
        raise ValueError(
            "profile.spatial is required; use null to explicitly decline the "
            "optional spatial covariance tier"
        )
    if value["spatial"] is not None:
        value["spatial"] = _validate_spatial(value["spatial"])
    _validate_extensions(value)
    notes = value.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError("profile.notes must be a string when present")
    return value


def _validate_extensions(profile):
    # Absent extension keys stay absent, preserving old normalized profile hashes.
    mechanism = profile.get("mechanism")
    if mechanism is not None and mechanism != _MECHANISM:
        raise ValueError(f"profile.mechanism must be null or {_MECHANISM!r}")
    temporal = profile.get("temporal")
    if temporal is not None:
        keys = {"signals", "lags", "centering", "normalization", "pair_domain"}
        if not isinstance(temporal, dict) or set(temporal) != keys:
            raise ValueError("temporal requires explicit signals/lags/centering/normalization/pair_domain")
        if temporal["signals"] != ["pv_speed"] or temporal["centering"] != "episode_mean":
            raise ValueError("temporal supports pv_speed with episode_mean centering")
        if temporal["normalization"] not in ("none", "lag0_variance") or temporal["pair_domain"] != "measurements_only_no_wrap":
            raise ValueError("invalid temporal normalization/pair_domain")
        _ordered_offsets(temporal["lags"], nonnegative=True)
        if temporal["lags"][0] != 0:
            raise ValueError("temporal lags must start at zero")
    alignment = profile.get("event_alignment")
    if alignment is not None:
        keys = {"event_id", "anchor", "offsets", "signals", "left_censored", "sample_domain", "aggregation"}
        if not isinstance(alignment, dict) or set(alignment) != keys:
            raise ValueError("event_alignment requires all explicit choices")
        if alignment["event_id"] not in {event["id"] for event in profile["events"]}:
            raise ValueError("event_alignment references unknown event_id")
        if alignment["anchor"] not in ("onset_step", "confirmation_step", "observed_start"):
            raise ValueError("invalid event_alignment anchor")
        if alignment["left_censored"] not in ("exclude", "include_observed_anchor"):
            raise ValueError("invalid left_censored handling")
        if alignment["anchor"] == "onset_step" and alignment["left_censored"] != "exclude":
            raise ValueError("left-censored onset is unknown and must be excluded")
        if alignment["sample_domain"] != "measurements_only_no_wrap" or alignment["aggregation"] != "within_episode_mean":
            raise ValueError("invalid event_alignment sample_domain/aggregation")
        signals = alignment["signals"]
        if not isinstance(signals, list) or not signals or any(x not in _SIGNALS for x in signals) or len(set(signals)) != len(signals):
            raise ValueError("event_alignment signals must be unique known signals")
        _ordered_offsets(alignment["offsets"], nonnegative=False)


def _ordered_offsets(values, *, nonnegative):
    if not isinstance(values, list) or not values or any(not _is_int(x) for x in values):
        raise ValueError("lags/offsets must be a nonempty integer list")
    if nonnegative and any(x < 0 for x in values):
        raise ValueError("lags must be nonnegative")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError("lags/offsets must be strictly increasing")


def require_scientific_profile(profile) -> dict:
    """Validate a profile and refuse technical-only scientific-campaign use."""

    return load_profile(profile, require_scientific=True)


def load_profile(source, *, require_scientific: bool = False) -> dict:
    """Load a profile from a mapping or a JSON file and validate it."""

    if isinstance(source, dict):
        return validate_profile(source, require_scientific=require_scientific)
    try:
        path = Path(os.fspath(source))
    except TypeError as exc:
        raise ValueError("analysis profile must be a mapping or a JSON path") from exc
    if os.path.islink(path):
        raise ValueError("analysis profile path must not be a symlink")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read analysis profile: {exc}") from exc
    value = _strict_json(raw, "analysis profile")
    return validate_profile(value, require_scientific=require_scientific)


# ---------------------------------------------------------------------------
# Episode context and array validation
# ---------------------------------------------------------------------------


def _array(arrays, name: str, shape=None) -> np.ndarray:
    if name not in arrays:
        raise ValueError(
            f"analysis requires array {name!r}; the saved episode does not "
            "provide it (no placeholder or time-accumulated substitute is used)"
        )
    value = arrays[name]
    if not isinstance(value, np.ndarray):
        raise ValueError(f"array {name!r} must be a NumPy array")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(
            f"array {name!r} has shape {tuple(value.shape)}, expected {tuple(shape)}"
        )
    return value


def _episode_context(payload, arrays, *, require_maps: bool) -> dict:
    payload = _mapping(payload, "episode payload")
    result = payload.get("result", payload)
    result = _mapping(result, "episode result")
    _require_keys(
        result,
        (
            "task_id",
            "execution_id",
            "config",
            "observer_profile",
            "steps_completed",
            "metrics",
            "control",
        ),
        "episode result",
    )
    config = _mapping(result["config"], "result.config")
    _require_keys(config, ("num_lanes", "road_length", "vmax_priority"), "result.config")
    steps = _positive_int(result["steps_completed"], "result.steps_completed")
    if result.get("measurement_steps", steps) != steps:
        raise ValueError("a complete episode must have steps == measurement_steps")
    lanes = _positive_int(config["num_lanes"], "config.num_lanes")
    road_length = _positive_int(config["road_length"], "config.road_length")
    vmax = _positive_int(config["vmax_priority"], "config.vmax_priority")

    pv_speed = _array(arrays, "pv_speed", (steps + 1,))
    pv_lane = _array(arrays, "pv_lane", (steps + 1,))
    pv_pos = _array(arrays, "pv_pos", (steps + 1,))
    pv_path = _array(arrays, "pv_path", (steps + 1,))
    window_hist = _array(arrays, "window_hist", (steps, 12, 3, lanes, 2, 6))
    initial_window_hist = _array(arrays, "initial_window_hist", (12, 3, lanes, 2, 6))
    window_bounds = _array(arrays, "window_bounds", (steps + 1, 12, 2))
    window_transitions = _array(
        arrays, "window_transitions", (steps, 12, 3, lanes, 2, 4)
    )
    global_hist = _array(arrays, "global_hist", (steps, lanes, 2, 6))
    global_lane_changes = _array(arrays, "global_lane_changes", (steps, lanes, 2))
    if require_maps:
        local_occupancy = _array(
            arrays, "local_occupancy", (steps, SPATIAL_CELLS, lanes)
        )
        if np.any((local_occupancy != 0) & (local_occupancy != 1)):
            raise ValueError("local_occupancy values must be 0 or 1")
    else:
        # The profile explicitly declined the spatial covariance tier.  A
        # full12 episode is sufficient for windows, events and frequencies.
        local_occupancy = None
    step = arrays.get("step")
    if step is None:
        step = np.arange(steps + 1, dtype=np.int64)
    elif not isinstance(step, np.ndarray) or tuple(step.shape) != (steps + 1,):
        raise ValueError("array 'step' must have shape (T+1,)")

    if not np.issubdtype(pv_speed.dtype, np.integer):
        raise ValueError("pv_speed must be an integer array")
    if np.any(pv_speed < 0) or np.any(pv_speed > 6):
        raise ValueError("pv_speed values must be in 0..6")
    if np.any(pv_lane < 0) or np.any(pv_lane >= lanes):
        raise ValueError("pv_lane values must be in 0..M-1")
    if np.any(window_hist < 0) or np.any(initial_window_hist < 0):
        raise ValueError("histogram arrays must be nonnegative")
    bounds = window_bounds.astype(np.int64, copy=False)
    measured_bounds = bounds[1:]
    if np.any(measured_bounds[..., 0] > 0) or np.any(measured_bounds[..., 1] < 0):
        raise ValueError("window bounds must include the centre cell")
    if np.any(measured_bounds[..., 0] > -1) or np.any(measured_bounds[..., 1] < 1):
        raise ValueError("canonical windows have a nonempty rear and front region")
    initial_bounds = bounds[0]
    if np.any(initial_bounds[..., 0] > -1) or np.any(initial_bounds[..., 1] < 1):
        raise ValueError("t=0 window context must have nonempty rear and front regions")

    task = payload.get("task")
    science = task.get("science") if isinstance(task, dict) else None
    if not isinstance(science, dict):
        science = {}
    attempt_id = payload.get("attempt_id")
    if attempt_id is not None:
        attempt_id = _safe_name(attempt_id, "episode attempt_id")
    return {
        "result": result,
        "steps": steps,
        "lanes": lanes,
        "road_length": road_length,
        "vmax": vmax,
        "pv_speed": pv_speed,
        "pv_lane": pv_lane,
        "pv_pos": pv_pos,
        "pv_path": pv_path,
        "window_hist": window_hist,
        "initial_window_hist": initial_window_hist,
        "window_bounds": bounds,
        "window_transitions": window_transitions,
        "global_hist": global_hist,
        "global_lane_changes": global_lane_changes,
        "local_occupancy": local_occupancy,
        "step": step,
        "attempt_id": attempt_id,
        "training_line": science.get("training_line"),
        "model_id": science.get("model_id"),
        "candidate_id": science.get("candidate_id"),
    }


# ---------------------------------------------------------------------------
# Exact sufficient-statistic reductions
# ---------------------------------------------------------------------------


def _scope_hist(hist, bounds, region: str, lane, group: str):
    """Select one window/region/lane/group scope.

    ``hist`` is (T, 3, M, 2, 6) and ``bounds`` is (T, 2).  Returns exact
    per-step N, S, Z and geometric A (integers, never NaN placeholders).
    """

    steps, _regions, lanes = hist.shape[:3]
    if region == "full":
        selected = hist.sum(axis=1)  # (T, M, 2, 6)
        width = bounds[:, 1] - bounds[:, 0] + 1
    else:
        index = {"rear": 0, "centre": 1, "front": 2}[region]
        selected = hist[:, index]
        if region == "rear":
            width = -bounds[:, 0]
        elif region == "centre":
            width = np.ones(steps, dtype=np.int64)
        else:
            width = bounds[:, 1]
    if lane == "all":
        selected = selected.sum(axis=1)  # (T, 2, 6)
        lane_count = lanes
    else:
        lane = int(lane)
        if not 0 <= lane < lanes:
            raise ValueError("scope lane is outside the saved lane axis")
        selected = selected[:, lane]
        lane_count = 1
    if group == "background":
        selected = selected.sum(axis=1)  # (T, 6)
    else:
        selected = selected[:, GROUPS.index(group)]
    n = selected.sum(axis=-1, dtype=np.int64)
    s = (selected.astype(np.int64) * np.arange(6, dtype=np.int64)).sum(axis=-1)
    z = selected[..., 0].astype(np.int64)
    area = width.astype(np.int64) * lane_count
    if np.any(area <= 0):
        raise ValueError("geometric area must be positive")
    return n, s, z, area


def _scope_transitions(transitions, region: str, lane, group: str):
    """Select transition counts (T, 3, M, 2, 4) for one scope."""

    if region == "full":
        selected = transitions.sum(axis=1)  # (T, M, 2, 4)
    else:
        selected = transitions[:, {"rear": 0, "centre": 1, "front": 2}[region]]
    if lane == "all":
        selected = selected.sum(axis=1)
    else:
        selected = selected[:, int(lane)]
    if group == "background":
        selected = selected.sum(axis=1)
    else:
        selected = selected[:, GROUPS.index(group)]
    counts = selected.sum(axis=0, dtype=np.int64)
    return [int(value) for value in counts]


def _reduce_scope(n, s, z, area, steps: int) -> dict:
    """Turn exact per-step integers into documented descriptive metrics."""

    N = int(n.sum())
    S = int(s.sum())
    Z = int(z.sum())
    A = int(area.sum())
    defined = int(np.count_nonzero(n))
    empty = steps - defined
    if N:
        pooled_speed = S / N
        pooled_stopped_fraction = Z / N
        deficit_sum = 5 * N - S
        speed_deficit = deficit_sum / N
        normalized_local_delay = deficit_sum / (5 * N)
    else:
        pooled_speed = None
        pooled_stopped_fraction = None
        deficit_sum = 5 * N - S
        speed_deficit = None
        normalized_local_delay = None
    return {
        "N": N,
        "S": S,
        "Z": Z,
        "A": A,
        "vehicle_time": N,
        "cell_time": A,
        "defined_steps": defined,
        "empty_steps": empty,
        "empty_fraction": empty / steps,
        "pooled_speed": pooled_speed,
        "pooled_stopped_fraction": pooled_stopped_fraction,
        "time_mean_density": float(
            np.mean(n.astype(np.float64) / area.astype(np.float64))
        ),
        "time_mean_flow": float(
            np.mean(s.astype(np.float64) / area.astype(np.float64))
        ),
        "cell_time_density": N / A if A else None,
        "cell_time_flow": S / A if A else None,
        "speed_deficit_sum": deficit_sum,
        "speed_deficit": speed_deficit,
        "normalized_local_delay": normalized_local_delay,
    }


def _window_summaries(ctx) -> list[dict]:
    steps = ctx["steps"]
    lanes = ctx["lanes"]
    records = []
    for index, window_id in enumerate(WINDOW_IDS):
        bounds = ctx["window_bounds"][1:, index]
        hist = ctx["window_hist"][:, index]
        for region in REGIONS:
            for lane in list(range(lanes)) + ["all"]:
                for group in GROUPS:
                    n, s, z, area = _scope_hist(hist, bounds, region, lane, group)
                    record = {
                        "window_id": window_id,
                        "region": region,
                        "lane": lane,
                        "group": group,
                    }
                    record.update(_reduce_scope(n, s, z, area, steps))
                    records.append(record)
    return records


def _window_transition_records(ctx) -> list[dict]:
    lanes = ctx["lanes"]
    by_scope = {
        (record["window_id"], record["region"], record["lane"], record["group"]): record
        for record in ctx["window_summaries"]
    }
    records = []
    for index, window_id in enumerate(WINDOW_IDS):
        transitions = ctx["window_transitions"][:, index]
        for region in REGIONS:
            for lane in list(range(lanes)) + ["all"]:
                for group in GROUPS:
                    counts = _scope_transitions(transitions, region, lane, group)
                    exposure = by_scope[(window_id, region, lane, group)]["N"]
                    records.append(
                        {
                            "window_id": window_id,
                            "region": region,
                            "lane": lane,
                            "group": group,
                            "exposure_N": exposure,
                            "categories": [
                                "stop_to_move",
                                "move_to_stop",
                                "acceleration",
                                "deceleration",
                            ],
                            "counts": counts,
                            "rates": [
                                count / exposure if exposure else None
                                for count in counts
                            ],
                        }
                    )
    return records


def _window_state_frequencies(ctx) -> list[dict]:
    steps = ctx["steps"]
    records = []
    for index, window_id in enumerate(WINDOW_IDS):
        bounds = ctx["window_bounds"][1:, index]
        hist = ctx["window_hist"][:, index]
        n, s, z, area = _scope_hist(hist, bounds, "full", "all", "background")
        stacked = np.stack((n, s, z, area), axis=1)
        values, counts = np.unique(stacked, axis=0, return_counts=True)
        states = [
            {
                "N": int(row[0]),
                "S": int(row[1]),
                "Z": int(row[2]),
                "A": int(row[3]),
                "count": int(count),
                "frequency": int(count) / steps,
            }
            for row, count in zip(values, counts)
        ]
        if sum(state["count"] for state in states) != steps:
            raise RuntimeError("state frequency counts do not sum to the horizon")
        records.append(
            {
                "window_id": window_id,
                "region": "full",
                "lane": "all",
                "group": "background",
                "steps": steps,
                "states": states,
            }
        )
    return records


def _window_geometry(ctx) -> list[dict]:
    measured_speed = ctx["pv_speed"][1:].astype(np.int64)
    records = []
    for index, window_id in enumerate(WINDOW_IDS):
        bounds = ctx["window_bounds"][1:, index]
        low = bounds[:, 0]
        high = bounds[:, 1]
        width = high - low + 1
        unique_widths, width_counts = np.unique(width, return_counts=True)
        kind, shape, scale = window_id.split("_")
        records.append(
            {
                "window_id": window_id,
                "kind": kind,
                "shape": shape,
                "scale": int(scale),
                "low_min": int(low.min()),
                "low_max": int(low.max()),
                "high_min": int(high.min()),
                "high_max": int(high.max()),
                "width_values": [
                    {"width": int(value), "steps": int(count)}
                    for value, count in zip(unique_widths, width_counts)
                ],
                "lane_cell_area": [int(value) for value in unique_widths],
                "all_lane_area": [int(value) * ctx["lanes"] for value in unique_widths],
                "floor_applies": kind == "speed",
                "pv_speed_below_1_steps": int(np.count_nonzero(measured_speed < 1)),
                "pv_speed_eq_0_steps": int(np.count_nonzero(measured_speed == 0)),
                "pv_speed_eq_1_steps": int(np.count_nonzero(measured_speed == 1)),
            }
        )
    return records


def _global_background(ctx) -> list[dict]:
    steps = ctx["steps"]
    lanes = ctx["lanes"]
    road_length = ctx["road_length"]
    records = []
    for lane in list(range(lanes)) + ["all"]:
        for group in GROUPS:
            hist = ctx["global_hist"]
            if lane == "all":
                selected = hist.sum(axis=1)  # (T, 2, 6)
                lane_count = lanes
            else:
                selected = hist[:, lane]
                lane_count = 1
            if group == "background":
                selected = selected.sum(axis=1)
            else:
                selected = selected[:, GROUPS.index(group)]
            n = selected.sum(axis=-1, dtype=np.int64)
            s = (selected.astype(np.int64) * np.arange(6, dtype=np.int64)).sum(axis=-1)
            z = selected[..., 0].astype(np.int64)
            area = np.full(steps, road_length * lane_count, dtype=np.int64)
            record = {"lane": lane, "group": group}
            record.update(_reduce_scope(n, s, z, area, steps))
            records.append(record)
    return records


def _global_lane_changes(ctx, global_background) -> list[dict]:
    lanes = ctx["lanes"]
    exposure = {
        (record["lane"], record["group"]): record["N"]
        for record in global_background
    }
    changes = ctx["global_lane_changes"]
    records = []
    for lane in list(range(lanes)) + ["all"]:
        for group in GROUPS:
            if lane == "all":
                selected = changes.sum(axis=1)
            else:
                selected = changes[:, lane]
            if group == "background":
                count = int(selected.sum())
            else:
                count = int(selected[:, GROUPS.index(group)].sum())
            denominator = exposure[(lane, group)]
            records.append(
                {
                    "lane": lane,
                    "group": group,
                    "changes": count,
                    "exposure_N": denominator,
                    "rate": count / denominator if denominator else None,
                }
            )
    return records


# ---------------------------------------------------------------------------
# PV trace, events and censoring
# ---------------------------------------------------------------------------


def _pv_summary(ctx) -> dict:
    steps = ctx["steps"]
    road_length = ctx["road_length"]
    speed = ctx["pv_speed"].astype(np.int64)
    lane = ctx["pv_lane"].astype(np.int64)
    pos = ctx["pv_pos"].astype(np.int64)
    path = ctx["pv_path"].astype(np.int64)
    measured_speed = speed[1:]
    distance = int(path[-1]) - int(path[0])
    stopped = int(np.count_nonzero(measured_speed == 0))
    lane_changes = int(np.count_nonzero(np.diff(lane)))
    cumulative = path - path[0]
    crossings = (np.flatnonzero(np.diff(cumulative // road_length)) + 1).tolist()
    lap_times = np.diff(np.r_[0, np.asarray(crossings, dtype=np.int64)]).tolist()
    trace = {
        "step": [int(value) for value in ctx["step"]],
        "speed": [int(value) for value in speed],
        "lane": [int(value) for value in lane],
        "pos": [int(value) for value in pos],
        "path": [int(value) for value in path],
        "cumulative_distance": [int(value) for value in cumulative],
        "includes_t0_context": True,
    }
    return {
        "distance": distance,
        "mean_speed": distance / steps,
        "normalized_time_loss": 1 - distance / (ctx["vmax"] * steps),
        "stopped_steps": stopped,
        "stopped_fraction": stopped / steps,
        "moving_steps": steps - stopped,
        "moving_mean_speed": distance / (steps - stopped) if steps > stopped else None,
        "speed_counts": [int(value) for value in np.bincount(measured_speed, minlength=7)],
        "lane_changes": lane_changes,
        "lane_change_rate": lane_changes / steps,
        "completed_laps": len(crossings),
        "lap_completion_steps": [int(value) for value in crossings],
        "completed_lap_times": [int(value) for value in lap_times],
        "mean_completed_lap_time": (
            float(np.mean(lap_times)) if lap_times else None
        ),
        "remaining_distance": int(distance % road_length),
        "no_completed_laps": not crossings,
        "initial": {
            "speed": int(speed[0]),
            "lane": int(lane[0]),
            "pos": int(pos[0]),
        },
        "trace": trace,
    }


def _mask_runs(mask) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, active in enumerate(mask):
        if bool(active) and start is None:
            start = index
        elif not bool(active) and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _event_stream(
    *,
    event: dict,
    mask,
    context_at_t0: bool,
    pv_speed,
    steps: int,
    kind: str,
    threshold,
    n=None,
    z=None,
    area=None,
) -> dict:
    """Build one confirmed, boundary-censored event stream.

    ``mask`` covers measured steps t=1..T (indices 0..T-1).  ``context_at_t0``
    is the same condition evaluated at the separate t=0 context snapshot.
    """

    events = []
    unconfirmed = 0
    unconfirmed_terminal = 0
    for start, end in _mask_runs(mask):
        length = end - start + 1
        if length < event["min_duration"]:
            unconfirmed += 1
            if end == steps - 1:
                unconfirmed_terminal += 1
            continue
        observed_start = start + 1
        observed_end = end + 1
        left_censored = bool(start == 0 and context_at_t0)
        right_censored = bool(end == steps - 1)
        onset = None if left_censored else observed_start
        end_step = None if right_censored else observed_end
        duration = (
            end_step - onset + 1
            if onset is not None and end_step is not None
            else None
        )
        exit_reason = None
        exit_observation = None
        restart = False
        restart_step = None
        empty_window_exit = False
        if not right_censored:
            next_index = end + 1
            # measured index next_index corresponds to t=next_index+1 in the
            # T+1 PV trace, because index 0 is the t=0 context snapshot.
            pv_exit = int(pv_speed[next_index + 1])
            if kind in ("pv", "blocking"):
                exit_observation = {
                    "step": observed_end + 1,
                    "pv_speed": pv_exit,
                }
                restart = pv_exit > threshold
                exit_reason = "pv_restart" if restart else "condition_ended"
            else:
                n_exit = int(n[next_index])
                z_exit = int(z[next_index])
                a_exit = int(area[next_index])
                exit_observation = {
                    "step": observed_end + 1,
                    "N": n_exit,
                    "Z": z_exit,
                    "A": a_exit,
                    "pv_speed": pv_exit,
                }
                if n_exit == 0:
                    exit_reason = "window_empty"
                    empty_window_exit = True
                elif z_exit < n_exit:
                    exit_reason = "vehicles_resumed"
                else:
                    exit_reason = "condition_ended"
                # PV restart is an observed PV fact, never inferred from the
                # local window becoming empty.
                restart = pv_exit > 0
            if restart:
                restart_step = observed_end + 1
        else:
            exit_reason = "right_censored_horizon"
        record = {
            "event_id": event["id"],
            "condition": event["condition"],
            "min_duration": event["min_duration"],
            "condition_scope": kind,
            "onset_step": onset,
            "observed_start": observed_start,
            "confirmation_step": observed_start + event["min_duration"] - 1,
            "observed_end": observed_end,
            "end_step": end_step,
            "duration_steps": duration,
            "observed_duration_steps": observed_end - observed_start + 1,
            "left_censored": left_censored,
            "right_censored": right_censored,
            "exit_reason": exit_reason,
            "exit_observation": exit_observation,
            "restart_observed": restart,
            "restart_step": restart_step,
            "empty_window_exit": empty_window_exit,
        }
        if kind in ("pv", "blocking"):
            record["speed_threshold"] = int(threshold)
        elif kind == "window":
            record["window"] = event["window"]
            record["region"] = event["region"]
            record["lane"] = event["lane"]
        events.append(record)

    fully_observed = [
        event["duration_steps"]
        for event in events
        if event["duration_steps"] is not None
    ]
    observed_lengths = [event["observed_duration_steps"] for event in events]
    exit_counts = {}
    for record in events:
        exit_counts[record["exit_reason"]] = exit_counts.get(record["exit_reason"], 0) + 1
    stream = {
        "event_id": event["id"],
        "condition": event["condition"],
        "min_duration": event["min_duration"],
        "confirmed_count": len(events),
        "left_censored_count": sum(1 for event in events if event["left_censored"]),
        "right_censored_count": sum(1 for event in events if event["right_censored"]),
        "fully_observed_count": len(fully_observed),
        "observed_restart_count": sum(
            1 for event in events if event["restart_observed"]
        ),
        "unconfirmed_runs": unconfirmed,
        "unconfirmed_terminal_runs": unconfirmed_terminal,
        "exit_reason_counts": exit_counts,
        "mean_observed_duration": (
            float(np.mean(observed_lengths)) if observed_lengths else None
        ),
        "descriptive_duration": {
            "mask": "fully_observed_only",
            "n": len(fully_observed),
            "mean": float(np.mean(fully_observed)) if fully_observed else None,
            "note": (
                "conditional descriptive duration, not a survival or "
                "censoring-adjusted estimate"
            ),
        },
        "events": events,
    }
    if kind in ("pv", "blocking"):
        stream["speed_threshold"] = int(threshold)
    elif kind == "window":
        stream["window"] = event["window"]
        stream["region"] = event["region"]
        stream["lane"] = event["lane"]
    if kind == "blocking":
        stream.update(copy.deepcopy(_BLOCKING_LOGIC))
        stream["initial_context"] = "unknown; first measured run is left-censored"
    return stream


def _events(ctx, profile) -> dict:
    steps = ctx["steps"]
    lanes = ctx["lanes"]
    pv_speed = ctx["pv_speed"].astype(np.int64)
    measured_speed = pv_speed[1:]
    streams = []
    for event in profile["events"]:
        condition = event["condition"]
        if condition == "pv_speed_eq_0":
            mask = measured_speed == 0
            context = bool(pv_speed[0] == 0)
            threshold = 0
            kind = "pv"
            n = z = area = None
        elif condition == "pv_speed_le_1":
            mask = measured_speed <= 1
            context = bool(pv_speed[0] <= 1)
            threshold = 1
            kind = "pv"
            n = z = area = None
        elif condition == _BLOCKING:
            gap = _integer_column(ctx["arrays"], "long_pv_leader_gap", steps, "int32", minimum=-1)
            eligible = _integer_column(ctx["arrays"], "native_phase_pv_native_eligible_count", steps, "int8", maximum=2)
            mask = (measured_speed == 0) & (gap == 0) & (eligible == 0)
            context = True  # no complete mixed-phase context before measurement 1
            threshold = 0
            kind = "blocking"
            n = z = area = None
        else:
            window_index = WINDOW_IDS.index(event["window"])
            if event["lane"] != "all" and int(event["lane"]) >= lanes:
                raise ValueError("event lane is outside the saved lane axis")
            hist = ctx["window_hist"][:, window_index]
            bounds = ctx["window_bounds"][1:, window_index]
            n, _s, z, area = _scope_hist(
                hist, bounds, event["region"], event["lane"], "background"
            )
            mask = (n > 0) & (z == n)
            initial = ctx["initial_window_hist"][window_index : window_index + 1]
            bounds0 = ctx["window_bounds"][0, window_index][None, :]
            n0, _s0, z0, _a0 = _scope_hist(
                initial, bounds0, event["region"], event["lane"], "background"
            )
            context = bool(n0[0] > 0 and z0[0] == n0[0])
            threshold = None
            kind = "window"
        streams.append(
            _event_stream(
                event=event,
                mask=mask,
                context_at_t0=context,
                pv_speed=pv_speed,
                steps=steps,
                kind=kind,
                threshold=threshold,
                n=n,
                z=z,
                area=area,
            )
        )
    summary = {
        "stream_count": len(streams),
        "confirmed_events": sum(stream["confirmed_count"] for stream in streams),
        "left_censored_events": sum(
            stream["left_censored_count"] for stream in streams
        ),
        "right_censored_events": sum(
            stream["right_censored_count"] for stream in streams
        ),
        "fully_observed_events": sum(
            stream["fully_observed_count"] for stream in streams
        ),
        "observed_restarts": sum(
            stream["observed_restart_count"] for stream in streams
        ),
        "unconfirmed_runs": sum(stream["unconfirmed_runs"] for stream in streams),
        "scientific_unit": "episode",
        "note": (
            "Events and windows within one trajectory do not increase "
            "independent n; fully observed durations are conditional "
            "descriptive values, not survival estimates."
        ),
    }
    return {"streams": streams, "summary": summary}


# ---------------------------------------------------------------------------
# Spatial covariance (exact technical measure; not Pearson correlation)
# ---------------------------------------------------------------------------


def _spatial_covariance(local_occupancy, spatial: dict) -> dict:
    occupancy = np.asarray(local_occupancy)
    if occupancy.ndim != 3:
        raise ValueError("local_occupancy must be (T, 41, M)")
    steps, cells, lanes = occupancy.shape
    if cells != SPATIAL_CELLS:
        raise ValueError("local_occupancy must contain the primary 41 cells")
    if np.any((occupancy != 0) & (occupancy != 1)):
        raise ValueError("local_occupancy values must be 0 or 1")
    values = occupancy.astype(np.float64, copy=False)
    centered = values - values.mean(axis=1, keepdims=True)
    variance0 = np.mean(centered * centered, axis=1)  # (T, M)
    lags = spatial["lags"]
    covariance = []
    covariance_by_lane = []
    pair_counts = []
    for lag in lags:
        if lag == 0:
            products = centered * centered
        else:
            products = centered[:, : cells - lag, :] * centered[:, lag:, :]
        per_frame_lane = products.mean(axis=1)  # (T, M)
        covariance.append(float(per_frame_lane.mean()))
        covariance_by_lane.append(
            [float(value) for value in per_frame_lane.mean(axis=0)]
        )
        pair_counts.append(int(steps * lanes * (cells - lag)))
    zero_frames = variance0 == 0.0
    total_frames = steps * lanes
    by_lane = []
    for lane in range(lanes):
        lane_variance = variance0[:, lane]
        zero_lane = int(np.count_nonzero(lane_variance == 0.0))
        by_lane.append(
            {
                "lane": lane,
                "pair_counts": [int(steps * (cells - lag)) for lag in lags],
                "covariance": [values[lane] for values in covariance_by_lane],
                "lag0_variance": float(lane_variance.mean()),
                "zero_variance_frames": zero_lane,
            }
        )
    return {
        "measure": spatial["measure"],
        "window": spatial["window"],
        "cells": cells,
        "offsets": [SPATIAL_OFFSETS[0], SPATIAL_OFFSETS[1]],
        "lanes": lanes,
        "steps": steps,
        "lags": list(lags),
        "centering": spatial["centering"],
        "normalization": spatial["normalization"],
        "pair_domain": spatial["pair_domain"],
        "pair_counts": pair_counts,
        "pair_count_definition": (
            "sum over frames and lanes of (41 - lag) valid within-frame/lane "
            "pairs; no end wrap and no periodic pairing"
        ),
        "lag0_variance": float(variance0.mean()),
        "zero_variance_frames": int(np.count_nonzero(zero_frames)),
        "zero_variance_total_frames": int(total_frames),
        "zero_variance_fraction": int(np.count_nonzero(zero_frames)) / total_frames,
        "covariance": covariance,
        "covariance0": covariance[lags.index(0)],
        "covariance_by_lane": covariance_by_lane,
        "by_lane": by_lane,
        "defined": True,
        "is_correlation": False,
        "measure_kind": "covariance_not_pearson_correlation",
        "source_arrays": ["local_occupancy"],
        "time_accumulated_spatial_hist_used": False,
        "inferential_unit": "episode",
        "pair_note": (
            "frame/lane/pair counts are descriptive sufficient statistics; "
            "they do not add independent scientific replicates"
        ),
        "note": (
            "Covariance at lag zero is defined even when the variance is "
            "zero; no normalization by variance is applied."
        ),
    }


# ---------------------------------------------------------------------------
# Explicit W5 extensions: recorded facts only, one episode at a time
# ---------------------------------------------------------------------------


def _integer_column(arrays, name, steps, dtype, *, minimum=0, maximum=None):
    values = _array(arrays, name, (steps,))
    if values.dtype != np.dtype(dtype):
        raise ValueError(f"{name} must have dtype {dtype}")
    if np.any(values < minimum) or (maximum is not None and np.any(values > maximum)):
        raise ValueError(f"{name} values outside the recorded domain")
    return values


def _integer_frequency(name, values, *, phase, sentinel=None, domain=None):
    values = np.asarray(values)
    valid = values >= 0
    selected = values[valid]
    unique, counts = np.unique(selected, return_counts=True)
    count_by_value = dict(zip(unique.tolist(), counts.tolist()))
    if domain is None:
        domain = sorted(count_by_value)
    denominator = int(selected.size)
    return {
        "name": name,
        "phase": phase,
        "contexts": int(values.size),
        "denominator": denominator,
        "denominator_definition": "contexts with an applicable observed value; excludes sentinel",
        "sentinel": sentinel,
        "sentinel_count": int(values.size - selected.size),
        "status": "observed" if denominator else "not_applicable",
        "mean": float(selected.mean()) if denominator else None,
        "states": [{"value": int(value), "count": int(count_by_value.get(value, 0)),
                    "frequency": count_by_value.get(value, 0) / denominator if denominator else None}
                   for value in domain],
    }


def _mechanism(ctx):
    arrays, steps = ctx["arrays"], ctx["steps"]
    records = []
    # Fixed raw contract; do not import diagnostics (it imports the simulator).
    columns = {
        "pre_pv_leader_gap": ("int32", -1, None, "no_leader"),
        "pre_pv_leader_type": ("int8", -1, 3, "no_leader"),
        "pre_pv_leader_speed": ("int8", -1, 6, "no_leader"),
        "pre_pv_leader_stopped": ("int8", -1, 1, "no_leader"),
        "long_pv_leader_gap": ("int32", -1, None, "no_leader"),
        "long_pv_leader_type": ("int8", -1, 3, "no_leader"),
        "long_pv_leader_speed": ("int8", -1, 6, "no_leader"),
        "long_pv_leader_final_speed": ("int8", -1, 6, "no_leader"),
        "long_pv_eq9_slack": ("int32", -1, None, "no_leader"),
        "long_pv_eq9_tight": ("int8", -1, 1, "no_leader"),
        "long_pv_acceleration_gate": ("int8", 0, 1, None),
        "long_pv_v1": ("int8", 0, 6, None),
    }
    for phase in ("pre", "native_phase"):
        for side in ("left", "right", "count"):
            columns[f"{phase}_pv_native_eligible_{side}"] = ("int8", 0, 2 if side == "count" else 1, None)
    for side in ("left", "right"):
        columns[f"pre_pv_native_{side}"] = ("int8", -1, 1, "lane_missing")
        columns[f"pre_pv_native_{side}_reason"] = ("int8", 0, 3, None)
    for phase in ("prev", "cur"):
        for bound in ("s1", "sS"):
            columns[f"long_pv_{phase}_gap_{bound}"] = ("int64", 0, None, None)
    for name, (dtype, minimum, maximum, sentinel) in columns.items():
        values = _integer_column(arrays, name, steps, dtype, minimum=minimum, maximum=maximum)
        phase = ("after_all_lateral_before_movement" if name.startswith("long_") else
                 "native_collector_original_pre_step_geometry" if name.startswith("native_phase_") else "pre_step")
        if name in ("long_pv_leader_final_speed", "long_pv_eq9_slack", "long_pv_eq9_tight"):
            phase = "longitudinal_final_speeds_in_post_lateral_geometry"
        records.append(_integer_frequency(name, values, phase=phase, sentinel=sentinel,
                                          domain=range(maximum + 1) if maximum is not None else None))
    for phase in ("pre", "native_phase"):
        left, right, count = (arrays[f"{phase}_pv_native_eligible_{side}"] for side in ("left", "right", "count"))
        if not np.array_equal(left + right, count):
            raise ValueError("native eligible count disagrees with directions")
        records.append(_integer_frequency(f"{phase}_pv_native_direction_set", left + 2 * right,
                                          phase="pre_step" if phase == "pre" else "native_collector_original_pre_step_geometry",
                                          domain=range(4)))
    for side in ("left", "right", "count"):
        if not np.array_equal(arrays[f"pre_pv_native_eligible_{side}"], arrays[f"native_phase_pv_native_eligible_{side}"]):
            raise ValueError("native collector eligibility must use original pre-step geometry")
    for phase in ("pre", "long"):
        absent = arrays[f"{phase}_pv_leader_gap"] == -1
        suffixes = ("type", "speed", "stopped") if phase == "pre" else ("type", "speed", "final_speed")
        for suffix in suffixes:
            if not np.array_equal(arrays[f"{phase}_pv_leader_{suffix}"] == -1, absent):
                raise ValueError("inconsistent no-leader sentinels")
        if phase == "long":
            slack, tight = arrays["long_pv_eq9_slack"], arrays["long_pv_eq9_tight"]
            if not np.array_equal(slack == -1, absent) or not np.array_equal(tight == -1, absent):
                raise ValueError("Eq9 no-leader sentinels disagree")
            if not np.array_equal(tight[~absent], (slack[~absent] == 0).astype(np.int8)):
                raise ValueError("Eq9 tightness disagrees with slack")
    trace = _array(arrays, "control_trace", (steps, len(_CONTROL_NAMES)))
    metadata = ctx["result"].get("control_columns")
    if trace.dtype != np.dtype("int64") or not isinstance(metadata, dict):
        raise ValueError("mechanism requires int64 control_trace and control_columns")
    for index, name in enumerate(_CONTROL_NAMES):
        column = _mapping(metadata.get(name, {}), f"control_columns.{name}")
        if column.get("index") != index or column.get("dtype") != "int64":
            raise ValueError(f"control column metadata mismatch: {name}")
    if not np.array_equal(trace[:, 0], np.arange(steps)) or np.any(trace[:, 1:] < -1):
        raise ValueError("invalid pre-step control contexts/counts")
    # B1/B2 historically store active_av=0 although the common active set is
    # absent. Preserve that distinction even when a producer stores zero.
    common = set(_CONTROL_NAMES[1:13]) | {"legal_nonstay_requests", "applied_legal_nonstay_requests"}
    for index, name in enumerate(_CONTROL_NAMES[1:], 1):
        values = trace[:, index]
        if ctx["result"].get("method") in ("B1", "B2") and name in common:
            values = np.full(steps, -1, dtype=np.int64)
        records.append(_integer_frequency("control_" + name, values, phase="pre_step_decision_context",
                                          sentinel="not_applicable"))
    # Fixed primary local counts at T decision contexts, including initial
    # context and excluding the final measured state (not T+1 samples).
    wi = WINDOW_IDS.index(PRIMARY_WINDOW)
    pre_hist = np.concatenate((ctx["initial_window_hist"][wi:wi + 1], ctx["window_hist"][:-1, wi]), axis=0)
    for region in ("rear", "centre", "front"):
        for group in ("HDV", "AV"):
            n, _, _, _ = _scope_hist(pre_hist, ctx["window_bounds"][:-1, wi], region, "all", group)
            records.append(_integer_frequency(f"primary_pre_{region}_{group}_count", n, phase="pre_step"))
    return {
        "tables": records,
        "decision_contexts": steps,
        "direction_set_codes": {"0": "none", "1": "left", "2": "right", "3": "both"},
        "inferential_unit": "episode",
        "note": "Native eligibility excludes stochastic draws, direction choice and reservation/conflict filtering. Longitudinal bounds and Eq9 tightness are observed constraints, not causal explanations. Active-set zones are not saved; zone counts cover HDV/AV and requested/applied interventions, active_av is the common-window total.",
    }


def _stop_intervals(stream):
    records = stream["events"]
    intervals = []
    for previous, following in zip(records, records[1:]):
        restart, onset = previous["restart_step"], following["onset_step"]
        if restart is not None and onset is not None:
            intervals.append({"previous_observed_end": previous["observed_end"],
                              "restart_step": restart, "next_onset_step": onset,
                              "observed_gap_steps": onset - restart,
                              "onset_to_onset_steps": onset - previous["onset_step"] if previous["onset_step"] is not None else None})
    gaps = [record["observed_gap_steps"] for record in intervals]
    return {"min_duration": stream["min_duration"], "intervals": intervals,
            "observed_interval_count": len(gaps), "mean_observed_gap": float(np.mean(gaps)) if gaps else None,
            "observed_restart_count": stream["observed_restart_count"],
            "definition": "Between consecutive confirmed stop records: [observed restart, next known onset); short unconfirmed stops may occur inside. Onset-to-onset is null for a left-censored prior stop. No terminal restart or terminal waiting interval is invented; duration threshold selects records, not a survival sample.",
            "inferential_unit": "episode"}


def _temporal(speed, profile):
    values = np.asarray(speed, dtype=np.float64)
    centered = values - values.mean()
    variance = float(np.mean(centered * centered))
    records = []
    for lag in profile["lags"]:
        pairs = max(0, len(values) - lag)
        covariance = float(np.dot(centered[:pairs], centered[lag:lag + pairs]) / pairs) if pairs else None
        value = covariance
        if profile["normalization"] == "lag0_variance":
            value = covariance / variance if pairs and variance > 0 else None
        records.append({"lag": lag, "pair_count": pairs, "covariance": covariance, "value": value})
    return {"signal": "pv_speed", "mean": float(values.mean()), "lag0_variance": variance,
            "records": records, "choices": copy.deepcopy(profile), "inferential_unit": "episode"}


def _event_alignment(ctx, events, profile):
    stream = next(s for s in events["streams"] if s["event_id"] == profile["event_id"])
    anchors = []
    for event in stream["events"]:
        if event["left_censored"] and profile["left_censored"] == "exclude":
            continue
        anchor = event[profile["anchor"]]
        if anchor is not None:
            anchors.append(anchor)
    signals = {"pv_speed": ctx["pv_speed"][1:]}
    if any(name != "pv_speed" for name in profile["signals"]):
        wi = WINDOW_IDS.index(PRIMARY_WINDOW)
        n, s, z, _ = _scope_hist(ctx["window_hist"][:, wi], ctx["window_bounds"][1:, wi], "front", "all", "background")
        signals.update(primary_front_N=n, primary_front_S=s, primary_front_Z=z)
    records = []
    for signal in profile["signals"]:
        for offset in profile["offsets"]:
            # t=0 is context, never an aligned sample. No wrapping/imputation.
            indices = [anchor + offset - 1 for anchor in anchors if 1 <= anchor + offset <= ctx["steps"]]
            values = signals[signal][indices]
            records.append({"signal": signal, "offset": offset, "valid_count": len(indices),
                            "sum": int(values.sum(dtype=np.int64)),
                            "mean": float(values.mean()) if indices else None})
    return {"choices": copy.deepcopy(profile), "confirmed_events": stream["confirmed_count"],
            "included_anchors": len(anchors), "records": records, "inferential_unit": "episode",
            "note": "Overlapping event windows remain within-episode descriptive observations; valid_count does not increase Monte Carlo n."}


# ---------------------------------------------------------------------------
# Pure public analysis
# ---------------------------------------------------------------------------


def analyze_episode(payload, arrays, profile) -> dict:
    """Analyze one published episode; pure, deterministic and NumPy-only."""

    normalized_profile = validate_profile(profile)
    require_maps = normalized_profile["spatial"] is not None
    ctx = _episode_context(payload, arrays, require_maps=require_maps)
    ctx["arrays"] = arrays
    steps = ctx["steps"]
    lanes = ctx["lanes"]
    result = ctx["result"]

    ctx["window_summaries"] = _window_summaries(ctx)
    window_summaries = ctx["window_summaries"]
    window_transitions = _window_transition_records(ctx)
    state_frequencies = _window_state_frequencies(ctx)
    window_geometry = _window_geometry(ctx)
    global_background = _global_background(ctx)
    global_lane_changes = _global_lane_changes(ctx, global_background)
    pv = _pv_summary(ctx)
    events = _events(ctx, copy.deepcopy(normalized_profile))
    if require_maps:
        spatial = _spatial_covariance(
            ctx["local_occupancy"], normalized_profile["spatial"]
        )
        spatial_status = "computed"
    else:
        spatial = None
        spatial_status = "not_requested"

    extensions = {
        "mechanism": _mechanism(ctx) if normalized_profile.get("mechanism") is not None else None,
        "temporal": _temporal(ctx["pv_speed"][1:], normalized_profile["temporal"]) if normalized_profile.get("temporal") is not None else None,
        "event_alignment": _event_alignment(ctx, events, normalized_profile["event_alignment"]) if normalized_profile.get("event_alignment") is not None else None,
    }
    stop_stream = next(stream for stream in events["streams"] if stream["condition"] == "pv_speed_eq_0")
    events["pv_stop_intervals"] = _stop_intervals(stop_stream)
    blocking_requested = any(event["condition"] == _BLOCKING for event in normalized_profile["events"])
    episode = {
        "task_id": result["task_id"],
        "execution_id": result["execution_id"],
        "attempt_id": ctx["attempt_id"],
        "purpose": result.get("purpose"),
        "split": result.get("split"),
        "method": result.get("method"),
        "training_line": ctx["training_line"],
        "model_id": ctx["model_id"],
        "candidate_id": ctx["candidate_id"],
        "density_index": result.get("density_index"),
        "replicate": result.get("replicate"),
        "priority_id": result.get("priority_id"),
        "observer_profile": result.get("observer_profile"),
        "measurement_steps": steps,
        "num_lanes": lanes,
        "road_length": ctx["road_length"],
        "config": copy.deepcopy(result["config"]),
    }
    analysis = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_kind": ANALYSIS_KIND,
        "scientific_unit": "episode",
        "independence": {
            "independent_episodes": 1,
            "events_frames_and_pairs_do_not_add_n": True,
            "cross_episode_ci": (
                "not computed here; crossed bootstrap and cross-episode "
                "statistics are owned by the statistics module"
            ),
        },
        "timing_convention": {
            "events": (
                "post-step measured states t=1..T; t=0 is a separate context "
                "snapshot and never a measured sample"
            ),
            "windows_and_histograms": "post-step measured states",
            "spatial_covariance": (
                "post-step local_occupancy frames"
                if require_maps
                else "not requested by the analysis profile"
            ),
            "pre_step_control_or_native_columns_used": extensions["mechanism"] is not None or blocking_requested,
            "operational_blocking_computed": blocking_requested,
        },
        "profile": normalized_profile,
        "profile_id": normalized_profile["profile_id"],
        "profile_sha256": schema.digest(normalized_profile),
        "technical_only": normalized_profile["technical_only"],
        "analysis_scope": (
            "scientific"
            if normalized_profile["purpose"] == "scientific"
            else "technical"
        ),
        "episode": episode,
        "windows": {
            "ids": list(WINDOW_IDS),
            "primary": PRIMARY_WINDOW,
            "regions": list(REGIONS),
            "lanes": list(range(lanes)) + ["all"],
            "groups": list(GROUPS),
            "summaries": window_summaries,
            "transitions": window_transitions,
            "state_frequencies": state_frequencies,
            "geometry": window_geometry,
        },
        "global_background": global_background,
        "global_lane_changes": global_lane_changes,
        "pv": pv,
        "events": events,
        **extensions,
        **{name + "_status": "computed" if block is not None else "not_requested" for name, block in extensions.items()},
        "operational_blocking_status": "computed" if blocking_requested else "not_requested",
        "operational_blocking": next((stream for stream in events["streams"] if stream["condition"] == _BLOCKING), None),
        "spatial_status": spatial_status,
        "spatial_covariance": spatial,
        "official": {
            "metrics": copy.deepcopy(result["metrics"]),
            "control": copy.deepcopy(result["control"]),
        },
    }
    try:
        encoded = schema.canonical_json(analysis)
    except ValueError as exc:
        raise ValueError(f"analysis is not finite canonical JSON: {exc}") from exc
    if len(encoded) > MAX_ANALYSIS_BYTES:
        raise ValueError(
            f"analysis exceeds the {MAX_ANALYSIS_BYTES} byte derivative limit"
        )
    return analysis


# ---------------------------------------------------------------------------
# Immutable publication
# ---------------------------------------------------------------------------


def _marker_path(input_path) -> Path:
    try:
        supplied = Path(os.fspath(input_path))
    except TypeError as exc:
        raise ValueError("episode input is not a filesystem path") from exc
    if os.path.islink(supplied):
        raise ValueError("episode input path must not be a symlink")
    if supplied.is_dir():
        marker = supplied / storage.MARKER_NAME
    elif supplied.name == storage.MARKER_NAME:
        marker = supplied
    else:
        raise ValueError("episode input must be an attempt directory or completion.json")
    if os.path.islink(marker):
        raise ValueError("episode completion marker must not be a symlink")
    if not marker.is_file():
        raise ValueError("episode completion marker is missing")
    return Path(os.path.abspath(marker))


def _ensure_directory(path: Path) -> None:
    try:
        path.mkdir()
    except FileExistsError:
        pass
    if os.path.islink(path) or not path.is_dir():
        raise ValueError(f"unsafe or non-directory publication path: {path.name}")


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


def _install_bytes(target: Path, encoded: bytes) -> None:
    """Publish exact bytes without ever replacing an existing file."""

    if target.exists():
        if os.path.islink(target) or not target.is_file():
            raise ValueError("existing analysis path is not a regular file")
        if target.read_bytes() != encoded:
            raise ValueError(
                "analysis already exists with different content; immutable "
                "derivative publication never overwrites"
            )
        return
    descriptor, temporary = tempfile.mkstemp(
        dir=os.fspath(target.parent), prefix=".analysis-", suffix=".part"
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, os.fspath(target))
        except FileExistsError as exc:
            raise ValueError(
                "analysis appeared concurrently; refusing to overwrite"
            ) from exc
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    _fsync_directory(target.parent)


def write_analysis(input_path, profile, output_root, *, require_scientific: bool = False) -> Path:
    """Analyze one published episode and publish an immutable derivative.

    The episode is read through :func:`physica_mc.storage.read_episode`, so the
    input is independently validated before any derivative is written.  An
    explicit ``spatial`` mapping needs mechanism maps, therefore
    ``required_profile='mechanism'`` and a full12 episode is rejected rather
    than fabricated from time-accumulated histograms.  An explicit
    ``spatial: null`` profile declines that optional tier and reads with
    ``required_profile='full12'``.  ``require_scientific=True`` is the explicit
    refusal point for using the technical fixture as scientific-campaign
    analysis.
    """

    normalized_profile = load_profile(
        profile, require_scientific=require_scientific
    )
    require_maps = normalized_profile["spatial"] is not None
    required_reader_profile = "mechanism" if require_maps else "full12"
    marker = _marker_path(input_path)
    sidecar_path = marker.parent / storage.SIDECAR_NAME
    if os.path.islink(sidecar_path):
        raise ValueError("episode arrays sidecar must not be a symlink")
    if not sidecar_path.is_file():
        raise ValueError("episode arrays sidecar is missing")
    marker_before = schema.file_sha256(marker)
    arrays_before = schema.file_sha256(sidecar_path)
    payload, arrays = storage.read_episode(
        marker, required_profile=required_reader_profile
    )
    marker_after = schema.file_sha256(marker)
    arrays_after = schema.file_sha256(sidecar_path)
    if marker_before != marker_after:
        raise ValueError("episode input changed while it was being analyzed")
    if arrays_before != arrays_after:
        raise ValueError("episode arrays changed while they were being analyzed")

    profile_hash = schema.digest(normalized_profile)
    analysis = analyze_episode(payload, arrays, normalized_profile)
    if analysis["profile_sha256"] != profile_hash:
        raise ValueError("analysis profile identity changed during analysis")

    sidecar = payload.get("sidecar")
    if not isinstance(sidecar, dict):
        raise ValueError("completed episode payload has no sidecar metadata")
    arrays_sha = sidecar.get("sha256")
    if not isinstance(arrays_sha, str) or not _HEX64.match(arrays_sha):
        raise ValueError("episode sidecar has no valid arrays SHA256")
    arrays_size = sidecar.get("size")
    if not _is_int(arrays_size) or int(arrays_size) < 0:
        raise ValueError("episode sidecar has no valid arrays size")
    attempt_id = payload.get("attempt_id")
    if attempt_id is None:
        raise ValueError("completed episode payload has no attempt_id")
    attempt_id = _safe_name(attempt_id, "attempt_id")
    task_id = analysis["episode"]["task_id"]
    if not isinstance(task_id, str) or not _HEX64.match(task_id):
        raise ValueError("episode task_id is not a SHA256 digest")

    source = schema.source_identity()
    analysis_id = schema.digest(
        {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "analysis_kind": ANALYSIS_KIND,
            "marker_sha256": marker_before,
            "arrays_sha256": arrays_sha,
            "profile_sha256": profile_hash,
            "required_reader_profile": required_reader_profile,
            "code_source_digest": source["source_digest"],
        }
    )
    analysis["analysis_id"] = analysis_id
    analysis["lineage"] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_kind": ANALYSIS_KIND,
        "analysis_id": analysis_id,
        "input": {
            "marker_sha256": marker_before,
            "arrays_sha256": arrays_sha,
            "arrays_size": int(arrays_size),
            "task_id": task_id,
            "attempt_id": attempt_id,
            "execution_id": analysis["episode"]["execution_id"],
            "observer_profile": analysis["episode"]["observer_profile"],
            "required_reader_profile": required_reader_profile,
        },
        "profile": {
            "profile_id": normalized_profile["profile_id"],
            "profile_sha256": profile_hash,
            "purpose": normalized_profile["purpose"],
            "technical_only": normalized_profile["technical_only"],
        },
        "code": {
            "source_digest": source["source_digest"],
            "packages": copy.deepcopy(source["packages"]),
        },
    }
    encoded = schema.canonical_json(analysis)
    if len(encoded) > MAX_ANALYSIS_BYTES:
        raise ValueError(
            f"analysis exceeds the {MAX_ANALYSIS_BYTES} byte derivative limit"
        )

    try:
        root = Path(os.path.abspath(os.fspath(output_root)))
    except TypeError as exc:
        raise ValueError("output_root is not a filesystem path") from exc
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("output_root must be a directory")
    _ensure_directory(root / task_id)
    _ensure_directory(root / task_id / attempt_id)
    analysis_root = root / task_id / attempt_id / "analysis"
    _ensure_directory(analysis_root)
    target_dir = analysis_root / analysis_id
    _ensure_directory(target_dir)
    target = target_dir / "analysis.json"
    _install_bytes(target, encoded)
    return target
