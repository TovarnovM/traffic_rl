"""One independent Physica A episode: warmup, role assignment, measurement.

The kernel is deliberately sequential and owns no runner, publisher or model
selection.  A production caller executes it in a fresh process and publishes
the returned ``(result, arrays)`` separately.  See ``docs/EPISODE_KERNEL.md``
for the task/execution boundary, control semantics, hashes and array columns.

The observer is passive: switching it off cannot change states, actions or RNG,
because nothing in the control/physics path reads observer output.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np

from .diagnostics import COLUMNS as DIAGNOSTIC_COLUMNS, observe_transition
from .controllers import (
    ACTION_INDEX_TO_DELTA,
    CommonCooldown,
    H_COOLDOWNS,
    OwnIntensityController,
    common_directions,
    left_if_safe_action,
    stay_action,
)
from .model_convert import load_bundle
from .models import infer_action
from .observer import WindowObserver
from snfs_traffic.control import LateralOverrideResult, _lateral_valid
from snfs_traffic.core import (
    SimulationParams,
    build_body_occupancy,
    build_lane_order,
    build_occupancy,
)
from snfs_traffic.metrics import PriorityMetricsAccumulator
from snfs_traffic.observations import PvGraphConfig, build_pv_graph_observation
from snfs_traffic.rules import PriorityYieldConfig, PriorityYieldController
from snfs_traffic.scenarios import (
    AV_VEH_TYPE,
    PriorityVehicleConfig,
    VehicleMix,
    assign_priority_vehicles,
    select_yielding_vehicle_ids,
)
from snfs_traffic.simulator import ScenarioConfig, TrafficSimulator


PROTOCOL = "physica-a-e2-native-eq9-v1"
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
FIXED_METHODS = frozenset(
    ("B1", "B2", "noop", "left_if_safe_original", "H1", "H2", "H4", "H8")
)
LEARNED_ARCHITECTURES = {"graph": "graph", "node": "node", "grid": "grid"}
HEURISTIC_C = {f"H{C}": C for C in H_COOLDOWNS}
SEED_ROLES = ("scenario", "warmup", "measurement", "pv_assignment", "av_selection")
OBSERVER_PROFILES = ("off", "full12", "mechanism")

STATE_FIELDS = (
    "vehicle_id",
    "lane",
    "pos",
    "vel",
    "length",
    "veh_type",
    "behavior_id",
    "alive",
    "last_lane_delta",
    "changed_lane",
    "controlled",
)

PHASES = (
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
)

# Fixed per-step rejection-reason columns; unexpected owning reasons go to
# "other" but the exact string is retained in the aggregate dictionary.
REJECTION_REASONS = (
    "out_of_bounds",
    "target_occupied",
    "unsafe",
    "override_conflict",
    "other",
)
_REJECTION_INDEX = {name: i for i, name in enumerate(REJECTION_REASONS)}

# All per-step control values are int64.  -1 means "not applicable for this
# control interface" (native B1 or rule B2); 0 is a real zero.
CONTROL_COLUMNS = (
    ("step", "int64", "measurement-relative pre-step index 0..T-1"),
    ("active_av", "int64", "active AV nodes in the common window; 0 for B1/B2"),
    (
        "selected_left",
        "int64",
        "controller action index 0 by active AV before common K1; -1 not applicable",
    ),
    (
        "selected_right",
        "int64",
        "controller action index 2 by active AV before common K1; -1 not applicable",
    ),
    (
        "selected_stay",
        "int64",
        "controller action index 1 by active AV before common K1; -1 not applicable",
    ),
    (
        "explicit_stay_commands",
        "int64",
        "active AVs sent delta=0 explicit stays, suppressing native lateral behaviour; -1 not applicable",
    ),
    (
        "available_left",
        "int64",
        "common-mask legal left opportunities; -1 not applicable",
    ),
    (
        "available_right",
        "int64",
        "common-mask legal right opportunities; -1 not applicable",
    ),
    (
        "eligible_nonstay",
        "int64",
        "union of common-mask left/right opportunities (private H_C not applied); -1 not applicable",
    ),
    (
        "common_throttled",
        "int64",
        "active AV decisions under the common K=1 cooldown before the step; -1 not applicable",
    ),
    (
        "private_throttled",
        "int64",
        "active AV decisions under an H_C private deadline; -1 not applicable",
    ),
    (
        "private_suppressed_left",
        "int64",
        "common-mask left opportunities forced to explicit stay by H_C; -1 not applicable",
    ),
    (
        "common_suppressed",
        "int64",
        "intended nonstay requests zeroed by common K=1 before simulation; -1 not applicable",
    ),
    (
        "requested_nonstay",
        "int64",
        "nonstay requests actually emitted to the simulator (rule requests for B2)",
    ),
    (
        "applied_nonstay",
        "int64",
        "emitted nonstay requests applied by the owning physics",
    ),
    (
        "rejected_nonstay",
        "int64",
        "emitted nonstay requests rejected by the owning physics",
    ),
    (
        "legal_nonstay_requests",
        "int64",
        "emitted nonstay requests permitted by the same common mask; -1 not applicable",
    ),
    (
        "applied_legal_nonstay_requests",
        "int64",
        "applied subset of legal nonstay requests; -1 not applicable",
    ),
    (
        "front_background",
        "int64",
        "pre-action background (HDV+AV) with signed distance > 0 (PV excluded)",
    ),
    (
        "centre_background",
        "int64",
        "pre-action background (HDV+AV) with signed distance == 0 (other lanes; PV excluded)",
    ),
    (
        "rear_background",
        "int64",
        "pre-action background (HDV+AV) with signed distance < 0 (PV excluded)",
    ),
    (
        "front_hdv",
        "int64",
        "pre-action HDV background with signed distance > 0",
    ),
    (
        "centre_hdv",
        "int64",
        "pre-action HDV background with signed distance == 0",
    ),
    (
        "rear_hdv",
        "int64",
        "pre-action HDV background with signed distance < 0",
    ),
    (
        "front_av",
        "int64",
        "pre-action AV background with signed distance > 0",
    ),
    (
        "centre_av",
        "int64",
        "pre-action AV background with signed distance == 0",
    ),
    (
        "rear_av",
        "int64",
        "pre-action AV background with signed distance < 0",
    ),
    (
        "requested_front",
        "int64",
        "emitted nonstay requests by AVs in the pre-action front zone",
    ),
    (
        "requested_centre",
        "int64",
        "emitted nonstay requests by AVs in the pre-action centre zone",
    ),
    (
        "requested_rear",
        "int64",
        "emitted nonstay requests by AVs in the pre-action rear zone",
    ),
    (
        "applied_front",
        "int64",
        "applied nonstay requests by AVs in the pre-action front zone",
    ),
    (
        "applied_centre",
        "int64",
        "applied nonstay requests by AVs in the pre-action centre zone",
    ),
    (
        "applied_rear",
        "int64",
        "applied nonstay requests by AVs in the pre-action rear zone",
    ),
)
_CONTROL_INDEX = {name: index for index, (name, _dtype, _text) in enumerate(CONTROL_COLUMNS)}

MECHANISM_COLUMNS = {
    "pre_pv_leader_gap": {
        "dtype": "int32",
        "description": "raw observed empty cells between the PV front and its nearest same-lane leader before the step; -1 when no leader exists (not a fabricated zero gap)",
    },
    "pre_pv_leader_type": {
        "dtype": "int8",
        "description": "veh_type of that observed leader (0 HDV, 1 AV, 2 BUS, 3 PV); -1 when none",
    },
    "pre_pv_leader_speed": {
        "dtype": "int8",
        "description": "raw observed speed of that leader before the step; -1 when none",
    },
    "pre_pv_leader_stopped": {
        "dtype": "int8",
        "description": "1 when the observed leader speed is 0, otherwise 0; -1 when no leader; this is not a causal blocking claim",
    },
    "pre_pv_native_left": {
        "dtype": "int8",
        "description": "owning _lateral_valid left safety/accessibility gate before the step: 1 gate passes, 0 gate does not pass, -1 lane_missing/not applicable. NOT the full native proposal: incentive, stochastic p_lane_change/direction draw and conflict outcome are not evaluated",
    },
    "pre_pv_native_right": {
        "dtype": "int8",
        "description": "owning _lateral_valid right safety/accessibility gate before the step: 1 gate passes, 0 gate does not pass, -1 lane_missing/not applicable. NOT the full native proposal: incentive, stochastic p_lane_change/direction draw and conflict outcome are not evaluated",
    },
    "pre_pv_native_left_reason": {
        "dtype": "int8",
        "description": "0 safety/accessibility gate passes, 1 target_occupied, 2 unsafe, 3 lane_missing; describes only that gate, never native incentive/desire",
    },
    "pre_pv_native_right_reason": {
        "dtype": "int8",
        "description": "0 safety/accessibility gate passes, 1 target_occupied, 2 unsafe, 3 lane_missing; describes only that gate, never native incentive/desire",
    },
    "measurement_step": {
        "dtype": "int64",
        "description": "measurement-relative physical step 0..T-1; t=0 is pre-measurement context, post-transition steps are 1..T",
    },
}
MECHANISM_COLUMNS.update(DIAGNOSTIC_COLUMNS)
# Internal test hook only; never consulted by control, physics, or the observer.
_DIAGNOSTICS_ENABLED = True

MECHANISM_NOTES = {
    "pre_pv_leader": "raw pre-step nearest same-lane leader gap/type/speed/stopped; gap=-1, type=-1, speed=-1 and stopped=-1 mean no leader, and no causal/blocking conclusion is encoded",
    "pre_pv_native_left_right": "only the owning _lateral_valid target-body-free and rear-gap-safe gate is recorded (1 pass, 0 fail, -1 lane_missing); native proposal also needs the incentive condition, a stochastic p_lane_change draw, a random direction choice when both sides are safe, and conflict resolution",
    "native_phase": "Both native collectors read original pre-step geometry after explicit acceptance/reservation, before any lanes change. Native-phase eligibility equals pre-action eligibility for the PV; excludes stochastic p/direction and reservation/conflict filtering.",
    "longitudinal": "All final lanes with pre-step positions/velocities reconstruct the exact post-lateral/pre-longitudinal phase. Previous/current cumulative gaps are candidate bounds for s=1 and s=S, not observed s/q/brake draws. Eq9 tightness is not proof of causal speed reduction.",
    "unavailable": [
        "native_stochastic_lane_change_draw",
        "native_direction_tie_break",
        "native_conflict_resolution_outcome",
        "longitudinal_s_q_brake_draws",
    ],
}
_MECHANISM_DTYPES = {
    name: np.dtype(spec["dtype"]) for name, spec in MECHANISM_COLUMNS.items()
}


class EpisodeIncomplete(RuntimeError):
    """A non-complete episode attempt; ``partial`` never claims completion."""

    def __init__(self, message: str, *, kind: str = "incomplete") -> None:
        super().__init__(message)
        self.kind = kind
        self.partial: dict = {}


def _schema():
    # Parent-owned schema module; imported lazily so controller/unit imports do
    # not depend on it and an unfinished integration fails at the boundary.
    return importlib.import_module("physica_mc.schema")


def _require_mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_int(value, name, *, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _require_float(value, name, *, low=None, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if low is not None and value < low:
        raise ValueError(f"{name} must be >= {low}")
    if high is not None and value > high:
        raise ValueError(f"{name} must be <= {high}")
    return value


def _require_seed(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer seed")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _state_sha256(state) -> str:
    digest = hashlib.sha256()
    for name in STATE_FIELDS:
        array = np.asarray(getattr(state, name))
        if array.ndim != 1:
            raise ValueError(f"state {name} must be one-dimensional")
        digest.update(name.encode("ascii"))
        digest.update(b":")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b":")
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(b":")
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _rng_sha256(rng) -> str:
    return hashlib.sha256(_canonical_json(rng.bit_generator.state)).hexdigest()


def _update_array(digest, tag: str, value) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(tag.encode("ascii"))
    digest.update(b":")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b":")
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(b":")
    digest.update(array.tobytes(order="C"))


def _step_trace_sha256(t, state, rng, action, requests, result) -> str:
    """Observer-neutral digest of one transition: actions, state, RNG, outcome."""

    digest = hashlib.sha256()
    digest.update(b"t:")
    digest.update(str(int(t)).encode("ascii"))
    for tag, value in (
        ("state", _state_sha256(state)),
        ("rng", _rng_sha256(rng)),
    ):
        digest.update(tag.encode("ascii"))
        digest.update(value.encode("ascii"))
    if action is not None:
        _update_array(digest, "action", action)
    digest.update(
        b"requests:"
        + _canonical_json([[int(k), int(v)] for k, v in sorted(requests.items())])
    )
    if result is not None:
        _update_array(digest, "vehicle_id", result.vehicle_id)
        _update_array(digest, "requested_delta", result.requested_lane_delta)
        _update_array(digest, "applied", result.applied)
        _update_array(digest, "valid", result.valid)
        digest.update(
            b"reasons:"
            + _canonical_json(list(result.rejection_reason))
        )
    return digest.hexdigest()


def _parse_deadline(deadline_utc):
    if deadline_utc is None:
        return None
    if isinstance(deadline_utc, dt.datetime):
        value = deadline_utc
    elif isinstance(deadline_utc, str):
        try:
            value = dt.datetime.fromisoformat(deadline_utc)
        except ValueError as exc:
            raise ValueError("deadline_utc must be an ISO8601 datetime") from exc
    else:
        raise TypeError("deadline_utc must be None, a timezone-aware datetime or ISO8601 text")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("deadline_utc must be timezone-aware")
    stamp = value.timestamp()
    if not math.isfinite(stamp):
        raise ValueError("deadline_utc is not finite")
    return stamp


def _check_deadline(deadline) -> None:
    if deadline is not None and time.time() >= deadline:
        raise EpisodeIncomplete(
            "absolute deadline reached during the episode", kind="deadline"
        )


def _make_params(config) -> SimulationParams:
    return SimulationParams(
        num_lanes=_require_int(config["num_lanes"], "num_lanes", minimum=1),
        road_length=_require_int(config["road_length"], "road_length", minimum=1),
        vmax_default=_require_int(config["vmax_hdv"], "vmax_hdv", minimum=0),
        vmax_controlled=_require_int(config["vmax_priority"], "vmax_priority", minimum=0),
        G=_require_int(config["G"], "G", minimum=0),
        q=_require_float(config["q"], "q", low=0.0, high=1.0),
        r=_require_float(config["r"], "r", low=0.0, high=1.0),
        S=_require_int(config["S"], "S", minimum=1),
        P1=_require_float(config["P1"], "P1", low=0.0, high=1.0),
        P2=_require_float(config["P2"], "P2", low=0.0, high=1.0),
        P3=_require_float(config["P3"], "P3", low=0.0, high=1.0),
        P4=_require_float(config["P4"], "P4", low=0.0, high=1.0),
        p_lane_change=_require_float(config["p_lane_change"], "p_lane_change", low=0.0, high=1.0),
    )


def _optimized_backend_available() -> bool:
    from snfs_traffic.backends import optimized_backend_available

    return bool(optimized_backend_available())


def _require_optimized_available(config) -> None:
    if config["backend"] != "optimized":
        return
    if not _optimized_backend_available():
        raise RuntimeError(
            "optimized backend requested but Numba kernels are unavailable; "
            "reference fallback is forbidden"
        )


def _require_optimized(sim, state) -> None:
    from snfs_traffic.core.indexing_numba import NUMBA_AVAILABLE as index_available
    from snfs_traffic.core.lane_change_numba import NUMBA_AVAILABLE as lane_available
    from snfs_traffic.core.longitudinal_numba import NUMBA_AVAILABLE as long_available

    if (
        sim.backend_name != "optimized"
        or not all((index_available, lane_available, long_available))
        or np.any(state.length[state.alive] != 1)
    ):
        raise RuntimeError(
            "optimized path prerequisites not met; reference fallback forbidden"
        )
    info = sim.last_step_info
    if info is not None and (
        info.used_reference_action_path or "reference" in info.backend_name
    ):
        raise RuntimeError("unexpected reference fallback")


def _validate_config(config, method, purpose) -> None:
    _require_mapping(config, "config")
    backend = config.get("backend")
    if backend not in ("optimized", "reference"):
        raise ValueError("backend must be 'optimized' or 'reference'")
    if not isinstance(config.get("validate"), bool):
        raise ValueError("config.validate must be boolean")
    density = _require_float(config.get("density"), "density", low=0.0, high=1.0)
    if density <= 0.0:
        raise ValueError("density must be > 0 for a closed-ring episode")
    _require_float(config.get("av_fraction"), "av_fraction", low=0.0, high=1.0)
    warmup = _require_int(config.get("warmup_steps"), "warmup_steps", minimum=0)
    measure = _require_int(config.get("measure_steps"), "measure_steps", minimum=1)
    for name in (
        "road_length",
        "num_lanes",
        "front_distance",
        "back_distance",
        "sensor_distance",
        "yield_cooldown_steps",
    ):
        _require_int(config.get(name), name, minimum=0)
    _require_int(config.get("yield_distance"), "yield_distance", minimum=1)
    _require_int(config.get("num_lanes"), "num_lanes", minimum=1)
    _require_int(config.get("road_length"), "road_length", minimum=1)
    _require_int(config.get("sensor_distance"), "sensor_distance", minimum=1)
    cooldown = _require_int(config.get("cooldown_steps"), "cooldown_steps", minimum=0)
    if cooldown != 1:
        raise ValueError("protocol v1 requires the common cooldown K=1")
    if purpose == "scientific":
        if backend != "optimized":
            raise ValueError("scientific purpose requires the optimized backend")
        if warmup != 1000 or measure != 2000:
            raise ValueError(
                "scientific purpose requires warmup_steps=1000 and measure_steps=2000"
            )


def _validate_science(task, schema):
    task = _require_mapping(task, "task")
    if task.get("schema_version") != 1:
        raise ValueError("unsupported task schema_version")
    science = _require_mapping(task.get("science"), "task.science")
    if science.get("protocol") != PROTOCOL:
        raise ValueError(f"unsupported protocol {science.get('protocol')!r}")
    purpose = science.get("purpose")
    if purpose not in ("technical", "scientific"):
        raise ValueError("purpose must be 'technical' or 'scientific'")
    if science.get("split") not in ("pilot", "validation", "fresh_test"):
        raise ValueError("unsupported split")
    method = science.get("method")
    if method not in METHODS:
        raise ValueError(f"unsupported method {method!r}")
    density_index = _require_int(science.get("density_index"), "density_index", minimum=0)
    replicate = _require_int(science.get("replicate"), "replicate", minimum=0)
    config = _require_mapping(science.get("config"), "task.science.config")
    seeds = _require_mapping(science.get("seeds"), "task.science.seeds")
    if tuple(sorted(seeds)) != tuple(sorted(SEED_ROLES)):
        raise ValueError("seeds must contain exactly the five protocol roles")
    seeds = {role: _require_seed(seeds[role], f"seeds.{role}") for role in SEED_ROLES}
    _validate_config(config, method, purpose)
    model_fields = {
        name: science.get(name)
        for name in ("model_id", "training_line", "candidate_id", "env_steps")
    }
    if method in LEARNED_ARCHITECTURES:
        if not isinstance(model_fields["model_id"], str) or not model_fields["model_id"]:
            raise ValueError("learned method requires a nonempty model_id")
        if not isinstance(model_fields["training_line"], str) or not model_fields["training_line"]:
            raise ValueError("learned method requires training_line")
        if not isinstance(model_fields["candidate_id"], str) or not model_fields["candidate_id"]:
            raise ValueError("learned method requires candidate_id")
        _require_int(model_fields["env_steps"], "env_steps", minimum=1)
    else:
        if any(value is not None for value in model_fields.values()):
            raise ValueError("fixed controllers require null model provenance fields")
    return {
        "task": task,
        "science": science,
        "purpose": purpose,
        "split": science["split"],
        "method": method,
        "density_index": density_index,
        "replicate": replicate,
        "config": config,
        "seeds": seeds,
        "model_fields": model_fields,
    }


def _validate_execution(execution, task, schema):
    execution = _require_mapping(execution, "execution")
    for name in ("execution_id", "task_id", "implementation", "profile"):
        if name not in execution:
            raise ValueError(f"execution is missing {name}")
    if execution["task_id"] != task.get("task_id"):
        raise ValueError("execution.task_id does not match the task")
    if not isinstance(execution["execution_id"], str) or not execution["execution_id"]:
        raise ValueError("execution_id must be a nonempty string")
    profile = _require_mapping(execution["profile"], "execution.profile")
    actual = schema.source_identity()
    implementation = execution["implementation"]
    if implementation != actual:
        # Accept canonical-equivalent transport (digest form or key order).
        try:
            digest_matches = implementation == schema.digest(actual)
        except Exception:
            digest_matches = False
        if not digest_matches:
            try:
                digest_matches = schema.digest(implementation) == schema.digest(actual)
            except Exception:
                digest_matches = False
        if not digest_matches:
            raise RuntimeError(
                "execution implementation identity does not match the actual source identity"
            )
    observer = profile.get("observer", "off")
    if observer is None:
        observer = "off"
    if observer not in OBSERVER_PROFILES:
        raise ValueError(f"unsupported observer profile {observer!r}")
    device = profile.get("device", "cpu")
    if not isinstance(device, str) or not device:
        raise ValueError("execution profile device must be a nonempty string")
    threads = _require_int(profile.get("threads", 1), "execution profile threads", minimum=1)
    bundle_path = execution.get("model_bundle")
    profile_bundle = profile.get("model_bundle")
    if bundle_path is not None and profile_bundle is not None and bundle_path != profile_bundle:
        raise ValueError("conflicting model_bundle transport values")
    if bundle_path is None:
        bundle_path = profile_bundle
    if bundle_path is not None and (
        not isinstance(bundle_path, str) or not bundle_path
    ):
        raise ValueError("model_bundle must be a nonempty path string")
    return {
        "execution": execution,
        "observer": observer,
        "device": device,
        "threads": threads,
        "model_bundle": bundle_path,
    }


def _pv_leader(state, pv_index: int, road_length: int):
    lane = int(state.lane[pv_index])
    pos = int(state.pos[pv_index])
    pv_id = int(state.vehicle_id[pv_index])
    same_lane = state.alive & (state.lane == lane) & (state.vehicle_id != pv_id)
    if not np.any(same_lane):
        return -1, -1, -1, -1
    forward = (state.pos[same_lane] - pos) % road_length
    local = int(np.argmin(forward))
    distance = int(forward[local])
    gap = distance - int(state.length[pv_index])
    if gap < 0:
        raise RuntimeError("overlapping PV/leader bodies in the owning physics")
    indices = np.flatnonzero(same_lane)
    leader = int(indices[local])
    speed = int(state.vel[leader])
    return gap, int(state.veh_type[leader]), speed, int(speed == 0)


def _pv_lateral_safety_gate(state, params, pv_index: int):
    """Owning `_lateral_valid` target-body-free and rear-gap-safe gate.

    This is only an accessibility/safety gate.  The native proposal/desire is
    stricter (incentive condition) and stochastic (p_lane_change, direction
    choice, conflict resolution); those are intentionally not evaluated here.
    """

    occupancy = build_occupancy(state, params)
    body_occupancy = build_body_occupancy(state, params)
    lane_order, lane_counts, _rank = build_lane_order(
        occupancy, n_vehicles=state.n_vehicles
    )
    lane = int(state.lane[pv_index])
    result = {}
    for name, delta in (("left", -1), ("right", 1)):
        target = lane + delta
        if not 0 <= target < params.num_lanes:
            result[name] = (-1, 3)
            continue
        ok, reason = _lateral_valid(
            state,
            lane_order,
            lane_counts,
            occupancy,
            body_occupancy,
            pv_index,
            target,
            params.road_length,
        )
        if ok:
            result[name] = (True, 0)
        elif reason == "target_occupied":
            result[name] = (False, 1)
        else:
            result[name] = (False, 2)
    return result


def _signed_distances(state, pv_index: int, road_length: int) -> np.ndarray:
    distances = (
        state.pos - int(state.pos[pv_index]) + road_length // 2
    ) % road_length - road_length // 2
    return distances.astype(np.int64, copy=False)


def _zone_index(distance: int) -> int:
    if distance > 0:
        return 0  # front
    if distance == 0:
        return 1  # centre
    return 2  # rear


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("result contains a non-finite float")
    return value


def _configure_threads(threads: int) -> None:
    import torch

    torch.set_num_threads(int(threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # An already initialized inter-op pool is configured by the caller.
        pass


def _load_model(bundle_path: str, science, config, device: str, threads: int):
    bundle = load_bundle(Path(bundle_path), device=device)
    architecture = LEARNED_ARCHITECTURES[science["method"]]
    if bundle.architecture != architecture:
        raise ValueError(
            f"bundle architecture {bundle.architecture!r} does not match method {science['method']!r}"
        )
    model_id = science["model_fields"]["model_id"]
    if bundle.identity != model_id:
        raise ValueError(
            "task model_id does not match the actual numeric bundle identity"
        )
    expected = {
        # Observation geometry, not training density/road length/horizon:
        # explicit transfer L=2000 must reuse the same model input geometry.
        "num_lanes": int(config["num_lanes"]),
        "front_distance": int(config["front_distance"]),
        "back_distance": int(config["back_distance"]),
        "sensor_distance": int(config["sensor_distance"]),
        "cooldown_steps": int(config["cooldown_steps"]),
        "max_nodes": int(config["num_lanes"])
        * (int(config["front_distance"]) + int(config["back_distance"]) + 1),
    }
    for key, value in expected.items():
        if bundle.config.get(key) != value:
            raise ValueError(
                f"bundle observation geometry {key}={bundle.config.get(key)!r} "
                f"does not match episode {value!r}"
            )
    _configure_threads(threads)
    return bundle


def _device_report(device: str, bundle) -> dict:
    report = {"requested": device, "torch_imported": bundle is not None}
    if bundle is None:
        report["type"] = "cpu" if device == "cpu" else device.split(":", 1)[0]
        return report
    import torch

    parameters = list(bundle.model.parameters())
    actual = str(parameters[0].device) if parameters else "cpu"
    report.update(
        type=actual.split(":", 1)[0],
        actual=actual,
        torch_version=torch.__version__,
        cuda_available=bool(torch.cuda.is_available()),
    )
    if actual.startswith("cuda"):
        index = torch.device(actual).index or 0
        report["cuda_device_name"] = torch.cuda.get_device_name(index)
        report["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(index))
    return report


def _control_summary(method, config, counters, reason_counts, steps) -> dict:
    interface = "native" if method == "B1" else "rule" if method == "B2" else "av_action"
    applicable = interface == "av_action"
    private_applicable = method in HEURISTIC_C
    eligible = counters["eligible_nonstay_decisions"] if applicable else None
    legal = counters["legal_nonstay_requests"] if applicable else None
    applied_legal = counters["applied_legal_nonstay_requests"] if applicable else None
    if not applicable:
        request_use = None
        applied_use = None
        use_status = "not_applicable"
    elif eligible > 0:
        request_use = legal / eligible
        applied_use = applied_legal / eligible
        use_status = "defined"
    else:
        request_use = None
        applied_use = None
        use_status = "undefined_zero_eligible"
    requested = counters["requested_nonstay"]
    applied = counters["applied_nonstay"]
    rejected = counters["rejected_nonstay"]
    if requested != applied + rejected:
        raise RuntimeError("emitted_nonstay must equal applied_nonstay + rejected_nonstay")
    return {
        "interface": interface,
        "steps": int(steps),
        "active_av_decisions": int(counters["active_av_decisions"]),
        "selected_left": counters["selected_left"] if applicable else None,
        "selected_right": counters["selected_right"] if applicable else None,
        "selected_stay": counters["selected_stay"] if applicable else None,
        "explicit_stay_commands": counters["explicit_stay_commands"] if applicable else None,
        "available_left_decisions": counters["available_left_decisions"] if applicable else None,
        "available_right_decisions": counters["available_right_decisions"] if applicable else None,
        "eligible_nonstay_decisions": eligible,
        "common_throttled_active_decisions": counters["common_throttled_active_decisions"] if applicable else None,
        "private_throttled_active_decisions": counters["private_throttled_active_decisions"] if (applicable and private_applicable) else None,
        "private_suppressed_left_opportunities": counters["private_suppressed_left_opportunities"] if (applicable and private_applicable) else None,
        "common_suppressed_nonstop_requests": counters["common_suppressed_nonstop_requests"] if applicable else None,
        "common_cooldown_steps": int(config["cooldown_steps"]),
        "private_cooldown_steps": HEURISTIC_C.get(method),
        "requested_nonstay": requested,
        "applied_nonstay": applied,
        "rejected_nonstay": rejected,
        "emitted_nonstay": requested,
        "legal_nonstay_requests": legal,
        "applied_legal_nonstay_requests": applied_legal,
        "request_opportunity_use": request_use,
        "applied_opportunity_use": applied_use,
        "opportunity_use_status": use_status,
        "applied_per_step": applied / steps,
        "applied_per_active_decision": (
            applied / counters["active_av_decisions"]
            if counters["active_av_decisions"]
            else None
        ),
        "applied_per_request": applied / requested if requested else None,
        "rejected_per_request": rejected / requested if requested else None,
        "rejection_reasons": {str(k): int(v) for k, v in sorted(reason_counts.items())},
        "rejection_reason_names": list(REJECTION_REASONS),
        "requested_front": int(counters["requested_front"]),
        "requested_centre": int(counters["requested_centre"]),
        "requested_rear": int(counters["requested_rear"]),
        "applied_front": int(counters["applied_front"]),
        "applied_centre": int(counters["applied_centre"]),
        "applied_rear": int(counters["applied_rear"]),
    }


def _control_columns() -> dict:
    return {
        name: {"index": index, "dtype": dtype, "description": text}
        for index, (name, dtype, text) in enumerate(CONTROL_COLUMNS)
    }


def _rejection_columns() -> dict:
    return {
        name: {
            "index": index,
            "description": "per-step count of rejected nonstay requests with this owning-physics reason",
        }
        for index, name in enumerate(REJECTION_REASONS)
    }


def _new_counters() -> dict:
    counters = {
        "active_av_decisions": 0,
        "selected_left": 0,
        "selected_right": 0,
        "selected_stay": 0,
        "explicit_stay_commands": 0,
        "available_left_decisions": 0,
        "available_right_decisions": 0,
        "eligible_nonstay_decisions": 0,
        "common_throttled_active_decisions": 0,
        "private_throttled_active_decisions": 0,
        "private_suppressed_left_opportunities": 0,
        "common_suppressed_nonstop_requests": 0,
        "legal_nonstay_requests": 0,
        "applied_legal_nonstay_requests": 0,
        "requested_nonstay": 0,
        "applied_nonstay": 0,
        "rejected_nonstay": 0,
        "requested_front": 0,
        "requested_centre": 0,
        "requested_rear": 0,
        "applied_front": 0,
        "applied_centre": 0,
        "applied_rear": 0,
    }
    return counters


def simulate_episode(task, *, execution=None, trace=False, deadline_utc=None):
    """Execute exactly one independent episode and return ``(result, arrays)``.

    ``result`` is JSON-compatible and never claims completion unless the full
    measurement horizon was observed.  ``arrays`` contains passive observer
    arrays (profile permitting) plus the documented control/mechanism arrays.
    """

    started = time.perf_counter()
    cpu_started = time.process_time()
    timings = {name: 0.0 for name in PHASES}
    schema = _schema()
    schema.validate_task(task)
    science = _validate_science(task, schema)
    if execution is None:
        execution = schema.make_execution(task)
    runtime = _validate_execution(execution, task, schema)
    deadline = _parse_deadline(deadline_utc)

    method = science["method"]
    config = science["config"]
    seeds = science["seeds"]
    purpose = science["purpose"]
    observer_profile = runtime["observer"]
    device = runtime["device"]
    threads = runtime["threads"]
    bundle_path = runtime["model_bundle"]
    if method in LEARNED_ARCHITECTURES and bundle_path is None:
        raise ValueError("learned method requires an explicit execution model_bundle path")
    if method in FIXED_METHODS and bundle_path is not None:
        raise ValueError("fixed controller must not receive a model_bundle")

    params = _make_params(config)
    road_length = int(config["road_length"])
    num_lanes = int(config["num_lanes"])
    measure_steps = int(config["measure_steps"])
    warmup_steps = int(config["warmup_steps"])

    counters = _new_counters()
    reason_counts: dict[str, int] = {}
    backend_paths: set[str] = set()
    warmup_completed = 0
    steps_completed = 0
    priority_id = None
    av_ids: tuple[int, ...] = ()
    warm_hash = pv_hash = av_hash = None
    initial_hash = final_hash = None
    initial_rng = final_rng = None
    metrics = None
    rolling = hashlib.sha256() if trace else None
    timings["setup"] = time.perf_counter() - started

    def partial(kind):
        summary = None
        if metrics is not None and steps_completed > 0:
            try:
                summary = metrics.summary().to_dict()
            except Exception:  # diagnostics must not replace the original failure
                summary = None
        return {
            "status": "partial" if (warmup_completed or steps_completed) else "failed",
            "kind": str(kind),
            "task_id": task.get("task_id"),
            "method": method,
            "measurement_steps_planned": measure_steps,
            "warmup_steps_completed": int(warmup_completed),
            "steps_completed": int(steps_completed),
            "priority_vehicle_id": priority_id,
            "initial_av_ids": [int(value) for value in av_ids],
            "observer_enabled": observer_profile != "off",
            "trace_enabled": bool(trace),
            "trace_sha256": rolling.hexdigest() if rolling is not None else None,
            "warm_state_sha256": warm_hash,
            "pv_assignment_state_sha256": pv_hash,
            "av_assignment_state_sha256": av_hash,
            "initial_state_sha256": initial_hash,
            "final_state_sha256": final_hash,
            "initial_rng_sha256": initial_rng,
            "final_rng_sha256": final_rng,
            "metrics": summary,
            "control_counters": dict(counters),
            "timings": {key: float(value) for key, value in timings.items()},
        }

    try:
        _check_deadline(deadline)
        if config["backend"] == "optimized":
            _require_optimized_available(config)

        # --- model loading (validated inputs before any expensive warmup) -----
        bundle = None
        if method in LEARNED_ARCHITECTURES:
            begin = time.perf_counter()
            bundle = _load_model(bundle_path, science, config, device, threads)
            timings["model_load"] += time.perf_counter() - begin

        # --- warmup: the unchanged scenario/warmup protocol, native physics ----
        begin = time.perf_counter()
        warm_sim = TrafficSimulator(
            params=params,
            backend=config["backend"],
            rng_seed=seeds["warmup"],
            scenario=ScenarioConfig(
                density=float(config["density"]),
                seed=seeds["scenario"],
                vehicle_mix=VehicleMix(),
            ),
            validate=bool(config["validate"]),
        )
        warm_state = warm_sim.reset(
            scenario_seed=seeds["scenario"], rng_seed=seeds["warmup"]
        )
        if config["backend"] == "optimized":
            _require_optimized(warm_sim, warm_state)
        for _ in range(warmup_steps):
            _check_deadline(deadline)
            warm_state = warm_sim.step()
            if config["backend"] == "optimized":
                _require_optimized(warm_sim, warm_state)
            warmup_completed += 1
        warm_hash = _state_sha256(warm_state)
        timings["warmup"] += time.perf_counter() - begin

        # --- PV then AV role assignment; B1 never receives AV tags ------------
        begin = time.perf_counter()
        assignment = assign_priority_vehicles(
            warm_state,
            PriorityVehicleConfig(
                count=1, placement="random", seed=seeds["pv_assignment"]
            ),
            road_length=road_length,
        )
        priority_id = int(assignment.vehicle_ids[0])
        pv_state = assignment.state
        pv_hash = _state_sha256(pv_state)
        timings["pv_assignment"] += time.perf_counter() - begin

        begin = time.perf_counter()
        av_assignment_applied = method != "B1"
        if av_assignment_applied:
            av_state, av_ids = select_yielding_vehicle_ids(
                pv_state,
                fraction=float(config["av_fraction"]),
                seed=seeds["av_selection"],
                tag_as_av=True,
            )
            av_hash = _state_sha256(av_state)
            initial_state = av_state
        else:
            initial_state = pv_state
        timings["av_assignment"] += time.perf_counter() - begin

        # --- measurement reset: fresh simulator RNG and mutable state ----------
        sim = TrafficSimulator(
            params=params,
            backend=config["backend"],
            rng_seed=seeds["measurement"],
            validate=bool(config["validate"]),
        )
        state = sim.reset(state=initial_state, rng_seed=seeds["measurement"])
        if config["backend"] == "optimized":
            _require_optimized(sim, state)
        initial_hash = _state_sha256(state)
        initial_rng = _rng_sha256(sim._rng)
        initial_alive = state.alive.copy()
        initial_live_count = int(initial_alive.sum())
        if initial_live_count < 1:
            raise RuntimeError("measurement state has no live vehicles")
        id_to_index = {
            int(vehicle_id): index
            for index, vehicle_id in enumerate(state.vehicle_id)
        }
        pv_index_values = np.flatnonzero(
            state.alive & (state.vehicle_id == priority_id)
        )
        if pv_index_values.size != 1:
            raise RuntimeError("priority vehicle must be alive exactly once")
        pv_index = int(pv_index_values[0])
        if av_assignment_applied and not set(int(value) for value in av_ids).issubset(
            id_to_index
        ):
            raise RuntimeError("selected AV ids are not present in the measurement state")

        graph_config = PvGraphConfig(
            front_distance=int(config["front_distance"]),
            back_distance=int(config["back_distance"]),
            sensor_distance=int(config["sensor_distance"]),
            cooldown_steps=int(config["cooldown_steps"]),
        )
        graph_config.validate_for(params)
        rule = None
        own = None
        if method == "B2":
            rule = PriorityYieldController(
                PriorityYieldConfig(
                    detection_distance=int(config["yield_distance"]),
                    cooldown_steps=int(config["yield_cooldown_steps"]),
                ),
                eligible_vehicle_ids=av_ids,
            )
        elif method in HEURISTIC_C:
            own = OwnIntensityController(HEURISTIC_C[method])
        common = CommonCooldown()

        observer = None
        if observer_profile != "off":
            observer = WindowObserver(
                state,
                pv_id=priority_id,
                av_ids=av_ids,
                road_length=road_length,
                num_lanes=num_lanes,
                horizon=measure_steps,
                profile=observer_profile,
            )

        metrics = PriorityMetricsAccumulator(
            params, priority_vehicle_ids=(priority_id,)
        )
        control_rows = np.zeros((measure_steps, len(CONTROL_COLUMNS)), dtype=np.int64)
        rejection_rows = np.zeros(
            (measure_steps, len(REJECTION_REASONS)), dtype=np.int64
        )
        mechanism = {
            name: np.zeros(measure_steps, dtype=dtype)
            for name, dtype in _MECHANISM_DTYPES.items()
        }
        mechanism["pre_pv_leader_gap"].fill(-1)
        mechanism["pre_pv_leader_type"].fill(-1)
        mechanism["pre_pv_leader_speed"].fill(-1)
        mechanism["measurement_step"][:] = np.arange(measure_steps, dtype=np.int64)
        measurement_started = time.perf_counter()

        for t in range(measure_steps):
            _check_deadline(deadline)

            # --- passive pre-step mechanism observations (no RNG, no mutation)
            begin = time.perf_counter()
            diagnostic_before = state.copy() if _DIAGNOSTICS_ENABLED else None
            leader_gap, leader_type, leader_speed, leader_stopped = _pv_leader(
                state, pv_index, road_length
            )
            safety_gate = _pv_lateral_safety_gate(state, params, pv_index)
            mechanism["pre_pv_leader_gap"][t] = leader_gap
            mechanism["pre_pv_leader_type"][t] = leader_type
            mechanism["pre_pv_leader_speed"][t] = leader_speed
            mechanism["pre_pv_leader_stopped"][t] = leader_stopped
            mechanism["pre_pv_native_left"][t] = int(safety_gate["left"][0])
            mechanism["pre_pv_native_right"][t] = int(safety_gate["right"][0])
            mechanism["pre_pv_native_left_reason"][t] = safety_gate["left"][1]
            mechanism["pre_pv_native_right_reason"][t] = safety_gate["right"][1]
            distances = _signed_distances(state, pv_index, road_length)
            alive_without_pv = initial_alive.copy()
            alive_without_pv[pv_index] = False
            av_mask = alive_without_pv & (state.veh_type == AV_VEH_TYPE)
            front_mask = alive_without_pv & (distances > 0)
            centre_mask = alive_without_pv & (distances == 0)
            rear_mask = alive_without_pv & (distances < 0)
            timings["mechanism"] += time.perf_counter() - begin

            graph = None
            action = None
            result = None
            decision = None
            requests: dict[int, int] = {}
            commands: dict[int, int] = {}
            active = 0
            selected_left = selected_right = selected_stay = 0
            explicit_stay = 0
            left_available = right_available = eligible = 0
            common_throttled = 0
            private_throttled = private_suppressed = 0
            common_suppressed = 0
            legal_requests = applied_legal = 0
            applicable = False
            private_applicable = method in HEURISTIC_C

            if method == "B1":
                begin = time.perf_counter()
                state = sim.step()
                timings["simulator"] += time.perf_counter() - begin
            elif method == "B2":
                begin = time.perf_counter()
                decision = rule.decide(state, params, sim.topology)
                overrides = dict(decision.overrides)
                timings["mask"] += time.perf_counter() - begin
                begin = time.perf_counter()
                if overrides:
                    state = sim.step_lateral_overrides(overrides)
                    result = sim.last_step_info.action_result
                    if not isinstance(result, LateralOverrideResult):
                        raise RuntimeError("override step returned no private lateral result")
                    rule.observe(result)
                else:
                    state = sim.step()
                timings["simulator"] += time.perf_counter() - begin
            else:
                applicable = True
                begin = time.perf_counter()
                graph = build_pv_graph_observation(
                    state,
                    params,
                    sim.topology,
                    priority_vehicle_id=priority_id,
                    config=graph_config,
                    cooldown_by_vehicle_id=common.remaining(),
                )
                timings["observation"] += time.perf_counter() - begin
                active = int(graph.active_count)
                mask_begin = time.perf_counter()
                if method == "noop":
                    action = stay_action(graph)
                elif method == "left_if_safe_original":
                    action = left_if_safe_action(graph)
                elif method in HEURISTIC_C:
                    action = own.action(graph, t)
                    private_throttled = own.last_throttled_decisions
                    private_suppressed = own.last_suppressed_left_opportunities
                else:
                    inference_begin = time.perf_counter()
                    action = infer_action(bundle.model, graph.as_dict())
                    timings["inference"] += time.perf_counter() - inference_begin
                    # Mask/suppression bookkeeping is timed after the forward pass.
                    mask_begin = time.perf_counter()
                if action.shape != graph.node_mask.shape or action.dtype != np.int64:
                    raise ValueError("controller action has an unexpected shape/dtype")
                left_mask, right_mask = common_directions(graph)
                left_available = int(left_mask.sum())
                right_available = int(right_mask.sum())
                eligible = int((left_mask | right_mask).sum())
                common_remaining = common.remaining()
                for slot in range(active):
                    vehicle_id = int(graph.vehicle_id[slot])
                    if common_remaining.get(vehicle_id, 0) > 0:
                        common_throttled += 1
                deltas = ACTION_INDEX_TO_DELTA[action]
                selected_left = int(np.sum(deltas[:active] == -1))
                selected_right = int(np.sum(deltas[:active] == 1))
                selected_stay = int(np.sum(deltas[:active] == 0))
                for slot in range(active):
                    vehicle_id = int(graph.vehicle_id[slot])
                    delta = int(deltas[slot])
                    requests[vehicle_id] = delta
                    if delta != 0:
                        legal = bool(
                            (delta == -1 and left_mask[slot])
                            or (delta == 1 and right_mask[slot])
                        )
                        if not legal:
                            raise RuntimeError(
                                "controller emitted a nonstay request outside the common action mask"
                            )
                        legal_requests += 1
                commands, common_suppressed = common.suppress(requests)
                explicit_stay = int(sum(delta == 0 for delta in commands.values()))
                timings["mask"] += time.perf_counter() - mask_begin
                begin = time.perf_counter()
                if commands:
                    state = sim.step_lateral_overrides(commands)
                    result = sim.last_step_info.action_result
                    if not isinstance(result, LateralOverrideResult):
                        raise RuntimeError("override step returned no private lateral result")
                else:
                    # Empty active set does not stop physical time or draw control RNG.
                    state = sim.step()
                timings["simulator"] += time.perf_counter() - begin
                if own is not None:
                    own.observe(result, t)
                common.update(result)

            if diagnostic_before is not None:
                begin = time.perf_counter()
                for name, value in observe_transition(diagnostic_before, state, params, pv_index).items():
                    mechanism[name][t] = value
                timings["mechanism"] += time.perf_counter() - begin

            if sim.last_step_info is not None:
                backend_paths.add(str(sim.last_step_info.backend_name))
            if config["backend"] == "optimized":
                _require_optimized(sim, state)
            if not np.array_equal(state.alive, initial_alive):
                raise RuntimeError("closed-ring population changed during measurement")
            if not bool(state.alive[pv_index]):
                raise RuntimeError("priority vehicle left the closed ring")

            requested = applied = rejected = 0
            requested_zones = [0, 0, 0]
            applied_zones = [0, 0, 0]
            if result is not None:
                nonstay = np.asarray(result.requested_lane_delta) != 0
                requested = int(nonstay.sum())
                applied_mask = nonstay & np.asarray(result.applied, dtype=bool)
                applied = int(applied_mask.sum())
                rejected = requested - applied
                for row in np.flatnonzero(nonstay):
                    vehicle_id = int(result.vehicle_id[int(row)])
                    if vehicle_id not in id_to_index:
                        raise RuntimeError("lateral result references an unknown physical ID")
                    zone = _zone_index(int(distances[id_to_index[vehicle_id]]))
                    requested_zones[zone] += 1
                    if bool(result.applied[int(row)]):
                        applied_zones[zone] += 1
                    else:
                        reason = str(result.rejection_reason[int(row)])
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                        rejection_rows[t, _REJECTION_INDEX.get(reason, _REJECTION_INDEX["other"])] += 1
                if applicable:
                    applied_legal = applied

            counters["active_av_decisions"] += active
            if applicable:
                counters["selected_left"] += selected_left
                counters["selected_right"] += selected_right
                counters["selected_stay"] += selected_stay
                counters["explicit_stay_commands"] += explicit_stay
                counters["available_left_decisions"] += left_available
                counters["available_right_decisions"] += right_available
                counters["eligible_nonstay_decisions"] += eligible
                counters["common_throttled_active_decisions"] += common_throttled
                counters["common_suppressed_nonstop_requests"] += common_suppressed
                counters["legal_nonstay_requests"] += legal_requests
                counters["applied_legal_nonstay_requests"] += applied_legal
                if private_applicable:
                    counters["private_throttled_active_decisions"] += private_throttled
                    counters["private_suppressed_left_opportunities"] += private_suppressed
            counters["requested_nonstay"] += requested
            counters["applied_nonstay"] += applied
            counters["rejected_nonstay"] += rejected
            counters["requested_front"] += requested_zones[0]
            counters["requested_centre"] += requested_zones[1]
            counters["requested_rear"] += requested_zones[2]
            counters["applied_front"] += applied_zones[0]
            counters["applied_centre"] += applied_zones[1]
            counters["applied_rear"] += applied_zones[2]

            row = control_rows[t]
            row[_CONTROL_INDEX["step"]] = t
            row[_CONTROL_INDEX["active_av"]] = active
            for name, value in (
                ("selected_left", selected_left),
                ("selected_right", selected_right),
                ("selected_stay", selected_stay),
                ("explicit_stay_commands", explicit_stay),
                ("available_left", left_available),
                ("available_right", right_available),
                ("eligible_nonstay", eligible),
                ("common_throttled", common_throttled),
                ("common_suppressed", common_suppressed),
                ("legal_nonstay_requests", legal_requests),
                ("applied_legal_nonstay_requests", applied_legal),
            ):
                row[_CONTROL_INDEX[name]] = value if applicable else -1
            row[_CONTROL_INDEX["private_throttled"]] = (
                private_throttled if private_applicable else -1
            )
            row[_CONTROL_INDEX["private_suppressed_left"]] = (
                private_suppressed if private_applicable else -1
            )
            row[_CONTROL_INDEX["requested_nonstay"]] = requested
            row[_CONTROL_INDEX["applied_nonstay"]] = applied
            row[_CONTROL_INDEX["rejected_nonstay"]] = rejected
            row[_CONTROL_INDEX["front_background"]] = int(np.sum(front_mask))
            row[_CONTROL_INDEX["centre_background"]] = int(np.sum(centre_mask))
            row[_CONTROL_INDEX["rear_background"]] = int(np.sum(rear_mask))
            row[_CONTROL_INDEX["front_hdv"]] = int(np.sum(front_mask & ~av_mask))
            row[_CONTROL_INDEX["centre_hdv"]] = int(np.sum(centre_mask & ~av_mask))
            row[_CONTROL_INDEX["rear_hdv"]] = int(np.sum(rear_mask & ~av_mask))
            row[_CONTROL_INDEX["front_av"]] = int(np.sum(front_mask & av_mask))
            row[_CONTROL_INDEX["centre_av"]] = int(np.sum(centre_mask & av_mask))
            row[_CONTROL_INDEX["rear_av"]] = int(np.sum(rear_mask & av_mask))
            row[_CONTROL_INDEX["requested_front"]] = requested_zones[0]
            row[_CONTROL_INDEX["requested_centre"]] = requested_zones[1]
            row[_CONTROL_INDEX["requested_rear"]] = requested_zones[2]
            row[_CONTROL_INDEX["applied_front"]] = applied_zones[0]
            row[_CONTROL_INDEX["applied_centre"]] = applied_zones[1]
            row[_CONTROL_INDEX["applied_rear"]] = applied_zones[2]

            begin = time.perf_counter()
            metrics.observe(
                state,
                decision=decision if method == "B2" else None,
                result=result if method == "B2" else None,
            )
            timings["metrics"] += time.perf_counter() - begin

            if rolling is not None:
                rolling.update(
                    bytes.fromhex(
                        _step_trace_sha256(
                            t,
                            state,
                            sim._rng,
                            action,
                            requests,
                            result,
                        )
                    )
                )
            steps_completed += 1

            if observer is not None:
                begin = time.perf_counter()
                observer.observe(state)
                timings["observer"] += time.perf_counter() - begin

        if steps_completed != measure_steps:
            raise EpisodeIncomplete("measurement horizon was not completed")
        _check_deadline(deadline)
        timings["measurement"] = time.perf_counter() - measurement_started

        observer_summary = None
        observer_arrays: dict[str, np.ndarray] = {}
        if observer is not None:
            begin = time.perf_counter()
            observer_summary, observer_arrays = observer.finish()
            timings["observer"] += time.perf_counter() - begin

        final_hash = _state_sha256(state)
        final_rng = _rng_sha256(sim._rng)
        summary = metrics.summary().to_dict()
        control = _control_summary(
            method, config, counters, reason_counts, measure_steps
        )

        kernel_arrays = {
            "control_trace": control_rows,
            "control_rejection": rejection_rows,
            **mechanism,
        }
        duplicate = set(kernel_arrays).intersection(observer_arrays)
        if duplicate:
            raise RuntimeError(f"kernel/observer array name collision: {sorted(duplicate)}")
        arrays = {**observer_arrays, **kernel_arrays}

        initial_live = int(initial_alive.sum())
        background_count = initial_live - 1
        realized_av_fraction = (
            float(len(av_ids)) / background_count if background_count > 0 else None
        )
        model_report = None
        model_fields = science["model_fields"]
        if bundle is not None:
            model_report = {
                "model_id": model_fields["model_id"],
                "training_line": model_fields["training_line"],
                "candidate_id": model_fields["candidate_id"],
                "env_steps": model_fields["env_steps"],
                "architecture": bundle.architecture,
                "bundle_identity": bundle.identity,
                "bundle_config_sha256": hashlib.sha256(
                    _canonical_json(bundle.config)
                ).hexdigest(),
            }
        timings["total"] = time.perf_counter() - started
        result = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "task_id": str(task["task_id"]),
            "execution_id": str(execution["execution_id"]),
            "purpose": purpose,
            "split": science["split"],
            "method": method,
            "architecture": LEARNED_ARCHITECTURES.get(method),
            "model_id": model_fields["model_id"],
            "training_line": model_fields["training_line"],
            "candidate_id": model_fields["candidate_id"],
            "env_steps": model_fields["env_steps"],
            "density_index": science["density_index"],
            "replicate": science["replicate"],
            "config": dict(config),
            "seeds": dict(seeds),
            "priority_id": int(priority_id),
            "initial_av_ids": [int(value) for value in av_ids],
            "av_assignment_applied": bool(av_assignment_applied),
            "requested_density": float(config["density"]),
            "requested_background_av_fraction": float(config["av_fraction"]),
            "realized_density": float(initial_live) / float(road_length * num_lanes),
            "realized_background_av_fraction": realized_av_fraction,
            "initial_vehicle_count": initial_live,
            "background_vehicle_count": background_count,
            "av_count": int(len(av_ids)),
            "background_av_count": int(len(av_ids)),
            "measurement_steps": measure_steps,
            "warmup_steps_completed": int(warmup_completed),
            "steps_completed": int(steps_completed),
            "warm_state_sha256": warm_hash,
            "pv_assignment_state_sha256": pv_hash,
            "av_assignment_state_sha256": av_hash,
            "initial_state_sha256": initial_hash,
            "final_state_sha256": final_hash,
            "initial_rng_sha256": initial_rng,
            "final_rng_sha256": final_rng,
            "trace_enabled": bool(trace),
            "trace_sha256": rolling.hexdigest() if rolling is not None else None,
            "model": model_report,
            "observer_profile": observer_profile,
            "observer_summary": observer_summary,
            "metrics": summary,
            "control": control,
            "control_columns": _control_columns(),
            "control_rejection_columns": _rejection_columns(),
            "mechanism_columns": _plain(MECHANISM_COLUMNS),
            "mechanism_notes": _plain(MECHANISM_NOTES),
            "backend_requested": config["backend"],
            "backend_paths": sorted(backend_paths),
            "device": _device_report(device, bundle),
            "timings": {key: float(value) for key, value in timings.items()},
            "resources": {
                "cpu_seconds": float(time.process_time() - cpu_started),
                "wall_seconds": float(time.perf_counter() - started),
                "process_peak_rss_kib": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ),
            },
            "deadline_utc": (
                dt.datetime.fromtimestamp(deadline, tz=dt.timezone.utc).isoformat()
                if deadline is not None
                else None
            ),
        }
        return _plain(result), arrays
    except EpisodeIncomplete as exc:
        exc.partial = partial(exc.kind)
        raise
    except Exception as exc:
        wrapped = EpisodeIncomplete(
            f"episode failed: {type(exc).__name__}: {exc}", kind="error"
        )
        wrapped.partial = partial("error")
        raise wrapped from exc
