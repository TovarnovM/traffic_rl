"""Technical fixtures for the one-episode kernel; no test claims a scientific run.

Tiny horizons and the reference backend are explicit technical fixtures only.
The parent-owned schema/storage modules are exercised through their declared
API; episode tests skip (not pass) when that integration module is absent.
"""

import datetime as dt
import json

import numpy as np
import pytest

from physica_mc import episode
from physica_mc.model_convert import load_bundle, save_bundle
from physica_mc.models import build_model

schema = pytest.importorskip("physica_mc.schema")


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
BASE_SEEDS = {
    "scenario": 101,
    "warmup": 202,
    "measurement": 303,
    "pv_assignment": 404,
    "av_selection": 505,
}
BUNDLE_CONFIG = {
    "road_length": 1000,
    "num_lanes": 4,
    "front_distance": 30,
    "back_distance": 10,
    "sensor_distance": 60,
    "cooldown_steps": 1,
}
FIXED_METHODS = (
    "B1",
    "B2",
    "noop",
    "left_if_safe_original",
    "H1",
    "H2",
    "H4",
    "H8",
)
HEURISTIC_METHODS = ("H1", "H2", "H4", "H8")


def config(**overrides):
    value = dict(DEFAULT_CONFIG)
    value.update(overrides)
    return value


def make_task(
    method,
    *,
    density=0.1,
    av_fraction=0.95,
    road_length=1000,
    num_lanes=4,
    warmup_steps=1,
    measure_steps=2,
    backend="reference",
    purpose="technical",
    split="pilot",
    seeds=None,
    recipe_seeds=False,
    model=None,
):
    cfg = config(
        density=density,
        av_fraction=av_fraction,
        road_length=road_length,
        num_lanes=num_lanes,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        backend=backend,
    )
    model_fields = {} if model is None else dict(model)
    kwargs = {}
    if not recipe_seeds:
        kwargs["seeds"] = dict(BASE_SEEDS if seeds is None else seeds)
    return schema.make_task(
        method,
        density=density,
        av_fraction=av_fraction,
        replicate=0,
        split=split,
        config=cfg,
        purpose=purpose,
        **kwargs,
        **model_fields,
    )


def run(task, *, observer="off", model_bundle=None, trace=False, deadline=None):
    kwargs = dict(device="cpu", threads=1, observer=observer)
    if model_bundle is not None:
        kwargs["model_bundle"] = model_bundle
    execution = schema.make_execution(task, **kwargs)
    return episode.simulate_episode(
        task, execution=execution, trace=trace, deadline_utc=deadline
    )


def column(trace, name):
    index = episode._CONTROL_INDEX[name]
    return trace[:, index]


def test_result_and_arrays_are_documented_and_json_compatible():
    task = make_task("B1", density=0.2, av_fraction=1.0)
    result, arrays = run(task, trace=True)

    json.dumps(result, allow_nan=False)
    assert list(result["control_columns"]) == [
        name for name, _dtype, _text in episode.CONTROL_COLUMNS
    ]
    assert list(result["control_rejection_columns"]) == list(
        episode.REJECTION_REASONS
    )
    assert set(result["mechanism_columns"]) == set(episode.MECHANISM_COLUMNS)
    assert result["mechanism_notes"]["unavailable"] == [
        "native_stochastic_lane_change_draw",
        "native_direction_tie_break",
        "native_conflict_resolution_outcome",
        "longitudinal_s_q_brake_draws",
    ]
    assert "incentive" in result["mechanism_notes"]["pre_pv_native_left_right"]
    assert "safety/accessibility" in result["mechanism_columns"][
        "pre_pv_native_left"
    ]["description"]
    assert result["trace_enabled"] is True
    assert isinstance(result["trace_sha256"], str) and len(result["trace_sha256"]) == 64
    assert result["observer_summary"] is None
    assert result["observer_profile"] == "off"
    assert result["steps_completed"] == 2
    assert result["measurement_steps"] == 2
    assert result["warmup_steps_completed"] == 1
    assert result["model"] is None
    for name in episode.PHASES:
        assert name in result["timings"]
    assert "cpu_seconds" in result["resources"]
    assert "wall_seconds" in result["resources"]
    assert result["resources"]["process_peak_rss_kib"] > 0

    assert arrays["control_trace"].shape == (2, len(episode.CONTROL_COLUMNS))
    assert arrays["control_trace"].dtype == np.int64
    assert arrays["control_rejection"].shape == (2, len(episode.REJECTION_REASONS))
    for name, spec in episode.MECHANISM_COLUMNS.items():
        assert arrays[name].shape == (2,)
        assert arrays[name].dtype == np.dtype(spec["dtype"])
    assert set(np.unique(arrays["pre_pv_native_left"]).tolist()) <= {-1, 0, 1}
    assert set(np.unique(arrays["pre_pv_native_right"]).tolist()) <= {-1, 0, 1}
    assert set(np.unique(arrays["pre_pv_leader_stopped"]).tolist()) <= {-1, 0, 1}
    assert set(np.unique(arrays["pre_pv_native_left_reason"]).tolist()) <= {0, 1, 2, 3}
    assert set(np.unique(arrays["pre_pv_native_right_reason"]).tolist()) <= {0, 1, 2, 3}


def test_result_preserves_validated_task_scientific_metadata():
    task = make_task("B1", density=0.2, av_fraction=1.0)
    result, _ = run(task)
    science = task["science"]
    assert result["task_id"] == task["task_id"]
    assert result["protocol"] == science["protocol"]
    assert result["purpose"] == science["purpose"]
    assert result["split"] == science["split"]
    assert result["method"] == science["method"]
    assert result["density_index"] == science["density_index"]
    assert result["replicate"] == science["replicate"]
    assert result["config"] == science["config"]
    assert result["seeds"] == science["seeds"]
    assert result["model_id"] == science.get("model_id")
    assert result["training_line"] == science.get("training_line")
    assert result["candidate_id"] == science.get("candidate_id")
    assert result["env_steps"] == science.get("env_steps")


def test_default_execution_uses_the_declared_full12_profile():
    task = make_task("B1", density=0.1, av_fraction=0.95)
    result, arrays = episode.simulate_episode(task)
    assert result["steps_completed"] == 2
    assert result["execution_id"]
    assert result["observer_profile"] == "full12"
    assert result["observer_summary"] is not None
    assert "window_hist" in arrays
    assert arrays["window_hist"].shape == (2, 12, 3, 4, 2, 6)


def test_population_density_and_preaction_conservation():
    task = make_task("B1", density=0.2, av_fraction=1.0)
    result, arrays = run(task)
    trace = arrays["control_trace"]

    assert result["initial_vehicle_count"] == 800  # floor(0.2 * 1000 * 4)
    assert result["background_vehicle_count"] == 799
    assert result["av_count"] == 0
    assert result["realized_density"] == pytest.approx(800 / (1000 * 4))
    assert result["realized_background_av_fraction"] == 0.0
    assert result["av_assignment_applied"] is False
    for t in range(2):
        background = (
            trace[t, episode._CONTROL_INDEX["front_background"]]
            + trace[t, episode._CONTROL_INDEX["centre_background"]]
            + trace[t, episode._CONTROL_INDEX["rear_background"]]
        )
        assert int(background) == result["background_vehicle_count"]
        for zone in ("front", "centre", "rear"):
            assert trace[t, episode._CONTROL_INDEX[f"{zone}_background"]] == (
                trace[t, episode._CONTROL_INDEX[f"{zone}_hdv"]]
                + trace[t, episode._CONTROL_INDEX[f"{zone}_av"]]
            )


@pytest.mark.parametrize("method", FIXED_METHODS)
def test_fixed_methods_episode_contract(method):
    task = make_task(method, density=0.2, av_fraction=1.0)
    result, arrays = run(task)
    control = result["control"]
    trace = arrays["control_trace"]

    assert result["steps_completed"] == 2
    assert result["method"] == method
    for key in (
        "mean_speed_priority",
        "flow_background",
        "mean_speed_background",
        "stopped_fraction_background",
        "completed_priority_laps",
        "rule_requests",
    ):
        assert key in result["metrics"]
    for key in (
        "warm_state_sha256",
        "pv_assignment_state_sha256",
        "initial_state_sha256",
        "final_state_sha256",
        "initial_rng_sha256",
        "final_rng_sha256",
    ):
        assert isinstance(result[key], str) and len(result[key]) == 64

    if method == "B1":
        assert control["interface"] == "native"
        assert result["av_assignment_state_sha256"] is None
        assert control["eligible_nonstay_decisions"] is None
        assert control["selected_left"] is None
        assert control["private_throttled_active_decisions"] is None
        assert control["requested_nonstay"] == 0
        assert result["initial_av_ids"] == []
        assert int(trace[0, episode._CONTROL_INDEX["active_av"]]) == 0
    elif method == "B2":
        assert control["interface"] == "rule"
        assert result["av_assignment_state_sha256"] is not None
        assert result["priority_id"] not in result["initial_av_ids"]
        assert len(set(result["initial_av_ids"])) == len(result["initial_av_ids"])
        assert control["eligible_nonstay_decisions"] is None
        assert control["request_opportunity_use"] is None
        assert control["opportunity_use_status"] == "not_applicable"
        assert control["requested_nonstay"] == result["metrics"]["rule_requests"]
        assert control["applied_nonstay"] == result["metrics"]["rule_applied"]
        assert control["rejected_nonstay"] == result["metrics"]["rule_rejected"]
    else:
        assert control["interface"] == "av_action"
        assert result["priority_id"] not in result["initial_av_ids"]
        assert len(set(result["initial_av_ids"])) == len(result["initial_av_ids"])
        assert control["active_av_decisions"] > 0
        assert control["opportunity_use_status"] in (
            "defined",
            "undefined_zero_eligible",
        )
        assert control["eligible_nonstay_decisions"] is not None
        for t in range(2):
            row_requests = int(trace[t, episode._CONTROL_INDEX["requested_nonstay"]])
            row_stays = int(trace[t, episode._CONTROL_INDEX["explicit_stay_commands"]])
            row_active = int(trace[t, episode._CONTROL_INDEX["active_av"]])
            assert row_requests + row_stays == row_active
            assert int(trace[t, episode._CONTROL_INDEX["selected_left"]]) + int(
                trace[t, episode._CONTROL_INDEX["selected_stay"]]
            ) == row_active
            assert int(trace[t, episode._CONTROL_INDEX["selected_right"]]) == 0

    assert control["emitted_nonstay"] == (
        control["applied_nonstay"] + control["rejected_nonstay"]
    )
    assert control["requested_nonstay"] == (
        control["requested_front"]
        + control["requested_centre"]
        + control["requested_rear"]
    )
    assert control["applied_nonstay"] == (
        control["applied_front"] + control["applied_centre"] + control["applied_rear"]
    )
    for t in range(2):
        assert int(trace[t, episode._CONTROL_INDEX["requested_nonstay"]]) == int(
            trace[t, episode._CONTROL_INDEX["applied_nonstay"]]
        ) + int(trace[t, episode._CONTROL_INDEX["rejected_nonstay"]])

    if method == "noop":
        assert control["requested_nonstay"] == 0
        assert control["selected_stay"] == control["active_av_decisions"]
        assert control["explicit_stay_commands"] == control["active_av_decisions"]
    if method == "left_if_safe_original":
        assert control["selected_right"] == 0
        assert control["requested_nonstay"] == control["selected_left"]
        assert control["private_throttled_active_decisions"] is None
    assert sum(control["rejection_reasons"].values()) == control["rejected_nonstay"]
    assert int(arrays["control_rejection"].sum()) == control["rejected_nonstay"]
    if method in HEURISTIC_METHODS:
        assert control["private_cooldown_steps"] == int(method[1:])
        assert (
            control["requested_nonstay"]
            + control["private_suppressed_left_opportunities"]
            == control["available_left_decisions"]
        )
        assert (
            control["private_throttled_active_decisions"]
            >= control["private_suppressed_left_opportunities"]
            >= 0
        )
        assert control["selected_left"] == control["requested_nonstay"]
        assert control["selected_left"] + control["selected_stay"] == (
            control["active_av_decisions"]
        )
        if method == "H1":
            # K=1 already blocks the only t+1 private window.
            assert control["private_suppressed_left_opportunities"] == 0


def test_h1_and_original_left_if_safe_have_full_parity():
    left_task = make_task("left_if_safe_original", density=0.2, av_fraction=1.0)
    h1_task = make_task("H1", density=0.2, av_fraction=1.0)
    left_result, _ = run(left_task, trace=True)
    h1_result, _ = run(h1_task, trace=True)

    assert left_result["initial_state_sha256"] == h1_result["initial_state_sha256"]
    assert left_result["final_state_sha256"] == h1_result["final_state_sha256"]
    assert left_result["final_rng_sha256"] == h1_result["final_rng_sha256"]
    assert left_result["trace_sha256"] == h1_result["trace_sha256"]
    assert left_result["metrics"] == h1_result["metrics"]
    for key in (
        "requested_nonstay",
        "applied_nonstay",
        "rejected_nonstay",
        "available_left_decisions",
        "eligible_nonstay_decisions",
        "explicit_stay_commands",
    ):
        assert left_result["control"][key] == h1_result["control"][key]


@pytest.mark.parametrize("profile", ["full12", "mechanism"])
def test_observer_profiles_are_neutral(profile):
    task = make_task("H2", density=0.2, av_fraction=1.0)
    off_result, off_arrays = run(task, observer="off", trace=True)
    on_result, on_arrays = run(task, observer=profile, trace=True)

    assert off_result["observer_summary"] is None
    assert on_result["observer_summary"] is not None
    for key in (
        "warm_state_sha256",
        "pv_assignment_state_sha256",
        "av_assignment_state_sha256",
        "initial_state_sha256",
        "final_state_sha256",
        "initial_rng_sha256",
        "final_rng_sha256",
        "trace_sha256",
    ):
        assert off_result[key] == on_result[key]
    assert off_result["metrics"] == on_result["metrics"]
    assert off_result["control"] == on_result["control"]
    assert off_result["initial_av_ids"] == on_result["initial_av_ids"]
    np.testing.assert_array_equal(
        off_arrays["control_trace"], on_arrays["control_trace"]
    )
    np.testing.assert_array_equal(
        off_arrays["control_rejection"], on_arrays["control_rejection"]
    )
    assert "window_hist" in on_arrays
    assert on_arrays["window_hist"].shape == (2, 12, 3, 4, 2, 6)
    if profile == "mechanism":
        assert on_arrays["local_occupancy"].shape == (2, 41, 4)
        assert on_arrays["local_speed"].shape == (2, 41, 4)
        assert on_arrays["local_type"].shape == (2, 41, 4)
        assert "local_occupancy" not in off_arrays


def test_reproducibility_and_optional_trace():
    task = make_task("B1", density=0.2, av_fraction=1.0)
    first, first_arrays = run(task, trace=True)
    second, second_arrays = run(task, trace=True)
    plain, plain_arrays = run(task, trace=False)

    assert first["trace_sha256"] == second["trace_sha256"]
    assert first["trace_sha256"] is not None
    assert plain["trace_sha256"] is None
    for key in (
        "initial_state_sha256",
        "final_state_sha256",
        "initial_rng_sha256",
        "final_rng_sha256",
    ):
        assert first[key] == second[key] == plain[key]
    assert first["metrics"] == second["metrics"] == plain["metrics"]
    assert first["control"] == second["control"] == plain["control"]
    for name in first_arrays:
        np.testing.assert_array_equal(first_arrays[name], second_arrays[name])
        np.testing.assert_array_equal(first_arrays[name], plain_arrays[name])


def test_seed_roles_are_independent_and_b1_ignores_av_fraction():
    base = make_task("B1", density=0.1, av_fraction=0.95)
    base_result, base_arrays = run(base)

    changed = dict(BASE_SEEDS, measurement=BASE_SEEDS["measurement"] + 1)
    measurement_result, _ = run(
        make_task("B1", density=0.1, av_fraction=0.95, seeds=changed)
    )
    assert measurement_result["warm_state_sha256"] == base_result["warm_state_sha256"]
    assert measurement_result["pv_assignment_state_sha256"] == base_result[
        "pv_assignment_state_sha256"
    ]
    assert measurement_result["initial_state_sha256"] == base_result[
        "initial_state_sha256"
    ]
    assert measurement_result["initial_rng_sha256"] != base_result["initial_rng_sha256"]
    assert measurement_result["final_rng_sha256"] != base_result["final_rng_sha256"]

    changed = dict(BASE_SEEDS, scenario=BASE_SEEDS["scenario"] + 1)
    scenario_result, _ = run(
        make_task("B1", density=0.1, av_fraction=0.95, seeds=changed)
    )
    assert scenario_result["warm_state_sha256"] != base_result["warm_state_sha256"]

    changed = dict(BASE_SEEDS, pv_assignment=BASE_SEEDS["pv_assignment"] + 1)
    pv_result, _ = run(
        make_task("B1", density=0.1, av_fraction=0.95, seeds=changed)
    )
    assert pv_result["warm_state_sha256"] == base_result["warm_state_sha256"]

    # B1 has no AV assignment, so the requested fraction cannot change it.
    low = make_task("B1", density=0.1, av_fraction=0.9)
    high = make_task("B1", density=0.1, av_fraction=1.0)
    low_result, low_arrays = run(low)
    high_result, high_arrays = run(high)
    assert low_result["initial_state_sha256"] == high_result["initial_state_sha256"]
    assert low_result["metrics"] == high_result["metrics"]
    assert low_result["control"] == high_result["control"]
    assert low_result["av_count"] == high_result["av_count"] == 0
    assert low_result["requested_background_av_fraction"] == 0.9
    assert high_result["requested_background_av_fraction"] == 1.0
    assert low_result["realized_background_av_fraction"] == 0.0
    for name in low_arrays:
        np.testing.assert_array_equal(low_arrays[name], high_arrays[name])
    np.testing.assert_array_equal(
        base_arrays["control_trace"], low_arrays["control_trace"]
    )


def test_av_selection_seed_does_not_change_warm_or_pv_assignment():
    base = make_task("H1", density=0.1, av_fraction=0.95)
    changed = dict(BASE_SEEDS, av_selection=BASE_SEEDS["av_selection"] + 1)
    base_result, _ = run(base)
    changed_result, _ = run(
        make_task("H1", density=0.1, av_fraction=0.95, seeds=changed)
    )
    assert base_result["warm_state_sha256"] == changed_result["warm_state_sha256"]
    assert base_result["pv_assignment_state_sha256"] == changed_result[
        "pv_assignment_state_sha256"
    ]
    assert base_result["av_assignment_state_sha256"] is not None


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    pytest.importorskip("torch")
    root = tmp_path_factory.mktemp("technical_model_bundles")
    output = {}
    for architecture in ("graph", "node", "grid"):
        model = build_model(architecture, BUNDLE_CONFIG)
        weights = {
            key: value.detach().cpu().numpy().copy()
            for key, value in model.state_dict().items()
        }
        save_bundle(
            weights,
            BUNDLE_CONFIG,
            architecture,
            root / architecture,
            {"technical_only": True, "purpose": "episode kernel fixture"},
        )
        identity = load_bundle(root / architecture).identity
        output[architecture] = (root / architecture, identity)
    return output


@pytest.mark.parametrize("architecture", ["graph", "node", "grid"])
def test_learned_methods_use_verified_bundle(architecture, bundles):
    bundle_dir, identity = bundles[architecture]
    task = make_task(
        architecture,
        density=0.2,
        av_fraction=1.0,
        model={
            "model_id": identity,
            "training_line": "technical-line",
            "candidate_id": f"{architecture}-candidate-0",
            "env_steps": 100,
        },
    )
    result, arrays = run(task, model_bundle=str(bundle_dir), trace=True)
    control = result["control"]

    assert result["method"] == architecture
    assert result["architecture"] == architecture
    assert result["model"]["bundle_identity"] == identity
    assert result["model"]["model_id"] == identity
    assert result["model"]["training_line"] == "technical-line"
    assert result["model"]["candidate_id"] == f"{architecture}-candidate-0"
    assert result["model"]["env_steps"] == 100
    assert result["model_id"] == identity
    assert result["training_line"] == "technical-line"
    assert result["candidate_id"] == f"{architecture}-candidate-0"
    assert result["env_steps"] == 100
    assert len(result["model"]["bundle_config_sha256"]) == 64
    assert result["device"]["torch_imported"] is True
    assert result["device"]["type"] == "cpu"
    assert result["device"]["actual"] == "cpu"
    assert control["interface"] == "av_action"
    assert control["active_av_decisions"] > 0
    assert control["legal_nonstay_requests"] <= control["eligible_nonstay_decisions"]
    assert (
        control["applied_legal_nonstay_requests"]
        <= control["legal_nonstay_requests"]
    )
    assert control["opportunity_use_status"] in ("defined", "undefined_zero_eligible")
    if control["opportunity_use_status"] == "defined":
        assert control["request_opportunity_use"] is not None
        assert control["applied_opportunity_use"] is not None
        assert 0.0 <= control["applied_opportunity_use"] <= control["request_opportunity_use"]
    assert control["emitted_nonstay"] == (
        control["applied_nonstay"] + control["rejected_nonstay"]
    )
    assert arrays["control_trace"].shape == (2, len(episode.CONTROL_COLUMNS))
    json.dumps(result, allow_nan=False)


def test_learned_bundle_identity_and_architecture_are_strict(bundles):
    graph_dir, graph_identity = bundles["graph"]
    wrong_identity = make_task(
        "graph",
        density=0.2,
        av_fraction=1.0,
        model={
            "model_id": "0" * 64,
            "training_line": "technical-line",
            "candidate_id": "graph-candidate-0",
            "env_steps": 100,
        },
    )
    with pytest.raises(RuntimeError, match="identity"):
        run(wrong_identity, model_bundle=str(graph_dir))

    wrong_architecture = make_task(
        "node",
        density=0.2,
        av_fraction=1.0,
        model={
            "model_id": graph_identity,
            "training_line": "technical-line",
            "candidate_id": "node-candidate-0",
            "env_steps": 100,
        },
    )
    with pytest.raises(RuntimeError, match="architecture"):
        run(wrong_architecture, model_bundle=str(graph_dir))


def test_learned_bundle_accepts_explicit_transfer_road_length(bundles):
    bundle_dir, identity = bundles["graph"]
    task = make_task(
        "graph",
        density=0.1,
        av_fraction=0.95,
        road_length=2000,
        num_lanes=4,
        model={
            "model_id": identity,
            "training_line": "technical-line",
            "candidate_id": "graph-candidate-0",
            "env_steps": 100,
        },
    )
    result, _ = run(task, model_bundle=str(bundle_dir))
    assert result["config"]["road_length"] == 2000
    assert result["config"]["warmup_steps"] == 1
    assert result["config"]["measure_steps"] == 2
    assert result["initial_vehicle_count"] == 800  # floor(0.1 * 2000 * 4)
    assert result["steps_completed"] == 2
    assert result["model"]["bundle_identity"] == identity


def test_learned_method_requires_transport_bundle(bundles):
    _, identity = bundles["graph"]
    task = make_task(
        "graph",
        density=0.2,
        av_fraction=1.0,
        model={
            "model_id": identity,
            "training_line": "technical-line",
            "candidate_id": "graph-candidate-0",
            "env_steps": 100,
        },
    )
    with pytest.raises((ValueError, RuntimeError)):
        execution = schema.make_execution(task)
        episode.simulate_episode(task, execution=execution)


def test_validate_config_enforces_scientific_profile():
    with pytest.raises(ValueError, match="optimized backend"):
        episode._validate_config(
            config(backend="reference", warmup_steps=1000, measure_steps=2000),
            "B1",
            "scientific",
        )
    with pytest.raises(ValueError, match="warmup_steps"):
        episode._validate_config(
            config(backend="optimized", warmup_steps=1, measure_steps=2000),
            "B1",
            "scientific",
        )
    with pytest.raises(ValueError, match="cooldown"):
        episode._validate_config(config(cooldown_steps=2), "noop", "technical")
    with pytest.raises(ValueError, match="backend"):
        episode._validate_config(config(backend="auto"), "B1", "technical")


def test_scientific_purpose_rejects_reference_backend():
    with pytest.raises(ValueError) as info:
        task = make_task(
            "B1",
            density=0.1,
            av_fraction=0.9,
            road_length=1000,
            num_lanes=4,
            warmup_steps=1000,
            measure_steps=2000,
            backend="reference",
            purpose="scientific",
            split="validation",
            recipe_seeds=True,
        )
        run(task)
    text = str(info.value).lower()
    assert "optimized" in text or "reference" in text or "technical" in text


def test_optimized_availability_failure_is_not_a_silent_fallback(monkeypatch):
    task = make_task(
        "B1",
        density=0.1,
        av_fraction=0.9,
        road_length=1000,
        num_lanes=4,
        warmup_steps=1000,
        measure_steps=2000,
        backend="optimized",
        purpose="scientific",
        split="validation",
        recipe_seeds=True,
    )
    monkeypatch.setattr(episode, "_optimized_backend_available", lambda: False)
    with pytest.raises(RuntimeError, match="optimized"):
        run(task)


def test_reference_backend_returned_for_optimized_is_rejected(monkeypatch):
    from snfs_traffic.core import get_reference_backend
    import snfs_traffic.simulator as simulator_module

    task = make_task(
        "B1",
        density=0.1,
        av_fraction=0.9,
        road_length=1000,
        num_lanes=4,
        warmup_steps=1000,
        measure_steps=2000,
        backend="optimized",
        purpose="scientific",
        split="validation",
        recipe_seeds=True,
    )
    monkeypatch.setattr(episode, "_optimized_backend_available", lambda: True)
    monkeypatch.setattr(
        simulator_module, "get_backend", lambda name="auto": get_reference_backend()
    )
    with pytest.raises(RuntimeError, match="fallback"):
        run(task)


def test_optimized_technical_episode_reaches_completion():
    from snfs_traffic.backends import optimized_backend_available

    if not optimized_backend_available():
        pytest.skip("Numba optimized backend is not available in this environment")
    task = make_task(
        "B1", density=0.1, av_fraction=0.9, backend="optimized", warmup_steps=1
    )
    result, _ = run(task)
    assert result["steps_completed"] == 2
    assert result["backend_paths"] == ["optimized"]


def test_expired_deadline_raises_partial_without_complete_result():
    task = make_task("B1", density=0.2, av_fraction=1.0)
    deadline = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)
    with pytest.raises(episode.EpisodeIncomplete) as info:
        run(task, deadline=deadline)
    partial = info.value.partial
    assert partial["kind"] == "deadline"
    assert partial["status"] == "failed"
    assert partial["steps_completed"] == 0
    assert partial["warmup_steps_completed"] == 0
    assert partial["metrics"] is None
    assert "observer_summary" not in partial
    assert "status" in partial and partial["status"] != "complete"


def test_step_failure_carries_partial_diagnostics(monkeypatch):
    task = make_task("B1", density=0.2, av_fraction=1.0, warmup_steps=0, measure_steps=3)
    calls = {"count": 0}
    original = episode._check_deadline

    def fail_on_third_check(deadline):
        calls["count"] += 1
        if calls["count"] >= 3:
            raise episode.EpisodeIncomplete("forced fixture failure", kind="error")
        return original(deadline)

    monkeypatch.setattr(episode, "_check_deadline", fail_on_third_check)
    with pytest.raises(episode.EpisodeIncomplete) as info:
        run(task)
    partial = info.value.partial
    assert partial["status"] == "partial"
    assert partial["steps_completed"] == 1
    assert partial["measurement_steps_planned"] == 3
    assert partial["metrics"] is not None
    assert partial["final_state_sha256"] is None
    assert "observer_summary" not in partial


@pytest.mark.parametrize("method", ["B1", "B2", "H2"])
@pytest.mark.parametrize("profile", ["off", "full12", "mechanism"])
def test_diagnostics_are_state_rng_action_and_observer_neutral(monkeypatch, method, profile):
    task = make_task(method, density=0.2, av_fraction=1.0, measure_steps=4)
    enabled, enabled_arrays = run(task, observer=profile, trace=True)
    monkeypatch.setattr(episode, "_DIAGNOSTICS_ENABLED", False)
    disabled, disabled_arrays = run(task, observer=profile, trace=True)
    for key in ("initial_state_sha256", "final_state_sha256", "initial_rng_sha256",
                "final_rng_sha256", "trace_sha256", "metrics", "control", "observer_summary"):
        assert enabled[key] == disabled[key]
    for name in enabled_arrays:
        if name not in episode.DIAGNOSTIC_COLUMNS:
            np.testing.assert_array_equal(enabled_arrays[name], disabled_arrays[name])


@pytest.mark.parametrize("profile", ["off", "full12", "mechanism"])
def test_w5_episode_arrays_pass_independent_reader(tmp_path, profile):
    from physica_mc import storage

    task = make_task("H2", density=0.2, av_fraction=1.0, measure_steps=4)
    execution = schema.make_execution(task, observer=profile, device="cpu", threads=1)
    result, arrays = episode.simulate_episode(task, execution=execution)
    marker = storage.publish_episode(tmp_path, task, execution, "w5", result, arrays)
    _, loaded = storage.read_episode(marker, required_profile=profile)
    for name in episode.MECHANISM_COLUMNS:
        np.testing.assert_array_equal(loaded[name], arrays[name])
