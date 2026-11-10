"""Metadata-only planning regressions; no model loading or simulation.

Prepared source only. Execution belongs to the parent's approved container.
"""
from copy import deepcopy

import pytest

from physica_mc import planning, schema, selection


def hashed(label):
    return schema.digest(label)


def rehash(plan):
    plan["plan_id"] = schema.digest({k: v for k, v in plan.items() if k != "plan_id"})
    return plan


def rebind(packet):
    packet["packet_sha256"] = schema.digest({k: v for k, v in packet.items() if k != "packet_sha256"})
    return packet


def proof():
    return {"kind": "dependency_equivalence", "scope": "B1_fraction", "verified": True,
            "evidence_sha256": hashed("explicit full dependency evidence"),
            "dependencies": ["all_inputs", "actions", "state", "rng"]}


def model(identity):
    return {"bundle_path": "/explicit/input/bundle", "manifest_sha256": hashed(identity),
            "identity": identity, "identity_verified": True,
            "eligibility_sha256": hashed(["scientific admission", identity])}


def campaign(*, w7=False, frozen=False):
    """Generate explicit synthetic hierarchy and candidate/model declarations."""
    lines, candidates, models = [], [], {}
    for method in ("graph", "node", "grid"):
        for scale in ((1.0, 0.5, 2.0) if method == "graph" and w7 else (1.0,)):
            for seed in range(3):
                name = f"{method}/scale={scale}/seed={seed}"
                lines.append({"architecture": method, "training_line": name,
                              "cohort": "main" if scale == 1.0 else "w7",
                              "scale": scale, "training_seed": str(seed)})
                group = {"family": "RL", "architecture": method, "training_line": name}
                for steps in planning.K13_ENV_STEPS:
                    candidate_id = f"checkpoint-{steps}"
                    identity = {"method": method, "training_line": name,
                                "candidate_id": candidate_id, "env_steps": steps}
                    model_id = hashed(["numeric identity", identity])
                    entry = model(identity)
                    models[model_id] = entry
                    candidates.append({"group": group, "candidate_id": candidate_id,
                                       "env_steps": steps, "C": None, "model_id": model_id,
                                       "manifest_sha256": entry["manifest_sha256"],
                                       "eligibility_sha256": entry["eligibility_sha256"]})
    for C in (1, 2, 4, 8):
        candidates.append({"group": {"family": "H", "architecture": "", "training_line": ""},
                           "candidate_id": f"H{C}", "C": C, "env_steps": 0,
                           "model_id": None, "manifest_sha256": None, "eligibility_sha256": None})
    bindings = {"scientific_version": "explicit-synthetic-science", "runtime": "synthetic-runtime",
                "stream_bank": "synthetic-new-stream-bank", "sap_sha256": hashed("SAP"),
                "seed_recipe_sha256": planning.seed_recipe_sha256(),
                "admission_sha256": hashed("explicit campaign admission, not converter provenance")}
    receipt = None
    if frozen:
        pools = {}
        for c in candidates:
            candidate = selection.candidate_from_json({k: c[k] for k in (
                "group", "candidate_id", "env_steps", "C")})
            pools.setdefault(candidate.group, []).append(candidate)
        selections = {}
        for group, pool in pools.items():
            pool.sort(key=lambda c: c.C if group.family == "H" else c.env_steps)
            choice = selection.Selection("selected", pool[0].candidate_id,
                                         tuple(c.candidate_id for c in pool))
            selections[group] = selection.GroupSelection(
                "complete", {c.candidate_id: 1.0 for c in pool},
                {c.candidate_id: dict.fromkeys(selection.CONDITIONS, 1.0) for c in pool},
                choice if group.family == "RL" else None,
                {tau: choice for tau in selection.THRESHOLDS},
            )
        receipt = selection.freeze_receipt(
            selections, pools, expected_groups=list(pools),
            provenance=selection.Provenance(*(bindings[k] for k in (
                "scientific_version", "runtime", "stream_bank"))),
        )
    return planning.make_freeze_binding(lines=lines, candidates=candidates,
                                        bindings=bindings, receipt=receipt), models


def learned_task():
    identity = {"method": "graph", "training_line": "line-A", "candidate_id": "anchor-A",
                "env_steps": 100 * 16384}
    task = schema.make_task("graph", model_id="explicit-numeric-identity", **{
        k: v for k, v in identity.items() if k != "method"})
    return task, {task["science"]["model_id"]: model(identity)}


def test_ordering_detachment_and_seed_blocks():
    tasks = [schema.make_task(method, density=density, av_fraction=fraction, replicate=2)
             for method in ("H1", "left_if_safe_original") for density in (0.1, 0.3)
             for fraction in planning.FRACTIONS]
    forward = planning.make_plan(tasks)
    assert forward == planning.make_plan(reversed(tasks))
    for density in (0.1, 0.3):
        blocks = [t for t in forward["tasks"] if t["science"]["config"]["density"] == density]
        assert len({schema.digest(t["science"]["seeds"]) for t in blocks}) == 1
        assert len({schema.pairing_key(t) for t in blocks}) == 1
    assert len({schema.digest(t["science"]["seeds"]) for t in tasks}) == 2
    assert len(forward["tasks"]) == 12  # H1/original have never been collapsed.
    tasks[0]["science"]["seeds"]["scenario"] += 1
    planning.validate_plan(forward)


def test_transport_changes_only_plan_identity_and_model_validation():
    task, models = learned_task()
    first = planning.make_plan([task], models=models)
    moved = deepcopy(models)
    moved[task["science"]["model_id"]]["bundle_path"] = "/relocated/weights"
    second = planning.make_plan([task], models=moved)
    assert first["plan_id"] != second["plan_id"]
    assert first["tasks"] == second["tasks"]
    for field, value in (("identity_verified", False), ("manifest_sha256", "bad")):
        broken = deepcopy(models)
        broken[task["science"]["model_id"]][field] = value
        with pytest.raises(ValueError):
            planning.make_plan([task], models=broken)
    broken = deepcopy(models)
    broken[task["science"]["model_id"]]["identity"]["training_line"] = "same-seed-other-line"
    with pytest.raises(ValueError):
        planning.make_plan([task], models=broken)
    with pytest.raises(ValueError):
        planning.make_plan([task])


def test_exclusive_idempotent_io_and_corruption(tmp_path):
    plan = planning.make_plan([schema.make_task("B1")])
    path = tmp_path / "plan.json"
    assert planning.write_plan(plan, path) == path
    before = path.read_bytes()
    assert planning.write_plan(plan, path) == path
    assert path.read_bytes() == before
    assert planning.load_plan(path) == plan
    with pytest.raises(ValueError):
        planning.write_plan(planning.make_plan([schema.make_task("B2")]), path)
    assert path.read_bytes() == before
    broken = tmp_path / "partial.json"
    broken.write_text('{"schema_version":', encoding="utf-8")
    with pytest.raises(ValueError):
        planning.write_plan(plan, broken)
    assert broken.read_text() == '{"schema_version":'
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"plan_id":"x","plan_id":"y"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        planning.load_plan(duplicate)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        planning.write_plan(plan, link)


def test_plan_corruption_and_duplicate_ids():
    task = schema.make_task("B1")
    plan = planning.make_plan([task])
    plan["phase"] = "validation"
    with pytest.raises(ValueError, match="hash"):
        planning.validate_plan(plan)
    with pytest.raises(ValueError, match="duplicate task"):
        planning.make_plan([task, task])
    plan = planning.make_plan([task])
    plan["tasks"][0]["science"]["replicate"] = 1
    with pytest.raises(ValueError):
        planning.validate_plan(rehash(plan))
    with pytest.raises(ValueError):
        planning.make_plan([schema.make_task("B1", split="validation")])
    with pytest.raises(ValueError):
        planning.make_plan([schema.make_task("B1", purpose="scientific")])


def test_aliases_need_explicit_dependency_proof():
    task = schema.make_task("B1", av_fraction=1.0)
    alias = {"alias_id": "explicit-reference", "task_id": task["task_id"], "label": "B1",
             "condition": {"density": 0.2, "fraction": 0.9, "split": "pilot", "replicate": 0},
             "proof": proof()}
    planning.make_plan([task], aliases=[alias])
    with pytest.raises(ValueError, match="duplicate alias"):
        planning.make_plan([task], aliases=[alias, alias])
    for mutation in ({"proof": {"kind": "equal_metrics"}}, {"task_id": "absent"},
                     {"proof": {"kind": "identical_task", "task_id": task["task_id"]}}):
        with pytest.raises(ValueError):
            planning.make_plan([task], aliases=[dict(alias, **mutation)])
    altered = deepcopy(alias)
    altered["proof"]["dependencies"] = ["metrics"]
    with pytest.raises(ValueError):
        planning.make_plan([task], aliases=[altered])


@pytest.mark.parametrize("w7,rl_count", [(False, 28080), (True, 28080 + 18720)])
def test_full_validation_counts(w7, rl_count):
    packet, models = campaign(w7=w7)
    plan, coverage = planning.make_validation_plan(freeze=packet, models=models,
                                                  b1_fraction_proof=proof())
    counts = {}
    for task in plan["tasks"]:
        method = task["science"]["method"]
        counts[method] = counts.get(method, 0) + 1
    assert sum(counts[m] for m in schema.LEARNED_METHODS) == rl_count
    assert [counts[f"H{c}"] for c in (1, 2, 4, 8)] == [240] * 4
    assert counts["B1"] == 80
    assert len(plan["aliases"]) == 240
    assert coverage["ready_inputs_complete"] is True
    assert coverage["missing_candidates"] == coverage["missing_lines"] == []
    assert coverage["expected_candidate_trajectories"] == rl_count + 960
    assert len(plan["tasks"]) == rl_count + 960 + 80
    assert len({t["science"]["training_line"] for t in plan["tasks"]
                if t["science"]["method"] == "graph"}) == (9 if w7 else 3)


def test_ready_subset_is_explicit_and_cannot_open_test_gate():
    packet, models = campaign()
    first = next(c for c in packet["candidates"] if c["group"]["family"] == "RL")
    ready = {first["model_id"]: models[first["model_id"]]}
    plan, coverage = planning.make_validation_plan(freeze=packet, models=ready)
    assert len(plan["tasks"]) == 240 + 960 + 240
    assert len(coverage["missing_candidates"]) == 116
    assert len(coverage["missing_lines"]) == 9
    assert coverage["ready_inputs_complete"] is False
    assert len(plan["freeze"]["candidates"]) == 121
    with pytest.raises(ValueError, match="completed freeze"):
        planning.make_frozen_test_plan(freeze=packet, models=ready)
    # Unavailable source identities remain explicit expected candidates.
    missing = deepcopy(packet)
    unavailable = next(c for c in missing["candidates"] if c == first)
    for name in ("model_id", "manifest_sha256", "eligibility_sha256"):
        unavailable[name] = None
    partial = rebind(missing)
    _, report = planning.make_validation_plan(freeze=partial, models={})
    assert len(report["missing_candidates"]) == 117


def test_catalog_order_and_incomplete_candidate_barrier():
    packet, models = campaign()
    reversed_packet = planning.make_freeze_binding(
        lines=reversed(packet["lines"]), candidates=reversed(packet["candidates"]),
        bindings=packet["bindings"],
    )
    assert packet == reversed_packet
    for mutation in ("missing", "anchor", "merged_line", "converter_provenance"):
        broken = deepcopy(packet)
        if mutation == "missing":
            broken["candidates"].pop()
        elif mutation == "anchor":
            next(c for c in broken["candidates"] if c["group"]["family"] == "RL")["env_steps"] = 1220 * 16384
        elif mutation == "merged_line":
            broken["lines"][1]["training_line"] = broken["lines"][0]["training_line"]
        else:
            next(c for c in broken["candidates"] if c["group"]["family"] == "RL")["eligibility_sha256"] = None
        with pytest.raises(ValueError):
            planning.make_validation_plan(freeze=rebind(broken), models={})


def test_global_frozen_union_deduplicates_labels_not_w7_lines():
    packet, models = campaign(w7=True, frozen=True)
    selected_ids = {c["model_id"] for c in packet["candidates"]
                    if c["group"]["family"] == "RL" and c["env_steps"] == planning.K13_ENV_STEPS[0]}
    ready = {key: models[key] for key in selected_ids}
    plan = planning.make_frozen_test_plan(freeze=packet, models=ready, b1_fraction_proof=proof())
    # Fifteen exact RL policies and one H policy, plus 540 physical B1 roads.
    assert len(plan["tasks"]) == (15 + 1) * 54 * 30 + 18 * 30
    assert len(plan["aliases"]) == (15 * 5 + 4) * 54 * 30 + 54 * 30
    by_line = {}
    for task in plan["tasks"]:
        s = task["science"]
        if s["method"] == "graph":
            by_line.setdefault(s["training_line"], set()).add(task["task_id"])
    assert len(by_line) == 9
    assert all(len(ids) == 1620 for ids in by_line.values())
    assert len(set.union(*by_line.values())) == 9 * 1620
    ready.pop(next(iter(ready)))
    with pytest.raises(ValueError, match="not ready"):
        planning.make_frozen_test_plan(freeze=packet, models=ready)


@pytest.mark.parametrize("mutation", ["hash", "missing_group", "missing_rule", "pool_hash",
                                       "seed", "runtime", "sap", "model", "missing_model"])
def test_corrupt_or_partial_freeze_cannot_open_test(mutation):
    packet, models = campaign(frozen=True)
    if mutation == "hash":
        packet["packet_sha256"] = "0" * 64
    elif mutation == "missing_group":
        packet["receipt"]["groups"].pop()
    elif mutation == "missing_rule":
        packet["receipt"]["selection_manifest"].pop()
    elif mutation == "pool_hash":
        packet["receipt"]["groups"][0]["pool_sha256"] = "0" * 64
    elif mutation == "seed":
        packet["bindings"]["seed_recipe_sha256"] = "0" * 64
    elif mutation == "runtime":
        packet["bindings"]["runtime"] = "different-runtime"
    elif mutation == "sap":
        packet["bindings"]["sap_sha256"] = None
    elif mutation == "model":
        candidate = next(c for c in packet["candidates"] if c["group"]["family"] == "RL"
                         and c["env_steps"] == planning.K13_ENV_STEPS[0])
        candidate["manifest_sha256"] = "0" * 64
    else:
        candidate = next(c for c in packet["candidates"] if c["group"]["family"] == "RL")
        for key in ("model_id", "manifest_sha256", "eligibility_sha256"):
            candidate[key] = None
    if mutation != "hash":
        # Rehash outer envelopes too: internal semantic validation must still fail.
        receipt = packet["receipt"]
        receipt["receipt_sha256"] = schema.digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
        rebind(packet)
    with pytest.raises(ValueError):
        planning.make_frozen_test_plan(freeze=packet, models=models)


def test_direct_scientific_tasks_require_binding_and_frozen_policy():
    scientific = schema.make_task("B1", split="fresh_test", purpose="scientific")
    with pytest.raises(ValueError):
        planning.make_plan([scientific], phase="fresh_test")
    packet, models = campaign(frozen=True)
    c = next(c for c in packet["candidates"] if c["group"]["family"] == "RL"
             and c["env_steps"] == planning.K13_ENV_STEPS[1])
    task = schema.make_task(c["group"]["architecture"], split="fresh_test", purpose="scientific",
                            model_id=c["model_id"], training_line=c["group"]["training_line"],
                            candidate_id=c["candidate_id"], env_steps=c["env_steps"])
    with pytest.raises(ValueError, match="not selected"):
        planning.make_plan([task], models=models, phase="fresh_test", freeze=packet)
    with pytest.raises(ValueError, match="H policy"):
        planning.make_plan([schema.make_task("H8", split="fresh_test", purpose="scientific")],
                           phase="fresh_test", freeze=packet)
    scientific["science"]["seeds"]["scenario"] += 1
    scientific["task_id"] = schema.digest(scientific["science"])
    with pytest.raises(ValueError, match="seed recipe"):
        planning.make_plan([scientific], phase="fresh_test", freeze=packet)


def test_infeasible_and_undefined_rules_are_retained_without_fallback():
    packet, models = campaign(frozen=True)
    receipt = packet["receipt"]
    pools, selections = {}, {}
    for entry in receipt["groups"]:
        group = selection.group_from_json(entry["group"])
        pools[group] = [selection.candidate_from_json(c) for c in entry["pool"]]
        outcome = selection.group_selection_from_json(entry["selection"])
        if group.family == "H":
            outcome = selection.GroupSelection(
                "complete", outcome.speed,
                {c.candidate_id: dict.fromkeys(selection.CONDITIONS, 0.5) for c in pools[group]},
                None, {tau: selection.Selection("infeasible", None) for tau in selection.THRESHOLDS},
            )
        else:
            outcome = selection.GroupSelection(
                "constraint_undefined", outcome.speed,
                {c.candidate_id: dict.fromkeys(selection.CONDITIONS, None) for c in pools[group]},
                outcome.s0,
                {tau: selection.Selection("constraint_undefined", None) for tau in selection.THRESHOLDS},
            )
        selections[group] = outcome
    receipt = selection.freeze_receipt(
        selections, pools, expected_groups=list(pools),
        provenance=selection.provenance_from_json(receipt["provenance"]),
    )
    packet = planning.make_freeze_binding(lines=packet["lines"], candidates=packet["candidates"],
                                         bindings=packet["bindings"], receipt=receipt)
    # Validate the bound outcomes without creating another large full-grid plan.
    plan = planning.make_plan([], models=models, phase="fresh_test", freeze=packet)
    rows = plan["freeze"]["receipt"]["selection_manifest"]
    assert sum(row["status"] == "selected" for row in rows) == 9  # S0 only
    assert sum(row["status"] == "infeasible" for row in rows) == 4
    assert sum(row["status"] == "constraint_undefined" for row in rows) == 36
    with pytest.raises(ValueError, match="H policy"):
        planning.make_plan([schema.make_task("H1", split="fresh_test", purpose="scientific")],
                           phase="fresh_test", freeze=packet)
