"""Immutable metadata plans and the E2 validation/fresh-test boundary.

No model files are opened here. Admission and equivalence evidence are explicit
caller attestations, never conclusions drawn from conversion or metric values.
Selector helpers are imported lazily only for scientific campaign packets.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

from . import schema

K13_ENV_STEPS = tuple(i * 16384 for i in (*range(100, 1201, 100), 1221))
FRACTIONS = (0.9, 0.95, 1.0)
VALIDATION_DENSITIES = (0.1, 0.2, 0.3, 0.4)
_MODEL_IDENTITY = ("method", "training_line", "candidate_id", "env_steps")


def _keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name}: unknown or missing fields")


def _text(value, name):
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        raise ValueError(f"{name}: expected nonempty text")


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name}: expected lowercase SHA256")


def _copy(value):
    return json.loads(schema.canonical_json(value))


def _group_key(group):
    _keys(group, ("family", "architecture", "training_line"), "group")
    return schema.canonical_json(group).decode()


def _candidate_key(candidate):
    return (_group_key(candidate["group"]), candidate["candidate_id"])


def _identity(candidate):
    return dict(zip(_MODEL_IDENTITY, (
        candidate["group"]["architecture"], candidate["group"]["training_line"],
        candidate["candidate_id"], candidate["env_steps"],
    )))


def _model(entry):
    _keys(entry, ("bundle_path", "manifest_sha256", "identity", "identity_verified",
                  "eligibility_sha256"), "model")
    _text(entry["bundle_path"], "bundle_path")
    _sha(entry["manifest_sha256"], "manifest_sha256")
    if entry["identity_verified"] is not True:
        raise ValueError("model identity must be explicitly verified")
    _keys(entry["identity"], _MODEL_IDENTITY, "model identity")
    identity = entry["identity"]
    if identity["method"] not in schema.LEARNED_METHODS:
        raise ValueError("model architecture must be graph/node/grid")
    for name in ("training_line", "candidate_id"):
        _text(identity[name], name)
    if type(identity["env_steps"]) is not int or identity["env_steps"] <= 0:
        raise ValueError("invalid model env_steps")
    if entry["eligibility_sha256"] is not None:
        _sha(entry["eligibility_sha256"], "eligibility_sha256")


def _proof(proof):
    _keys(proof, ("kind", "scope", "verified", "evidence_sha256", "dependencies"), "proof")
    if (proof["kind"] != "dependency_equivalence" or proof["scope"] != "B1_fraction"
            or proof["verified"] is not True
            or proof["dependencies"] != ["all_inputs", "actions", "state", "rng"]):
        raise ValueError("B1 reuse needs explicit full dependency-equivalence proof")
    _sha(proof["evidence_sha256"], "equivalence evidence")


def _validate_alias(alias, tasks):
    _keys(alias, ("alias_id", "task_id", "label", "condition", "proof"), "alias")
    _text(alias["alias_id"], "alias_id")
    _text(alias["label"], "label")
    if alias["task_id"] not in tasks:
        raise ValueError("alias references an absent canonical task")
    science = tasks[alias["task_id"]]["science"]
    condition = alias["condition"]
    _keys(condition, ("density", "fraction", "split", "replicate"), "alias condition")
    if (type(condition["density"]) is not float
            or condition["density"] != science["config"]["density"]
            or condition["split"] != science["split"]
            or type(condition["replicate"]) is not int
            or condition["replicate"] != science["replicate"]
            or type(condition["fraction"]) is not float
            or not 0 <= condition["fraction"] <= 1):
        raise ValueError("alias condition differs from canonical block")
    if alias["proof"] == {"kind": "identical_task", "task_id": alias["task_id"]}:
        if condition["fraction"] != science["config"]["av_fraction"]:
            raise ValueError("identical-task alias changes fraction")
    else:
        _proof(alias["proof"])
        if science["method"] != "B1" or condition["fraction"] not in FRACTIONS:
            raise ValueError("fraction proof applies only to B1")


def _catalog(lines, candidates):
    """Check explicit full expectations, independently of ready model inputs."""
    if not isinstance(lines, list) or not isinstance(candidates, list):
        raise ValueError("lines and candidates must be arrays")
    line_map = {}
    seeds = set()
    counts = {}
    for line in lines:
        _keys(line, ("architecture", "training_line", "cohort", "scale", "training_seed"), "line")
        architecture, name = line["architecture"], line["training_line"]
        if architecture not in schema.LEARNED_METHODS:
            raise ValueError("unknown line architecture")
        _text(name, "training_line")
        _text(line["training_seed"], "training_seed")
        if type(line["scale"]) is not float:
            raise ValueError("training scale must be an explicit float")
        if line["cohort"] == "main":
            if line["scale"] != 1.0:
                raise ValueError("main lines require scale=1")
        elif line["cohort"] == "w7":
            if architecture != "graph" or line["scale"] not in (0.5, 2.0):
                raise ValueError("W7 additions are graph scale=0.5/2")
        else:
            raise ValueError("unknown line cohort")
        key = (architecture, name)
        seed = (architecture, line["scale"], line["training_seed"])
        if key in line_map or seed in seeds:
            raise ValueError("duplicate training identity")
        line_map[key] = line
        seeds.add(seed)
        count_key = (line["cohort"], architecture, line["scale"])
        counts[count_key] = counts.get(count_key, 0) + 1
    expected = {("main", method, 1.0): 3 for method in schema.LEARNED_METHODS}
    if any(line["cohort"] == "w7" for line in lines):
        expected.update({("w7", "graph", scale): 3 for scale in (0.5, 2.0)})
    if counts != expected:
        raise ValueError("expected nine main lines and either zero or six W7 additions")
    pools = {}
    identities = set()
    models = set()
    for candidate in candidates:
        _keys(candidate, ("group", "candidate_id", "env_steps", "C", "model_id",
                          "manifest_sha256", "eligibility_sha256"), "candidate")
        key = _candidate_key(candidate)
        _text(candidate["candidate_id"], "candidate_id")
        if key in identities:
            raise ValueError("duplicate candidate identity")
        identities.add(key)
        group = candidate["group"]
        if group["family"] == "H":
            if group["architecture"] or group["training_line"]:
                raise ValueError("H cannot have a training identity")
            if type(candidate["C"]) is not int or candidate["C"] not in (1, 2, 4, 8):
                raise ValueError("invalid H candidate")
            if type(candidate["env_steps"]) is not int or candidate["env_steps"] != 0:
                raise ValueError("H env_steps must be zero")
            if any(candidate[k] is not None for k in ("model_id", "manifest_sha256", "eligibility_sha256")):
                raise ValueError("H has no model identity")
            anchor = candidate["C"]
        elif group["family"] == "RL":
            if (group["architecture"], group["training_line"]) not in line_map:
                raise ValueError("candidate belongs to an unexpected line")
            if (type(candidate["env_steps"]) is not int or candidate["env_steps"] not in K13_ENV_STEPS
                    or candidate["C"] is not None):
                raise ValueError("candidate must use the exact K13 anchor schedule")
            anchor = candidate["env_steps"]
            if candidate["model_id"] is None:
                if candidate["manifest_sha256"] is not None or candidate["eligibility_sha256"] is not None:
                    raise ValueError("unavailable model cannot claim hashes")
            else:
                _text(candidate["model_id"], "model_id")
                if candidate["model_id"] in models:
                    raise ValueError("one model identity cannot represent different candidates/lines")
                models.add(candidate["model_id"])
                _sha(candidate["manifest_sha256"], "candidate manifest hash")
                _sha(candidate["eligibility_sha256"], "explicit scientific eligibility")
        else:
            raise ValueError("unknown candidate family")
        pool = pools.setdefault(key[0], [])
        pool.append(anchor)
    expected_groups = [{"family": "H", "architecture": "", "training_line": ""}] + [
        {"family": "RL", "architecture": method, "training_line": name}
        for method, name in line_map
    ]
    if set(pools) != {_group_key(group) for group in expected_groups}:
        raise ValueError("missing expected candidate group")
    for group in expected_groups:
        anchors = (1, 2, 4, 8) if group["family"] == "H" else K13_ENV_STEPS
        if sorted(pools[_group_key(group)]) != list(anchors):
            raise ValueError("missing or duplicated expected candidate anchors")
    return expected_groups


def make_freeze_binding(*, lines, candidates, bindings, receipt=None):
    """Bind E2 expectations/admission; receipt=None is validation-only.

    See PLANNING.md for the JSON-only line, candidate and binding records.
    A completed receipt is produced by selection.freeze_receipt, not here.
    """
    packet = _copy({"schema_version": 1, "verified": True,
                    "lines": list(lines), "candidates": list(candidates),
                    "bindings": bindings, "receipt": receipt})
    packet["lines"].sort(key=lambda x: (x["architecture"], x["training_line"]))
    packet["candidates"].sort(key=_candidate_key)
    packet["packet_sha256"] = schema.digest(packet)
    _validate_freeze(packet, completed=receipt is not None)
    return packet


def _validate_freeze(packet, *, completed):
    _keys(packet, ("schema_version", "verified", "lines", "candidates", "bindings",
                   "receipt", "packet_sha256"), "freeze binding")
    if type(packet["schema_version"]) is not int or packet["schema_version"] != 1 or packet["verified"] is not True:
        raise ValueError("freeze binding requires explicit verified admission")
    _sha(packet["packet_sha256"], "packet hash")
    if schema.digest({k: v for k, v in packet.items() if k != "packet_sha256"}) != packet["packet_sha256"]:
        raise ValueError("freeze packet hash mismatch")
    groups = _catalog(packet["lines"], packet["candidates"])
    bindings = packet["bindings"]
    _keys(bindings, ("scientific_version", "runtime", "stream_bank", "sap_sha256",
                     "seed_recipe_sha256", "admission_sha256"), "bindings")
    for name in ("scientific_version", "runtime", "stream_bank"):
        _text(bindings[name], name)
    for name in ("sap_sha256", "seed_recipe_sha256", "admission_sha256"):
        _sha(bindings[name], name)
    if bindings["seed_recipe_sha256"] != seed_recipe_sha256():
        raise ValueError("seed recipe binding mismatch")
    receipt = packet["receipt"]
    if receipt is None:
        if completed:
            raise ValueError("fresh_test requires the global completed freeze")
        return
    if any(c["group"]["family"] == "RL" and c["model_id"] is None for c in packet["candidates"]):
        raise ValueError("cannot freeze with missing candidate model identities")
    # Reuse public selector conversion/freezing helpers. The envelope verifier
    # alone does not check complete groups, rules, or internally consistent hashes.
    from . import selection
    selection.verify_freeze_receipt(receipt)
    selections, pools = {}, {}
    for entry in receipt["groups"]:
        group = selection.group_from_json(entry["group"])
        if group in pools:
            raise ValueError("duplicate frozen group")
        pools[group] = [selection.candidate_from_json(c) for c in entry["pool"]]
        selections[group] = selection.group_selection_from_json(entry["selection"])
    rebuilt = selection.freeze_receipt(
        selections, pools,
        provenance=selection.provenance_from_json({k: bindings[k] for k in (
            "scientific_version", "runtime", "stream_bank")}),
        expected_groups=[selection.group_from_json(g) for g in groups],
    )
    if schema.canonical_json(rebuilt) != schema.canonical_json(receipt):
        raise ValueError("freeze receipt is not a complete canonical selection")
    expected_pool = sorted(({
        k: c[k] for k in ("group", "candidate_id", "env_steps", "C")
    } for c in packet["candidates"]), key=_candidate_key)
    actual_pool = sorted((c for entry in receipt["groups"] for c in entry["pool"]), key=_candidate_key)
    if expected_pool != actual_pool:
        raise ValueError("freeze pool differs from expected candidate identities")


def seed_recipe_sha256():
    """Hash the schema-owned seed recipe and full canonical density indexing."""
    return schema.digest({"bases": schema.SPLIT_BASES, "offsets": schema.ROLE_OFFSETS,
                          "density_stride": schema.DENSITY_STEP_INDEX,
                          "replicate_stride": schema.REPLICATE_STEP_INDEX,
                          "densities": list(schema.DENSITY_GRID)})


def make_plan(tasks, *, models=None, aliases=None, phase="technical", freeze=None):
    """Create a detached, canonically ordered run-contract-v1 plan."""
    plan = _copy({"schema_version": 1, "phase": phase, "tasks": list(tasks),
                  "models": {} if models is None else models,
                  "aliases": [] if aliases is None else list(aliases), "freeze": freeze})
    plan["tasks"].sort(key=lambda t: t["task_id"])
    plan["aliases"].sort(key=lambda a: a["alias_id"])
    plan["plan_id"] = schema.digest(plan)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    """Reject corrupted, ambiguous, or scientifically unbound plans."""
    schema.canonical_json(plan)
    _keys(plan, ("schema_version", "plan_id", "phase", "tasks", "models", "aliases", "freeze"), "plan")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise ValueError("unsupported plan schema")
    _sha(plan["plan_id"], "plan_id")
    if schema.digest({k: v for k, v in plan.items() if k != "plan_id"}) != plan["plan_id"]:
        raise ValueError("plan hash mismatch")
    phase = plan["phase"]
    if phase not in ("technical", "validation", "fresh_test"):
        raise ValueError("unknown plan phase")
    if not isinstance(plan["tasks"], list) or not isinstance(plan["aliases"], list) or not isinstance(plan["models"], dict):
        raise ValueError("invalid plan arrays/registry")
    scientific = phase != "technical"
    packet = plan["freeze"]
    if packet is not None:
        _validate_freeze(packet, completed=phase == "fresh_test")
    elif scientific:
        raise ValueError("scientific plans require explicit admission/freeze bindings")
    candidates = {} if packet is None else {_candidate_key(c): c for c in packet["candidates"]}
    selected = set()
    if phase == "fresh_test":
        selected = {(_group_key(row["group"]), row["candidate_id"])
                    for row in packet["receipt"]["selection_manifest"] if row["status"] == "selected"}
    for model_id, entry in plan["models"].items():
        _text(model_id, "model_id")
        _model(entry)
    tasks = {}
    for task in plan["tasks"]:
        schema.validate_task(task)
        if task["task_id"] in tasks:
            raise ValueError("duplicate task ID")
        tasks[task["task_id"]] = task
        science = task["science"]
        if science["split"] != ("pilot" if phase == "technical" else phase) or science["purpose"] != ("scientific" if scientific else "technical"):
            raise ValueError("task purpose/split disagrees with plan phase")
        method = science["method"]
        if method in schema.LEARNED_METHODS:
            entry = plan["models"].get(science["model_id"])
            if entry is None or entry["identity"] != {k: science[k] for k in _MODEL_IDENTITY}:
                raise ValueError("learned task requires an exact verified model/training identity")
            if scientific:
                key = (_group_key({"family": "RL", "architecture": method,
                                   "training_line": science["training_line"]}), science["candidate_id"])
                candidate = candidates.get(key)
                if candidate is None or any(candidate[k] != entry[k] for k in ("manifest_sha256", "eligibility_sha256")) or candidate["model_id"] != science["model_id"] or candidate["env_steps"] != science["env_steps"]:
                    raise ValueError("model differs from explicitly eligible candidate")
                if phase == "fresh_test" and key not in selected:
                    raise ValueError("test policy was not selected in the global freeze")
        elif phase == "fresh_test" and method.startswith("H"):
            if not any(_candidate_key(c) in selected and c["C"] == int(method[1:]) for c in candidates.values() if c["group"]["family"] == "H"):
                raise ValueError("H policy was not selected in the global freeze")
        if scientific:
            if science["seeds"] != schema.seed_record(phase, science["density_index"], science["replicate"]):
                raise ValueError("scientific task differs from bound seed recipe")
    ids = [a["alias_id"] for a in plan["aliases"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate alias ID")
    for alias in plan["aliases"]:
        _validate_alias(alias, tasks)
    if plan["tasks"] != sorted(plan["tasks"], key=lambda t: t["task_id"]) or plan["aliases"] != sorted(plan["aliases"], key=lambda a: a["alias_id"]):
        raise ValueError("plan arrays must use canonical ID ordering")


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def load_plan(path):
    """Read and independently validate JSON, including duplicate-key rejection."""
    with open(path, "r", encoding="utf-8") as stream:
        plan = json.load(stream, object_pairs_hook=_json_object)
    validate_plan(plan)
    return plan


def write_plan(plan, destination):
    """Exclusive creation; identical existing content is the only retry allowed.

    A failed write may leave a partial file. It is rejected, never overwritten
    or removed; operator reconciliation is required before another destination.
    """
    validate_plan(plan)
    destination = Path(destination)
    data = schema.canonical_json(plan) + b"\n"
    try:
        with destination.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if destination.is_symlink() or schema.canonical_json(load_plan(destination)) != schema.canonical_json(plan):
            raise ValueError("destination has different contents or is a symlink")
    return destination


def _ready(candidate, models):
    if candidate["group"]["family"] == "H":
        return True
    entry = models.get(candidate["model_id"])
    if entry is None:
        return False
    _model(entry)
    if (entry["identity"] != _identity(candidate)
            or any(entry[k] != candidate[k] for k in ("manifest_sha256", "eligibility_sha256"))):
        raise ValueError("ready model does not match declared eligible candidate")
    return True


def _tasks_for(candidate, split, densities, repeats):
    learned = candidate["group"]["family"] == "RL"
    method = candidate["group"]["architecture"] if learned else f"H{candidate['C']}"
    fields = {k: candidate[k] for k in ("model_id", "candidate_id", "env_steps")} if learned else {}
    if learned:
        fields["training_line"] = candidate["group"]["training_line"]
    return [schema.make_task(method, density=density, av_fraction=fraction,
                             replicate=replicate, split=split, purpose="scientific", **fields)
            for density in densities for fraction in FRACTIONS for replicate in range(repeats)]


def _alias(task, label, *, fraction=None, proof=None):
    science = task["science"]
    alias = {"task_id": task["task_id"], "label": label,
             "condition": {"density": science["config"]["density"],
                           "fraction": science["config"]["av_fraction"] if fraction is None else fraction,
                           "split": science["split"], "replicate": science["replicate"]},
             "proof": {"kind": "identical_task", "task_id": task["task_id"]} if proof is None else proof}
    alias["alias_id"] = schema.digest(alias)
    return alias


def _baselines(split, densities, repeats, proof):
    if proof is not None:
        _proof(proof)
    tasks, aliases = [], []
    for density in densities:
        for replicate in range(repeats):
            for fraction in (FRACTIONS if proof is None else (1.0,)):
                task = schema.make_task("B1", density=density, av_fraction=fraction,
                                        replicate=replicate, split=split, purpose="scientific")
                tasks.append(task)
                for logical_fraction in ((fraction,) if proof is None else FRACTIONS):
                    aliases.append(_alias(task, "B1", fraction=logical_fraction, proof=proof))
    return tasks, aliases


def make_validation_plan(*, freeze, models, b1_fraction_proof=None):
    """Return (plan, coverage) for all ready E2 candidates, never a selection.

    The full expected catalog remains hashed inside plan.freeze even if no
    learned bundle is ready. Coverage is derivable metadata, not run completion.
    """
    _validate_freeze(freeze, completed=False)
    tasks, aliases = _baselines("validation", VALIDATION_DENSITIES, 20, b1_fraction_proof)
    missing = []
    for candidate in freeze["candidates"]:
        if not _ready(candidate, models):
            missing.append({"group": candidate["group"], "candidate_id": candidate["candidate_id"]})
            continue
        tasks.extend(_tasks_for(candidate, "validation", VALIDATION_DENSITIES, 20))
    plan = make_plan(tasks, models=models, aliases=aliases, phase="validation", freeze=freeze)
    missing_groups = {_group_key(item["group"]): item["group"] for item in missing}
    coverage = {"ready_inputs_complete": not missing,
                "missing_candidates": sorted(missing, key=_candidate_key),
                "missing_lines": [missing_groups[k] for k in sorted(missing_groups)],
                "expected_candidate_trajectories": len(freeze["candidates"]) * 240,
                "physical_tasks": len(tasks), "b1_logical_references": 240}
    return plan, coverage


def make_frozen_test_plan(*, freeze, models, b1_fraction_proof=None):
    """Full 54-cell x 30 union; aliases share tasks only within exact policies.

    Includes B1 references and all selected rules. Other control panels must be
    explicit tasks passed through make_plan; no automatic H1/original collapse.
    """
    _validate_freeze(freeze, completed=True)
    tasks, aliases = _baselines("fresh_test", schema.DENSITY_GRID, 30, b1_fraction_proof)
    catalog = {_candidate_key(c): c for c in freeze["candidates"]}
    policies = {}
    for row in freeze["receipt"]["selection_manifest"]:
        if row["status"] != "selected":
            continue
        key = (_group_key(row["group"]), row["candidate_id"])
        candidate = catalog[key]
        if not _ready(candidate, models):
            raise ValueError("selected test model is not ready; no incomplete test union")
        if key not in policies:
            policies[key] = _tasks_for(candidate, "fresh_test", schema.DENSITY_GRID, 30)
            tasks.extend(policies[key])
        label = _group_key(row["group"]) + ":" + row["rule"]
        aliases.extend(_alias(task, label) for task in policies[key])
    return make_plan(tasks, models=models, aliases=aliases, phase="fresh_test", freeze=freeze)
