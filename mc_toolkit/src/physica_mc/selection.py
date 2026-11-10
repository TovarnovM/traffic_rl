"""Isolated E2 selection layer for the Physica A Monte Carlo toolkit.

The grouping, validation-table typing, paired-ratio rule, exact-tie ordering and
status distinctions are a port of the tested E2 selection owner (the
``select_group`` block of ``analysis/e2.py``, lines 150-376) with unchanged
semantics.  Only the transport helpers are new: strict JSON conversion for the
selector inputs/outputs and an immutable freeze-receipt builder/verifier so a
CLI can expose selection without re-implementing scientific rules.

This module is pure standard library + NumPy.  It imports no simulator, model,
torch, Slurm or research-project code and never reads files.

Scientific contract (see ``docs/SELECTION_STATISTICS.md`` and protocol §10):

* one group is one RL training line (13 K13 checkpoints) or the H family
  (C in {1, 2, 4, 8}); a group never pools training seeds or architectures;
* validation is 4 densities x 3 fractions x 20 environment blocks;
* candidate objective V is the equal-weight mean of per-condition PV-speed
  means, summed with ``math.fsum`` in canonical order, unrounded binary64;
* per condition R is the mean of 20 per-road paired q_bg(policy)/q_bg(B1)
  ratios, not a ratio of means;
* S0 is the per-line speed argmax; S_tau requires R >= tau in all 12 cells;
  ties resolve exactly: smaller env-step checkpoint, then stable candidate ID,
  or C=1,2,4,8 for H;
* full matrix with no feasible candidate is ``infeasible``; missing/corrupt
  rows are ``incomplete``; q_bg(B1)==0 makes only the ratio rule
  ``constraint_undefined`` and leaves a complete speed table usable for S0.
  There is no best-seed choice, no fallback and no complete-case subset.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import math

import numpy as np


CONDITIONS = tuple((rho, fraction) for rho in (.1, .2, .3, .4) for fraction in (.9, .95, 1.0))
THRESHOLDS = (.93, .95, .97, .99)
H_CANDIDATES = (1, 2, 4, 8)
K13_BATCH_STEPS = 16384
K13_ITERATIONS = tuple(range(100, 1201, 100)) + (1221,)
K13_ENV_STEPS = tuple(iteration * K13_BATCH_STEPS for iteration in K13_ITERATIONS)
EXPECTED_POOL_SIZE = {"H": 4, "RL": 13}
VALIDATION_SPLIT = "validation"

SELECTION_STATUSES = frozenset({"selected", "infeasible", "incomplete", "constraint_undefined"})
GROUP_STATUSES = frozenset({"complete", "incomplete", "constraint_undefined"})
FREEZE_SCHEMA_VERSION = 1
FREEZE_KIND = "physica_mc_selection_freeze_v1"


@dataclass(frozen=True)
class Group:
    family: str  # "H" or "RL"; one architecture/training line, never a seed pool.
    architecture: str = ""
    training_line: str = ""


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    group: Group
    env_steps: int = 0
    C: int | None = None


@dataclass(frozen=True)
class Provenance:
    scientific_version: str
    runtime: str
    stream_bank: str


@dataclass(frozen=True)
class ValidationKey:
    split: str
    density: float
    fraction: float
    replicate: int
    pairing_id: str
    stream_id: str


@dataclass(frozen=True)
class ValidationRow:
    candidate_id: str
    group: Group
    key: ValidationKey
    provenance: Provenance
    pv_speed: float
    q_bg: float


@dataclass(frozen=True)
class BaselineRow:
    key: ValidationKey
    provenance: Provenance
    q_bg: float  # B1 only; not an arbitrary comparator policy.


@dataclass(frozen=True)
class Selection:
    status: str  # selected, infeasible, incomplete, constraint_undefined
    selected: str | None
    feasible: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroupSelection:
    status: str
    speed: dict[str, float]
    ratios: dict[str, dict[tuple[float, float], float | None]]
    s0: Selection | None
    thresholds: dict[float, Selection]


def _validate_group(group: Group) -> None:
    if group.family not in ("H", "RL"):
        raise ValueError("unknown family")
    if group.family == "RL" and (not group.architecture or not group.training_line):
        raise ValueError("RL requires explicit architecture and training line")
    if group.family == "H" and (group.architecture or group.training_line):
        raise ValueError("H has no architecture/training line")


def _rank(group: Group, candidate: Candidate):
    return (candidate.C,) if group.family == "H" else (candidate.env_steps, candidate.candidate_id)


def _normalise_candidates(group: Group, expected_candidates: Iterable[Candidate]):
    """Validate one explicit candidate pool; return its canonical order."""
    candidates = tuple(expected_candidates)
    pool: dict[str, Candidate] = {}
    for candidate in candidates:
        if candidate.group != group or not isinstance(candidate.candidate_id, str) or not candidate.candidate_id:
            raise ValueError("wrong group or empty candidate ID")
        if candidate.candidate_id in pool:
            raise ValueError("duplicate candidate")
        if group.family == "H":
            if type(candidate.C) is not int or candidate.C not in H_CANDIDATES or candidate.env_steps != 0:
                raise ValueError("invalid H candidate")
        elif type(candidate.env_steps) is not int or candidate.env_steps <= 0 or candidate.C is not None:
            raise ValueError("invalid RL checkpoint")
        pool[candidate.candidate_id] = candidate
    expected_size = EXPECTED_POOL_SIZE[group.family]
    if len(pool) > expected_size:
        raise ValueError("extra candidate")
    if group.family == "H" and len({p.C for p in candidates}) != len(candidates):
        raise ValueError("duplicate C")
    order = tuple(sorted(pool, key=lambda candidate_id: _rank(group, pool[candidate_id])))
    return candidates, pool, expected_size, order


def select_group(
    group: Group, expected_candidates: Iterable[Candidate],
    rows: Iterable[ValidationRow], baselines: Iterable[BaselineRow],
    *, provenance: Provenance,
) -> GroupSelection:
    """Select from a synthetic validation table, independently for ONE group.

    The explicit expected pool is four C values or one line's K13 checkpoints;
    missing candidate rows give incomplete, never complete-case selection. A
    complete RL speed table still determines S0 when B1 is incomplete; only
    constrained selection then remains incomplete.
    Baselines are 240 explicit logical B1 references, including fractions; 80
    physical B1 runs are referenced three times with identical flow and common
    block-specific pairing_id/stream_id across fractions. Neither identity may
    be reused for another (density, replicate); stream_id is not stream_bank.
    Identity joins
    use condition/environment plus exact pairing/stream identity, not row order.
    Pairing identifies the common environment/warmed geometry/PV, NOT the full
    post-AV-tag state (B1 legitimately has different vehicle types). Provenance
    describes common dynamics/runtime/seed bank. Keys must be validation only.
    This validates equality of supplied identities, not their real-world truth.
    """
    _validate_group(group)
    if not all(isinstance(value, str) and value for value in (
        provenance.scientific_version, provenance.runtime, provenance.stream_bank
    )):
        raise ValueError("explicit provenance required")
    candidates, pool, expected_size, order = _normalise_candidates(group, expected_candidates)

    blocks = {}
    pairing_blocks = {}
    stream_blocks = {}

    def cell(key):
        if key.split != VALIDATION_SPLIT:
            raise ValueError("only validation split is allowed")
        if (isinstance(key.density, (bool, np.bool_)) or isinstance(key.fraction, (bool, np.bool_))
                or (key.density, key.fraction) not in CONDITIONS
                or type(key.replicate) is not int or not 0 <= key.replicate < 20):
            raise ValueError("extra condition/environment row")
        if not isinstance(key.pairing_id, str) or not key.pairing_id or not isinstance(key.stream_id, str) or not key.stream_id:
            raise ValueError("explicit pairing/stream identity required")
        block = key.density, key.replicate
        identity = key.pairing_id, key.stream_id
        if block in blocks and blocks[block] != identity:
            raise ValueError("pairing/stream identity must be common across fractions")
        for token, owners in ((key.pairing_id, pairing_blocks), (key.stream_id, stream_blocks)):
            if token in owners and owners[token] != block:
                raise ValueError("block-specific pairing/stream identity reused across blocks")
            owners[token] = block
        blocks[block] = identity
        return key.density, key.fraction, key.replicate

    def number(value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(value) or value < 0:
            raise ValueError("speed and flow must be finite and nonnegative")
        return float(value)

    baseline = {}
    baseline_flow = {}
    for row in baselines:
        k = cell(row.key)
        if row.provenance != provenance:
            raise ValueError("baseline provenance mismatch")
        flow = number(row.q_bg)
        block = row.key.density, row.key.replicate
        if block in baseline_flow and baseline_flow[block] != flow:
            raise ValueError("conflicting B1 flow for the same physical block reference")
        baseline_flow[block] = flow
        if k in baseline:
            raise ValueError("duplicate B1 row")
        baseline[k] = row
    table = {candidate_id: {} for candidate_id in pool}
    for row in rows:
        k = cell(row.key)
        if row.group != group or row.candidate_id not in pool:
            raise ValueError("wrong line/group or extra candidate row")
        if row.provenance != provenance:
            raise ValueError("candidate provenance mismatch")
        number(row.pv_speed)
        number(row.q_bg)
        if k in table[row.candidate_id]:
            raise ValueError("duplicate candidate row")
        if k in baseline and row.key != baseline[k].key:
            raise ValueError("B1 pairing/stream mismatch")
        table[row.candidate_id][k] = row
    speed_incomplete = len(pool) != expected_size or any(len(values) != 240 for values in table.values())
    if speed_incomplete:
        missing = Selection("incomplete", None)
        return GroupSelection("incomplete", {}, {}, missing if group.family == "RL" else None, {tau: missing for tau in THRESHOLDS})

    speed, ratios = {}, {}
    for candidate_id in order:
        values = table[candidate_id]
        speed[candidate_id] = math.fsum(
            math.fsum(float(values[rho, f, rep].pv_speed) for rep in range(20)) / 20
            for rho, f in CONDITIONS
        ) / 12

    def choose(feasible):
        if not feasible:
            return Selection("infeasible", None)
        # Stable order resolves only EXACT binary64 objective ties.
        return Selection("selected", max(feasible, key=speed.__getitem__), tuple(feasible))

    s0 = choose(order) if group.family == "RL" else None
    if len(baseline) != 240:
        missing = Selection("incomplete", None)
        return GroupSelection("incomplete", speed, {}, s0, {tau: missing for tau in THRESHOLDS})

    for candidate_id in order:
        values = table[candidate_id]
        ratios[candidate_id] = {}
        for rho, f in CONDITIONS:
            if any(baseline[rho, f, rep].q_bg == 0 for rep in range(20)):
                ratios[candidate_id][rho, f] = None
            else:
                quotients = [float(values[rho, f, rep].q_bg) / float(baseline[rho, f, rep].q_bg) for rep in range(20)]
                if not all(math.isfinite(q) for q in quotients):
                    raise ValueError("nonfinite paired ratio")
                ratios[candidate_id][rho, f] = math.fsum(quotients) / 20

    undefined = any(value is None for cells in ratios.values() for value in cells.values())
    thresholds = {}
    for tau in THRESHOLDS:
        thresholds[tau] = Selection("constraint_undefined", None) if undefined else choose([
            candidate_id for candidate_id in order
            if all(value >= tau for value in ratios[candidate_id].values())
        ])
    return GroupSelection("constraint_undefined" if undefined else "complete", speed, ratios, s0, thresholds)


# ---------------------------------------------------------------------------
# Strict JSON transport.  Every helper rejects unknown/missing keys and
# non-finite numbers instead of coercing transport errors into scientific ones.
# ---------------------------------------------------------------------------

def _object(value, what):
    if not isinstance(value, Mapping):
        raise ValueError(f"{what} must be a JSON object")
    return value


def _keys(value, expected, what):
    if set(value) != set(expected):
        raise ValueError(f"{what} must contain exactly {sorted(expected)}")


def _text(value, what):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a non-empty string")
    return value


def _optional_text(value, what):
    if value is None:
        return None
    return _text(value, what)


def _number(value, what):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{what} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{what} must be finite")
    return result


def _integer(value, what, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{what} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{what} must be >= {minimum}")
    return result


def _array(value, what):
    if not isinstance(value, list):
        raise ValueError(f"{what} must be a JSON array")
    return value


def _status(value, allowed, what):
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{what} must be one of {sorted(allowed)}")
    return value


def group_to_json(group: Group) -> dict:
    return {
        "family": group.family,
        "architecture": group.architecture,
        "training_line": group.training_line,
    }


def group_from_json(value) -> Group:
    value = _object(value, "group")
    _keys(value, ("family", "architecture", "training_line"), "group")
    for field in ("architecture", "training_line"):
        if not isinstance(value[field], str):
            raise ValueError(f"group.{field} must be a string (empty for H)")
    return Group(
        _text(value["family"], "group.family"),
        value["architecture"],
        value["training_line"],
    )


def provenance_to_json(provenance: Provenance) -> dict:
    return {
        "scientific_version": provenance.scientific_version,
        "runtime": provenance.runtime,
        "stream_bank": provenance.stream_bank,
    }


def provenance_from_json(value) -> Provenance:
    value = _object(value, "provenance")
    _keys(value, ("scientific_version", "runtime", "stream_bank"), "provenance")
    return Provenance(
        _text(value["scientific_version"], "provenance.scientific_version"),
        _text(value["runtime"], "provenance.runtime"),
        _text(value["stream_bank"], "provenance.stream_bank"),
    )


def candidate_to_json(candidate: Candidate) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "group": group_to_json(candidate.group),
        "env_steps": candidate.env_steps,
        "C": candidate.C,
    }


def candidate_from_json(value) -> Candidate:
    value = _object(value, "candidate")
    _keys(value, ("candidate_id", "group", "env_steps", "C"), "candidate")
    C = None if value["C"] is None else _integer(value["C"], "candidate.C")
    return Candidate(
        _text(value["candidate_id"], "candidate.candidate_id"),
        group_from_json(value["group"]),
        _integer(value["env_steps"], "candidate.env_steps", 0),
        C,
    )


def validation_key_to_json(key: ValidationKey) -> dict:
    return {
        "split": key.split,
        "density": key.density,
        "fraction": key.fraction,
        "replicate": key.replicate,
        "pairing_id": key.pairing_id,
        "stream_id": key.stream_id,
    }


def validation_key_from_json(value) -> ValidationKey:
    value = _object(value, "validation key")
    _keys(value, ("split", "density", "fraction", "replicate", "pairing_id", "stream_id"), "validation key")
    return ValidationKey(
        _text(value["split"], "key.split"),
        _number(value["density"], "key.density"),
        _number(value["fraction"], "key.fraction"),
        _integer(value["replicate"], "key.replicate", 0),
        _text(value["pairing_id"], "key.pairing_id"),
        _text(value["stream_id"], "key.stream_id"),
    )


def validation_row_to_json(row: ValidationRow) -> dict:
    return {
        "candidate_id": row.candidate_id,
        "group": group_to_json(row.group),
        "key": validation_key_to_json(row.key),
        "provenance": provenance_to_json(row.provenance),
        "pv_speed": row.pv_speed,
        "q_bg": row.q_bg,
    }


def validation_row_from_json(value) -> ValidationRow:
    value = _object(value, "validation row")
    _keys(value, ("candidate_id", "group", "key", "provenance", "pv_speed", "q_bg"), "validation row")
    return ValidationRow(
        _text(value["candidate_id"], "row.candidate_id"),
        group_from_json(value["group"]),
        validation_key_from_json(value["key"]),
        provenance_from_json(value["provenance"]),
        _number(value["pv_speed"], "row.pv_speed"),
        _number(value["q_bg"], "row.q_bg"),
    )


def baseline_row_to_json(row: BaselineRow) -> dict:
    return {
        "key": validation_key_to_json(row.key),
        "provenance": provenance_to_json(row.provenance),
        "q_bg": row.q_bg,
    }


def baseline_row_from_json(value) -> BaselineRow:
    value = _object(value, "baseline row")
    _keys(value, ("key", "provenance", "q_bg"), "baseline row")
    return BaselineRow(
        validation_key_from_json(value["key"]),
        provenance_from_json(value["provenance"]),
        _number(value["q_bg"], "baseline.q_bg"),
    )


def selection_to_json(selection: Selection) -> dict:
    return {
        "status": _status(selection.status, SELECTION_STATUSES, "selection.status"),
        "selected": _optional_text(selection.selected, "selection.selected"),
        "feasible": [_text(candidate_id, "selection.feasible entry") for candidate_id in selection.feasible],
    }


def selection_from_json(value) -> Selection:
    value = _object(value, "selection")
    _keys(value, ("status", "selected", "feasible"), "selection")
    feasible = tuple(_text(candidate_id, "selection.feasible entry")
                     for candidate_id in _array(value["feasible"], "selection.feasible"))
    if len(set(feasible)) != len(feasible):
        raise ValueError("selection.feasible must not repeat candidate IDs")
    return Selection(
        _status(value["status"], SELECTION_STATUSES, "selection.status"),
        _optional_text(value["selected"], "selection.selected"),
        feasible,
    )


def group_selection_to_json(selection: GroupSelection) -> dict:
    thresholds = selection.thresholds
    if thresholds and set(thresholds) != set(THRESHOLDS):
        raise ValueError("thresholds must be either empty or cover exactly THRESHOLDS")
    ratio_cells = {}
    for candidate_id, cells in selection.ratios.items():
        if cells and set(cells) != set(CONDITIONS):
            raise ValueError("ratios must be either empty or cover exactly the 12 conditions")
        ratio_cells[_text(candidate_id, "ratios candidate ID")] = [
            {
                "density": rho,
                "fraction": fraction,
                "value": None if cells[rho, fraction] is None else _number(cells[rho, fraction], "ratios value"),
            }
            for rho, fraction in CONDITIONS if (rho, fraction) in cells
        ]
    return {
        "status": _status(selection.status, GROUP_STATUSES, "group selection.status"),
        "speed": {
            _text(candidate_id, "speed candidate ID"): _number(value, "speed value")
            for candidate_id, value in selection.speed.items()
        },
        "ratios": ratio_cells,
        "s0": None if selection.s0 is None else selection_to_json(selection.s0),
        "thresholds": [
            {"tau": tau, "selection": selection_to_json(thresholds[tau])} for tau in THRESHOLDS
        ] if thresholds else [],
    }


def group_selection_from_json(value) -> GroupSelection:
    value = _object(value, "group selection")
    _keys(value, ("status", "speed", "ratios", "s0", "thresholds"), "group selection")
    speed = {}
    for candidate_id, candidate_value in _object(value["speed"], "group selection.speed").items():
        speed[_text(candidate_id, "speed candidate ID")] = _number(candidate_value, "speed value")
    ratios = {}
    for candidate_id, entries in _object(value["ratios"], "group selection.ratios").items():
        cells = {}
        for entry in _array(entries, "ratio entries"):
            entry = _object(entry, "ratio entry")
            _keys(entry, ("density", "fraction", "value"), "ratio entry")
            cell = (_number(entry["density"], "ratio density"), _number(entry["fraction"], "ratio fraction"))
            if cell not in CONDITIONS:
                raise ValueError("ratio cell is not one of the 12 validation conditions")
            if cell in cells:
                raise ValueError("duplicate ratio cell")
            cells[cell] = None if entry["value"] is None else _number(entry["value"], "ratio value")
        if cells and set(cells) != set(CONDITIONS):
            raise ValueError("ratio cells must be either empty or cover exactly the 12 conditions")
        ratios[_text(candidate_id, "ratios candidate ID")] = cells
    thresholds = {}
    for entry in _array(value["thresholds"], "group selection.thresholds"):
        entry = _object(entry, "threshold entry")
        _keys(entry, ("tau", "selection"), "threshold entry")
        tau = _number(entry["tau"], "threshold tau")
        if tau not in THRESHOLDS or tau in thresholds:
            raise ValueError("unexpected or duplicate threshold")
        thresholds[tau] = selection_from_json(entry["selection"])
    if thresholds and set(thresholds) != set(THRESHOLDS):
        raise ValueError("thresholds must be either empty or cover exactly THRESHOLDS")
    return GroupSelection(
        _status(value["status"], GROUP_STATUSES, "group selection.status"),
        speed,
        ratios,
        None if value["s0"] is None else selection_from_json(value["s0"]),
        thresholds,
    )


def select_group_json(
    *, group, candidates, rows, baselines, provenance,
) -> dict:
    """JSON-in/JSON-out wrapper for a CLI; no scientific rule lives here.

    ``group``, ``provenance`` and every array element use the object/array
    shapes produced by the ``*_to_json`` helpers above.
    """
    return group_selection_to_json(select_group(
        group_from_json(group),
        (candidate_from_json(item) for item in _array(candidates, "candidates")),
        (validation_row_from_json(item) for item in _array(rows, "rows")),
        (baseline_row_from_json(item) for item in _array(baselines, "baselines")),
        provenance=provenance_from_json(provenance),
    ))


# ---------------------------------------------------------------------------
# Immutable freeze receipt.  The helper consumes only frozen validation
# selections; it never reads test rows and cannot run the selector again.
# ---------------------------------------------------------------------------

def _canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def threshold_rule(tau: float) -> str:
    """Canonical S_tau rule label, e.g. ``0.95 -> "S_0.95"``."""
    if tau not in THRESHOLDS:
        raise ValueError("tau must be one of THRESHOLDS")
    return f"S_{tau:.2f}"


def _group_order(group: Group):
    return group.family, group.architecture, group.training_line


def _check_frozen_selection(group: Group, selection: GroupSelection, pool: Mapping[str, Candidate]) -> None:
    if selection.status not in ("complete", "constraint_undefined"):
        raise ValueError("selection is undetermined/incomplete; refusing to freeze")
    if set(selection.speed) != set(pool):
        raise ValueError("speed table must cover exactly the frozen candidate pool")
    for value in selection.speed.values():
        _number(value, "speed value")
    if set(selection.ratios) != set(pool):
        raise ValueError("ratio table must cover exactly the frozen candidate pool")
    undefined = False
    for cells in selection.ratios.values():
        if set(cells) != set(CONDITIONS):
            raise ValueError("ratio table must cover all 12 validation conditions")
        for value in cells.values():
            if value is None:
                undefined = True
            else:
                _number(value, "ratio value")
    if undefined != (selection.status == "constraint_undefined"):
        raise ValueError("group status must mark exactly the undefined ratio constraints")
    if set(selection.thresholds) != set(THRESHOLDS):
        raise ValueError("all four S_tau/H_tau rules are required for freeze")

    def check_choice(choice: Selection, *, undefined_allowed: bool) -> None:
        if choice.status not in ("selected", "infeasible", "constraint_undefined"):
            raise ValueError("selection rule is undetermined/incomplete; refusing to freeze")
        feasible = tuple(choice.feasible)
        if len(set(feasible)) != len(feasible) or any(candidate_id not in pool for candidate_id in feasible):
            raise ValueError("feasible set must contain unique frozen candidate IDs")
        canonical = tuple(sorted(feasible, key=lambda candidate_id: _rank(group, pool[candidate_id])))
        if feasible != canonical:
            raise ValueError("feasible set must follow canonical candidate order")
        if choice.status == "selected":
            if choice.selected not in pool or choice.selected not in feasible:
                raise ValueError("selected candidate must be a frozen feasible candidate")
        elif choice.status == "infeasible":
            if choice.selected is not None or feasible:
                raise ValueError("infeasible selection cannot name a candidate")
        else:
            if not undefined_allowed or choice.selected is not None or feasible:
                raise ValueError("constraint_undefined selection is inconsistent")

    if group.family == "RL":
        if selection.s0 is None:
            raise ValueError("RL freeze requires an S0 selection")
        check_choice(selection.s0, undefined_allowed=False)
    elif selection.s0 is not None:
        raise ValueError("H has no S0 speed-selected rule")
    for tau in THRESHOLDS:
        choice = selection.thresholds[tau]
        if undefined and choice.status != "constraint_undefined":
            raise ValueError("undefined ratio constraints must make every tau rule undefined")
        if not undefined and choice.status == "constraint_undefined":
            raise ValueError("constraint_undefined requires an undefined ratio cell")
        check_choice(choice, undefined_allowed=undefined)


def freeze_receipt(
    selections: Mapping[Group, GroupSelection],
    candidate_pools: Mapping[Group, Sequence[Candidate]],
    *,
    provenance: Provenance,
    expected_groups: Iterable[Group],
) -> dict:
    """Build a deterministic freeze receipt for an explicit, complete group set.

    ``expected_groups`` is caller-supplied on purpose: this layer must not infer
    architectures, training lines or source admission.  Every expected group
    must have its complete 4/13 candidate pool and a determined selection for
    S0 and all four S_tau (RL) or all four H_tau (H); otherwise the helper
    refuses to freeze.  The returned JSON-native receipt records the pools,
    full 12-cell ratio tables, objectives, statuses, a selection manifest, the
    labels that alias one candidate, per-group hashes and a whole-receipt
    SHA-256.  Only validation selections are accepted; no test output is an
    argument and the selector is not run again.
    """
    expected = tuple(expected_groups)
    if not expected:
        raise ValueError("expected_groups must be explicit and non-empty")
    if len(set(expected)) != len(expected):
        raise ValueError("duplicate expected group")
    if set(selections) != set(expected):
        raise ValueError("selections must cover exactly expected_groups")
    if set(candidate_pools) != set(expected):
        raise ValueError("candidate_pools must cover exactly expected_groups")
    if not all(isinstance(value, str) and value for value in (
        provenance.scientific_version, provenance.runtime, provenance.stream_bank
    )):
        raise ValueError("explicit provenance required")

    groups_payload = []
    manifest = []
    aliases = []
    rule_counts: dict[str, int] = {}
    for group in sorted(expected, key=_group_order):
        _validate_group(group)
        _, pool, expected_size, order = _normalise_candidates(group, candidate_pools[group])
        if len(pool) != expected_size:
            raise ValueError("freeze requires the complete expected candidate pool")
        selection = selections[group]
        _check_frozen_selection(group, selection, pool)

        selection_payload = group_selection_to_json(selection)
        pool_payload = [candidate_to_json(pool[candidate_id]) for candidate_id in order]
        groups_payload.append({
            "group": group_to_json(group),
            "pool": pool_payload,
            "pool_sha256": _digest(pool_payload),
            "selection": selection_payload,
            "selection_sha256": _digest(selection_payload),
        })

        rules = []
        if group.family == "RL":
            rules.append(("S0", selection.s0))
        for tau in THRESHOLDS:
            rules.append((
                threshold_rule(tau) if group.family == "RL" else f"H_{tau:.2f}",
                selection.thresholds[tau],
            ))
        per_candidate_rules: dict[str, list[str]] = {}
        for rule, choice in rules:
            candidate = pool[choice.selected] if choice.selected is not None else None
            manifest.append({
                "group": group_to_json(group),
                "rule": rule,
                "status": choice.status,
                "candidate_id": choice.selected,
                "C": None if candidate is None else candidate.C,
                "env_steps": None if candidate is None else candidate.env_steps,
            })
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
            if choice.selected is not None:
                per_candidate_rules.setdefault(choice.selected, []).append(rule)
        for candidate_id in order:
            if candidate_id in per_candidate_rules:
                candidate = pool[candidate_id]
                aliases.append({
                    "group": group_to_json(group),
                    "candidate_id": candidate_id,
                    "C": candidate.C,
                    "env_steps": candidate.env_steps,
                    "rules": per_candidate_rules[candidate_id],
                })

    receipt = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "kind": FREEZE_KIND,
        "frozen": True,
        "selection_split": VALIDATION_SPLIT,
        "test_outputs_accepted": False,
        "provenance": provenance_to_json(provenance),
        "groups": groups_payload,
        "selection_manifest": manifest,
        "policy_aliases": aliases,
        "coverage": {
            "expected_groups": len(expected),
            "frozen_groups": len(groups_payload),
            "RL_groups": sum(group.family == "RL" for group in expected),
            "H_groups": sum(group.family == "H" for group in expected),
            "rules": dict(sorted(rule_counts.items())),
        },
    }
    receipt["receipt_sha256"] = _digest(receipt)
    return receipt


def verify_freeze_receipt(receipt) -> None:
    """Recompute the receipt digest and reject a mutated or partial freeze."""
    receipt = _object(receipt, "freeze receipt")
    digest = _text(receipt.get("receipt_sha256"), "receipt.receipt_sha256")
    payload = dict(receipt)
    payload.pop("receipt_sha256")
    if not hmac.compare_digest(_digest(payload), digest):
        raise ValueError("freeze receipt digest mismatch")
    if receipt.get("schema_version") != FREEZE_SCHEMA_VERSION or receipt.get("kind") != FREEZE_KIND:
        raise ValueError("unsupported freeze receipt")
    if receipt.get("frozen") is not True or receipt.get("test_outputs_accepted") is not False:
        raise ValueError("freeze receipt is not an immutable validation-only freeze")
    if receipt.get("selection_split") != VALIDATION_SPLIT:
        raise ValueError("freeze receipt is not a validation selection")


def to_json(value) -> str:
    """Canonical UTF-8 JSON text for selector/freeze transport (no NaN/Inf)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
