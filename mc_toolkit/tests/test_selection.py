"""Selector regression cases plus JSON/freeze transport tests.

The selector cases are ported from the tested E2 owner suite.  All tables are
complete 13x240 or 4x240 synthetic rows generated in memory: no simulator,
weights, historical CSV, container or RNG state is involved.
"""

from copy import deepcopy
from dataclasses import replace
import math
import random

import pytest

from physica_mc.selection import (
    BaselineRow,
    Candidate,
    CONDITIONS,
    EXPECTED_POOL_SIZE,
    Group,
    GroupSelection,
    H_CANDIDATES,
    K13_ENV_STEPS,
    K13_ITERATIONS,
    Provenance,
    Selection,
    THRESHOLDS,
    ValidationKey,
    ValidationRow,
    baseline_row_from_json,
    baseline_row_to_json,
    candidate_from_json,
    candidate_to_json,
    freeze_receipt,
    group_from_json,
    group_selection_from_json,
    group_selection_to_json,
    group_to_json,
    provenance_from_json,
    provenance_to_json,
    select_group,
    select_group_json,
    threshold_rule,
    to_json,
    validation_key_from_json,
    validation_key_to_json,
    validation_row_from_json,
    validation_row_to_json,
    verify_freeze_receipt,
)

PROVENANCE = Provenance("synthetic-dynamics", "synthetic-runtime", "new-validation-bank")
H = Group("H")


def fixture(group=H):
    candidates = (
        [Candidate(f"H{c}", group, C=c) for c in (1, 2, 4, 8)]
        if group.family == "H" else
        [Candidate(f"{group.training_line}-{i:02}", group, env_steps=i * 16384) for i in range(1, 14)]
    )
    baseline = []
    for rho, fraction in CONDITIONS:
        for rep in range(20):
            key = ValidationKey("validation", rho, fraction, rep, f"block-{rho}-{rep}", f"streams-{rho}-{rep}")
            baseline.append(BaselineRow(key, PROVENANCE, 1.0))
    rows = [ValidationRow(p.candidate_id, group, b.key, PROVENANCE, float(i + 1), 1.0)
            for i, p in enumerate(candidates) for b in baseline]
    return candidates, rows, baseline


def select(data, group=H):
    return select_group(group, *data, provenance=PROVENANCE)


def test_authoritative_candidate_pool_constants():
    assert H_CANDIDATES == (1, 2, 4, 8)
    assert len(K13_ITERATIONS) == 13
    assert K13_ITERATIONS[:3] == (100, 200, 300)
    assert K13_ITERATIONS[-2:] == (1200, 1221)
    assert K13_ENV_STEPS == tuple(iteration * 16384 for iteration in K13_ITERATIONS)
    assert K13_ENV_STEPS[-1] == 20004864
    assert EXPECTED_POOL_SIZE == {"H": 4, "RL": 13}
    assert [threshold_rule(tau) for tau in THRESHOLDS] == ["S_0.93", "S_0.95", "S_0.97", "S_0.99"]


def test_threshold_reselection_all_cells_and_exact_inclusive_boundary():
    pool, rows, baseline = fixture()
    ratios = {"H1": .93, "H2": .95, "H4": .97, "H8": .99}
    # Fastest H1, then H2/H4/H8. Each tau genuinely reselects from the pool.
    rows = [replace(r, pv_speed=5 - next(i for i, p in enumerate(pool) if p.candidate_id == r.candidate_id),
                    q_bg=ratios[r.candidate_id]) for r in rows]
    result = select((pool, rows, baseline))
    assert result.status == "complete"
    assert result.s0 is None  # No new speed-selected heuristic rule.
    assert [result.thresholds[t].selected for t in THRESHOLDS] == ["H1", "H2", "H4", "H8"]
    # One bad cell suffices; pooling 11 passing cells cannot hide it.
    rows = [replace(r, q_bg=.94) if r.candidate_id == "H2" and (r.key.density, r.key.fraction) == CONDITIONS[-1] else r for r in rows]
    assert select((pool, rows, baseline)).thresholds[.95].selected == "H4"


def test_mean_of_paired_ratios_not_ratio_of_means_and_shuffle_invariance():
    pool, rows, baseline = fixture()
    baseline = [replace(b, q_bg=1.0 if b.key.replicate < 10 else 9.0) for b in baseline]
    # mean(2/1, 4/9) = 11/9; ratio of means = 3/5.
    rows = [replace(r, q_bg=2.0 if r.key.replicate < 10 else 4.0,
                    pv_speed=r.pv_speed + r.key.density + r.key.replicate / 100.) for r in rows]
    expected = select((pool, rows, baseline))
    assert expected.ratios["H1"][CONDITIONS[0]] == math.fsum([2.0] * 10 + [4 / 9] * 10) / 20
    assert expected.speed["H8"] == math.fsum(
        math.fsum(4. + rho + rep / 100. for rep in range(20)) / 20
        for rho, fraction in CONDITIONS
    ) / 12
    assert expected.thresholds[.99].selected == "H8"
    rng = random.Random(44)
    rng.shuffle(pool)
    rng.shuffle(rows)
    rng.shuffle(baseline)
    assert select((pool, rows, baseline)) == expected


def test_exact_H_ties_and_full_precision_not_rendered_ties():
    pool, rows, baseline = fixture()
    rows = [replace(r, pv_speed=1.0) for r in rows]
    assert select((pool, rows, baseline)).thresholds[.95].selected == "H1"
    rows = [replace(r, pv_speed=math.nextafter(1.0, math.inf)) if r.candidate_id == "H8" else r for r in rows]
    result = select((pool, rows, baseline))
    assert result.speed["H8"] > result.speed["H1"]
    assert result.thresholds[.95].selected == "H8"


def test_RL_exact_ties_steps_then_ID_and_near_ties():
    group = Group("RL", "graph_ppo", "line-a")
    pool, rows, baseline = fixture(group)
    # Two stable IDs may refer to the same step boundary; ID is secondary.
    pool[1] = replace(pool[1], env_steps=pool[0].env_steps)
    rows = [replace(r, pv_speed=2.0) for r in rows]
    result = select((pool, rows, baseline), group)
    assert result.s0.selected == pool[0].candidate_id
    rows = [replace(r, pv_speed=2.000000000000001) if r.candidate_id == pool[-1].candidate_id else r for r in rows]
    assert select((pool, rows, baseline), group).s0.selected == pool[-1].candidate_id


def test_three_independent_training_lines_k_of_three_not_best_seed():
    results = []
    for line, offset, flow in (("line-a", 0., 1.), ("line-b", 100., .96), ("line-c", 200., .90)):
        group = Group("RL", "graph_ppo", line)
        pool, rows, baseline = fixture(group)
        # Different within-line winners, even though line-c has highest speeds.
        winner = pool[len(results)].candidate_id
        rows = [replace(r, pv_speed=offset + (20. if r.candidate_id == winner else 1.), q_bg=flow) for r in rows]
        result = select((pool, rows, baseline), group)
        assert result.s0.selected == winner
        results.append(result)
    assert sum(r.thresholds[.95].status == "selected" for r in results) == 2
    assert results[2].thresholds[.95].selected is None
    assert all(r.s0.status == "selected" for r in results)
    assert sum(r.thresholds[.99].status == "selected" for r in results) == 1


def test_missing_pool_candidate_rows_and_B1_are_incomplete_not_infeasible():
    pool, rows, baseline = fixture()
    for data in ((pool, rows[:-1], baseline),
                 (pool[:-1], [r for r in rows if r.candidate_id != "H8"], baseline),
                 (pool, rows, baseline[:-1])):
        result = select(data)
        assert result.status == "incomplete"
        assert all(s.status == "incomplete" and s.selected is None for s in result.thresholds.values())
    group = Group("RL", "node_mlp_ppo", "missing-line")
    pool, rows, baseline = fixture(group)
    result = select((pool, [], baseline), group)
    assert result.s0.status == "incomplete"


def test_missing_B1_preserves_complete_RL_speed_selection():
    group = Group("RL", "graph_ppo", "line-a")
    pool, rows, baseline = fixture(group)
    result = select((pool, rows, baseline[:-1]), group)
    assert result.status == "incomplete"
    assert result.s0.status == "selected"
    assert result.s0.selected == pool[-1].candidate_id
    assert result.speed[pool[-1].candidate_id] == 13.
    assert all(s.status == "incomplete" and s.selected is None for s in result.thresholds.values())
    # Candidate coverage, unlike B1 coverage, remains necessary for S0.
    assert select((pool, rows[:-1], baseline), group).s0.status == "incomplete"


def test_coordinated_fraction_identity_change_and_cross_block_collision_rejected():
    pool, rows, baseline = fixture()

    def change_fraction(key):
        if key.density == .1 and key.replicate == 0 and key.fraction == .95:
            return replace(key, pairing_id="other-pair", stream_id="other-stream")
        return key

    # Candidate and B1 still match each other cell-wise: cross-fraction identity
    # must independently reject this coordinated change.
    with pytest.raises(ValueError, match="common across fractions"):
        select((pool, [replace(r, key=change_fraction(r.key)) for r in rows],
                [replace(b, key=change_fraction(b.key)) for b in baseline]))

    for field, token in (("pairing_id", "block-0.1-0"), ("stream_id", "streams-0.1-0")):
        def collide(key):
            return replace(key, **{field: token}) if key.density == .1 and key.replicate == 1 else key

        with pytest.raises(ValueError, match="reused across blocks"):
            select((pool, [replace(r, key=collide(r.key)) for r in rows],
                    [replace(b, key=collide(b.key)) for b in baseline]))


def test_same_physical_B1_reference_requires_identical_flow():
    pool, rows, baseline = fixture()
    baseline[0] = replace(baseline[0], q_bg=.5)
    with pytest.raises(ValueError, match="conflicting B1 flow"):
        select((pool, rows, baseline))


def test_complete_none_feasible_is_infeasible_without_fallback():
    pool, rows, baseline = fixture()
    result = select((pool, [replace(r, q_bg=.90) for r in rows], baseline))
    assert result.status == "complete"
    assert all(s.status == "infeasible" and s.selected is None for s in result.thresholds.values())


def test_zero_B1_undefined_constraints_preserves_valid_S0():
    group = Group("RL", "grid_cnn_ppo", "line-a")
    pool, rows, baseline = fixture(group)
    baseline = [replace(b, q_bg=0.) if b.key.density == .1 and b.key.replicate == 0 else b for b in baseline]
    result = select((pool, rows, baseline), group)
    assert result.status == "constraint_undefined"
    assert result.s0.selected == pool[-1].candidate_id
    assert result.speed[pool[-1].candidate_id] == 13.
    assert result.ratios[pool[0].candidate_id][CONDITIONS[0]] is None
    assert all(s.status == "constraint_undefined" and s.selected is None for s in result.thresholds.values())


@pytest.mark.parametrize("bad", [-1., math.nan, math.inf, -math.inf, True])
def test_invalid_baseline_input_is_not_zero_or_infeasibility(bad):
    pool, rows, baseline = fixture()
    baseline[0] = replace(baseline[0], q_bg=bad)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        select((pool, rows, baseline))


def test_duplicate_extra_wrong_line_split_pairing_and_provenance_rejected():
    pool, rows, baseline = fixture()
    bad_keys = [replace(rows[0].key, split="fresh_test"),
                replace(rows[0].key, split="historical_E1_test"),
                replace(rows[0].key, density=.05),
                replace(rows[0].key, replicate=20),
                replace(rows[0].key, pairing_id="wrong-block"),
                replace(rows[0].key, stream_id="wrong-stream")]
    bad_rows = [replace(rows[0], key=k) for k in bad_keys] + [
        replace(rows[0], group=Group("RL", "graph_ppo", "other-line")),
        replace(rows[0], candidate_id="unregistered"),
        replace(rows[0], provenance=replace(PROVENANCE, runtime="wrong-runtime")),
        replace(rows[0], pv_speed=math.nan),
        replace(rows[0], q_bg=-1.),
    ]
    for bad_row in bad_rows:
        with pytest.raises(ValueError):
            select((pool, [bad_row] + rows[1:], baseline))
    for data in ((pool, rows + rows[:1], baseline),
                 (pool, rows, baseline + baseline[:1]),
                 (pool + pool[:1], rows, baseline),
                 (pool, rows, [replace(baseline[0], provenance=replace(PROVENANCE, scientific_version="other"))] + baseline[1:])):
        with pytest.raises(ValueError):
            select(data)


def test_extra_RL_pool_and_invalid_candidate_metadata_rejected():
    group = Group("RL", "graph_ppo", "line-a")
    pool, rows, baseline = fixture(group)
    with pytest.raises(ValueError, match="extra candidate"):
        select((pool + [Candidate("extra", group, env_steps=999)], rows, baseline), group)
    with pytest.raises(ValueError, match="checkpoint"):
        select(([replace(pool[0], env_steps=True)] + pool[1:], rows, baseline), group)
    hpool, hrows, hbaseline = fixture()
    with pytest.raises(ValueError, match="H candidate"):
        select(([replace(hpool[0], C=1.0)] + hpool[1:], hrows, hbaseline))


# ---------------------------------------------------------------------------
# JSON transport and freeze receipt
# ---------------------------------------------------------------------------

def test_json_round_trip_and_json_entry_point():
    group = Group("RL", "graph_ppo", "line-a")
    pool, rows, baseline = fixture(group)
    result = select((pool, rows, baseline), group)
    assert group_selection_from_json(group_selection_to_json(result)) == result
    for candidate in pool:
        assert candidate_from_json(candidate_to_json(candidate)) == candidate
    for row in rows[::97]:
        assert validation_row_from_json(validation_row_to_json(row)) == row
        assert baseline_row_from_json(baseline_row_to_json(baseline[0])) == baseline[0]
    assert validation_key_from_json(validation_key_to_json(rows[0].key)) == rows[0].key
    assert group_from_json(group_to_json(group)) == group
    assert provenance_from_json(provenance_to_json(PROVENANCE)) == PROVENANCE
    payload = select_group_json(
        group=group_to_json(group),
        candidates=[candidate_to_json(candidate) for candidate in pool],
        rows=[validation_row_to_json(row) for row in rows],
        baselines=[baseline_row_to_json(row) for row in baseline],
        provenance=provenance_to_json(PROVENANCE),
    )
    assert payload == group_selection_to_json(result)
    assert "NaN" not in to_json(payload)
    assert payload["s0"]["selected"] == result.s0.selected


def test_incomplete_selection_json_round_trip():
    group = Group("RL", "graph_ppo", "line-a")
    pool, rows, baseline = fixture(group)
    result = select((pool, rows[:-1], baseline), group)
    assert result.status == "incomplete"
    assert group_selection_from_json(group_selection_to_json(result)) == result


def test_json_transport_rejects_unknown_missing_nonfinite_and_bool_values():
    group = Group("RL", "graph_ppo", "line-a")
    pool, rows, baseline = fixture(group)
    result = select((pool, rows, baseline), group)

    candidate = candidate_to_json(pool[0])
    candidate["extra"] = 1
    with pytest.raises(ValueError, match="exactly"):
        candidate_from_json(candidate)
    candidate = candidate_to_json(pool[0])
    candidate["env_steps"] = True
    with pytest.raises(ValueError, match="integer"):
        candidate_from_json(candidate)
    row = validation_row_to_json(rows[0])
    row["q_bg"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        validation_row_from_json(row)
    key = validation_key_to_json(rows[0].key)
    key.pop("stream_id")
    with pytest.raises(ValueError, match="exactly"):
        validation_key_from_json(key)
    payload = group_selection_to_json(result)
    payload["thresholds"] = payload["thresholds"][:1]
    with pytest.raises(ValueError, match="exactly"):
        group_selection_from_json(payload)
    payload = group_selection_to_json(result)
    payload["ratios"][pool[0].candidate_id][1]["value"] = True
    with pytest.raises(ValueError, match="number"):
        group_selection_from_json(payload)


def frozen_case():
    groups = [Group("H"), Group("RL", "graph_ppo", "line-a"), Group("RL", "node_mlp_ppo", "line-b")]
    selections, pools = {}, {}
    for group in groups:
        pool, rows, baseline = fixture(group)
        # Exact full-precision ties make every rule select the first canonical
        # candidate, so several labels alias one policy.
        rows = [replace(row, pv_speed=1.0, q_bg=1.0) for row in rows]
        selections[group] = select((pool, rows, baseline), group)
        pools[group] = pool
    return groups, selections, pools


def test_freeze_receipt_is_complete_deterministic_and_verifiable():
    groups, selections, pools = frozen_case()
    receipt = freeze_receipt(selections, pools, provenance=PROVENANCE, expected_groups=groups)
    verify_freeze_receipt(receipt)
    assert receipt["frozen"] is True
    assert receipt["selection_split"] == "validation"
    assert receipt["test_outputs_accepted"] is False
    assert receipt["coverage"] == {
        "expected_groups": 3,
        "frozen_groups": 3,
        "RL_groups": 2,
        "H_groups": 1,
        "rules": {
            "H_0.93": 1, "H_0.95": 1, "H_0.97": 1, "H_0.99": 1,
            "S0": 2, "S_0.93": 2, "S_0.95": 2, "S_0.97": 2, "S_0.99": 2,
        },
    }
    assert len(receipt["selection_manifest"]) == 4 + 2 * 5
    assert len(receipt["policy_aliases"]) == 3
    h_alias = receipt["policy_aliases"][0]
    assert h_alias["group"] == group_to_json(Group("H"))
    assert h_alias["rules"] == ["H_0.93", "H_0.95", "H_0.97", "H_0.99"]
    for alias in receipt["policy_aliases"][1:]:
        assert alias["rules"] == ["S0", "S_0.93", "S_0.95", "S_0.97", "S_0.99"]
    assert all(len(group["pool"]) in (4, 13) for group in receipt["groups"])
    assert freeze_receipt(selections, pools, provenance=PROVENANCE, expected_groups=groups) == receipt


def test_freeze_receipt_detects_tampering_and_missing_digest():
    groups, selections, pools = frozen_case()
    receipt = freeze_receipt(selections, pools, provenance=PROVENANCE, expected_groups=groups)
    tampered = deepcopy(receipt)
    tampered["selection_manifest"][0]["candidate_id"] = "not-selected"
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_freeze_receipt(tampered)
    tampered = deepcopy(receipt)
    tampered["frozen"] = False
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_freeze_receipt(tampered)
    tampered = deepcopy(receipt)
    tampered.pop("receipt_sha256")
    with pytest.raises(ValueError, match="receipt_sha256"):
        verify_freeze_receipt(tampered)
    assert to_json(receipt).count("receipt_sha256") == 1


def test_freeze_refuses_missing_groups_pools_rules_and_incomplete_selection():
    groups, selections, pools = frozen_case()
    with pytest.raises(ValueError, match="exactly expected_groups"):
        freeze_receipt(selections, pools, provenance=PROVENANCE, expected_groups=groups[:-1])
    with pytest.raises(ValueError, match="duplicate expected group"):
        freeze_receipt(selections, pools, provenance=PROVENANCE, expected_groups=groups + groups[:1])
    with pytest.raises(ValueError, match="exactly expected_groups"):
        freeze_receipt({groups[0]: selections[groups[0]]}, pools, provenance=PROVENANCE, expected_groups=groups)

    incomplete_pools = dict(pools)
    incomplete_pools[groups[1]] = pools[groups[1]][:-1]
    with pytest.raises(ValueError, match="complete expected candidate pool"):
        freeze_receipt(selections, incomplete_pools, provenance=PROVENANCE, expected_groups=groups)

    pool, rows, baseline = fixture(groups[1])
    incomplete = select((pool, rows[:-1], baseline), groups[1])
    incomplete_selections = dict(selections)
    incomplete_selections[groups[1]] = incomplete
    with pytest.raises(ValueError, match="undetermined/incomplete"):
        freeze_receipt(incomplete_selections, pools, provenance=PROVENANCE, expected_groups=groups)

    missing_rule = dict(selections)
    missing_rule[groups[1]] = replace(
        selections[groups[1]],
        thresholds={tau: value for tau, value in selections[groups[1]].thresholds.items() if tau != .99},
    )
    with pytest.raises(ValueError, match="all four"):
        freeze_receipt(missing_rule, pools, provenance=PROVENANCE, expected_groups=groups)

    h_with_s0 = dict(selections)
    h_with_s0[groups[0]] = replace(
        selections[groups[0]], s0=Selection("selected", "H1", ("H1",)),
    )
    with pytest.raises(ValueError, match="H has no S0"):
        freeze_receipt(h_with_s0, pools, provenance=PROVENANCE, expected_groups=groups)

    bad_pool_group = dict(pools)
    bad_pool_group[groups[1]] = [
        replace(candidate, group=Group("RL", "graph_ppo", "other-line")) for candidate in pools[groups[1]]
    ]
    with pytest.raises(ValueError, match="wrong group"):
        freeze_receipt(selections, bad_pool_group, provenance=PROVENANCE, expected_groups=groups)
