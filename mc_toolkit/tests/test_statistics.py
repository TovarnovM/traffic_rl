"""Crossed-bootstrap tests: small explicit technical fixtures, no campaign data.

All fixtures hold already-reduced scalar outcomes.  Bootstrap sizes are
deliberately tiny (5-11 resamples) and exist only to test exact stream/design
semantics; the frozen SAP default remains 49 999 and is never executed here.
"""

import hashlib
import math
import random

import numpy as np
import pytest

from physica_mc import selection as selection_layer
from physica_mc import statistics as stats


def grid(densities=(0.2,), fractions=(0.95,), n_env=4):
    return stats.GridSpec(tuple(densities), tuple(fractions), n_env)


def fixed_side(label, spec, fn):
    return stats.FixedSide(label, {
        cell: np.asarray([fn(cell, env) for env in range(spec.n_env)], dtype=np.float64)
        for cell in spec.cells
    })


def line_side(architecture, rule, line_ids, spec, fn):
    return stats.LineSide(architecture, rule, tuple(line_ids), {
        cell: np.asarray(
            [[fn(cell, index, env) for env in range(spec.n_env)] for index in range(len(line_ids))],
            dtype=np.float64,
        )
        for cell in spec.cells
    })


def collect(chunks):
    return np.concatenate(tuple(chunks), axis=0)


def test_source_sap_identities_and_keyed_seed():
    assert stats.BOOTSTRAP_B == 49999
    assert stats.CONFIDENCE == 0.95
    assert stats.EXPECTED_NUMPY_VERSION == "2.1.2"
    assert stats.MASTER_SEED == 2026102401
    assert stats.QUANTILE_METHOD == "linear"
    assert stats.SOURCE_SAP["resamples"] == 49999
    assert stats.SOURCE_SAP["selection_resampled"] is False
    assert stats.SOURCE_SAP["simultaneous_or_multiplicity_controlled"] is False
    assert stats.environment_identity(0.05) == "env|rho=0.05"
    assert stats.environment_identity(0.1) == "env|rho=0.10"
    assert stats.environment_identity(0.9) == "env|rho=0.90"
    with pytest.raises(ValueError, match="two-decimal"):
        stats.environment_identity(0.025)
    assert stats.training_identity("S0", "graph_ppo") == "train|rule=S0|architecture=graph_ppo"
    assert stats.training_identity("S_0.95", "node_mlp_ppo") == "train|rule=S_0.95|architecture=node_mlp_ppo"
    for identity in (
        "env|rho=0.05",
        "env|rho=0.90",
        "train|rule=S0|architecture=graph_ppo",
        "train|rule=S_0.95|architecture=grid_cnn_ppo",
    ):
        expected = int.from_bytes(
            hashlib.sha256(f"2026102401|{identity}".encode("utf-8")).digest()[:8], "little",
        )
        assert stats.stream_seed(2026102401, identity) == expected
    with pytest.raises(ValueError):
        stats.training_identity("S_0.96", "graph_ppo")
    with pytest.raises(ValueError):
        stats.training_identity("S0", "other_arch")


def test_rule_labels_match_selector_rule_labels():
    assert stats.RULES == ("S0",) + tuple(selection_layer.threshold_rule(tau) for tau in selection_layer.THRESHOLDS)
    for tau in selection_layer.THRESHOLDS:
        assert stats.rule_label(tau) == selection_layer.threshold_rule(tau)


def test_mean_and_sample_sd_are_undefined_below_two():
    assert stats.mean_sample_sd([]) == (None, None)
    mean, sample_sd = stats.mean_sample_sd([2.0])
    assert mean == 2.0 and sample_sd is None
    mean, sample_sd = stats.mean_sample_sd([1.0, 2.0, 3.0])
    assert mean == 2.0 and sample_sd == 1.0
    cell = stats.summarize_cell([2.0])
    assert cell.complete and cell.mean == 2.0 and cell.sample_sd is None and cell.n == 1
    cell = stats.summarize_cell([1.0, np.nan])
    assert not cell.complete and cell.mean is None and cell.sample_sd is None
    assert cell.n == 2 and cell.n_defined == 1


def test_equal_weight_grid_mean_requires_the_full_planned_grid():
    spec = stats.GridSpec((0.1, 0.2), (0.9, 0.95), 2)
    values = {cell: float(index) for index, cell in enumerate(spec.cells)}
    assert stats.equal_weight_grid_mean(values, spec) == math.fsum(values.values()) / 4
    missing = dict(values)
    missing.pop(spec.cells[0])
    assert stats.equal_weight_grid_mean(missing, spec) is None
    undefined = dict(values)
    undefined[spec.cells[1]] = None
    assert stats.equal_weight_grid_mean(undefined, spec) is None
    extra = dict(values)
    extra[(0.3, 0.95)] = 1.0
    with pytest.raises(ValueError, match="unplanned"):
        stats.equal_weight_grid_mean(extra, spec)


def test_paired_ratio_is_divide_then_average_with_zero_denominator_undefined():
    assert stats.paired_ratios([2.0, 4.0], [1.0, 9.0]) == (2.0, 4 / 9)
    assert stats.paired_ratios([3.0, 4.0], [0.0, 2.0]) == (None, 2.0)
    assert stats.mean_paired_ratio([2.0, 4.0], [1.0, 9.0]) == math.fsum([2.0, 4 / 9]) / 2
    assert stats.mean_paired_ratio([2.0, 4.0], [1.0, 9.0]) != (2.0 + 4.0) / (1.0 + 9.0)
    assert stats.mean_paired_ratio([3.0, 4.0], [0.0, 2.0]) is None
    assert stats.mean_paired_ratio([], []) is None
    for bad in (-1.0, math.inf, math.nan, True):
        with pytest.raises(ValueError):
            stats.paired_ratios([1.0], [bad])


def test_environment_streams_are_chunk_invariant_and_density_independent():
    common = dict(n_env=5, B=11)
    a = collect(stats.environment_index_chunks(0.2, chunk_size=1, **common))
    b = collect(stats.environment_index_chunks(0.2, chunk_size=3, **common))
    c = collect(stats.environment_index_chunks(0.2, chunk_size=100, **common))
    assert a.dtype == np.int64 and a.shape == (11, 5)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)
    assert ((a >= 0) & (a < 5)).all()
    np.testing.assert_array_equal(a, collect(stats.environment_index_chunks(0.2, chunk_size=2, **common)))
    other = collect(stats.environment_index_chunks(0.3, chunk_size=2, **common))
    assert not np.array_equal(a, other)
    # A different B is a prefix of the same stream, not a re-seeded stream.
    short = collect(stats.environment_index_chunks(0.2, n_env=5, B=4, chunk_size=3))
    np.testing.assert_array_equal(short, a[:4])


def test_training_streams_are_chunk_invariant_and_identity_specific():
    a = collect(stats.training_index_chunks("S0", "graph_ppo", 3, B=9, chunk_size=1))
    b = collect(stats.training_index_chunks("S0", "graph_ppo", 3, B=9, chunk_size=4))
    assert a.dtype == np.int64 and a.shape == (9, 3)
    np.testing.assert_array_equal(a, b)
    assert ((a >= 0) & (a < 3)).all()
    np.testing.assert_array_equal(a, collect(stats.training_index_chunks("S0", "graph_ppo", 3, B=9, chunk_size=100)))
    assert not np.array_equal(a, collect(stats.training_index_chunks("S0", "node_mlp_ppo", 3, B=9, chunk_size=4)))
    assert not np.array_equal(a, collect(stats.training_index_chunks("S_0.95", "graph_ppo", 3, B=9, chunk_size=4)))
    single = collect(stats.training_index_chunks("S_0.95", "grid_cnn_ppo", 1, B=9, chunk_size=2))
    assert single.shape == (9, 1) and not single.any()


def test_crossed_bootstrap_matches_independent_two_line_fixture():
    spec = grid(n_env=4)
    left_values = {(0.2, 0.95): np.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])}
    right_values = {(0.2, 0.95): np.asarray([10.0, 20.0, 30.0, 40.0])}
    left = stats.LineSide("graph_ppo", "S_0.95", ("a", "b"), left_values)
    right = stats.FixedSide("H_0.95", right_values)
    B = 5
    result = stats.crossed_bootstrap(
        [stats.Contrast("graph_095-H095", left, right)], grid=spec, B=B, chunk_size=2,
    )["graph_095-H095"]

    env = collect(stats.environment_index_chunks(0.2, 4, B=B, chunk_size=100))
    train = collect(stats.training_index_chunks("S_0.95", "graph_ppo", 2, B=B, chunk_size=100))
    expected = np.empty(B, dtype=np.float64)
    for replicate in range(B):
        drawn = [
            left_values[(0.2, 0.95)][train[replicate, row], env[replicate, column]]
            for row in range(2) for column in range(4)
        ]
        right_drawn = right_values[(0.2, 0.95)][env[replicate]]
        expected[replicate] = math.fsum(drawn) / 8.0 - math.fsum(right_drawn) / 4.0
    np.testing.assert_array_equal(result.replicates, expected)
    assert result.point == 4.5 - 25.0
    quantiles = np.quantile(expected, [0.025, 0.975], method="linear")
    assert result.ci_low == float(quantiles[0]) and result.ci_high == float(quantiles[1])
    assert result.status == "complete"
    assert result.n_lines_left == 2 and result.n_lines_right is None
    assert result.interval_kind == "crossed_approximate"
    assert result.n_cells_planned == result.n_cells_complete == 1
    assert result.cells[(0.2, 0.95)].left.n == 8
    assert result.cells[(0.2, 0.95)].right.n == 4
    assert result.numpy_version == np.__version__


def test_one_line_interval_is_environment_only_conditional():
    spec = grid(n_env=4)
    left_values = {(0.2, 0.95): np.asarray([[2.0, 4.0, 6.0, 8.0]])}
    right_values = {(0.2, 0.95): np.asarray([1.0, 2.0, 3.0, 4.0])}
    left = stats.LineSide("node_mlp_ppo", "S_0.95", ("only-line",), left_values)
    right = stats.FixedSide("H_0.95", right_values)
    B = 7
    result = stats.crossed_bootstrap(
        [stats.Contrast("node_095-H095", left, right)], grid=spec, B=B, chunk_size=3,
    )["node_095-H095"]
    env = collect(stats.environment_index_chunks(0.2, 4, B=B, chunk_size=100))
    train = collect(stats.training_index_chunks("S_0.95", "node_mlp_ppo", 1, B=B, chunk_size=100))
    assert not train.any()
    expected = np.asarray([
        math.fsum(left_values[(0.2, 0.95)][0, env[replicate]]) / 4.0
        - math.fsum(right_values[(0.2, 0.95)][env[replicate]]) / 4.0
        for replicate in range(B)
    ])
    np.testing.assert_array_equal(result.replicates, expected)
    assert result.n_lines_left == 1 and result.n_lines_right is None
    assert result.interval_kind == "environment_only_conditional_on_single_training_line"
    assert "environment-only" in result.interval_label
    assert "between-training variation is not estimated" in result.interval_label


def test_zero_line_contrast_is_undefined_not_missing_or_available_case():
    spec = grid(n_env=4)
    left = stats.LineSide("grid_cnn_ppo", "S_0.95", (), {})
    right = stats.FixedSide("H_0.95", {(0.2, 0.95): np.asarray([1.0, 2.0, 3.0, 4.0])})
    result = stats.crossed_bootstrap(
        [stats.Contrast("grid_095-H095", left, right)], grid=spec, B=5, chunk_size=2,
    )["grid_095-H095"]
    assert result.status == "undefined_no_training_lines"
    assert result.point is None and result.ci_low is None and result.ci_high is None
    assert result.replicates is None
    assert result.n_lines_left == 0 and result.n_lines_right is None
    assert result.interval_kind == "undefined"
    assert not result.cells[(0.2, 0.95)].complete
    right_line = stats.LineSide("node_mlp_ppo", "S0", (), {})
    with pytest.raises(ValueError, match="at most three"):
        stats.LineSide("node_mlp_ppo", "S0", ("a", "b", "c", "d"), {})


def test_fraction_aliases_share_one_environment_draw_per_density():
    def value(cell, index, env):
        return float(index * 10 + env)

    def fixed_value(cell, env):
        return float(env)

    one = stats.GridSpec((0.2,), (0.9,), 4)
    three = stats.GridSpec((0.2,), (0.9, 0.95, 1.0), 4)
    result_one = stats.crossed_bootstrap(
        [stats.Contrast("same", line_side("graph_ppo", "S0", ("a", "b"), one, value),
                        fixed_side("B1", one, fixed_value))],
        grid=one, B=9, chunk_size=2,
    )["same"]
    result_three = stats.crossed_bootstrap(
        [stats.Contrast("same", line_side("graph_ppo", "S0", ("a", "b"), three, value),
                        fixed_side("B1", three, fixed_value))],
        grid=three, B=9, chunk_size=2,
    )["same"]
    # Identical fraction-cell values and a d=1/8-grid statistic make the three
    # identical cell contributions average back exactly to the one-cell value.
    np.testing.assert_array_equal(result_one.replicates, result_three.replicates)
    estimates = {cell: summary.estimate for cell, summary in result_three.cells.items()}
    assert len(set(estimates.values())) == 1
    assert result_one.point == estimates[(0.2, 0.9)]
    # The fixed B1 comparator is a single controller, never three cloned lines.
    assert result_three.n_lines_right is None
    assert result_three.cells[(0.2, 0.9)].right.n == 4


def test_distinct_densities_have_independent_env_streams_and_stable_cells():
    spec = stats.GridSpec((0.1, 0.2, 0.3), (0.95,), 2)
    base = {0.1: 10.0, 0.2: 20.0, 0.3: 30.0}

    def make(offset):
        def value(cell, index, env):
            return base[cell[0]] + (offset if cell[0] == 0.1 else 0.0) + index + env
        return line_side("graph_ppo", "S0", ("a", "b"), spec, value)

    def fixed_value(cell, env):
        return float(env)

    first = stats.crossed_bootstrap(
        [stats.Contrast("c", make(0.0), fixed_side("B1", spec, fixed_value))],
        grid=spec, B=7, chunk_size=2,
    )["c"]
    second = stats.crossed_bootstrap(
        [stats.Contrast("c", make(3.0), fixed_side("B1", spec, fixed_value))],
        grid=spec, B=7, chunk_size=2,
    )["c"]
    assert second.cells[(0.2, 0.95)].estimate == first.cells[(0.2, 0.95)].estimate
    assert second.cells[(0.3, 0.95)].estimate == first.cells[(0.3, 0.95)].estimate
    assert second.cells[(0.1, 0.95)].estimate == first.cells[(0.1, 0.95)].estimate + 3.0
    assert second.point == first.point + 1.0  # +3 in one of three equal-weight cells
    assert not np.array_equal(
        collect(stats.environment_index_chunks(0.1, 2, B=7, chunk_size=2)),
        collect(stats.environment_index_chunks(0.2, 2, B=7, chunk_size=2)),
    )


def test_missing_planned_data_disables_full_grid_without_available_case():
    spec = stats.GridSpec((0.2,), (0.9, 0.95), 3)
    left_values = {
        (0.2, 0.9): np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        (0.2, 0.95): np.asarray([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]),
    }
    right_values = {
        (0.2, 0.9): np.asarray([1.0, 2.0, 3.0]),
        (0.2, 0.95): np.asarray([1.0, 2.0, 3.0]),
    }
    full = stats.crossed_bootstrap(
        [stats.Contrast("c", stats.LineSide("graph_ppo", "S0", ("a", "b"), left_values),
                        stats.FixedSide("H", right_values))],
        grid=spec, B=5, chunk_size=2,
    )["c"]
    assert full.status == "complete"
    assert full.n_cells_complete == 2

    missing = dict(left_values)
    missing.pop((0.2, 0.95))
    result = stats.crossed_bootstrap(
        [stats.Contrast("c", stats.LineSide("graph_ppo", "S0", ("a", "b"), missing),
                        stats.FixedSide("H", right_values))],
        grid=spec, B=5, chunk_size=2,
    )["c"]
    assert result.status == "missing_planned_cells"
    assert result.point is None and result.ci_low is None and result.ci_high is None
    assert result.replicates is None
    assert result.n_cells_complete == 1
    assert result.cells[(0.2, 0.9)].complete and result.cells[(0.2, 0.9)].estimate is not None
    assert not result.cells[(0.2, 0.95)].complete
    assert "no available-case" in result.interval_label

    nan_values = {cell: values.copy() for cell, values in left_values.items()}
    nan_values[(0.2, 0.95)][1, 2] = np.nan
    result = stats.crossed_bootstrap(
        [stats.Contrast("c", stats.LineSide("graph_ppo", "S0", ("a", "b"), nan_values),
                        stats.FixedSide("H", right_values))],
        grid=spec, B=5, chunk_size=2,
    )["c"]
    assert result.status == "missing_planned_cells"
    assert not result.cells[(0.2, 0.95)].complete


def test_zero_b1_ratio_is_undefined_but_speed_contrast_remains_complete():
    assert stats.paired_ratios([3.0, 4.0], [0.0, 2.0]) == (None, 2.0)
    assert stats.mean_paired_ratio([3.0, 4.0], [0.0, 2.0]) is None
    spec = stats.GridSpec((0.2,), (0.95,), 3)
    speed_left = stats.LineSide("grid_cnn_ppo", "S_0.95", ("line",), {
        (0.2, 0.95): np.asarray([[3.0, 6.0, 9.0]]),
    })
    speed_fixed = stats.FixedSide("H_0.95", {(0.2, 0.95): np.asarray([1.0, 2.0, 3.0])})
    speed = stats.crossed_bootstrap(
        [stats.Contrast("speed", speed_left, speed_fixed)], grid=spec, B=5, chunk_size=2,
    )["speed"]
    assert speed.status == "complete"
    # An undefined B1 ratio enters only its own contrast as NaN, not speeds.
    ratio_fixed = stats.FixedSide("B1_ratio", {(0.2, 0.95): np.asarray([np.nan, 2.0, 3.0])})
    ratio = stats.crossed_bootstrap(
        [stats.Contrast("ratio", speed_left, ratio_fixed)], grid=spec, B=5, chunk_size=2,
    )["ratio"]
    assert ratio.status == "missing_planned_cells"
    assert ratio.point is None


def test_shuffle_and_rechunk_invariance_and_shared_architecture_draws():
    spec = stats.GridSpec((0.1, 0.2), (0.9, 0.95), 3)

    def values(prefix):
        return {
            cell: {line: [prefix + index * 10 + env for env in range(spec.n_env)]
                   for index, line in enumerate(("a", "b", "c"))}
            for cell in spec.cells
        }

    graph = values(1.0)
    node = values(100.0)

    def side(architecture, rule, line_ids, table):
        return stats.LineSide(architecture, rule, tuple(line_ids), {
            cell: np.asarray([table[cell][line_id] for line_id in line_ids], dtype=np.float64)
            for cell in spec.cells
        })

    contrasts = [
        stats.Contrast("graph-node", side("graph_ppo", "S0", ("a", "b", "c"), graph),
                       side("node_mlp_ppo", "S0", ("a", "b", "c"), node)),
        stats.Contrast("graph-grid", side("graph_ppo", "S0", ("a", "b", "c"), graph),
                       side("grid_cnn_ppo", "S0", ("a", "b", "c"), node)),
    ]
    base = stats.crossed_bootstrap(contrasts, grid=spec, B=11, chunk_size=1)
    rechunk = stats.crossed_bootstrap(contrasts, grid=spec, B=11, chunk_size=4)
    for label in ("graph-node", "graph-grid"):
        np.testing.assert_array_equal(base[label].replicates, rechunk[label].replicates)
        assert base[label].point == rechunk[label].point
        assert base[label].ci_low == rechunk[label].ci_low
        assert base[label].ci_high == rechunk[label].ci_high

    permuted = [
        stats.Contrast("graph-node", side("graph_ppo", "S0", ("c", "a", "b"), graph),
                       side("node_mlp_ppo", "S0", ("c", "a", "b"), node)),
        stats.Contrast("graph-grid", side("graph_ppo", "S0", ("b", "c", "a"), graph),
                       side("grid_cnn_ppo", "S0", ("b", "c", "a"), node)),
    ]
    shuffled = stats.crossed_bootstrap(list(reversed(permuted)), grid=spec, B=11, chunk_size=3)
    for label in ("graph-node", "graph-grid"):
        np.testing.assert_array_equal(base[label].replicates, shuffled[label].replicates)
        assert base[label].point == shuffled[label].point
    # Graph S0 draws are keyed by rule+architecture and shared by both
    # contrasts: combining them consumes the stream exactly once per replicate.
    separate = {
        label: stats.crossed_bootstrap([contrast], grid=spec, B=11, chunk_size=3)[label]
        for label, contrast in {
            "graph-node": stats.Contrast("graph-node", side("graph_ppo", "S0", ("a", "b", "c"), graph),
                                         side("node_mlp_ppo", "S0", ("a", "b", "c"), node)),
            "graph-grid": stats.Contrast("graph-grid", side("graph_ppo", "S0", ("a", "b", "c"), graph),
                                         side("grid_cnn_ppo", "S0", ("a", "b", "c"), node)),
        }.items()
    }
    for label in ("graph-node", "graph-grid"):
        np.testing.assert_array_equal(base[label].replicates, separate[label].replicates)


def test_all_lines_retained_and_fixed_controller_not_expanded():
    spec = stats.GridSpec((0.2,), (0.95,), 3)
    left = line_side("graph_ppo", "S_0.95", ("line-b", "line-a", "line-c"), spec,
                     lambda cell, index, env: float(index + env))
    right = fixed_side("H_0.95", spec, lambda cell, env: float(env))
    result = stats.crossed_bootstrap(
        [stats.Contrast("c", left, right)], grid=spec, B=5, chunk_size=2,
    )["c"]
    assert result.n_lines_left == 3 and result.n_lines_right is None
    assert result.line_ids_left == ("line-a", "line-b", "line-c")
    assert result.line_ids_right is None
    assert result.cells[(0.2, 0.95)].left.n == 3 * 3
    assert result.cells[(0.2, 0.95)].right.n == 3
    assert result.interval_kind == "crossed_approximate"
    with pytest.raises(ValueError, match="at most three"):
        stats.LineSide("graph_ppo", "S_0.95", ("a", "b", "c", "d"), {})
    with pytest.raises(ValueError, match="unique"):
        stats.LineSide("graph_ppo", "S_0.95", ("a", "a"), {})
    with pytest.raises(ValueError, match="non-empty"):
        stats.FixedSide("", {})


def test_interval_and_report_labels_state_descriptive_pointwise_conditionality():
    spec = stats.GridSpec((0.2,), (0.95,), 3)
    left = line_side("graph_ppo", "S_0.95", ("a", "b"), spec, lambda cell, index, env: float(index + env))
    right = fixed_side("H_0.95", spec, lambda cell, env: float(env))
    result = stats.crossed_bootstrap(
        [stats.Contrast("c", left, right)], grid=spec, B=5, chunk_size=2,
    )["c"]
    assert "pointwise" in result.interval_label
    assert "descriptive" in result.interval_label
    assert "not simultaneous" in result.interval_label
    assert "not multiplicity-controlled" in result.interval_label
    assert "conditional" in result.interval_label
    assert "p-value" not in result.interval_label
    assert result.master_seed == stats.MASTER_SEED
    assert result.resamples == 5
    assert result.quantile_method == "linear"
    assert result.confidence == 0.95


def test_input_shape_missing_and_extra_cells_are_rejected():
    spec = stats.GridSpec((0.2,), (0.95,), 3)
    left = stats.LineSide("graph_ppo", "S0", ("a", "b"), {
        (0.2, 0.95): np.asarray([[1.0, 2.0, 3.0]]),
    })
    right = stats.FixedSide("H", {(0.2, 0.95): np.asarray([1.0, 2.0, 3.0])})
    with pytest.raises(ValueError, match="shape"):
        stats.crossed_bootstrap(
            [stats.Contrast("c", left, right)], grid=spec, B=5, chunk_size=2,
        )
    left = stats.LineSide("graph_ppo", "S0", ("a", "b"), {
        (0.2, 0.95): np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
    })
    with pytest.raises(ValueError, match="unplanned cells"):
        stats.crossed_bootstrap(
            [stats.Contrast("c", left, stats.FixedSide("H", {
                (0.2, 0.9): np.asarray([1.0, 2.0, 3.0]),
                (0.2, 0.95): np.asarray([1.0, 2.0, 3.0]),
            }))],
            grid=spec, B=5, chunk_size=2,
        )
    bad = stats.LineSide("graph_ppo", "S0", ("a", "b"), {
        (0.2, 0.95): np.asarray([[1.0, 2.0, np.inf], [1.0, 2.0, 3.0]]),
    })
    with pytest.raises(ValueError, match="infinite"):
        stats.crossed_bootstrap(
            [stats.Contrast("c", bad, right)], grid=spec, B=5, chunk_size=2,
        )
    with pytest.raises(ValueError, match="unique"):
        stats.crossed_bootstrap(
            [stats.Contrast("c", left, right), stats.Contrast("c", left, right)],
            grid=spec, B=5, chunk_size=2,
        )
    with pytest.raises(ValueError, match="confidence"):
        stats.crossed_bootstrap([stats.Contrast("c", left, right)], grid=spec, B=5, confidence=1.0)


def test_random_cell_and_contrast_order_does_not_change_results():
    spec = stats.GridSpec((0.1, 0.2), (0.9, 1.0), 3)
    data = {
        cell: np.asarray([[float(index * 10 + env + int(round(cell[0] * 10))) for env in range(3)] for index in range(2)])
        for cell in spec.cells
    }
    fixed = {cell: np.asarray([float(env) for env in range(3)]) for cell in spec.cells}

    def make(order_cells):
        left = stats.LineSide("node_mlp_ppo", "S_0.95", ("b", "a"), {
            cell: data[cell] for cell in order_cells
        })
        right = stats.FixedSide("H_0.95", {cell: fixed[cell] for cell in order_cells})
        return stats.Contrast("c", left, right)

    first = stats.crossed_bootstrap([make(spec.cells)], grid=spec, B=9, chunk_size=2)["c"]
    order = list(spec.cells)
    random.Random(7).shuffle(order)
    second = stats.crossed_bootstrap([make(order)], grid=spec, B=9, chunk_size=2)["c"]
    np.testing.assert_array_equal(first.replicates, second.replicates)
    assert first.point == second.point
    assert first.ci_low == second.ci_low and first.ci_high == second.ci_high
