"""Technical-only tiny B fixtures. Never run a simulator or campaign bootstrap."""
import math
import numpy as np
import pytest
from physica_mc import statistics as core
import inference


def fixture():
    grid = core.GridSpec((.1, .2), (.9, 1.), 5)
    a = core.LineSide('graph_ppo', 'S0', ('c', 'a', 'b'),
        {cell: np.arange(15, dtype=float).reshape(3, 5) + cell[0] for cell in grid.cells})
    b = core.LineSide('node_mlp_ppo', 'S0', ('z', 'x', 'y'),
        {cell: np.arange(15, dtype=float).reshape(3, 5)**2 / 17 for cell in grid.cells})
    return grid, core.Contrast('G-N', a, b)


def test_extended_adapter_exact_main_parity_and_chunk_invariance():
    grid, c = fixture()
    expected = core.crossed_bootstrap([c], grid=grid, B=17, chunk_size=4)[c.label]
    # Explicit scale mapping exercises extension implementation on main keys.
    for chunk in (1, 4, 19):
        actual = inference.bootstrap([c], grid=grid, scales={c.label: (1., 1.)},
                                     B=17, chunk_size=chunk, technical=True)[c.label]
        np.testing.assert_array_equal(actual.replicates, expected.replicates)
        assert actual.point == expected.point
        assert actual.ci_low == pytest.approx(expected.ci_low, abs=1e-14)
        assert actual.ci_high == pytest.approx(expected.ci_high, abs=1e-14)


def test_w7_and_length_keys_are_independent_without_fake_architectures():
    keys = [inference.training_key('S0', 'graph_ppo', s) for s in (.5, 1., 2.)]
    assert keys == ['train|rule=S0|architecture=graph_ppo|penalty_scale=0.5',
                    'train|rule=S0|architecture=graph_ppo',
                    'train|rule=S0|architecture=graph_ppo|penalty_scale=2.0']
    assert inference.environment_key(.2, 2000) == 'env|L=2000|rho=0.20'
    streams = [np.concatenate(list(inference.index_chunks(k, 3, B=11))) for k in keys]
    assert not np.array_equal(streams[0], streams[1])
    assert not np.array_equal(streams[1], streams[2])
    grid, c = fixture()
    # Same genuine architecture, different W7 line rosters and keys.
    right = core.LineSide('graph_ppo', 'S0', ('d', 'e', 'f'), c.right.values)
    c = core.Contrast('half-main', c.left, right)
    first = inference.bootstrap([c], grid=grid, scales={c.label: (.5, 1.)}, B=11,
                                chunk_size=2, technical=True)[c.label]
    second = inference.bootstrap([c], grid=grid, scales={c.label: (.5, 1.)}, B=11,
                                 chunk_size=7, technical=True)[c.label]
    np.testing.assert_array_equal(first.replicates, second.replicates)
    length = inference.bootstrap([c], grid=grid, road_length=2000,
                                 scales={c.label: (.5, 1.)}, B=11, technical=True)[c.label]
    assert not np.array_equal(length.replicates, first.replicates)


def test_missing_grid_is_na_and_final_settings_are_guarded():
    grid, c = fixture()
    left = core.LineSide('graph_ppo', 'S0', c.left.line_ids, {grid.cells[0]: c.left.values[grid.cells[0]]})
    c = core.Contrast('missing', left, c.right)
    result = inference.bootstrap([c], grid=grid, road_length=2000, B=7, technical=True)[c.label]
    assert result.point is result.ci_low is result.replicates is None
    assert result.status == 'missing_planned_cells'
    with pytest.raises(ValueError, match='final inference'):
        inference.bootstrap([c], grid=grid, B=7)


def test_pair_gate_missing_line_overlap_zero_and_mean_of_ratios():
    right = {i: i+1 for i in range(6)}
    left = {line: {i: 2 for i in range(5)} for line in ('a', 'b', 'c')}
    result = inference.paired_summary(left, right, line_ids=('a', 'b', 'c'), ratio=True)
    assert result['value'] == math.fsum(2/(i+1) for i in range(5))/5
    assert result['n'] == 5
    assert inference.paired_summary(left, right, line_ids=('a', 'b', 'missing'))['value'] is None
    left['c'] = {i: 2 for i in range(1, 6)}
    assert inference.paired_summary(left, right, line_ids=('a', 'b', 'c'))['n'] == 4
    left['c'] = left['a']
    right[0] = 0
    assert inference.paired_summary(left, right, line_ids=('a', 'b', 'c'), ratio=True)['status'] == 'undefined_denominator'


def test_all90_illustration_outer_line_rule_ties_and_partial_refusal():
    rows = [dict(training_line=line, replicate=r, environment=str(r), paired_pv_effect=float(r))
            for line in ('c', 'a', 'b') for r in range(30)]
    selected = inference.select_illustration(rows, line_ids=('a', 'b', 'c'))
    assert selected['training_line'] == 'a'
    assert selected['replicate'] == 14
    assert selected['n'] == 90
    assert inference.select_illustration(rows[:-1], line_ids=('a', 'b', 'c'))['status'] == 'deferred'
    with pytest.raises(ValueError, match='duplicate'):
        inference.select_illustration(rows+[rows[0]], line_ids=('a', 'b', 'c'))
