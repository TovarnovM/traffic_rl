"""Frozen descriptive statistics; W7/length stream extensions outside sealed code."""
from dataclasses import replace
from math import fsum

import numpy as np
from physica_mc import statistics as core


def environment_key(density, road_length=1000):
    """SAP environment stream; L2000 never shares the L1000 road draw."""
    key = core.environment_identity(density)
    if road_length not in (1000, 2000):
        raise ValueError('unfrozen road length')
    return key if road_length == 1000 else key.replace('env|', 'env|L=2000|')


def training_key(rule, architecture, scale=1.0):
    """SAP whole-training stream, including independent W7 cohorts."""
    key = core.training_identity(rule, architecture)
    if scale not in (0.5, 1.0, 2.0):
        raise ValueError('unfrozen penalty scale')
    if scale != 1.0:
        if rule != 'S0' or architecture != 'graph_ppo':
            raise ValueError('W7 is Graph S0 only')
        key += f'|penalty_scale={scale:.1f}'
    return key


def index_chunks(identity, size, *, B=49999, chunk_size=512):
    """Exact keyed PCG64 int64 draws, invariant to chunk boundaries."""
    if min(size, B, chunk_size) < 1:
        raise ValueError('positive draw dimensions required')
    rng = np.random.Generator(np.random.PCG64(core.stream_seed(2026102401, identity)))
    for start in range(0, B, chunk_size):
        yield rng.integers(0, size, size=(min(chunk_size, B-start), size), dtype=np.int64)


def bootstrap(contrasts, *, grid, road_length=1000, scales=None, B=49999,
              chunk_size=512, technical=False):
    """Return native BootstrapResults with explicit W7 keys per contrast side.

    ``scales`` maps contrast label to (left_scale, right_scale). Fixed sides
    consume no training stream. Tiny B is permitted only for technical tests.
    Main calls delegate to the sealed core. No selection or roster is inferred.
    """
    contrasts = tuple(contrasts)
    if type(B) is not int or type(chunk_size) is not int or min(B, chunk_size) < 1:
        raise ValueError("positive integer B and chunk_size required")
    if not technical and (np.__version__ != '2.1.2' or B != 49999 or grid.n_env != 30):
        raise ValueError('final inference requires NumPy2.1.2, B49999, n30')
    environment_key(grid.densities[0], road_length)
    if not scales and road_length == 1000:
        return core.crossed_bootstrap(contrasts, grid=grid, B=B, chunk_size=chunk_size)
    scales = scales or {}
    if len({c.label for c in contrasts}) != len(contrasts) or not contrasts:
        raise ValueError('nonempty unique contrast labels required')
    prepared, streams, results = {}, {}, {}
    for c in contrasts:
        sides = []
        for side, scale in zip((c.left, c.right), scales.get(c.label, (1., 1.))):
            p = core._prepare_side(side, grid)
            if p.kind == 'line':
                p = replace(p, identity=training_key(p.rule, p.architecture, scale))
                if p.identity in streams and streams[p.identity] != p.line_ids:
                    raise ValueError('inconsistent roster for training stream')
                streams[p.identity] = p.line_ids
            sides.append(p)
        prepared[c.label] = tuple(sides)
    env = {d: index_chunks(environment_key(d, road_length), grid.n_env, B=B,
                           chunk_size=chunk_size) for d in grid.densities}
    train = {key: index_chunks(key, len(ids), B=B, chunk_size=chunk_size)
             for key, ids in streams.items() if ids}
    complete = {}
    summaries = {}
    for label, (left, right) in prepared.items():
        cells = {cell: core._cell_summary(cell, left, right) for cell in grid.cells}
        summaries[label] = cells
        if left.n_lines != 0 and right.n_lines != 0 and all(s.complete for s in cells.values()):
            complete[label] = np.empty(B)
    if complete:
        for start in range(0, B, chunk_size):
            e = {d: next(s) for d, s in env.items()}
            t = {key: next(s) for key, s in train.items()}
            for label, values in complete.items():
                left, right = prepared[label]
                total = np.zeros(min(chunk_size, B-start))
                for cell in grid.cells:
                    total += (core._replicate_mean(left, cell, e[cell[0]], t)
                              - core._replicate_mean(right, cell, e[cell[0]], t))
                values[start:start+len(total)] = total / len(grid.cells)
    for label, (left, right) in prepared.items():
        values = complete.get(label)
        status = ('undefined_no_training_lines' if left.n_lines == 0 or right.n_lines == 0
                  else 'missing_planned_cells' if values is None else 'complete')
        lo, hi = (None, None) if values is None else np.quantile(values, [(1.0-.95)/2.0, 1.0-(1.0-.95)/2.0], method='linear')
        kind = core._interval_kind(left, right)
        results[label] = core._result(
            label=label, status=status, reason='' if status == 'complete' else status,
            left=left, right=right, grid=grid, cells=summaries[label],
            point=None if values is None else core.equal_weight_grid_mean(
                {c: s.estimate for c, s in summaries[label].items()}, grid),
            ci_low=None if lo is None else float(lo), ci_high=None if hi is None else float(hi),
            confidence=.95, master_seed=2026102401, B=B, replicates=values,
            interval_kind=kind, interval_label=core.ENVIRONMENT_ONLY_LABEL if kind.startswith("environment_only") else core.INTERVAL_LABEL)
    return results


def paired_summary(left, right, *, line_ids, min_repeats=5, ratio=False):
    """Common compatible-environment descriptive mean; never shrink frozen k.

    Left maps every frozen line to environment->value; right is a fixed
    environment->value mapping. Environment keys must include length and seeds.
    Undefined denominators invalidate the mean, without epsilon or deletion.
    """
    if min_repeats < 5:
        raise ValueError('minimum is five distinct environments')
    roster = tuple(line_ids)
    common = set(right)
    for line in roster:
        common &= set(left.get(line, {}))
    result = dict(value=None, sd=None, n=len(common), k=sum(bool(left.get(l)) for l in roster),
                  k_planned=len(roster), status='insufficient_paired_support')
    if not roster or len(common) < min_repeats:
        return result
    values = []
    for env in sorted(common, key=str):
        denominator = right[env]
        if denominator is None or (ratio and denominator == 0):
            result['status'] = 'undefined_denominator'
            return result
        raw = [left[line][env] for line in roster]
        if any(v is None for v in raw):
            result['status'] = 'undefined_outcome'
            return result
        values.append(fsum(v/denominator if ratio else v-denominator for v in raw)/len(roster))
    result['value'], result['sd'] = core.mean_sample_sd(values)
    result['status'] = 'partial_exploratory'
    return result


def select_illustration(records, *, line_ids, replicates=range(30)):
    """Frozen outer median-line then inner median-environment selection, all90."""
    from physica_mc.reporting import select_illustration_point
    lines, repeats = tuple(sorted(line_ids)), tuple(replicates)
    if len(lines) != 3 or len(set(lines)) != 3 or len(repeats) != 30 or len(set(repeats)) != 30:
        raise ValueError('illustrations require three frozen lines and thirty repeats')
    rows = list(records)
    keys = [(r['training_line'], r['replicate']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate illustration pair')
    if set(keys) != {(line, rep) for line in lines for rep in repeats}:
        return dict(status='deferred', reason='requires_all90_pairs', n=len(keys))
    by_key = dict(zip(keys, rows))
    means = {line: fsum(by_key[line, r]['paired_pv_effect'] for r in sorted(repeats))/30
             for line in lines}
    if not all(np.isfinite(v) for v in means.values()):
        raise ValueError('nonfinite illustration effect')
    median = float(np.quantile(list(means.values()), .5, method='linear'))
    line = min(lines, key=lambda l: (abs(means[l]-median), l))
    candidates = [dict(r, seed=r['replicate']) for r in rows]
    chosen = select_illustration_point(candidates, training_line=line)
    row = by_key[line, chosen['seed']]
    return dict(row, status='selected', n=90, line_means=means,
                median_line_mean=median, selection=chosen)
