"""Redraw the current Physica A global figures from included derived rows."""
import argparse
import csv
import gzip
from inputs import INPUTS, RENDER, verify_inputs
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

PV = 'mean_speed_priority'
STOP, LAP, AVAIL = 'stopped_fraction_priority', 'mean_priority_lap_time', 'lap_availability'
CELL = 'paired_PV_speed_difference_final'
DOMAIN = 'grid_PV_speed_difference'
COUNTS = ('R_min', 'pass95_count', 'fail95_count', 'undefined_count')
RL = ('Graph', 'Node', 'Grid')
FIXED = ('B1', 'B2', 'noop', 'original', 'H2')
FRACTIONS = (.9, .95, 1.)
DENSITIES = tuple(i / 100 for i in range(5, 91, 5))
FULL54 = {(d, f) for d in DENSITIES for f in FRACTIONS}
ADDED = (.5, .7, .96, .97, .98, .99)
CI_NOTE = ('Stored 95% descriptive intervals conditional on frozen selection; '
           'not FDR-adjusted or simultaneous. SD is never plotted as CI. '
           'Fixed controls are physical controls, not additional training samples. '
           'Training ordinals are local to method/rule/scale, not paired trainings.')
EXPECTED = {
    ('S0', 'Graph−Node'): (-.07781152263374487, -.16196205761316868, .0016875977366255306),
    ('S0', 'Graph−Grid'): (.09185648148148148, -.0030835853909464295, .19012173353909467),
    ('C95', 'Graph−H2'): (.09485262345679013, .06399595679012342, .12558336419753094),
    ('C95', 'Node−H2'): (.13148786008230454, .08195989197530866, .17156115226337446),
    ('C95', 'Grid−H2'): (-.011089094650205779, -.04303070473251018, .016071455761316862),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(row, key='value'):
    value = float(row[key])
    require(math.isfinite(value), f'nonfinite {key}: {row}')
    return value


def select(rows, **keys):
    return [r for r in rows if all(r[k] == str(v) for k, v in keys.items())]


def one(rows, **keys):
    found = select(rows, **keys)
    require(len(found) == 1, f'expected exactly one row: {keys}; got {len(found)}')
    return found[0]


def roster(method, rule='S0', cohort='1.0'):
    if method in FIXED:
        return ['']
    count = 2 if (method, rule) == ('Graph', 'C95') else 3
    prefix = {'0.5': 'w7-half-s', '2.0': 'w7-double-s'}.get(cohort, f'E3{method.lower()}_s')
    return [prefix + str(i) for i in range(1, count + 1)]


def branch(rows, method, metric=PV, rule='S0', cohort='1.0', panel='main', length=1000, line='all'):
    """Exact source branch. Fixed references never inherit an RL rule or scale."""
    fixed = method in FIXED
    return select(rows, method=method, metric=metric, panel=panel,
                  road_length=length, cohort='1.0' if fixed else cohort,
                  rule='fixed' if fixed else rule, training_line='' if fixed else line)


def check_row(row):
    metric = row['metric']
    missing_lap = metric == LAP and row['value'] == ''
    value = math.nan if missing_lap else number(row)
    require(metric in (PV, STOP, LAP, AVAIL, 'R', CELL, DOMAIN, *COUNTS), 'unsupported metric')
    require(row['n'] == '30' and row['n_planned'] == '30', f'n != 30: {row}')
    base = row['method'].split('−')[0]
    expected_k = (len(roster(base, row['rule'], row['cohort']))
                  if row['training_line'] == 'all' else 1)
    require(row['k'] == (str(expected_k) if base in RL else ''), f'wrong k: {row}')
    if metric in (STOP, LAP, AVAIL):
        defined, support = number(row, 'n_defined'), number(row, 'support')
        require(row['training_line'] != 'all' and defined.is_integer() and 0 <= defined <= 30
                and support.is_integer() and 0 <= support <= 30, f'invalid scalar counts: {row}')
        require(row['family'] == ('mobility' if metric == STOP else 'laps')
                and row['units'] == ('steps' if metric == LAP else 'fraction')
                and row['interval_kind'] == 'descriptive_sd'
                and row['status'] == ('no_completed_laps' if missing_lap else 'complete_descriptive'),
                f'unsupported scalar: {row}')
        require(row['support_kind'] == ('episodes_with_completed_laps' if metric == AVAIL else 'episode')
                and row['na_reason'] == ('no_completed_laps' if metric == LAP and defined < 30 else ''),
                f'wrong scalar provenance: {row}')
        if metric == LAP:
            require(support == defined and missing_lap == (defined == 0)
                    and (missing_lap or value > 0), f'invalid conditional lap mean: {row}')
        else:
            require(defined == 30 and 0 <= value <= 1
                    and (math.isclose(value * 30, support, abs_tol=1e-12) if metric == AVAIL else support == 30),
                    f'invalid fraction/support: {row}')
    elif row['metric'] == PV:
        require(row['status'] == 'complete_descriptive' and row['interval_kind'] == 'descriptive_sd',
                f'incomplete scalar: {row}')
        require(number(row, 'support') == 30, f'wrong scalar support: {row}')
    elif row['metric'] == 'R':
        # Native paired_summary labels even complete n30 ratios partial_exploratory.
        require(row['status'] == 'partial_exploratory' and row['interval_kind'] == 'none'
                and number(row, 'support') == 30, f'unsupported ratio: {row}')
    elif row['metric'] in (CELL, DOMAIN):
        require(row['status'] == 'complete', f'incomplete contrast: {row}')
        if row['metric'] == CELL:
            require(number(row, 'support') == 1, 'cell contrast must cover one cell')
        ci_endpoints(row)
    else:
        require(row['status'] == 'complete_descriptive' and row['x'] == 'full54'
                and row['x_label'] == 'planned domain' and number(row, 'support') == 54,
                f'incomplete full54 summary: {row}')
    return value


def ci_endpoints(row):
    require(row['interval_kind'] not in ('', 'none', 'descriptive_sd'), 'SD is not CI')
    low, high = number(row, 'ci_low'), number(row, 'ci_high')
    require(low <= high, 'reversed CI endpoints')
    return low, high


def cells(rows, expected):
    actual = [(number(r, 'density'), number(r, 'fraction')) for r in rows]
    require(len(actual) == len(set(actual)) and set(actual) == set(expected),
            f'duplicate/missing/unplanned cells: expected {len(expected)}, got {len(actual)}')
    identities = {tuple(r[k] for k in ('method', 'metric', 'rule', 'road_length',
                                       'cohort', 'training_line', 'model_id')) for r in rows}
    require(len(identities) == 1, 'mixed curve identities/rules/lengths/weights')
    for row in rows:
        check_row(row)
    return sorted(rows, key=lambda r: (number(r, 'density'), number(r, 'fraction')))


def mean54(rows):
    rows = cells(rows, FULL54)
    identities = {(r['panel'], r['road_length'], r['rule'], r['method'],
                   r['cohort'], r['training_line'], r['model_id'], r['metric']) for r in rows}
    require(len(identities) == 1 and rows[0]['metric'] == PV
            and rows[0]['panel'] == 'main' and rows[0]['road_length'] == '1000'
            and rows[0]['training_line'] != 'all', 'mixed/non-line full54 mean')
    return math.fsum(number(r) for r in rows) / 54


def merged_curve(rows, method, metric, density, fractions, *, length=1000, line='all'):
    result = []
    for fraction in fractions:
        panel = ('E5_L2000' if length == 2000 else
                 'main' if fraction in FRACTIONS else 'E5_L1000')
        source = branch(rows, method, metric, panel=panel, length=length, line=line)
        result.append(one(source, density=str(density), fraction=str(fraction)))
    return cells(result, {(density, f) for f in fractions})


def load(path):
    digest = hashlib.sha256()
    with gzip.open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    kept = []
    with gzip.open(path, 'rt', newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        for lineno, row in enumerate(reader, 2):
            if (row['panel'] in ('main', 'W7', 'E5_L1000', 'E5_L2000')
                    and row['metric'] in (PV, STOP, LAP, AVAIL, CELL, DOMAIN, 'R', *COUNTS)
                    and row['rule'] in ('fixed', 'S0', 'S_0.93', 'C95')):
                kept.append(dict(row, source_row=str(lineno)))
        require(lineno - 1 == 136626, f'unexpected source row count: {lineno - 1}')
    return kept, columns + ['source_row'], digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')



def validation_candidates(path, cohort):
    require(cohort in ('0.5', '2.0'), 'unexpected validation cohort')
    metadata_bytes = path.with_suffix('.json').read_bytes()
    metadata = json.loads(metadata_bytes)
    identity = dict(family='selection', panel='validation', cohort=cohort, rule='candidate',
                    road_length='1000', metric='validation_V', units='cells/step',
                    estimand='equal12-cell validation PV speed', phase='',
                    x_label='checkpoint environment steps')
    require(metadata['mode'] == metadata['source_manifest']['mode'] == 'completed_validation'
            and metadata['id'] == path.stem and metadata['points'] == path.name
            and metadata['plotted_rows'] == 39
            and metadata['identity'] == dict(identity, layout='complete', page=1,
                                            facets=[['', '', '', '', '', 'Graph']]),
            'wrong validation metadata identity/completion')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames + ['source_row', 'source', 'source_sha256']
        rows = [dict(r, source_row=str(i), source=str(path), source_sha256=sha)
                for i, r in enumerate(reader, 2)]
    expected = dict(identity, method='Graph', n='20', n_planned='20', k='1', k_planned='3',
                    support='12', support_kind='complete_cells', status='completed_validation',
                    interval_kind='none', sd='', ci_low='', ci_high='', na_reason='')
    grid = {(line, f'iter_{i:06d}', str(i * 16384))
            for line in roster('Graph', cohort=cohort) for i in (*range(100, 1201, 100), 1221)}
    require(len(rows) == 39 and {(r['training_line'], r['candidate_id'], r['env_steps'])
                               for r in rows} == grid, 'incomplete/duplicate validation candidate grid')
    for r in rows:
        require(all(r.get(k) == v for k, v in expected.items()) and r['x'] == r['env_steps'],
                'wrong native validation identity/budget/support')
        number(r)
    return rows, columns, dict(source=str(path), source_sha256=sha,
                              metadata_source=str(path.with_suffix('.json')),
                              metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest())


def write_csv(path, rows, columns):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def latex(value):
    return ''.join({'_': r'\_', '%': r'\%', '&': r'\&', '#': r'\#',
                    '−': r'\ensuremath{-}'}.get(c, c) for c in str(value))


def table(path, rows, columns):
    # Display precision only; CSV/JSON sidecars retain every native value.
    headings = {'rule': 'Rule', 'method': 'Method / contrast', 'x': 'Domain',
                'value': 'Difference', 'ci_low': r'95\% low', 'ci_high': r'95\% high',
                'k': 'k', 'support': 'Cells', 'training_line': 'Training line',
                'R_min': 'min R', 'pass95_count': 'Pass .95',
                'fail95_count': 'Fail .95', 'undefined_count': 'Undefined'}
    lines = ['% Standalone tabular; four-decimal display, exact values in CSV/JSON.',
             r'\begin{tabular}{' + 'l' * len(columns) + '}', r'\hline',
             ' & '.join(headings[k] for k in columns) + r' \\', r'\hline']
    for row in rows:
        values = []
        for key in columns:
            value = row[key]
            if key in ('value', 'ci_low', 'ci_high'):
                values.append(f'${float(value):+.4f}$')
            elif key == 'R_min':
                values.append(f'${float(value):.4f}$')
            elif key in COUNTS[1:]:
                require(float(value).is_integer(), 'non-integer count')
                values.append(str(int(float(value))))
            else:
                values.append(latex(value))
        lines.append(' & '.join(values) + r' \\')
    path.write_text('\n'.join(lines + [r'\hline', r'\end{tabular}']) + '\n')


def validate_rosters(rows):
    for rule in ('S0', 'S_0.93', 'C95'):
        for method in RL:
            source = select(rows, panel='main', method=method, rule=rule, metric=PV)
            require({r['training_line'] for r in source} == {'all', *roster(method, rule)},
                    f'unexpected frozen roster: {method}/{rule}')
            for line in ['all', *roster(method, rule)]:
                cells(branch(rows, method, rule=rule, line=line), FULL54)
            if rule == 'S_0.93':
                for line in roster(method):
                    a = cells(branch(rows, method, line=line), FULL54)
                    b = cells(branch(rows, method, rule=rule, line=line), FULL54)
                    for left, right in zip(a, b):
                        for key in ('model_id', 'candidate_id', 'policy_id', 'env_steps', 'value', 'sd'):
                            require(left[key] == right[key], f'S093 differs from S0: {key}')
    extensions = [('W7', scale, 1000, {(d, f) for d in (.1, .2, .3, .6) for f in FRACTIONS})
                  for scale in ('0.5', '1.0', '2.0')]
    extensions += [('E5_L1000', '1.0', 1000, {(d, f) for d in (.2, .3, .6) for f in ADDED}),
                   ('E5_L2000', '1.0', 2000, {(d, f) for d in (.2, .3, .6) for f in (.9, 1.)})]
    for panel, cohort, length, planned in extensions:
        for metric in (PV, 'R'):
            expected = roster('Graph', cohort=cohort) + (['all'] if metric == PV else [])
            source = select(rows, panel=panel, cohort=cohort, method='Graph', rule='S0', metric=metric)
            require({r['training_line'] for r in source} == set(expected), 'unexpected extension roster')
            for line in expected:
                cells(branch(rows, 'Graph', metric, panel=panel, cohort=cohort,
                             length=length, line=line), planned)


def build(source, output):
    require(not output.exists(), f'output already exists: {output}')
    verify_inputs(source)
    manifest_bytes = (source / 'global-data' / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    require(manifest['mode'] == 'complete' and manifest['row_counts']['episodes.csv'] == 30690
            and manifest['row_counts']['points.csv'] == 136626, 'wrong native completed dataset')
    rows, columns, sha = load(source / 'global-data' / 'points.csv.gz')

    validate_rosters(rows)
    output.mkdir()  # exclusive; never overwrite an earlier or concurrent build
    common = dict(source=str(source / 'global-data' / 'points.csv.gz'), source_sha256=sha,
                  source_hash_semantics='SHA256 of original decompressed CSV bytes',
                  native_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                  native_episode_count=manifest['row_counts']['episodes.csv'],
                  semantics=CI_NOTE, source_row_number='1-based CSV record number; header=1',
                  source_filter='main/W7/E5 panels; PV, stopped fraction, lap time/availability, R, full54 counts, native cell/domain contrasts')
    sys.path.insert(0, str(RENDER))
    from style import RC, method_style
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC)
    plt.rcParams['pdf.fonttype'] = 42

    def figure(nrows=2, ncols=3):
        return plt.subplots(nrows, ncols, figsize=(7.1, 2.05 * nrows + .5),
                            squeeze=False, sharey='row')

    def curve(ax, data, x, label, *, linestyle=None, marker=None, thin=False, ci=False, categorical=False):
        require(bool(data), f'empty curve: {label}')
        data = sorted(data, key=lambda r: number(r, x))
        values = [check_row(row) for row in data]
        style = method_style(data[0]['method'])
        if linestyle is not None:
            style['linestyle'] = linestyle
        if marker is not None:
            style['marker'] = marker
        if thin:
            style.update(linewidth=.8, markersize=2.4, alpha=.75)
        xs = [f'{number(r, x):g}' if categorical else number(r, x) for r in data]
        ax.plot(xs, values, label=label, **style)
        if ci:
            for r in data:
                low, high = ci_endpoints(r)
                ax.vlines(number(r, x), low, high, color=style['color'], linewidth=.8)
                ax.plot([number(r, x)] * 2, [low, high], marker='_', linestyle='none',
                        color=style['color'], markersize=3)
        return data

    def save(name, fig, axes, used, description, details=(), *, provenance=None, native_columns=None):
        handles = {}
        for i, ax in enumerate(axes.flat):
            ax.text(.01, 1.04, chr(97 + i), transform=ax.transAxes, weight='bold')
            hs, ls = ax.get_legend_handles_labels()
            handles.update(zip(ls, hs))
        fig.legend(handles.values(), handles.keys(), loc='lower center',
                   ncol=min(5, len(handles)), frameon=False)
        fig.tight_layout(rect=(0, .15, 1, 1), h_pad=1.8, w_pad=1.1)
        native = sorted({(r.get('source', ''), r['source_row']): r for r in used}.values(),
                        key=lambda r: (r.get('source', ''), int(r['source_row'])))
        write_csv(output / (name + '.csv'), native, columns if native_columns is None else native_columns)
        write_json(output / (name + '.json'), dict(common if provenance is None else provenance, scope=description,
                   metrics=sorted({r['metric'] for r in native}), selection=details,
                   source_rows=[r['source_row'] if provenance is None else
                                dict(source=r['source'], source_row=r['source_row']) for r in native]))
        fig.savefig(output / (name + '.png'), dpi=225)
        fig.savefig(output / (name + '.pdf'))
        plt.close(fig)

    fig, axes = figure()
    used = []
    for j, f in enumerate(FRACTIONS):
        for i, methods in enumerate((FIXED, (*RL, 'H2', 'B2'))):
            for method in methods:
                data = cells(branch(rows, method), FULL54)
                used += curve(axes[i, j], select(data, fraction=f), 'density', method,
                              thin=i == 1 and method == 'B2')
            axes[i, j].set(title=f'f = {f:g}', xlabel='Density')
        axes[0, 0].set_ylabel('PV speed [cells/step]')
        axes[1, 0].set_ylabel('PV speed [cells/step]')
    # Same physical metric in both rows uses a single common Y range.
    limits = [ax.get_ylim() for ax in axes.flat]
    for ax in axes.flat:
        ax.set_ylim(min(v[0] for v in limits), max(v[1] for v in limits))
    save('main_mobility', fig, axes, used, 'main L1000 S0; true fixed hierarchy and RL family means; no CI')

    fig, axes = figure()
    used = []
    for i, method in enumerate(('Graph−Node', 'Graph−Grid')):
        data = cells(select(rows, panel='main', rule='S0', method=method, metric=CELL,
                            cohort='1.0', road_length='1000', training_line='all'), FULL54)
        for j, f in enumerate(FRACTIONS):
            used += curve(axes[i, j], select(data, fraction=f), 'density', method, ci=True)
            axes[i, j].axhline(0, color='.4', lw=.8, ls=':')
            axes[i, j].set(title=f'f = {f:g}', xlabel='Density')
        axes[i, 0].set_ylabel(method + '\nPV difference [cells/step]')
    limits = [ax.get_ylim() for ax in axes.flat]
    for ax in axes.flat:
        ax.set_ylim(min(v[0] for v in limits), max(v[1] for v in limits))
    save('main_architecture', fig, axes, used, 'main S0 native cell means and 95% CI; zero reference; no win counts')

    fig, axes = figure(1, 2)
    used, derived = [], []
    for j, rule in enumerate(('S0', 'C95')):
        for method in (*RL, 'H2', 'original'):
            for ordinal, line in enumerate(roster(method, rule), 1):
                data = branch(rows, method, rule=rule, line=line)
                y = mean54(data)
                ref = one(branch(rows, method, 'R_min', rule=rule, line=line), x='full54')
                check_row(ref)
                style = method_style(method)
                axes[0, j].plot(number(ref), y, linestyle='none', marker=style['marker'],
                                color=style['color'], markerfacecolor=style['markerfacecolor'], label=method)
                if line:
                    axes[0, j].annotate(str(ordinal), (number(ref), y), xytext=(4, 3),
                                        textcoords='offset points', fontsize=8)
                used += data + [ref]
                derived.append(dict(facet=rule, method=method, training_line=line,
                                    ordinal=ordinal if line else None, x=number(ref), y=y,
                                    expression='math.fsum(float(row[value]) for row in exact FULL54) / 54',
                                    x_source_row=ref['source_row'],
                                    y_source_rows=[r['source_row'] for r in data]))
        axes[0, j].axvline(.95, color='.4', ls=':', lw=.9)
        axes[0, j].set(title=rule, xlabel='Per-line minimum R over full54')
    axes[0, 0].set_ylabel('Mean PV speed [cells/step]')
    xlimits = [ax.get_xlim() for ax in axes.flat]
    for ax in axes.flat:
        ax.set_xlim(min(v[0] for v in xlimits), max(v[1] for v in xlimits))
    save('main_cost_screen', fig, axes, used, 'Equal-cell PV means vs stored per-line R_min; no frontier or uncertainty', derived)

    for name, fractions, lengths in [('main_penetration', tuple(sorted(FRACTIONS + ADDED)), (1000,)),
                                     ('main_length', (.9, 1.), (1000, 2000))]:
        fig, axes = figure()
        used = []
        for j, density in enumerate((.2, .3, .6)):
            for i, metric in enumerate((PV, 'R')):
                for length in lengths:
                    for method in ('Graph', 'H2', 'B2'):
                        for line in (roster(method) if metric == 'R' else ['all']):
                            data = merged_curve(rows, method, metric, density, fractions,
                                                length=length, line=line)
                            label = method + (f' L{length}' if len(lengths) == 2 else '')
                            used += curve(axes[i, j], data, 'fraction', label,
                                          linestyle='-' if length == 1000 else '--',
                                          thin=metric == 'R' and method == 'Graph',
                                          categorical=name == 'main_penetration')
                axes[i, j].set(title=f'Density = {density:g}', xlabel='AV fraction (evaluated levels)' if name == 'main_penetration' else 'Fraction f')
                if name == 'main_penetration':
                    axes[i, j].tick_params(axis='x', rotation=60)
                if metric == 'R':
                    axes[i, j].axhline(.95, color='.5', ls=':', lw=.8)
            axes[0, 0].set_ylabel('PV speed [cells/step]')
            axes[1, 0].set_ylabel('Per-line R')
        save(name, fig, axes, used,
             'Graph S0 family PV; three separate Graph R lines; true fixed H2/B2. '
             'main L1000 supplies .90/.95/1; E5_L1000 supplies six added fractions; '
             'E5_L2000 supplies L2000 .90/1 only. No pooling, cross-length CI, or full54 claim for E5.',
             dict(lengths=lengths, fractions=fractions, densities=[.2, .3, .6],
                  connecting_lines='visual guides between stored points only',
                  fraction_axis='equally spaced evaluated levels, not linear' if name == 'main_penetration' else 'linear'))

    # Separate PV/R keeps all nine ratio curves visible at publication width.
    for metric, suffix in ((PV, 'pv'), ('R', 'r')):
        fig, axes = figure(1, 3)
        used = []
        for j, f in enumerate(FRACTIONS):
            for cohort, dash, marker in [('0.5', ':', 'o'), ('1.0', '-', 's'), ('2.0', '--', '^')]:
                for line in (roster('Graph', cohort=cohort) if metric == 'R' else ['all']):
                    data = cells(branch(rows, 'Graph', metric, panel='W7', cohort=cohort, line=line),
                                 {(d, f0) for d in (.1, .2, .3, .6) for f0 in FRACTIONS})
                    label = f'×{cohort}, line {line[-1]}' if metric == 'R' else f'Graph ×{cohort}'
                    point_marker = ('o', 's', '^')[int(line[-1]) - 1] if metric == 'R' else marker
                    used += curve(axes[0, j], select(data, fraction=f), 'density',
                                  label, linestyle=dash, marker=point_marker, thin=metric == 'R')
            axes[0, j].set(title=f'f = {f:g}', xlabel='Density')
            if metric == 'R':
                axes[0, j].axhline(.95, color='.5', ls=':', lw=.8)
        axes[0, 0].set_ylabel('Per-line R' if metric == 'R' else 'PV speed [cells/step]')
        save('supp_penalties_' + suffix, fig, axes, used,
             'W7 S0 Graph ×0.5/1/2; color denotes Graph. PV family means: scale uses pattern/marker. '
             'R: pattern denotes scale and marker identifies each of three lines within it; no ordinal pairing.')

    for metric, name, ylabel in [(PV, 'supp_main_lines', 'PV speed [cells/step]'),
                                 (STOP, 'supp_stopped', 'Stopped fraction'),
                                 (LAP, 'supp_lap_time', 'PV lap time [steps]'),
                                 (AVAIL, 'supp_lap_availability', 'Lap availability')]:
        fig, axes = figure(3, 3)
        used = []
        for i, method in enumerate(RL):
            for j, f in enumerate(FRACTIONS):
                for ordinal, (line, dash) in enumerate(zip(roster(method), ('-', '--', ':')), 1):
                    data = cells(branch(rows, method, metric, line=line), FULL54)
                    used += curve(axes[i, j], select(data, fraction=f), 'density',
                                  f'{method} {ordinal}', linestyle=dash, thin=True)
                for control in (('H2', 'B2') if metric == PV else FIXED):
                    used += curve(axes[i, j], select(cells(branch(rows, control, metric), FULL54), fraction=f),
                                  'density', control)
                axes[i, j].set(title=f'{method}; f = {f:g}', xlabel='Density')
            axes[i, 0].set_ylabel(ylabel)
        limits = [ax.get_ylim() for ax in axes.flat]
        for ax in axes.flat:
            # Keep the planned density extent even when terminal lap times are NA.
            ax.set(xlim=(0, .95), xticks=(.1, .3, .5, .7, .9))
            ax.set_ylim((0, 1) if metric in (STOP, AVAIL) else
                        (min(v[0] for v in limits), max(v[1] for v in limits)))
        save(name, fig, axes, used, 'main L1000 S0 individual frozen training lines; full54 each; no CI/SD bands or pairing',
             dict(training_lines={m: roster(m) for m in RL},
                  missing_lap='Validated no_completed_laps only: line gap; full planned density extent retained, including NA tails; native CSV unchanged',
                  availability='Observed fraction of 30 episodes completing laps; zero is valid; missing is rejected'))

    validation_provenance = dict(sources=[], source_row_number=common['source_row_number'],
                                semantics='Native validation objective; no CI, inference or reselection. '
                                '20 environments × 12 cells; 13 candidates per training; ordinals local to scale.')
    with plt.rc_context({'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8}):
        fig, axes = figure(1, 2)
        fig.set_size_inches(7.1, 3)
        used = []
        for j, (cohort, digest) in enumerate((('0.5', '595e1ddf43c0'), ('2.0', '82ae3e1ff6ae'))):
            stem = f'selection--validation--{cohort}--candidate--1000--validation-V--complete--1--{digest}'
            path = source / 'validation' / (stem + '.csv')
            data, validation_columns, provenance = validation_candidates(path, cohort)
            used += data
            validation_provenance['sources'].append(provenance)
            for ordinal, (line, dash, marker) in enumerate(zip(roster('Graph', cohort=cohort),
                                                              ('-', '--', ':'), ('o', 's', '^')), 1):
                points = sorted(select(data, training_line=line), key=lambda r: number(r, 'env_steps'))
                style = dict(method_style('Graph'), linestyle=dash, marker=marker, markersize=3)
                axes[0, j].plot([number(r, 'env_steps') for r in points], [number(r) for r in points],
                                label=f'Graph training {ordinal}', **style)
            axes[0, j].set(title=f'Penalty ×{cohort}', xlabel='Checkpoint environment steps')
        axes[0, 0].set_ylabel('Validation objective [cells/step]')
        save('validation_penalties', fig, axes, used, 'Half/double penalty native validation candidates',
             provenance=validation_provenance, native_columns=validation_columns)

    domains = select(rows, metric=DOMAIN)
    expected_keys = set()
    main_domains = ('selection12', 'transfer42', 'density_band_low', 'density_band_middle', 'density_band_high', 'full54')
    for rule, method in EXPECTED:
        expected_keys.update(('main', '1.0', rule, method, x) for x in main_domains)
    for cohort in ('0.5', '2.0'):
        expected_keys.update(('W7', cohort, 'S0', 'Graph−Graph', x) for x in ('selection9', 'transfer3', 'all12'))
    for panel, domain in [('E5_L1000', 'added18'), ('E5_L2000', 'length6')]:
        expected_keys.update((panel, '1.0', 'S0', 'Graph−' + m, domain) for m in ('B1', 'B2', 'H2'))
    keys = [(r['panel'], r['cohort'], r['rule'], r['method'], r['x']) for r in domains]
    require(len(keys) == 42 and set(keys) == expected_keys, 'expected all 42 unique domain effects')
    support = dict(selection12=12, transfer42=42, density_band_low=9, density_band_middle=9,
                   density_band_high=36, full54=54, selection9=9, transfer3=3, all12=12, added18=18, length6=6)
    for r in domains:
        check_row(r)
        require(number(r, 'support') == support[r['x']] and r['x_label'] == 'planned domain'
                and r['road_length'] == ('2000' if r['panel'] == 'E5_L2000' else '1000'), 'wrong domain support/length')
    main5 = []
    for (rule, method), values in EXPECTED.items():
        r = one(domains, panel='main', rule=rule, method=method, x='full54')
        require(tuple(number(r, k) for k in ('value', 'ci_low', 'ci_high')) == values,
                f'native main effect differs: {rule}/{method}: {r}')
        main5.append(r)
    write_csv(output / 'main_effects.csv', main5, columns)
    table(output / 'main_effects.tex', main5, ['rule', 'method', 'value', 'ci_low', 'ci_high', 'k', 'support'])
    domains.sort(key=lambda r: (r['panel'], r['cohort'], r['rule'], r['method'], r['x']))
    write_csv(output / 'all42domain_effects.csv', domains, columns)
    # Six small, self-contained tabular environments in one input file.
    parts = []
    for panel, cohort, rule in sorted({(r['panel'], r['cohort'], r['rule']) for r in domains}):
        subset = select(domains, panel=panel, cohort=cohort, rule=rule)
        part = output / 'all42domain_effects.tex'
        table(part, subset, ['method', 'x', 'value', 'ci_low', 'ci_high', 'k', 'support'])
        heading = latex(panel + ' / scale ' + cohort + ' / ' + rule)
        parts.append(r'\par\noindent ' + heading + '\n' + r'\par\nopagebreak[4]' + '\n' + part.read_text())

    (output / 'all42domain_effects.tex').write_text('\n\n'.join(parts))
    audit = []
    for rule, methods in [('S0', RL), ('S_0.93', RL), ('C95', RL), ('fixed', FIXED)]:
        for method in methods:
            for line in roster(method, rule):
                for metric in COUNTS:
                    r = one(branch(rows, method, metric, rule=rule, line=line), x='full54')
                    check_row(r)
                    audit.append(r)
    write_csv(output / 'background_audit.csv', audit, columns)
    audit_wide = []
    for rule, method, line in dict.fromkeys((r['rule'], r['method'], r['training_line']) for r in audit):
        group = select(audit, rule=rule, method=method, training_line=line)
        audit_wide.append(dict(rule=rule, method=method, training_line=line,
                               **{metric: one(group, metric=metric)['value'] for metric in COUNTS}))
    table(output / 'background_audit.tex', audit_wide, ['rule', 'method', 'training_line', *COUNTS])

    illustrative = []
    for rule in ('S0', 'C95'):
        for method in RL:
            illustrative.append(one(branch(rows, method, rule=rule), density='0.3', fraction='1.0'))
    for method in FIXED:
        illustrative.append(one(branch(rows, method), density='0.3', fraction='1.0'))
    write_json(output / 'evidence.json', dict(common, illustrative_cell=illustrative, main5=main5,
               all42domain_effects=domains, background_audit=audit,
               background_note='pass95 always uses .95, including S_0.93. S_0.93 weights verified identical to S0; no new n.',
               derived_cost_screen=derived, gains='No derived gains or new intervals computed.'))
    write_json(output / 'COMPLETE.json', dict(sources=[common, validation_provenance], figures=12, status='complete'))
    print(f'Wrote {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=INPUTS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == '__main__':
    rc = 0
    try:
        main()
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 1
    except BaseException:
        traceback.print_exc()
        rc = 1
    finally:
        gc.collect()
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(rc)
