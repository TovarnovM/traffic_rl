"""Redraw current Physica A local figures from compact native scientific rows."""
import argparse
import csv
import gzip
from inputs import INPUTS, RENDER, verify_inputs
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

from build_global_assets import latex, require, write_csv, write_json

MAIN_F = (.9, .95, 1.)
RHO = (.1, .2, .3, .6)
ADDED_F = (.5, .7, .96, .97, .98, .99)
WINDOWS = tuple(f'{kind}_{shape}_{m}' for kind in ('fixed', 'speed')
                for m in (1, 2, 3) for shape in ('sym', 'front'))
PRIMARY = 'fixed_front_2'
METRICS = ('pooled_speed', 'pooled_stopped_fraction', 'normalized_local_delay',
           'cell_time_density', 'cell_time_flow', 'empty_fraction')
COUNTS = ('N', 'S', 'Z', 'A')
EVENTS = ('pv_stop', 'pv_low_progress', 'front_stagnation', 'operational_blocking')
EVENT_FIELDS = ('confirmed_count', 'fully_observed_count', 'left_censored_count',
                'right_censored_count', 'descriptive_duration/mean')
EVENT_METRICS = tuple(f'{e}/{m}' for e in EVENTS for m in EVENT_FIELDS)
SPATIAL = 'connected_occupancy_covariance'
TEMPORAL = ('pv_speed/covariance', 'pv_speed/value')
ONSET = ('primary_front_N', 'primary_front_S', 'primary_front_Z', 'pv_speed')
FILES = ('points.csv', 'manifest.json', 'coverage.csv', 'episodes.csv')
UNITS = dict(zip(METRICS, ('cells/step', 'fraction', 'fraction', 'vehicles/cell',
                          'vehicles/step', 'fraction')),
             N='vehicle-time', S='vehicle-cells', Z='vehicle-time', A='cell-time',
             connected_occupancy_covariance='occupancy squared')
UNITS.update({m: 'steps' if m.endswith('/mean') else 'count' for m in EVENT_METRICS})
UNITS.update(dict(zip(TEMPORAL, ('(cells/step)^2', 'normalized covariance'))))
UNITS.update(dict(zip(ONSET, ('count', 'count', 'count', 'cells/step'))))
NOTES = {
    'temporal': 'Native PV-speed covariance and normalized covariance, lags 0..40 steps. '
                'Temporal-pair support is not independent replication. Zero-variance NA stays NA; '
                'normalized covariance is not clipped to exact +/-1. No SD/CI bands.',
    'onset': 'PV-stop onset anchor; left-censored events excluded. Only valid measured '
             'in-horizon offsets -20..20, without wrap or interpolation. Within-episode onset '
             'means then equal defined-episode means; n_defined and event support may vary '
             'with offset. Events are not independent replicates or causal proof. Native '
             'primary_front_S unit count is a legacy label for instantaneous speed sum; '
             'preserved unchanged in sidecars, not plotted. Missing onset rows may have an empty '
             'native na_reason with n_defined=0, support=0 and status=missing; no cause is imputed. No SD/CI bands.',

    'sampling': 'n=30 environments per line/control; n_defined is separate. Supports '
                'are vehicle-time, events or lag pairs, never independent replications. SD is not CI.',
    'values': 'Native equal-episode means copied exactly; no family pooling, imputation, '
              'new uncertainty, recomputation of N/S/Z/A, or B1 alias as additional n.',
    'geometry': 'Primary fixed_front_2/full/all/background: A=328000 cell-steps '
                '(41 x 4 x 2000), including PV site; only N/S/Z exclude PV.',
    'events': 'Confirmed one-step-inclusive counts, displayed as events/episode; native '
              'count units retained. Duration is native descriptive_duration/mean '
              '[steps], conditional on complete events; availability is n_defined/30 '
              'in these same rows. No mean_observed_duration used.',
    'mechanism': 'Operational blocking mixes post-step PV speed, post-lateral leader '
                 'gap and native pre-step eligibility. Not causal HDV blocking; '
                 'marginal frequencies are never multiplied to infer a joint event.',
    'spatial': 'Native lane-all averages within-lane connected occupancy covariances, '
               'not Pearson correlations. All four physical lanes retained separately.',
    'sensitivity': 'Graph line minus same-cell H2 native normalized_local_delay; '
                   'simple point differences, not paired CI or individual-driver delays. '
                   'Speed windows use each episode post-step max(PV speed,1); realised areas need not match.',
}


def expected_scopes():
    result = {}
    for panel, length, densities, fractions in (
            ('E4', 1000, RHO, MAIN_F),
            ('E5_L1000', 1000, RHO[1:], ADDED_F),
            ('E5_L2000', 2000, RHO[1:], (.9, 1.))):
        for d in densities:
            for f in fractions:
                result[f'{panel}-L{length}-rho{d:.2f}-f{f:.2f}'] = dict(
                    panel=panel, road_length=length, density=d, fraction=f)
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_hash(path, expected):
    require(sha256(path) == expected, f'checksum mismatch: {path}')


def gate(progress):
    expected = expected_scopes()
    require(progress.get('status') == 'complete', 'extraction not complete')
    require(set(progress.get('finished', {})) == set(expected), 'expected exactly 36 scopes')
    identity = progress['identity']
    require(all(identity.get(k) for k in ('snapshot', 'freeze', 'binding', 'native_source', 'files'))
            and identity.get('cache_version') == 2, 'incomplete frozen identity')
    for cid, scope in expected.items():
        entry = progress['finished'][cid]
        require(entry.get('complete') is True and entry.get('scope') == scope, f'incomplete/identity mismatch: {cid}')
        require(set(entry.get('checksums', {})) == set(FILES), f'incomplete checksums: {cid}')
    return expected


def branches(panel):
    methods = ('Graph', 'Node', 'Grid') if panel == 'E4' else ('Graph',)
    return [(m, f'E3{m.lower()}_s{i}') for m in methods for i in (1, 2, 3)] + [
        (m, '') for m in ('B1', 'B2', 'H2')]


def exact_branch(row):
    method, line = row['method'], row['training_line']
    return ((method, line) in branches(row['panel']) and row['cohort'] == '1.0'
            and row['rule'] == ('fixed' if method in ('B1', 'B2', 'H2') else 'S0'))


def keep(row):
    if not exact_branch(row):
        return False
    family, metric = row['family'], row['metric']
    if family == 'windows':
        return ((row['region'], row['lane'], row['vehicle_group']) == ('full', 'all', 'background')
                and ((row['window'] == PRIMARY and metric in (*METRICS, *COUNTS)) or
                     (row['panel'] == 'E4' and row['method'] in ('Graph', 'H2')
                      and row['window'] in WINDOWS and metric == METRICS[2])))
    if family == 'events':
        return metric in EVENT_METRICS
    if family in ('temporal', 'onset'):
        return (metric in (TEMPORAL if family == 'temporal' else ONSET)
                and not any(row[k] for k in ('window', 'region', 'lane', 'vehicle_group', 'phase')))

    return (family == 'spatial' and row['panel'] == 'E4' and metric == SPATIAL
            and row['window'] == PRIMARY and row['lane'] in ('all', '0', '1', '2', '3')
            and not any(row[k] for k in ('region', 'vehicle_group', 'phase')))


def plot_value(row):
    # Matplotlib breaks lines at NaN; sidecars keep the original empty string.
    if row['value'] == '':
        return math.nan
    value = float(row['value'])
    require(math.isfinite(value), 'unexpected nonfinite native value')
    return value


def check_row(row):
    require(row['n'] == row['n_planned'] == '30', 'incomplete environment support')
    nd = int(row['n_defined'])
    require(0 <= nd <= 30 and (row['value'] == '') == (nd == 0), 'invalid n_defined/value')
    # Native onset export omits a reason for zero valid aligned samples.
    empty_onset = (row['family'] == 'onset' and nd == 0 and row['status'] == 'missing'
                   and row['support'] in ('0', '0.0'))
    require(row['value'] != '' or bool(row['na_reason']) or empty_onset, 'NA without reason')
    require(row['interval_kind'] == 'descriptive_sd' and not row['ci_low'] and not row['ci_high'],
            'unexpected interval; no CI is consumed')
    require(row['units'] == UNITS[row['metric']], f'unit mismatch: {row["metric"]}')
    plot_value(row)
    if row['family'] == 'windows':
        require(row['phase'] == 'post-step measured states', 'wrong window phase')
        if row['metric'] == 'A':
            require(row['value'] != '' and float(row['value']) == 328000, 'primary area mismatch')
    elif row['family'] == 'events':
        phase = ('mixed-phase operational' if row['metric'].startswith('operational_blocking/')
                 else 'post-step measured states')
        require(row['phase'] == phase and not any(row[k] for k in
                ('window', 'region', 'lane', 'vehicle_group')), 'wrong event branch')
    elif row['family'] == 'spatial':
        require(row['x_label'] == 'lag (cells)' and row['x'] in {str(i) for i in range(41)},
                'wrong spatial lag')
    elif row['family'] in ('temporal', 'onset'):
        temporal = row['family'] == 'temporal'
        require(keep(row), 'wrong temporal/onset branch')
        require(row['x_label'] == ('lag (steps)' if temporal else 'onset offset (steps)')
                and row['x'] in {str(i) for i in (range(41) if temporal else range(-20, 21))},
                'wrong temporal lag/onset offset')
        require(row['support_kind'] == ('temporal_pairs_not_replicates' if temporal
                                       else 'events_not_replicates'), 'wrong temporal/onset support')



def key(row):
    return (row['panel'], int(row['road_length']), float(row['density']), float(row['fraction']),
            row['method'], row['training_line'], row['metric'], row['window'], row['lane'], row['x'])


def lookup(index, panel, length, density, fraction, method, line, metric,
           window=PRIMARY, lane='all', lag=None):
    if metric in EVENT_METRICS:
        window, lane = '', ''
    x = str(lag) if lag is not None else str(density)
    return index[(panel, length, density, fraction, method, line, metric, window, lane, x)]


def transfer(index, method, line, metric, density, fractions, length):
    result = []
    for fraction in fractions:
        panel = ('E5_L2000' if length == 2000 else
                 'E4' if fraction in MAIN_F else 'E5_L1000')
        result.append(lookup(index, panel, length, density, fraction, method, line, metric))
    return result


def difference(left, right):
    require(left['units'] == right['units'] == 'fraction', 'difference units')
    value = plot_value(left) - plot_value(right)
    return dict(value=value if math.isfinite(value) else None,
                left_file=left['source_file'], left_row=left['source_row'], left_value=left['value'],
                right_file=right['source_file'], right_row=right['source_row'], right_value=right['value'],
                left_n_defined=left['n_defined'], right_n_defined=right['n_defined'],
                left_na_reason=left['na_reason'], right_na_reason=right['na_reason'],
                na_reason='' if math.isfinite(value) else 'undefined_native_operand',
                expression='float(left.value) - float(right.value)', units='fraction')


def check_grid(index, scopes):
    # Assert the whole requested grid before creating any output, including NA rows.
    for scope in scopes.values():
        for method, line in branches(scope['panel']):
            args = (index, scope['panel'], scope['road_length'], scope['density'], scope['fraction'], method, line)
            for metric in (*METRICS, *COUNTS, *EVENT_METRICS):
                lookup(*args, metric)
            for metrics, offsets in ((TEMPORAL, range(41)), (ONSET, range(-20, 21))):
                for metric in metrics:
                    for offset in offsets:
                        lookup(*args, metric, window='', lane='', lag=offset)
            if scope['panel'] == 'E4':
                for lane in ('all', '0', '1', '2', '3'):
                    for lag in range(41):
                        lookup(*args, SPATIAL, lane=lane, lag=lag)
                if method in ('Graph', 'H2'):
                    for window in WINDOWS:
                        lookup(*args, METRICS[2], window=window)


def load(source):
    verify_inputs(source)
    compact = json.loads((source / 'local/provenance.json').read_text())
    scopes = gate(compact)
    identity = compact['identity']
    profile_hashes = {}
    for name in ('full12', 'mechanism'):
        path = source / f'local/analysis-{name}.json'
        original = f'/workspace/.local/science-test-freeze-20261109-v2/analysis-{name}.json'
        checked_hash(path, identity['files'][original])
        profile = json.loads(path.read_text())
        profile_hashes[name] = hashlib.sha256(json.dumps(
            profile, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        require(all(e['min_duration'] == 1 for e in profile['events']), 'event duration filter changed')
    with gzip.open(source / 'local/episode_identity.csv.gz', 'rt', newline='', encoding='utf-8') as stream:
        episodes = list(csv.DictReader(stream))
    require({r['scope_id'] for r in episodes} == set(scopes), 'episode scope mismatch')
    episode_union, coverage_by_scope = set(), {}
    for cid, scope in scopes.items():
        entry, manifest = compact['finished'][cid], compact['scopes'][cid]
        require(manifest['snapshot_sha256'] == identity['snapshot']
                and manifest['freeze_sha256'] == identity['freeze']
                and manifest['profiles'] == profile_hashes and manifest['scope'] == scope,
                f'manifest identity mismatch: {cid}')
        coverage = {}
        for row in manifest['coverage']:
            require(row['n'] == row['n_planned'] == '30' and row['status'] == 'eligible', 'bad coverage')
            pair = row['method'], row['training_line']
            require(pair not in coverage, 'duplicate physical group')
            coverage[pair] = row
        require(set(coverage) == set(branches(scope['panel']))
                and {r['group_id'] for r in coverage.values()} == set(entry['groups'])
                and len(coverage) == manifest['row_counts']['coverage.csv'], 'coverage roster mismatch')
        coverage_by_scope[cid] = coverage
        ids, groups = set(), {}
        scope_episodes = [r for r in episodes if r['scope_id'] == cid]
        require([int(r['source_row']) for r in scope_episodes] == list(range(2, len(scope_episodes) + 2)),
                'episode source coordinates changed')
        for row in scope_episodes:
            require(row['task_id'] not in ids and row['code_source_digest'] == identity['native_source'],
                    'duplicate episode/source mismatch')
            ids.add(row['task_id'])
            groups.setdefault(row['group_id'], []).append(row['replicate'])
        require(set(groups) == set(entry['groups']) and all(len(v) == len(set(v)) == 30 for v in groups.values())
                and len(ids) == entry['physical_episodes'] == manifest['row_counts']['episodes.csv'],
                'episode coverage mismatch')
        require(manifest['row_counts']['points.csv'] == entry['counts']['selected'], 'source row count mismatch')
        episode_union.update(ids)
    require(len(episode_union) == compact['unique_physical_episodes'] == 7770,
            'expected 7770 distinct physical episodes; aliases are not new n')
    rows, index, coordinates, seen_scopes = [], {}, set(), set()
    with gzip.open(source / 'local/points.csv.gz', 'rt', newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        require(columns == compact['columns'] and len(columns) == len(set(columns)), 'invalid header')
        for row in reader:
            require(None not in row and None not in row.values(), 'malformed CSV')
            cid = Path(row['source_file']).parent.name
            require(cid in scopes, 'unknown source scope')
            scope, entry = scopes[cid], compact['finished'][cid]
            require(row['source_file'] == f'/workspace/.local/manuscript-revision-20261110/article-data/{cid}/points.csv'
                    and row['source_sha256'] == entry['checksums']['points.csv']
                    and 2 <= int(row['source_row']) <= entry['counts']['selected'] + 1,
                    'invalid original source coordinate/hash')
            coordinate = row['source_file'], row['source_row']
            require(coordinate not in coordinates, 'duplicate projected coordinate')
            coordinates.add(coordinate)
            seen_scopes.add(cid)
            require(row['panel'] == scope['panel'] and all(float(row[k]) == scope[k]
                    for k in ('road_length', 'density', 'fraction')), 'point outside recorded scope')
            require(keep(row), 'unexpected projected branch')
            check_row(row)
            physical = coverage_by_scope[cid][row['method'], row['training_line']]
            require(all(row[k] == physical[k] for k in
                        ('model_id', 'policy_id', 'candidate_id', 'env_steps', 'profile', 'cohort')),
                    'point physical identity mismatch')
            require(key(row) not in index, 'duplicate selected point')
            index[key(row)] = row
            rows.append(row)
    require(len(rows) == compact['selected_rows'] and seen_scopes == set(scopes), 'projection coverage mismatch')
    check_grid(index, scopes)
    return rows, index, columns, dict(
        scientific_provenance=compact, progress_sha256=compact['progress_sha256'], notes=NOTES,
        source_row_number='1-based CSV record number, header=1', unique_physical_episodes=7770,
        verification='Included payloads, frozen identities and projected support; excluded raw archives are not revalidated.')



TRACE_UNITS = dict(speed='cells/step', lane='physical lane', cumulative_distance='cells')
RECORDED_NOTE = ('12 frozen paired illustrations: two episodes from each prespecified all-90 pool '
                 '(three lines x 30 environments); selection n=90 is not population evidence. '
                 'Full 2000-step maps and storyboards remain working companion data; dense AV '
                 'outlines are not publication maps. Native step 0 retained; figures show 1..2000.')


def recorded_series(rows):
    series = {(m, metric): {} for m in ('Graph', 'H2') for metric in TRACE_UNITS}
    for r in rows:
        branch, step, value = (r['method'], r['metric']), float(r['x']), plot_value(r)
        require(branch in series and r['phase'] == 'post-step' and r['x_label'] == 'step'
                and r['units'] == TRACE_UNITS[r['metric']], 'invalid native trace branch')
        require(step.is_integer() and 0 <= step <= 2000 and step not in series[branch], 'invalid trace step')
        require(math.isnan(value) or (r['metric'] == 'speed' and 0 <= value <= 6)
                or (r['metric'] == 'lane' and value in (0, 1, 2, 3))
                or (r['metric'] == 'cumulative_distance' and value >= 0), 'invalid physical lane/speed/distance')
        series[branch][int(step)] = value
    require(all(set(v) == set(range(2001)) for v in series.values()), 'incomplete trace coverage')
    return series


def load_recorded(source):
    verify_inputs(source)
    native = source
    manifest = native / 'global-data/illustrations.json'
    manifest_text = manifest.read_text()
    entries, lexical = json.loads(manifest_text), json.loads(manifest_text, parse_float=str)
    expected = {(d, f) for d in RHO for f in MAIN_F}
    cells = [(e['identity']['density'], e['identity']['fraction']) for e in entries]
    require(len(cells) == 12 and set(cells) == expected, 'expected exactly 12 recorded cells')
    selected = dict(zip(cells, zip(entries, lexical)))
    paths = sorted((native / 'recorded').glob('*--PV-trace--speed-lane-distance--all--*.json'))
    require(len(paths) == 12, 'expected exactly 12 native traces')
    result, seen = [], set()
    for path in paths:
        meta = json.loads(path.read_text())
        identity, origin = meta['identity'], meta['source_manifest']
        cell = identity['density'], identity['fraction']
        require(cell in selected and cell not in seen, 'unexpected/duplicate trace cell')
        seen.add(cell)
        entry, exact = selected[cell]
        e, ids = entry['selection_evidence'], {m: entry['pair'][m]['episode_id'] for m in ('Graph', 'H2')}
        require(entry['identity'] == dict(density=cell[0], fraction=cell[1], road_length=1000,
                panel='E4', rule='S0', cohort=1.) and origin['identity'] == entry['identity']
                and identity == dict(entry['identity'], family='illustrations', metric='PV-trace',
                                     layout='speed-lane-distance'), 'recorded identity mismatch')
        require(entry['mode'] == meta['mode'] == origin['mode'] == 'final'
                and entry['status'] == e['status'] == 'selected'
                and origin['selection_evidence'] == e and origin['episode_ids'] == ids
                and all(e[m] == ids[m] for m in ids), 'selection evidence/episode ID mismatch')
        require(e['n'] == e['n_planned'] == 90 and e['k'] == e['k_planned'] == 3
                and e['scope'] == 'complete_E4_cell' and e['rule'] == 'frozen_median_line_then_environment'
                and e['cell'] == dict(density=cell[0], fraction=cell[1], road_length=1000)
                and e['training_line'] in ('E3graph_s1', 'E3graph_s2', 'E3graph_s3')
                and e['replicate'] in range(30) and e['selection']['seed'] == e['replicate']
                and e['selection']['candidates'] == 30 and all(e['selection'][k] == e[k]
                for k in ('training_line', 'environment', 'paired_pv_effect')), 'inconsistent frozen selection')
        csv_path = path.with_suffix('.csv.gz')
        require(meta['points'] == path.with_suffix('.csv').name, 'native CSV reference mismatch')
        provenance = meta['release_provenance']
        require(provenance['selection_sha256'] == sha256(manifest), 'selection checksum mismatch')
        with gzip.open(csv_path, 'rt', newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            require(reader.fieldnames == meta['projection']['columns']
                    and len(set(reader.fieldnames)) == len(reader.fieldnames)
                    and all(None not in r and None not in r.values() for r in rows), 'invalid trace CSV')
        require(all(r['source_row'] == str(i) and r['episode_id'] == ids[r['method']]
                    and r['selection_pool'] == '90' and r['displayed_episodes'] == '2'
                    for i, r in enumerate(rows, 2)), 'trace source mapping/episode mismatch')
        recorded_series(rows)
        require(len(rows) == meta['plotted_rows'], 'native trace row count mismatch')
        for number, row in enumerate(rows, 2):
            row.update(provenance, source_row=str(number), episode_id=ids[row['method']],
                       selection_pool=90, displayed_episodes=2)
        summary = dict(rho=exact['identity']['density'], f=exact['identity']['fraction'],
                       training_line=e['training_line'], replicate=e['replicate'],
                       paired_pv_effect=exact['selection_evidence']['paired_pv_effect'],
                       Graph=ids['Graph'], H2=ids['H2'], selection_pool=90, displayed_episodes=2)
        result.append((cell, rows, dict(selection=entry, **provenance), summary))
    return sorted(result, key=lambda item: item[0])


def build(source, output=None):
    require(output is not None, 'an explicit new output directory is required')
    require(not output.exists(), f'output already exists: {output}')
    rows, index, columns, common = load(source)
    recorded = load_recorded(source)
    sys.path.insert(0, str(RENDER))
    from style import RC, method_style
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC)
    plt.rcParams['pdf.fonttype'] = 42
    output.mkdir()  # exclusive; a failure leaves evidence and no COMPLETE receipt

    def sidecar(name, native, native_columns=None, **details):
        unique = {(r['source_file'], int(r['source_row'])): r for r in native}
        ordered = [unique[k] for k in sorted(unique)]
        write_csv(output / f'{name}.csv', ordered, native_columns or columns)
        write_json(output / f'{name}.json', dict(common, **details,
                   source_rows=[dict(file=r['source_file'], row=r['source_row'], sha256=r['source_sha256'])
                                for r in ordered]))

    def save(name, fig, axes, used, **details):
        handles = {}
        for i, ax in enumerate(axes.flat):
            ax.text(.01, 1.03, chr(97 + i), transform=ax.transAxes, weight='bold')
            hs, ls = ax.get_legend_handles_labels()
            handles.update(zip(ls, hs))
        if handles:
            fig.legend(handles.values(), handles.keys(), loc='lower center',
                       ncol=4 if len(handles) > 6 else len(handles), frameon=False)
        fig.tight_layout(rect=(0, .12 if handles else 0, 1, 1))
        sidecar(name, used, **details)
        fig.savefig(output / f'{name}.png', dpi=225)
        fig.savefig(output / f'{name}.pdf')
        plt.close(fig)

    def curve(ax, data, x, length_comparison=False, categorical=False):
        row = data[0]
        method, line = row['method'], row['training_line']
        ordinal = int(line[-1]) if line else 0
        style = method_style(method)
        style.update(linewidth=1., markersize=3)
        label = method + (f' {ordinal}' if line else '')
        if length_comparison:
            # Length owns dashes; Graph ordinals own markers here so all remain visible.
            length = int(row['road_length'])
            style['linestyle'] = '-' if length == 1000 else '--'
            if line:
                style['marker'] = ('o', 's', '^')[ordinal - 1]
            label += f' L{length}'
        elif line:
            style['linestyle'] = ('-', '--', ':')[ordinal - 1]
        xs = [f'{float(r[x]):g}' if categorical else float(r[x]) for r in data]
        ax.plot(xs, [plot_value(r) for r in data], label=label, **style)

    def primary_figure(name, metrics, method='Graph'):
        fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.8), squeeze=False, sharey='row')
        used = []
        for i, metric in enumerate(metrics):
            for j, f in enumerate(MAIN_F):
                for m, line in branches('E4'):
                    if m not in (method, 'B1', 'B2', 'H2'):
                        continue
                    data = [lookup(index, 'E4', 1000, d, f, m, line, metric) for d in RHO]
                    curve(axes[i, j], data, 'density')
                    used += data
                axes[i, j].set(title=f'f = {f:g}', xlabel='Density', xticks=RHO)
            label = {'pooled_speed': 'Background speed', 'pooled_stopped_fraction': 'Stopped fraction',
                     'normalized_local_delay': 'Local speed-deficit proxy', 'cell_time_density': 'Local density',
                     'cell_time_flow': 'Local flow', 'empty_fraction': 'Empty fraction',
                     'pv_stop/confirmed_count': 'PV stops',
                     'operational_blocking/confirmed_count': 'Operational blocking'}[metric]
            unit = 'events/episode' if metric in EVENT_METRICS else UNITS[metric]
            axes[i, 0].set_ylabel(f'{label}\n[{unit}]')
        save(name, fig, axes, used, selection='12 E4 cells; native individual lines and fixed controls; no CI')

    for name, method in (('main_local_cost', 'Graph'), ('supp_local_node', 'Node'), ('supp_local_grid', 'Grid')):
        primary_figure(name, METRICS[:2], method)
    primary_figure('main_mechanism', ('pv_stop/confirmed_count', 'operational_blocking/confirmed_count'))
    primary_figure('supp_local_density_flow', METRICS[3:5])
    primary_figure('supp_local_delay_empty', (METRICS[2], METRICS[5]))

    fig, axes = plt.subplots(4, 3, figsize=(7.1, 9), squeeze=False, sharey=True)
    used = []
    for i, d in enumerate(RHO):
        for j, f in enumerate(MAIN_F):
            ax = axes[i, j]
            for method, line in branches('E5_L1000'):
                data = [lookup(index, 'E4', 1000, d, f, method, line, SPATIAL, lag=lag) for lag in range(41)]
                curve(ax, data, 'x')
                used += data
            ax.axhline(0, color='.4', lw=.6)
            ax.set(title=f'rho={d:g}, f={f:g}', xlabel='Lag (cells)')
        axes[i, 0].set_ylabel('Connected covariance\n[occupancy squared]')
    save('supp_spatial', fig, axes, used)

    for name, metric, offsets, xlabel, ylabel in (
            ('supp_temporal', TEMPORAL[1], range(41), 'Lag (steps)', 'Normalized PV-speed\ncovariance'),
            ('supp_onset', 'primary_front_Z', range(-20, 21), 'Onset offset (steps)',
             'Front stopped background\nvehicles [count]')):
        fig, axes = plt.subplots(4, 3, figsize=(7.1, 9), squeeze=False, sharey=True)
        used = []
        for i, d in enumerate(RHO):
            for j, f in enumerate(MAIN_F):
                ax = axes[i, j]
                for method, line in branches('E5_L1000'):
                    data = [lookup(index, 'E4', 1000, d, f, method, line, metric,
                                   window='', lane='', lag=offset) for offset in offsets]
                    curve(ax, data, 'x')
                    used += data
                ax.axhline(0, color='.4', lw=.6)
                if name == 'supp_onset':
                    ax.axvline(0, color='.4', lw=.6)
                ax.set(title=f'rho={d:g}, f={f:g}', xlabel=xlabel)
            axes[i, 0].set_ylabel(ylabel)
        save(name, fig, axes, used)

    for name, fractions, lengths in (('supp_local_penetration', tuple(sorted(MAIN_F + ADDED_F)), (1000,)),
                                    ('supp_local_length', (.9, 1.), (1000, 2000))):
        fig, axes = plt.subplots(2, 3, figsize=(7.1, 5), squeeze=False, sharey='row')
        used = []
        for j, d in enumerate(RHO[1:]):
            for i, metric in enumerate(METRICS[:2]):
                for length in lengths:
                    for method, line in branches('E5_L1000'):
                        if len(lengths) == 2 and method == 'B2':
                            continue
                        data = transfer(index, method, line, metric, d, fractions, length)
                        curve(axes[i, j], data, 'fraction', len(lengths) == 2, categorical=len(lengths) == 1)
                        used += data
                axes[i, j].set(title=f'rho = {d:g}', xlabel='AV fraction (evaluated levels)' if len(lengths) == 1 else 'Fraction f')
                if len(lengths) == 2:
                    axes[i, j].set_xticks(fractions)
                else:
                    axes[i, j].tick_params(axis='x', rotation=60)
            axes[0, 0].set_ylabel('Background speed\n[cells/step]')
            axes[1, 0].set_ylabel('Stopped fraction\n[fraction]')
        save(name, fig, axes, used, line_encoding='Graph ordinal dashes, or ordinal markers when dashes encode length',
             fraction_axis='equally spaced evaluated levels, not linear' if len(lengths) == 1 else 'linear')

    fig, axes = plt.subplots(3, 1, figsize=(7.1, 9), squeeze=False)
    used, derived, matrices = [], [], []
    cells = [(d, f) for d in RHO for f in MAIN_F]
    for ordinal in (1, 2, 3):
        matrix = []
        for d, f in cells:
            values = []
            for window in WINDOWS:
                left = lookup(index, 'E4', 1000, d, f, 'Graph', f'E3graph_s{ordinal}', METRICS[2], window=window)
                right = lookup(index, 'E4', 1000, d, f, 'H2', '', METRICS[2], window=window)
                delta = difference(left, right)
                derived.append(dict(delta, density=d, fraction=f, window=window, training_line=left['training_line']))
                values.append(math.nan if delta['value'] is None else delta['value'])
                used += [left, right]
            matrix.append(values)
        matrices.append(matrix)
    limit = max((abs(r['value']) for r in derived if r['value'] is not None), default=0) or 1.
    for i, matrix in enumerate(matrices):
        ax = axes[i, 0]
        cmap = plt.get_cmap('RdBu_r').copy()
        cmap.set_bad('.8')
        im = ax.imshow(matrix, vmin=-limit, vmax=limit, cmap=cmap, aspect='auto')
        ax.set(title=f'Graph {i + 1} minus H2', yticks=range(12),
               yticklabels=[f'{d:g} / {f:g}' for d, f in cells], ylabel='rho / f',
               xticks=range(12), xticklabels=[w + (' *' if w == PRIMARY else '') for w in WINDOWS])
        ax.tick_params(axis='x', rotation=60)
        ax.grid(False)
        ax.axvline(WINDOWS.index(PRIMARY) - .5, color='black', lw=.8)
        ax.axvline(WINDOWS.index(PRIMARY) + .5, color='black', lw=.8)
        fig.colorbar(im, ax=ax, label='Speed-deficit proxy difference\n(grey = NA)', fraction=.025, pad=.02)
    save('supp_window_sensitivity', fig, axes, used, derived=derived, primary='fixed_front_2 (*)')
    write_csv(output / 'window_differences.csv', derived, list(derived[0]))

    sidecar('primary_summary', [r for r in rows if r['family'] == 'windows'
                               and r['window'] == PRIMARY and r['metric'] in METRICS])
    sidecar('events_summary', [r for r in rows if r['family'] == 'events'],
            duration_availability='n_defined / n in descriptive_duration/mean rows; no zero filling')
    sidecar('context_counts', [r for r in rows if r['family'] == 'windows' and r['metric'] in COUNTS])
    sidecar('spatial_per_lane', [r for r in rows if r['family'] == 'spatial'],
            scope='All 12 E4 cells, all methods/individual lines, all plus four physical lanes, 41 lags')

    for family in ('temporal', 'onset'):
        sidecar(f'{family}_summary', [r for r in rows if r['family'] == family],
                scope='All 36 E4/E5 scopes, all native individual branches and controls, 41 points',
                support='Native n, n_defined, support, NA and units retained; no pooling/recomputation')

    pairs, captions, table_rows = [], [], []

    for (d, f), native_rows, provenance, summary in recorded:
        name = f'recorded_rho{d:.2f}_f{f:.2f}'
        series = recorded_series(native_rows)
        fig, axes = plt.subplots(3, 2, figsize=(7.1, 5.5), squeeze=False, sharex=True, sharey='row')
        for i, metric in enumerate(TRACE_UNITS):
            for j, method in enumerate(('Graph', 'H2')):
                style = method_style(method)
                style.update(marker='', linewidth=.7)
                axes[i, j].plot(range(1, 2001), [series[method, metric][x] for x in range(1, 2001)],
                                drawstyle='steps-post' if metric == 'lane' else 'default', **style)
                axes[i, j].set(xlim=(1, 2000), title=method if i == 0 else '',
                               xlabel='Recorded step' if i == 2 else '')
                if metric == 'speed':
                    axes[i, j].set_ylim(0, 6)
                elif metric == 'lane':
                    axes[i, j].set(ylim=(-.1, 3.1), yticks=(0, 1, 2, 3))
            axes[i, 0].set_ylabel(f'{metric.replace("_", " ").capitalize()}\n[{TRACE_UNITS[metric]}]')
        save(name, fig, axes, native_rows, native_columns=list(native_rows[0]),
             recorded=provenance, notes=RECORDED_NOTE, selection_pool=90, displayed_episodes=2)
        pairs.append(summary)
        line, rep = latex(summary['training_line']), summary['replicate']
        captions.append(r'\suppfig{revision_' + name + '.pdf}{Recorded PV speed, physical lane and '
                        f'cumulative distance, $\\rho={d:.2f}$, $f={f:.2f}$; Graph line {line}, rep {rep}; '
                        'one paired illustration selected from 90 prespecified pairs (three lines by '
                        r'30 environments), two displayed episodes; not population evidence.}{fig:recorded-'
                        + f'rho{d:.2f}-f{f:.2f}' + '}')
        table_rows.append(f'{d:.2f} & {f:.2f} & {line} & {rep} & '
                          + f'{float(summary["paired_pv_effect"]):.4f}' + r' \\')
    write_csv(output / 'recorded_pairs.csv', pairs, list(pairs[0]))
    write_json(output / 'recorded_pairs.json', dict(pairs=pairs, notes=RECORDED_NOTE,
               sources=[item[2] for item in recorded]))
    (output / 'recorded_figures.tex').write_text('\n'.join(captions) + '\n')
    (output / 'recorded_pairs.tex').write_text('\n'.join([
        r'\begin{table}[htbp]\centering',
        r'\caption{Frozen representative Graph/H2 pairs for all 12 E4 cells. Each pair is selected '
        r'from all 90 prespecified pairs; two episodes are displayed. Replicate indices are 0--29. '
        r'PV-speed effects (Graph minus H2, cells per step) are descriptive, not population estimates.}\label{tab:recorded-pairs}',
        r'\begin{tabular}{rrlrr}\hline', r'$\rho$ & $f$ & Graph line & Replicate & Paired PV effect \\\hline',
        *table_rows, r'\hline\end{tabular}', r'\end{table}', '']))

    write_json(output / 'README.json', dict(notes=NOTES, figures=24, recorded=RECORDED_NOTE, continuity='Provided by global-assets: stopped fraction, lap time and availability.',
               native_columns='All columns preserved in source CSV sidecars; derived rows are separate.',
               provenance='Each JSON includes frozen progress and exact input checksums.',
               windows=list(WINDOWS), primary_window=PRIMARY))
    write_json(output / 'COMPLETE.json', dict(progress_sha256=common['progress_sha256'], scopes=36, figures=24,
               recorded=RECORDED_NOTE,
               checksums={p.name: sha256(p) for p in sorted(output.iterdir())}))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=INPUTS)
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing paths are refused')
    args = parser.parse_args()
    require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')),
            'thread environment must be 1')
    print(build(args.source.resolve(), args.output.resolve() if args.output else None), flush=True)
    return 0


if __name__ == '__main__':
    rc = 1
    try:
        rc = main()
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)  # known NumPy teardown workaround; normal GC stays enabled
