"""Hand-built compact metadata and diagnostic fixtures; no RAW or simulator."""
import copy
import csv
import json
import lzma
from pathlib import Path
import pytest
import data


def task(tid, rep, method='graph', line='line-a', fraction=.9, length=1000):
    return dict(task_id=tid, science=dict(method=method, training_line=line,
        model_id='model' if line else None, candidate_id='candidate' if line else None,
        env_steps=100, replicate=rep, seeds={k: rep for k in ('scenario', 'warmup', 'measurement', 'pv_assignment')},
        config=dict(road_length=length, density=.2, av_fraction=fraction)))


def fixture(n=5):
    tasks = {f't{r}': task(f't{r}', r) for r in range(30)}
    policies = {'line-a/candidate': dict(method='graph', model_id='model', rules=['S0', 'S_0.95'], scale=1.)}
    frozen = dict(by_id=tasks, detailed=set(), aliases=[], panels={
        'main': dict(physical_task_ids=list(tasks), policies=list(policies))})
    snapshot = dict(records=[dict(task_id=f't{r}', attempt_id='a') for r in range(n)])
    return frozen, snapshot, policies


def test_distinct_metadata_gate_and_duplicate_attempts():
    f, s, p = fixture(4)
    assert data._scan(f, s, p)['task_ids'] == []
    s['records'].append(dict(task_id='t4', attempt_id='a'))
    result = data._scan(f, s, p)
    assert len(result['task_ids']) == 5
    assert result['coverage'][0]['n'] == 5
    s['records'].append(dict(task_id='t4', attempt_id='b'))
    with pytest.raises(ValueError, match='duplicate'):
        data._scan(f, s, p)


def test_aliases_are_logical_only_and_native_identity_retained():
    f, s, p = fixture()
    scan = data._scan(f, s, p)
    identity = scan['identities']['t0']
    projected = list(data.projections(identity, f, p))
    assert {r['rule'] for r in projected} == {'S0', 'C95'}
    assert {r['task_id'] for r in projected} == {'t0'}
    assert projected[0]['method'] == 'Graph'
    assert projected[0]['native_method'] == 'graph'
    b = task('b', 0, 'B1', '', 1.)
    p = {'B1': dict(method='B1', rules=[])}
    f = dict(aliases=[dict(task_id='b', panel='main', requested_fractions=[.9, .95, 1.])],
             panels={'main': dict(physical_task_ids=['b'], policies=['B1'])})
    assert len(list(data.projections(data._identity(b, p), f, p))) == 3


def items(n=5, values=None):
    for rep in range(n):
        identity = dict(task_id=str(rep), replicate=rep)
        projection = dict(panel='main', cohort=1., rule='S0', method='Graph', training_line='a',
                          model_id='m', road_length=1000, density=.2, fraction=.9)
        yield dict(identity=identity, projections=[projection], measures=[dict(family='windows',
            metric='pooled_speed', window='fixed_front_2', value=None if values is None else values[rep],
            support=0 if values is None else 10, support_kind='vehicle_time')])


def test_empty_windows_and_four_vs_five_and_duplicate_replicates():
    assert data.build_points(items())[0]['value'] is None
    four = data.build_points(items(4, [1]*4))[0]
    assert four['status'] == 'deferred_below_min_repeats'
    five = data.build_points(items(5, [1, 2, 3, 4, 5]))[0]
    assert five['value'] == 3
    assert five['n'] == 5 and five['k_planned'] == 3
    assert five['interval_kind'] == 'descriptive_sd'
    raw = list(items())
    with pytest.raises(ValueError, match='duplicate'):
        data.build_points(raw+[raw[0]])


def test_sparse_frequency_zeros_are_only_valid_with_observed_denominator():
    raw = list(items())
    for rep, item in enumerate(raw):
        item['measures'] = [dict(family='state', metric='N_frequency', x=rep % 2,
                                x_label='N', value=1., support=10, support_kind='steps')]
    rows = data.build_points(raw)
    assert sorted(r['value'] for r in rows) == [.4, .6]
    assert all(r['n'] == 5 for r in rows)


def test_mirror_binding_and_escape_rejection(tmp_path, monkeypatch):
    import physica_test_store as store
    payload = tmp_path/'mirror'/'outputs'/'worker'/'attempt'/'result.json'
    payload.parent.mkdir(parents=True)
    payload.write_text('{}')
    ref = store.file_ref(payload)
    record = dict(result_path='outputs/worker/attempt/result.json', result_sha256=ref['sha256'], result_bytes=ref['bytes'])
    snapshot = dict(shared_dir='/not-mounted', document_sha256='unchanged')
    before = copy.deepcopy(snapshot)
    assert data.result_path(snapshot, record, tmp_path/'mirror') == payload
    assert snapshot == before
    with pytest.raises(ValueError):
        data.result_path(snapshot, dict(record, result_path='../result.json'), tmp_path/'mirror')
    link = tmp_path/'link'
    link.symlink_to(tmp_path/'mirror', target_is_directory=True)
    with pytest.raises(ValueError):
        data.result_path(snapshot, record, link)
    payload.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='pinned result changed'):
        data.result_path(snapshot, record, tmp_path/'mirror')


def test_diagnostic_scope_projection_preserves_native_values():
    analysis = dict(windows=dict(summaries=[dict(window_id='fixed_front_2', region='front', lane=2,
        group='HDV', N=0, pooled_speed=None)], transitions=[dict(window_id='fixed_front_2', region='front',
        lane=2, group='HDV', exposure_N=0, categories=['stop_to_move'], counts=[0], rates=[None])],
        state_frequencies=[dict(window_id='fixed_front_2', region='full', lane='all', group='background',
            steps=10, states=[dict(N=0, S=0, Z=0, A=41, count=10, frequency=1.)])]),
        events=dict(streams=[dict(event_id='stop0', confirmed_count=0, mean_observed_duration=None, events=[])]),
        mechanism=dict(tables=[dict(name='native_gate', phase='pre-step', denominator=0,
            sentinel_count=10, contexts=10, mean=None, states=[dict(value=0, frequency=None)])]),
        temporal=dict(records=[dict(lag=0, pair_count=10, covariance=0., value=None)]),
        event_alignment=dict(records=[dict(signal='pv_speed', offset=-1, mean=None, valid_count=0)]))
    rows = list(data.diagnostic_measures(analysis))
    assert {'windows', 'transitions', 'state', 'events', 'mechanism', 'temporal', 'onset'} <= {r['family'] for r in rows}
    speed = next(r for r in rows if r['metric'] == 'pooled_speed')
    assert speed['value'] is None and speed['lane'] == 2 and speed['vehicle_group'] == 'HDV'
    gate = next(r for r in rows if r['metric'] == 'native_gate/frequency')
    assert gate['phase'] == 'pre-step' and gate['value'] is None
    assert speed['na_reason'] == 'empty_exposure'
    temporal = next(r for r in rows if r['metric'] == 'pv_speed/value')
    assert temporal['na_reason'] == 'zero_variance'


def test_completed_validation_output_is_separate_and_keeps_candidates(tmp_path):
    root, output = tmp_path/'input', tmp_path/'output'
    root.mkdir()
    (root/'summary.json').write_text(json.dumps(dict(data_complete=True, final=True, missing=0)))
    selection = dict(groups=[dict(group=dict(architecture='graph', family='RL', training_line='line-a'),
        selection=dict(speed={'candidate': 2.}, ratios={'candidate': [dict(density=.2, fraction=.9, value=.96)]},
            thresholds=[dict(tau=.95, selection=dict(feasible=['candidate'], selected='candidate', status='selected'))]))])
    (root/'selection.json').write_text(json.dumps(selection))
    with lzma.open(root/'runs.jsonl.xz', 'wt') as f:
        for rep in range(5):
            f.write(json.dumps(dict(status='valid', target_task=task(str(rep), rep), attempt_id='attempt',
                metrics=dict(mean_speed_priority=2., mean_priority_lap_time=None), control={}))+'\n')
    manifest = data.extract_validation(output, root)
    assert manifest['dataset_kind'] == 'completed_validation'
    with (output/'points.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert all(r['panel'] == 'validation' for r in rows)
    assert {'validation_V', 'validation_R', 'validation_feasible'} <= {r['metric'] for r in rows}
    assert list(rows[0])[:len(data.POINT_COLUMNS)] == data.POINT_COLUMNS
    assert (output/'episodes.csv').exists()
    assert 'windows' in manifest['deferred_families']


@pytest.mark.parametrize('n,baseline,covered,undefined', [(5, 1., False, None), (30, 1., True, 0), (30, 0., True, 1)])
def test_ratio_domain_counts_separate_missing_from_zero_baseline(n, baseline, covered, undefined):
    frozen, snapshot, policies = fixture(n)
    policies['B1'] = dict(method='B1', rules=[])
    for rep in range(30):
        entry = task('base'+str(rep), rep, method='B1', line='')
        frozen['by_id'][entry['task_id']] = entry
        if rep < n:
            snapshot['records'].append(dict(task_id=entry['task_id'], attempt_id='a'))
    frozen['panels']['main'] = dict(physical_task_ids=list(frozen['by_id']), policies=list(policies))
    scan = data._scan(frozen, snapshot, policies)
    episodes = [dict(scan['identities'][tid], mean_speed_priority=3.,
                     flow_background=baseline if tid.startswith('base') else 2.) for tid in scan['task_ids']]
    points = data.comparison_points(episodes, scan)
    pv = data.build_points(dict(identity=row,
        projections=list(data.projections(row, frozen, policies)),
        measures=list(data.scalar_measures({'metrics': {'mean_speed_priority': row['mean_speed_priority']}})))
        for row in episodes)
    match_fields = ('panel', 'cohort', 'rule', 'road_length', 'density', 'fraction',
                    'method', 'training_line', 'model_id', 'candidate_id', 'x', 'x_label')
    for ratio in (p for p in points if p['metric'] == 'R'):
        matches = [p for p in pv if all(p.get(k, '') == ratio.get(k, '') for k in match_fields)]
        assert len(matches) == 1
        assert ratio['policy_id'] == matches[0]['policy_id']
        if ratio['training_line']:
            assert ratio['model_id'] == 'model' and ratio['candidate_id'] == 'candidate'
    assert all(p.get('k') is None and p.get('k_planned') is None for p in points if p['method'] == 'B1')
    domain = [p for p in points if p['method'] == 'Graph' and p['rule'] == 'S0' and p['x'] == 'full54']
    assert next(p for p in domain if p['metric'] == 'undefined_count')['value'] == undefined
    if not covered:
        assert all(p['value'] is None and p['status'] == 'missing_planned_cells' for p in domain)
    elif undefined:
        assert next(p for p in domain if p['metric'] == 'R_min')['value'] is None
        assert all(p['status'] == 'undefined_ratio' for p in domain)


def test_functional_estimands_preserve_cell_and_domain_coordinates(monkeypatch):
    import inference
    bootstrap = inference.bootstrap
    monkeypatch.setattr(inference, 'bootstrap', lambda *args, **kwargs:
                        bootstrap(*args, **kwargs, B=17, technical=True))
    frozen, snapshot, policies = fixture(30)
    policies['line-a/candidate']['rules'] = ['S_0.95']
    for method, line in [('graph', 'b'), ('graph', 'c'), ('H2', ''), ('B1', '')]:
        policy = line+'/candidate' if line else 'H/C=2' if method == 'H2' else method
        policies[policy] = dict(method=method, model_id=line or None,
                                rules=['S_0.95'] if line else [], scale=1.)
        for rep in range(30):
            entry = task((line or method)+str(rep), rep, method=method, line=line)
            entry['science']['model_id'] = line or None
            frozen['by_id'][entry['task_id']] = entry
    for tid, entry in list(frozen['by_id'].items()):
        other = copy.deepcopy(entry)
        other['task_id'] = tid+'-dense'
        other['science']['config']['density'] = .6
        frozen['by_id'][other['task_id']] = other
    frozen['panels']['main'] = dict(physical_task_ids=list(frozen['by_id']), policies=list(policies))
    snapshot['records'] = [dict(task_id=tid, attempt_id='a') for tid in frozen['by_id']]
    scan = data._scan(frozen, snapshot, policies)
    episodes = [dict(row, mean_speed_priority=2.+10*row['density'] if row['method'] == 'Graph' else 1.,
                     flow_background=2*row['density'] if row['method'] == 'Graph' else 1.)
                for row in scan['identities'].values()]
    points = data.comparison_points(episodes, scan, final=True)
    cells = sorted((p for p in points if p['metric'] == 'paired_PV_speed_difference_final'),
                   key=lambda p: p['x'])
    assert len(cells) == 2
    assert len({p['estimand'] for p in cells}) == 1
    assert [p['x'] for p in cells] == [p['density'] for p in cells] == [.2, .6]
    assert all(p['fraction'] == .9 and p['x_label'] == 'density' for p in cells)
    assert [p['value'] for p in cells] == pytest.approx([3., 7.])
    grids = [p for p in points if p['metric'] == 'grid_PV_speed_difference']
    assert len({p['estimand'] for p in grids}) == 1
    domains = {p['x']: p for p in grids}
    assert [domains[d]['value'] for d in ('selection12', 'transfer42', 'full54')] == pytest.approx([3., 7., 5.])
    assert all(p['x_label'] == 'planned domain' for p in grids)
    for p in cells + [domains[d] for d in ('selection12', 'transfer42', 'full54')]:
        assert p['status'] == 'complete' and p['interval_kind'] != 'none'
        assert p['ci_low'] is not None and p['ci_high'] is not None
        assert p['n'] == 30 and p['k'] == 3
    for metric, expected in [('R_min', [.4, 1.2]), ('pass95_count', [0, 1]),
                             ('fail95_count', [1, 0]), ('undefined_count', [0, 0])]:
        rows = [p for p in points if p['metric'] == metric and p['method'] == 'Graph'
                and p['training_line'] == 'line-a']
        assert rows and len({p['estimand'] for p in rows}) == 1
        domains = {p['x']: p for p in rows}
        assert [domains[d]['value'] for d in ('selection12', 'transfer42')] == pytest.approx(expected)
        assert all(p['x_label'] == 'planned domain' for p in rows)


@pytest.mark.parametrize('field', ['model_id', 'candidate_id', 'policy_id'])
def test_ratio_rejects_conflicting_planned_line_identity(field):
    frozen, snapshot, policies = fixture()
    scan = data._scan(frozen, snapshot, policies)
    # An unobserved repeat in another cell must still agree with the line.
    scan['identities']['t29']['density'] = .3
    scan['identities']['t29'][field] = 'conflicting'
    policies['conflicting'] = policies['line-a/candidate']
    frozen['panels']['main']['policies'].append('conflicting')
    with pytest.raises(ValueError, match='conflicting frozen per-line ratio identities'):
        data.comparison_points([], scan)


def test_missing_training_line_and_full_grid_remain_na():
    f, s, p = fixture()
    # Three planned Graph lines, only one observed, plus a fixed H2 reference.
    for line in ('b', 'c'):
        policy = line+'/candidate'
        p[policy] = dict(method='graph', model_id=line, rules=['S0', 'S_0.95'], scale=1.)
        for rep in range(30):
            t = task(line+str(rep), rep, line=line)
            t['science']['model_id'] = line
            f['by_id'][t['task_id']] = t
    p['H/C=2'] = dict(method='H2', rules=['H_0.95'])
    for rep in range(30):
        t = task('h'+str(rep), rep, method='H2', line='')
        f['by_id'][t['task_id']] = t
    f['panels']['main'] = dict(physical_task_ids=list(f['by_id']), policies=list(p))
    scan = data._scan(f, s, p)
    rows = [dict(scan['identities'][tid], mean_speed_priority=3., flow_background=1.) for tid in scan['task_ids']]
    points = data.comparison_points(rows, scan)
    contrasts = [r for r in points if r['family'] == 'contrasts' and r['method'] == 'Graph−H2']
    assert contrasts and all(r['value'] is None for r in contrasts)
    assert all(r.get('ci_low') is None for r in contrasts)


def test_inapplicable_mechanism_does_not_disappear_from_episode_n():
    raw = list(items())
    for rep, item in enumerate(raw):
        item['measures'] = [] if rep == 4 else [dict(family='mechanism', metric='gate/frequency',
            x=0, x_label='native category', value=1., support=10, support_kind='applicable_contexts')]
    point = data.build_points(raw)[0]
    assert point['n'] == 5 and point['n_defined'] == 4
    assert point['value'] == 1.


def test_family_means_keep_missing_frozen_roster_and_length_separate():
    f, s, p = fixture()
    other = task('other', 0, line='b', length=2000)
    other['science']['model_id'] = 'other-model'
    p['b/candidate'] = dict(method='graph', model_id='other-model', rules=['S0'])
    f['by_id']['other'] = other
    f['panels']['main']['physical_task_ids'].append('other')
    f['panels']['main']['policies'].append('b/candidate')
    scan = data._scan(f, s, p)
    rows = [dict(scan['identities'][tid], mean_speed_priority=3.) for tid in scan['task_ids']]
    points = data.family_points(rows, scan)
    speed = [r for r in points if r['metric'] == 'mean_speed_priority']
    assert all(r['value'] == 3. for r in speed if r['road_length'] == 1000)
    assert all(r['value'] is None for r in speed if r['road_length'] == 2000)


def empty_analysis():
    return dict(windows=dict(summaries=[], transitions=[], state_frequencies=[]),
                events=dict(streams=[]))


def test_native_laps_availability_and_speed_distribution():
    raw = list(items())
    for rep, item in enumerate(raw):
        analysis = empty_analysis()
        analysis['pv'] = dict(completed_laps=2 if rep == 0 else 0,
                              mean_completed_lap_time=12. if rep == 0 else None,
                              speed_counts=[3, 2, 0, 0, 0, 0, 0])
        item['measures'] = list(data.diagnostic_measures(analysis))
    points = data.build_points(raw)
    availability = next(p for p in points if p['metric'] == 'lap_availability')
    assert availability['value'] == .2
    assert availability['support'] == 1 and availability['n'] == 5
    assert availability['n_planned'] == 30
    duration = next(p for p in points if p['metric'] == 'mean_completed_lap_time')
    assert duration['value'] == 12 and duration['n_defined'] == 1
    assert duration['support'] == 2 and duration['na_reason'] == 'no_completed_laps'
    speed = [p for p in points if p['metric'] == 'pv_speed_counts']
    assert len(speed) == 7 and all(p['units'] == 'count' for p in speed)


def test_sparse_event_bins_include_zero_event_episodes_but_not_unrequested():
    raw = list(items(6))
    for rep, item in enumerate(raw):
        analysis = empty_analysis()
        if rep < 5:
            events = [dict(duration_steps=3, observed_duration_steps=3,
                           left_censored=False, right_censored=False)] if rep == 0 else []
            analysis['events']['streams'] = [dict(event_id='stop0', min_duration=2,
                speed_threshold=0, lane=2, confirmed_count=len(events), events=events,
                descriptive_duration=dict(n=len(events), mean=3. if events else None),
                exit_reason_counts={'pv_restart': 1} if events else {})]
        item['measures'] = list(data.diagnostic_measures(analysis))
    points = data.build_points(raw)
    bins = [p for p in points if '/duration_steps/' in p['metric']]
    assert len(bins) == 1
    assert bins[0]['value'] == .2 and bins[0]['n_defined'] == 5 and bins[0]['n'] == 6
    # With exactly the five observed episodes the required regression is n=5.
    observed = next(p for p in data.build_points(raw[:5]) if '/duration_steps/' in p['metric'])
    assert observed['value'] == .2 and observed['n'] == 5
    exits = next(p for p in points if p['metric'].endswith('/exit_reason_counts'))
    assert exits['value'] == .2
    assert not any(p['metric'].endswith(('/min_duration', '/speed_threshold', '/lane')) for p in points)


def test_censored_durations_and_native_na_reasons():
    analysis = empty_analysis()
    analysis['events']['streams'] = [dict(event_id='stop0', confirmed_count=1,
        descriptive_duration=dict(n=0, mean=None), events=[dict(duration_steps=None,
            observed_duration_steps=4, left_censored=True, right_censored=True)])]
    measures = list(data.diagnostic_measures(analysis))
    raw = list(items())
    for item in raw:
        item['measures'] = measures
    points = data.build_points(raw)
    assert not any('/duration_steps/' in p['metric'] for p in points)
    assert next(p for p in points if '/observed_duration_steps/' in p['metric'])['value'] == 1
    assert next(p for p in points if '/descriptive_duration/' in p['metric'])['status'] == 'all_censored'
    for status in ('not_applicable', 'undefined_zero_eligible'):
        for item in raw:
            item['measures'] = list(data.scalar_measures(dict(control=dict(
                request_opportunity_use=None, opportunity_use_status=status))))
        point = data.build_points(raw)[0]
        assert point['status'] == status and point['na_reason'] == status


def test_sparse_mechanism_empty_domain_has_no_applicable_zero():
    raw = list(items())
    for rep, item in enumerate(raw):
        analysis = empty_analysis()
        analysis['mechanism'] = dict(tables=[dict(name='pre_pv_leader_gap', phase='pre_step',
            denominator=0 if rep == 4 else 10, contexts=10, sentinel_count=10 if rep == 4 else 0,
            mean=None if rep == 4 else rep,
            states=[] if rep == 4 else [dict(value=rep, frequency=1.)])])
        item['measures'] = list(data.diagnostic_measures(analysis))
    points = [p for p in data.build_points(raw) if p['metric'].endswith('/frequency')]
    assert len(points) == 4
    assert all(p['value'] == .25 and p['n_defined'] == 4 and p['n'] == 5 for p in points)


@pytest.mark.parametrize('metric,units', [
    ('pre_pv_leader_speed/frequency', 'fraction'),
    ('pre_pv_leader_speed/mean', 'cells/step'),
    ('pre_pv_leader_type/mean', 'native category code'),
    ('pre_pv_leader_gap/mean', 'cells'),
    ('lane_change_rate_background', 'changes/vehicle-time'),
    ('lane_change_rate_priority', 'changes/step'), ('speed_floor_steps', 'steps'),
    ('rates', 'transitions/vehicle-time'), ('time_mean_flow', 'vehicles/step'),
    ('cell_time_density', 'vehicles/cell'), ('N', 'vehicle-time'),
    ('A', 'cell-time'), ('speed_deficit_sum', 'vehicle-cells'),
    ('stop0/confirmed_count', 'count'), ('pv_speed_counts', 'count'),
    ('spatial_hist_speed0_count', 'count'),
    ('stop0/descriptive_duration/mean', 'steps'),
    ('connected_occupancy_covariance', 'occupancy squared'),
    ('pv_speed/value', 'normalized covariance'),
    ('applied_per_step', 'interventions/step'), ('applied_per_request', 'fraction'),
])
def test_native_units(metric, units):
    assert data._units(metric) == units


def test_validation_frozen_line_cohort_selected_steps_and_terminal_budget(tmp_path):
    root, frozen, output = tmp_path/'input', tmp_path/'freeze', tmp_path/'output'
    root.mkdir()
    frozen.mkdir()
    (root/'summary.json').write_text(json.dumps(dict(data_complete=True, final=True, missing=0)))
    (root/'selection.json').write_text(json.dumps(dict(groups=[dict(
        group=dict(architecture='graph', family='RL', training_line='line-a'),
        selection=dict(speed={'early': 2., 'terminal': 1.}, ratios={}, thresholds=[]))])))
    (frozen/'policies.json').write_text(json.dumps({'selected': dict(
        group=dict(training_line='line-a'), scale=.5, model_id='selected-model')}))
    (frozen/'selections.csv').write_text('training_line,rule,candidate_id,env_steps,model_id,status\n'
                                      'line-a,S0,early,100,selected-model,selected\n')
    with lzma.open(root/'runs.jsonl.xz', 'wt') as f:
        for candidate, steps in [('early', 100), ('terminal', 20004864)]:
            for rep in range(5):
                t = task(candidate+str(rep), rep)
                t['science'].update(candidate_id=candidate, env_steps=steps, model_id=candidate+'-model')
                f.write(json.dumps(dict(status='valid', target_task=t, attempt_id='a',
                                       metrics=dict(mean_speed_priority=2.), control={}))+'\n')
    data.extract_validation(output, root, frozen_root=frozen)
    with (output/'points.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert {r['cohort'] for r in rows} == {'0.5'}
    assert {r['candidate_id'] for r in rows} == {'early', 'terminal'}
    assert {r['model_id'] for r in rows} == {'early-model', 'terminal-model'}
    with (output/'selections.csv').open() as f:
        selections = list(csv.DictReader(f))
    chosen = next(r for r in selections if r['row_kind'] == 'frozen_selection')
    assert chosen['selected_env_steps'] == '100' and chosen['terminal_env_steps'] == '20004864'
    assert chosen['terminal_budget_source'] == 'validation_pool_terminal_boundary'
    assert chosen['continuity'] == 'not_established' and chosen['runtime_seconds'] == ''


def illustration_fixture():
    tasks, policies = {}, {'H/C=2': dict(method='H2', rules=[])}
    for line in ('a', 'b', 'c'):
        policies[line] = dict(method='graph', model_id=line, rules=['S0'])
        for rep in range(30):
            t = task(line+str(rep), rep, line=line)
            t['science']['model_id'] = line
            tasks[t['task_id']] = t
    for rep in range(30):
        t = task('h'+str(rep), rep, method='H2', line='')
        tasks[t['task_id']] = t
    frozen = dict(by_id=tasks, detailed=set(tasks), aliases=[], panels={
        'E4_mechanisms': dict(physical_task_ids=list(tasks), policies=list(policies))})
    snapshot = dict(records=[dict(task_id=tid, attempt_id='a', result_sha256='recorded') for tid in tasks],
                    coverage=dict(complete=False))
    scan = data._scan(frozen, snapshot, policies)
    episodes = [dict(identity, mean_speed_priority=2. if identity['method'] == 'Graph' else 1.)
                for identity in scan['identities'].values()]
    return episodes, scan


def test_illustration_export_native_trace_maps_and_cell_evidence(tmp_path, monkeypatch):
    from contextlib import contextmanager
    import numpy as np
    import physica_test_store as store
    from physica_mc import postprocess
    episodes, scan = illustration_fixture()
    scalar = data.selected_illustrations(episodes, scan)[0]
    assert scalar['status'] == 'deferred' and scalar['reason'] == 'recorded_export_not_requested'
    assert 'pair' not in scalar and 'paths' not in scalar
    partial = data.selected_illustrations(episodes[:-1], scan)[0]
    assert partial['status'] == 'deferred' and partial['reason'] == 'requires_all90_pairs'
    arrays = dict(step=np.array([0, 1, 1000, 2000]),
                  local_speed=np.full((3, 41, 4), -1, dtype=np.int8),
                  local_type=np.full((3, 41, 4), -1, dtype=np.int8))
    trace = dict(step=[0, 1, 1000, 2000], speed=[2, 1, 0, 3], lane=[0, 0, 1, 1],
                 pos=[17, 18, 18, 21], path=[17, 18, 18, 21],
                 cumulative_distance=[0, 1, 1, 4], includes_t0_context=True)
    calls = []
    @contextmanager
    def materialize(path, output, *, expected_task, required_profile):
        assert required_profile == 'mechanism'
        yield None, expected_task, arrays
    def analyze(payload, recorded, profile):
        assert recorded is arrays and profile == {'frozen': 'profile'}
        calls.append(payload['task_id'])
        return dict(episode=dict(task_id=payload['task_id']), pv=dict(trace=trace))
    monkeypatch.setattr(store, 'materialize_episode', materialize)
    monkeypatch.setattr(postprocess, 'analyze_episode', analyze)
    monkeypatch.setattr(data, 'result_path', lambda *args: Path('unused'))
    entry = data.selected_illustrations(episodes, scan, output_dir=tmp_path/'illustrations',
                                       analysis_profile={'frozen': 'profile'})[0]
    assert entry['mode'] == 'final' and entry['selection_evidence']['n'] == 90
    assert entry['selection_evidence']['scope'] == 'complete_E4_cell'
    assert len(calls) == 2 and set(entry['pair']) == {'Graph', 'H2'}
    for record in entry['pair'].values():
        assert not Path(record['analysis_path']).is_absolute()
        saved = json.loads((tmp_path/record['analysis_path']).read_text())
        assert saved['pv']['trace'] == trace
        assert saved['episode']['task_id'] == record['episode_id']
        with np.load(tmp_path/record['arrays_path']) as maps:
            assert set(maps.files) == {'local_speed', 'local_type', 'map_steps'}
            np.testing.assert_array_equal(maps['map_steps'], arrays['step'][1:])
            np.testing.assert_array_equal(maps['local_speed'], arrays['local_speed'])


def test_cached_points_partition_matches_all_groups_with_aliases_and_missing_bins(tmp_path, monkeypatch):
    import gzip
    from collections import defaultdict
    tasks, policies, detailed, aliases = {}, {}, set(), []
    for name, model, scale, profile in [('b1', '', 1., 'full12'),
                                       ('a', 'model-a', .5, 'full12'),
                                       ('b', 'model-b', 2., 'full12'),
                                       ('c', 'model-a', .5, 'mechanism')]:
        policy = name if model else 'B1'
        policies[policy] = dict(method='graph' if model else 'B1', model_id=model,
                                rules=['S0', 'S_0.95'] if model else [], scale=scale)
        for rep in range(5):
            tid = name+str(rep)
            t = task(tid, rep, method='graph' if model else 'B1',
                     line='same-line' if model else '', fraction=.9 if model else 1.)
            t['science']['model_id'] = model or None
            tasks[tid] = t
            if profile == 'mechanism':
                detailed.add(tid)
            if not model:
                aliases.extend(dict(task_id=tid, panel=panel, requested_fractions=[.9, .95, 1.])
                               for panel in ('main', 'E4_mechanisms', 'W7_S0'))
    frozen = dict(by_id=tasks, detailed=detailed, aliases=aliases, panels={
        panel: dict(physical_task_ids=list(tasks), policies=list(policies))
        for panel in ('main', 'E4_mechanisms', 'W7_S0')})
    scan = data._scan(frozen, dict(records=[dict(task_id=tid, attempt_id='a') for tid in tasks]), policies)
    paths, all_items = defaultdict(list), []
    for tid, identity in scan['identities'].items():
        rep = identity['replicate']
        measures = [dict(family='state', metric='N_frequency', x=rep % 2,
                         x_label='N', value=1., support=10, support_kind='steps')]
        if rep < 4:
            measures.append(dict(family='mechanism', metric='gate/frequency',
                                 x=0, x_label='native category', value=1., support=10,
                                 support_kind='applicable_contexts'))
        measures.append(dict(family='windows', metric='pooled_speed', value=None,
                             support=0, na_reason='empty_exposure'))
        item = dict(identity=identity, measures=measures)
        path = tmp_path/(tid+'.json.gz')
        with gzip.open(path, 'wt') as stream:
            json.dump(item, stream)
        paths[identity['group_id']].append(path)
        all_items.append(dict(item, projections=list(data.projections(identity, frozen, policies))))
    expected = data.build_points(all_items)
    original = data.build_points
    visited = []
    def build(episodes, **kwargs):
        episodes = list(episodes)
        gids = {item['identity']['group_id'] for item in episodes}
        assert len(gids) == 1
        visited.append(next(iter(gids)))
        return original(episodes, **kwargs)
    monkeypatch.setattr(data, 'build_points', build)
    stream = data._cached_points(dict(reversed(list(paths.items()))), frozen, policies, min_repeats=5)
    assert visited == []
    first = next(stream)
    assert visited == [min(paths)]
    actual = [first, *stream]
    assert visited == sorted(paths)
    # Compare full keyed rows, including exact values, status and support.
    assert sorted(json.dumps(r, sort_keys=True) for r in actual) == sorted(
        json.dumps(r, sort_keys=True) for r in expected)
    assert len(paths) == 4
    assert {r['fraction'] for r in actual if r['method'] == 'B1'} == {.9, .95, 1.}
    assert {r['panel'] for r in actual if r['method'] == 'B1'} == {'main', 'E4', 'W7'}
    assert {r['value'] for r in actual if r['family'] == 'state'} == {.4, .6}
    assert all(r['n_defined'] == 4 and r['n'] == 5 for r in actual if r['family'] == 'mechanism')
    assert all(r['status'] == 'empty_exposure' for r in actual if r['family'] == 'windows')


def test_csv_explicit_columns_writes_each_row_before_requesting_next(tmp_path, monkeypatch):
    written = []
    original = csv.DictWriter
    def writer(*args, **kwargs):
        instance = original(*args, **kwargs)
        write = instance.writerow
        def record(row):
            result = write(row)
            written.append(row)
            return result
        instance.writerow = record
        return instance
    monkeypatch.setattr(csv, 'DictWriter', writer)
    class Once:
        used = False
        def __iter__(self):
            assert not self.used
            self.used = True
            for i in range(3):
                # Header plus all prior rows must already have been written.
                assert len(written) == i+1
                yield dict(value=i, metadata={'i': i})
        def __len__(self):
            raise AssertionError('explicit-column export must not materialize rows')
    path = tmp_path/'rows.csv'
    assert data._csv(path, Once(), ['value', 'metadata']) == 3
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r['value'] for r in rows] == ['0', '1', '2']
    assert json.loads(rows[-1]['metadata']) == {'i': 2}


def test_publish_consumes_points_once_and_records_counts(tmp_path):
    class Once:
        used = False
        def __iter__(self):
            assert not self.used
            self.used = True
            yield dict(family='mobility', value=0)
            yield dict(family='windows', value=None)
        def __len__(self):
            raise AssertionError('points must stream')
    manifest = data._publish(tmp_path, manifest={'mode': 'complete'}, points=Once(),
                             episodes=iter([{'task_id': 'a'}]), coverage=iter([{'n': 5}]))
    assert manifest['row_counts'] == {'points.csv': 2, 'episodes.csv': 1,
                                      'coverage.csv': 1, 'selections.csv': 0}
    assert 'mobility' not in manifest['deferred_families']
    assert 'windows' in manifest['deferred_families']
    assert json.loads((tmp_path/'manifest.json').read_text()) == manifest


@pytest.mark.parametrize('failed_table', ['points', 'coverage'])
def test_publish_failure_does_not_replace_manifest(tmp_path, failed_table):
    previous = '{"mode": "partial_exploratory"}\n'
    (tmp_path/'manifest.json').write_text(previous)
    (tmp_path/'points.csv').write_text('previous points\n')
    def broken():
        yield dict(family='mobility', value=1)
        raise ValueError('interrupted rows')
    kwargs = dict(points=iter([dict(family='mobility', value=1)]), coverage=())
    kwargs[failed_table] = broken()
    with pytest.raises(ValueError, match='interrupted rows'):
        data._publish(tmp_path, manifest={'mode': 'complete'}, episodes=(), **kwargs)
    assert (tmp_path/'manifest.json').read_text() == previous
    if failed_table == 'points':
        assert (tmp_path/'points.csv').read_text() == 'previous points\n'


@pytest.mark.parametrize('method', ['graph', 'node', 'grid'])
def test_mixed_profile_roster_fixed_h2_and_stale_cache(tmp_path, method):
    import gzip
    tasks, policies = {}, {'H/C=2': dict(method='H2', rules=['H_0.93', 'H_0.95'])}
    detailed = set()
    for line in ('a', 'b', 'c', ''):
        if line:
            policies[line] = dict(method=method, model_id=line, rules=['S0', 'S_0.95'])
        for rep in range(5):
            tid = (line or 'h') + str(rep)
            t = task(tid, rep, method=method if line else 'H2', line=line)
            t['science']['model_id'] = line or None
            tasks[tid] = t
            if line in ('b', 'c'):
                detailed.add(tid)
    frozen = dict(by_id=tasks, detailed=detailed, aliases=[], panels={
        panel: dict(physical_task_ids=list(tasks), policies=list(policies))
        for panel in ('main', 'E4_mechanisms', 'W7_S0', 'E5_fractions', 'E5_length')})
    scan = data._scan(frozen, dict(records=[dict(task_id=t, attempt_id='a') for t in tasks]), policies)
    rows = [dict(i, mean_speed_priority=2. if not i['training_line'] else 5.,
                 flow_background=1.) for i in scan['identities'].values()]
    family = [p for p in data.family_points(rows, scan) if p['metric'] == 'mean_speed_priority']
    assert family and all(p['value'] == 5. and p['n'] == 5 and p['k'] == 3
                          and p['profile'] == 'mixed' for p in family)
    assert len(family) == 6
    missing = [r for r in rows if r['training_line'] != 'c']
    assert all(p['value'] is None and p['n'] == 0 for p in data.family_points(missing, scan))
    four = [r for r in rows if not (r['training_line'] == 'c' and r['replicate'] == 4)]
    assert all(p['value'] is None and p['n'] == 4 for p in data.family_points(four, scan))
    contrasts = [p for p in data.comparison_points(rows, scan)
                 if p['method'] == data.METHODS[method] + '−H2' and p['metric'] == 'paired_PV_speed_difference']
    assert contrasts and all(p['value'] == 3. and p['n'] == 5 for p in contrasts)
    paths = []
    for row in rows:
        if row['training_line']:
            projected = list(data.projections(row, frozen, policies))
            assert all(p['rule'] == 'S0' for p in projected if p['panel'] != 'main')
            continue
        projected = list(data.projections(row, frozen, policies))
        assert len(projected) == 5 and {p['rule'] for p in projected} == {'fixed'}
        path = tmp_path / (row['task_id'] + '.json.gz')
        with gzip.open(path, 'wt') as stream:
            json.dump(dict(identity=row, projections=[dict(p, rule='H_0.93') for p in projected],
                           measures=list(data.scalar_measures({'metrics': {'mean_speed_priority': 2.}}))), stream)
        paths.append(path)
    cached = list(data._cached_points({'h': paths}, frozen, policies, min_repeats=5))
    assert len(cached) == 5
    assert all(p['rule'] == 'fixed' and p['value'] == 2. and p['n'] == 5 for p in cached)


def test_fresh_rules_preserve_main_sensitivity_and_refresh_stale_cache(tmp_path):
    import gzip
    frozen, snapshot, policies = fixture()
    rules = ['S0', 'S_0.93', 'S_0.95', 'S_0.97', 'S_0.99']
    policies['line-a/candidate']['rules'] = rules
    frozen['panels'].update({panel: dict(frozen['panels']['main']) for panel in
                            ('E4_mechanisms', 'W7_S0', 'E5_fractions', 'E5_length')})
    scan = data._scan(frozen, snapshot, policies)
    paths, episodes = [], []
    for tid in scan['task_ids']:
        identity = scan['identities'][tid]
        episodes.append(dict(identity, mean_speed_priority=2., flow_background=1.))
        path = tmp_path / (tid + '.json.gz')
        with gzip.open(path, 'wt') as stream:
            json.dump(dict(identity=identity,
                           projections=[dict(identity, panel='main', rule=r) for r in rules],
                           measures=list(data.scalar_measures({'metrics': {'mean_speed_priority': 2.}}))), stream)
        paths.append(path)
    expected = {('main', r) for r in ('S0', 'S_0.93', 'C95', 'S_0.97', 'S_0.99')} | {
        (panel, 'S0') for panel in ('E4', 'W7', 'E5_L1000', 'E5_L2000')}
    for rows in (data._cached_points({'g': paths}, frozen, policies, min_repeats=5),
                 data.family_points(episodes, scan), data.comparison_points(episodes, scan)):
        rows = list(rows)
        assert {(r['panel'], r['rule']) for r in rows} == expected
    assert policies['line-a/candidate']['rules'] == rules
