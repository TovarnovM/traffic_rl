"""Read-only paper consumer. Native episodes stay physical; aliases are projections.

All numerical imports are lazy so metadata scanning also works on the host.
Outputs are deterministic CSV/JSON plus compact per-episode gzip caches. No
collector, simulator, model inference, or mutable native state is used here.
"""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import lzma
import math
from pathlib import Path

POINT_COLUMNS = 'family,panel,cohort,rule,method,training_line,model_id,road_length,density,fraction,metric,window,region,lane,vehicle_group,phase,x,x_label,value,sd,ci_low,ci_high,interval_kind,n,n_planned,k,k_planned,support,support_kind,status,units,estimand'.split(',')
EXTRA_COLUMNS = ["n_defined", 'native_method', 'policy_id', 'profile', 'candidate_id', 'env_steps', 'na_reason']
FAMILIES = 'mobility background ratio contrasts laps control selection coverage windows transitions state events mechanism spatial temporal onset profiles'.split()
PANELS = {'main': 'main', 'E4_mechanisms': 'E4', 'W7_S0': 'W7',
          'E5_fractions': 'E5_L1000', 'E5_length': 'E5_L2000'}
METHODS = {'graph': 'Graph', 'node': 'Node', 'grid': 'Grid',
           'left_if_safe_original': 'original'}
CACHE_VERSION = 2


def _json(path):
    with Path(path).open() as f:
        return json.load(f)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def _csv(path, rows, columns=None):
    if columns is None:
        rows = list(rows)
        columns = sorted({key for row in rows for key in row}) or ['status']
    count = 0
    tmp = Path(path).with_suffix('.csv.tmp')
    with tmp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v
                             for k, v in row.items()})
            count += 1
    tmp.replace(path)
    return count


def _policy(science, policies):
    model = science.get('model_id')
    for key, policy in policies.items():
        if model and policy.get('model_id') == model:
            return key, policy
    method = science['method']
    key = 'H/C=2' if method == 'H2' else method
    return key, policies.get(key, {'method': method, 'rules': [], 'cohort': 'control'})


def _identity(task, policies=None, profile=''):
    s = task['science']
    c = s['config']
    policy_id, policy = _policy(s, policies or {})
    method = policy.get('method', s['method'])
    return dict(task_id=task['task_id'], native_method=s['method'],
                method=METHODS.get(method, method), policy_id=policy_id,
                training_line=s.get('training_line') or '', model_id=s.get('model_id') or '',
                candidate_id=s.get('candidate_id') or '', env_steps=s.get('env_steps'),
                road_length=c['road_length'], density=c['density'], fraction=c['av_fraction'],
                replicate=s['replicate'], profile=profile, cohort=policy.get('scale', 1.0),
                environment=_digest({'L': c['road_length'], 'density': c['density'],
                    'seeds': {k: s['seeds'][k] for k in ('scenario', 'warmup', 'measurement', 'pv_assignment')}}))


def scan_metadata(frozen_root, index, *, min_repeats=5, groups=None):
    """Metadata-only frozen coverage, distinct-repeat gate, and pinned snapshot.

    Optional groups are exact physical group IDs returned in ``coverage``.
    Returns admitted immutable documents for extraction; no archives are opened.
    """
    import dataset
    import physica_test_plan as plan
    if min_repeats < 5:
        raise ValueError('min_repeats cannot be below five')
    frozen = plan.load_freeze(frozen_root, verify=False)
    snapshot = dataset.load_snapshot(index)
    if snapshot['freeze_sha256'] != plan.FREEZE_SHA256 or snapshot['plan_id'] != plan.PLAN_ID:
        raise ValueError('snapshot belongs to another freeze')
    policies = _json(Path(frozen_root)/'policies.json')
    return _scan(frozen, snapshot, policies, min_repeats, groups)


def _scan(frozen, snapshot, policies, min_repeats=5, groups=None):
    accepted = {}
    for record in snapshot['records']:
        tid = record['task_id']
        if tid in accepted:
            raise ValueError('duplicate accepted physical task/attempt')
        if tid not in frozen['by_id']:
            raise ValueError('unknown accepted task')
        accepted[tid] = record
    coverage, identities, members = {}, {}, defaultdict(list)
    for tid, task in frozen['by_id'].items():
        profile = 'mechanism' if tid in frozen['detailed'] else 'full12'
        if tid in accepted and accepted[tid].get("observer", profile) != profile:
            raise ValueError("accepted profile differs from frozen task")
        identity = _identity(task, policies, profile)
        science = {k: v for k, v in task['science'].items() if k not in ('replicate', 'seeds')}
        gid = _digest(dict(science=science, profile=profile))
        identity['group_id'] = gid
        identities[tid] = identity
        members[gid].append(tid)
    wanted = set(members) if groups is None else set(groups)
    if not wanted <= members.keys():
        raise ValueError('unknown selected physical group')
    eligible = set()
    for gid, tids in sorted(members.items()):
        repeats = {identities[t]['replicate'] for t in tids if t in accepted}
        planned = {identities[t]['replicate'] for t in tids}
        if len(planned) != len(tids):
            raise ValueError('duplicate frozen physical replicate')
        admitted = len(repeats) >= min_repeats
        row = dict(identities[tids[0]], group_id=gid, n=len(repeats), n_planned=len(planned),
                   status='eligible' if admitted else 'deferred_below_min_repeats', selected=gid in wanted)
        coverage[gid] = row
        if admitted and gid in wanted:
            eligible.update(t for t in tids if t in accepted)
    return dict(frozen=frozen, snapshot=snapshot, policies=policies, identities=identities,
                coverage=list(coverage.values()), task_ids=sorted(eligible), min_repeats=min_repeats)


def result_path(snapshot, record, staged_root=None):
    """Resolve an exact accepted file in shared storage or an explicit local mirror.

    A mirror preserves result_path layout and original snapshot/hash/binding.
    No symlinks, parent traversals, or fallback search for other attempts.
    """
    import physica_test_plan as plan
    import physica_test_store as store
    path = plan.contained(staged_root or snapshot['shared_dir'], record['result_path'])
    ref = store.file_ref(path)
    if ref['sha256'] != record['result_sha256'] or ref['bytes'] != record['result_bytes']:
        raise ValueError('pinned result changed')
    return path


def projections(identity, frozen, policies):
    """Logical facets only: shared checkpoints and B1 roads stay single episodes."""
    tid = identity['task_id']
    policy = policies[identity['policy_id']]
    if "_projection_index" not in frozen:
        aliases_by_task = defaultdict(dict)
        for a in frozen["aliases"]:
            aliases_by_task[a["task_id"]][a["panel"]] = a["requested_fractions"]
        frozen["_projection_index"] = (aliases_by_task,
            {name: set(p["physical_task_ids"]) for name, p in frozen["panels"].items()})
    aliases_by_task, panel_ids = frozen["_projection_index"]
    aliases = aliases_by_task.get(tid, {})
    for native_panel, panel in frozen['panels'].items():
        if tid not in panel_ids[native_panel] or identity['policy_id'] not in panel['policies']:
            continue
        rules = (policy.get('rules') or ['fixed']) if identity['training_line'] else ['fixed']
        if identity['training_line']:
            planned = {'S0', 'S_0.93', 'S_0.95', 'S_0.97', 'S_0.99'} if native_panel == 'main' else {'S0'}
            rules = [r for r in rules if r in planned]
        for fraction in aliases.get(native_panel, [identity['fraction']]):
            for rule in rules:
                yield dict(identity, panel=PANELS[native_panel], fraction=fraction,
                           rule={'S_0.95': 'C95', 'H_0.95': 'C95'}.get(rule, rule))


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _metric_family(name):
    if 'lap' in name:
        return 'laps'
    if 'background' in name or 'q_bg' == name:
        return 'background'
    return 'mobility'


def scalar_measures(result):
    """Project recorded official scalars, retaining control NA as missing."""
    for block, family in (('metrics', None), ('control', 'control')):
        for name, value in result.get(block, {}).items():
            if name in ('common_cooldown_steps', 'private_cooldown_steps'):
                continue
            if value is None or _number(value):
                reason = ''
                if value is None:
                    reason = 'no_completed_laps' if 'lap_time' in name else 'missing'
                    if 'opportunity_use' in name:
                        reason = result[block].get('opportunity_use_status', 'missing')
                yield dict(family=family or _metric_family(name), metric=name, value=value,
                           na_reason=reason,
                           units=_units(name), support_kind='episode', estimand='mean of episode scalars')

    if "mean_priority_lap_time" in result.get("metrics", {}):
        yield dict(family="laps", metric="lap_availability",
                   value=int(result["metrics"]["mean_priority_lap_time"] is not None),
                   units="fraction", support=int(result["metrics"]["mean_priority_lap_time"] is not None), support_kind="episodes_with_completed_laps",
                   estimand="fraction of episodes with recorded completed lap time")


def _units(metric):
    leaf = metric.rsplit('/', 1)[-1]
    if '/duration_steps/' in metric or '/observed_duration_steps/' in metric:
        return 'events'
    if leaf == 'frequency' or leaf.endswith('_frequency'):
        return 'fraction'
    if leaf in ('counts', 'denominator', 'sentinel_count', 'contexts', 'exit_reason_counts') or leaf.endswith(('_count', '_counts')):
        return 'count'
    if leaf == 'mean' and '/' in metric:
        if 'duration' in metric:
            return 'steps'
        if 'gap' in metric or 'slack' in metric:
            return 'cells'
        if 'speed' in metric or 'long_pv_v1/' in metric:
            return 'cells/step'
        if any(s in metric for s in ('type/', 'reason/', 'direction_set/')):
            return 'native category code'
        if any(metric.endswith(s+'/mean') for s in ('_stopped', '_tight', '_gate',
                                                    '_eligible_left', '_eligible_right',
                                                    '_native_left', '_native_right')):
            return 'fraction'
        return 'count'

    if metric == 'pv_speed/value':
        return 'normalized covariance'
    if 'covariance' in metric:
        return 'occupancy squared' if 'occupancy' in metric else '(cells/step)^2'
    if leaf == 'applied_per_step':
        return 'interventions/step'
    if leaf in ('applied_per_active_decision', 'applied_per_request', 'rejected_per_request'):
        return 'fraction'
    if leaf == 'rates':
        return 'transitions/vehicle-time'
    if leaf in ('lane_change_rate_all', 'lane_change_rate_background'):
        return 'changes/vehicle-time'
    if leaf in ('lane_change_rate', 'lane_change_rate_priority'):
        return 'changes/step'
    if 'flow' in metric or metric == 'q_bg':
        return 'vehicles/step'
    if 'density' in metric:
        return 'vehicles/cell'
    if leaf in ('N', 'vehicle_time', 'Z'):
        return 'vehicle-time'
    if leaf in ('A', 'cell_time'):
        return 'cell-time'
    if leaf in ('S', 'speed_deficit_sum'):
        return 'vehicle-cells'
    if any(s in metric for s in ('fraction', 'ratio', 'delay', 'opportunity_use', 'time_loss', 'lap_availability')):
        return 'fraction'
    if 'duration' in metric or 'lap_time' in metric:
        return 'steps'
    if leaf.endswith('_steps') or leaf == 'steps':
        return 'steps'
    if 'speed' in metric:
        return 'cells/step'
    if leaf in ('distance', 'remaining_distance'):
        return 'cells'
    return 'count'


def diagnostic_measures(analysis, arrays=None):
    """Scope-aware projections of native postprocessor outputs, without new definitions."""
    def scope(r):
        return dict(window=r.get('window_id', ''), region=r.get('region', ''),
                    lane=r.get('lane', ''), vehicle_group=r.get('group', ''),
                    phase='post-step measured states')
    def emit(family, metric, value, **kw):
        return dict(family=family, metric=metric, value=value, units=_units(metric),
                    estimand='equal-episode mean of native diagnostic', **kw)
    pv = analysis.get('pv', {})
    for key, value in pv.items():
        if value is None or _number(value):
            yield emit(_metric_family(key), key, value,
                       na_reason='no_completed_laps' if key == 'mean_completed_lap_time' and value is None else '',
                       support=pv.get('completed_laps') if 'lap' in key else None,
                       support_kind='completed_laps' if 'lap' in key else 'episode')
    if 'completed_laps' in pv:
        success = int(pv['completed_laps'] > 0)
        yield emit('laps', 'lap_availability', success, support=success,
                   support_kind='episodes_with_completed_laps')
    for speed, count in enumerate(pv.get('speed_counts', [])):
        yield emit('mobility', 'pv_speed_counts', count, x=speed, x_label='speed (cells/step)',
                   support_kind='measured_steps')
    for r in analysis['windows']['summaries']:
        for key, value in r.items():
            if value is None or _number(value):
                if key != 'lane':
                    yield emit('windows', key, value, **scope(r), support=r['N'], support_kind='vehicle_time',
                               na_reason='empty_exposure' if value is None and r['N'] == 0 else '')
    for r in analysis['windows']['transitions']:
        for i, category in enumerate(r['categories']):
            for metric in ('counts', 'rates'):
                yield emit('transitions', metric, r[metric][i], **scope(r), x=category,
                           x_label='transition', support=r['exposure_N'], support_kind='vehicle_time',
                           na_reason='empty_exposure' if metric == 'rates' and not r['exposure_N'] else '')
    for r in analysis['windows']['state_frequencies']:
        # Marginals of the recorded joint N/S/Z/A table are time occupancy,
        # not additional samples or newly defined episode outcomes.
        for variable in ('N', 'S', 'Z', 'A'):
            counts = defaultdict(int)
            for state in r['states']:
                counts[state[variable]] += state['count']
            for value, count in sorted(counts.items()):
                yield emit('state', variable+'_frequency', count/r['steps'], **scope(r),
                           x=value, x_label=variable, support=r['steps'], support_kind='measured_steps')
    for stream in analysis['events']['streams']:
        phase = 'mixed-phase operational' if stream.get('condition') == 'operational_pv_blocking' else 'post-step measured states'
        for key, value in stream.items():
            if key not in ('min_duration', 'speed_threshold', 'lane') and (value is None or _number(value)):
                yield emit('events', stream['event_id']+'/'+key, value,
                           phase="mixed-phase operational" if stream.get("condition") == "operational_pv_blocking" else "post-step measured states", support=stream.get("confirmed_count"),
                           support_kind='events_not_replicates',
                           na_reason='no_events' if value is None and not stream.get('confirmed_count') else '')
        duration = stream.get('descriptive_duration')
        if duration:
            yield emit('events', stream['event_id']+'/descriptive_duration/mean', duration['mean'],
                       phase=phase, support=duration['n'], support_kind='fully_observed_events',
                       na_reason=('all_censored' if stream['confirmed_count'] else 'no_events') if duration['mean'] is None else '')
        yield emit('events', stream['event_id']+'/exit_reason_counts', None, phase=phase,
                   x_label='exit reason', support=1, support_kind='events_not_replicates', _context=True)
        for reason, count in stream.get('exit_reason_counts', {}).items():
            yield emit('events', stream['event_id']+'/exit_reason_counts', count, phase=phase,
                       x=reason, x_label='exit reason', support=count, support_kind='events_not_replicates')
        # Context markers establish observed zeros without inventing duration bins.
        for category in ('duration_steps', 'observed_duration_steps'):
            for left in (False, True):
                for right in (False, True):
                    yield emit('events', stream['event_id']+'/'+category+f'/left={left}/right={right}', None,
                               x_label='steps', support=1, support_kind='events_not_replicates', _context=True)
            counts = defaultdict(int)
            for event in stream.get('events', []):
                if event.get(category) is not None:
                    counts[(event[category], event.get('left_censored'), event.get('right_censored'))] += 1
            for (duration, left, right), count in sorted(counts.items(), key=str):
                yield emit('events', stream['event_id']+'/'+category+f'/left={left}/right={right}', count,
                           x=duration, x_label='steps', support=count, support_kind='events_not_replicates')
    mechanism = analysis.get('mechanism')
    if mechanism:
        for table in mechanism['tables']:
            for key in ('mean', 'denominator', 'sentinel_count', 'contexts'):
                yield emit('mechanism', table['name']+'/'+key, table.get(key), phase=table['phase'],
                           support=table['denominator'], support_kind='applicable_contexts',
                           na_reason='not_applicable' if table.get(key) is None and not table['denominator'] else '')
            yield emit('mechanism', table['name']+'/frequency', None, phase=table['phase'],
                       x_label='native category', support=table['denominator'],
                       support_kind='applicable_contexts', _context=True)
            for state in table['states']:
                yield emit('mechanism', table['name']+'/frequency', state['frequency'], phase=table['phase'],
                           x=state['value'], x_label='native category', support=table['denominator'],
                           support_kind='applicable_contexts',
                           na_reason='not_applicable' if state['frequency'] is None else '')
    spatial = analysis.get('spatial_covariance')
    if spatial:
        for table in [dict(spatial, lane='all')] + spatial['by_lane']:
            for lag, value, count in zip(spatial['lags'], table['covariance'], table['pair_counts']):
                yield emit('spatial', 'connected_occupancy_covariance', value, lane=table['lane'],
                           window=spatial['window'], x=lag, x_label='lag (cells)', support=count,
                           support_kind='spatial_pairs_not_replicates')
    for r in (analysis.get('temporal') or {}).get('records', []):
        for metric in ('covariance', 'value'):
            yield emit('temporal', 'pv_speed/'+metric, r[metric], x=r['lag'], x_label='lag (steps)',
                       support=r['pair_count'], support_kind='temporal_pairs_not_replicates',
                       na_reason=('zero_variance' if r['pair_count'] else 'empty_exposure') if r[metric] is None else '')
    for r in (analysis.get('event_alignment') or {}).get('records', []):
        yield emit('onset', r['signal'], r['mean'], x=r['offset'], x_label='onset offset (steps)',
                   support=r['valid_count'], support_kind='events_not_replicates')
    if arrays is not None and 'spatial_hist' in arrays:
        hist = arrays['spatial_hist']
        # Raw accumulated bins only; no invented speed/delay normalization.
        for index in __import__('numpy').ndindex(hist.shape):
            yield emit('profiles', f'spatial_hist_speed{index[3]}_count', int(hist[index]),
                       x=index[0]-36, x_label='offset (cells)', lane=index[1],
                       vehicle_group='HDV' if index[2] == 0 else 'AV', support_kind='accumulated_bin_count')


def build_points(episodes, *, min_repeats=5, n_planned=30):
    """Aggregate compact episode projections; SD is descriptive, never a CI.

    Input items have identity, projections and measures. Only distinct physical
    episodes contribute to each logical point. No training-line pooling here.
    """
    if min_repeats < 5:
        raise ValueError('minimum is five')
    accum, frequency_contexts, projection_repeats = {}, {}, {}
    reasons = defaultdict(set)
    for episode in episodes:
        identity = episode['identity']
        for projection in episode["projections"]:
            projection_fields = {k: projection.get(k) for k in POINT_COLUMNS[:10]+EXTRA_COLUMNS}
            projection_key = json.dumps(projection_fields, sort_keys=True)
            projection_repeats.setdefault(projection_key, set()).add(identity["replicate"])
            for measure in episode['measures']:
                row = {k: projection.get(k) for k in POINT_COLUMNS[:10]+EXTRA_COLUMNS}
                row.update(measure)
                row.setdefault('x', row['density'])
                row.setdefault('x_label', 'density')
                fields = {k: v for k, v in row.items() if k not in ('value', 'support', 'na_reason', '_context')}
                if measure.get('_context'):
                    context = json.dumps({k: v for k, v in fields.items() if k != 'x'}, sort_keys=True)
                    frequency_contexts.setdefault(context, {})[identity['replicate']] = measure.get('support')
                    continue
                key = json.dumps(fields, sort_keys=True)
                if measure.get('na_reason'):
                    reasons[key].add(measure['na_reason'])
                state = accum.setdefault(key, [fields, {}, {}, projection_key])
                rep = identity['replicate']
                if rep in state[1]:
                    raise ValueError('duplicate physical replicate in projection')
                state[1][rep] = measure.get('value')
                state[2][rep] = measure.get("support")
                if measure["family"] == "state" or (measure["family"] == "mechanism" and measure["metric"].endswith("/frequency")):
                    context = json.dumps({k: v for k, v in fields.items() if k != "x"}, sort_keys=True)
                    frequency_contexts.setdefault(context, {})[rep] = measure.get("support")
    points = []
    for key in sorted(accum):
        fields, observations, supports, projection_key = accum[key]
        context = json.dumps({k: v for k, v in fields.items() if k != "x"}, sort_keys=True)
        for rep, denominator in frequency_contexts.get(context, {}).items():
            if rep not in observations:
                observations[rep] = 0.0 if denominator else None
                supports[rep] = 0 if fields['family'] == 'events' else denominator
                if not denominator and fields['family'] == 'mechanism':
                    reasons[key].add('not_applicable')
        for rep in projection_repeats[projection_key]:
            observations.setdefault(rep, None)
        values = [v for v in observations.values() if _number(v)]
        n, defined = len(observations), len(values)
        status = 'partial_exploratory' if n < n_planned else 'complete_descriptive'
        if n < min_repeats:
            status = 'deferred_below_min_repeats'
        elif not defined:
            status = '|'.join(sorted(reasons[key])) or 'missing'
        value = math.fsum(values)/defined if defined and n >= min_repeats else None
        sd = math.sqrt(math.fsum((v-value)**2 for v in values)/(defined-1)) if defined > 1 and value is not None else None
        points.append(dict(fields, value=value, sd=sd, interval_kind='descriptive_sd',
                           n=n, n_defined=defined, n_planned=n_planned, k=1 if fields.get('training_line') else None,
                           k_planned=3 if fields.get('training_line') else None,
                           support=math.fsum(v for v in supports.values() if _number(v)) if any(_number(v) for v in supports.values()) else defined,
                           status=status, na_reason='|'.join(sorted(reasons[key]))))
    return points


def _publish(output, *, manifest, episodes, points, coverage=(), selections=(), illustrations=()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    present = set()
    def point_rows():
        for row in points:
            if row.get('value') is not None:
                present.add(row['family'])
            yield row
    row_counts = {'points.csv': _csv(output/'points.csv', point_rows(), POINT_COLUMNS+EXTRA_COLUMNS)}
    for name, rows in [('episodes.csv', episodes), ('coverage.csv', coverage), ('selections.csv', selections)]:
        row_counts[name] = _csv(output/name, rows)
    _write_json(output/'illustrations.json', list(illustrations))
    manifest.update(schema_version=1, tables=['points.csv', 'episodes.csv', 'coverage.csv', 'selections.csv'],
                    row_counts=row_counts,
                    deferred_families={f: 'no supported recorded data in this extraction' for f in FAMILIES if f not in present})
    _write_json(output/'manifest.json', manifest)
    return manifest


def _cached_points(cache_paths, frozen, policies, *, min_repeats):
    """Release each physical group's aggregation before loading the next group."""
    def items(paths):
        for path in paths:
            with gzip.open(path, 'rt') as f:
                item = json.load(f)
            item['projections'] = list(projections(item['identity'], frozen, policies))
            yield item
    for group_id in sorted(cache_paths):
        yield from build_points(items(cache_paths[group_id]), min_repeats=min_repeats)


def extract_fresh(output_dir, frozen_root, index, *, groups=None, staged_root=None,
                  min_repeats=5, diagnostics=True, final=False, illustrations=False):
    """Stream eligible accepted episodes once; reuse task/attempt/profile keyed caches.

    ``groups`` is an explicit list from scan_metadata; no automatic live refresh.
    Full-scope point tables retain all Full12/lane/group axes. Aggregation and
    CSV export retain at most one physical group's diagnostic points; that group
    can still be large. Use bounded physical-group batches to limit total work.
    ``staged_root`` mirrors relative result_path files without editing the index.
    """
    import physica_test_store as store
    from physica_mc import postprocess, reporting
    scan = scan_metadata(frozen_root, index, groups=groups, min_repeats=min_repeats)
    snapshot, frozen = scan['snapshot'], scan['frozen']
    output = Path(output_dir)
    cache = output/'cache'
    cache.mkdir(parents=True, exist_ok=True)
    records = {r['task_id']: r for r in snapshot['records']}
    scalars, cache_paths = [], defaultdict(list)
    profiles = {p: _json(Path(frozen_root)/f'analysis-{p}.json') for p in ('full12', 'mechanism')}
    for tid in scan['task_ids']:
        record = records[tid]
        identity = dict(scan['identities'][tid], attempt_id=record['attempt_id'])
        profile = profiles[record['observer']]
        key = _digest(dict(record=record, binding=snapshot['input_binding'], profile=profile,
                           diagnostics=diagnostics, cache_version=CACHE_VERSION))
        destination = cache/(key+'.json.gz')
        if destination.exists():
            with gzip.open(destination, 'rt') as f:
                item = json.load(f)
        else:
            path = result_path(snapshot, record, staged_root)
            envelope = store.read_result(path, expected_task_id=tid, expected_attempt_id=record['attempt_id'],
                expected_binding=snapshot['input_binding'], required_profile=record['observer'], verify_archive=False)
            result = envelope['canonical_marker']['result']
            measures = list(scalar_measures(result))
            scalar = dict(identity, **{m['metric']: m['value'] for m in measures})
            scalar['opportunity_use_status'] = result.get('control', {}).get('opportunity_use_status')
            if diagnostics:
                with store.materialize_episode(path, cache, expected_task=frozen['by_id'][tid],
                                               required_profile=record['observer']) as (_, payload, arrays):
                    analysis = postprocess.analyze_episode(payload, arrays, profile)
                    analysis['lineage'] = {'input': {'result_sha256': record['result_sha256']}, 'code': {'source_digest': envelope['source_digest']}}
                    scalar.update(reporting.episode_scalar_row(analysis))
                    measures = [m for m in measures if m['metric'] != 'lap_availability']
                    measures.extend(diagnostic_measures(analysis, arrays))
                    del analysis, arrays, payload
            scalar.update(identity)
            item = dict(identity=identity, scalar=scalar, measures=measures)
            temp = destination.with_suffix('.tmp')
            with gzip.open(temp, 'wt') as f:
                json.dump(item, f, sort_keys=True, separators=(',', ':'), allow_nan=False)
            temp.replace(destination)
            del measures, scalar, result, envelope
        scalars.append(item['scalar'])
        cache_paths[identity['group_id']].append(destination)
        del item
    def points():
        yield from _cached_points(cache_paths, frozen, scan['policies'], min_repeats=min_repeats)
        yield from comparison_points(scalars, scan, final=final)
        yield from family_points(scalars, scan)
        yield from (dict(r, family="coverage", panel=p["panel"], rule=p["rule"], fraction=p["fraction"], metric="accepted_distinct_replicates",
                        x=r["density"], x_label="density", value=r["n"], units="episodes",
                        support_kind="physical_episodes", estimand="metadata only; aliases do not add n")
                    for r in scan["coverage"]
                    for p in projections(scan["identities"][r["task_id"]], frozen, scan["policies"]))
    with (Path(frozen_root)/'selections.csv').open() as f:
        selections = list(csv.DictReader(f))
    illustrations = selected_illustrations(scalars, scan, output_dir=output/'illustrations' if illustrations else None, staged_root=staged_root, analysis_profile=profiles['mechanism'])
    return _publish(output, manifest=dict(dataset_kind='fresh_test', mode="complete" if final and snapshot["coverage"].get("complete") and len(scalars) == len(frozen["by_id"]) else "partial_exploratory",
        snapshot={'sha256': snapshot['document_sha256'], 'created_at_utc': snapshot['created_at_utc'],
                  'index': str(index), 'shared_dir': snapshot['shared_dir']},
        freeze_sha256=snapshot['freeze_sha256'], SAP=_json(Path(frozen_root)/'SAP.json'),
        profiles={p: _digest(v) for p, v in profiles.items()}, min_repeats=min_repeats,
        sources={'frozen_root': str(frozen_root), 'staged_root': str(staged_root) if staged_root else None},
        caveats=['Snapshot cutoff is not live progress.', 'Frames/events do not increase n.',
                 'SD is descriptive; no available-case final grids.']),
        episodes=scalars, points=points(), coverage=scan['coverage'], selections=selections,
        illustrations=illustrations)


def selected_illustrations(episodes, scan, *, output_dir=None, staged_root=None, analysis_profile=None):
    """Select complete E4 Graph S0/H2 pairs; optionally export recorded traces/maps.

    With no output directory this is a scalar-only selection. Incomplete cells
    return explicit deferred entries. The exporter opens only the selected pair.
    """
    from inference import select_illustration
    frozen, policies = scan['frozen'], scan['policies']
    rows = {r['task_id']: r for r in episodes}
    planned = defaultdict(lambda: defaultdict(dict))
    for tid, identity in scan['identities'].items():
        for p in projections(identity, frozen, policies):
            if p['panel'] == 'E4' and (p['method'] == 'H2' or (p['method'] == 'Graph' and p['rule'] == 'S0')):
                cell = (p['road_length'], p['density'], p['fraction'])
                line = p['training_line'] if p['method'] == 'Graph' else 'H2'
                planned[cell][line][p['replicate']] = tid
    entries = []
    for cell, lines in sorted(planned.items()):
        roster = sorted(line for line in lines if line != 'H2')
        pairs = []
        for line in roster:
            for rep, tid in sorted(lines[line].items()):
                hid = lines.get('H2', {}).get(rep)
                g, h = rows.get(tid), rows.get(hid)
                if g is None or h is None or g['environment'] != h['environment']:
                    continue
                gv, hv = g.get('mean_speed_priority'), h.get('mean_speed_priority')
                if _number(gv) and _number(hv):
                    pairs.append(dict(training_line=line, replicate=rep, environment=g['environment'],
                                      paired_pv_effect=gv-hv, Graph=tid, H2=hid))
        selection = select_illustration(pairs, line_ids=roster)
        selection['cell'] = dict(zip(('road_length', 'density', 'fraction'), cell))
        identity = dict(selection['cell'], panel='E4', rule='S0', cohort=1.0)
        if selection['status'] != 'selected':
            entries.append(dict(selection, identity=identity, mode='deferred'))
            continue
        evidence = dict(selection, n_planned=90, k=3, k_planned=3,
                        scope='complete_E4_cell', rule='frozen_median_line_then_environment')
        if output_dir is None:
            entries.append(dict(status='deferred', reason='recorded_export_not_requested',
                                identity=identity, mode='deferred', selection_evidence=evidence))
            continue
        if analysis_profile is None:
            raise ValueError('recorded export requires the frozen mechanism analysis profile')
        import numpy as np
        import physica_test_store as store
        from physica_mc import postprocess
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        records = {r['task_id']: r for r in scan['snapshot']['records']}
        pair = {}
        for method in ('Graph', 'H2'):
            tid = selection[method]
            name = tid+'-'+_digest(dict(result=records[tid]['result_sha256'],
                                        profile=analysis_profile, cache_version=CACHE_VERSION))
            destination, trace_path = root/(name+'.npz'), root/(name+'.json')
            if not destination.exists() or not trace_path.exists():
                path = result_path(scan['snapshot'], records[tid], staged_root)
                with store.materialize_episode(path, root, expected_task=frozen['by_id'][tid],
                                               required_profile='mechanism') as (_, payload, arrays):
                    analysis = postprocess.analyze_episode(payload, arrays, analysis_profile)
                    _write_json(trace_path, dict(episode=analysis['episode'], pv={'trace': analysis['pv']['trace']}))
                    np.savez_compressed(destination, local_speed=arrays['local_speed'],
                                        local_type=arrays['local_type'], map_steps=arrays['step'][1:])
                    del analysis, payload, arrays
            pair[method] = dict(episode_id=tid,
                                analysis_path=str(trace_path.relative_to(root.parent)),
                                arrays_path=str(destination.relative_to(root.parent)))
        entries.append(dict(status='selected', mode='final', identity=identity,
                            selection_evidence=evidence, pair=pair))
    return entries


def extract_validation(output_dir, completed_root, *, min_repeats=5, frozen_root=None):
    """Stream completed scalar validation records and all frozen candidate scores.

    Does not reselect or relabel checkpoint validation as training return.
    Historical RAW references are never opened. All candidates are retained.
    frozen_root supplies training-line cohorts and actual frozen selections;
    without it, RL cohorts remain explicitly unknown. Terminal budgets are
    maximum recorded candidate steps, not proof of training continuity.
    """
    root = Path(completed_root)
    summary, selection = _json(root/'summary.json'), _json(root/'selection.json')
    if not summary.get('data_complete') or not summary.get('final') or summary.get('missing') != 0:
        raise ValueError('completed validation required')
    line_scales, frozen_selections = {}, []
    if frozen_root is not None:
        for policy in _json(Path(frozen_root)/'policies.json').values():
            line = policy.get('group', {}).get('training_line')
            if line:
                scale = policy['scale']
                if line in line_scales and line_scales[line] != scale:
                    raise ValueError('conflicting frozen training-line scales')
                line_scales[line] = scale
        with (Path(frozen_root)/'selections.csv').open() as f:
            frozen_selections = list(csv.DictReader(f))
    scalars, seen, candidates = [], set(), {}

    def episodes():
        with lzma.open(root/'runs.jsonl.xz', 'rt') as f:
            for line in f:
                raw = json.loads(line)
                if raw.get('status') != 'valid':
                    raise ValueError('nonvalid row in completed validation')
                task = raw.get('target_task') or raw['task']
                identity = _identity(task, profile=raw.get('validation', {}).get('actual_profile', ''))
                identity['cohort'] = line_scales.get(identity['training_line'], 'unknown') if identity['training_line'] else 1.0
                tid = identity['task_id']
                if tid in seen:
                    raise ValueError('duplicate completed validation physical task')
                seen.add(tid)
                identity.update(attempt_id=raw['attempt_id'], panel='validation', rule='candidate')
                # Model-less native H candidates remain H1/H2/H4/H8, never original.
                identity['policy_id'] = identity['training_line']+'/'+identity['candidate_id'] if identity['training_line'] else identity['method']
                measures = list(scalar_measures(raw))
                scalar = dict(identity, **{m['metric']: m['value'] for m in measures})
                scalar['opportunity_use_status'] = raw.get('control', {}).get('opportunity_use_status')
                scalars.append(scalar)
                candidates[identity['training_line'], identity['candidate_id'] or identity['method']] = identity
                yield dict(identity=identity, projections=[identity], measures=measures)
    points = build_points(episodes(), min_repeats=min_repeats, n_planned=20)
    # Selected checkpoint steps are not a terminal training budget. The maximum
    # recorded candidate is only the boundary of this completed validation pool.
    terminal = {}
    for (line, _), identity in candidates.items():
        steps = identity.get('env_steps')
        if line and _number(steps):
            terminal[line] = max(terminal.get(line, 0), steps)
    selections = [dict(row, row_kind='frozen_selection',
                       cohort=line_scales.get(row['training_line'], 'unknown') if row['training_line'] else 1.0,
                       selected_env_steps=row.get('env_steps'),
                       terminal_env_steps=terminal.get(row['training_line']),
                       terminal_budget_source='validation_pool_terminal_boundary',
                       continuity='not_established', runtime_seconds=None)
                  for row in frozen_selections]
    for line, steps in sorted(terminal.items()):
        selections.append(dict(row_kind='training_budget', training_line=line,
                               cohort=line_scales.get(line, 'unknown'), terminal_env_steps=steps,
                               terminal_budget_source='validation_pool_terminal_boundary',
                               continuity='not_established', runtime_seconds=None))
    for group in selection['groups']:
        info, scores = group['group'], group['selection']
        line = info['training_line']
        for candidate, speed in scores['speed'].items():
            identity = candidates.get((line, candidate), {})
            base = dict(panel='validation', cohort=identity.get('cohort', line_scales.get(line, 'unknown') if line else 1.0), rule='candidate',
                        method=METHODS.get(info['architecture'], candidate), training_line=line,
                        model_id=identity.get('model_id'), candidate_id=candidate,
                        env_steps=identity.get('env_steps'), x=identity.get('env_steps'),
                        x_label='checkpoint environment steps', n=20, n_planned=20,
                        k=1 if line else None, k_planned=3 if line else None,
                        status='completed_validation', interval_kind='none', road_length=1000)
            points.append(dict(base, family='selection', metric='validation_V', value=speed,
                               units='cells/step', support=12, support_kind='complete_cells',
                               estimand='equal12-cell validation PV speed'))
            ratios = scores['ratios'].get(candidate, [])
            for r in ratios:
                points.append(dict(base, family='ratio', metric='validation_R', value=r['value'],
                                   density=r['density'], fraction=r['fraction'], units='ratio',
                                   estimand='mean paired per-environment policy/B1 flow'))
            thresholds = scores['thresholds']
            for threshold in thresholds:
                tau, chosen = threshold['tau'], threshold['selection']
                feasible = candidate in chosen['feasible']
                points.append(dict(base, family='selection', rule=f'C{round(tau*100)}',
                                   metric='validation_feasible', value=int(feasible), units='boolean',
                                   estimand='frozen validation point screen'))
                selections.append(dict(info, row_kind='validation_threshold', cohort=identity.get('cohort', 'unknown'),
                                       model_id=identity.get('model_id'), candidate_id=candidate, env_steps=identity.get('env_steps'),
                                       terminal_env_steps=terminal.get(line), terminal_budget_source='validation_pool_terminal_boundary',
                                       tau=tau, feasible=feasible, selected=candidate == chosen['selected'],
                                       status=chosen['status']))
    coverage = [dict(panel='validation', status='completed_validation', physical_episodes=len(seen),
                     missing=summary['missing'], n_planned=20)]
    return _publish(output_dir, manifest=dict(dataset_kind='completed_validation', mode='completed_validation',
                    sources={'root': str(root), 'frozen_root': str(frozen_root) if frozen_root else None}, snapshot={'summary': summary}, min_repeats=min_repeats,
                    caveats=['Validation candidates, not training return or fresh test.',
                             'No RAW arrays are implied by historical seals; local diagnostics deferred.',
                             'Training budget is the terminal validation-pool boundary, not verified exact continuity.',
                             'Training cohorts are unknown without frozen training-line metadata.']),
                    episodes=scalars, points=points, selections=selections, coverage=coverage,
                    illustrations=[dict(status='deferred', reason='fresh_E4_pairs_required')])


def comparison_points(episodes, scan, *, final=False):
    """Paired R, frozen direct contrasts and equal-cell domains from physical scalars.

    Final intervals are opt-in and require every planned training/environment
    value. Partial cell means use >=5 common environments and the entire roster.
    Missing grids emit NA; the roster and denominators never shrink.
    """
    import numpy as np
    from physica_mc import statistics as stats
    from inference import bootstrap, paired_summary
    frozen, policies = scan['frozen'], scan['policies']
    observed = {r['task_id']: r for r in episodes}
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    # group -> cell -> training line (or fixed) -> replicate -> (identity, scalar)
    for tid, identity in scan['identities'].items():
        for p in projections(identity, frozen, policies):
            rule = p['rule'] if p['training_line'] else 'fixed'
            key = (p['panel'], p['road_length'], p['method'], p['cohort'], rule)
            cell, line = (p['density'], p['fraction']), p['training_line'] or 'fixed'
            groups[key][cell][line][p['replicate']] = (identity, observed.get(tid))
    points = []
    def base(key):
        panel, length, method, cohort, rule = key
        return dict(panel=panel, road_length=length, method=method, cohort=cohort,
                    rule=rule, training_line='all', n_planned=30,
                    k_planned=3 if method in ('Graph', 'Node', 'Grid') else None,
                    interval_kind='none', units='cells/step')
    def vectors(cells, metric, cell, roster):
        result = {}
        for line in roster:
            result[line] = {identity['environment']: row.get(metric)
                           for identity, row in cells.get(cell, {}).get(line, {}).values()
                           if row is not None}
        return result
    def roster(cells):
        return sorted({line for c in cells.values() for line in c})
    def domains(key, cells):
        panel = key[0]
        allcells = sorted(cells)
        if panel == 'main':
            selection = [c for c in allcells if c[0] in (.1, .2, .3, .4)]
            yield 'selection12', selection
            yield 'transfer42', [c for c in allcells if c not in selection]
            for name, lo, hi in [('low', .05, .15), ('middle', .2, .3), ('high', .35, .9)]:
                yield 'density_band_'+name, [c for c in allcells if lo <= c[0] <= hi]
            yield 'full54', allcells
        elif panel == 'W7':
            yield 'selection9', [c for c in allcells if c[0] != .6]
            yield 'transfer3', [c for c in allcells if c[0] == .6]
            yield 'all12', allcells
        else:
            yield {'E4': 'all12', 'E5_L1000': 'added18', 'E5_L2000': 'length6'}[panel], allcells
    for key, cells in sorted(groups.items()):
        reference_key = (key[0], key[1], 'B1', 1.0, 'fixed')
        reference = groups.get(reference_key, {})
        for line in roster(cells):
            identity_fields = ('model_id', 'candidate_id', 'policy_id')
            identities = {tuple(identity[field] for field in identity_fields)
                          for cell in cells.values()
                          for identity, _ in cell.get(line, {}).values()}
            if len(identities) != 1:
                raise ValueError('conflicting frozen per-line ratio identities')
            line_identity = dict(zip(identity_fields, identities.pop()))
            ratios, fully_observed = {}, {}
            for cell in sorted(cells):
                left = vectors(cells, 'flow_background', cell, [line])
                right = vectors(reference, 'flow_background', cell, ['fixed'])['fixed']
                result = paired_summary(left, right, line_ids=[line], min_repeats=scan['min_repeats'], ratio=True)
                full = len(left[line]) == len(right) == result['n'] == 30
                ratios[cell] = result['value'] if full else None
                fully_observed[cell] = full
                points.append(dict(base(key) | result | line_identity, family='ratio', metric='R',
                    training_line='' if line == 'fixed' else line, k=None if line == 'fixed' else 1,
                    density=cell[0], fraction=cell[1],
                    x=cell[0], x_label='density', units='ratio', k_planned=3 if line != 'fixed' else None,
                    support=result['n'], support_kind='common_compatible_environments',
                    estimand='mean paired per-environment policy/B1 background flow'))
            for domain, domain_cells in domains(key, cells):
                values = [ratios.get(c) for c in domain_cells]
                covered = bool(values) and all(fully_observed.get(c, False) for c in domain_cells)
                complete = covered and all(v is not None for v in values)
                for metric, value in [('R_min', min(values) if complete else None),
                                      ('pass95_count', sum(v >= .95 for v in values if v is not None) if covered else None),
                                      ('fail95_count', sum(v < .95 for v in values if v is not None) if covered else None),
                                      ('undefined_count', sum(v is None for v in values) if covered else None)]:
                    points.append(dict(base(key), family='ratio', metric=metric, value=value,
                        training_line='' if line == 'fixed' else line, k=None if line == 'fixed' else 1,
                        x=domain, x_label='planned domain',
                        n=min((sum(row is not None for _, row in cells[c].get(line, {}).values()) for c in domain_cells), default=0),
                        support=sum(v is not None for v in values), support_kind='defined_complete_planned_cells',
                        status='complete_descriptive' if complete else 'undefined_ratio' if covered else 'missing_planned_cells', units='ratio' if metric == 'R_min' else 'cells',
                        estimand='planned-domain summary of paired policy/B1 background-flow ratios; no available-case minimum'))
    requests = []
    for key in groups:
        panel, length, method, cohort, rule = key
        comparators = []
        if panel == 'main' and method == 'Graph' and rule == 'S0':
            comparators = [('Node', 1., 'S0'), ('Grid', 1., 'S0')]
        if panel == 'main' and method in ('Graph', 'Node', 'Grid') and rule == 'C95':
            comparators = [('H2', 1., 'fixed')]
        if panel.startswith('E5') and method == 'Graph' and rule == 'S0':
            comparators = [(c, 1., 'fixed') for c in ('H2', 'B2', 'B1')]
        if panel == 'W7' and method == 'Graph' and rule == 'S0' and cohort in (.5, 2.):
            comparators = [('Graph', 1., 'S0')]
        for method2, scale2, rule2 in comparators:
            requests.append((key, (panel, length, method2, scale2, rule2)))
    architectures = {'Graph': 'graph_ppo', 'Node': 'node_mlp_ppo', 'Grid': 'grid_cnn_ppo'}
    for key, right_key in sorted(requests):
        left, right = groups[key], groups.get(right_key, {})
        ls, rs = roster(left), roster(right)
        label = key[2]+'−'+right_key[2]
        common_by_cell = {}
        for cell in sorted(left):
            lv = vectors(left, 'mean_speed_priority', cell, ls)
            rv = vectors(right, 'mean_speed_priority', cell, rs)
            common = set.intersection(*(set(v) for v in [*lv.values(), *rv.values()])) if lv and rv else set()
            finite = all(_number(v[e]) for v in [*lv.values(), *rv.values()] for e in common)
            values = [math.fsum(lv[l][e] for l in ls)/len(ls)-math.fsum(rv[r][e] for r in rs)/len(rs)
                      for e in sorted(common)] if finite else []
            supported = len(common) >= scan['min_repeats'] and finite
            value = math.fsum(values)/len(values) if supported else None
            complete = supported and len(common) == 30
            common_by_cell[cell] = value if complete else None
            points.append(dict(base(key), family='contrasts', method=label, metric='paired_PV_speed_difference',
                density=cell[0], fraction=cell[1], x=cell[0], x_label='density', value=value,
                sd=stats.mean_sample_sd(values)[1] if supported else None,
                interval_kind='descriptive_sd', n=len(common), k=len(ls),
                support=len(common), support_kind='common_compatible_environments',
                status='complete_descriptive' if complete else 'partial_exploratory' if supported else 'insufficient_paired_support',
                estimand='first minus comparator; equal frozen whole lines'))
        inference_domains = list(domains(key, left))
        if final:
            inference_domains.extend(("cell:"+str(cell), [cell]) for cell in sorted(left))
        for domain, domain_cells in inference_domains:
            complete = bool(domain_cells) and all(common_by_cell[c] is not None for c in domain_cells)
            result = None
            if complete and final:
                grid = stats.GridSpec(tuple(sorted({c[0] for c in domain_cells})),
                                      tuple(sorted({c[1] for c in domain_cells})), 30)
                def side(k, cells, lines):
                    arrays = {}
                    for cell in grid.cells:
                        if cell not in cells:
                            continue
                        arrays[cell] = np.asarray([[cells[cell][line][r][1]['mean_speed_priority']
                                                    for r in sorted(cells[cell][line])] for line in lines])
                    if lines == ['fixed']:
                        return stats.FixedSide(k[2], {c: a[0] for c, a in arrays.items()})
                    return stats.LineSide(architectures[k[2]], 'S_0.95' if k[4] == 'C95' else k[4], tuple(lines), arrays)
                contrast = stats.Contrast(label, side(key, left, ls), side(right_key, right, rs))
                result = bootstrap([contrast], grid=grid, road_length=key[1],
                                   scales={label: (key[3], right_key[3])})[label]
            points.append(dict(base(key), family='contrasts', method=label, metric='grid_PV_speed_difference',
                x=domain, x_label='planned domain', value=result.point if result else None,
                ci_low=result.ci_low if result else None, ci_high=result.ci_high if result else None,
                interval_kind=result.interval_kind if result else 'none', n=30 if complete else None,
                k=len(ls), support=sum(common_by_cell[c] is not None for c in domain_cells),
                support_kind='complete_planned_cells', status=result.status if result else 'final_inference_deferred' if complete else 'missing_planned_cells',
                estimand="first minus comparator; equal cells in planned domain, equal frozen whole lines; whole training × environment"))
            if domain.startswith("cell:"):
                cell = domain_cells[0]
                points[-1].update(density=cell[0], fraction=cell[1], x=cell[0], x_label="density",
                                  metric="paired_PV_speed_difference_final",
                                  estimand="first minus comparator; equal frozen whole lines; whole training × environment")
    return points


def family_points(episodes, scan):
    """Absolute family means with the frozen roster and common-environment gate."""
    from physica_mc.statistics import mean_sample_sd
    metrics = ('mean_speed_priority', 'mean_priority_time_loss', 'stopped_fraction_priority',
               'flow_background', 'mean_speed_background', 'stopped_fraction_background',
               'requested_nonstay', 'applied_nonstay', 'rejected_nonstay',
               'request_opportunity_use', 'applied_opportunity_use')
    observed = {row['task_id']: row for row in episodes}
    groups = defaultdict(lambda: defaultdict(dict))
    profiles = defaultdict(set)
    for tid, identity in scan['identities'].items():
        if not identity['training_line']:
            continue
        for p in projections(identity, scan['frozen'], scan['policies']):
            key = tuple(p[k] for k in ('panel', 'cohort', 'rule', 'method', 'road_length', 'density', 'fraction'))
            groups[key][p['training_line']][identity['environment']] = observed.get(tid)
            profiles[key].add(p['profile'])
    points = []
    for key, lines in sorted(groups.items()):
        common = set.intersection(*(set(e for e, row in rows.items() if row is not None) for rows in lines.values()))
        k = sum(any(row is not None for row in rows.values()) for rows in lines.values())
        for metric in metrics:
            values = []
            for env in sorted(common):
                raw = [rows[env].get(metric) for rows in lines.values()]
                if all(_number(v) for v in raw):
                    values.append(math.fsum(raw)/len(lines))
            supported = len(common) >= scan['min_repeats'] and len(values) == len(common)
            mean, sd = mean_sample_sd(values) if supported else (None, None)
            base = dict(zip(('panel', 'cohort', 'rule', 'method', 'road_length', 'density', 'fraction'), key))
            base['profile'] = next(iter(profiles[key])) if len(profiles[key]) == 1 else 'mixed'
            points.append(dict(base, family='control' if metric in metrics[6:] else _metric_family(metric),
                training_line='all', metric=metric, value=mean, sd=sd, interval_kind='descriptive_sd',
                n=len(common), n_defined=len(values), n_planned=30, k=k, k_planned=3,
                x=base['density'], x_label='density', support=len(common),
                support_kind='common_compatible_environments', units=_units(metric),
                status='complete_descriptive' if supported and len(common) == 30 else 'partial_exploratory' if supported else 'insufficient_or_undefined_support',
                estimand='equal frozen whole-line means; no training-line pairing'))
    return points
