"""Bounded, read-only snapshot staging and paper-output entry points.

Host-safe commands import only the existing stdlib dataset consumer. Numeric
extraction/rendering commands are added at the same boundary, not to the runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

TOOLKIT = Path(__file__).resolve().parents[2]
DEFAULT_FREEZE = TOOLKIT / '.local/science-test-freeze-20261109-v2'


def dataset_api():
    path = str(TOOLKIT / 'operations/fresh_test')
    if path not in sys.path:
        sys.path.insert(0, path)
    import dataset
    return dataset


def scan_snapshot(index):
    """Describe the published cutoff, not live scheduler progress."""
    snapshot = dataset_api().load_snapshot(index)
    cells = snapshot['coverage']['cell']
    counts = [entry['accepted'] for entry in cells.values()]
    return {
        'snapshot_time': snapshot['created_at_utc'],
        'snapshot_sha256': snapshot['document_sha256'],
        'analysis_status': snapshot['analysis_status'],
        'physical_accepted': len(snapshot['records']),
        'physical_planned': snapshot['coverage']['planned'],
        'groups': len(cells), 'groups_ge5': sum(n >= 5 for n in counts),
        'minimum_repeats': min(counts, default=0),
        'maximum_repeats': max(counts, default=0),
        'panels': snapshot['coverage'].get('panel', {}),
        'note': 'Published snapshot only; may lag native completion. No live queue queried.',
    }


def choose_stage_ids(snapshot, frozen, *, panels=(), densities=(), fractions=(),
                     road_lengths=(), min_repeats=5):
    """Select whole eligible physical groups before inspecting any outcomes.

    Groups may support individual-line previews. Paired comparisons and family
    summaries still require their full roster and common roads in data.py.
    """
    if min_repeats < 5:
        raise ValueError('preview requires at least five unique repeats')
    api = dataset_api()
    allowed = set(frozen['by_id'])
    if panels:
        unknown = set(panels) - set(frozen['panels'])
        if unknown:
            raise ValueError('unknown panels: ' + ', '.join(sorted(unknown)))
        allowed &= set().union(*(set(frozen['panels'][p]['physical_task_ids']) for p in panels))
    alias_fractions = {}
    for alias in frozen['aliases']:
        if not panels or alias['panel'] in panels:
            alias_fractions.setdefault(alias['task_id'], set()).update(alias['requested_fractions'])
    groups = {}
    accepted = {record['task_id'] for record in snapshot['records']}
    for tid in sorted(allowed):
        science = frozen['by_id'][tid]['science']
        cfg = science['config']
        if densities and cfg['density'] not in densities:
            continue
        if road_lengths and cfg['road_length'] not in road_lengths:
            continue
        if fractions and cfg['av_fraction'] not in fractions:
            if not set(fractions).intersection(alias_fractions.get(tid, ())):
                continue
        group = {key: value for key, value in science.items() if key not in ('replicate', 'seeds')}
        gid = api.plan.digest(group)
        groups.setdefault(gid, []).append(tid)
    chosen, deferred = [], []
    for gid, members in sorted(groups.items()):
        available = [tid for tid in members if tid in accepted]
        repeats = [frozen['by_id'][tid]['science']['replicate'] for tid in available]
        if len(repeats) != len(set(repeats)):
            raise ValueError('duplicate environment replicate in a physical group')
        if len(available) >= min_repeats:
            chosen.extend(available)
        else:
            deferred.append({'group_id': gid, 'accepted': len(available),
                             'planned': len(members), 'reason': 'fewer_than_min_repeats'})
    return sorted(chosen), deferred


def stage_snapshot(index, freeze, output, *, include_archives=False,
                   max_bytes=1024**3, **scope):
    """Copy only selected accepted artifacts into a local mirrored source root.

    The original index remains byte-identical. stage.json records the subset;
    neither native acceptance nor the original checksum is rewritten. Archive
    checks remain with materialize_episode when consumed, not another audit here.
    """
    api = dataset_api()
    source = Path(index).absolute()
    raw = source.read_bytes()
    # Validate the exact bytes held here, rather than re-reading a moving pointer.
    snapshot = json.loads(raw)
    check = dict(snapshot)
    digest = check.pop('document_sha256')
    api.plan.require(api.plan.digest(check) == digest
                     and snapshot['format'] == 'physica-test-dataset-v1', 'invalid snapshot')
    records = {row['task_id']: row for row in snapshot['records']}
    api.plan.require(len(records) == len(snapshot['records']) == snapshot['coverage']['accepted'],
                     'duplicate or inconsistent accepted records')
    frozen = api.plan.load_freeze(freeze, verify=False)
    api.plan.require(snapshot['freeze_sha256'] == api.plan.FREEZE_SHA256
                     and snapshot['plan_id'] == api.plan.PLAN_ID
                     and records.keys() <= frozen['by_id'].keys(), 'snapshot belongs to another plan')
    if max_bytes <= 0:
        raise ValueError('staging byte limit must be positive')
    chosen, deferred = choose_stage_ids(snapshot, frozen, **scope)
    total = sum(records[tid]['result_bytes'] +
                (records[tid]['archive_bytes'] if include_archives else 0) for tid in chosen)
    if total > max_bytes:
        raise ValueError(f'selected staging needs {total} bytes; limit is {max_bytes}; narrow the scope')
    root = Path(output).absolute()
    api.plan.require(root.resolve() == root, 'staging root must not contain symlinks')
    api.plan.require(not root.is_relative_to(Path(snapshot['shared_dir']))
                     and not root.is_relative_to(Path(freeze).absolute()),
                     'staging must be outside the live run and frozen input roots')
    root.mkdir(parents=True, exist_ok=True)
    frozen_index = root / 'index.json'
    if frozen_index.exists():
        api.plan.require(frozen_index.read_bytes() == raw, 'output already contains a different snapshot')
    else:
        with frozen_index.open('xb') as stream:
            stream.write(raw)
    mirror = root / 'results'
    mirror.mkdir(exist_ok=True)
    copied = 0
    for tid in chosen:
        record = records[tid]
        original = api.episode_path(snapshot, tid)
        target = api.plan.contained(mirror, record['result_path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        items = [(original, target, record['result_bytes'])]
        if include_archives:
            items.append((original.parent / 'episode.tar.gz', target.parent / 'episode.tar.gz',
                          record['archive_bytes']))
        for src, dest, expected_bytes in items:
            if dest.exists():
                api.plan.require(dest.is_file() and dest.stat().st_size == expected_bytes,
                                 'staged artifact has incompatible size')
                continue
            # Exclusive creation prevents overwriting any previously staged artifact.
            with src.open('rb') as inp, dest.open('xb') as out:
                shutil.copyfileobj(inp, out, length=1024 * 1024)
            api.plan.require(dest.stat().st_size == expected_bytes, 'source changed while staging')
            copied += expected_bytes
        ref = api.store.file_ref(target)
        api.plan.require(ref['sha256'] == record['result_sha256'], 'staged result differs from snapshot')
    result = {
        'format': 'physica-paper-stage-v1', 'snapshot_sha256': digest,
        'snapshot_time': snapshot['created_at_utc'], 'source_index': str(source),
        'source_shared_dir': snapshot['shared_dir'], 'result_root': str(mirror),
        'task_ids': chosen, 'archives_staged': include_archives,
        'selected_bytes': total, 'newly_copied_bytes': copied,
        'deferred_groups': deferred, 'scope': scope,
        'note': 'Group readiness is not paired/family readiness; no numeric outcomes used for staging.',
    }
    stage_file = root / 'stage.json'
    if stage_file.exists():
        prior = json.loads(stage_file.read_text(encoding='utf-8'))
        if prior['snapshot_sha256'] != digest or prior['task_ids'] != chosen:
            raise ValueError('staging directory already belongs to another selection')
    with stage_file.open('w', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def _scope_arguments(command):
    command.add_argument('--panel', nargs='+', dest='panels', default=[])
    command.add_argument('--density', nargs='+', type=float, dest='densities', default=[])
    command.add_argument('--fraction', nargs='+', type=float, dest='fractions', default=[])
    command.add_argument('--road-length', nargs='+', type=int, dest='road_lengths', default=[])
    command.add_argument('--min-repeats', type=int, default=5)


def extract_command(*, output, freeze, index=None, stage=None, diagnostics=False,
                    final=False, illustrations=False, **scope):
    """Wire the original snapshot, bounded physical scope and local source mirror."""
    dataset_api()  # make the existing consumer imports available, without a runner
    import data
    mirror = None
    staged_ids = None
    if stage is not None:
        stage = Path(stage)
        info = json.loads((stage / 'stage.json').read_text(encoding='utf-8'))
        index = stage / 'index.json'
        mirror = stage / 'results'
        if (diagnostics or illustrations) and info['task_ids'] and not info['archives_staged']:
            raise ValueError('diagnostics/illustrations require a stage created with --include-archives')
        staged_ids = set(info['task_ids'])
    else:
        raw = Path(index).read_bytes()
        original = json.loads(raw)
        root = Path(output).absolute()
        if root.resolve() != root or root.is_relative_to(Path(freeze).absolute()) or root.is_relative_to(Path(original['shared_dir'])):
            raise ValueError('output must be outside live/frozen inputs and contain no symlinks')
        pins = root / 'snapshots'
        pins.mkdir(parents=True, exist_ok=True)
        index = pins / (hashlib.sha256(raw).hexdigest() + '.json')
        if not index.exists():
            with index.open('xb') as stream:
                stream.write(raw)
        elif index.read_bytes() != raw:
            raise ValueError('pinned snapshot changed')
    scan = data.scan_metadata(freeze, index, min_repeats=scope.get('min_repeats', 5))
    if final and not scan['snapshot']['coverage']['complete']:
        raise ValueError('--final requires a complete published snapshot; use descriptive preview otherwise')
    root = Path(output).absolute()
    protected = [Path(freeze).absolute(), Path(scan['snapshot']['shared_dir'])]
    if mirror is not None:
        protected.append(mirror.absolute())
    if root.resolve() != root or any(root.is_relative_to(path) for path in protected):
        raise ValueError('output must be outside live/frozen/staged inputs and contain no symlinks')
    if stage is not None and info['snapshot_sha256'] != scan['snapshot']['document_sha256']:
        raise ValueError('stage manifest belongs to a different snapshot')
    tids, _ = choose_stage_ids(scan['snapshot'], scan['frozen'], **scope)
    chosen = set(tids)
    if staged_ids is not None:
        chosen &= staged_ids
    groups = sorted({scan['identities'][tid]['group_id'] for tid in chosen})
    # Different native and consumer group IDs are intentional: the consumer
    # also carries observer profile. Never rewrite a native group checksum.
    expected = {tid for tid in scan['task_ids'] if scan['identities'][tid]['group_id'] in groups}
    if final and chosen != set(scan['frozen']['by_id']):
        raise ValueError('--final requires all planned physical scalar episodes; use preview for bounded subsets')
    if not expected <= chosen:
        raise ValueError('staged subset truncates an accepted physical group')
    if diagnostics and len({(scan['identities'][tid]['road_length'],
                            scan['identities'][tid]['density'],
                            scan['identities'][tid]['fraction'])
                           for tid in chosen if scan['identities'][tid]['method'] != 'B1'}) > 1:
        raise ValueError('diagnostic extraction is cell-batched; select one length/density/fraction per output')
    del scan  # extraction reloads the immutable pin; do not retain two full plans
    report = data.extract_fresh(output, freeze, index, groups=groups, staged_root=mirror,
                                min_repeats=scope.get('min_repeats', 5), diagnostics=diagnostics,
                                final=final, illustrations=illustrations)
    return {'output': str(output), 'mode': report['mode'], 'selected_tasks': len(chosen),
            'selected_groups': len(groups), 'diagnostics': diagnostics,
            'row_counts': report.get('row_counts', {}),
            'deferred_families': report['deferred_families']}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest='command', required=True)
    scan = sub.add_parser('scan', help='read published snapshot coverage; no live queue query')
    scan.add_argument('--index', type=Path, required=True)
    stage = sub.add_parser('stage', help='copy a bounded eligible subset to a local mirrored root')
    stage.add_argument('--index', type=Path, required=True)
    stage.add_argument('--freeze', type=Path, default=DEFAULT_FREEZE)
    stage.add_argument('--output', type=Path, required=True)
    _scope_arguments(stage)
    stage.add_argument('--include-archives', action='store_true')
    stage.add_argument('--max-bytes', type=int, default=1024**3)
    extract = sub.add_parser('extract', help='extract fresh scalar tables; optional one-cell diagnostics')
    source = extract.add_mutually_exclusive_group(required=True)
    source.add_argument('--index', type=Path)
    source.add_argument('--stage', type=Path)
    extract.add_argument('--freeze', type=Path, default=DEFAULT_FREEZE)
    extract.add_argument('--output', type=Path, required=True)
    _scope_arguments(extract)
    extract.add_argument('--diagnostics', action='store_true')
    extract.add_argument('--final', action='store_true', help='enable frozen inference only for complete estimands')
    extract.add_argument('--illustrations', action='store_true', help='export only all90-selected recorded pairs')
    validation = sub.add_parser('validation', help='extract the completed validation, not fresh results')
    validation.add_argument('--completed-root', type=Path,
                            default=TOOLKIT / '.local/science-validation-20261108-full195/validation-results-completed')
    validation.add_argument('--output', type=Path, required=True)
    validation.add_argument('--freeze', type=Path, default=DEFAULT_FREEZE)
    validation.add_argument('--min-repeats', type=int, default=5)
    catalog = sub.add_parser('catalog', help='list available and deferred families from extracted tables')
    catalog.add_argument('--data', type=Path, required=True)
    render = sub.add_parser('render', help='draw an offline atlas from extracted tables, without RAW reads')
    render.add_argument('--data', type=Path, required=True)
    render.add_argument('--output', type=Path, required=True)
    render.add_argument('--family', nargs='+', dest='families')
    render.add_argument('--panel', nargs='+', dest='panels',
                        help='normalized plotting panels: main E4 W7 E5_L1000 E5_L2000 validation')
    render.add_argument('--metric', nargs='+', dest='metrics')
    render.add_argument('--table-row-limit', type=int, default=300)
    render.add_argument('--all-local-scopes', action='store_true',
                        help='expand marginal local cuts to the full Cartesian scope; plan resources first')
    render.add_argument('--formats', nargs='+', choices=['png', 'pdf', 'svg'], default=['png', 'pdf'])
    render.add_argument('--no-heatmaps', dest='heatmaps', action='store_false')
    render.add_argument('--no-conceptual', dest='conceptual', action='store_false')
    tables = sub.add_parser('tables', help='export compact article/companion tables')
    tables.add_argument('--data', type=Path, required=True)
    tables.add_argument('--output', type=Path, required=True)
    return result


def main(argv=None):
    args = vars(parser().parse_args(argv))
    command = args.pop('command')
    if command == 'scan':
        report = scan_snapshot(**args)
    elif command == 'stage':
        detail = stage_snapshot(**args)
        report = {key: value for key, value in detail.items()
                  if key not in ('task_ids', 'deferred_groups')}
        report.update(selected_tasks=len(detail['task_ids']),
                      deferred_groups=len(detail['deferred_groups']),
                      stage_manifest=str(args['output'].absolute() / 'stage.json'))
    elif command == 'extract':
        report = extract_command(**args)
    elif command == 'validation':
        import data
        result = data.extract_validation(args['output'], args['completed_root'],
                                         min_repeats=args['min_repeats'], frozen_root=args['freeze'])
        report = {'output': str(args['output']), 'mode': result['mode'],
                  'deferred_families': result['deferred_families']}
    elif command == 'catalog':
        import render
        report = render.available_families(args['data'])
        report.pop('source_manifest', None)
    elif command == 'render':
        import render
        data_dir = args.pop('data')
        output_dir = args.pop('output')
        result = render.render_atlas(data_dir, output_dir, **args)
        report = {'output': str(output_dir), 'mode': result['mode'],
                  'figures': len(result['artifacts']), 'deferred': result.get('deferred', [])}
    elif command == 'tables':
        import render
        result = render.export_tables(args['data'], args['output'])
        report = {'output': str(args['output']), 'tables': result}
    else:
        raise ValueError('unsupported command')
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
