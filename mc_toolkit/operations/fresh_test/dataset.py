"""Incremental accepted-result view; no dispatch, simulation or statistical inference."""
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parent / 'source'
sys.path.insert(0, str(SOURCE))
import physica_test_plan as plan
import physica_test_store as store


def coverage(frozen, accepted):
    """Physical denominators; overlapping panels and B1 aliases never inflate N."""
    ids = set(accepted)
    plan.require(ids <= frozen['by_id'].keys(), 'unknown accepted task')
    groups = {'repeat': defaultdict(list), 'density': defaultdict(list),
              'profile': defaultdict(list), 'cell': defaultdict(list)}
    cells = {}
    for tid, task in frozen['by_id'].items():
        science = task['science']
        cell = {key: value for key, value in science.items() if key not in ('replicate', 'seeds')}
        cell_id = plan.digest(cell)
        cells[cell_id] = cell
        groups['repeat'][str(science['replicate'])].append(tid)
        groups['density'][f"{science['config']['density']:.2f}"].append(tid)
        groups['profile'][plan.observer_for(frozen, tid)].append(tid)
        groups['cell'][cell_id].append(tid)
    groups['panel'] = {key: value['physical_task_ids'] for key, value in frozen['panels'].items()}
    result = {}
    for group, entries in groups.items():
        result[group] = {}
        for key, members in entries.items():
            n = len(ids.intersection(members))
            result[group][key] = {'planned': len(members), 'accepted': n,
                                  'missing': len(members) - n, 'complete': n == len(members)}
            if group == 'cell':
                result[group][key]['science_without_environment'] = cells[key]
    result.update(planned=len(frozen['by_id']), accepted=len(ids),
                  missing=len(frozen['by_id']) - len(ids), complete=ids == frozen['by_id'].keys(),
                  physical_counts=True, panels_overlap=True)
    return result


def verify_record(record, run, frozen, binding):
    """Trust native acceptance; check the small scientific envelope, not the archive again."""
    receipt = record['receipt']
    tid, aid = receipt['task_id'], receipt['attempt_id']
    plan.require(tid in frozen['by_id'], 'native receipt not in frozen physical plan')
    observer = plan.observer_for(frozen, tid)
    if receipt['status'] != 'success':
        plan.require(receipt['status'] == 'failed', 'unexpected native receipt status')
        return None
    value = receipt['result']
    plan.require(value['format'] == 'physica-test-reference-v1'
                 and value['task_id'] == tid and value['attempt_id'] == aid
                 and value['input_binding'] == binding, 'accepted reference identity mismatch')
    output = Path(record['output_dir'])
    expected_output = Path(run.shared) / 'outputs' / record['worker_id'] / aid
    plan.require(output == expected_output and store.safe(output) == output,
                 'native output identity mismatch')
    native_artifacts = {item['path']: item['size'] for item in receipt['artifacts']}
    for key, basename in (('result', 'result.json'), ('archive', 'episode.tar.gz')):
        ref = value[key]
        plan.require(ref['path'] == basename and native_artifacts.get(basename) == ref['bytes'],
                     'missing native artifact registration')
    plan.require(store.file_ref(output / 'result.json') == value['result'], 'result envelope changed')
    row = store.read_result(output / 'result.json', expected_task_id=tid, expected_attempt_id=aid,
                            expected_binding=binding, required_profile=observer, verify_archive=False)
    plan.require(row['archive'] == value['archive'], 'envelope/native archive descriptor disagreement')
    return {'task_id': tid, 'attempt_id': aid, 'sequence': record['sequence'],
            'result_path': (output / 'result.json').relative_to(run.shared).as_posix(),
            'result_sha256': value['result']['sha256'], 'result_bytes': value['result']['bytes'],
            'archive_sha256': value['archive']['sha256'], 'archive_bytes': value['archive']['bytes'],
            'observer': observer}


def _checked_document(path, kind):
    value = store.read_json(Path(path).absolute())
    expected = value.pop('document_sha256')
    plan.require(plan.digest(value) == expected and value['format'] == kind,
                 'view/cache checksum or format mismatch')
    value['document_sha256'] = expected
    return value


def _write_document(path, value):
    # Single collector flock owns these mutable pointers; native artifacts stay immutable.
    from slurm_map.common import atomic_write
    path = store.safe(Path(path).absolute())
    value = dict(value)
    value['document_sha256'] = plan.digest(value)
    atomic_write(path, plan.encoded(value))
    return value


def refresh(run, frozen, *, binding, state_root, dataset_root, max_records=4096):
    """One bounded metadata-only refresh; native queue and episode archives are untouched."""
    plan.require(type(max_records) is int and 0 < max_records <= 100000, 'invalid refresh bound')
    state_root, dataset_root = Path(state_root).absolute(), Path(dataset_root).absolute()
    state_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    store.local_root(state_root)
    store.safe(dataset_root)
    with (state_root / 'collector.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = state_root / 'collector-state.json'
        if state_path.exists():
            state = _checked_document(state_path, 'physica-test-collector-v1')
            plan.require(state['input_binding'] == binding and state['shared_dir'] == str(run.shared),
                         'collector state belongs to another run')
            state.pop('document_sha256')
        else:
            state = {'format': 'physica-test-collector-v1', 'input_binding': binding,
                     'shared_dir': str(run.shared), 'cursor': 0, 'accepted': {}, 'last_failures': {}}
        seen = 0
        while seen < max_records:
            page = run.completed(after=state['cursor'], limit=min(256, max_records - seen))
            if not page:
                break
            for record in page:
                plan.require(type(record['sequence']) is int and record['sequence'] > state['cursor'],
                             'non-monotone native completion cursor')
                item = verify_record(record, run, frozen, binding)
                tid = record['receipt']['task_id']
                if item is not None:
                    previous = state['accepted'].get(tid)
                    plan.require(previous is None or previous == item, 'conflicting accepted attempt')
                    state['accepted'][tid] = item
                    state['last_failures'].pop(tid, None)
                else:
                    plan.require(tid not in state['accepted'], 'accepted success regressed to failure')
                    # This is a bounded view, not a duplicate traceback archive.
                    tail = ''.join(c if 32 <= ord(c) < 127 else ' '
                                   for c in record['receipt']['error'][-256:])
                    state['last_failures'][tid] = {
                        'attempt_id': record['receipt']['attempt_id'], 'worker_id': record['worker_id'],
                        'sequence': record['sequence'], 'error_tail_ascii': tail}
                state['cursor'] = record['sequence']
                seen += 1
        snapshot = {
            'format': 'physica-test-dataset-v1', 'input_binding': binding,
            'freeze_sha256': plan.FREEZE_SHA256, 'plan_id': plan.PLAN_ID,
            'frozen_inputs': str(frozen['root']),
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'shared_dir': str(run.shared), 'collected_through_sequence': state['cursor'],
            'records': sorted(state['accepted'].values(), key=lambda row: row['task_id']),
            'coverage': coverage(frozen, state['accepted']),
            'last_observed_failures': state['last_failures'],
            'native_status': run.status(),
            'analysis_status': 'complete' if len(state['accepted']) == len(frozen['by_id']) else 'partial_exploratory',
            'note': 'Accepted physical episodes only. No imputation, test reselection or early stopping. '
                    'Native status can be newer than this bounded collection cutoff. '
                    'Last observed failures may since have entered explicit recovery. '
                    'Coverage is accepted-task coverage, not a new archive audit; archives are checked on read.',
        }
        snapshot = _write_document(dataset_root / 'index.json', snapshot)
        _write_document(state_path, state)
        return snapshot


def load_snapshot(path):
    """Load once and hold this document while plotting; never mix moving index versions."""
    snapshot = _checked_document(path, 'physica-test-dataset-v1')
    records = snapshot['records']
    plan.require(len({row['task_id'] for row in records}) == len(records), 'duplicate physical task in view')
    plan.require(snapshot['coverage']['accepted'] == len(records), 'view count mismatch')
    return snapshot


def iter_results(snapshot, *, task_ids=None, verify_archives=False):
    """Yield the complete scalar/provenance envelopes from a pinned view (stdlib only)."""
    wanted = None if task_ids is None else set(task_ids)
    for record in snapshot['records']:
        if wanted is not None and record['task_id'] not in wanted:
            continue
        path = plan.contained(snapshot['shared_dir'], record['result_path'])
        ref = store.file_ref(path)
        plan.require(ref['sha256'] == record['result_sha256'] and ref['bytes'] == record['result_bytes'],
                     'pinned result changed')
        yield store.read_result(path, expected_task_id=record['task_id'],
                                expected_attempt_id=record['attempt_id'],
                                expected_binding=snapshot['input_binding'], required_profile=record['observer'],
                                verify_archive=verify_archives)


def episode_path(snapshot, task_id):
    """Resolve an accepted episode for store.materialize_episode, never an orphan file."""
    for record in snapshot['records']:
        if record['task_id'] == task_id:
            path = plan.contained(snapshot['shared_dir'], record['result_path'])
            ref = store.file_ref(path)
            plan.require(ref['sha256'] == record['result_sha256'] and ref['bytes'] == record['result_bytes'],
                         'pinned result changed')
            return path
    raise KeyError('task is not accepted in this snapshot: ' + task_id)
