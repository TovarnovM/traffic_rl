"""Durable, process-per-trajectory execution. No simulator imports on the host."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import traceback
import uuid


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        with tmp.open('xb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            try:
                os.link(tmp, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ValueError(f'immutable input differs: {path}')
        else:
            os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        tmp.unlink(missing_ok=True)
    return path


@contextmanager
def _lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _deadline(value):
    date = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if date.tzinfo is None:
        raise ValueError('deadline must include timezone')
    return date.timestamp()


def _positive(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f'{name} must be an explicit integer')
    return value


def _owner(pid=None):
    pid = os.getpid() if pid is None else pid
    # starttime is field 22; comm may contain spaces and parentheses.
    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': stat[19],
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'host': socket.gethostname()}


def _alive(owner):
    if not owner or owner['host'] != socket.gethostname():
        return None
    try:
        return _owner(owner['pid']) == owner
    except FileNotFoundError:
        return False
    except OSError:
        return None


def _runtime(execution):
    deps = {}
    for name in ('numpy', 'numba', 'torch', 'physica-mc'):
        try:
            deps[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            deps[name] = None
    modules = {name: str(Path(module.__file__).resolve())
               for name, module in tuple(sys.modules.items())
               if name in ('physica_mc', 'snfs_traffic', 'numpy', 'numba', 'torch')
               and getattr(module, '__file__', None)}
    return {'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'dependencies': deps,
            'module_paths': modules, 'execution': execution,
            'owner': _owner()}


def publish_spool(source_marker, output_root):
    from .storage import read_episode, publish_episode
    payload, arrays = read_episode(source_marker)
    marker = publish_episode(output_root, payload['task'], payload['execution'],
                             payload['attempt_id'], payload['result'], arrays)
    read_episode(marker, expected_task=payload['task'],
                 expected_execution=payload['execution'])
    return marker


def _registry(plan, root):
    return Path(root).resolve() / '.runs' / plan['plan_id']


def _execution(task, plan, device, threads, observer):
    from .schema import make_execution
    bundle = None
    if task['science']['model_id'] is not None:
        from .model_convert import read_manifest
        model = plan['models'][task['science']['model_id']]
        bundle = model['bundle_path']
        meta = read_manifest(bundle, expected_sha256=model['manifest_sha256'])
        if meta['identity'] != task['science']['model_id']:
            raise ValueError('model identity differs from task')
    return make_execution(task, device=device if bundle else 'cpu', threads=threads,
                          observer=observer, model_bundle=bundle)


def collect_plan(plan, output_root, *, required_profile='full12'):
    from .planning import validate_plan
    from .storage import read_episode
    validate_plan(plan)
    root = Path(output_root).resolve()
    ledger_path = _registry(plan, root) / 'ledger.json'
    ledger = _read(ledger_path) if ledger_path.exists() else {'tasks': {}}
    if ledger_path.exists() and ledger.get('plan') != plan:
        raise ValueError('ledger plan identity mismatch')
    report = {'accepted': {}, 'missing': [], 'invalid': [], 'conflicting': {},
              'canonical': {}, 'lifecycle': {}, 'status': 'incomplete'}
    for task in plan['tasks']:
        tid = task['task_id']
        record = ledger['tasks'].get(tid)
        valid = []
        for directory in sorted((root / tid).glob('*')):
            marker = directory / 'completion.json'
            try:
                if not record:
                    raise ValueError('no bound execution in run ledger')
                payload, _ = read_episode(marker, expected_task=task,
                                          expected_execution=record['execution'],
                                          required_profile=required_profile)
                if payload['attempt_id'] not in {a['attempt_id'] for a in record['attempts']}:
                    raise ValueError('attempt is not owned by this ledger')
                valid.append(str(marker))
            except Exception as exc:
                report['invalid'].append({'marker': str(marker), 'error': str(exc)})
        if len(valid) > 1:
            report['conflicting'][tid] = valid
        elif len(valid) == 1:
            report['accepted'][tid] = valid[0]
            report['canonical'][tid] = valid[0]
        else:
            report['missing'].append(tid)
        report['lifecycle'][tid] = [
            {'attempt_id': a['attempt_id'], 'state': a['state'],
             'lifecycle': a.get('lifecycle', 'unknown'), 'owner': a.get('owner'),
             'reconciliation': a.get('reconciliation')}
            for a in record['attempts']] if record else []
    if not report['missing'] and not report['invalid'] and not report['conflicting']:
        report['status'] = 'complete'
    report['data_status'] = report['status']
    report['lifecycle_complete'] = all(a['lifecycle'] == 'exited'
                                     for attempts in report['lifecycle'].values() for a in attempts)
    if not report['lifecycle_complete']:
        report['status'] = 'incomplete'
    return report


def collect_shards(parent_plan, shards, *, required_profile='full12'):
    """Collect an exact partition, preserving each shard's execution ledger."""
    from .planning import validate_plan, load_plan
    validate_plan(parent_plan)
    expected = {t['task_id']: t for t in parent_plan['tasks']}
    seen, roots, reports = set(), [], []
    for shard in shards:
        plan = load_plan(shard['plan']) if isinstance(shard['plan'], str) else shard['plan']
        validate_plan(plan)
        root = Path(shard['output_root']).resolve()
        if any(root == old or root in old.parents or old in root.parents for old in roots):
            raise ValueError('shards require disjoint output roots')
        roots.append(root)
        if plan['phase'] != parent_plan['phase'] or plan['freeze'] != parent_plan['freeze']:
            raise ValueError('shard science boundary differs from parent')
        for task in plan['tasks']:
            tid = task['task_id']
            if tid in seen or expected.get(tid) != task:
                raise ValueError('overlapping or unexpected shard task/science')
            seen.add(tid)
        reports.append(collect_plan(plan, root, required_profile=required_profile))
    if seen != set(expected):
        raise ValueError('shards do not exactly cover parent tasks')
    result = {'accepted': {}, 'canonical': {}, 'lifecycle': {}, 'conflicting': {},
              'missing': [], 'invalid': []}
    for report in reports:
        for key in ('accepted', 'canonical', 'lifecycle', 'conflicting'):
            result[key].update(report[key])
        for key in ('missing', 'invalid'):
            result[key].extend(report[key])
    result['data_status'] = 'complete' if all(r['data_status'] == 'complete' for r in reports) else 'incomplete'
    result['lifecycle_complete'] = all(r['lifecycle_complete'] for r in reports)
    result['status'] = result['data_status'] if result['lifecycle_complete'] else 'incomplete'
    return result


def reconcile_owner(plan, output_root, *, campaign_root, token, element_id,
                    attempt_id, evidence):
    """Import exact operator ownership evidence, without querying/stopping work."""
    from . import slurm
    from .planning import validate_plan
    validate_plan(plan)
    root = Path(output_root).resolve()
    campaign_root = Path(campaign_root).resolve()
    with _lock(root / '.runs' / 'coordinator.lock'), _lock(campaign_root / 'campaign.lock'):
        rec = _read(campaign_root / 'campaign.json')['submissions'][token]
        frozen = _read(campaign_root / token / 'prepared.json')
        for key in ('token', 'campaign_id', 'cluster', 'commands', 'inputs', 'compose'):
            if rec[key] != frozen[key]:
                raise ValueError('submission differs from immutable preparation')
        element = rec['elements'][element_id]
        cleanup = element.get('cleanup_evidence', {})
        if (element['state'] not in slurm.TERMINAL or element['cleanup'] != 'verified'
                or not element.get('controller') or not element.get('node')
                or evidence.get('container_id') != cleanup.get('container_id')
                or not evidence.get('container_id')
                or not isinstance(evidence.get('provenance'), str)
                or not evidence['provenance'].strip()):
            raise ValueError('exact terminal container evidence required')
        index = int(element_id.split('_')[1]) if '_' in element_id else 0
        expected_id = f"{rec['job_id']}_{index}" if rec['count'] > 1 else rec['job_id']
        if element_id != expected_id or not 0 <= index < rec['count']:
            raise ValueError('element identity mismatch')
        slurm._validate_cleanup(rec, token, element_id, cleanup)
        # Legacy preparations omitted run_requests. Supplied mappings still
        # have to match the immutable hash inventory and exact command argv.
        binding = frozen['run_requests'][index] if frozen.get('run_requests') else evidence['run_request']
        host, container = binding['host_path'], binding['container_path']
        if (rec['commands'][index][-5:] != ['-m', 'physica_mc.cli', 'run', '--request', container]
                or host not in frozen['inputs'] or slurm._hash(host) != frozen['inputs'][host]):
            raise ValueError('immutable run request binding mismatch')
        request = _read(host)
        bound_plan = request['plan']
        if isinstance(bound_plan, str):
            compose_path = frozen['compose']
            if slurm._hash(compose_path) != frozen['inputs'][compose_path]:
                raise ValueError('frozen Compose changed')
            candidates = []
            for volume in _read(compose_path)['services']['compute'].get('volumes', []):
                target = Path(volume['target'])
                if Path(bound_plan) == target or target in Path(bound_plan).parents:
                    candidates.append(Path(volume['source']) / Path(bound_plan).relative_to(target))
            if len(candidates) != 1:
                raise ValueError('plan requires one exact frozen mount binding')
            plan_path = str(candidates[0])
            if plan_path not in frozen['inputs'] or slurm._hash(plan_path) != frozen['inputs'][plan_path]:
                raise ValueError('plan not in immutable input inventory')
            bound_plan = _read(plan_path)
        path = _registry(plan, root) / 'ledger.json'
        ledger = _read(path)
        if bound_plan != plan or ledger['plan'] != plan or Path(request['output_root']).resolve() != root:
            raise ValueError('run request/ledger plan or output mismatch')
        defaults = {'workers': 1, 'device': 'cpu', 'threads': 1, 'observer': 'full12', 'trace': False}
        settings = {key: request.get(key, value) for key, value in defaults.items()}
        settings.update(budget=request['budget'], spool_root=str(Path(request['spool_root']).resolve()))
        if settings != ledger['settings']:
            raise ValueError('immutable run settings mismatch')
        matches = [(tid, r, a) for tid, r in ledger['tasks'].items() for a in r['attempts']
                   if a['attempt_id'] == attempt_id]
        if len(matches) != 1:
            raise ValueError('attempt must belong to this ledger')
        tid, record, attempt = matches[0]
        episode = _read(attempt['request'])
        if (slurm._hash(attempt['request']) != evidence.get('request_sha256')
                or episode['attempt_id'] != attempt_id or episode['task'] not in plan['tasks']
                or episode['task']['task_id'] != tid or episode['execution'] != record['execution']
                or episode['output_root'] != str(root) or episode['spool_root'] != settings['spool_root']
                or episode['deadline_utc'] != settings['budget']['deadline_utc']
                or episode['trace'] != settings['trace']
                or episode.get('model') != plan['models'].get(episode['task']['science']['model_id'])):
            raise ValueError('immutable episode request mismatch')
        owner = attempt.get('owner')
        receipt_path = Path(episode['receipt'])
        if receipt_path.exists():
            receipt = _read(receipt_path)
            if receipt['attempt_id'] != attempt_id or (owner and owner != receipt.get('owner')):
                raise ValueError('receipt owner/attempt mismatch')
            owner = receipt.get('owner')
        if 'owner' not in evidence or evidence['owner'] != owner:
            raise ValueError('exact owner attestation required')
        provenance = {'campaign_root': str(campaign_root), 'token': token,
                      'element_id': element_id, 'job_id': rec['job_id'],
                      'element': element, 'run_request': binding, 'evidence': evidence}
        if attempt.get('reconciliation') and attempt['reconciliation'] != provenance:
            raise ValueError('owner reconciliation is immutable')
        attempt.update(lifecycle='exited', owner=owner, reconciliation=provenance)
        _write(path, ledger)
        return {'status': 'reconciled', 'attempt_id': attempt_id, 'lifecycle': 'exited',
                'attempts_consumed': ledger['attempts_consumed'], 'provenance': provenance}


def episode_request(request):
    """Internal fresh child entry. Receipt is separate from scientific payload."""
    from . import schema, storage
    request = dict(request)
    receipt = Path(request['receipt'])
    owner = _owner()
    state = {'state': 'running', 'owner': owner, 'attempt_id': request['attempt_id']}
    _write(receipt, state, exclusive=True)
    stage = 'simulation'
    try:
        if request['execution']['implementation'] != schema.source_identity():
            raise ValueError('implementation changed after intent; freeze code before execution')
        if request.get('model'):
            from .model_convert import read_manifest
            meta = read_manifest(request['model']['bundle_path'],
                                 expected_sha256=request['model']['manifest_sha256'])
            if meta['identity'] != request['task']['science']['model_id']:
                raise ValueError('model identity mismatch')
        from .episode import simulate_episode
        start = time.monotonic()
        result, arrays = simulate_episode(request['task'], execution=request['execution'],
                                          trace=request['trace'], deadline_utc=request['deadline_utc'])
        state['runtime'] = _runtime(request['execution'])
        state['model'] = request.get('model')
        state['phase_timings'] = result.get('timings')
        state['resources'] = result.get('resources')
        state['actual_device'] = result.get('device')
        stage = 'spool'
        marker = storage.publish_episode(request['spool_root'], request['task'],
                                          request['execution'], request['attempt_id'], result, arrays)
        storage.read_episode(marker, expected_task=request['task'],
                             expected_execution=request['execution'])
        state.update(spool_marker=str(marker), state='spooled')
        _write(receipt, state)
        stage = 'publication'
        publish_start = time.monotonic()
        output = publish_spool(marker, request['output_root'])
        state.update(state='complete', marker=str(output),
                     wall_seconds=time.monotonic() - start,
                     publication_seconds=time.monotonic() - publish_start)
        _write(receipt, state)
        return state
    except BaseException as exc:
        state.update(state='failed', stage=stage, error=str(exc),
                     traceback=traceback.format_exc(), runtime=_runtime(request['execution']))
        partial = getattr(exc, 'partial', None)
        if partial is not None:
            state['partial'] = partial
        _write(receipt, state)
        raise


def run_plan(plan, *, output_root, spool_root, budget, workers=1,
             device='cpu', threads=1, observer='full12', trace=False):
    from .planning import validate_plan
    from .storage import read_episode
    validate_plan(plan)
    for key in ('max_attempts', 'max_workers', 'cpu_limit', 'max_gpu_workers'):
        _positive(budget[key], key, zero=key == 'max_gpu_workers')
    _positive(workers, 'workers')
    _positive(threads, 'threads')
    end = _deadline(budget['deadline_utc'])
    if workers > budget['max_workers'] or workers * threads > budget['cpu_limit']:
        raise ValueError('worker/thread allocation exceeds approved CPU budget')
    if device != 'cpu' and (workers != 1 or budget['max_gpu_workers'] < 1):
        raise ValueError('GPU episodes require one serial worker and GPU approval')
    output_root, spool_root = Path(output_root).resolve(), Path(spool_root).resolve()
    if output_root == spool_root or output_root in spool_root.parents or spool_root in output_root.parents:
        raise ValueError('spool and output must be separate persistent roots')
    registry = _registry(plan, output_root)
    with _lock(output_root / '.runs' / 'coordinator.lock'), _lock(registry / 'coordinator.lock'):
        # Serialize the root and retain ownership after a coordinator exits.
        foreign_blocked = False
        task_ids = {t['task_id'] for t in plan['tasks']}
        for foreign_path in (output_root / '.runs').glob('*/ledger.json'):
            if foreign_path == registry / 'ledger.json':
                continue
            foreign = _read(foreign_path)
            if any(tid in task_ids and r['attempts'] for tid, r in foreign['tasks'].items()):
                raise ValueError('physical task owned by another plan; use disjoint roots')
            foreign_blocked |= any(a.get('lifecycle') != 'exited'
                                   for r in foreign['tasks'].values() for a in r['attempts'])
        path = registry / 'ledger.json'
        settings = {'budget': budget, 'workers': workers, 'device': device,
                    'threads': threads, 'observer': observer, 'trace': trace,
                    'spool_root': str(spool_root)}
        if path.exists():
            ledger = _read(path)
            if ledger['settings'] != settings or ledger['plan'] != plan:
                raise ValueError('immutable run settings/plan changed')
        else:
            ledger = {'schema_version': 1, 'plan': plan, 'settings': settings,
                      'attempts_consumed': 0, 'tasks': {}}
        for task in plan['tasks']:
            execution = _execution(task, plan, device, threads, observer)
            old = ledger['tasks'].setdefault(task['task_id'], {'execution': execution, 'attempts': []})
            if old['execution'] != execution:
                raise ValueError('source/profile changed since run intent')
        ledger['coordinator'] = _owner()
        _write(path, ledger)
        # No automatic retries of failed physics or unknown owners. Only verified
        # completed spool may be replayed, after exact owner is known to be gone.
        for task in plan['tasks']:
            rec = ledger['tasks'][task['task_id']]
            for attempt in rec['attempts']:
                if attempt.get('request'):
                    receipt_path = Path(attempt['request']).with_name('receipt.json')
                    if receipt_path.exists():
                        receipt = _read(receipt_path)
                        if receipt.get('attempt_id') != attempt['attempt_id']:
                            raise ValueError('receipt attempt identity mismatch')
                        if attempt.get('owner') and attempt['owner'] != receipt.get('owner'):
                            raise ValueError('receipt owner generation mismatch')
                        attempt['owner'] = receipt.get('owner')
                        attempt['receipt'] = receipt
                if attempt.get('lifecycle') != 'exited':
                    alive = _alive(attempt.get('owner'))
                    if alive is not False:
                        attempt['state'] = 'unknown'
                        continue
                    attempt['lifecycle'] = 'exited'
                marker = spool_root / task['task_id'] / attempt['attempt_id'] / 'completion.json'
                if marker.exists():
                    try:
                        read_episode(marker, expected_task=task, expected_execution=rec['execution'])
                        attempt['marker'] = str(publish_spool(marker, output_root))
                        attempt['state'] = 'complete'
                    except Exception as exc:
                        attempt.update(state='failed', error=str(exc))
                elif attempt['state'] in ('intent', 'running', 'unknown'):
                    attempt['state'] = 'unknown'
        _write(path, ledger)
        coverage = collect_plan(plan, output_root, required_profile=observer)
        blocked = foreign_blocked or any(a.get('lifecycle') != 'exited'
                                        for r in ledger['tasks'].values() for a in r['attempts'])
        pending = [t for t in plan['tasks'] if t['task_id'] not in coverage['accepted']
                   and not ledger['tasks'][t['task_id']]['attempts']
                   and not (output_root / t['task_id']).exists()]
        if blocked:
            pending = []
        active = {}
        try:
            while pending or active:
                while pending and len(active) < workers and time.time() < end and ledger['attempts_consumed'] < budget['max_attempts']:
                    task = pending.pop(0)
                    tid, aid = task['task_id'], uuid.uuid4().hex
                    rec = ledger['tasks'][tid]
                    directory = registry / 'attempts' / aid
                    request = {'task': task, 'execution': rec['execution'], 'attempt_id': aid,
                               'output_root': str(output_root), 'spool_root': str(spool_root),
                               'receipt': str(directory / 'receipt.json'), 'trace': trace,
                               'deadline_utc': budget['deadline_utc'],
                               'model': plan['models'].get(task['science']['model_id'])}
                    request_path = _write(directory / 'request.json', request, exclusive=True)
                    attempt = {'attempt_id': aid, 'state': 'intent', 'lifecycle': 'unknown',
                               'owner': None, 'request': str(request_path)}
                    rec['attempts'].append(attempt)
                    ledger['attempts_consumed'] += 1
                    _write(path, ledger)  # consume budget BEFORE Popen, including uncertain launch
                    log = (directory / 'child.log').open('ab')
                    env = dict(os.environ)
                    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
                        env[key] = str(threads)
                    try:
                        process = subprocess.Popen([sys.executable, '-m', 'physica_mc.cli', 'episode',
                                                    '--request', str(request_path)], env=env,
                                                   stdout=log, stderr=subprocess.STDOUT)
                    except OSError as exc:
                        attempt.update(state='failed', lifecycle='exited', error=str(exc))
                        _write(path, ledger)
                        log.close()
                        continue
                    try:
                        attempt['owner'] = _owner(process.pid)
                    except FileNotFoundError:
                        pass
                    attempt['state'] = 'running'
                    active[aid] = (process, log, attempt, request)
                    _write(path, ledger)
                for aid, (process, log, attempt, request) in list(active.items()):
                    if time.time() >= end and process.poll() is None:
                        # Popen owns this unreaped child; never signal by UID/name.
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    rc = process.poll()
                    if rc is None:
                        continue
                    log.close()
                    attempt.update(state='failed', lifecycle='exited', returncode=rc)
                    receipt = Path(request['receipt'])
                    if receipt.exists():
                        child = _read(receipt)
                        attempt['receipt'] = child
                        attempt['owner'] = child.get('owner', attempt['owner'])
                    marker = output_root / request['task']['task_id'] / aid / 'completion.json'
                    if marker.exists():
                        try:
                            read_episode(marker, expected_task=request['task'], expected_execution=request['execution'])
                            attempt.update(state='complete', marker=str(marker))
                        except Exception as exc:
                            attempt['error'] = str(exc)
                    del active[aid]
                    _write(path, ledger)
                if not active and (time.time() >= end or ledger['attempts_consumed'] >= budget['max_attempts']):
                    break
                if active:
                    time.sleep(0.05)
        finally:
            # Exceptions do not imply child termination. Preserve uncertainty.
            for process, log, attempt, request in active.values():
                attempt['state'] = 'unknown'
                log.close()
            _write(path, ledger)
        report = collect_plan(plan, output_root, required_profile=observer)
        ledger['canonical'] = report['canonical']
        _write(path, ledger)
        report['attempts_consumed'] = ledger['attempts_consumed']
        report['dispatch_blocked'] = blocked
        report['lifecycle_complete'] = all(a.get('lifecycle') == 'exited'
                                         for r in ledger['tasks'].values() for a in r['attempts'])
        if not report['lifecycle_complete'] or blocked:
            report['status'] = 'incomplete'
        return report
