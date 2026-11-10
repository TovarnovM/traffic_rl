"""Standard-library host adapter for the installed Slurm Docker runner.

Preparation never submits. Submission requires an explicit live=True argument;
unit tests inject command explicitly. No Docker, SSH or cancellation fallback.
"""
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import uuid

from .runner import _deadline, _lock, _positive, _read, _write

RUNNER = '/usr/local/libexec/slurm-docker/runner.py'
LABEL = 'org.physica.mc.'
RESERVED = 'org.research.slurm.'
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _command(argv, timeout=30):
    return subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=False)


def _validate_local_cluster(cluster, command):
    """Fail closed before dispatch; local access needs no accounting DB."""
    try:
        response = command(['scontrol', 'show', 'config'], timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError('local cluster configuration unavailable') from exc
    if response.returncode != 0 or len(response.stdout) > 1024 * 1024:
        raise ValueError('local cluster configuration unavailable or exceeds bound')
    entries = re.findall(r'^[ \t]*ClusterName\b[^\r\n]*', response.stdout, re.MULTILINE)
    if len(entries) != 1:
        raise ValueError('local cluster configuration requires exactly one ClusterName')
    match = re.fullmatch(
        r'[ \t]*ClusterName[ \t]*=[ \t]*([A-Za-z0-9][A-Za-z0-9_.-]{0,100})[ \t]*', entries[0])
    if not match:
        raise ValueError('local cluster configuration has invalid ClusterName')
    if match[1] != cluster:
        raise ValueError('requested cluster does not match local ClusterName')


def _budget(budget):
    for key in ('max_jobs', 'max_concurrent', 'cpu_limit', 'max_attempts'):
        _positive(budget[key], key)
    _positive(budget['max_gpu_concurrent'], 'max_gpu_concurrent', zero=True)
    if type(budget['max_cpu_hours']) not in (int, float) or not math.isfinite(budget['max_cpu_hours']) or budget['max_cpu_hours'] <= 0:
        raise ValueError('finite positive max_cpu_hours required')
    if (type(budget['max_gpu_hours']) not in (int, float)
            or not math.isfinite(budget['max_gpu_hours']) or budget['max_gpu_hours'] < 0):
        raise ValueError('finite nonnegative max_gpu_hours required')
    _deadline(budget['deadline_utc'])


def _token(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', value):
        raise ValueError('unsafe campaign/partition/node token')
    return value


def prepare_submission(*, campaign_root, campaign_id, budget, compose,
                       commands, inputs, image_digest, profile, cluster, attempts_per_element, run_requests):
    """Freeze prepared JSON Compose and literal argv per element; no execution.

    inputs is {absolute_file: sha256}, including code, models and task/requests.
    compose is an already prepared JSON-native Compose mapping. No resource or
    visibility settings are manufactured. commands preserve its entrypoint.
    """
    _budget(budget)
    _token(campaign_id)
    _token(cluster)
    root = Path(campaign_root).resolve()
    if not commands or not all(isinstance(c, list) and c and all(isinstance(s, str) and s and '\x00' not in s for s in c) for c in commands):
        raise ValueError('nonempty literal command argv required per element')
    if not inputs:
        raise ValueError('explicit frozen code/model/task input hashes required')
    for path, digest in inputs.items():
        if not Path(path).is_absolute() or any(c in path for c in ('\n', '\r', '\\')) or _hash(path) != digest:
            raise ValueError('input hash mismatch: ' + path)
    if not re.fullmatch(r'[^\s]+@sha256:[0-9a-f]{64}', image_digest):
        raise ValueError('operator-provided exact cached image digest required')
    for key in ('cpus', 'memory_mib', 'wall_seconds', 'concurrency'):
        _positive(profile[key], key)
    if type(profile['gpu_tokens']) is not int or profile['gpu_tokens'] not in (0, 1):
        raise ValueError('gpu_tokens must be 0 or 1')
    _token(profile['partition'])
    for key in ('node', 'constraint'):
        if profile.get(key):
            _token(profile[key])
    if profile['wall_seconds'] % 60:
        raise ValueError('wall_seconds must be a whole number of Slurm minutes')
    if len(attempts_per_element) != len(commands):
        raise ValueError('explicit attempt allocation required for every element')
    for count in attempts_per_element:
        _positive(count, 'attempt allocation')
    n, concurrent = len(commands), min(len(commands), profile['concurrency'])
    if len(run_requests) != n:
        raise ValueError('one host-container run-request binding per element required')
    for index, binding in enumerate(run_requests):
        host, container = binding['host_path'], binding['container_path']
        if host not in inputs or not Path(container).is_absolute():
            raise ValueError('run request must be hashed with absolute container path')
        if commands[index][-5:] != ['-m', 'physica_mc.cli', 'run', '--request', container]:
            raise ValueError('each element must invoke the bounded run CLI')
        request = _read(host)
        inner = request['budget']
        for key in ('max_attempts', 'max_workers', 'cpu_limit', 'max_gpu_workers'):
            _positive(inner[key], key, zero=key == 'max_gpu_workers')
        if (inner['max_attempts'] != attempts_per_element[index]
                or _deadline(inner['deadline_utc']) > _deadline(budget['deadline_utc'])
                or inner['cpu_limit'] > profile['cpus']
                or inner['max_gpu_workers'] > profile['gpu_tokens']):
            raise ValueError('element budget exceeds its reserved allocation')
        workers, threads = request.get('workers', 1), request.get('threads', 1)
        _positive(workers, 'workers')
        _positive(threads, 'threads')
        if workers > inner['max_workers'] or workers * threads > inner['cpu_limit']:
            raise ValueError('element allocation exceeds CPU reservation')
    if (n > budget['max_jobs'] or sum(attempts_per_element) > budget['max_attempts']
            or concurrent > budget['max_concurrent']
            or concurrent * profile['cpus'] > budget['cpu_limit']
            or concurrent * profile['gpu_tokens'] > budget['max_gpu_concurrent']):
        raise ValueError('prepared submission exceeds budget')
    if n * profile['cpus'] * profile['wall_seconds'] / 3600 > budget['max_cpu_hours']:
        raise ValueError('reserved CPU-hours exceed budget')
    if n * profile['gpu_tokens'] * profile['wall_seconds'] / 3600 > budget['max_gpu_hours']:
        raise ValueError('reserved GPU-hours exceed budget')
    # JSON roundtrip makes an owned copy without YAML dependencies on the host.
    compose = json.loads(json.dumps(compose))
    if set(compose.get('services', {})) != {'compute'}:
        raise ValueError('prepare exactly one compute service')
    service = compose['services']['compute']
    if service.get('image') != image_digest:
        raise ValueError('Compose image does not match exact digest')
    if any(key in service for key in ('build', 'env_file', 'depends_on', 'develop', 'post_start', 'pre_stop')):
        raise ValueError('external/mutable Compose inputs or hooks are not supported')
    if '${' in json.dumps(compose):
        raise ValueError('resolve Compose interpolation explicitly before freezing')
    labels = service.setdefault('labels', {})
    if not isinstance(labels, dict) or any(k.startswith((RESERVED, LABEL)) for k in labels):
        raise ValueError('reserved ownership labels already set')
    for volume in service.get('volumes', []):
        if not isinstance(volume, dict) or volume.get('type') != 'bind' or not Path(volume.get('source', '')).is_absolute():
            raise ValueError('use explicit absolute bind mounts')
        if volume.get('bind', {}).get('create_host_path') is not False:
            raise ValueError('bind.create_host_path must be false')
    with _lock(root / 'campaign.lock'):
        campaign_path = root / 'campaign.json'
        if campaign_path.exists():
            campaign = _read(campaign_path)
            if (campaign['budget'], campaign['campaign_id'], campaign['cluster']) != (budget, campaign_id, cluster):
                raise ValueError('campaign identity/budget is immutable')
        else:
            campaign = {'campaign_id': campaign_id, 'cluster': cluster, 'budget': budget, 'submissions': {}}
        token = uuid.uuid4().hex
        directory = root / token
        directory.mkdir(parents=True)
        (directory / 'logs').mkdir()
        labels.update({LABEL + 'campaign': campaign_id, LABEL + 'submission': token})
        compose_path = _write(directory / 'compose.json', compose, exclusive=True)
        batch = directory / 'job.sbatch'
        node_inputs = dict(inputs)
        node_inputs[str(compose_path)] = _hash(compose_path)
        check = ('printf ' + shlex.quote('%s\n') + ' ' +
                 shlex.join([digest + '  ' + name for name, digest in node_inputs.items()]) +
                 ' | sha256sum --check --status')
        lines = ['#!/bin/bash', '#SBATCH --nodes=1', '#SBATCH --ntasks=1',
                 '#SBATCH --no-requeue', 'set -euo pipefail',
                 ': "${SLURM_JOB_ID:?real allocation required}"',
                 check,
                 'case "${SLURM_ARRAY_TASK_ID:-0}" in']
        for index, argv in enumerate(commands):
            command = [RUNNER, '--compose', str(compose_path), '--service', 'compute',
                       '--gpu-tokens', str(profile['gpu_tokens']), '--', *argv]
            lines.append(f'{index}) exec {shlex.join(command)} ;;')
        lines.extend(['*) exit 64 ;;', 'esac', ''])
        with batch.open('x') as stream:
            stream.write('\n'.join(lines))
            stream.flush()
            os.fsync(stream.fileno())
        hashes = dict(inputs)
        hashes[str(compose_path)] = _hash(compose_path)
        hashes[str(batch)] = _hash(batch)
        record = {'token': token, 'comment': f'physica-mc:{campaign_id}:{token}',
                  'campaign_id': campaign_id, 'cluster': cluster, 'state': 'prepared',
                  'run_requests': run_requests, 'profile': profile, 'commands': commands, 'inputs': hashes,
                  'image_digest': image_digest, 'batch': str(batch),
                  'compose': str(compose_path), 'count': n, 'attempts': sum(attempts_per_element),
                  'attempts_per_element': attempts_per_element, 'concurrency': concurrent,
                  'cleanup': 'unknown', 'data_acceptance': 'unverified', 'elements': {}}
        _write(directory / 'prepared.json', record, exclusive=True)
        campaign['submissions'][token] = record
        _write(campaign_path, campaign)
        return record


def submit_submission(campaign_root, token, *, live=False, command=None):
    if not live and command is None:
        raise ValueError('submission requires explicit live=True; tests must inject command')
    command = _command if command is None else command
    root = Path(campaign_root).resolve()
    with _lock(root / 'campaign.lock'):
        path = root / 'campaign.json'
        campaign = _read(path)
        rec = campaign['submissions'][token]
        if rec['state'] != 'prepared':
            raise ValueError('submission already attempted; reconcile, never blind resubmit')
        budget, profile = campaign['budget'], rec['profile']
        _budget(budget)
        for name, digest in rec['inputs'].items():
            if _hash(name) != digest:
                raise ValueError('frozen input changed: ' + name)
        previous = [r for r in campaign['submissions'].values() if r['state'] != 'prepared']
        # Reserve worst-case walltime cost, never refund an ambiguous submission.
        if (sum(r['count'] for r in previous) + rec['count'] > budget['max_jobs']
                or sum(r['attempts'] for r in previous) + rec['attempts'] > budget['max_attempts']):
            raise ValueError('persistent total job/attempt budget exhausted')
        hours = sum(r['count'] * r['profile']['cpus'] * r['profile']['wall_seconds'] / 3600 for r in [*previous, rec])
        if hours > budget['max_cpu_hours']:
            raise ValueError('persistent CPU-hour budget exhausted')
        gpu_hours = sum(r['count'] * r['profile']['gpu_tokens'] * r['profile']['wall_seconds'] / 3600
                        for r in [*previous, rec])
        if gpu_hours > budget['max_gpu_hours']:
            raise ValueError('persistent GPU-hour budget exhausted')
        active = [r for r in previous if r['state'] != 'terminal' or r['cleanup'] != 'verified']
        if any(r['state'] == 'unknown' for r in active):
            raise ValueError('unknown submission blocks dispatch')
        for key, limit in [('concurrency', budget['max_concurrent']), ('cpu', budget['cpu_limit']), ('gpu', budget['max_gpu_concurrent'])]:
            def cost(r):
                return r['concurrency'] * (1 if key == 'concurrency' else r['profile']['cpus' if key == 'cpu' else 'gpu_tokens'])
            if sum(cost(r) for r in [*active, rec]) > limit:
                raise ValueError('global concurrency/resource reservation exceeded')
        _validate_local_cluster(campaign['cluster'], command)
        remaining = _deadline(budget['deadline_utc']) - time.time()
        if remaining < profile['wall_seconds'] + 60:
            raise ValueError('insufficient deadline for finite job walltime')
        # Slurm deadline also prevents a queued job starting too late to finish.
        argv = ['sbatch', '--parsable',
                '--job-name=mc-' + token, '--comment=' + rec['comment'],
                '--nodes=1', '--ntasks=1', '--no-requeue',
                '--partition=' + profile['partition'], '--cpus-per-task=' + str(profile['cpus']),
                '--mem=' + str(profile['memory_mib']) + 'M',
                '--time=' + str(math.ceil(profile['wall_seconds'] / 60)),
                '--deadline=' + time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(_deadline(budget['deadline_utc']))),
                '--chdir=' + str(root / token),
                '--output=' + str(root / token / 'logs' / '%A_%a.out')]
        if rec['count'] > 1:
            argv.append(f"--array=0-{rec['count'] - 1}%{rec['concurrency']}")
        if profile['gpu_tokens']:
            argv.append('--gres=gpu_slot:1')
        for key, flag in [('node', '--nodelist='), ('constraint', '--constraint=')]:
            if profile.get(key):
                argv.append(flag + profile[key])
        argv.append(rec['batch'])
        rec.update(state='unknown', submit_argv=argv, submit_time=time.time(), submit_uid=os.getuid())
        _write(path, campaign)  # durable unique token and budget reservation BEFORE sbatch
        try:
            result = command(argv, timeout=30)
            rec['submit_returncode'] = result.returncode
            rec['submit_stdout'] = result.stdout[:4096]
            rec['submit_stderr'] = getattr(result, 'stderr', '')[:4096]
            match = re.fullmatch(r'([1-9][0-9]*)(?:;([A-Za-z0-9_.-]+))?\s*', result.stdout)
            if result.returncode != 0 or not match or (match[2] and match[2] != campaign['cluster']):
                rec['error'] = 'ambiguous sbatch receipt; exact-token reconciliation required'
            else:
                _bind_job(rec, match[1])
        except (OSError, subprocess.TimeoutExpired) as exc:
            rec['error'] = str(exc)
        _write(path, campaign)
        return rec


def _bind_job(rec, job):
    rec.update(state='submitted', job_id=job)
    ids = [f'{job}_{i}' for i in range(rec['count'])] if rec['count'] > 1 else [job]
    rec['elements'] = {jid: {'state': 'unknown', 'cleanup': 'unknown'} for jid in ids}


def _fields(line):
    return dict(part.split('=', 1) for part in shlex.split(line) if '=' in part)


def _validate_cleanup(rec, token, jid, evidence):
    element = rec['elements'][jid]
    index = int(jid.split('_')[1]) if '_' in jid else 0
    labels = evidence['labels']
    expected = {LABEL + 'campaign': rec['campaign_id'], LABEL + 'submission': token,
                RESERVED + 'cluster': rec['cluster'], RESERVED + 'job': element.get('job_id'),
                RESERVED + 'node': element.get('node'), RESERVED + 'uid': str(rec['submit_uid']),
                RESERVED + 'array_job': rec['job_id'] if '_' in jid else 'none',
                RESERVED + 'array_task': str(index) if '_' in jid else 'none'}
    if (element['state'] not in TERMINAL or not element.get('job_id') or not element.get('node')
            or any(labels.get(k) != v for k, v in expected.items())
            or not labels.get(RESERVED + 'nonce') or not evidence.get('container_id')
            or evidence.get('command') != rec['commands'][index]
            or evidence.get('containers_absent') is not True
            or evidence.get('networks_absent') is not True):
        raise ValueError('cleanup evidence does not establish exact ownership/absence')


# This reviewed runner exits before its first Docker command for this message.
_PRECREATION_RUNNER_SHA256 = '6628196afd7960de719f960947cacdbc8d165f0224da91843dc2775000d8ea11'
_PRECREATION_LOG = b'slurm-docker: array context mismatch\n'


def _validate_precreation(root, rec, token, jid, evidence, row, now):
    def reject():
        raise ValueError('precreation evidence does not establish exact ownership/absence')

    if row is None:
        reject()
    index = int(jid.split('_')[1]) if '_' in jid else 0
    expected = dict(campaign_id=rec['campaign_id'], submission=token,
                    cluster=rec['cluster'], uid=str(rec['submit_uid']),
                    job_id=row.get('JobId'), node=row.get('NodeList'),
                    array_job_id=rec['job_id'] if '_' in jid else 'none',
                    array_task_id=str(index) if '_' in jid else 'none')
    exit_code = re.fullmatch(r'([0-9]+):([0-9]+)', row.get('ExitCode', ''))
    if (row.get('JobState') != 'FAILED' or not exit_code
            or not any(int(value) for value in exit_code.groups())
            or not re.fullmatch(r'[1-9][0-9]*', row.get('JobId', ''))
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', row.get('NodeList', ''))
            or row.get('NodeList') in ('None', 'none', 'Unknown')
            or (rec['profile'].get('node') and row['NodeList'] != rec['profile']['node'])
            or ('_' in jid and (row.get('ArrayJobId') != rec['job_id']
                                or row.get('ArrayTaskId') != str(index)))
            or ('_' not in jid and ('ArrayJobId' in row or 'ArrayTaskId' in row))
            or evidence.get('identity') != expected
            or evidence.get('command') != rec['commands'][index]
            or evidence.get('containers_absent') is not True
            or evidence.get('networks_absent') is not True
            or not isinstance(evidence.get('provenance'), str)
            or not evidence['provenance'].strip()
            or evidence.get('runner_sha256') != _PRECREATION_RUNNER_SHA256
            or any(key in evidence for key in ('container_id', 'nonce', 'labels'))):
        reject()
    try:
        prepared = _read(root / token / 'prepared.json')
        mutable = {'state', 'cleanup', 'data_acceptance', 'elements'}
        if (any(rec.get(key) != value for key, value in prepared.items() if key not in mutable)
                or prepared['token'] != token
                or prepared['batch'] != str(root / token / 'job.sbatch')
                or prepared['compose'] != str(root / token / 'compose.json')):
            reject()
        for name in (prepared['batch'], prepared['compose']):
            if _hash(name) != prepared['inputs'][name]:
                reject()
        # Require the operator's actual controller offset, not the host timezone.
        end = datetime.fromisoformat(evidence['controller_end_time'])
        if (end.tzinfo is None
                or end.replace(tzinfo=None).isoformat(timespec='seconds') != row.get('EndTime')):
            reject()
        for key in ('containers_observed_at', 'networks_observed_at'):
            observed = evidence[key]
            if (type(observed) not in (int, float) or not math.isfinite(observed)
                    or not max(end.timestamp(), rec['submit_time'], now - 300) <= observed <= now):
                reject()
        log = Path(evidence['failure_log_path'])
        if (not log.is_absolute() or str(log) != row.get('StdOut')
                or log.resolve().parent != (root / token / 'logs').resolve()):
            reject()
        with log.open('rb') as stream:
            content = stream.read(len(_PRECREATION_LOG) + 1)
        if (content != _PRECREATION_LOG
                or evidence.get('failure_log_sha256') != hashlib.sha256(content).hexdigest()):
            reject()
    except (KeyError, OSError, TypeError, ValueError, OverflowError):
        reject()


def reconcile_submission(campaign_root, token, *, command=None, cleanup_evidence=None):
    """Bounded local config and job queries; no accounting or resubmission.

    cleanup_evidence: operator-observed per-element exact labels, CID, node,
    command, and containers_absent/networks_absent booleans. Never inferred from
    queue absence. kind=precreation_failure requires a fresh controller row
    and the narrow operator proof documented in EXECUTION.md.
    This function does not access Docker or SSH.
    """
    command = _command if command is None else command
    root = Path(campaign_root).resolve()
    with _lock(root / 'campaign.lock'):
        path = root / 'campaign.json'
        campaign = _read(path)
        rec = campaign['submissions'][token]
        if rec['state'] == 'prepared':
            raise ValueError('no submission intent to reconcile')
        _validate_local_cluster(campaign['cluster'], command)
        if rec['cluster'] != campaign['cluster']:
            raise ValueError('submission cluster does not match campaign')
        argv = ['scontrol', 'show', 'job']
        if rec.get('job_id'):
            argv.append(rec['job_id'])
        argv.append('--oneliner')
        response = command(argv, timeout=30)
        if response.returncode != 0 or len(response.stdout) > 8 * 1024 * 1024:
            rec['reconcile_error'] = 'controller unavailable or response exceeds bound'
            _write(path, campaign)
            return rec
        matches = []
        for line in response.stdout.splitlines():
            row = _fields(line)
            if row.get('Comment') == rec['comment']:
                if (row.get('JobName') != 'mc-' + token or row.get('Command') != rec['batch']
                        or row.get('WorkDir') != str(root / token)
                        or not row.get('UserId', '').endswith(f"({rec['submit_uid']})")):
                    raise ValueError('exact-token record has conflicting owner/command')
                matches.append(row)
        bases = {r.get('ArrayJobId', r.get('JobId')) for r in matches}
        if not rec.get('job_id'):
            if len(bases) != 1 or not re.fullmatch(r'[1-9][0-9]*', next(iter(bases), '')):
                rec['reconcile_error'] = 'submission remains unknown; no unique exact-token job'
                _write(path, campaign)
                return rec
            _bind_job(rec, bases.pop())
        fresh_rows = {}
        for row in matches:
            if row.get('ArrayJobId', row.get('JobId')) != rec['job_id']:
                raise ValueError('duplicate jobs for one submission token')
            jid = (row['ArrayJobId'] + '_' + row['ArrayTaskId']) if 'ArrayTaskId' in row else row['JobId']
            if jid not in rec['elements']:
                continue  # compressed ranges are not element receipts
            fresh_rows.setdefault(jid, []).append(row)
            element = rec['elements'][jid]
            if (row.get('JobState') not in TERMINAL
                    or element.get('job_id') != row['JobId']
                    or element.get('node') != row.get('NodeList')
                    or (element.get('cleanup_evidence', {}).get('kind') == 'precreation_failure'
                        and element.get('controller') != row)):
                element['cleanup'] = 'unknown'
                element.pop('cleanup_evidence', None)
            rec['elements'][jid].update(state=row.get('JobState', 'unknown'),
                                        job_id=row['JobId'], node=row.get('NodeList'),
                                        exit_code=row.get('ExitCode'), controller=row,
                                        observed_at=time.time())
        for jid, evidence in (cleanup_evidence or {}).items():
            element = rec['elements'][jid]
            if evidence.get('kind') == 'precreation_failure':
                current = fresh_rows.get(jid, [])
                _validate_precreation(root, rec, token, jid, evidence,
                                      current[0] if len(current) == 1 else None, time.time())
            elif evidence.get('kind') is None:
                _validate_cleanup(rec, token, jid, evidence)
            else:
                raise ValueError('unknown cleanup evidence kind')
            element.update(cleanup='verified', cleanup_evidence=evidence)
        if rec['elements']:
            rec['state'] = 'terminal' if all(e['state'] in TERMINAL for e in rec['elements'].values()) else 'submitted'
            rec['cleanup'] = 'verified' if all(e['cleanup'] == 'verified' for e in rec['elements'].values()) else 'unknown'
        _write(path, campaign)
        return rec
