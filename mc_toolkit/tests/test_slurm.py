"""Pure mocked host orchestration; never execute Slurm or Docker."""
from datetime import datetime, timedelta, timezone
import json
import subprocess
from types import SimpleNamespace

import pytest

from physica_mc import slurm


@pytest.fixture(autouse=True)
def no_real_commands(monkeypatch):
    monkeypatch.setattr(slurm, '_command', lambda *a, **k: pytest.fail('real cluster command forbidden'))


def prepared(tmp_path, *, count=2, budget_changes=None, gpu_tokens=0, service_changes=None):
    data = tmp_path / 'input.json'
    data.write_text('{}')
    digest = 'registry/example@sha256:' + 'a' * 64
    budget = dict(deadline_utc=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                  max_jobs=4, max_concurrent=2, cpu_limit=4, max_gpu_concurrent=0,
                  max_attempts=4, max_cpu_hours=2, max_gpu_hours=0)
    budget.update(budget_changes or {})
    inner = dict(deadline_utc=budget['deadline_utc'], max_attempts=1,
                 max_workers=1, cpu_limit=2, max_gpu_workers=0)
    data.write_text(json.dumps({'budget': inner}))
    return slurm.prepare_submission(
        campaign_root=tmp_path / 'campaign', campaign_id='technical', cluster='research_docker',
        budget=budget, compose={'services': {'compute': {'image': digest, 'entrypoint': [],
                            'environment': {'OMP_NUM_THREADS': '1'}, **(service_changes or {})}}},
        commands=[['python', '-m', 'physica_mc.cli', 'run', '--request', f'/inputs/{i}.json'] for i in range(count)],
        inputs={str(data): slurm._hash(data)}, image_digest=digest,
        attempts_per_element=[1] * count,
        run_requests=[{'host_path': str(data), 'container_path': f'/inputs/{i}.json'} for i in range(count)],
        profile=dict(cpus=2, memory_mib=2048, wall_seconds=60, concurrency=2,
                     gpu_tokens=gpu_tokens, partition='docker_cpu'))


def reply(stdout, returncode=0):
    return SimpleNamespace(stdout=stdout, stderr='', returncode=returncode)


def local_command(stdout):
    def command(argv, timeout):
        assert timeout == 30
        assert not any(arg.startswith('--clusters') for arg in argv)
        if argv == ['scontrol', 'show', 'config']:
            return reply('ClusterName = research_docker\n')
        assert argv[0] == 'sbatch' or argv[:3] == ['scontrol', 'show', 'job']
        return reply(stdout)
    return command


def test_exact_array_submission_and_budget(tmp_path):
    rec = prepared(tmp_path)
    calls = []
    def command(argv, timeout):
        calls.append(argv)
        assert timeout == 30
        assert not any(arg.startswith('--clusters') for arg in argv)
        if argv == ['scontrol', 'show', 'config']:
            return reply('ClusterName = research_docker\n')
        assert argv[0] == 'sbatch'
        campaign = slurm._read(tmp_path / 'campaign' / 'campaign.json')
        intent = campaign['submissions'][rec['token']]
        assert intent['state'] == 'unknown'
        assert intent['submit_argv'] == argv
        assert intent['count'] == 2 and intent['attempts'] == 2
        assert intent['profile'] == rec['profile']
        return reply('123;research_docker\n')
    done = slurm.submit_submission(tmp_path / 'campaign', rec['token'], command=command)
    assert set(done['elements']) == {'123_0', '123_1'}
    assert calls[0] == ['scontrol', 'show', 'config']
    assert len(calls) == 2
    assert '--array=0-1%2' in calls[1]
    assert '--nodes=1' in calls[1] and '--ntasks=1' in calls[1] and '--no-requeue' in calls[1]
    assert '--comment=' + rec['comment'] in calls[1]
    text = open(rec['batch']).read()
    assert slurm.RUNNER in text and '--service compute --gpu-tokens 0' in text
    with pytest.raises(ValueError, match='already attempted'):
        slurm.submit_submission(tmp_path / 'campaign', rec['token'], command=command)


def test_timeout_unknown_and_disappearance_never_resubmits(tmp_path):
    rec = prepared(tmp_path)
    def timeout(argv, timeout):
        if argv == ['scontrol', 'show', 'config']:
            return reply('ClusterName = research_docker\n')
        assert argv[0] == 'sbatch'
        raise subprocess.TimeoutExpired(argv, timeout)
    result = slurm.submit_submission(tmp_path / 'campaign', rec['token'], command=timeout)
    assert result['state'] == 'unknown'
    result = slurm.reconcile_submission(tmp_path / 'campaign', rec['token'], command=local_command(''))
    assert result['state'] == 'unknown'
    with pytest.raises(ValueError):
        slurm.submit_submission(tmp_path / 'campaign', rec['token'], command=lambda *a, **k: pytest.fail('resubmit'))


def rows(rec, root, state='COMPLETED'):
    return '\n'.join(
        f"JobId={123+i} ArrayJobId=123 ArrayTaskId={i} Comment={rec['comment']} "
        f"JobName=mc-{rec['token']} Command={rec['batch']} WorkDir={root / rec['token']} "
        f"UserId=test({slurm.os.getuid()}) NodeList=node01 JobState={state} ExitCode=0:0"
        for i in range(2))


def test_exact_comment_recovery_and_cleanup_separate(tmp_path):
    rec = prepared(tmp_path)
    root = tmp_path / 'campaign'
    slurm.submit_submission(root, rec['token'], command=local_command('ambiguous'))
    result = slurm.reconcile_submission(root, rec['token'], command=local_command(rows(rec, root)))
    assert result['job_id'] == '123' and result['state'] == 'terminal'
    assert result['cleanup'] == 'unknown' and result['data_acceptance'] == 'unverified'
    wrong = rows(rec, root).replace('NodeList=node01', 'NodeList=other')
    evidence = {'123_0': {'labels': {}, 'containers_absent': True, 'networks_absent': True}}
    with pytest.raises(ValueError, match='ownership'):
        slurm.reconcile_submission(root, rec['token'], command=local_command(wrong), cleanup_evidence=evidence)


def test_modified_inputs_refused_and_no_implicit_live(tmp_path):
    rec = prepared(tmp_path)
    root = tmp_path / 'campaign'
    with pytest.raises(ValueError, match='live'):
        slurm.submit_submission(root, rec['token'])
    (tmp_path / 'input.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='changed'):
        slurm.submit_submission(root, rec['token'], command=lambda *a, **k: pytest.fail('changed input launched'))


def test_preparation_resource_guard(tmp_path):
    with pytest.raises(ValueError, match='budget'):
        prepared(tmp_path, budget_changes={'cpu_limit': 1})


def test_exact_cleanup_evidence_and_missing_queue_preserves_terminal(tmp_path):
    rec = prepared(tmp_path)
    root = tmp_path / 'campaign'
    slurm.submit_submission(root, rec['token'], command=local_command('123'))
    slurm.reconcile_submission(root, rec['token'], command=local_command(rows(rec, root)))
    evidence = {}
    for index in range(2):
        evidence[f'123_{index}'] = {
            'labels': {slurm.LABEL + 'campaign': 'technical', slurm.LABEL + 'submission': rec['token'],
                       slurm.RESERVED + 'cluster': 'research_docker',
                       slurm.RESERVED + 'job': str(123 + index),
                       slurm.RESERVED + 'uid': str(slurm.os.getuid()),
                       slurm.RESERVED + 'node': 'node01', slurm.RESERVED + 'nonce': f'nonce-{index}',
                       slurm.RESERVED + 'array_job': '123', slurm.RESERVED + 'array_task': str(index)},
            'container_id': f'cid-{index}', 'command': rec['commands'][index],
            'containers_absent': True, 'networks_absent': True}
    result = slurm.reconcile_submission(root, rec['token'], command=local_command(''), cleanup_evidence=evidence)
    assert result['state'] == 'terminal' and result['cleanup'] == 'verified'
    assert result['data_acceptance'] == 'unverified'


def test_unknown_submission_globally_blocks_next_prepared(tmp_path):
    first = prepared(tmp_path)
    root = tmp_path / 'campaign'
    campaign = slurm._read(root / 'campaign.json')
    # Build a second immutable submission with exactly the same approval.
    compose = {'services': {'compute': {'image': first['image_digest']}}}
    data = str(tmp_path / 'input.json')
    second = slurm.prepare_submission(
        campaign_root=root, campaign_id='technical', cluster='research_docker',
        budget=campaign['budget'], compose=compose, commands=first['commands'],
        inputs={data: slurm._hash(data)}, image_digest=first['image_digest'], profile=first['profile'],
        attempts_per_element=[1, 1], run_requests=[{'host_path': data, 'container_path': f'/inputs/{i}.json'} for i in range(2)])
    slurm.submit_submission(root, first['token'], command=local_command('ambiguous'))
    with pytest.raises(ValueError, match='unknown'):
        slurm.submit_submission(root, second['token'], command=lambda *a, **k: pytest.fail('unknown permits launch'))


@pytest.mark.parametrize('hours', [-1, True, float('nan'), float('inf')])
def test_gpu_hours_must_be_explicit_finite(tmp_path, hours):
    with pytest.raises(ValueError, match='max_gpu_hours'):
        prepared(tmp_path, budget_changes={'max_gpu_hours': hours})


def test_gpu_hours_preparation_and_compose_preserved(tmp_path):
    with pytest.raises(ValueError, match='GPU-hours'):
        prepared(tmp_path, gpu_tokens=1, budget_changes={'max_gpu_concurrent': 2})
    service = {'gpus': [{'driver': 'nvidia', 'count': 'all'}], 'cpus': 2,
               'mem_limit': '2g', 'environment': {'CUDA_VISIBLE_DEVICES': '0'}}
    rec = prepared(tmp_path, gpu_tokens=1, budget_changes={'max_gpu_concurrent': 2, 'max_gpu_hours': 4},
                   service_changes=service)
    actual = slurm._read(rec['compose'])['services']['compute']
    assert all(actual[key] == value for key, value in service.items())
    assert rec['run_requests'] == slurm._read(tmp_path / 'campaign' / rec['token'] / 'prepared.json')['run_requests']


@pytest.mark.parametrize('state,cleanup', [('unknown', 'unknown'), ('terminal', 'unknown'), ('terminal', 'verified')])
def test_gpu_hours_never_refunded(tmp_path, state, cleanup):
    first = prepared(tmp_path, gpu_tokens=1,
                     budget_changes={'max_gpu_concurrent': 2, 'max_gpu_hours': 0.05})
    root = tmp_path / 'campaign'
    campaign = slurm._read(root / 'campaign.json')
    data = str(tmp_path / 'input.json')
    second = slurm.prepare_submission(
        campaign_root=root, campaign_id='technical', cluster='research_docker',
        budget=campaign['budget'], compose={'services': {'compute': {'image': first['image_digest']}}},
        commands=first['commands'], inputs={data: slurm._hash(data)}, image_digest=first['image_digest'],
        profile=first['profile'], attempts_per_element=[1, 1], run_requests=first['run_requests'])
    slurm.submit_submission(root, first['token'], command=local_command('123'))
    campaign = slurm._read(root / 'campaign.json')
    prior = campaign['submissions'][first['token']]
    prior.update(state=state, cleanup=cleanup)
    for element in prior['elements'].values():
        element.update(state='FAILED', cleanup=cleanup)
    slurm._write(root / 'campaign.json', campaign)
    with pytest.raises(ValueError, match='GPU-hour budget'):
        slurm.submit_submission(root, second['token'], command=lambda *a, **k: pytest.fail('GPU hours exhausted'))


@pytest.mark.parametrize('state,cleanup', [('submitted', 'unknown'), ('terminal', 'unknown')])
def test_unresolved_reservations_block_concurrency(tmp_path, state, cleanup):
    first = prepared(tmp_path)
    root = tmp_path / 'campaign'
    campaign = slurm._read(root / 'campaign.json')
    data = str(tmp_path / 'input.json')
    second = slurm.prepare_submission(
        campaign_root=root, campaign_id='technical', cluster='research_docker',
        budget=campaign['budget'], compose={'services': {'compute': {'image': first['image_digest']}}},
        commands=first['commands'], inputs={data: slurm._hash(data)}, image_digest=first['image_digest'],
        profile=first['profile'], attempts_per_element=[1, 1], run_requests=first['run_requests'])
    slurm.submit_submission(root, first['token'], command=local_command('123'))
    campaign = slurm._read(root / 'campaign.json')
    campaign['submissions'][first['token']].update(state=state, cleanup=cleanup)
    slurm._write(root / 'campaign.json', campaign)
    with pytest.raises(ValueError, match='reservation'):
        slurm.submit_submission(root, second['token'], command=lambda *a, **k: pytest.fail('reserved resources'))


@pytest.mark.parametrize('change', ['RUNNING', 'node02'])
def test_later_controller_observation_invalidates_old_cleanup(tmp_path, change):
    rec = prepared(tmp_path)
    root = tmp_path / 'campaign'
    slurm.submit_submission(root, rec['token'], command=local_command('123'))
    slurm.reconcile_submission(root, rec['token'], command=local_command(rows(rec, root)))
    path = root / 'campaign.json'
    campaign = slurm._read(path)
    record = campaign['submissions'][rec['token']]
    record['cleanup'] = 'verified'
    for element in record['elements'].values():
        element.update(cleanup='verified', cleanup_evidence={'old': 'observation'})
    slurm._write(path, campaign)
    observed = rows(rec, root, state='RUNNING') if change == 'RUNNING' else rows(rec, root).replace('node01', 'node02')
    result = slurm.reconcile_submission(root, rec['token'], command=local_command(observed))
    assert result['cleanup'] == 'unknown'
    assert all(e['cleanup'] == 'unknown' and 'cleanup_evidence' not in e for e in result['elements'].values())
    if change == 'RUNNING':
        assert result['state'] == 'submitted'


def test_missing_gpu_hours_is_not_implicit_approval(tmp_path):
    rec = prepared(tmp_path)
    root = tmp_path / 'campaign'
    path = root / 'campaign.json'
    campaign = slurm._read(path)
    del campaign['budget']['max_gpu_hours']
    slurm._write(path, campaign)
    before = path.read_bytes()
    with pytest.raises((KeyError, ValueError), match='max_gpu_hours'):
        slurm.submit_submission(root, rec['token'], command=lambda *a, **k: pytest.fail('missing approval'))
    assert path.read_bytes() == before


@pytest.mark.parametrize('config', [
    'ClusterName = foreign\n', '', 'OtherClusterName = research_docker\n',
    'ClusterName = research_docker\nClusterName = research_docker\n',
    'ClusterName = research_docker\nClusterName = foreign\n',
    'ClusterName =\n', 'ClusterName = research_docker extra\n',
    'ClusterName = research_docker\n' + 'x' * (1024 * 1024),
    'nonzero', 'timeout', 'unavailable',
])
@pytest.mark.parametrize('operation', ['submit', 'reconcile'])
def test_local_cluster_preflight_failure_preserves_record(tmp_path, config, operation):
    rec = prepared(tmp_path)
    root = tmp_path / 'campaign'
    if operation == 'reconcile':
        slurm.submit_submission(root, rec['token'], command=local_command('123'))
    path = root / 'campaign.json'
    before = path.read_bytes()
    calls = []
    def command(argv, timeout):
        calls.append(argv)
        assert argv == ['scontrol', 'show', 'config']
        assert timeout == 30
        if config == 'timeout':
            raise subprocess.TimeoutExpired(argv, timeout)
        if config == 'unavailable':
            raise OSError('controller unavailable')
        return reply(config, returncode=1 if config == 'nonzero' else 0)
    operation = slurm.submit_submission if operation == 'submit' else slurm.reconcile_submission
    with pytest.raises(ValueError, match='cluster'):
        operation(root, rec['token'], command=command)
    assert calls == [['scontrol', 'show', 'config']]
    assert path.read_bytes() == before


def test_local_reconcile_preserves_legacy_submit_argv(tmp_path):
    rec = prepared(tmp_path)
    root = tmp_path / 'campaign'
    slurm.submit_submission(root, rec['token'], command=local_command('123'))
    path = root / 'campaign.json'
    campaign = slurm._read(path)
    legacy_argv = campaign['submissions'][rec['token']]['submit_argv']
    legacy_argv.insert(2, '--clusters=research_docker')
    slurm._write(path, campaign)
    calls = []
    def command(argv, timeout):
        calls.append(argv)
        return local_command(rows(rec, root))(argv, timeout)
    result = slurm.reconcile_submission(root, rec['token'], command=command)
    assert calls == [['scontrol', 'show', 'config'],
                     ['scontrol', 'show', 'job', '123', '--oneliner']]
    assert result['state'] == 'terminal' and result['cleanup'] == 'unknown'
    assert result['submit_argv'] == legacy_argv
    assert slurm._read(path)['submissions'][rec['token']]['submit_argv'] == legacy_argv


def precreation_case(tmp_path):
    rec = prepared(tmp_path)
    root = tmp_path / 'campaign'
    slurm.submit_submission(root, rec['token'], command=local_command('123'))
    now = slurm.time.time()
    end = datetime.fromtimestamp(now, timezone.utc).replace(microsecond=0)
    controller, evidence = [], {}
    for index, line in enumerate(rows(rec, root, state='FAILED').splitlines()):
        log = root / rec['token'] / 'logs' / f'123_{index}.out'
        log.write_bytes(b'slurm-docker: array context mismatch\n')
        controller.append(line.replace('ExitCode=0:0', 'ExitCode=1:0') +
                          f" EndTime={end.replace(tzinfo=None).isoformat()} StdOut={log}")
        evidence[f'123_{index}'] = dict(
            kind='precreation_failure',
            identity=dict(campaign_id='technical', submission=rec['token'],
                          cluster='research_docker', uid=str(slurm.os.getuid()),
                          job_id=str(123 + index), node='node01',
                          array_job_id='123', array_task_id=str(index)),
            command=rec['commands'][index], containers_absent=True, networks_absent=True,
            containers_observed_at=now, networks_observed_at=now,
            controller_end_time=end.isoformat(),
            provenance='Operator: exact identity-filtered containers and networks inspected; runner verified',
            runner_sha256='6628196afd7960de719f960947cacdbc8d165f0224da91843dc2775000d8ea11',
            failure_log_path=str(log), failure_log_sha256=slurm._hash(log))
    return root, rec, '\n'.join(controller), evidence


def test_precreation_failure_cleanup_without_container_identity(tmp_path):
    root, rec, controller, evidence = precreation_case(tmp_path)
    before = slurm._read(root / 'campaign.json')
    result = slurm.reconcile_submission(root, rec['token'], command=local_command(controller),
                                        cleanup_evidence=evidence)
    assert result['state'] == 'terminal' and result['cleanup'] == 'verified'
    assert result['data_acceptance'] == 'unverified'
    assert result['attempts'] == rec['attempts'] and result['count'] == rec['count']
    assert slurm._read(root / 'campaign.json')['budget'] == before['budget']
    for jid, element in result['elements'].items():
        assert element['cleanup_evidence'] == evidence[jid]
        assert 'container_id' not in element['cleanup_evidence']
        assert 'labels' not in element['cleanup_evidence']
    changed = controller.replace('ExitCode=1:0', 'ExitCode=2:0')
    result = slurm.reconcile_submission(root, rec['token'], command=local_command(changed))
    assert result['cleanup'] == 'unknown'


@pytest.mark.parametrize('field', [
    'campaign_id', 'submission', 'cluster', 'uid', 'job_id', 'node', 'array_job_id', 'array_task_id'])
def test_precreation_rejects_wrong_identity(tmp_path, field):
    root, rec, controller, evidence = precreation_case(tmp_path)
    evidence['123_0']['identity'][field] = 'wrong'
    with pytest.raises(ValueError, match='precreation'):
        slurm.reconcile_submission(root, rec['token'], command=local_command(controller),
                                    cleanup_evidence=evidence)


@pytest.mark.parametrize('case', [
    'queue_absence', 'duplicate', 'completed', 'zero_exit', 'multi_node', 'wrong_job',
    'wrong_array', 'missing_log', 'changed_log', 'unknown_phase', 'wrong_log_path',
    'wrong_log_hash', 'wrong_runner', 'no_provenance', 'uid_only', 'terminal_only',
    'containers_present', 'networks_present', 'stale', 'future', 'before_end',
    'naive_end', 'wrong_end', 'wrong_command', 'changed_batch', 'changed_prepared', 'invented_cid'])
def test_precreation_rejects_insufficient_proof(tmp_path, case):
    root, rec, controller, evidence = precreation_case(tmp_path)
    proof = evidence['123_0']
    log = root / rec['token'] / 'logs' / '123_0.out'
    if case == 'queue_absence':
        slurm.reconcile_submission(root, rec['token'], command=local_command(controller))
        controller = ''
    elif case == 'duplicate':
        controller += '\n' + controller.splitlines()[0]
    elif case == 'completed':
        controller = controller.replace('JobState=FAILED', 'JobState=COMPLETED')
    elif case == 'zero_exit':
        controller = controller.replace('ExitCode=1:0', 'ExitCode=0:0')
    elif case == 'multi_node':
        controller = controller.replace('NodeList=node01', 'NodeList=node[01-02]')
        proof['identity']['node'] = 'node[01-02]'
    elif case == 'wrong_job':
        controller = controller.replace('JobId=123 Array', 'JobId=999 Array')
    elif case == 'wrong_array':
        controller = controller.replace('ArrayTaskId=0', 'ArrayTaskId=9')
    elif case == 'missing_log':
        log.unlink()
    elif case in ('changed_log', 'unknown_phase'):
        log.write_text('slurm-docker: arbitrary failure\n')
        if case == 'unknown_phase':
            proof['failure_log_sha256'] = slurm._hash(log)
    elif case == 'wrong_log_path':
        proof['failure_log_path'] = evidence['123_1']['failure_log_path']
    elif case == 'wrong_log_hash':
        proof['failure_log_sha256'] = '0' * 64
    elif case == 'wrong_runner':
        proof['runner_sha256'] = '0' * 64
    elif case == 'no_provenance':
        proof['provenance'] = ' '
    elif case == 'uid_only':
        proof['identity'] = {'uid': str(slurm.os.getuid())}
    elif case == 'terminal_only':
        proof.clear()
        proof['kind'] = 'precreation_failure'
    elif case == 'containers_present':
        proof['containers_absent'] = False
    elif case == 'networks_present':
        proof['networks_absent'] = False
    elif case == 'stale':
        proof['containers_observed_at'] -= 301
    elif case == 'future':
        proof['networks_observed_at'] += 3600
    elif case == 'before_end':
        proof['containers_observed_at'] = datetime.fromisoformat(proof['controller_end_time']).timestamp() - 1
    elif case == 'naive_end':
        proof['controller_end_time'] = proof['controller_end_time'][:-6]
    elif case == 'wrong_end':
        proof['controller_end_time'] = '2000-01-01T00:00:00+00:00'
    elif case == 'wrong_command':
        proof['command'] = ['true']
    elif case == 'changed_batch':
        with open(rec['batch'], 'a') as stream:
            stream.write('\ntrue\n')
    elif case == 'changed_prepared':
        campaign = slurm._read(root / 'campaign.json')
        campaign['submissions'][rec['token']]['commands'][0] = ['true']
        proof['command'] = ['true']
        slurm._write(root / 'campaign.json', campaign)
    elif case == 'invented_cid':
        proof['container_id'] = 'invented'
    expected_error = 'duplicate jobs for one submission token' if case == 'wrong_job' else 'precreation'
    with pytest.raises(ValueError, match=expected_error):
        slurm.reconcile_submission(root, rec['token'], command=local_command(controller),
                                    cleanup_evidence=evidence)
    persisted = slurm._read(root / 'campaign.json')['submissions'][rec['token']]
    assert persisted['cleanup'] == 'unknown'


def test_precreation_releases_only_concurrency_for_approved_ordinary_job(tmp_path):
    root, rec, controller, evidence = precreation_case(tmp_path)
    campaign = slurm._read(root / 'campaign.json')
    data = str(tmp_path / 'input.json')
    next_rec = slurm.prepare_submission(
        campaign_root=root, campaign_id=rec['campaign_id'], cluster=rec['cluster'],
        budget=campaign['budget'], compose={'services': {'compute': {'image': rec['image_digest']}}},
        commands=rec['commands'][:1], inputs={data: slurm._hash(data)},
        image_digest=rec['image_digest'], profile=rec['profile'],
        attempts_per_element=[1], run_requests=rec['run_requests'][:1])
    with pytest.raises(ValueError, match='reservation'):
        slurm.submit_submission(root, next_rec['token'], command=local_command('999'))
    slurm.reconcile_submission(root, rec['token'], command=local_command(controller),
                               cleanup_evidence=evidence)
    result = slurm.submit_submission(root, next_rec['token'], command=local_command('999'))
    assert result['state'] == 'submitted'
    assert not any(arg.startswith('--array=') for arg in result['submit_argv'])
    campaign = slurm._read(root / 'campaign.json')
    assert sum(r['count'] for r in campaign['submissions'].values()) == 3
    assert sum(r['attempts'] for r in campaign['submissions'].values()) == 3
    assert campaign['submissions'][rec['token']]['job_id'] == '123'
