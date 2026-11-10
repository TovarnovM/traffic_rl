"""Parent-run tests only: tiny technical episodes, never cluster commands."""
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path

import pytest

from physica_mc import runner, schema, storage
from physica_mc.planning import make_plan


def budget(**changes):
    value = dict(deadline_utc=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
                 max_attempts=4, max_workers=2, cpu_limit=4, max_gpu_workers=0)
    value.update(changes)
    return value


def task(replicate=0):
    return schema.make_task('B1', density=0.05, replicate=replicate,
                            config={'warmup_steps': 1, 'measure_steps': 3, 'backend': 'reference'})


def run(tmp_path, plan, limits, **kwargs):
    return runner.run_plan(plan, output_root=tmp_path / 'output',
                           spool_root=tmp_path / 'spool', budget=limits, **kwargs)


def test_fresh_children_resume_and_spool(tmp_path):
    plan = make_plan([task(0), task(1)])
    limits = budget()
    result = run(tmp_path, plan, limits, workers=2, trace=True)
    assert result['status'] == 'complete'
    assert result['attempts_consumed'] == 2
    ledger = runner._read(runner._registry(plan, tmp_path / 'output') / 'ledger.json')
    owners = [record['attempts'][0]['receipt']['owner'] for record in ledger['tasks'].values()]
    assert all(owner['pid'] != os.getpid() for owner in owners)
    assert len({(o['pid'], o['start_ticks']) for o in owners}) == 2
    assert all(o['boot_id'] for o in owners)
    again = run(tmp_path, plan, limits, workers=2, trace=True)
    assert again['canonical'] == result['canonical']
    assert again['attempts_consumed'] == 2
    for marker in result['canonical'].values():
        payload, _ = storage.read_episode(marker)
        source = tmp_path / 'spool' / payload['task']['task_id'] / payload['attempt_id'] / 'completion.json'
        assert source.exists()
        assert runner.publish_spool(source, tmp_path / 'output') == Path(marker)


def test_guards_before_launch(tmp_path, monkeypatch):
    plan = make_plan([task()])
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not launch'))
    with pytest.raises(ValueError, match='CPU'):
        run(tmp_path, plan, budget(cpu_limit=1), workers=2)
    with pytest.raises(ValueError, match='GPU'):
        run(tmp_path, plan, budget(), device='cuda')
    result = run(tmp_path, plan, budget(deadline_utc='2000-01-01T00:00:00Z'))
    assert result['status'] == 'incomplete'
    assert result['attempts_consumed'] == 0


def test_unknown_intent_blocks_after_lock_release(tmp_path, monkeypatch):
    plan = make_plan([task()])
    limits = budget(deadline_utc='2000-01-01T00:00:00Z')
    run(tmp_path, plan, limits)
    path = runner._registry(plan, tmp_path / 'output') / 'ledger.json'
    ledger = runner._read(path)
    ledger['attempts_consumed'] = 1
    ledger['tasks'][task()['task_id']]['attempts'] = [
        {'attempt_id': 'uncertain', 'state': 'intent', 'owner': None, 'lifecycle': 'unknown'}]
    runner._write(path, ledger)
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **k: pytest.fail('blind resimulation'))
    result = run(tmp_path, plan, limits)
    assert result['status'] == 'incomplete'
    assert result['attempts_consumed'] == 1
    assert result['lifecycle'][task()['task_id']][0]['state'] == 'unknown'


def test_publication_failure_replays_without_simulation(tmp_path, monkeypatch):
    from physica_mc import episode
    t = task()
    request = dict(task=t, execution=schema.make_execution(t), attempt_id='stable-attempt',
                   output_root=str(tmp_path / 'output'), spool_root=str(tmp_path / 'spool'),
                   receipt=str(tmp_path / 'receipt.json'), trace=True,
                   deadline_utc=budget()['deadline_utc'], model=None)
    real_publish = runner.publish_spool
    monkeypatch.setattr(runner, 'publish_spool', lambda *a, **k: (_ for _ in ()).throw(OSError('publication unavailable')))
    with pytest.raises(OSError, match='unavailable'):
        runner.episode_request(request)
    receipt = runner._read(request['receipt'])
    assert receipt['state'] == 'failed' and receipt['stage'] == 'publication'
    assert not (tmp_path / 'output').exists()
    monkeypatch.setattr(episode, 'simulate_episode', lambda *a, **k: pytest.fail('must reuse spool'))
    marker = real_publish(receipt['spool_marker'], tmp_path / 'output')
    assert storage.read_episode(marker)[0]['attempt_id'] == 'stable-attempt'


def test_partial_failure_has_diagnostics_no_completion(tmp_path, monkeypatch):
    from physica_mc import episode
    def fail(*args, **kwargs):
        error = episode.EpisodeIncomplete('partial')
        error.partial = {'steps_completed': 1}
        raise error
    monkeypatch.setattr(episode, 'simulate_episode', fail)
    t = task()
    request = dict(task=t, execution=schema.make_execution(t), attempt_id='partial',
                   output_root=str(tmp_path / 'out'), spool_root=str(tmp_path / 'spool'),
                   receipt=str(tmp_path / 'receipt.json'), trace=False,
                   deadline_utc=budget()['deadline_utc'])
    with pytest.raises(episode.EpisodeIncomplete):
        runner.episode_request(request)
    assert runner._read(request['receipt'])['partial'] == {'steps_completed': 1}
    assert not list(tmp_path.rglob('completion.json'))


def test_duplicate_owned_attempts_are_conflicts_and_orphans_invalid(tmp_path):
    t = task()
    plan = make_plan([t])
    result = run(tmp_path, plan, budget())
    marker = result['canonical'][t['task_id']]
    payload, arrays = storage.read_episode(marker)
    storage.publish_episode(tmp_path / 'output', t, payload['execution'], 'duplicate', payload['result'], arrays)
    path = runner._registry(plan, tmp_path / 'output') / 'ledger.json'
    ledger = runner._read(path)
    ledger['tasks'][t['task_id']]['attempts'].append(
        {'attempt_id': 'duplicate', 'state': 'complete', 'lifecycle': 'exited'})
    runner._write(path, ledger)
    orphan = tmp_path / 'output' / t['task_id'] / 'orphan'
    orphan.mkdir()
    (orphan / 'arrays.npz').write_bytes(b'partial')
    collected = runner.collect_plan(plan, tmp_path / 'output')
    assert t['task_id'] in collected['conflicting']
    assert not collected['accepted'] and not collected['canonical']
    assert collected['invalid']


def test_attempt_budget_persists(tmp_path):
    plan = make_plan([task(0), task(1)])
    limits = budget(max_attempts=1)
    first = run(tmp_path, plan, limits)
    second = run(tmp_path, plan, limits)
    assert first['attempts_consumed'] == second['attempts_consumed'] == 1
    assert second['status'] == 'incomplete'
    assert len(second['accepted']) == 1


def test_source_guard_and_exclusive_child_attempt(tmp_path, monkeypatch):
    from physica_mc import episode
    t = task()
    execution = schema.make_execution(t)
    request = dict(task=t, execution=execution, attempt_id='guarded',
                   output_root=str(tmp_path / 'out'), spool_root=str(tmp_path / 'spool'),
                   receipt=str(tmp_path / 'receipt.json'), trace=False,
                   deadline_utc=budget()['deadline_utc'])
    monkeypatch.setattr(schema, 'source_identity', lambda: {'changed': True})
    monkeypatch.setattr(episode, 'simulate_episode', lambda *a, **k: pytest.fail('source guard bypassed'))
    with pytest.raises(ValueError, match='implementation changed'):
        runner.episode_request(request)
    assert runner._read(request['receipt'])['state'] == 'failed'
    with pytest.raises(ValueError, match='immutable'):
        runner.episode_request(request)


def test_sequential_parallel_preserve_state_and_rng(tmp_path):
    plan = make_plan([task(2), task(3)])
    serial = run(tmp_path / 'serial', plan, budget(), trace=True)
    parallel = run(tmp_path / 'parallel', plan, budget(), workers=2, trace=True)
    assert serial['status'] == parallel['status'] == 'complete'
    for tid in serial['canonical']:
        left, _ = storage.read_episode(serial['canonical'][tid])
        right, _ = storage.read_episode(parallel['canonical'][tid])
        assert left['task'] == right['task']
        for key in ('initial_state_sha256', 'final_state_sha256', 'initial_rng_sha256', 'final_rng_sha256'):
            assert left['result'][key] == right['result'][key]


@pytest.mark.parametrize('alive', [True, None, False])
@pytest.mark.parametrize('accepted', [False, True])
def test_prior_owner_reserves_all_dispatch(tmp_path, monkeypatch, alive, accepted):
    plan = make_plan([task(0), task(1)])
    limits = budget()
    # Initialize durable settings without launching or changing the budget.
    monkeypatch.setattr(runner.time, 'time', lambda: runner._deadline(limits['deadline_utc']) + 1)
    run(tmp_path, plan, limits)
    path = runner._registry(plan, tmp_path / 'output') / 'ledger.json'
    ledger = runner._read(path)
    tid = task(0)['task_id']
    ledger['tasks'][tid]['attempts'] = [dict(attempt_id='old', state='intent',
                                           lifecycle='unknown', owner=None)]
    ledger['attempts_consumed'] = 1
    runner._write(path, ledger)
    monkeypatch.setattr(runner.time, 'time', lambda: 0)
    monkeypatch.setattr(runner, '_alive', lambda owner: alive)
    if accepted:
        collect = runner.collect_plan
        def accepted_first(*args, **kwargs):
            result = collect(*args, **kwargs)
            result['accepted'][tid] = 'verified-existing-marker'
            return result
        monkeypatch.setattr(runner, 'collect_plan', accepted_first)
    calls = []
    def launch(*args, **kwargs):
        calls.append(args)
        raise OSError('known launch failure')
    monkeypatch.setattr(runner.subprocess, 'Popen', launch)
    result = run(tmp_path, plan, limits)
    assert len(calls) == (1 if alive is False else 0)
    assert result['attempts_consumed'] == (2 if alive is False else 1)
    assert result['dispatch_blocked'] is (alive is not False)


def test_root_lock_serializes_different_plans(tmp_path, monkeypatch):
    first, second = make_plan([task(0)]), make_plan([task(0), task(1)])
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **k: pytest.fail('root locked'))
    with runner._lock(tmp_path / 'output' / '.runs' / 'coordinator.lock'):
        for plan in (first, second):
            with pytest.raises(BlockingIOError):
                run(tmp_path, plan, budget())
    assert not list((tmp_path / 'output').glob('.runs/*/ledger.json'))


def test_foreign_plan_attempt_never_adopted_after_lock_release(tmp_path, monkeypatch):
    first, second = make_plan([task(0)]), make_plan([task(0), task(1)])
    limits = budget()
    monkeypatch.setattr(runner.time, 'time', lambda: runner._deadline(limits['deadline_utc']) + 1)
    run(tmp_path, first, limits)
    path = runner._registry(first, tmp_path / 'output') / 'ledger.json'
    ledger = runner._read(path)
    ledger['tasks'][task(0)['task_id']]['attempts'] = [dict(attempt_id='old', lifecycle='unknown')]
    runner._write(path, ledger)
    monkeypatch.setattr(runner.time, 'time', lambda: 0)
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **k: pytest.fail('foreign owner'))
    with pytest.raises(ValueError, match='owned by another plan'):
        run(tmp_path, second, limits)
    # Even nonoverlapping work in this root cannot consume resources while
    # foreign children remain unreconciled.
    result = run(tmp_path, make_plan([task(2)]), limits)
    assert result['dispatch_blocked'] and result['attempts_consumed'] == 0


def test_collect_shards_exact_coverage_and_separate_lifecycle(tmp_path):
    parent = make_plan([task(0), task(1)])
    shards, canonical = [], {}
    for index in range(2):
        plan = make_plan([task(index)])
        root = tmp_path / str(index)
        report = run(root, plan, budget())
        canonical.update(report['canonical'])
        shards.append(dict(plan=plan, output_root=root / 'output'))
    report = runner.collect_shards(parent, shards)
    assert report['status'] == report['data_status'] == 'complete'
    assert report['canonical'] == canonical
    path = runner._registry(shards[0]['plan'], shards[0]['output_root']) / 'ledger.json'
    ledger = runner._read(path)
    ledger['tasks'][task(0)['task_id']]['attempts'][0]['lifecycle'] = 'unknown'
    runner._write(path, ledger)
    report = runner.collect_shards(parent, shards)
    assert report['data_status'] == 'complete' and report['status'] == 'incomplete'
    assert report['canonical'] == canonical and not report['lifecycle_complete']
    with pytest.raises(ValueError, match='exactly cover'):
        runner.collect_shards(parent, shards[:1])
    with pytest.raises(ValueError, match='overlapping'):
        runner.collect_shards(parent, [shards[0], dict(shards[0], output_root=tmp_path / 'other')])
    with pytest.raises(ValueError, match='unexpected'):
        runner.collect_shards(parent, [dict(plan=make_plan([task(9)]), output_root=tmp_path / 'other')])
    with pytest.raises(ValueError, match='disjoint'):
        runner.collect_shards(parent, [shards[0], dict(shards[1], output_root=shards[0]['output_root'])])
    wrong_profile = runner.collect_shards(parent, shards, required_profile='not-full12')
    assert wrong_profile['status'] == 'incomplete' and not wrong_profile['accepted']


@pytest.fixture
def external_owner(tmp_path):
    from physica_mc import slurm
    from types import SimpleNamespace
    plan = make_plan([task()])
    limits = budget(max_attempts=1, max_workers=1, cpu_limit=1)
    result = run(tmp_path, plan, limits)
    output = tmp_path / 'output'
    path = runner._registry(plan, output) / 'ledger.json'
    ledger = runner._read(path)
    attempt = ledger['tasks'][task()['task_id']]['attempts'][0]
    owner = dict(attempt['owner'], host='old-container-hostname')
    receipt_path = Path(attempt['request']).with_name('receipt.json')
    receipt = runner._read(receipt_path)
    receipt['owner'] = owner
    runner._write(receipt_path, receipt)
    attempt.update(owner=owner, lifecycle='unknown', state='unknown')
    runner._write(path, ledger)
    request_path = runner._write(tmp_path / 'run.json', dict(
        plan=plan, output_root=str(output), spool_root=str(tmp_path / 'spool'), budget=limits))
    campaign_root = tmp_path / 'campaign'
    digest = 'registry/test@sha256:' + 'a' * 64
    rec = slurm.prepare_submission(
        campaign_root=campaign_root, campaign_id='recovery', cluster='test',
        budget=dict(deadline_utc=limits['deadline_utc'], max_jobs=1, max_concurrent=1,
                    cpu_limit=1, max_gpu_concurrent=0, max_attempts=1, max_cpu_hours=1, max_gpu_hours=0),
        compose={'services': {'compute': {'image': digest}}},
        commands=[['python', '-m', 'physica_mc.cli', 'run', '--request', '/inputs/run.json']],
        inputs={str(request_path): slurm._hash(request_path)}, image_digest=digest,
        profile=dict(cpus=1, memory_mib=1024, wall_seconds=60, concurrency=1, gpu_tokens=0, partition='cpu'),
        attempts_per_element=[1], run_requests=[dict(host_path=str(request_path), container_path='/inputs/run.json')])
    def reply(text):
        return SimpleNamespace(stdout=text, stderr='', returncode=0)
    def local_reply(text):
        def command(argv, **kwargs):
            if argv == ['scontrol', 'show', 'config']:
                return reply('ClusterName = test\n')
            return reply(text)
        return command
    slurm.submit_submission(campaign_root, rec['token'], command=local_reply('123'))
    row = (f"JobId=123 Comment={rec['comment']} JobName=mc-{rec['token']} "
           f"Command={rec['batch']} WorkDir={campaign_root / rec['token']} "
           f"UserId=test({os.getuid()}) NodeList=node01 JobState=FAILED ExitCode=1:0")
    cleanup = dict(labels={
        slurm.LABEL + 'campaign': 'recovery', slurm.LABEL + 'submission': rec['token'],
        slurm.RESERVED + 'cluster': 'test', slurm.RESERVED + 'job': '123',
        slurm.RESERVED + 'node': 'node01', slurm.RESERVED + 'uid': str(os.getuid()),
        slurm.RESERVED + 'nonce': 'exact-nonce', slurm.RESERVED + 'array_job': 'none',
        slurm.RESERVED + 'array_task': 'none'}, container_id='exact-container',
        command=rec['commands'][0], containers_absent=True, networks_absent=True)
    slurm.reconcile_submission(campaign_root, rec['token'], command=local_reply(row),
                               cleanup_evidence={'123': cleanup})
    evidence = dict(container_id='exact-container', owner=owner,
                    request_sha256=slurm._hash(attempt['request']),
                    provenance='Operator test; exact request/owner observed in this CID; terminal and absence observed.')
    args = dict(plan=plan, output_root=output, campaign_root=campaign_root, token=rec['token'],
                element_id='123', attempt_id=attempt['attempt_id'], evidence=evidence)
    return args, path, limits, result


@pytest.mark.parametrize('legacy', [False, True])
def test_external_owner_reconciliation_and_spool_replay(external_owner, monkeypatch, legacy):
    import shutil
    args, path, limits, original = external_owner
    if legacy:
        frozen_path = args['campaign_root'] / args['token'] / 'prepared.json'
        frozen = runner._read(frozen_path)
        args['evidence']['run_request'] = frozen.pop('run_requests')[0]
        runner._write(frozen_path, frozen)
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **k: pytest.fail('reconciliation must not simulate'))
    assert runner._alive(args['evidence']['owner']) is None
    report = runner.reconcile_owner(**args)
    assert report['lifecycle'] == 'exited' and report['attempts_consumed'] == 1
    assert runner.reconcile_owner(**args) == report
    ledger = runner._read(path)
    assert ledger['settings']['budget'] == limits
    assert ledger['tasks'][task()['task_id']]['attempts'][0]['state'] == 'unknown'
    shutil.rmtree(args['output_root'] / task()['task_id'])
    replay = runner.run_plan(args['plan'], output_root=args['output_root'],
                             spool_root=ledger['settings']['spool_root'], budget=limits)
    assert replay['status'] == 'complete' and replay['canonical'] == original['canonical']
    assert replay['attempts_consumed'] == 1


@pytest.mark.parametrize('field,value', [('container_id', 'foreign'), ('owner', {'pid': 1}),
                                        ('request_sha256', '0' * 64), ('provenance', '')])
def test_external_owner_rejects_unbound_evidence(external_owner, field, value):
    args, path, _, _ = external_owner
    before = path.read_bytes()
    args['evidence'][field] = value
    with pytest.raises(ValueError):
        runner.reconcile_owner(**args)
    assert path.read_bytes() == before


@pytest.mark.parametrize('mutation', ['nonterminal', 'no_cleanup', 'wrong_node', 'changed_request'])
def test_external_owner_requires_exact_submission_evidence(external_owner, mutation):
    args, path, _, _ = external_owner
    before = path.read_bytes()
    campaign_path = args['campaign_root'] / 'campaign.json'
    campaign = runner._read(campaign_path)
    rec = campaign['submissions'][args['token']]
    element = rec['elements']['123']
    if mutation == 'nonterminal':
        element['state'] = 'RUNNING'
    elif mutation == 'no_cleanup':
        element['cleanup'] = 'unknown'
    elif mutation == 'wrong_node':
        element['node'] = 'foreign'
    else:
        Path(rec['run_requests'][0]['host_path']).write_text('{}')
    runner._write(campaign_path, campaign)
    with pytest.raises(ValueError):
        runner.reconcile_owner(**args)
    assert path.read_bytes() == before


def test_real_accepted_output_does_not_release_live_owner_capacity(tmp_path, monkeypatch):
    plan = make_plan([task(0), task(1)])
    limits = budget()
    with monkeypatch.context() as clock:
        clock.setattr(runner.time, 'time', lambda: runner._deadline(limits['deadline_utc']) + 1)
        run(tmp_path, plan, limits)
    path = runner._registry(plan, tmp_path / 'output') / 'ledger.json'
    ledger = runner._read(path)
    record = ledger['tasks'][task(0)['task_id']]
    directory = path.parent / 'attempts' / 'prior'
    request = dict(task=task(0), execution=record['execution'], attempt_id='prior',
                   output_root=str(tmp_path / 'output'), spool_root=str(tmp_path / 'spool'),
                   receipt=str(directory / 'receipt.json'), trace=False,
                   deadline_utc=limits['deadline_utc'], model=None)
    request_path = runner._write(directory / 'request.json', request, exclusive=True)
    receipt = runner.episode_request(request)
    record['attempts'].append(dict(attempt_id='prior', state='complete', lifecycle='unknown',
                                   owner=receipt['owner'], request=str(request_path)))
    ledger['attempts_consumed'] = 1
    runner._write(path, ledger)
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **k: pytest.fail('live owner reserves capacity'))
    result = run(tmp_path, plan, limits)
    assert task(0)['task_id'] in result['accepted']
    assert result['attempts_consumed'] == 1 and result['dispatch_blocked']
    assert result['status'] == 'incomplete'


def test_collect_shards_rejects_insufficient_bound_profile(tmp_path):
    plan = make_plan([task()])
    run(tmp_path, plan, budget(), observer='off')
    report = runner.collect_shards(plan, [dict(plan=plan, output_root=tmp_path / 'output')])
    assert report['invalid'] and not report['canonical'] and report['status'] == 'incomplete'
