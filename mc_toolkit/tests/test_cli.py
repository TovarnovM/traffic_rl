"""Parent executes these short real CLI integrations inside the container."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys

from physica_mc import cli


def invoke(tmp_path, command, request, *extra):
    path = tmp_path / (command + '.json')
    path.write_text(json.dumps(request))
    result = subprocess.run([sys.executable, '-m', 'physica_mc.cli', command,
                             '--request', str(path), *extra], capture_output=True, text=True, timeout=90)
    return result, json.loads(result.stdout)


def test_help_stdlib_only():
    # -S excludes site packages; only the owned source directory is made visible.
    code = "import sys; sys.path.insert(0, sys.argv[1]); from physica_mc.cli import main; main(['--help'])"
    root = str(Path(cli.__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, '-S', '-c', code, root], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'slurm-prepare' in result.stdout


def test_real_short_cli_plan_run_verify_collect(tmp_path):
    result, task = invoke(tmp_path, 'task', {'method': 'B1', 'density': 0.05,
                        'config': {'warmup_steps': 1, 'measure_steps': 2, 'backend': 'reference'}})
    assert result.returncode == 0, result.stdout + result.stderr
    plan_path = tmp_path / 'plan-output.json'
    result, plan = invoke(tmp_path, 'plan', {'tasks': [task], 'destination': str(plan_path)})
    assert result.returncode == 0, result.stdout + result.stderr
    request = dict(plan=str(plan_path), output_root=str(tmp_path / 'output'),
                   spool_root=str(tmp_path / 'spool'), budget=dict(
                       deadline_utc=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
                       max_attempts=1, max_workers=1, cpu_limit=1, max_gpu_workers=0))
    result, report = invoke(tmp_path, 'run', request)
    assert result.returncode == 0, result.stdout + result.stderr
    marker = report['canonical'][task['task_id']]
    result, verified = invoke(tmp_path, 'verify', {'path': marker, 'required_profile': 'full12'})
    assert result.returncode == 0 and verified['task_id'] == task['task_id']
    result, collected = invoke(tmp_path, 'collect', {'plan': str(plan_path), 'output_root': request['output_root']})
    assert result.returncode == 0 and len(collected['accepted']) == 1
    result, repeated = invoke(tmp_path, 'run', request)
    assert result.returncode == 0 and repeated['attempts_consumed'] == 1


def test_failed_cli_is_nonzero_and_convert_requires_trust(tmp_path):
    result, error = invoke(tmp_path, 'verify', {'path': str(tmp_path / 'missing')})
    assert result.returncode != 0 and error['status'] == 'failed'
    result, error = invoke(tmp_path, 'convert', {'path': 'missing', 'destination': 'never-created'})
    assert result.returncode != 0 and '--trusted-legacy' in error['error']


def test_collect_explicit_shards_cli(tmp_path):
    from physica_mc import schema
    from physica_mc.planning import make_plan, write_plan
    tasks = [schema.make_task('B1', density=0.05, replicate=i,
                             config={'warmup_steps': 1, 'measure_steps': 2, 'backend': 'reference'})
             for i in range(2)]
    parent = make_plan(tasks)
    shards = []
    for index, task in enumerate(tasks):
        path = tmp_path / f'plan-{index}.json'
        write_plan(make_plan([task]), path)
        shards.append(dict(plan=str(path), output_root=str(tmp_path / f'output-{index}')))
    result, report = invoke(tmp_path, 'collect', dict(plan=parent, shards=shards))
    assert result.returncode == 1 and report['status'] == 'incomplete'
    assert set(report['missing']) == {task['task_id'] for task in tasks}
    assert report['canonical'] == {} and report['data_status'] == 'incomplete'
    result, report = invoke(tmp_path, 'collect', dict(plan=parent, shards=shards[:1]))
    assert result.returncode == 1 and 'exactly cover' in report['error']


def test_reconcile_owner_cli_dispatch(tmp_path, monkeypatch):
    from physica_mc import runner, schema
    from physica_mc.planning import make_plan, write_plan
    plan = make_plan([schema.make_task('B1', density=0.05)])
    path = tmp_path / 'plan.json'
    write_plan(plan, path)
    calls = []
    def reconcile(**request):
        calls.append(request)
        return {'status': 'reconciled', 'lifecycle': 'exited'}
    monkeypatch.setattr(runner, 'reconcile_owner', reconcile)
    request = dict(plan=str(path), output_root='output', campaign_root='campaign', token='token',
                   element_id='123_0', attempt_id='attempt', evidence={'provenance': 'operator'})
    request_path = tmp_path / 'reconcile.json'
    request_path.write_text(json.dumps(request))
    assert cli.main(['reconcile-owner', '--request', str(request_path)]) == 0
    assert calls == [dict(request, plan=plan)]
