"""JSON request CLI; importing this module and --help need only stdlib."""
import argparse
import json
from pathlib import Path


def _load(path):
    def reject(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def _dispatch(command, request, *, trusted_legacy=False, live=False):
    if command == 'task':
        from .schema import make_task
        return make_task(**request)
    if command == 'plan':
        from .planning import make_plan, write_plan
        request = dict(request)
        destination = request.pop('destination')
        plan = make_plan(**request)
        write_plan(plan, destination)
        return plan
    if command == 'convert':
        from .model_convert import convert_checkpoint
        if not trusted_legacy:
            raise ValueError('legacy conversion requires --trusted-legacy')
        return {'status': 'complete', 'bundle': str(convert_checkpoint(**request, trusted_legacy=True))}
    if command == 'episode':
        from .runner import episode_request
        return episode_request(request)
    if command in ('run', 'collect', 'reconcile-owner'):
        from .planning import load_plan
        from .runner import run_plan, collect_plan, collect_shards, reconcile_owner
        request = dict(request)
        if isinstance(request['plan'], str):
            request['plan'] = load_plan(request['plan'])
        if command == 'reconcile-owner':
            return reconcile_owner(**request)
        if command == 'collect' and 'shards' in request:
            request['parent_plan'] = request.pop('plan')
            return collect_shards(**request)
        return (run_plan if command == 'run' else collect_plan)(**request)
    if command == 'verify':
        from .storage import read_episode
        payload, _ = read_episode(**request)
        return {'status': 'complete', 'task_id': payload['task']['task_id'],
                'execution_id': payload['execution']['execution_id'], 'attempt_id': payload['attempt_id']}
    if command == 'publish':
        from .runner import publish_spool
        return {'status': 'complete', 'marker': str(publish_spool(**request))}
    if command == 'analyze':
        from .postprocess import write_analysis
        return {'status': 'complete', 'analysis': str(write_analysis(**request))}
    if command == 'export':
        from .reporting import stream_episode_tables
        return stream_episode_tables(**request)
    if command == 'select':
        from .selection import select_group_json
        return select_group_json(**request)
    if command == 'freeze':
        from . import selection as s
        groups = [s.group_from_json(g) for g in request['expected_groups']]
        selections, pools = {}, {}
        for row in request['groups']:
            group = s.group_from_json(row['group'])
            if group in pools:
                raise ValueError('duplicate freeze group')
            pools[group] = [s.candidate_from_json(c) for c in row['candidates']]
            selections[group] = s.group_selection_from_json(row['selection'])
        return s.freeze_receipt(selections, pools,
                                provenance=s.provenance_from_json(request['provenance']),
                                expected_groups=groups)
    if command.startswith('slurm-'):
        from . import slurm
        if command == 'slurm-prepare':
            return slurm.prepare_submission(**request)
        if command == 'slurm-submit':
            if not live:
                raise ValueError('Slurm submission requires --live')
            return slurm.submit_submission(**request, live=True)
        return slurm.reconcile_submission(**request)
    raise ValueError('unknown command')


def _successful(result):
    status = result.get('status', result.get('state'))
    if status in ('failed', 'incomplete', 'unknown', 'running', 'spooled', 'constraint_undefined'):
        return False
    if (result.get('counts', {}).get('invalid', 0)
            or result.get('counts', {}).get('points_missing_planned', 0)):
        return False
    if result.get('invalid') or result.get('conflicting') or result.get('missing'):
        return False
    if status == 'terminal':
        return result.get('cleanup') == 'verified' and result.get('data_acceptance') == 'complete'
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('task', 'plan', 'convert', 'episode', 'run', 'verify', 'collect',
                 'publish', 'analyze', 'export', 'select', 'freeze', 'slurm-prepare',
                 'slurm-submit', 'slurm-reconcile', 'reconcile-owner'):
        child = sub.add_parser(name, help='JSON request; see docs/EXECUTION.md')
        child.add_argument('--request', required=True, help='JSON keyword arguments for the documented API')
        child.add_argument('--output', help='write immutable JSON response as well as stdout')
        if name == 'convert':
            child.add_argument('--trusted-legacy', action='store_true')
        if name == 'slurm-submit':
            child.add_argument('--live', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = _dispatch(args.command, _load(args.request),
                           trusted_legacy=getattr(args, 'trusted_legacy', False),
                           live=getattr(args, 'live', False))
        if args.output:
            from .runner import _write
            _write(args.output, result, exclusive=True)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if _successful(result) else 1
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error': str(exc), 'error_type': type(exc).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
