"""Bridge complete native local publications to the frozen paper dataset contract.

Runtime operation: verifies NPZ data and packages archives; never runs simulation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from prepare_mc import verify_frozen, sha

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'mc_toolkit/operations/fresh_test'),
                str(ROOT / 'mc_toolkit/operations/fresh_test/source')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import dataset
    import physica_test_plan as science
    import physica_test_store as store
    from physica_mc import planning, schema
    from physica_mc.runner import collect_plan

    require = science.require
    output = store.safe(args.output.absolute())
    require(not output.exists(), 'use a new output directory; retry failures with another directory')
    frozen = science.load_freeze(verify_frozen(), verify=False)
    require(schema.source_identity()['source_digest'] == store.SOURCE, 'scientific source changed')
    inputs, publications, executions = {}, {}, {}
    # Hash exact inputs before and after collection/packaging. Never inspect private receipts.
    pinned = {}

    def pin(path):
        path = store.safe(Path(path).absolute())
        value = {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
        pinned[path] = value
        return value

    for profile in ('full12', 'mechanism'):
        request_path = args.requests.absolute() / (profile + '-run.json')
        collect_path = args.requests.absolute() / (profile + '-collect.json')
        request_ref, collect_ref = pin(request_path), pin(collect_path)
        request, collection = store.read_json(request_path), store.read_json(collect_path)
        plan_path = Path(request['plan'])
        plan_ref = pin(plan_path)
        execution_plan = planning.load_plan(plan_path)
        require(collection == {'plan': request['plan'], 'output_root': request['output_root'],
                               'required_profile': profile}, 'collect/run request mismatch')
        require(request['observer'] == profile and request['threads'] == 1
                and request['device'] == 'cpu' and not request.get('trace', False),
                'wrong local execution profile')
        expected = [task for tid, task in frozen['by_id'].items()
                    if science.observer_for(frozen, tid) == profile]
        ids = {task['task_id'] for task in expected}
        models = execution_plan['models']
        require(models.keys() == frozen['plan']['models'].keys(), 'selected model set changed')
        for mid, model in models.items():
            require({k: v for k, v in model.items() if k != 'bundle_path'} ==
                    {k: v for k, v in frozen['plan']['models'][mid].items() if k != 'bundle_path'},
                    'selected model identity changed')
        rebuilt = planning.make_plan(expected, models=models,
            aliases=[a for a in frozen['plan']['aliases'] if a['task_id'] in ids],
            phase=frozen['plan']['phase'], freeze=frozen['plan']['freeze'])
        require(execution_plan == rebuilt, 'execution plan is not the exact frozen profile partition')
        root = store.safe(Path(request['output_root']))
        ledger_path = root / '.runs' / execution_plan['plan_id'] / 'ledger.json'
        ledger_ref = pin(ledger_path)
        # Native ledgers can exceed the small-envelope JSON limit.
        with ledger_path.open() as stream:
            ledger = json.load(stream)
        settings = {key: request[key] for key in
                    ('budget', 'workers', 'device', 'threads', 'observer', 'spool_root')}
        settings['trace'] = False
        require(ledger['settings'] == settings and ledger['plan'] == execution_plan,
                'native ledger differs from exact execution request')
        bound_executions = {tid: ledger['tasks'][tid]['execution']['execution_id'] for tid in ids}
        del ledger
        report = collect_plan(execution_plan, root, required_profile=profile)
        require(report['status'] == 'complete' and set(report['accepted']) == ids,
                'native collection incomplete, invalid, conflicting, or lifecycle unfinished: ' + profile)
        require(not publications.keys() & ids, 'duplicate physical task across profiles')
        for tid, marker in report['accepted'].items():
            publications[tid] = dict(pin(marker), observer=profile)
            executions[tid] = bound_executions[tid]
        inputs[profile] = {'execution_plan_id': execution_plan['plan_id'],
                           'plan': plan_ref, 'run_request': request_ref,
                           'collect_request': collect_ref, 'ledger': ledger_ref}
        del bound_executions, report
    require(publications.keys() == frozen['by_id'].keys(), 'incomplete original physical union')
    ingestion = {'format': 'physica-local-mc-ingestion-v1',
                 'logical_plan_id': science.PLAN_ID, 'freeze_sha256': science.FREEZE_SHA256,
                 'source_digest': store.SOURCE, 'inputs': inputs,
                 'publications': publications, 'output_dir': str(output)}
    binding = science.digest(ingestion)
    output.mkdir(parents=True, exist_ok=False)
    store.atomic_json(output / 'ingestion.json', ingestion)
    records = []
    for tid, publication in sorted(publications.items()):
        marker_path = Path(publication['path'])
        marker = store.read_json(marker_path)
        require(sha(marker_path) == publication['sha256'], 'accepted marker changed')
        aid, profile = marker['attempt_id'], publication['observer']
        destination = science.contained(output, f'results/{tid}/{aid}')
        destination.mkdir(parents=True, exist_ok=False)
        refs = store.publish_episode(marker_path, destination, input_binding=binding,
            observer=profile, expected_task=frozen['by_id'][tid],
            expected_execution=executions[tid],
            provenance={'kind': 'new-local-native-ingestion', 'input_binding': binding,
                        'logical_plan_id': science.PLAN_ID, 'freeze_sha256': science.FREEZE_SHA256,
                        'execution_plan_id': inputs[profile]['execution_plan_id'],
                        'native_publication': publication})
        records.append({'task_id': tid, 'attempt_id': aid, 'observer': profile,
                        'result_path': (destination / 'result.json').relative_to(output).as_posix(),
                        'result_sha256': refs['result']['sha256'], 'result_bytes': refs['result']['bytes'],
                        'archive_sha256': refs['archive']['sha256'], 'archive_bytes': refs['archive']['bytes']})
    for path, ref in pinned.items():
        require(path.stat().st_size == ref['bytes'] and sha(path) == ref['sha256'],
                'input changed during ingestion: ' + str(path))
    verify_frozen()
    snapshot = {'format': 'physica-test-dataset-v1', 'input_binding': binding,
                'freeze_sha256': science.FREEZE_SHA256, 'plan_id': science.PLAN_ID,
                'frozen_inputs': str(frozen['root']), 'shared_dir': str(output),
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'records': records, 'coverage': dataset.coverage(frozen, publications),
                'analysis_status': 'complete',
                'provenance': {'kind': 'new-local-native-ingestion', 'manifest': 'ingestion.json'},
                'note': 'New native attempts; original logical science identities, not an original-run snapshot.'}
    snapshot['document_sha256'] = science.digest(snapshot)
    # Exclusive atomic index publication is the completion marker for the whole dataset.
    store.atomic_json(output / 'index.json', snapshot)
    print(output / 'index.json')


if __name__ == '__main__':
    main()
