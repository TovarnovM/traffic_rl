"""Prepare local native MC plans/requests from unchanged frozen scientific inputs.

This command reads metadata and hashes files; it never runs an episode.
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')


def verify_frozen():
    frozen = HERE / 'configs/frozen'
    refs = json.loads((HERE / 'configs/frozen-files.json').read_text())
    for name, ref in refs.items():
        path = frozen / name
        if path.stat().st_size != ref['bytes'] or sha(path) != ref['sha256']:
            raise ValueError('changed frozen input: ' + name)
    return frozen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--spool', type=Path, required=True)
    p.add_argument('--deadline-utc', required=True)
    p.add_argument('--workers', type=int, required=True)
    p.add_argument('--max-attempts', type=int, required=True,
                   help='separate attempt cap for each of the two profile plans')
    args = p.parse_args()
    if args.workers < 1 or args.max_attempts < 1:
        raise ValueError('positive resource limits required')
    from datetime import datetime
    deadline = datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00'))
    if deadline.utcoffset() is None or deadline.utcoffset().total_seconds() != 0:
        raise ValueError('deadline must be timezone-aware UTC')
    from physica_mc.planning import load_plan, make_plan, write_plan
    from physica_mc.schema import source_identity
    if source_identity()["source_digest"] != "3ff57f9e84d10390a6a184358332ea12bdf400011f2b6008d6d767030678492e":
        raise ValueError("MC package tree differs from the frozen scientific source")
    frozen = verify_frozen()
    original = load_plan(frozen / 'future-plan.json')
    seal = json.loads((frozen / 'FREEZE_MANIFEST.json').read_text())
    requirements = json.loads((frozen / 'execution-requirements.json').read_text())
    # No dependency on a release catalog layout: match native manifest bytes.
    manifests = {}
    for path in sorted(args.weights.resolve().rglob('manifest.json')):
        manifests.setdefault(sha(path), []).append(path.parent)
    models = {}
    for model_id, entry in original['models'].items():
        candidates = manifests.get(entry['manifest_sha256'], [])
        if not candidates:
            raise ValueError('missing selected model: ' + model_id)
        bundle = candidates[0]
        for name in ('manifest.json', 'weights.npz'):
            ref = seal['artifacts'][f'models/{model_id}/{name}']
            path = bundle / name
            if path.stat().st_size != ref['bytes'] or sha(path) != ref['sha256']:
                raise ValueError('selected bundle changed: ' + model_id)
        models[model_id] = dict(entry, bundle_path=str(bundle))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    detailed = set(requirements['mechanism_task_ids'])
    receipts = {}
    for observer in ('full12', 'mechanism'):
        tasks = [t for t in original['tasks'] if (t['task_id'] in detailed) == (observer == 'mechanism')]
        ids = {t['task_id'] for t in tasks}
        aliases = [a for a in original['aliases'] if a['task_id'] in ids]
        plan = make_plan(tasks, models=models, aliases=aliases,
                         phase=original['phase'], freeze=original['freeze'])
        destination = output / (observer + '-plan.json')
        write_plan(plan, destination)
        request = {'plan': str(destination),
                   'output_root': str(args.results.resolve() / observer),
                   'spool_root': str(args.spool.resolve() / observer),
                   'workers': args.workers, 'threads': 1, 'device': 'cpu', 'observer': observer,
                   'budget': {'deadline_utc': args.deadline_utc, 'max_attempts': args.max_attempts,
                              'max_workers': args.workers, 'cpu_limit': args.workers, 'max_gpu_workers': 0}}
        write(output / (observer + '-run.json'), request)
        write(output / (observer + '-collect.json'),
              {k: request[k] for k in ('plan', 'output_root')} | {'required_profile': observer})
        receipts[observer] = {'plan_id': plan['plan_id'], 'tasks': len(tasks)}
    write(output / 'transport.json', {'original_plan_id': original['plan_id'],
          'local_plans': receipts, 'models': models,
          'note': 'Only transport paths and execution partition changed; task/freeze identities unchanged.'})


if __name__ == '__main__':
    main()
