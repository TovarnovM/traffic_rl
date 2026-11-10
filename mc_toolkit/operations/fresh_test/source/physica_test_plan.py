"""Read-only frozen inputs and deterministic dispatch order (stdlib only)."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

FREEZE_SHA256 = 'cf3717f4e55cde16ff2f013a9813f95c98caa070fac4094bc9030d3de84ddad4'
PLAN_ID = 'aa65ef46e5ab9594f28844ac6573befd2f2424305334b45b21ec3a86788e548c'
SOURCE_DIGEST = '3ff57f9e84d10390a6a184358332ea12bdf400011f2b6008d6d767030678492e'
IMAGE = 'workpc411:5000/jg-gpu-base@sha256:c5a9f7f9f743e6a5514c87c1bf84324e14de5018be3688c834b2b867d89f5b8d'
TASK_COUNT = 30690
MECHANISM_COUNT = 9930
HEX64 = re.compile(r'[0-9a-f]{64}\Z')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def contained(root, relative):
    root, relative = Path(root).absolute(), Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts,
            'unsafe relative artifact path')
    target = root / relative
    require(target.resolve() == target and target.is_relative_to(root),
            'artifact path contains a symlink or escapes its root')
    return target


def priority(task):
    """FIFO registration key, not a barrier or a change in scientific identity."""
    science = task['science']
    return science['replicate'], -science['config']['density'], task['task_id']


def load_freeze(root, *, verify=True):
    """Verify once for preparation; later read the already admitted immutable inputs."""
    root = Path(root).absolute()
    seal_path = contained(root, 'FREEZE_MANIFEST.json')
    if verify:
        require(sha256(seal_path) == FREEZE_SHA256, 'not the authoritative v2 freeze seal')
    seal = load_json(seal_path)
    if verify:
        for name, ref in seal['artifacts'].items():
            path = contained(root, name)
            require(path.stat().st_size == ref['bytes'] and sha256(path) == ref['sha256'],
                    'sealed artifact changed: ' + name)
    plan = load_json(root / 'future-plan.json')
    requirements = load_json(root / 'execution-requirements.json')
    panels = load_json(root / 'panels.json')
    aliases = load_json(root / 'b1-logical-references.json')
    tasks = plan['tasks']
    by_id = {t['task_id']: t for t in tasks}
    detailed = set(requirements['mechanism_task_ids'])
    require(plan['phase'] == 'fresh_test' and plan['plan_id'] == PLAN_ID,
            'wrong frozen phase or plan identity')
    require(len(tasks) == len(by_id) == TASK_COUNT and len(plan['models']) == 17,
            'wrong frozen physical task/model counts')
    require(len(detailed) == len(requirements['mechanism_task_ids']) == MECHANISM_COUNT
            and detailed <= by_id.keys(), 'wrong detailed physical task set')
    require(requirements['default_observer'] == 'full12'
            and requirements['all_tasks_collect_full12'] is True
            and requirements['source_digest'] == SOURCE_DIGEST,
            'unexpected execution requirements')
    return {'root': root, 'seal': seal, 'plan': plan, 'by_id': by_id,
            'detailed': detailed, 'panels': panels, 'aliases': aliases}


def ordered_tasks(frozen):
    return sorted(frozen['by_id'].values(), key=priority)


def observer_for(frozen, task_id):
    require(task_id in frozen['by_id'], 'unknown task')
    return 'mechanism' if task_id in frozen['detailed'] else 'full12'


def order_digest(frozen):
    h = hashlib.sha256()
    for task in ordered_tasks(frozen):
        h.update(encoded({'task_id': task['task_id'], 'priority': priority(task),
                          'observer': observer_for(frozen, task['task_id'])}) + b'\n')
    return h.hexdigest()


def summary(frozen):
    tasks = ordered_tasks(frozen)
    return {'format': 'physica-test-preparation-v1', 'freeze_sha256': FREEZE_SHA256,
            'plan_id': PLAN_ID, 'source_digest': SOURCE_DIGEST,
            'tasks': len(tasks), 'models': len(frozen['plan']['models']),
            'profiles': dict(Counter(observer_for(frozen, t['task_id']) for t in tasks)),
            'tasks_per_repeat': dict(sorted(Counter(t['science']['replicate'] for t in tasks).items())),
            'dispatch_order': ['environment_repeat_ASC', 'density_DESC', 'task_id_ASC'],
            'completion_barriers': False, 'order_sha256': order_digest(frozen),
            'first': {'task_id': tasks[0]['task_id'], 'priority': priority(tasks[0])},
            'last': {'task_id': tasks[-1]['task_id'], 'priority': priority(tasks[-1])},
            'panels': {k: v['physical_tasks'] for k, v in frozen['panels'].items()},
            'canonical_uncompressed_npz_bytes': 399930729300,
            'launch_authorized': False}


def scientific_tree(root):
    """Metadata-only equivalent of pinned schema.source_identity; never import science."""
    root = Path(root).absolute()
    packages = {}
    for name in ('physica_mc', 'snfs_traffic'):
        stem = contained(root, name)
        require(stem.is_dir(), 'missing scientific package: ' + name)
        files = {}
        for path in sorted(stem.rglob('*')):
            require(not path.is_symlink(), 'scientific package contains symlink')
            if path.is_dir() or '__pycache__' in path.parts or path.suffix in ('.pyc', '.pyo'):
                continue
            if path.is_file():
                files[path.relative_to(stem).as_posix()] = sha256(path)
        require(files, 'empty scientific package')
        packages[name] = {'digest': digest({'name': name, 'files': files}), 'files': files}
    identity = {'schema_version': 1, 'packages': packages}
    identity['source_digest'] = digest(identity)
    require(identity['source_digest'] == SOURCE_DIGEST, 'scientific source differs from3ff freeze')
    return identity


def model_transport(frozen, model_root, *, verify=True):
    """Change transport paths only; manifest/task/model identities remain frozen."""
    root = Path(model_root).absolute()
    models = {}
    for model_id, model in frozen['plan']['models'].items():
        bundle = contained(root, model_id)
        if verify:
            for name in ('manifest.json', 'weights.npz'):
                ref = frozen['seal']['artifacts'][f'models/{model_id}/{name}']
                path = contained(bundle, name)
                require(path.is_file() and path.stat().st_size == ref['bytes']
                        and sha256(path) == ref['sha256'], 'transported selected model changed')
        models[model_id] = {'bundle_path': str(bundle),
                            'manifest_sha256': model['manifest_sha256']}
    return models


def task_items(frozen, settings, binding):
    """Iterator accepted by SlurmMap.submit, with no scheduling or RNG side effects."""
    require(isinstance(binding, str) and HEX64.fullmatch(binding), 'invalid input binding')
    models = model_transport(frozen, settings['model_root'], verify=False)
    for task in ordered_tasks(frozen):
        payload = {name: settings[name] for name in
                   ('science_root', 'deadline_utc', 'max_task_seconds', 'publication_margin_seconds')}
        payload.update(task=task, model=models.get(task['science']['model_id']),
                       observer=observer_for(frozen, task['task_id']), input_binding=binding)
        yield {'task_id': task['task_id'], 'payload': payload}
