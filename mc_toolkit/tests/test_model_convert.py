import hashlib
import json
import pickle
import shutil
import subprocess
import sys
import types
import zipfile
import numpy as np
import pytest
from physica_mc.models import build_model
from physica_mc.model_convert import (
    LegacyCompatibilityError, _InertLegacyMetadataClass, convert_checkpoint,
    load_bundle, read_bundle_weights, read_manifest, save_bundle,
    validate_checkpoint, _legacy_digest,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_legacy_global(module, name):
    """Expose a temporary fake class so pickle can emit an unavailable global."""
    previous = sys.modules.get(module)
    module_obj = types.ModuleType(module)
    token = type(name, (), {'__module__': module, '__qualname__': name})
    module_obj.__dict__[name] = token
    sys.modules[module] = module_obj
    return token, previous


def restore_legacy_global(module, previous):
    if previous is None:
        sys.modules.pop(module, None)
    else:
        sys.modules[module] = previous


def write_state(path, state_path, complete_path, state):
    state_path.write_bytes(pickle.dumps(state))
    complete = json.loads(complete_path.read_text())
    complete['files'][state_path.relative_to(path).as_posix()] = digest(state_path)
    complete_path.write_text(json.dumps(complete))


def weights(architecture='node'):
    return {k: v.detach().numpy().copy() for k, v in build_model(architecture, {}).state_dict().items()}


@pytest.fixture
def bundle(tmp_path):
    return save_bundle(weights(), {}, 'node', tmp_path / 'bundle', {'pilot_only': True})


def test_bundle_roundtrip_relocation_and_immutable(bundle, tmp_path):
    first = load_bundle(bundle)
    moved = tmp_path / 'moved'
    shutil.copytree(bundle.parent, moved)
    second = load_bundle(moved, expected_sha256=digest(bundle))
    assert first.identity == second.identity and first.config == second.config
    assert first.manifest['provenance'] == {'pilot_only': True}
    for key, value in first.model.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), second.model.state_dict()[key].numpy())
    with pytest.raises(FileExistsError):
        save_bundle(weights(), {}, 'node', bundle.parent, {})
    with pytest.raises(ValueError, match='SHA256'):
        load_bundle(bundle, expected_sha256='0' * 64)


def test_metadata_and_numeric_reads_import_neither_torch_nor_ray(bundle):
    code = '''
import sys
from physica_mc.model_convert import read_manifest, read_bundle_weights
read_manifest(sys.argv[1])
read_bundle_weights(sys.argv[1])
assert not any(k == 'torch' or k.startswith('torch.') or k == 'ray' or k.startswith('ray.') for k in sys.modules)
'''
    subprocess.run([sys.executable, '-c', code, str(bundle)], check=True)


def rewrite_manifest(bundle, change):
    meta = json.loads(bundle.read_text())
    change(meta)
    bundle.write_text(json.dumps(meta))


def test_path_rejection(bundle):
    rewrite_manifest(bundle, lambda meta: meta.update(weights_file='../escape.npz'))
    with pytest.raises(ValueError, match='path'):
        read_manifest(bundle)


def test_corrupt_archive(bundle):
    with (bundle.parent / 'weights.npz').open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match='SHA256'):
        read_bundle_weights(bundle)


@pytest.mark.parametrize('kind', ['object', 'float64', 'complex64', 'nan', 'path', 'duplicate'])
def test_malicious_npz_even_with_updated_archive_hash(bundle, kind):
    archive_path = bundle.parent / 'weights.npz'
    meta, arrays = read_bundle_weights(bundle)
    key = next(iter(arrays))
    if kind == 'path':
        with zipfile.ZipFile(archive_path, 'w') as archive:
            archive.writestr('../escape.npy', b'bad')
    elif kind == 'duplicate':
        with zipfile.ZipFile(archive_path, 'w') as archive:
            archive.writestr(key + '.npy', b'bad')
            archive.writestr(key + '.npy', b'bad')
    else:
        if kind == 'nan':
            arrays[key].flat[0] = np.nan
        else:
            arrays[key] = arrays[key].astype(kind)
        np.savez(archive_path, **arrays)
    rewrite_manifest(bundle, lambda meta: meta.update(weights_sha256=digest(archive_path)))
    with pytest.raises(ValueError):
        read_bundle_weights(bundle)


def test_symlink_bundle_rejected(bundle, tmp_path):
    link = tmp_path / 'linked'
    link.symlink_to(bundle.parent, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        read_bundle_weights(link)


def checkpoint(tmp_path, architecture='node', state=None):
    root = tmp_path / 'phase'
    path = root / 'checkpoints/iter_000001'
    state_path = path / 'algorithm/policies/default_policy/policy_state.pkl'
    state_path.parent.mkdir(parents=True)
    spec = {'architecture': architecture, 'train_batch_size': 8192,
            'env': {'road_length': 1000, 'num_lanes': 4}, 'model': {}}
    identity = {'architecture': architecture, 'seed': 7, 'owner': 'fixture',
                'config_digest': _legacy_digest(spec), 'source_digest': 'raw-source-hash'}
    owner = {'identity': identity, 'scientific': spec, 'pilot_only': True,
             'caveats': ['fixture is not scientific admission']}
    (root / 'run_config.json').write_text(json.dumps(owner))
    raw_weights = weights(architecture)
    if architecture != 'graph':
        raw_weights = {'core.' + key: value for key, value in raw_weights.items()}
    state_path.write_bytes(pickle.dumps({'weights': raw_weights} if state is None else state))
    complete = {'identity': identity, 'snapshot': {'iteration': 1, 'sampled': 8192, 'trained': 8192,
                                                'weights': _legacy_digest(raw_weights)},
                'payload': 'algorithm', 'files': {state_path.relative_to(path).as_posix(): digest(state_path)}}
    (path / 'COMPLETE.json').write_text(json.dumps(complete))
    return path, state_path


@pytest.mark.parametrize('architecture', ['graph', 'node', 'grid'])
def test_classic_declared_numeric_state_conversion(tmp_path, architecture):
    path, _ = checkpoint(tmp_path, architecture)
    manifest = convert_checkpoint(path, tmp_path / 'converted', trusted_legacy=True)
    result = load_bundle(manifest)
    assert result.architecture == architecture
    provenance = result.manifest['provenance']
    assert provenance['source_run_config']['pilot_only'] is True
    assert provenance['source_complete']['identity']['seed'] == 7
    assert provenance['parity_status'] == 'unverified'
    assert provenance['admission_status'] == 'not-established-by-converter'


def test_legacy_requires_trust(tmp_path):
    with pytest.raises(ValueError, match='trusted_legacy'):
        convert_checkpoint(tmp_path / 'absent', tmp_path / 'out')


def test_real_e3_metadata_class_tokens_are_narrowly_mapped(tmp_path):
    path, state_path = checkpoint(tmp_path, 'graph')
    source = pickle.loads(state_path.read_bytes())
    previous = {}
    try:
        policy_class, previous['snfs_traffic.rl.graph_ppo'] = install_legacy_global(
            'snfs_traffic.rl.graph_ppo', 'FactorizedPPOTorchPolicy')
        callbacks_class, previous['train_pv_graph_ppo'] = install_legacy_global(
            'train_pv_graph_ppo', 'PvTrainingMetricsCallbacks')
        state_path.write_bytes(pickle.dumps({
            'weights': source['weights'],
            'policy_spec': {'policy_class': policy_class,
                            'config': {'callbacks': callbacks_class}},
        }))
    finally:
        for module, old in previous.items():
            restore_legacy_global(module, old)
    complete_path = path / 'COMPLETE.json'
    complete = json.loads(complete_path.read_text())
    complete['files'][state_path.relative_to(path).as_posix()] = digest(state_path)
    complete_path.write_text(json.dumps(complete))

    manifest = convert_checkpoint(path, tmp_path / 'converted', trusted_legacy=True)
    meta, weights = read_bundle_weights(manifest)
    assert _legacy_digest(weights) == complete['snapshot']['weights']
    assert meta['provenance']['metadata_class_compatibility'] == [
        {'module': 'snfs_traffic.rl.graph_ppo', 'name': 'FactorizedPPOTorchPolicy',
         'serialized_role': 'PolicySpec.policy_class',
         'handling': 'inert-local-placeholder-not-imported-or-instantiated'},
        {'module': 'train_pv_graph_ppo', 'name': 'PvTrainingMetricsCallbacks',
         'serialized_role': 'PPOConfig.callbacks',
         'handling': 'inert-local-placeholder-not-imported-or-instantiated'},
    ]
    with pytest.raises(LegacyCompatibilityError, match='must not be instantiated'):
        _InertLegacyMetadataClass()


@pytest.mark.parametrize('module,name', [
    ('snfs_traffic.rl.graph_ppo', 'PvGraphTorchModel'),
    ('train_pv_graph_ppo', 'PvTrainingMetricsCallbacksV2'),
])
def test_other_legacy_source_references_remain_hard_failures(tmp_path, module, name):
    path, state_path = checkpoint(tmp_path, 'graph')
    source = pickle.loads(state_path.read_bytes())
    token, previous = install_legacy_global(module, name)
    try:
        write_state(path, state_path, path / 'COMPLETE.json',
                    {'weights': source['weights'],
                     'policy_spec': {'policy_class': token}})
    finally:
        restore_legacy_global(module, previous)
    with pytest.raises(LegacyCompatibilityError, match='unshipped'):
        convert_checkpoint(path, tmp_path / 'out', trusted_legacy=True)


def test_tamper_before_deserialization(tmp_path):
    path, state_path = checkpoint(tmp_path)
    state_path.write_bytes(b'not a pickle')
    with pytest.raises(ValueError, match='tampered'):
        convert_checkpoint(path, tmp_path / 'out', trusted_legacy=True)


def test_inventory_extra_and_traversal(tmp_path):
    path, _ = checkpoint(tmp_path)
    (path / 'extra').write_text('unlisted')
    with pytest.raises(ValueError, match='extra'):
        validate_checkpoint(path)
    (path / 'extra').unlink()
    complete_path = path / 'COMPLETE.json'
    complete = json.loads(complete_path.read_text())
    complete['payload'] = '../escape'
    complete_path.write_text(json.dumps(complete))
    with pytest.raises(ValueError, match='path'):
        validate_checkpoint(path)


def test_unshipped_source_pickle_is_rejected(tmp_path):
    path, state_path = checkpoint(tmp_path)
    state_path.write_bytes(b'csnfs_traffic.rl.graph_ppo\nPvGraphTorchModel\n.')
    complete_path = path / 'COMPLETE.json'
    complete = json.loads(complete_path.read_text())
    complete['files'][state_path.relative_to(path).as_posix()] = digest(state_path)
    complete_path.write_text(json.dumps(complete))
    with pytest.raises(LegacyCompatibilityError, match='unshipped'):
        convert_checkpoint(path, tmp_path / 'out', trusted_legacy=True)


def test_unknown_policy_state(tmp_path):
    path, _ = checkpoint(tmp_path, state={'other': {}})
    with pytest.raises(ValueError, match='Policy-state'):
        convert_checkpoint(path, tmp_path / 'out', trusted_legacy=True)
