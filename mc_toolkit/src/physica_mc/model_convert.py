"""Versioned numeric bundles and explicit trusted classic RLlib conversion.

Metadata and numeric readers need only NumPy; torch is imported by model building.
Legacy pickle conversion is an explicitly trusted offline operation, not a sandbox.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import zipfile

import numpy as np
from .models import build_model, normalize_config, validate_weights

FORMAT = 'physica-mc-model-v1'
MAX_JSON = 4 * 1024 * 1024
MAX_WEIGHTS = 256 * 1024 * 1024
MAX_CHECKPOINT = 2 * 1024 * 1024 * 1024


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _safe_path(path):
    path = Path(os.path.abspath(path))
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('symlink paths are forbidden')
    return path


def _child(root, relative):
    if not isinstance(relative, str) or '\\' in relative:
        raise ValueError('invalid relative path')
    parts = PurePosixPath(relative)
    if parts.is_absolute() or not parts.parts or any(p in ('.', '..') for p in relative.split('/')):
        raise ValueError('unsafe relative path')
    return _safe_path(root.joinpath(*parts.parts))


def _read(path, limit):
    path = _safe_path(path)
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f'missing or oversized file: {path.name}')
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('size limit exceeded')
    return data


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'invalid JSON constant: {value}')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def _descriptors(weights):
    return {key: {'shape': list(value.shape), 'dtype': 'float32',
                  'sha256': _digest(value.tobytes(order='C'))}
            for key, value in sorted(weights.items())}


def _identity(metadata):
    return _digest(_canonical({key: metadata[key] for key in
                              ('format', 'architecture', 'config', 'provenance', 'arrays')}))


def read_manifest(path, *, expected_sha256=None):
    """Read/validate JSON only. expected_sha256 pins the manifest file bytes."""
    path = _safe_path(path)
    if path.is_dir():
        path = path / 'manifest.json'
    raw = _read(path, MAX_JSON)
    if expected_sha256 is not None and _digest(raw) != expected_sha256:
        raise ValueError('manifest SHA256 mismatch')
    meta = _json(raw)
    required = {'format', 'architecture', 'config', 'provenance', 'arrays', 'identity',
                'weights_file', 'weights_sha256', 'config_sha256', 'provenance_sha256'}
    if not isinstance(meta, dict) or set(meta) != required or meta['format'] != FORMAT:
        raise ValueError('unsupported model manifest')
    if meta['weights_file'] != 'weights.npz':
        raise ValueError('unsupported weights path')
    if normalize_config(meta['architecture'], meta['config']) != meta['config']:
        raise ValueError('noncanonical configuration')
    if not isinstance(meta['provenance'], dict):
        raise ValueError('invalid provenance')
    for key in ('config', 'provenance'):
        if _digest(_canonical(meta[key])) != meta[key + '_sha256']:
            raise ValueError(f'{key} hash mismatch')
    if _identity(meta) != meta['identity']:
        raise ValueError('bundle identity mismatch')
    return meta


def read_bundle_weights(path, *, expected_sha256=None):
    """Validate the entire numeric bundle without importing torch or Ray."""
    path = _safe_path(path)
    manifest_path = path / 'manifest.json' if path.is_dir() else path
    meta = read_manifest(manifest_path, expected_sha256=expected_sha256)
    raw = _read(_child(manifest_path.parent, meta['weights_file']), MAX_WEIGHTS)
    if _digest(raw) != meta['weights_sha256']:
        raise ValueError('weights archive SHA256 mismatch')
    descriptors = meta['arrays']
    if not isinstance(descriptors, dict) or not 1 <= len(descriptors) <= 4096:
        raise ValueError('invalid array inventory')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if len(members) != len(descriptors) or {m.filename for m in members} != {k + '.npy' for k in descriptors}:
            raise ValueError('archive member mismatch')
        if sum(m.file_size for m in members) > MAX_WEIGHTS:
            raise ValueError('expanded archive exceeds size limit')
        for member in members:
            if member.flag_bits & 1 or member.compress_type != zipfile.ZIP_STORED:
                raise ValueError('only uncompressed, unencrypted NPZ is supported')
            if '/' in member.filename or '\\' in member.filename:
                raise ValueError('unsafe archive member path')
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version != (1, 0):
                    raise ValueError('unsupported NPY version')
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=10000)
                count = 1
                for dim in shape:
                    if type(dim) is not int or dim < 0:
                        raise ValueError('invalid array shape')
                    count *= dim
                if dtype != np.dtype('float32') or len(shape) > 4 or fortran or count * 4 > MAX_WEIGHTS:
                    raise ValueError('invalid numeric array header')
                if stream.tell() + count * 4 != member.file_size:
                    raise ValueError('array byte length mismatch')
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        weights = validate_weights({key: archive[key] for key in archive.files})
    if _descriptors(weights) != descriptors:
        raise ValueError('array hashes/dtypes/shapes mismatch')
    return meta, weights


@dataclass(frozen=True)
class ModelBundle:
    model: object
    config: dict
    architecture: str
    identity: str
    manifest: dict


def load_bundle(path, *, device='cpu', expected_sha256=None):
    meta, weights = read_bundle_weights(path, expected_sha256=expected_sha256)
    model = build_model(meta['architecture'], meta['config'], weights, device=device)
    return ModelBundle(model, meta['config'], meta['architecture'], meta['identity'], meta)


def save_bundle(weights, config, architecture, destination, provenance):
    """Create a fresh bundle directory; publish manifest last, never overwrite."""
    weights = validate_weights(weights)
    config = normalize_config(architecture, config)
    if not isinstance(provenance, Mapping):
        raise ValueError('provenance must be a JSON mapping')
    provenance = _json(_canonical(dict(provenance)))
    # Validate actual architecture keys and dimensions before publishing anything.
    build_model(architecture, config, weights, device='cpu')
    metadata = dict(format=FORMAT, architecture=architecture, config=config,
                    provenance=provenance, arrays=_descriptors(weights))
    metadata['identity'] = _identity(metadata)
    metadata.update(config_sha256=_digest(_canonical(config)),
                    provenance_sha256=_digest(_canonical(provenance)), weights_file='weights.npz')
    destination = _safe_path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    weights_path = destination / 'weights.npz'
    with weights_path.open('xb') as stream:
        np.savez(stream, **dict(sorted(weights.items())))
        stream.flush()
        os.fsync(stream.fileno())
    metadata['weights_sha256'] = _hash_file(weights_path)
    raw = _canonical(metadata)
    if len(raw) > MAX_JSON:
        raise ValueError('manifest exceeds size limit')
    # Link a complete private marker into place: readers never see partial JSON.
    temporary = destination / '.manifest.tmp'
    with temporary.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    manifest = destination / 'manifest.json'
    os.link(temporary, manifest)
    temporary.unlink()
    descriptor = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return manifest


class LegacyCompatibilityError(ValueError):
    """The supplied pickle requires source compatibility not shipped here."""


# The E3 classic PolicySpec serializes these class objects as metadata. They
# are not training code needed to read weights. Map only these exact tokens to
# one inert local stand-in: no legacy module is imported and the class is never
# instantiated or called. Any other reference below remains a hard failure.
_LEGACY_METADATA_CLASS_TOKENS = {
    ('snfs_traffic.rl.graph_ppo', 'FactorizedPPOTorchPolicy'): 'PolicySpec.policy_class',
    ('train_pv_graph_ppo', 'PvTrainingMetricsCallbacks'): 'PPOConfig.callbacks',
}
_LEGACY_SOURCE_MODULES = ('snfs_traffic', 'train_pv_graph_ppo', 'research', 'physica_a_revision_r1')


class _InertLegacyMetadataClass:
    """Placeholder for an unused serialized metadata class; never executable."""

    def __new__(cls, *args, **kwargs):
        raise LegacyCompatibilityError('legacy metadata class token must not be instantiated')


class _LegacyUnpickler(pickle.Unpickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapped_metadata_classes = []

    def find_class(self, module, name):
        token = (module, name)
        if token in _LEGACY_METADATA_CLASS_TOKENS:
            self.mapped_metadata_classes.append(token)
            return _InertLegacyMetadataClass
        # Trusted pickle still executes code. This guard enforces the supported
        # deployment boundary, and is explicitly NOT a security sandbox.
        if any(module == prefix or module.startswith(prefix + '.')
               for prefix in _LEGACY_SOURCE_MODULES):
            raise LegacyCompatibilityError(f'unshipped legacy source reference: {module}.{name}')
        return super().find_class(module, name)


def _legacy_digest(value):
    """E3 typed length-framed identity, retained exactly for numeric state/config."""
    digest = hashlib.sha256()

    def emit(tag, data=b''):
        digest.update(tag.encode() + b':' + str(len(data)).encode() + b':' + data)

    def visit(item):
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise ValueError('object arrays unsupported')
            emit('array', str(item.dtype).encode())
            visit(tuple(item.shape))
            emit('bytes', item.tobytes(order='C'))
        elif isinstance(item, dict):
            emit('dict')
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            emit(type(item).__name__, str(len(item)).encode())
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (str, bool, int, float, bytes)):
            emit(type(item).__name__, item if isinstance(item, bytes) else repr(item).encode())
        elif isinstance(item, np.generic):
            visit(item.item())
        else:
            raise ValueError(f'unsupported legacy digest type: {type(item).__name__}')
    visit(value)
    return digest.hexdigest()


def validate_checkpoint(path, *, run_config_path=None):
    """Validate the E3 publication envelope, inventory and owner before pickle."""
    path = _safe_path(path)
    if path.parent.name != 'checkpoints' or not path.is_dir():
        raise ValueError('expected an explicitly staged checkpoints/iter_NNNNNN directory')
    owner_path = _safe_path(run_config_path) if run_config_path is not None else path.parent.parent / 'run_config.json'
    owner_raw = _read(owner_path, MAX_JSON)
    complete_raw = _read(path / 'COMPLETE.json', MAX_JSON)
    owner, complete = _json(owner_raw), _json(complete_raw)
    if set(complete) != {'identity', 'snapshot', 'payload', 'files'}:
        raise ValueError('unsupported COMPLETE format')
    if complete['identity'] != owner['identity']:
        raise ValueError('checkpoint owner identity mismatch')
    snapshot = complete['snapshot']
    iteration = snapshot['iteration']
    if type(iteration) is not int or iteration < 1 or path.name != f'iter_{iteration:06d}':
        raise ValueError('unpublished checkpoint iteration')
    spec = owner['scientific']
    if _legacy_digest(spec) != complete['identity']['config_digest']:
        raise ValueError('source scientific config digest mismatch')
    if (snapshot['sampled'] != iteration * spec['train_batch_size'] or
            snapshot['trained'] != snapshot['sampled']):
        raise ValueError('checkpoint counters mismatch')
    files = complete['files']
    if not isinstance(files, dict) or not 1 <= len(files) <= 4096:
        raise ValueError('invalid checkpoint inventory')
    for relative in files:
        _child(path, relative)
    actual = {}
    total = 0
    for child in path.rglob('*'):
        _safe_path(child)
        if child.is_dir():
            continue
        if not child.is_file():
            raise ValueError('unsupported checkpoint file type')
        if child == path / 'COMPLETE.json':
            continue
        total += child.stat().st_size
        if total > MAX_CHECKPOINT or len(actual) >= 4096:
            raise ValueError('checkpoint exceeds size limit')
        actual[child.relative_to(path).as_posix()] = _hash_file(child)
    if actual != files:
        raise ValueError('missing, extra or tampered checkpoint files')
    payload = _child(path, complete['payload'])
    state_path = _child(payload, 'policies/default_policy/policy_state.pkl')
    if state_path.relative_to(path).as_posix() not in files:
        raise ValueError('missing classic default policy state')
    return owner, complete, state_path, _digest(owner_raw), _digest(complete_raw)


def convert_checkpoint(path, destination, *, trusted_legacy=False, run_config_path=None):
    """Convert declared classic Policy-state weights; never restore an Algorithm.

    Only use on trusted, immutable staged inputs. Missing legacy class imports
    are a hard compatibility failure; no training source imports or guessed weights.
    """
    if trusted_legacy is not True:
        raise ValueError('legacy pickle requires explicit trusted_legacy=True')
    owner, complete, state_path, owner_sha, complete_sha = validate_checkpoint(path, run_config_path=run_config_path)
    raw = _read(state_path, MAX_CHECKPOINT)
    relative = state_path.relative_to(_safe_path(path)).as_posix()
    if _digest(raw) != complete['files'][relative]:
        raise ValueError('policy state changed after validation')
    try:
        unpickler = _LegacyUnpickler(io.BytesIO(raw))
        state = unpickler.load()
    except (ImportError, AttributeError) as exc:
        raise LegacyCompatibilityError(f'legacy dependency unavailable: {exc}') from exc
    if not isinstance(state, dict) or not isinstance(state.get('weights'), dict):
        raise ValueError('unsupported classic Policy-state: expected numeric weights dict')
    architecture = complete['identity']['architecture']
    spec = owner['scientific']
    if spec['architecture'] != architecture:
        raise ValueError('scientific architecture mismatch')
    config = dict(spec['env'])
    config.update(spec['model'])
    weights = state['weights']
    if _legacy_digest(weights) != complete['snapshot']['weights']:
        raise ValueError('source snapshot weight digest mismatch')
    if architecture in ('node', 'grid'):
        if not weights or not all(isinstance(k, str) and k.startswith('core.') for k in weights):
            raise ValueError('non-graph Policy weights must have exact core. prefix')
        weights = {key[5:]: value for key, value in weights.items()}
    provenance = dict(source_format='classic-rllib-policy-state-e3-envelope',
                      source_complete=complete, source_run_config=owner,
                      source_complete_sha256=complete_sha, source_run_config_sha256=owner_sha,
                      source_policy_state_sha256=_digest(raw),
                      conversion_validation='strict-numeric-weights-only',
                      parity_status='unverified', admission_status='not-established-by-converter')
    if unpickler.mapped_metadata_classes:
        provenance['metadata_class_compatibility'] = [
            dict(module=module, name=name,
                 serialized_role=_LEGACY_METADATA_CLASS_TOKENS[(module, name)],
                 handling='inert-local-placeholder-not-imported-or-instantiated')
            for module, name in sorted(set(unpickler.mapped_metadata_classes))
        ]
    return save_bundle(weights, config, architecture, destination, provenance)
