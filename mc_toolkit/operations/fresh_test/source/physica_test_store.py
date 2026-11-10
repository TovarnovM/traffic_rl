"""Lossless transport of canonical episodes; metadata reads use only stdlib."""
import contextlib
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tarfile
import tempfile
import zlib

SOURCE = '3ff57f9e84d10390a6a184358332ea12bdf400011f2b6008d6d767030678492e'
MAX_JSON = 32 * 1024 * 1024
MAX_NPZ = (1 << 30) + 16 * 1024 * 1024
MAX_ARCHIVE = MAX_NPZ + MAX_JSON + 1024 * 1024
NAMES = ('arrays.npz', 'completion.json')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def hex64(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'invalid SHA256')
    return value


def safe(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'absolute untraversed path required')
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink path forbidden')
    return path


def local_root(path):
    path = safe(path)
    require(path.is_dir(), 'local scratch directory missing')
    # Longest matching Linux mount determines the backing filesystem.
    mounts = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        mount = Path(left.split()[4].replace('\\040', ' '))
        if path == mount or mount in path.parents:
            mounts.append((len(mount.parts), right.split()[0]))
    require(mounts and max(mounts)[1] in {'ext4', 'ext3', 'ext2', 'xfs', 'btrfs', 'tmpfs', 'overlay', 'zfs'},
            'scratch must be on a recognized local filesystem')
    return path


def regular(path, limit):
    path = safe(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= limit, 'invalid regular file size/type')
    return path


def file_ref(path, limit=MAX_ARCHIVE):
    path = regular(path, limit)
    h = hashlib.sha256()
    total = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            total += len(block)
            require(total <= limit, 'file grew past limit')
            h.update(block)
    return {'path': path.name, 'sha256': h.hexdigest(), 'bytes': total}


def _pairs(items):
    value = {}
    for key, item in items:
        require(key not in value, 'duplicate JSON key')
        value[key] = item
    return value


def _float(text):
    value = float(text)
    require(math.isfinite(value), 'nonfinite JSON number')
    return value


def _constant(text):
    raise ValueError('invalid JSON constant: ' + text)


def read_json(path):
    path = regular(path, MAX_JSON)
    with path.open('rb') as stream:
        data = stream.read(MAX_JSON + 1)
    require(len(data) <= MAX_JSON, 'JSON too large')
    return json.loads(data, object_pairs_hook=_pairs, parse_float=_float, parse_constant=_constant)


def _sync(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _install(source, destination, *, expected=None):
    """Fsync an owned copy, publish exclusively, then hash the destination once."""
    destination = safe(destination)
    if expected is None:
        expected = file_ref(source)
    else:
        source = regular(source, MAX_ARCHIVE)
        hex64(expected['sha256'])
        require(type(expected['bytes']) is int and 0 < expected['bytes'] <= MAX_ARCHIVE, 'invalid expected size')
    fd, name = tempfile.mkstemp(prefix='.' + destination.name + '-', suffix='.pending', dir=destination.parent)
    temporary = Path(name)
    linked = False
    try:
        with os.fdopen(fd, 'wb') as out, Path(source).open('rb') as inp:
            shutil.copyfileobj(inp, out, 1048576)
            out.flush()
            os.fsync(out.fileno())
        os.link(temporary, destination)
        linked = True
        _sync(destination.parent)
        final = file_ref(destination)
        require((final['sha256'], final['bytes']) == (expected['sha256'], expected['bytes']), 'NFS readback mismatch')
        return final
    except BaseException:
        if linked and destination.exists() and destination.stat().st_ino == temporary.stat().st_ino:
            destination.unlink()
        raise
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    data = encoded(value) + b'\n'
    require(len(data) <= MAX_JSON, 'JSON exceeds limit')
    with tempfile.TemporaryDirectory(prefix='physica-json-', dir=local_root('/tmp')) as directory:
        source = Path(directory) / 'value.json'
        source.write_bytes(data)
        return _install(source, path, expected={'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})


def _header(name, size):
    member = tarfile.TarInfo(name)
    member.size = size
    member.mode = 0o600
    member.mtime = 0
    return member.tobuf(format=tarfile.USTAR_FORMAT)


def _validate(row, expected_task_id, expected_attempt_id, expected_binding, required_profile):
    require(row['format'] == 'physica-test-result-v1', 'wrong result format')
    hex64(row['task_id']); hex64(row['input_binding'])
    require(re.fullmatch('[0-9a-f]{32}', row['attempt_id']), 'invalid attempt ID')
    require(row['observer'] in ('full12', 'mechanism'), 'invalid observer')
    require(row['source_digest'] == SOURCE, 'wrong source')
    for actual, expected in ((row['task_id'], expected_task_id), (row['attempt_id'], expected_attempt_id),
                             (row['input_binding'], expected_binding), (row['observer'], required_profile)):
        require(expected is None or actual == expected, 'result identity/profile mismatch')
    marker = row['canonical_marker']
    task, execution, result = marker['task'], marker['execution'], marker['result']
    require(marker['status'] == 'complete' and marker['attempt_id'] == row['attempt_id']
            and task['task_id'] == row['task_id'], 'canonical identity mismatch')
    require(hashlib.sha256(encoded(task['science'])).hexdigest() == row['task_id'], 'task science digest mismatch')
    require(execution['implementation']['source_digest'] == SOURCE
            and execution['task_id'] == row['task_id'] and execution['profile']['observer'] == row['observer']
            and execution['profile']['threads'] == 1 and execution['profile']['device'] == 'cpu'
            and result['observer_profile'] == row['observer']
            and result['task_id'] == row['task_id'] and result['execution_id'] == execution['execution_id'],
            'canonical execution/profile mismatch')
    config = task['science']['config']
    require(config['measure_steps'] == 2000 and config['warmup_steps'] == 1000 and config['num_lanes'] == 4,
            'wrong frozen horizon/lanes')
    require(set(row['members']) == set(NAMES), 'wrong canonical members')
    for name, descriptor in row['members'].items():
        hex64(descriptor['sha256'])
        require(type(descriptor['bytes']) is int and 0 < descriptor['bytes'] <= (MAX_NPZ if name == NAMES[0] else MAX_JSON), 'member size limit')
    marker_bytes = encoded(marker)
    require(row['members']['completion.json'] == {'sha256': hashlib.sha256(marker_bytes).hexdigest(),
            'bytes': len(marker_bytes)}, 'canonical marker hash/size mismatch')
    sidecar = marker['sidecar']
    require(sidecar['path'] == 'arrays.npz' and sidecar['size'] == row['members']['arrays.npz']['bytes']
            and sidecar['sha256'] == row['members']['arrays.npz']['sha256'], 'sidecar descriptor mismatch')
    archive = row['archive']
    require(archive['path'] == 'episode.tar.gz', 'wrong archive basename')
    hex64(archive['sha256'])
    require(type(archive['bytes']) is int and 0 < archive['bytes'] <= MAX_ARCHIVE, 'archive size limit')


def read_result(path, *, expected_task_id=None, expected_attempt_id=None,
                expected_binding=None, required_profile=None, verify_archive=True):
    """Read scalar envelope; does not establish native acceptance or scientific validity."""
    path = safe(path)
    require(path.name == 'result.json', 'result commit marker required')
    row = read_json(path)
    _validate(row, expected_task_id, expected_attempt_id, expected_binding, required_profile)
    if verify_archive:
        require(file_ref(path.parent / row['archive']['path']) == row['archive'], 'archive hash/size mismatch')
    return row


def _unpack(archive, directory, row):
    """Strict single gzip stream and deterministic two-member tar, no extraction API."""
    tar_path = directory / 'transport.tar'
    sizes = [row['members'][name]['bytes'] for name in NAMES]
    content_size = sum(512 + ((size + 511) // 512) * 512 for size in sizes)
    tar_size = ((content_size + 1024 + 10239) // 10240) * 10240
    decoder = zlib.decompressobj(31)
    total = 0
    with regular(archive, MAX_ARCHIVE).open('rb') as inp, tar_path.open('xb') as out:
        for block in iter(lambda: inp.read(65536), b''):
            while block:
                decoded = decoder.decompress(block, min(1048576, tar_size - total + 1))
                total += len(decoded)
                require(total <= tar_size and not decoder.unused_data, 'gzip trailing data/size overflow')
                out.write(decoded)
                block = decoder.unconsumed_tail
        require(decoder.eof and total == tar_size, 'truncated gzip/tar')
    with tar_path.open('rb') as stream:
        for name, size in zip(NAMES, sizes):
            require(stream.read(512) == _header(name, size), 'unsafe/noncanonical tar header')
            destination = directory / row['task_id'] / row['attempt_id'] / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            remaining = size
            with destination.open('xb') as out:
                while remaining:
                    block = stream.read(min(1048576, remaining))
                    require(block, 'truncated member')
                    out.write(block)
                    remaining -= len(block)
            require(stream.read((-size) % 512) == b'\0' * ((-size) % 512), 'invalid tar padding')
            descriptor = file_ref(destination)
            require({k: descriptor[k] for k in ('sha256', 'bytes')} == row['members'][name], 'canonical member hash mismatch')
        require(stream.read() == b'\0' * (tar_size - content_size), 'extra tar members/trailing data')
    tar_path.unlink()
    marker = destination.parent / 'completion.json'
    require(read_json(marker) == row['canonical_marker'], 'envelope differs from canonical marker')
    return marker


@contextlib.contextmanager
def materialize_episode(path, local_scratch, *, expected_task=None, expected_execution=None, required_profile=None):
    row = read_result(path, expected_task_id=expected_task['task_id'] if expected_task else None,
                      required_profile=required_profile)
    with tempfile.TemporaryDirectory(prefix='physica-read-', dir=local_root(local_scratch)) as temporary:
        marker = _unpack(Path(path).parent / 'episode.tar.gz', Path(temporary), row)
        from physica_mc import schema, storage
        require(schema.source_identity()['source_digest'] == SOURCE, 'Reader source mismatch')
        complete, arrays = storage.read_episode(marker, expected_task=expected_task,
            expected_execution=expected_execution, required_profile=row['observer'])
        yield marker, complete, arrays


def publish_episode(marker_path, output_dir, *, input_binding, observer, provenance,
                    expected_task=None, expected_execution=None):
    marker_path, output_dir = safe(marker_path), safe(output_dir)
    local_root(marker_path.parent)
    require(output_dir.is_dir(), 'native output directory missing')
    require(not any((output_dir / name).exists() for name in ('result.json', 'episode.tar.gz')), 'attempt outputs already exist')
    from physica_mc import schema, storage
    require(schema.source_identity()['source_digest'] == SOURCE, 'Reader source mismatch')
    marker, arrays = storage.read_episode(marker_path, expected_task=expected_task,
        expected_execution=expected_execution, required_profile=observer)
    del arrays
    # The original Reader already verified the immutable NPZ hash and size.
    ref = file_ref(marker_path, MAX_JSON)
    members = {'arrays.npz': {'sha256': marker['sidecar']['sha256'], 'bytes': marker['sidecar']['size']},
               'completion.json': {key: ref[key] for key in ('sha256', 'bytes')}}
    installed = None
    with tempfile.TemporaryDirectory(prefix='physica-pack-', dir=local_root(marker_path.parent)) as temporary:
        archive = Path(temporary) / 'episode.tar.gz'
        with archive.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0, compresslevel=1) as compressed:
            with tarfile.open(fileobj=compressed, mode='w', format=tarfile.USTAR_FORMAT) as tar:
                for name in NAMES:
                    info = tarfile.TarInfo.frombuf(_header(name, members[name]['bytes']), 'utf-8', 'strict')
                    with (marker_path.parent / name).open('rb') as source:
                        tar.addfile(info, source)
        row = {'format': 'physica-test-result-v1', 'task_id': marker['task']['task_id'],
               'attempt_id': marker['attempt_id'], 'input_binding': input_binding, 'observer': observer,
               'source_digest': SOURCE, 'canonical_marker': marker, 'archive': file_ref(archive),
               'members': members, 'provenance': provenance}
        _validate(row, None, None, None, observer)
        try:
            installed = _install(archive, output_dir / 'episode.tar.gz', expected=row['archive'])
            result = atomic_json(output_dir / 'result.json', row)
        except BaseException:
            if installed and not (output_dir / 'result.json').exists():
                (output_dir / 'episode.tar.gz').unlink()
                _sync(output_dir)
            raise
    return {'result': result, 'archive': installed}
