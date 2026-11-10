"""Release-relative paths and integrity of the included scientific payloads."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / 'results/physica_a_v1.1/inputs'
RENDER = ROOT / 'mc_toolkit/operations/paper_outputs'
INVENTORY_SHA256 = 'a0a65baa05b15366fa2b97953b0e329611df8d81de1c984ee2e1055f61f04226'


def verify_inputs(source):
    source = Path(source)
    payload = (source / 'inventory.json').read_bytes()
    if hashlib.sha256(payload).hexdigest() != INVENTORY_SHA256:
        raise ValueError('publication input inventory checksum mismatch')
    for name, entry in json.loads(payload)['files'].items():
        path = source / name
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError(f'input outside release: {name}')
        data = path.read_bytes()
        if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError(f'publication input checksum mismatch: {name}')
        if 'payload_sha256' in entry:
            raw = gzip.decompress(data)
            if hashlib.sha256(raw).hexdigest() != entry['payload_sha256']:
                raise ValueError(f'decompressed payload checksum mismatch: {name}')
