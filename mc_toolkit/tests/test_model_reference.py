"""Compare shipped numeric bundles with the immutable original-policy oracle.

The oracle artifacts are development data under .local/model_reference and are
not part of a clean install. This test skips when they are absent and never
imports the original Physica RLlib source or a historical checkout.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from physica_mc.model_convert import load_bundle, read_bundle_weights
from physica_mc.models import infer_logits

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / '.local/model_reference'
GOLDEN = REFERENCE / 'golden'
BUNDLES = ROOT / '.local/models_compat_v1'
RECEIPT = REFERENCE / 'parity_receipt.json'
CASES = {'graph_s1': 'graph', 'node_s1': 'node', 'grid_s1': 'grid'}
OBSERVATION_KEYS = (
    'node_features', 'node_mask', 'neighbor_index', 'neighbor_mask',
    'edge_features', 'action_mask', 'global_features',
)

pytestmark = pytest.mark.skipif(
    not (RECEIPT.is_file() and GOLDEN.is_dir() and all(
        (BUNDLES / f'{case}_bundle/manifest.json').is_file() for case in CASES)),
    reason='local original-policy oracle artifacts and numeric bundles are unavailable',
)


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_converted_bundles_match_original_policy_logits_and_actions():
    receipt = json.loads(RECEIPT.read_text())
    assert receipt['status'] == 'pass'
    assert receipt['fixture_backend'] == 'reference'
    tolerance = receipt['tolerance']
    fixtures = json.loads((GOLDEN / 'fixtures.json').read_text())
    observation_ids = [row['observation_id'] for row in fixtures]
    assert len(observation_ids) == 16
    observation_archive = np.load(GOLDEN / 'observations.npz', allow_pickle=False)
    observations = {
        observation_id: {
            key: observation_archive[f'{observation_id}__{key}']
            for key in OBSERVATION_KEYS
        }
        for observation_id in observation_ids
    }

    for case, architecture in CASES.items():
        entry = next(row for row in receipt['architectures'] if row['case'] == case)
        assert entry['status'] == 'pass'
        assert entry['architecture'] == architecture
        assert entry['weights_before_after_equal'] is True
        assert entry['ray_initialized_during_oracle'] is False
        assert entry['logits']['bitwise_exact'] == len(observation_ids)
        assert entry['logits']['max_abs_diff'] == 0.0
        assert entry['actions']['exact'] == len(observation_ids)
        assert entry['actions']['all_legal_under_mask'] is True

        bundle_dir = BUNDLES / f'{case}_bundle'
        manifest_path = bundle_dir / 'manifest.json'
        manifest_sha = file_sha256(manifest_path)
        assert manifest_sha == entry['manifest_sha256']
        meta, _weights = read_bundle_weights(bundle_dir, expected_sha256=manifest_sha)
        assert meta['provenance']['parity_status'] == 'unverified'
        assert meta['provenance']['admission_status'] == 'not-established-by-converter'
        assert meta['provenance']['source_policy_state_sha256'] == entry['source_policy_state_sha256']

        golden_path = GOLDEN / f'{case}.npz'
        assert file_sha256(golden_path) == entry['golden_sha256']
        golden = np.load(golden_path, allow_pickle=False)
        bundle = load_bundle(bundle_dir, expected_sha256=manifest_sha)
        for observation_id in observation_ids:
            observation = observations[observation_id]
            got_logits = infer_logits(bundle.model, observation)
            expected_logits = golden[f'{observation_id}__logits']
            expected_action = golden[f'{observation_id}__actions']
            np.testing.assert_allclose(
                got_logits, expected_logits,
                atol=tolerance['atol'], rtol=tolerance['rtol'])
            got_action = np.argmax(got_logits, axis=-1).astype(np.int64)
            np.testing.assert_array_equal(got_action, expected_action)
            mask = np.asarray(observation['action_mask'], dtype=bool)
            assert mask[np.arange(mask.shape[0]), got_action].all()

    assert 'snfs_traffic.rl.graph_ppo' not in sys.modules
    assert 'snfs_traffic.rl.nongraph_ppo' not in sys.modules
    assert 'train_pv_graph_ppo' not in sys.modules
