"""Metadata-only staging regressions; no simulation or live queue access."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cli


def fixture_plan(tmp_path, counts=(5, 4)):
    api = cli.dataset_api()
    shared = tmp_path / 'native'
    shared.mkdir()
    by_id, records, aliases = {}, [], []
    for index, count in enumerate(counts):
        method = 'B1' if index == 0 else 'graph'
        for replicate in range(6):
            tid = f'{index:02d}{replicate:062d}'
            science = {'method': method, 'training_line': '' if index == 0 else 'g1',
                       'replicate': replicate, 'seeds': {'environment': replicate},
                       'config': {'density': .2, 'av_fraction': 0 if index == 0 else .95,
                                  'road_length': 1000}}
            by_id[tid] = {'task_id': tid, 'science': science}
            if index == 0:
                aliases.append({'panel': 'main', 'task_id': tid,
                                'requested_fractions': [.9, .95, 1.]})
            if replicate < count:
                result = shared / 'outputs' / tid / 'result.json'
                result.parent.mkdir(parents=True)
                result.write_text(json.dumps({'fixture_only': tid}), encoding='utf-8')
                ref = api.store.file_ref(result)
                records.append({'task_id': tid, 'attempt_id': tid[:32], 'observer': 'full12',
                                'result_path': result.relative_to(shared).as_posix(),
                                'result_sha256': ref['sha256'], 'result_bytes': ref['bytes'],
                                'archive_bytes': 1, 'archive_sha256': '0' * 64})
    frozen = {'by_id': by_id, 'panels': {'main': {'physical_task_ids': list(by_id)}},
              'aliases': aliases}
    snapshot = {'format': 'physica-test-dataset-v1', 'freeze_sha256': api.plan.FREEZE_SHA256,
                'plan_id': api.plan.PLAN_ID, 'input_binding': '1' * 64,
                'created_at_utc': 'fixture-cutoff', 'shared_dir': str(shared),
                'records': records, 'coverage': {'accepted': len(records)},
                'analysis_status': 'partial_exploratory'}
    snapshot['document_sha256'] = api.plan.digest(snapshot)
    index = tmp_path / 'published.json'
    index.write_bytes(api.plan.encoded(snapshot))
    return frozen, snapshot, index


def test_at_least_five_and_aliases_not_extra_repeats(tmp_path):
    frozen, snapshot, _ = fixture_plan(tmp_path)
    selected, deferred = cli.choose_stage_ids(snapshot, frozen, panels=['main'], fractions=[.95])
    assert len(selected) == 5
    assert all(frozen['by_id'][tid]['science']['method'] == 'B1' for tid in selected)
    assert len(deferred) == 1 and deferred[0]['accepted'] == 4
    with pytest.raises(ValueError, match='at least five'):
        cli.choose_stage_ids(snapshot, frozen, min_repeats=4)


def test_no_cross_length_or_duplicate_repeat(tmp_path):
    frozen, snapshot, _ = fixture_plan(tmp_path)
    selected, _ = cli.choose_stage_ids(snapshot, frozen, road_lengths=[2000])
    assert not selected
    first = snapshot['records'][0]['task_id']
    second = snapshot['records'][1]['task_id']
    frozen['by_id'][second]['science']['replicate'] = frozen['by_id'][first]['science']['replicate']
    with pytest.raises(ValueError, match='duplicate environment'):
        cli.choose_stage_ids(snapshot, frozen)


def test_stage_preserves_index_and_original_bytes(tmp_path, monkeypatch):
    frozen, snapshot, index = fixture_plan(tmp_path)
    api = cli.dataset_api()
    monkeypatch.setattr(api.plan, 'load_freeze', lambda *_a, **_k: frozen)
    original = index.read_bytes()
    output = tmp_path / 'stage'
    report = cli.stage_snapshot(index, tmp_path / 'freeze', output, fractions=[.95])
    assert len(report['task_ids']) == 5
    assert (output / 'index.json').read_bytes() == original == index.read_bytes()
    assert not list(output.rglob('episode.tar.gz'))
    for record in snapshot['records']:
        if record['task_id'] in report['task_ids']:
            assert (output / 'results' / record['result_path']).read_bytes() == (
                Path(snapshot['shared_dir']) / record['result_path']).read_bytes()
    second = cli.stage_snapshot(index, tmp_path / 'freeze', output, fractions=[.95])
    assert second['newly_copied_bytes'] == 0


def test_limit_rejects_before_copy_and_protects_inputs(tmp_path, monkeypatch):
    frozen, snapshot, index = fixture_plan(tmp_path)
    api = cli.dataset_api()
    monkeypatch.setattr(api.plan, 'load_freeze', lambda *_a, **_k: frozen)
    output = tmp_path / 'stage'
    with pytest.raises(ValueError, match='limit'):
        cli.stage_snapshot(index, tmp_path / 'freeze', output, max_bytes=1)
    assert not output.exists()
    for root in (Path(snapshot['shared_dir']) / 'bad', tmp_path / 'freeze' / 'bad'):
        with pytest.raises(ValueError, match='outside the live run'):
            cli.stage_snapshot(index, tmp_path / 'freeze', root)
        assert not root.exists()


def test_staging_rejects_changed_snapshot_checksum(tmp_path):
    _, snapshot, index = fixture_plan(tmp_path)
    snapshot['records'].pop()
    index.write_text(json.dumps(snapshot), encoding='utf-8')
    with pytest.raises(ValueError, match='invalid snapshot'):
        cli.stage_snapshot(index, tmp_path / 'freeze', tmp_path / 'out')
