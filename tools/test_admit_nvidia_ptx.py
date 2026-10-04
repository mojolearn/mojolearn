"""Admission revalidates real synthetic receipts, not precomputed PASS labels."""
import copy
import json
from types import SimpleNamespace

import pytest

import admit_nvidia_ptx as a
import nvidia_baseline_qualification as q
from test_nvidia_baseline_qualification import campaign  # noqa: F401


def save(path, doc):
    path.write_text(json.dumps(doc))


@pytest.fixture
def evidence(campaign, tmp_path, monkeypatch):
    api = a.admission_api()
    h = campaign.h
    h.FIXTURES = list(api.NVIDIA_FIXTURES)
    h._h = lambda x: f'{x:016x}'
    monkeypatch.setattr(q, 'inventory', lambda _: dict(lanes=['ridge'], fixtures=h.FIXTURES, parts=list(api.NVIDIA_PARTS),
                                                     excluded={}, applicability_gaps=[]))
    train, heldout = q.fixture_witnesses(h, h.FIXTURES)
    for path in campaign.receipts:
        receipt = q.read(path)
        column_path = path.parent / receipt['column_file']
        column = q.read(column_path)
        cell = column['cells']['ridge/base']
        cell.update(parts=[{'predict': 'd' * 16}] * 2, reload=['d' * 16] * 2)
        column.update(fixtures=train, heldout=heldout, heldout_seed=1,
                      lane_revisions={}, batch_revisions={},
                      cells={'ridge/' + f: copy.deepcopy(cell) for f in h.FIXTURES})
        save(column_path, column)
        receipt.update(fixtures=h.FIXTURES, column_sha256=q.sha(column_path), driver_version='580.159.04')
        save(path, receipt)
    references, reference_hashes = {}, {}
    for vendor in ('apple', 'amd'):
        doc = q.read(campaign.receipts[0].with_suffix('.column.json'))
        doc.update(vendor=vendor, repeats=1)
        for name in ('fixtures', 'heldout'):
            doc[name] = {f: doc[name][f] for f in api.SHARED_FIXTURES}
        doc['cells'] = {k: v for k, v in doc['cells'].items() if k.split('/')[1] in api.SHARED_FIXTURES}
        for cell in doc['cells'].values():
            for key, value in cell.items():
                if isinstance(value, list):
                    cell[key] = value[:1]
        path = tmp_path / (vendor + '.json')
        save(path, doc)
        references[vendor], reference_hashes[vendor] = path, q.sha(path)
    script = tmp_path / 'configuration.py'
    script.write_text('# synthetic configuration collector\n')
    witnesses = []
    for path in campaign.receipts[:2]:
        receipt = q.read(path)
        hw = receipt['hardware']
        doc = dict(schema='mojolearn.cuda-runtime-config-witness.v1',
                   source_commit=receipt['source_commit'], script_sha256=q.sha(script),
                   source_commit_file_sha256=receipt['installed_source']['core_commit_sha256'],
                   timestamp_utc='2026-10-04T00:00:00+00:00', cuDriverGetVersion_return=0,
                   cuDriverGetVersion=13000,
                   device=dict(uuid=hw['uuid'], name=hw['name'],
                               compute_capability='.'.join(map(str, hw['compute_capability'])),
                               driver_version=receipt['driver_version']))
        witness = tmp_path / (hw['uuid'] + '.json')
        save(witness, doc)
        witnesses.append(witness)
    return SimpleNamespace(campaign=campaign, references=references, hashes=reference_hashes,
                           witnesses=witnesses, script=script)


def build(e):
    return a.build(e.campaign.manifest, e.campaign.receipts, e.references, e.hashes, e.witnesses, e.script)


def test_complete_evidence_keeps_shared_and_nvidia_scopes_distinct(evidence):
    record, reports = build(evidence)
    assert record['qualified'] and len(record['configurations']) == 2
    assert len(record['coverage']['shared']['fixtures']) == 3
    assert len(record['coverage']['nvidia']['fixtures']) == 9
    assert record['coverage']['shared']['comparison_sha256'] == a.encoded_sha(reports['shared-comparison.json'])
    assert record['coverage']['nvidia']['comparison_sha256'] == a.encoded_sha(reports['nvidia-comparison.json'])
    assert not q.read(evidence.campaign.manifest)['identical_qualified']


@pytest.mark.parametrize('change', [
    lambda d: d.update(cuDriverGetVersion_return=1),
    lambda d: d.update(cuDriverGetVersion=0),
    lambda d: d.update(script_sha256='0' * 64),
    lambda d: d.update(source_commit='0' * 40),
    lambda d: d.update(source_commit_file_sha256='0' * 64),
    lambda d: d['device'].update(driver_version='581.1'),
    lambda d: d['device'].update(uuid='GPU-other'),
    lambda d: d['device'].update(compute_capability='9.9'),
])
def test_configuration_is_bound_to_observed_run(evidence, change):
    path = evidence.witnesses[0]
    doc = q.read(path)
    change(doc)
    save(path, doc)
    with pytest.raises(ValueError):
        build(evidence)


def test_rehashed_vendor_mismatch_still_refuses_admission(evidence):
    path = evidence.references['amd']
    doc = q.read(path)
    doc['cells']['ridge/base']['batchgrad'] = ['f' * 16]
    save(path, doc)
    evidence.hashes['amd'] = q.sha(path)
    with pytest.raises(ValueError, match='does not match'):
        build(evidence)


def test_missing_full_fixture_cannot_be_admitted_as_prototype(evidence):
    path = evidence.campaign.receipts[0]
    receipt = q.read(path)
    receipt['fixtures'] = ['base']
    save(path, receipt)
    with pytest.raises(ValueError):
        build(evidence)


def test_duplicate_configuration_witness_refuses(evidence):
    evidence.witnesses.append(evidence.witnesses[0])
    with pytest.raises(ValueError, match='duplicate witness'):
        build(evidence)


@pytest.mark.parametrize('field', ['parts', 'reload'])
def test_nvidia_only_fixture_components_cannot_escape_admission(evidence, field):
    path = evidence.campaign.receipts[0]
    receipt = q.read(path)
    column_path = path.parent / receipt['column_file']
    column = q.read(column_path)
    column['cells']['ridge/ties'][field] = ([{'predict': 'f' * 16}] * 2 if field == 'parts' else ['f' * 16] * 2)
    save(column_path, column)
    receipt['column_sha256'] = q.sha(column_path)
    save(path, receipt)
    with pytest.raises(ValueError):
        build(evidence)
