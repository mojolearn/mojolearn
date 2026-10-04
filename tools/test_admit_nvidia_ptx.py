"""Admission revalidates real synthetic receipts, not precomputed PASS labels."""
import copy
import json
from types import SimpleNamespace

import pytest

import admit_nvidia_ptx as a
import nvidia_baseline_qualification as q
from test_nvidia_baseline_qualification import add_receipt, campaign  # noqa: F401


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


def with_ampere(evidence):
    campaign = evidence.campaign
    hopper_native = add_receipt(campaign, 2, 'hopper-native', 'native-reference', [9, 0], 'GPU-H100')
    ampere = add_receipt(campaign, 0, 'ampere', 'baseline', [8, 0], 'GPU-A100')
    witness = q.read(evidence.witnesses[0])
    witness['device'].update(uuid='GPU-A100', name='GPU-A100', compute_capability='8.0')
    path = evidence.witnesses[0].parent / 'GPU-A100.json'
    save(path, witness)
    campaign.receipts += [hopper_native, ampere]
    evidence.witnesses.append(path)
    return ampere


def test_native_absent_device_is_admitted_under_its_own_two_scopes(evidence):
    plain, _ = build(evidence)
    assert 'native_absent' not in plain['coverage']
    with_ampere(evidence)
    record, reports = build(evidence)
    absent = record['coverage']['native_absent']
    assert [row['compute_capability'] for row in absent['configurations']] == [[8, 0]]
    assert len(record['configurations']) == 3
    assert len(absent['nvidia']['fixtures']) == 9 and len(absent['shared']['fixtures']) == 3
    assert absent['nvidia']['native_reference_capabilities'] == [[8, 9], [9, 0]]
    assert absent['nvidia']['comparison_sha256'] == a.encoded_sha(reports['nvidia-comparison.json'])
    assert absent['shared']['comparison_sha256'] == a.encoded_sha(reports['native-absent-shared-comparison.json'])
    assert absent['shared']['comparison_sha256'] != record['coverage']['shared']['comparison_sha256']
    assert len(reports['native-absent-shared-comparison.json']) == 1
    api = a.admission_api()
    key = dict(source_commit=record['source_commit'], manifest_sha256=record['manifest_sha256'])
    api.validate_admission(record, configuration=absent['configurations'][0], **key)
    for change in (lambda c: c.update(configurations=[]),
                   lambda c: c.update(configurations=[dict(c['configurations'][0], driver_version='1.2')]),
                   lambda c: c['nvidia'].update(native_reference_capabilities=[[8, 9]]),
                   lambda c: c['nvidia'].update(native_reference_capabilities=[[8, 9], [8, 0]]),
                   lambda c: c['nvidia'].update(fixtures=['base', 'denormal', 'odd']),
                   lambda c: c['shared'].update(vendors=['hip']),
                   lambda c: c.pop('shared')):
        broken = copy.deepcopy(record)
        change(broken['coverage']['native_absent'])
        with pytest.raises(ValueError, match='native-absent'):
            api.validate_admission(broken, **key)


def test_native_absent_device_needs_witness_and_both_comparisons(evidence):
    ampere = with_ampere(evidence)
    missing = evidence.witnesses.pop()
    with pytest.raises(ValueError, match='Missing measured CUDA configuration'):
        build(evidence)
    evidence.witnesses.append(missing)
    # Its nine-fixture column differs from the native references on one NVIDIA-only fixture.
    receipt = q.read(ampere)
    column = q.read(ampere.parent / receipt['column_file'])
    column['cells']['ridge/ties'].update(infer=['f' * 16] * 2, reload=['f' * 16] * 2)
    column_path = ampere.parent / 'ampere.column.json'
    save(column_path, column)
    receipt.update(column_file=column_path.name, column_sha256=q.sha(column_path))
    save(ampere, receipt)
    with pytest.raises(ValueError, match='Bitwise or structural result mismatch'):
        build(evidence)


def test_native_absent_device_cannot_replace_a_native_capability(evidence):
    with_ampere(evidence)
    del evidence.campaign.receipts[1]   # drop the cap 9.0 baseline
    del evidence.witnesses[1]
    with pytest.raises(ValueError, match='distinct GPU'):
        build(evidence)


def test_undeclared_exclusions_are_recorded_in_nvidia_coverage_only(evidence, monkeypatch):
    record, _ = build(evidence)
    assert record['coverage']['nvidia']['undeclared_exclusions'] == []
    monkeypatch.setattr(q, 'UNDECLARED_EXCLUSIONS', (('ridge', 'batch'),))
    for path in evidence.campaign.receipts:
        receipt = q.read(path)
        column_path = path.parent / receipt['column_file']
        column = q.read(column_path)
        for cell in column['cells'].values():
            cell.update(batch=['n/a:UNDECLARED'] * 2, batch_verdict='N/A')
        save(column_path, column)
        receipt['column_sha256'] = q.sha(column_path)
        save(path, receipt)
    record, reports = build(evidence)
    assert record['coverage']['nvidia']['undeclared_exclusions'] == [dict(lane='ridge', part='batch')]
    assert 'undeclared_exclusions' not in record['coverage']['shared']
    assert reports['nvidia-comparison.json']['excluded_parts'] == 9
    api = a.admission_api()
    for bad in ([dict(lane='other', part='batch')], [dict(lane='ridge', part='nope')],
                [dict(lane='ridge', part='batch')] * 2, 'ridge/batch'):
        broken = copy.deepcopy(record)
        broken['coverage']['nvidia']['undeclared_exclusions'] = bad
        with pytest.raises(ValueError, match='undeclared exclusion'):
            api.validate_admission(broken, source_commit=record['source_commit'],
                                   manifest_sha256=record['manifest_sha256'])
    broken = copy.deepcopy(record)
    broken['coverage']['shared']['undeclared_exclusions'] = []
    with pytest.raises(ValueError, match='shared scope'):
        api.validate_admission(broken, source_commit=record['source_commit'],
                               manifest_sha256=record['manifest_sha256'])


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
