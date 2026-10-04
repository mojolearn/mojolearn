"""Portable-path evidence must fail closed on missing/tampered provenance."""
import copy
import json
import sys
import types
from types import SimpleNamespace

import pytest

import identity_columns
import nvidia_baseline_qualification as q


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    h = SimpleNamespace(LANES={'ridge': None}, FIXTURES=['base'], LANE_REVISIONS={},
                        BATCH_REVISIONS={}, RLPAIR={}, PROPERTY_PARTS=('batchgrad', 'batchscale', 'ragged', 'stepfull'),
                        NARROWABLE_PARTS=('batch', 'rlpair', 'batchgrad', 'batchscale', 'ragged', 'stepfull'),
                        BATCH_ALONE=16, BATCH_SPLIT=(1, 8), fixture=lambda f: (1, 2, 3),
                        heldout=lambda f: 4, _h=lambda x: str(x),
                        _part_protocol=lambda part, alone: {'part': part, 'alone': alone})
    monkeypatch.setattr(q, 'modules', lambda: (h, identity_columns.vref()))
    monkeypatch.setattr(q, 'inventory', lambda _: dict(lanes=['ridge'], fixtures=['base'], excluded={}, applicability_gaps=[]))
    payload = tmp_path / 'identical' / '_mojolearn.so'
    payload.parent.mkdir()
    payload.write_bytes(b'fixture-payload')
    manifest = dict(schema='mojolearn.ptx-baseline.v1', code_format='ptx-baseline', vendor='cuda',
                    target='sm_80', source_commit='a' * 40, source_dirty=False, experimental=True,
                    identical_qualified=False, errors=[], files=[dict(file='identical/_mojolearn.so',
                    sha256=q.sha(payload), numeric_mode='identical', ptx_modules=[dict(target='sm_80', sha256='c' * 64)])])
    manifest_path = tmp_path / 'PTX_BASELINE.json'
    q.write(manifest_path, manifest)
    train, heldout = q.fixture_witnesses(h, ['base'])
    cell = dict(verdict='STABLE', hashes=['d' * 16] * 2,
                parts=[{'predict': 'd' * 16}] * 2, reload=['d' * 16] * 2)
    for part in ('infer', 'model', 'batch', *h.PROPERTY_PARTS):
        cell[part] = ['d' * 16] * 2
        cell[part + '_verdict'] = 'STABLE'
    column = dict(complete=True, commit='a' * 40, mode='identical', vendor='nvidia', repeats=2,
                  parts_collected=list(h.NARROWABLE_PARTS), fixtures=train, heldout=heldout,
                  cells={'ridge/base': cell}, package={'bindings': [dict(module='_mojolearn', sha256=q.sha(payload))]},
                  batch_protocol=dict(alone=16, split=[1, 8, 'n'], prefix='1,7,full-1', enabled=True))
    for part in h.PROPERTY_PARTS:
        column[part + '_protocol'] = h._part_protocol(part, 16)
    receipts = []
    for i, (role, cap) in enumerate([('baseline', [8, 0]), ('baseline', [9, 0]), ('native-reference', [8, 9])]):
        col_path = tmp_path / f'{i}.column.json'
        q.write(col_path, column)
        fmt = 'ptx-baseline' if role == 'baseline' else 'native'
        receipt = dict(schema=q.SCHEMA, collector='same-process-loaded-bindings-v1', role=role,
                       source_commit='a' * 40, harness_sha256=q.harness_digest(), exit_code=0,
                       hardware=dict(uuid=f'GPU-{i}', name=f'GPU {i}', compute_capability=cap),
                       driver_version='test-driver', lanes=['ridge'], fixtures=['base'],
                       column_file=col_path.name, column_sha256=q.sha(col_path),
                       selection=dict(requested=fmt, selected=fmt, native_fallback=False,
                                      manifest_sha256=q.sha(manifest_path), loaded_files=[dict(module='_mojolearn',
                                      file='identical/_mojolearn.so', sha256=q.sha(payload))]))
        receipt['selection']['runtime_receipt'] = dict(schema='mojolearn.ptx-baseline-selection.v1',
            source_commit='a' * 40, manifest_sha256=q.sha(manifest_path), requested=fmt, selected=fmt,
            native_fallback=False, loaded_files=[dict(file='identical/_mojolearn.so', sha256=q.sha(payload))])
        commit_text = 'a' * 40 + '\n'
        receipt['installed_source'] = dict(core_commit=commit_text,
            core_commit_sha256=q.hashlib.sha256(commit_text.encode()).hexdigest(), inventories=[])
        if role == 'native-reference':
            member = 'mojolearn/cuda_native/sm_89/identical/_mojolearn.so'
            receipt['selection']['loaded_files'][0]['installed_member'] = member
            doc = dict(schema='mojolearn.linux-payload.v1', source_commit='a' * 40,
                split=dict(distribution='mojolearn-nvidia-sm89', native_members=[member]),
                extensions={member: q.sha(payload)})
            receipt['installed_source']['inventories'] = [dict(distribution='mojolearn-nvidia-sm89',
                document=doc, document_sha256=q.document_sha(doc), installed_text_sha256='e' * 64)]
        path = tmp_path / f'{i}.json'
        q.write(path, receipt)
        receipts.append(path)
    return SimpleNamespace(manifest=manifest_path, receipts=receipts, payload=payload, h=h)


def alter(path, mutation):
    obj = q.read(path)
    mutation(obj)
    path.write_text(json.dumps(obj))


def alter_column(campaign, mutation):
    receipt = campaign.receipts[0]
    column = receipt.parent / q.read(receipt)['column_file']
    alter(column, mutation)
    alter(receipt, lambda r: r.update(column_sha256=q.sha(column)))


def test_complete_observed_agreement_does_not_enable_identical(campaign):
    result = q.check(campaign.manifest, campaign.receipts)
    assert result['status'] == 'OBSERVED_CONFIGURATION_AGREEMENT'
    assert result['compared_parts'] == 10
    assert result['full_applicable_single_gpu_coverage']
    assert not result['identical_qualified'] and not result['release_qualified']
    assert not result['future_drivers_qualified'] and not result['universal_gpu_support']


def test_mismatch_names_exact_part_without_dumping_column(campaign):
    alter_column(campaign, lambda c: c['cells']['ridge/base'].update(hashes=['e' * 16] * 2))
    with pytest.raises(ValueError, match='Bitwise or structural result mismatch: ') as error:
        q.check(campaign.manifest, campaign.receipts)
    detail = json.loads(str(error.value).split(': ', 1)[1])
    assert detail['count'] == 1
    assert detail['receipt'] == str(campaign.receipts[1])
    assert detail['examples'] == [dict(part='ridge/base/train',
                                      reference='e' * 16, actual='d' * 16)]


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(source_commit='b' * 40),
    lambda r: r.update(harness_sha256='b' * 64),
    lambda r: r.update(collector='manually-entered'),
    lambda r: r.update(exit_code=1),
    lambda r: r.update(column_sha256='b' * 64),
    lambda r: r.update(driver_version=''),
    lambda r: r.pop('installed_source'),
    lambda r: r['installed_source'].update(core_commit='b' * 40 + '\n'),
    lambda r: r['hardware'].update(uuid=''),
    lambda r: r['hardware'].update(compute_capability=[7, 5]),
    lambda r: r['selection'].update(selected='native'),
    lambda r: r['selection'].update(native_fallback=True),
    lambda r: r['selection'].update(manifest_sha256='b' * 64),
    lambda r: r['selection'].pop('runtime_receipt'),
    lambda r: r['selection']['runtime_receipt'].update(loaded_files=[]),
    lambda r: r['selection']['runtime_receipt'].update(native_fallback=True),
    lambda r: r['selection'].update(loaded_files=[]),
    lambda r: r['selection']['loaded_files'][0].update(sha256='b' * 64),
    lambda r: r['selection']['loaded_files'][0].update(file='fast/_mojolearn.so'),
])
def test_receipt_tampering_is_refused(campaign, mutation):
    alter(campaign.receipts[0], mutation)
    with pytest.raises(ValueError):
        q.check(campaign.manifest, campaign.receipts)


@pytest.mark.parametrize('mutation', [
    lambda c: c.update(complete=False),
    lambda c: c.update(commit='b' * 40),
    lambda c: c.update(vendor='amd'),
    lambda c: c.update(mode='fast'),
    lambda c: c.update(repeats=1),
    lambda c: c.update(skipped=['ridge']),
    lambda c: c.update(cells={}),
    lambda c: c.update(fixtures={}),
    lambda c: c.update(heldout={}),
    lambda c: c.update(batch_protocol={}),
    lambda c: c.update(stepfull_protocol={}),
    lambda c: c.update(parts_collected=[]),
    lambda c: c['cells']['ridge/base'].update(hashes=['e' * 16] * 2),
    lambda c: c['cells']['ridge/base'].update(batch=['n/a:skipped'] * 2, batch_verdict='N/A'),
    lambda c: c['cells']['ridge/base'].update(batch=['n/a:UNDECLARED'] * 2, batch_verdict='N/A'),
    lambda c: c['cells']['ridge/base'].pop('model'),
    lambda c: c['cells']['ridge/base'].update(infer_verdict='MOVED'),
    lambda c: c['cells']['ridge/base'].update(probe_error='failure'),
])
def test_bad_columns_remain_bad_even_with_rehashed_receipt(campaign, mutation):
    alter_column(campaign, mutation)
    with pytest.raises(ValueError):
        q.check(campaign.manifest, campaign.receipts)


def undeclare_batch(campaign, receipts):
    for receipt in receipts:
        column = receipt.parent / q.read(receipt)['column_file']
        alter(column, lambda c: c['cells']['ridge/base'].update(batch=['n/a:UNDECLARED'] * 2, batch_verdict='N/A'))
        alter(receipt, lambda r: r.update(column_sha256=q.sha(column)))


def test_pinned_undeclared_part_is_reported_excluded_never_compared(campaign, monkeypatch):
    monkeypatch.setattr(q, 'UNDECLARED_EXCLUSIONS', (('ridge', 'batch'), ('absent-lane', 'batch')))
    undeclare_batch(campaign, campaign.receipts)
    result = q.check(campaign.manifest, campaign.receipts)
    assert result['compared_parts'] == 9 and result['excluded_parts'] == 1
    assert result['undeclared_exclusions'] == [dict(lane='ridge', part='batch', fixtures=['base'],
                                                    value='n/a:UNDECLARED')]


def test_pinned_undeclared_part_must_be_undeclared_in_every_column(campaign, monkeypatch):
    monkeypatch.setattr(q, 'UNDECLARED_EXCLUSIONS', (('ridge', 'batch'),))
    # Every column still carries a batch hash: the pin is stale and refuses.
    with pytest.raises(ValueError, match='Pinned undeclared exclusion carries another value: ridge/base/batch'):
        q.check(campaign.manifest, campaign.receipts)
    undeclare_batch(campaign, campaign.receipts[:2])
    with pytest.raises(ValueError, match='Pinned undeclared exclusion carries another value'):
        q.check(campaign.manifest, campaign.receipts)


def test_unpinned_undeclared_part_still_fails_in_every_column(campaign):
    undeclare_batch(campaign, campaign.receipts)
    with pytest.raises(ValueError, match='Unverified part: ridge/base/batch'):
        q.check(campaign.manifest, campaign.receipts)


def test_default_pin_names_only_the_two_measured_lanes():
    assert q.UNDECLARED_EXCLUSIONS == (('gbdt-class-weights', 'batch'), ('gbdt-multiclass-offgrid', 'batch'))
    assert q.undeclared_scope(['ridge'], ['base']) == []


def test_actual_payload_bytes_are_checked(campaign):
    campaign.payload.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='Payload bytes changed'):
        q.check(campaign.manifest, campaign.receipts)


def test_missing_native_reference_cannot_qualify(campaign):
    with pytest.raises(ValueError, match='native reference'):
        q.check(campaign.manifest, campaign.receipts[:2], prototype=True)


def test_one_architecture_is_only_prototype_evidence(campaign):
    receipts = [campaign.receipts[0], campaign.receipts[2]]
    with pytest.raises(ValueError, match='distinct GPU'):
        q.check(campaign.manifest, receipts)
    result = q.check(campaign.manifest, receipts, prototype=True)
    assert result['status'] == 'PROTOTYPE_AGREEMENT'
    assert not result['full_applicable_single_gpu_coverage']


def test_scope_cannot_shrink_to_whatever_was_recorded(campaign, monkeypatch):
    monkeypatch.setattr(q, 'inventory', lambda _: dict(lanes=['ridge', 'kmeans'], fixtures=['base', 'ties'],
                                                     excluded={}, applicability_gaps=[]))
    with pytest.raises(ValueError, match='Missing full applicable'):
        q.check(campaign.manifest, campaign.receipts)
    assert q.check(campaign.manifest, campaign.receipts, prototype=True)['status'] == 'PROTOTYPE_AGREEMENT'


def test_inventory_ambiguity_blocks_full_admission(campaign, monkeypatch):
    monkeypatch.setattr(q, 'inventory', lambda _: dict(lanes=['ridge'], fixtures=['base'],
                                                     excluded={}, applicability_gaps=['unresolved route']))
    with pytest.raises(ValueError, match='applicability metadata'):
        q.check(campaign.manifest, campaign.receipts)


def test_manifest_rejects_dirty_source_and_wrong_target(campaign):
    manifest = q.read(campaign.manifest)
    for field, value in [('source_dirty', True), ('source_commit', 'main'), ('target', 'sm_90a')]:
        bad = copy.deepcopy(manifest)
        bad[field] = value
        with pytest.raises(ValueError):
            q.manifest_files(bad)


@pytest.mark.parametrize('fallback', [False, True])
def test_collector_binds_actual_loaded_files_to_runtime_path(campaign, monkeypatch, fallback):
    vref = identity_columns.vref()
    monkeypatch.setattr(q, 'modules', lambda: (campaign.h, vref))
    sample = q.read(campaign.receipts[0])
    column = q.read(campaign.receipts[0].parent / sample['column_file'])
    runtime = sample['selection']['runtime_receipt']
    runtime['native_fallback'] = fallback
    backend = types.ModuleType('mojolearn._backend')
    backend.gpu_plugin = lambda: {'code_format': 'ptx-baseline'}
    backend.baseline_selection_receipt = lambda: runtime
    installed = campaign.manifest.parent / 'site-packages' / 'mojolearn'
    baseline_root = installed / 'cuda_ptx' / 'sm_80'
    installed_payload = baseline_root / 'identical' / '_mojolearn.so'
    installed_payload.parent.mkdir(parents=True)
    installed_payload.write_bytes(campaign.payload.read_bytes())
    (installed / 'identity_columns').mkdir()
    (installed / 'identity_columns/COMMIT').write_text('a' * 40 + '\n')
    backend._BASELINE_ROOT = str(baseline_root)
    verifier = types.ModuleType('mojolearn._verify')
    verifier.binding_artifacts = lambda: [dict(module='_mojolearn', file=str(installed_payload), sha256=q.sha(installed_payload))]
    package = types.ModuleType('mojolearn')
    package.__file__ = str(installed / '__init__.py')
    package.vendor = lambda: 'cuda'
    package.numeric_mode = lambda: 'identical'
    package._backend = backend
    monkeypatch.setitem(sys.modules, 'mojolearn', package)
    monkeypatch.setitem(sys.modules, 'mojolearn._backend', backend)
    monkeypatch.setitem(sys.modules, 'mojolearn._verify', verifier)
    monkeypatch.setattr(q.subprocess, 'check_output', lambda cmd, **kwargs:
                        'a' * 40 if cmd[0] == 'git' else 'GPU-0, test GPU, 8.0, test-driver')
    monkeypatch.setattr(q.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0))

    def run_harness():
        path = q.Path(sys.argv[sys.argv.index('--json') + 1])
        q.write(path, column)
        return 0

    campaign.h.main = run_harness
    out = campaign.manifest.parent / 'collected.json'
    args = SimpleNamespace(manifest=campaign.manifest, out=out, lanes='ridge', fixtures='base', role='baseline')
    if fallback:
        with pytest.raises(ValueError, match='forced-baseline runtime receipt'):
            q.collect(args)
        assert not out.exists()
    else:
        receipt = q.collect(args)
        assert receipt['selection']['runtime_receipt'] == runtime
        assert receipt['column_sha256'] == q.sha(out.with_suffix('.column.json'))
        assert receipt['selection']['loaded_files'][0]['file'] == 'identical/_mojolearn.so'


@pytest.mark.parametrize('mutation', [
    lambda r: r['installed_source'].update(inventories=[]),
    lambda r: r['installed_source']['inventories'][0]['document'].update(source_commit='b' * 40),
    lambda r: r['installed_source']['inventories'][0].update(document_sha256='b' * 64),
    lambda r: r['selection']['loaded_files'][0].update(installed_member='mojolearn/cuda_ptx/sm_80/identical/_mojolearn.so'),
    lambda r: r['installed_source']['inventories'][0]['document']['extensions'].update(
        {'mojolearn/cuda_native/sm_89/identical/_mojolearn.so': 'b' * 64}),
])
def test_old_or_unwitnessed_native_wheels_cannot_borrow_harness_source(campaign, mutation):
    alter(campaign.receipts[2], mutation)
    with pytest.raises(ValueError):
        q.check(campaign.manifest, campaign.receipts)


def test_installed_core_source_is_checked_independently_of_checkout(tmp_path):
    package = tmp_path / 'mojolearn'
    (package / 'identity_columns').mkdir(parents=True)
    (package / 'identity_columns/COMMIT').write_text('b' * 40 + '\n')
    with pytest.raises(ValueError, match='Installed core source differs'):
        q.installed_source_evidence(package, None, 'baseline', 'a' * 40, [])


def test_native_collection_reads_installed_distribution_inventory(tmp_path):
    package = tmp_path / 'mojolearn'
    (package / 'identity_columns').mkdir(parents=True)
    (package / 'identity_columns/COMMIT').write_text('a' * 40 + '\n')
    member = 'mojolearn/cuda_native/sm_89/identical/_mojolearn.so'
    binary = tmp_path / member
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b'native test')
    docs = {name: dict(schema='mojolearn.linux-payload.v1', source_commit='a' * 40,
                      split=dict(distribution=name, native_members=[]), extensions={})
            for name in ('mojolearn', 'mojolearn-nvidia-sm89')}
    docs['mojolearn-nvidia-sm89']['split']['native_members'] = [member]
    docs['mojolearn-nvidia-sm89']['extensions'][member] = q.sha(binary)
    backend = SimpleNamespace(gpu_plugin=lambda: dict(payloads=['mojolearn-nvidia-sm89']),
        _find_distribution=lambda name, paths: SimpleNamespace(read_text=lambda file: json.dumps(docs[name])))
    loaded = [dict(module='_mojolearn', file=str(binary), sha256=q.sha(binary))]
    evidence = q.installed_source_evidence(package, backend, 'native-reference', 'a' * 40, loaded)
    assert loaded[0]['installed_member'] == member and len(evidence['inventories']) == 2
    docs['mojolearn-nvidia-sm89']['source_commit'] = 'b' * 40
    with pytest.raises(ValueError, match='inventory source differs'):
        q.installed_source_evidence(package, backend, 'native-reference', 'a' * 40, loaded)
    docs['mojolearn-nvidia-sm89']['source_commit'] = 'a' * 40
    docs['mojolearn-nvidia-sm89']['extensions'][member] = 'b' * 64
    with pytest.raises(ValueError, match='Loaded native bytes differ'):
        q.installed_source_evidence(package, backend, 'native-reference', 'a' * 40, loaded)
