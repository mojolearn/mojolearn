"""Evidence must survive invalid, incomplete and incomparable control records."""
import copy
import types
import pytest
import verification_matrix as matrix
import verification_evidence as evidence


def test_fragment_changes_invalidate_inventory_without_rebuilding(tmp_path):
    sources = matrix.load('python/mojolearn/_verification_sources.py', '_test_inventory_sources')
    main, fragment = tmp_path/'harness.py', tmp_path/'lane.py'
    main.write_text('# unchanged harness\n')
    fragment.write_text('value = 1\n')
    harness = types.SimpleNamespace(__file__=str(main), lane_fragment_paths=lambda: [('x', str(fragment))])
    before = sources.harness_sources(harness)
    fragment.write_text('value = 2\n')
    after = sources.harness_sources(harness)
    assert before['harness'] == after['harness']
    assert before['lane/x'] != after['lane/x']


def cell(value='a' * 16, part='train'):
    return {('hashes' if part == 'train' else part): [value] * 2,
            ('verdict' if part == 'train' else part + '_verdict'): 'STABLE'}


def test_source_fingerprint_survives_packaged_harness_but_detects_new_api(tmp_path):
    sources = matrix.load('python/mojolearn/_verification_sources.py', '_test_public_sources')
    (tmp_path/'ops.py').write_text('value = 1\n')
    before = sources.public_sources(tmp_path)
    (tmp_path/'_identity_break.py').write_text('# packaged harness\n')
    assert sources.public_sources(tmp_path) == before
    (tmp_path/'new_api.py').write_text('def new(): pass\n')
    assert sources.public_sources(tmp_path) != before


@pytest.mark.parametrize('bad', [None, {}, {'verdict': 'REFUSED', 'hashes': ['error'] * 2},
    {'verdict': 'N/A', 'hashes': ['n/a:no-check'] * 2},
    {'verdict': 'STABLE', 'hashes': ['a' * 16, 'b' * 16]}])
def test_invalid_clean_or_changed_arm_is_not_a_negative_control(bad):
    assert not matrix.negative_control_moves(cell('b' * 16), bad, 'train')
    assert not matrix.negative_control_moves(bad, cell(), 'train')


def test_changed_bytes_and_explicit_failed_batch_assertion_are_controls():
    assert matrix.negative_control_moves(cell('b' * 16), cell(), 'train')
    assert not matrix.negative_control_moves(cell(), cell(), 'train')
    # every cell is fitted once: one fit per arm is a control when the bytes differ
    once = {'verdict': 'STABLE', 'hashes': ['a' * 16]}
    assert matrix.negative_control_moves(cell('b' * 16), once, 'train')
    assert not matrix.negative_control_moves(once, cell(), 'train')
    bad = {'batch_verdict': 'BATCH_MOVED', 'batch': ['BATCH_MOVED: changed row'] * 2}
    assert matrix.negative_control_moves(bad, cell(part='batch'), 'batch')
    bad['batch'] = ['RuntimeError: missing binding'] * 2
    assert not matrix.negative_control_moves(bad, cell(part='batch'), 'batch')


def record():
    return dict(commit='a' * 40, mode='identical',
                fixtures={'base': {'X': 'input'}}, heldout={'base': {'X': 'held'}},
                batch_protocol={'alone': 16}, package={'par_devices': '0'}, cells={})


@pytest.mark.parametrize('field,value', [('commit', 'b' * 40), ('mode', 'fast'),
    ('fixtures', {}), ('heldout', {}), ('batch_protocol', {'alone': 1}),
    ('lane_revisions', {'par-x': 'changed'}), ('package', {'fixture_n': 12})])
def test_pair_cannot_cross_input_revision_protocol_or_mode(field, value):
    clean = record()
    changed = copy.deepcopy(clean)
    changed[field] = value
    assert not evidence.same_cell_context(clean, changed, 'par-x/base', 'batch')


def test_pair_context_and_distinct_devices():
    clean = record()
    assert evidence.same_cell_context(clean, copy.deepcopy(clean), 'par-x/base', 'batch')
    for raw, expected in [('0,1', (0, 1)), ('0,0', ()), ('-1,0', ()), ('wrong', ())]:
        clean['package']['par_devices'] = raw
        assert evidence.devices(clean) == expected


def test_multi_admission_reuses_all_other_exclusions():
    calls = []
    def admit(j, path):
        calls.append(j)
        return 'probe' if 'probe' in path else None
    j = record()
    j['package']['par_devices'] = '0,1'
    c = dict(record=j, sabotage=False, cls='nvidia', rel='clean.json')
    reference = types.SimpleNamespace(admit=admit)
    assert evidence.admitted_multi(c, reference)
    assert calls[-1]['package']['par_devices'] == '0'
    assert j['package']['par_devices'] == '0,1'  # original witness stays intact
    c['rel'] = 'probe/clean.json'
    assert not evidence.admitted_multi(c, reference)
    c['rel'], c['sabotage'] = 'clean.json', True
    assert not evidence.admitted_multi(c, reference)


def test_both_packers_share_digest_checked_ctr_payload(tmp_path):
    from pathlib import Path
    import verification_ctr_payload as payload
    root = Path(__file__).resolve().parents[1]
    entries = payload.model_entries(root)
    assert len(entries) == 18 and all(path.is_file() for path in entries.values())
    # A packaging staging tree with a declared but corrupted model must refuse.
    package = tmp_path/'python/mojolearn'
    package.mkdir(parents=True)
    (package/'host_surface.py').write_text("GBDT_CTR_MODELS_DIR = 'models'\n")
    (package/'_verification_ctr_models.py').write_text("MODEL_SHA256 = {'x.npz': '" + 'a'*64 + "'}\n")
    (tmp_path/'models').mkdir()
    (tmp_path/'models/x.npz').write_bytes(b'wrong')
    with pytest.raises(ValueError, match='missing or changed'):
        payload.model_entries(tmp_path)
