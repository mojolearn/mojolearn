"""Coverage is an inspectable contract, never a substitute for execution."""
import types
import numpy as np
import pytest
from mojolearn import __main__ as cli, _verify_all as va, _verify_reference as vr
from mojolearn import _verification_coverage as coverage
from mojolearn._verification_catalog import ENTRIES


def test_all_246_appendix_entries_are_preserved_and_resolve():
    h = va.load_harness()
    assert len(ENTRIES) == len({e['id'] for e in ENTRIES}) == 246
    assert len({e['group'] for e in ENTRIES}) == 12
    for entry in ENTRIES:
        assert entry['lanes'] or entry.get('alternative_gate')
        assert set(entry['lanes']) <= set(h.LANES), entry
    report = coverage.inventory(h, vr.load_table(), 'cpu')
    mapped = {l for e in ENTRIES for l in e['lanes']}
    assert set(report['additional_lanes']) == set(h.LANES) - mapped
    assert report['execution'] == 'not run'
    assert report['lanes']['select-d']['status'] == 'withheld'
    assert report['lanes']['select-d']['properties']['batch']['kind'] == 'check'
    assert report['lanes']['bpe-trainer']['properties']['batch']['kind'] == 'not_applicable'
    assert report['lanes']['transformer']['properties']['batchgrad']['command'] == 'verify --batch-checks'


def test_inspection_and_batch_flags_route_to_suite():
    for flag in ('--coverage', '--batch-checks'):
        assert cli._wants_suite(cli.build_parser().parse_args(['verify', flag]))


def test_explicit_pending_cpu_lane_can_run_but_is_not_promoted():
    h = va.load_harness()
    lanes, _ = va.select_lanes(h, vr.load_table(), 'cpu', 'full', ['select-d'])
    assert lanes == ['select-d']
    assert 'select-d' not in va.host_surface().public_reference_lanes()
    for name in ('bpe-trainer', 'cross-val-folds'):
        assert name in va.host_surface().public_reference_lanes()


def test_partial_lane_is_not_counted_as_end_to_end():
    rows = [dict(lane='x', state=vr.IDENTICAL), dict(lane='x', state=vr.OWED)]
    assert va.lane_counts(rows, ['x'])['checked'] == 0
    assert va.lanes_line(dict(cells=rows)).startswith('checked 0 of 1')


def test_local_batch_pass_without_hash_is_still_owed():
    rows = va.judge_rows([dict(lane='x', fixture='base', part='batch',
                              value='0123456789abcdef', error=None)], dict(cells={}))
    assert rows[0]['local_check'] == 'passed'
    assert rows[0]['state'] == vr.OWED
    counts = {s: int(s == vr.OWED) for s in vr.STATES}
    assert va.verdict(counts)[0] != 0


def test_opt_in_probes_reuse_fit_and_detect_drift():
    fits, probes = [], []
    def fit(*args):
        fits.append(1)
        return object()
    def probe(part, *args):
        probes.append(part)
        return ('BATCH_MOVED: wrong rows' if part == 'batchgrad' else 'n/a:test'), None, []
    h = types.SimpleNamespace(LANES={'x': fit}, BATCH_ALONE=1,
        EXTRA_PARTS={p: () for p in ('stepfull', 'batchgrad', 'batchscale', 'ragged')},
        _train_hash=lambda f: '0123456789abcdef',
        _probe_fit=lambda *args: ('n/a:function', 'n/a:no-save', None, None),
        _probe_batch=lambda *args: ('n/a:test', None), _probe_part=probe)
    result = va.run_cell(h, None, 'x', 'base', (None, None, None), np.zeros((2, 2)), 2,
                         extra_parts=('batchgrad', 'batchscale', 'ragged'))
    assert len(fits) == 2
    assert probes == ['stepfull', 'batchgrad', 'batchscale', 'ragged'] * 2
    value, error = result['batchgrad']
    assert vr.judge(value, None, error)[0] == vr.DIVERGENT


def test_select_d_cpu_chooses_first_stationary_order_and_preserves_input(monkeypatch):
    from mojolearn import _tsa_impl as tsa, _backend
    from mojolearn import Array
    calls = []
    def kpss(y, **kw):
        calls.append(kw)
        return Array.from_list([1, 0, 0] if kw['d'] == 0 else [0, 1, 0], '<u1')
    monkeypatch.setattr(_backend, 'vendor', lambda: 'cpu')
    monkeypatch.setattr(tsa, 'kpss_test', kpss)
    data = np.arange(90, dtype=np.float32).reshape(30, 3)
    before = data.tobytes()
    assert tsa.select_d(data, d_max=2).tolist() == [0, 1, 2]
    assert [c['d'] for c in calls] == [0, 1]
    assert data.tobytes() == before
    calls.clear()
    assert tsa.select_d(data, D=1, s=12).tolist() == [0, 1, 1]
    assert len(calls) == 1 and calls[0]['D'] == 1
    with pytest.raises(ValueError, match='d_max must satisfy'):
        tsa.select_d(data, D=1, s=12, d_max=2)


def test_a_withheld_lane_blocks_a_full_scope_pass():
    counts = {state: int(state == vr.IDENTICAL) for state in vr.STATES}
    assert va.verdict(counts) == (0, "VERIFIED")
    assert va.verdict(counts, {"pending": "no reference"}) == (va.EXIT_NO_REFERENCE, "INCOMPLETE")
    counts[vr.DIVERGENT] = 1
    assert va.verdict(counts, {"pending": "no reference"})[0] == va.EXIT_MISMATCH


def test_missing_portable_models_cannot_silently_pass(tmp_path):
    rows = va.run_models(None, None, {}, pkg_dir=str(tmp_path))
    assert len(rows) == 1 and rows[0]['value'] is None
    judged = va.judge_rows(rows, dict(cells={}))
    assert judged[0]['state'] == vr.REFUSED


def test_sampler_replay_failure_is_not_misreported_as_missing_reference():
    value, error = va._collapse(['RLPAIR_MOVED: changed logit'] * 2, [])
    assert vr.judge(value, None, error)[0] == vr.DIVERGENT
    assert va._value_kind(value) == 'moved'


def test_reference_builder_can_include_extended_checks(tmp_path, monkeypatch):
    import hashlib, json
    h = types.SimpleNamespace(LANES={'x': None}, FIXTURES=['base'], LANE_REVISIONS={},
        __file__=__file__, BATCH_ALONE=16,
        _part_protocol=lambda part, alone: dict(part=part, alone=alone),
        _rlpair_protocol=lambda: dict(enabled=True), fixture=lambda f: (np.zeros(1),)*3,
        heldout=lambda f: np.zeros(1), _h=lambda a: hashlib.sha256(a.tobytes()).hexdigest()[:16])
    fingerprint = h._h(np.zeros(1))
    cell = dict(verdict='STABLE', hashes=['a'*16]*2)
    for part in vr.OPTIONAL_PARTS:
        cell[part] = ['b'*16]*2
        cell[part+'_verdict'] = 'STABLE'
    record = dict(mode='identical', commit='a'*40, vendor='cpu-test',
        fixtures={'base': dict(X=fingerprint, y_clf=fingerprint, y_reg=fingerprint)},
        heldout={'base': dict(X=fingerprint)}, cells={'x/base': cell})
    for part in vr.OPTIONAL_PARTS:
        record[part+'_protocol'] = h._rlpair_protocol() if part == 'rlpair' else h._part_protocol(part, h.BATCH_ALONE)
    path = tmp_path/'cpu.json'
    path.write_text(json.dumps(record))
    monkeypatch.setattr(vr, '_commit_time', lambda *args: 1)
    table = vr.build_table([str(path)], h, str(tmp_path), parts=vr.PARTS+vr.OPTIONAL_PARTS)
    for part in vr.OPTIONAL_PARTS:
        assert table['cells']['x/base'][part]['ref'] == 'b'*16
    ordinary = vr.build_table([str(path)], h, str(tmp_path))
    assert not (set(ordinary['cells']['x/base']) & set(vr.OPTIONAL_PARTS))
    record['batchgrad_protocol']['alone'] = 99
    path.write_text(json.dumps(record))
    changed = vr.build_table([str(path)], h, str(tmp_path), parts=vr.PARTS+vr.OPTIONAL_PARTS)
    assert 'batchgrad' not in changed['cells']['x/base']
    assert 'rlpair' in changed['cells']['x/base']


def test_historical_evidence_is_exposed_without_release_certification():
    h = va.load_harness()
    report = coverage.inventory(h, vr.load_table(), 'cpu')
    assert report['evidence_provenance']['release_qualified'] is False
    assert report['evidence_provenance']['sources']
    for row in report['entries']:
        assert row['evidence_summary']['release_qualified'] is False
    for lane in report['lanes'].values():
        assert 'historical_evidence' in lane
        assert lane['release_qualified'] is False
    assert 'not qualification of this wheel' in coverage.format_human(report)


@pytest.mark.parametrize('change', ['missing', 'inputs', 'heldout', 'protocol', 'harness'])
def test_equal_hashes_do_not_hide_incomparable_experiments(change):
    import copy
    def document(vendor):
        return dict(format='mojolearn.verify-all-report.v1', device=dict(vendor=vendor),
            cells=[dict(lane='x', fixture='base', part='batch', value='a'*16, state='IDENTICAL')],
            verification_contract=dict(harness_sha256='b'*64,
                fixtures={'base': {'X': 'input'}}, heldout={'base': {'X': 'held'}},
                protocols={'batch': {'alone': 16}}))
    a, b = document('cuda'), document('hip')
    assert va.compare_documents(a, b)['verdict'] == 'AGREE'
    if change == 'missing':
        b.pop('verification_contract')
    else:
        key = dict(inputs='fixtures', heldout='heldout', protocol='protocols', harness='harness_sha256')[change]
        b['verification_contract'][key] = {} if key != 'harness_sha256' else 'c'*64
    result = va.compare_documents(a, b)
    assert result['exit'] == va.EXIT_CANNOT_RUN and result['agree'] == 0
    assert result['verdict'] == 'INCOMPARABLE' and result['context_problems']


def test_bundled_ctr_models_are_complete_and_digest_checked(tmp_path):
    from pathlib import Path
    from mojolearn import _verification_ctr_models as ctr
    h = va.load_harness()
    assert len(ctr.MODEL_SHA256) == len(h.FIXTURES) * 2
    assert set(("gbdt-categorical-ctr-tables", "gbdt-tensor-ctr-tables")) <= set(va.host_surface().public_reference_lanes())
    for lane in ('gbdt-categorical-ctr-tables', 'gbdt-tensor-ctr-tables'):
        for fixture in h.FIXTURES:
            assert Path(ctr.resolve_model(lane, fixture)).is_file()
    target = tmp_path / 'verify_reference/ctr_models'
    target.mkdir(parents=True)
    (target/'gbdt-categorical-ctr-tables.base.npz').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='digest mismatch'):
        ctr.resolve_model('gbdt-categorical-ctr-tables', 'base', tmp_path)
    with pytest.raises(RuntimeError, match='missing verification'):
        ctr.resolve_model('gbdt-tensor-ctr-tables', 'base', tmp_path)
