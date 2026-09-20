"""Coverage is an inspectable contract, never a substitute for execution."""
import types
import numpy as np
import pytest
from mojolearn import __main__ as cli, _verify_all as va, _verify_reference as vr
from mojolearn import _verification_coverage as coverage
from mojolearn._verification_catalog import ENTRIES


def test_reference_support_does_not_count_old_conflicted_or_na_as_numeric_evidence():
    records = [dict(dir='records', file=c+'.json', vendor=c, commit='a'*40)
               for c in ('cpu', 'amd')]
    table = dict(records=records, cells={
        'x/base': dict(train=dict(ref='a'*16, cols={'cpu': 0, 'amd': [1, 'b'*16]})),
        'x/ties': dict(train=dict(ref='a'*16, conflict=True, cols={'cpu': 0})),
        'x/odd': dict(train=dict(ref='n/a:no-operation', cols={'cpu': 0})),
    })
    fixtures = ['base', 'ties', 'odd', 'missing']
    row = coverage.reference_support(table, 'x', fixtures, ['train'])['train']
    assert row['numerical_fixtures'] == 1
    assert row['not_applicable_fixtures'] == 1
    assert row['missing_or_conflicted_fixtures'] == 2
    assert row['agreeing_device_classes'] == dict(cpu=1, apple=0, nvidia=0, amd=0)
    stale = coverage.reference_support(table, 'x', fixtures, ['train'], stale=True)['train']
    assert stale['stale_fixtures'] == 4
    assert stale['numerical_fixtures'] == 0
    assert not any(stale['agreeing_device_classes'].values())


def test_all_246_appendix_entries_are_preserved_and_resolve():
    h = va.load_harness()
    assert len(ENTRIES) == len({e['id'] for e in ENTRIES}) == 246
    assert len({e['group'] for e in ENTRIES}) == 12
    for entry in ENTRIES:
        assert entry['lanes'] or entry.get('alternative_gate')
        assert set(entry['lanes']) <= set(h.LANES), entry
    report = coverage.inventory(h, vr.load_table(), 'cpu')
    stale = set(vr.stale_reference_lanes(vr.load_table(), h))
    for name, row in report['lanes'].items():
        if row['reason'] == 'stale reference':
            assert name in stale
    mapped = {l for e in ENTRIES for l in e['lanes']}
    assert set(report['additional_lanes']) == set(h.LANES) - mapped
    assert report['execution'] == 'not run'
    assert report['lanes']['select-d']['status'] == 'available'
    assert report['lanes']['holtwinters']['status'] == 'available'
    assert report['lanes']['select-d']['properties']['batch']['kind'] == 'check'
    assert report['lanes']['bpe-trainer']['properties']['batch']['kind'] == 'not_applicable'
    assert report['lanes']['transformer']['properties']['batchgrad']['command'] == 'verify --batch-checks'


def test_inspection_and_batch_flags_route_to_suite():
    for flag in ('--coverage', '--batch-checks'):
        assert cli._wants_suite(cli.build_parser().parse_args(['verify', flag]))


def test_a_pending_reason_annotates_and_only_a_blocking_one_withholds(monkeypatch):
    """A PENDING REASON STOPPED DECIDING VISIBILITY (Andrew, 2026-09-20).

    This test used to assert the opposite: that a lane in PUBLIC_PENDING_LANES
    was NOT in the public set, whatever its reason said. That is the rule that
    hid 76 lanes. Now the dict annotates, and only three classes of reason --
    `no reference`, `no cpu route`, `stale reference` -- stop the verifier
    comparing a lane. Both directions are checked, because a rule that only
    ever admits is not a rule."""
    surface = va.host_surface()
    h = va.load_harness()

    # an ANNOTATING reason leaves the lane comparable and public
    monkeypatch.setitem(surface.PUBLIC_PENDING_LANES, 'ols', 'unwatched')
    assert 'ols' in surface.public_reference_lanes()
    assert 'ols' in surface.comparable_lanes(list(h.LANES), 'cpu')
    lanes, _ = va.select_lanes(h, vr.load_table(), 'cpu', 'full', ['ols'])
    assert lanes == ['ols']

    # a BLOCKING reason does not hide it either; it withholds the comparison
    monkeypatch.setitem(surface.PUBLIC_PENDING_LANES, 'ols', 'no reference')
    assert 'ols' not in surface.comparable_lanes(list(h.LANES), 'cpu')
    assert 'ols' in surface.public_lane_scope(list(h.LANES)), "a blocked lane is still public"
    assert surface.lane_exposure(list(h.LANES))['ols']['status'] == surface.LANE_OWED

    for name in ('bpe-trainer', 'cross-val-folds'):
        assert name in surface.public_reference_lanes()


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


def test_an_unreferenced_lane_no_longer_blocks_a_pass_but_a_wrong_one_does():
    """THE GATE WAS LOOSENED ON PURPOSE, AND EXACTLY TWICE (Andrew,
    2026-09-20). This test asserted the old rule, that any withheld or
    unreferenced lane cost the run its pass. It now asserts the new one and
    the line that was kept: DIVERGENT and REFUSED still gate, because they
    mean something WENT WRONG rather than something is missing, and passing
    over either would make the word mean nothing.

    `verdict`'s second argument is now the GATING gaps only, which is why an
    unreferenced lane never reaches it."""
    counts = {state: int(state == vr.IDENTICAL) for state in vr.STATES}
    assert va.verdict(counts) == (0, "VERIFIED")
    assert va.verdict(counts, {}) == (0, "VERIFIED"), "an unreferenced lane is not a gate"
    counts[vr.OWED] = 7
    assert va.verdict(counts) == (0, "VERIFIED"), "owed parts do not gate either"
    counts[vr.DIVERGENT] = 1
    assert va.verdict(counts)[0] == va.EXIT_MISMATCH
    counts[vr.DIVERGENT] = 0
    counts[vr.REFUSED] = 1
    assert va.verdict(counts)[0] == va.EXIT_CANNOT_RUN


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


def test_probe_error_cannot_be_hidden_by_matching_hashes():
    value='0123456789abcdef'
    assert va._collapse([value,value],['model reload failed']) == (None,'model reload failed')
    state, detail=vr.judge(value,dict(ref=value),error='model reload failed')
    assert state == vr.REFUSED and detail == 'model reload failed'
    assert vr.judge('BATCH_MOVED: broken row',dict(ref=value),error='other failure')[0] == vr.DIVERGENT


@pytest.mark.parametrize('value', [123, [], {}, True, '', 'error: failed', 'n/a:'])
def test_invalid_probe_value_cannot_become_a_matching_reference(value):
    assert vr.judge(value,dict(ref=value))[0] == vr.REFUSED


def test_failed_reload_refuses_the_model_part_even_when_saved_bytes_match():
    digest='0123456789abcdef'
    h=types.SimpleNamespace(LANES={'x':lambda *a:object()},BATCH_ALONE=1,
        EXTRA_PARTS={'stepfull':()},_train_hash=lambda f:digest,
        _probe_fit=lambda *a:(digest,digest,None,'model: reload failed'),
        _probe_batch=lambda *a:('n/a:function',None),
        _probe_part=lambda *a:('n/a:no-decode-state',None,[]))
    parts=va.run_cell(h,None,'x','base',(None,None,None),np.zeros((1,1)),2)
    assert parts['train']==(digest,None)
    assert parts['model']==(None,'model: reload failed')
    rows=va.judge_rows([dict(lane='x',fixture='base',part='model',value=parts['model'][0],error=parts['model'][1])],
                      dict(cells={'x/base':{'model':dict(ref=digest)}}))
    assert rows[0]['state']==vr.REFUSED


def test_no_lane_reports_an_admission_the_table_cannot_support():
    """A LANE MUST NEVER READ STRICTER THAN THE TABLE IT CAME FROM.

    This began as `test_scoped_admission_is_visible_without_upgrading_legacy_lanes`,
    guarding `merge_reference_lanes`: a few lanes grafted onto a legacy base
    are strict, the base is NOT, and the report has to say both. The hazard it
    was built for is one direction only, a lane claiming more than its
    evidence.

    On 2026-09-17 lane/reference-regen rebuilt the whole table with
    `--emit-reference --batch-checks` over every committed record, so there is
    no legacy base left to protect: every cell in the shipped table was
    re-derived under `min_repeats=2`, the input witness and the property
    protocol, and `build_table` writes that policy into the file. The
    assertion that the global policy reads `legacy` was therefore asserting
    the ABSENCE of the regeneration, and it is gone. The invariant is kept and
    now runs the other way: no lane may report weaker than the table either,
    which is what a dropped `lane_admission` entry would look like.

    `merge_reference_lanes` itself stays covered, on synthetic tables that do
    not depend on what the shipped one happens to hold today:
    `test_verify_reference_admit.py`.
    """
    table = vr.load_table()
    report = coverage.inventory(va.load_harness(), table, 'cpu')
    strict = dict(min_repeats=2, input_witness_required=True, property_protocol_required=True)
    assert report['reference_admission_policy'] == strict
    # the lanes the scoped path admitted onto the old legacy base, and one
    # (`ols`) that was legacy at the time: all of them read the same policy
    # now, because all of them came out of the same strict global build
    for lane in ('embedding', 'embedding-sort', 'ivf-euclidean', 'ols'):
        assert report['lanes'][lane]['reference_admission']['policy'] == strict, lane
        assert report['lanes'][lane]['status'] == 'available', lane
    # and nothing anywhere claims a policy the table does not carry
    for name, lane in report['lanes'].items():
        policy = lane['reference_admission']['policy']
        assert policy == strict, (name, policy)


def _absence_harness():
    import hashlib
    return types.SimpleNamespace(
        LANES={'x': None}, FIXTURES=['base'], LANE_REVISIONS={}, __file__=__file__,
        BATCH_ALONE=16, BATCH_SPLIT=(1, 7),
        _part_protocol=lambda part, alone: dict(part=part, alone=alone),
        _rlpair_protocol=lambda: dict(enabled=True),
        fixture=lambda f: (np.zeros(1),) * 3, heldout=lambda f: np.zeros(1),
        _h=lambda a: hashlib.sha256(a.tobytes()).hexdigest()[:16])


def test_an_absent_part_is_named_where_build_table_skips_it(tmp_path, monkeypatch):
    """AN ABSENT PART USED TO BE A BARE `continue` (2026-09-20).

    `build_table` skipped every unusable or uncollected part with no log line
    and no flag, and the only signal was a smaller N in `use <path>: ... N
    cell parts` -- a number with nothing to compare it against. A column
    missing four of its nine parts and a whole one printed the same shape of
    line. This holds the denominator, the per-part counts, and the two
    reasons apart: NOT RUN (a hole in the column) and RAN BUT UNUSABLE (a
    moved, refused, skipped or single-repeat cell the column did collect).
    """
    import json
    h = _absence_harness()
    fingerprint = h._h(np.zeros(1))
    # train is usable; infer RAN and MOVED; every other part was never run.
    cell = dict(verdict='STABLE', hashes=['a' * 16] * 2,
                infer=['a' * 16, 'c' * 16], infer_verdict='MOVED')
    record = dict(mode='identical', commit='a' * 40, vendor='cpu-test',
                  fixtures={'base': dict(X=fingerprint, y_clf=fingerprint, y_reg=fingerprint)},
                  heldout={'base': dict(X=fingerprint)}, cells={'x/base': cell})
    path = tmp_path / 'cpu.json'
    path.write_text(json.dumps(record))
    monkeypatch.setattr(vr, '_commit_time', lambda *args: 1)
    logs = []
    parts = vr.PARTS + vr.OPTIONAL_PARTS
    table = vr.build_table([str(path)], h, str(tmp_path), parts=parts, log=logs.append)
    use = [l for l in logs if l.startswith('use ')]
    assert len(use) == 1, logs
    # the denominator, which the old line did not carry at all
    assert f'1 of {len(parts)} cell parts' in use[0], use[0]
    # and the two reasons, held apart
    assert 'infer x1 (not usable' in use[0], use[0]
    assert 'stepfull x1 (not run)' in use[0], use[0]
    assert table['absent_parts']['infer'] == {
        'not usable (moved, refused or skipped)': 1}
    assert table['absent_parts']['stepfull'] == {'not run': 1}
    assert 'train' not in table['absent_parts']
    assert any(l.startswith('absent stepfull: 1 cell parts over 1 admitted columns')
               for l in logs), logs


def test_a_whole_column_reports_no_absence_at_all(tmp_path, monkeypatch):
    """The control for the test above. A line that always says `absent` says
    nothing, so a column carrying every part must print none of it."""
    import json
    h = _absence_harness()
    fingerprint = h._h(np.zeros(1))
    cell = dict(verdict='STABLE', hashes=['a' * 16] * 2)
    for part in ('infer', 'model', 'batch', 'stepfull') + vr.OPTIONAL_PARTS:
        cell[part] = ['b' * 16] * 2
        cell[part + '_verdict'] = 'STABLE'
    record = dict(mode='identical', commit='a' * 40, vendor='cpu-test',
                  fixtures={'base': dict(X=fingerprint, y_clf=fingerprint, y_reg=fingerprint)},
                  heldout={'base': dict(X=fingerprint)}, cells={'x/base': cell})
    for part in vr.OPTIONAL_PARTS:
        record[part + '_protocol'] = (h._rlpair_protocol() if part == 'rlpair'
                                      else h._part_protocol(part, h.BATCH_ALONE))
    record['stepfull_protocol'] = h._part_protocol('stepfull', h.BATCH_ALONE)
    record['batch_protocol'] = dict(alone=16, split=[1, 7, 'n'], prefix='1,7,full-1', enabled=True)
    path = tmp_path / 'cpu.json'
    path.write_text(json.dumps(record))
    monkeypatch.setattr(vr, '_commit_time', lambda *args: 1)
    logs = []
    parts = vr.PARTS + vr.OPTIONAL_PARTS
    table = vr.build_table([str(path)], h, str(tmp_path), parts=parts, log=logs.append)
    use = [l for l in logs if l.startswith('use ')][0]
    assert f'{len(parts)} of {len(parts)} cell parts' in use, use
    assert 'absent' not in use, use
    assert table['absent_parts'] == {}
