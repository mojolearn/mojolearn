# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`_verify_reference.admit` and the DIRECTORY a column happens to sit in.

    cd python && python3 -m mojolearn.tests.test_verify_reference_admit

THE TRAP (found 2026-09-16, lane/sabotage-evidence). `admit` matched every
token of `_EXCLUDED_NAME_TOKENS` against the WHOLE lowercased path, so a
perfectly clean column was refused because of the name of a directory above
it. A negative control is recorded BESIDE the clean column it is a control
for, in one record directory, and naming that directory after the thing it
records is the obvious thing to do. Two committed clean columns were being
silently discarded for their neighbor's sin:

    2026-09-15_metrics-sabotage-coverage/cpu-prod.json
    2026-09-15_ties-sabotage/x86-runpod/cpu-x86.json

A silent refusal is the bad failure here. The column is not reported as
rejected; it simply stops feeding the shipped reference table, and the next
person to put a clean column in a sensibly named directory loses it the same
way with no error to read.

WHY THE FIX IS NARROW. `partial`, `probe`, `unfixed` and `post-merge-smoke`
are MEANINGFUL as directory markers: `2026-09-14_kmeans-sqrt-fix/unfixed/`
holds columns taken with the bug still present, and admitting those would feed
known-wrong hashes into the table. Only `sabotage` needs to match the file
name alone, because a sabotage build ALSO says so in its own metadata
(`host.families[*].sabotage`, and the `*_sabotage` harness flags), which is
the stronger check and the one that still refuses it. Measured over the 495
committed columns: 220 admitted before, 222 after, nothing newly refused, and
no column carrying a sabotage signal admitted.

The last three tests are the blast-radius guard. They are the reason this fix
is narrow rather than "match the basename", which would have admitted eight
`unfixed/` and `probe/` columns.
"""
import sys

import pytest

from mojolearn import _verify_reference as vref

RECORDS = "bench/results/identity_break"


def clean_column():
    """The smallest column `admit` accepts: identical mode, a real commit, one
    device, default fixtures, no sabotage signal of any kind."""
    return dict(
        cells={"ols/base": {"verdict": "STABLE", "hashes": ["a" * 16]}},
        mode="identical",
        commit="1334beeedd92",
        vendor="cpu-amd-epyc-9754-128-core-processor",
        package={},
        host={"families": {"_mojolearn_estimators_host": {"column": "cpu", "sabotage": False}}},
    )


def full_part_column():
    column = clean_column()
    column.update(parts_collected=['batch', 'batchgrad', 'batchscale', 'ragged', 'rlpair', 'stepfull'],
                  partial_column=False, parts_omitted=[], repeats=1)
    cell = column['cells']['ols/base']
    for part in ('infer', 'model', 'batch', 'batchgrad', 'batchscale', 'ragged', 'stepfull'):
        cell[part] = ['n/a:not-applicable']
        cell[part + '_verdict'] = 'N/A'
    return column


@pytest.mark.parametrize('part', ['infer', 'model', 'batch', 'stepfull', 'batchgrad', 'batchscale', 'ragged'])
@pytest.mark.parametrize('field', ['values', 'verdict'])
def test_modern_full_column_cannot_silently_drop_a_declared_part(part, field):
    column = full_part_column()
    del column['cells']['ols/base'][part if field == 'values' else part + '_verdict']
    assert vref.admit(column, 'clean.json') == f'incomplete declared part: ols/base {part}'


def test_modern_full_column_preserves_explicit_na_and_failed_probes():
    column = full_part_column()
    assert vref.admit(column, 'clean.json') is None
    column['cells']['ols/base'].update(model=[None], model_verdict='REFUSED', probe_error='save failed')
    assert vref.admit(column, 'clean.json') is None
    assert vref._part_value(column['cells']['ols/base'], 'model') is None


def test_modern_full_column_preserves_failed_training_evidence():
    column = full_part_column()
    column['cells']['ols/base'] = dict(verdict='REFUSED', error='fit failed', hashes=[])
    assert vref.admit(column, 'clean.json') is None


def test_modern_full_column_cannot_shrink_declared_parts_or_repeats():
    column = full_part_column()
    column['parts_collected'].remove('batch')
    assert vref.admit(column, 'clean.json') == 'incomplete full-part declaration'
    column = full_part_column()
    column['repeats'] = 2
    assert 'incomplete declared repeats' in vref.admit(column, 'clean.json')


# ------------------------------------------------- the trap this test exists for

def test_clean_column_in_a_sabotage_named_directory_is_admitted():
    """FAILS before the fix. This is the whole point of the file."""
    path = f"{RECORDS}/2026-09-15_ties-sabotage/x86-runpod/cpu-x86.json"
    assert vref.admit(clean_column(), path) is None, (
        "a clean column was refused for the name of a directory above it"
    )


def test_clean_column_in_a_sabotage_coverage_directory_is_admitted():
    path = f"{RECORDS}/2026-09-15_metrics-sabotage-coverage/cpu-prod.json"
    assert vref.admit(clean_column(), path) is None


# ------------------------------------------- a sabotage column is STILL refused

def test_sabotage_column_refused_by_its_own_file_name():
    path = f"{RECORDS}/2026-09-15_kmeans-transform/cpu-apple-m4.host-sabotage.json"
    assert vref.admit(clean_column(), path) is not None


def test_sabotage_column_refused_by_its_binding_read_back():
    """The metadata path, which does not care what anything is named. This is
    what `forest_host_sabotage()` was failing to report for the CTR arm."""
    j = clean_column()
    j["host"]["families"]["_mojolearn_forest_host"] = {"column": "cpu", "sabotage": True}
    assert vref.admit(j, f"{RECORDS}/2026-09-16_negative-controls/cpu.json") is not None


def test_sabotage_column_refused_by_its_harness_flag():
    j = clean_column()
    j["batch_sabotage"] = True
    assert vref.admit(j, f"{RECORDS}/2026-09-16_negative-controls/cpu.json") is not None


# --------------------------------------------------------- the blast-radius guard

def test_unfixed_directory_is_still_refused():
    """`unfixed/` holds columns taken with the bug present. Admitting them
    would feed known-wrong hashes into the shipped table."""
    path = f"{RECORDS}/2026-09-14_kmeans-sqrt-fix/unfixed/apple-m4.json"
    assert vref.admit(clean_column(), path) is not None


def test_probe_directory_is_still_refused():
    path = "bench/results/e1g/2026-09-14_142050-amd-probe/remote/mamba2_probe/identity_42.json"
    assert vref.admit(clean_column(), path) is not None


def test_documented_metal_incident_is_quarantined_without_hiding_clean_gp_records():
    import json
    from pathlib import Path
    incident = Path(__file__).resolve().parents[3] / RECORDS / "2026-09-15_gp-sample-y" / "metal-transient"
    for path in incident.glob("*.json"):
        column = json.loads(path.read_text())
        assert "quarantined Metal incident" in vref.admit(column, str(path))
    assert (incident / "apple-m4.merged-rerun1.json").is_file()
    assert vref.admit(clean_column(), str(incident.parent / "apple-m4.json")) is None
    assert vref.admit(clean_column(), "other-record/metal-transient/apple-m4.json") is None


def test_partial_and_smoke_directories_are_still_refused():
    for token in ("partial", "post-merge-smoke"):
        path = f"{RECORDS}/2026-09-14_{token}-run/apple-m4.json"
        assert vref.admit(clean_column(), path) is not None, token


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


@pytest.mark.parametrize('values,verdict', [
    (['a'*16], 'STABLE'), (['a'*16,'b'*16], 'STABLE'),
    (['ERROR: bad']*2, 'STABLE'), (['n/a:skipped']*2, 'N/A'),
    (['a'*16]*2, 'N/A'), (['n/a:no-check']*2, 'STABLE'),
    ('a'*16, 'STABLE'), ([None,None], 'STABLE')])
def test_reference_requires_consistent_repeated_typed_values(values, verdict):
    assert vref._part_value(dict(batch=values, batch_verdict=verdict), 'batch', min_repeats=2) is None


def test_reference_accepts_real_repeats_and_explicit_nonapplicability():
    assert vref._part_value(dict(hashes=['a'*16]*2, verdict='STABLE'), 'train') == 'a'*16
    assert vref._part_value(dict(batch=['n/a:global-reduction']*2, batch_verdict='N/A'), 'batch') == 'n/a:global-reduction'


@pytest.mark.parametrize('broken', ['none','one_repeat','missing_input','missing_heldout','batch_protocol','decode_protocol'])
def test_builder_requires_witnesses_repeats_and_standard_property_protocols(tmp_path, monkeypatch, broken):
    import json, types
    h=types.SimpleNamespace(LANES={'x':None}, FIXTURES=['base'], LANE_REVISIONS={}, __file__=__file__,
        _h=lambda x:'f'*16, fixture=lambda f:(0,0,0), heldout=lambda f:0,
        BATCH_ALONE=16,BATCH_SPLIT=(3,7),_part_protocol=lambda part,alone:dict(length=32))
    c=dict(verdict='STABLE',hashes=['a'*16]*2)
    for part in ('infer','batch','stepfull'):
        c[part]=['b'*16]*2;c[part+'_verdict']='STABLE'
    j=dict(mode='identical',commit='a'*40,vendor='cpu-test',cells={'x/base':c},
        fixtures={'base':dict(X='f'*16,y_clf='f'*16,y_reg='f'*16)},heldout={'base':dict(X='f'*16)},
        batch_protocol=dict(alone=16,split=[3,7,'n'],prefix='1,7,full-1',enabled=True),
        stepfull_protocol=dict(length=32))
    if broken=='one_repeat':
        for part in ('hashes','infer','batch','stepfull'):c[part]=c[part][:1]
    elif broken=='missing_input':j['fixtures']={}
    elif broken=='missing_heldout':j['heldout']={}
    elif broken=='batch_protocol':j['batch_protocol']['alone']=1
    elif broken=='decode_protocol':j.pop('stepfull_protocol')
    path=tmp_path/'cpu.json';path.write_text(json.dumps(j))
    monkeypatch.setattr(vref,'_commit_time',lambda *a:1)
    table=vref.build_table([str(path)],h,str(tmp_path))
    parts=set(table['cells'].get('x/base',{}))
    expected={'none':{'train','infer','batch','stepfull'},'one_repeat':set(),'missing_input':set(),
        'missing_heldout':{'train'},'batch_protocol':{'train','infer','stepfull'},
        'decode_protocol':{'train','infer','batch'}}
    assert parts==expected[broken]
    assert table['admission_policy']==dict(vref.ADMISSION_POLICY, cpu_host_only_witnesses=1)


def scoped_tables():
    import copy
    base = dict(format=vref.FORMAT, fixtures={'base': {'X': 'x'}}, heldout={'base': {'X': 'h'}},
                records=[{'class': 'cpu', 'commit': 'a' * 40}], harness_sha256='old',
                lane_revisions={'old': 'v1'},
                cells={'old/base': {'train': {'ref': 'a' * 16, 'cols': {'cpu': 0}}}})
    candidate = copy.deepcopy(base)
    candidate.update(admission_policy=dict(min_repeats=2, input_witness_required=True,
                                          property_protocol_required=True),
                     harness_sha256='new', lane_revisions={'new': 'v2'})
    candidate['cells'] = {'new/base': {part: {'ref': 'b' * 16, 'cols': {'cpu': 0}}
                                     for part in ('train', 'infer', 'model', 'batch')}}
    return base, candidate


def test_scoped_admission_preserves_unselected_cells_and_legacy_policy():
    import copy
    base, candidate = scoped_tables()
    saved = copy.deepcopy(base)
    merged = vref.merge_reference_lanes(base, candidate, ['new'])
    assert base == saved
    assert merged['cells']['old/base'] == base['cells']['old/base']
    assert merged['records'][0] == base['records'][0]
    assert merged['cells']['new/base']['train']['cols'] == {'cpu': 1}
    assert 'admission_policy' not in merged
    assert merged['harness_sha256'] == 'old'
    assert merged['lane_revisions'] == {'old': 'v1', 'new': 'v2'}
    assert merged['lane_admission']['new']['policy']['min_repeats'] == 2


@pytest.mark.parametrize('failure', ['legacy', 'fixture', 'heldout', 'missing', 'conflict', 'lost-part'])
def test_scoped_admission_refuses_incomplete_or_incomparable_evidence(failure):
    base, candidate = scoped_tables()
    if failure == 'legacy':
        candidate.pop('admission_policy')
    elif failure in ('fixture', 'heldout'):
        candidate['fixtures' if failure == 'fixture' else 'heldout'] = {}
    elif failure == 'missing':
        candidate['cells']['new/base'].pop('model')
    elif failure == 'conflict':
        candidate['cells']['new/base']['train']['conflict'] = True
    elif failure == 'lost-part':
        base['cells']['new/base'] = {'stepfull': {'ref': 'c' * 16, 'cols': {'cpu': 0}}}
    with pytest.raises(vref.TableError):
        vref.merge_reference_lanes(base, candidate, ['new'])


def test_incomplete_checkpoint_is_refused_even_with_a_clean_filename():
    column = clean_column()
    column['complete'] = False
    assert vref.admit(column, RECORDS + '/new/cpu-clean.json') == 'incomplete identity_break checkpoint'
    column['complete'] = True
    assert vref.admit(column, RECORDS + '/new/cpu-clean.json') is None


# ------------------------------------------ the partial column (2026-09-20)
# `complete` is a RUN-LEVEL flag and never inspected which parts a column
# carries, so a four-part column and a nine-part column were admitted here
# identically. The harness now stamps a narrowed run, and these watch the
# refusal FIRE, watch a whole column still pass, and watch the historical
# corpus stay admissible (the key is absent there, which must read as "not
# narrowed", not as "unknown").

def test_a_partial_column_is_refused_and_names_the_parts_it_left_out():
    column = clean_column()
    assert vref.admit(column, RECORDS + '/new/cpu-clean.json') is None
    column['partial_column'] = True
    column['parts_omitted'] = ['batchscale', 'stepfull']
    why = vref.admit(column, RECORDS + '/new/cpu-clean.json')
    assert why == 'partial column: the run left out batchscale, stepfull'
    # and it is a DIFFERENT refusal from the incomplete one, because they are
    # different faults: one run stopped early, the other never asked
    column['complete'] = False
    assert vref.admit(column, RECORDS + '/new/cpu-clean.json') == 'incomplete identity_break checkpoint'


def test_a_partial_column_that_did_not_name_its_parts_is_still_refused():
    column = clean_column()
    column['partial_column'] = True
    assert vref.admit(column, RECORDS + '/new/cpu-clean.json') == (
        'partial column: the run left out parts it did not name')


@pytest.mark.parametrize('value', [False, None])
def test_a_column_that_left_nothing_out_is_admitted(value):
    column = clean_column()
    column['partial_column'] = value
    column['parts_omitted'] = []
    assert vref.admit(column, RECORDS + '/new/cpu-clean.json') is None


def test_the_key_being_absent_reads_as_not_narrowed():
    """Every column committed before 2026-09-20 lacks `partial_column`.
    Refusing those would discard the corpus over a question they were never
    asked, so absence must read falsey and admit exactly as before."""
    column = clean_column()
    assert 'partial_column' not in column
    assert vref.admit(column, RECORDS + '/new/cpu-clean.json') is None


@pytest.mark.parametrize('second,admitted', [('agrees', True), ('differs', False), ('absent', False)])
def test_one_fit_on_nvidia_and_amd_is_a_reference(tmp_path, monkeypatch, second, admitted):
    """EVERY CELL IS FITTED ONCE. One fit is a value; the same hash from a
    second device class is what makes it a reference. One class alone, or two
    classes that disagree, give no reference."""
    import json, types
    h=types.SimpleNamespace(LANES={'x':None}, FIXTURES=['base'], LANE_REVISIONS={}, __file__=__file__,
        _h=lambda x:'f'*16, fixture=lambda f:(0,0,0), heldout=lambda f:0,
        BATCH_ALONE=16,BATCH_SPLIT=(3,7),_part_protocol=lambda part,alone:dict(length=32))
    def column(vendor, value):
        return dict(mode='identical',commit='a'*40,vendor=vendor,
            cells={'x/base':dict(verdict='STABLE',hashes=[value])},
            fixtures={'base':dict(X='f'*16,y_clf='f'*16,y_reg='f'*16)},heldout={'base':dict(X='f'*16)})
    paths=[tmp_path/'amd.json']
    paths[0].write_text(json.dumps(column('amd-test','a'*16)))
    if second!='absent':
        paths.append(tmp_path/'nvidia.json')
        paths[1].write_text(json.dumps(column('nvidia-test','a'*16 if second=='agrees' else 'b'*16)))
    monkeypatch.setattr(vref,'_commit_time',lambda *a:1)
    table=vref.build_table([str(p) for p in paths],h,str(tmp_path),parts=('train',))
    ref=table['cells'].get('x/base',{}).get('train',{}).get('ref')
    assert (ref=='a'*16) is admitted
    if admitted:
        assert set(table['cells']['x/base']['train']['cols'])=={'amd','nvidia'}


def test_scoped_admission_accepts_the_two_witness_policy():
    base, candidate = scoped_tables()
    candidate['admission_policy'] = dict(vref.ADMISSION_POLICY)
    merged = vref.merge_reference_lanes(base, candidate, ['new'])
    assert merged['lane_admission']['new']['policy'] == vref.ADMISSION_POLICY


def backend_witness_column(backend='hip', vendor='x86_64'):
    j = clean_column()
    j.pop('host')
    j.update(vendor=vendor, complete=True, fixtures={'base': {'X': 'x'}}, heldout={'base': {'X': 'h'}})
    j['resume_signature'] = dict(schema=1, source_sha256='a'*64, environment_sha256='b'*64,
        options=dict(require_backend=backend, require_cpu=backend == 'cpu', vendor=None),
        provenance={k:j[k] for k in ('commit', 'mode', 'vendor', 'fixtures', 'heldout')})
    return j


@pytest.mark.parametrize('backend,vendor,expected', [('metal','arm64','apple'),
    ('hip','x86_64','amd'), ('cuda','x86_64','nvidia'), ('cpu','cpu-amd-epyc','cpu')])
def test_enforced_backend_witness_recovers_architecture_labels_without_mutation(backend, vendor, expected):
    import copy
    j = backend_witness_column(backend, vendor)
    before = copy.deepcopy(j)
    assert vref.record_device_class(j, 'gmm.gpu.json') == (expected, None)
    assert vref.admit(j, 'gmm.gpu.json') is None
    assert j == before


@pytest.mark.parametrize('broken', ['cpu_label','cpu_flag','host','vendor_option','commit',
    'fixtures','digest','schema','incomplete','unknown_backend','unknown_label'])
def test_backend_witness_refuses_contradictory_or_unbound_metadata(broken):
    j = backend_witness_column()
    sig = j['resume_signature']
    if broken == 'cpu_label':
        j['vendor'] = sig['provenance']['vendor'] = 'cpu-amd-epyc'
    elif broken == 'cpu_flag': sig['options']['require_cpu'] = True
    elif broken == 'host': j['host'] = {'families': {'example': {}}}
    elif broken == 'vendor_option': sig['options']['vendor'] = 'nvidia-h100'
    elif broken == 'commit': sig['provenance']['commit'] = '0'*40
    elif broken == 'fixtures': sig['provenance']['fixtures'] = {}
    elif broken == 'digest': sig['source_sha256'] = ''
    elif broken == 'schema': sig['schema'] = 999
    elif broken == 'incomplete': j['complete'] = False
    elif broken == 'unknown_backend': sig['options']['require_backend'] = 'mystery'
    elif broken == 'unknown_label': j['vendor'] = sig['provenance']['vendor'] = 'mystery'
    cls, why = vref.record_device_class(j, 'gmm.gpu.json')
    assert cls is None and why
    assert vref.admit(j, 'gmm.gpu.json') is not None


def test_architecture_without_backend_witness_remains_unclassified():
    j = backend_witness_column()
    del j['resume_signature']
    assert vref.record_device_class(j, 'gmm.gpu.json') == (None, None)
    assert vref.record_device_class(clean_column(), 'cpu.json') == ('cpu', None)


@pytest.mark.parametrize('backend,suffix', [('hip','gpu'), ('cpu','cpudefault'), ('cpu','cpu3')])
def test_registered_partial_fit_lane_filename_is_not_a_partial_run(backend, suffix):
    lane = 'x-prep-nb-partial'
    j = backend_witness_column(backend, 'cpu-test' if backend == 'cpu' else 'x86_64')
    j['cells'] = {lane+'/base': j['cells']['ols/base']}
    j['resume_signature']['options']['lanes'] = lane
    path = f'clean/{lane}.{suffix}.json'
    assert vref.admit(j, path, known_lanes={lane}) is None
    assert vref.admit(j, path) is not None, 'unregistered names retain refusal'


@pytest.mark.parametrize('broken', ['partial_flag','partial_dir','probe_dir','unfixed_dir',
    'sabotage_flag','unknown_lane','wrong_scope','extra_lane','wrong_suffix','no_witness'])
def test_partial_fit_name_exception_cannot_hide_partial_or_bad_evidence(broken):
    lane = 'x-cluster-minibatch-partial'
    j = backend_witness_column()
    j['cells'] = {lane+'/base': j['cells']['ols/base']}
    j['resume_signature']['options']['lanes'] = lane
    path = f'clean/{lane}.gpu.json'
    known = {lane}
    if broken == 'partial_flag': j['partial_column'] = True
    elif broken.endswith('_dir'): path = broken[:-4]+'/'+path
    elif broken == 'sabotage_flag': j['batch_sabotage'] = True
    elif broken == 'unknown_lane': known = set()
    elif broken == 'wrong_scope': j['resume_signature']['options']['lanes'] = 'ols'
    elif broken == 'extra_lane': j['cells']['ols/base'] = j['cells'][lane+'/base']
    elif broken == 'wrong_suffix': path = f'clean/{lane}.partial.gpu.json'
    elif broken == 'no_witness': del j['resume_signature']
    assert vref.admit(j, path, known_lanes=known) is not None


# THE PTX COLUMN (Andrew 2026-10-10: PTX is a normal target; no flag). A record
# labelled nvidia-ptx-<gpu>-sm80 is the `ptx` device class; it is judged against
# the Apple == NVIDIA == AMD reference and never decides or corroborates one.

def test_a_ptx_label_is_the_ptx_device_class():
    assert vref.PTX_CLASS in vref.CLASSES
    assert vref.device_class('nvidia-ptx-l40s-sm80', 'x.json') == 'ptx'
    assert vref.device_class('x86_64', 'nvidia-ptx-l40s-sm80.identical.json') == 'ptx'
    assert vref.device_class('nvidia-l40s-sm_89', 'x.json') == 'nvidia'


def test_a_cuda_backend_record_labelled_ptx_is_the_ptx_column():
    j = backend_witness_column('cuda', 'nvidia-ptx-l40s-sm80')
    assert vref.record_device_class(j, 'nvidia-ptx-l40s-sm80.identical.json') == ('ptx', None)
    assert vref.admit(j, 'nvidia-ptx-l40s-sm80.identical.json') is None
    other = backend_witness_column('hip', 'nvidia-ptx-l40s-sm80')
    cls, why = vref.record_device_class(other, 'nvidia-ptx-l40s-sm80.identical.json')
    assert cls is None and why


def _ptx_table(tmp_path, monkeypatch, columns):
    """build_table over one train cell; `columns` is [(file stem, vendor label, value)]."""
    import json, types
    h=types.SimpleNamespace(LANES={'x':None}, FIXTURES=['base'], LANE_REVISIONS={}, __file__=__file__,
        _h=lambda x:'f'*16, fixture=lambda f:(0,0,0), heldout=lambda f:0,
        BATCH_ALONE=16,BATCH_SPLIT=(3,7),_part_protocol=lambda part,alone:dict(length=32))
    paths=[]
    for stem, vendor, value in columns:
        path=tmp_path/f'{stem}.json'
        path.write_text(json.dumps(dict(mode='identical',commit='a'*40,vendor=vendor,
            cells={'x/base':dict(verdict='STABLE',hashes=[value])},
            fixtures={'base':dict(X='f'*16,y_clf='f'*16,y_reg='f'*16)},heldout={'base':dict(X='f'*16)})))
        paths.append(path)
    monkeypatch.setattr(vref,'_commit_time',lambda *a:1)
    return vref.build_table([str(p) for p in paths],h,str(tmp_path),parts=('train',))


def test_an_agreeing_ptx_column_is_listed_and_never_decides(tmp_path, monkeypatch):
    # The PTX file sorts last, so it would be "newest" if it could decide.
    table=_ptx_table(tmp_path, monkeypatch, [('amd','amd-test','a'*16), ('nvidia','nvidia-test','a'*16),
                                             ('nvidia-ptx-zz','nvidia-ptx-test','a'*16)])
    ent=table['cells']['x/base']['train']
    assert ent['ref']=='a'*16 and not ent.get('conflict')
    assert set(ent['cols'])=={'amd','nvidia','ptx'} and isinstance(ent['cols']['ptx'], int)
    assert table['ptx_divergent']==[]


def test_a_differing_ptx_value_is_listed_and_the_reference_stays(tmp_path, monkeypatch):
    table=_ptx_table(tmp_path, monkeypatch, [('amd','amd-test','a'*16), ('nvidia','nvidia-test','a'*16),
                                             ('nvidia-ptx-zz','nvidia-ptx-test','b'*16)])
    ent=table['cells']['x/base']['train']
    assert ent['ref']=='a'*16 and not ent.get('conflict')
    assert ent['cols']['ptx'][1]=='b'*16
    assert isinstance(ent['cols']['amd'], int) and isinstance(ent['cols']['nvidia'], int)
    assert [(r['cell'], r['part'], r['ptx'], r['reference']) for r in table['ptx_divergent']] == \
        [('x/base', 'train', 'b'*16, 'a'*16)]


def test_apple_difference_cannot_be_admitted_as_an_alternative_reference(tmp_path, monkeypatch):
    table = _ptx_table(tmp_path, monkeypatch, [
        ('amd', 'amd-test', 'a' * 16), ('nvidia', 'nvidia-test', 'a' * 16),
        ('zz-apple', 'apple-test', 'b' * 16)])
    # The newest Apple value has no independent corroboration. It cannot
    # become a passing Apple-only reference, even when NVIDIA and AMD agree.
    assert not table['cells']


@pytest.mark.parametrize('other', ['apple', 'cpu'])
def test_other_native_class_can_corroborate_single_nvidia_sample(tmp_path, monkeypatch, other):
    table = _ptx_table(tmp_path, monkeypatch, [
        ('nvidia', 'nvidia-test', 'a' * 16), (other, other + '-test', 'a' * 16)])
    assert table['cells']['x/base']['train']['ref'] == 'a' * 16


@pytest.mark.parametrize('columns', [
    [('nvidia-ptx-zz','nvidia-ptx-test','a'*16)],
    [('nvidia','nvidia-test','a'*16), ('nvidia-ptx-zz','nvidia-ptx-test','a'*16)],
    [('amd','amd-test','a'*16), ('nvidia-ptx-zz','nvidia-ptx-test','a'*16)]])
def test_a_ptx_column_never_corroborates_a_reference(tmp_path, monkeypatch, columns):
    table=_ptx_table(tmp_path, monkeypatch, columns)
    assert 'x/base' not in table['cells']
    assert table['ptx_divergent']==[]


# ------------------------------------------ the cpu column decides (2026-10-24)
# Andrew 2026-10-24: "the cpu column should be REQUIRED too." The builder keeps
# every cpu value it reads, agreeing (a bare index) or not (a pair, or a
# conflict at one commit); tools/identity_admission_guard.py then refuses any
# selected numerical part whose cpu value is not the Apple == NVIDIA == AMD
# reference. These watch the builder hand the guard each case.

def _admission_guard():
    import importlib.util, pathlib
    path = pathlib.Path(__file__).resolve().parents[3] / 'tools/identity_admission_guard.py'
    spec = importlib.util.spec_from_file_location('identity_admission_guard_for_vref', path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    return guard


def _cpu_table(tmp_path, monkeypatch, cpu_value, cpu_commit):
    import json, types
    h = types.SimpleNamespace(LANES={'x': None}, FIXTURES=['base'], LANE_REVISIONS={}, __file__=__file__,
        _h=lambda x: 'f' * 16, fixture=lambda f: (0, 0, 0), heldout=lambda f: 0,
        BATCH_ALONE=16, BATCH_SPLIT=(3, 7), _part_protocol=lambda part, alone: dict(length=32))
    paths = []
    for stem, vendor, value, commit in [('amd', 'amd-test', 'a' * 16, 'a' * 40),
                                        ('apple', 'apple-test', 'a' * 16, 'a' * 40),
                                        ('nvidia', 'nvidia-test', 'a' * 16, 'a' * 40),
                                        ('cpu', 'cpu-test', cpu_value, cpu_commit)]:
        path = tmp_path / f'{stem}.json'
        path.write_text(json.dumps(dict(mode='identical', commit=commit, vendor=vendor,
            cells={'x/base': dict(verdict='STABLE', hashes=[value])},
            fixtures={'base': dict(X='f' * 16, y_clf='f' * 16, y_reg='f' * 16)},
            heldout={'base': dict(X='f' * 16)})))
        paths.append(path)
    monkeypatch.setattr(vref, '_commit_time', lambda root, commit, cache: 2 if commit == 'a' * 40 else 1)
    return vref.build_table([str(p) for p in paths], h, str(tmp_path), parts=('train',))


def _guard_problems(table):
    import copy
    guard = _admission_guard()
    base = copy.deepcopy(table)
    base['cells'] = {}
    guard.REQUIRED_PARTS = frozenset(('train',))
    return guard.validate(base, table, ['x'], (frozenset(), ()))['problems']


@pytest.mark.parametrize('case', ['agree', 'same-commit', 'older-commit'])
def test_the_cpu_column_must_equal_the_gpu_reference(tmp_path, monkeypatch, case):
    cpu_value = 'a' * 16 if case == 'agree' else 'b' * 16
    cpu_commit = 'b' * 40 if case == 'older-commit' else 'a' * 40
    table = _cpu_table(tmp_path, monkeypatch, cpu_value, cpu_commit)
    ent = table['cells']['x/base']['train']
    problems = _guard_problems(table)
    if case == 'agree':
        assert ent['ref'] == 'a' * 16 and isinstance(ent['cols']['cpu'], int)
        assert problems == []
    elif case == 'same-commit':
        # a disagreement inside one record: no reference at all
        assert ent.get('conflict') and ent['ref'] is None
        assert problems == ['x/base/train: missing or conflicted reference']
    else:
        # an older cpu value stays VISIBLE as a pair, and the guard refuses it
        assert ent['ref'] == 'a' * 16 and ent['cols']['cpu'][1] == 'b' * 16
        assert problems == [f"x/base/train: cpu column older than the GPU reference "
                            f"(cpu@{'b' * 9} vs gpu@{'a' * 9}); re-record the cpu column"]


def schema2_backend_column(backend='cuda', vendor='nvidia-ptx-a40-sm80'):
    import hashlib, json
    column = backend_witness_column(backend, vendor)
    signature = column['resume_signature']
    signature['schema'] = 2
    signature['environment_digests'] = {'MOJOLEARN_VENDOR': hashlib.sha256(backend.encode()).hexdigest(),
                                        'OMP_NUM_THREADS': hashlib.sha256(b'4').hexdigest()}
    signature['environment_sha256'] = hashlib.sha256(
        json.dumps(signature['environment_digests'], sort_keys=True).encode()).hexdigest()
    return column


@pytest.mark.parametrize('backend,vendor,expected', [('cuda','nvidia-ptx-a40-sm80','ptx'),
    ('cuda','nvidia-a40','nvidia'), ('metal','arm64','apple'), ('hip','amd-mi300x','amd'),
    ('cpu','cpu-amd-epyc','cpu')])
def test_schema2_backend_witness_preserves_device_class_and_raw_record(backend, vendor, expected):
    import copy
    column = schema2_backend_column(backend, vendor)
    before = copy.deepcopy(column)
    assert vref.record_device_class(column, 'column.identical.json') == (expected, None)
    assert vref.admit(column, 'column.identical.json') is None
    assert column == before


@pytest.mark.parametrize('broken', ['missing','not_dict','raw_value','bad_key','lease','aggregate',
    'incomplete','vendor','commit','schema_type','future_schema'])
def test_schema2_backend_witness_refuses_corruption(broken):
    column = schema2_backend_column()
    signature = column['resume_signature']
    if broken == 'missing': signature.pop('environment_digests')
    elif broken == 'not_dict': signature['environment_digests'] = []
    elif broken == 'raw_value': signature['environment_digests']['OMP_NUM_THREADS'] = '4'
    elif broken == 'bad_key': signature['environment_digests']['UNBOUND'] = 'a'*64
    elif broken == 'lease': signature['environment_digests']['MOJOLEARN_SLOT_TOKEN'] = 'a'*64
    elif broken == 'aggregate': signature['environment_sha256'] = '0'*64
    elif broken == 'incomplete': column['complete'] = False
    elif broken == 'vendor': signature['options']['vendor'] = 'amd-mi300x'
    elif broken == 'commit': signature['provenance']['commit'] = '0'*40
    elif broken == 'schema_type': signature['schema'] = True
    elif broken == 'future_schema': signature['schema'] = 3
    device, reason = vref.record_device_class(column, 'column.identical.json')
    assert device is None and reason
    assert vref.admit(column, 'column.identical.json') is not None
