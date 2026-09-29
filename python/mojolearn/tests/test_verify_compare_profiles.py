"""Document comparisons cover combined routine reports without rerunning fits."""
import copy
import json
from types import SimpleNamespace

import pytest
from mojolearn import _verify_all as va


def document(device='apple', *, cross=True):
    report = dict(format=va.COMPARE_INPUT_FORMAT, device=dict(device=device, device_class=device),
                  verdict='VERIFIED', detail='test fixture', bindings=[],
                  verification_contract=dict(harness_sha256='a'*64,
                      fixtures={'base': {'X': 'a'*16}}, heldout={'base': {'X': 'b'*16}},
                      protocols={'batch': {'test': 1}}, batch_revisions={}),
                  cells=[dict(lane='ols', fixture='base', part='train', value='a'*16, state='IDENTICAL')])
    if cross:
        rows = [dict(lane='ols', fixture='base', part=p, gpu='b'*16, cpu='b'*16,
                     agree=True, na=False, seconds=1) for p in ('infer','batch')]
        report['cross_check'] = va._cross_check_result('metal', ['ols'], ['base'], rows, {})
    return report


def test_both_gpu_and_cpu_cross_check_results_are_compared():
    a,b=document(),document('nvidia')
    r=va.compare_documents(a,b)
    assert r['verdict']=='AGREE' and r['agree']==5
    b['cross_check']['cells'][0]['cpu']='c'*16
    # Deliberately leave the old agree/complete/passed summary: hashes win.
    r=va.compare_documents(a,b)
    assert r['exit']==1 and r['differ']==1


def test_two_machines_agreeing_on_the_same_local_divergence_cannot_pass():
    a,b=document(),document('nvidia')
    for d in (a,b):d['cross_check']['cells'][0]['cpu']='c'*16
    r=va.compare_documents(a,b)
    assert r['verdict']=='AGREED ON A DIVERGENT ANSWER'


@pytest.mark.parametrize('fault',['missing','both-missing','failed-child','different-scope'])
def test_missing_or_failed_cross_checks_are_not_silently_dropped(fault):
    a,b=document(),document('nvidia')
    if fault=='missing':b['cross_check']['cells'].pop()
    if fault=='both-missing':
        a['cross_check']['cells'].pop();b['cross_check']['cells'].pop()
    if fault=='failed-child':b['cross_check']['skipped']={'worker':'nonzero exit'}
    if fault=='different-scope':b.pop('cross_check')
    r=va.compare_documents(a,b)
    assert r['verdict']=='INCOMPLETE' and r['exit']==4


def test_duplicate_cross_check_cells_are_malformed():
    a,b=document(),document('nvidia')
    b['cross_check']['cells'].append(copy.deepcopy(b['cross_check']['cells'][0]))
    assert va.compare_documents(a,b)['verdict']=='MALFORMED'


def test_matching_na_cross_check_reasons_are_not_counted_as_matches():
    a,b=document(),document('nvidia')
    for d in (a,b):
        d['cross_check']['cells'][1].update(gpu='n/a:no row axis',cpu='n/a:no row axis',na=True,agree=None)
    r=va.compare_documents(a,b)
    assert r['verdict']=='AGREE' and r['agree']==3 and r['n_a']==2


def test_cross_check_commitments_bind_values_and_scope_but_not_timing_or_order():
    a=document();nonce='0'*32;original=va.commitment_digest(a,nonce)
    b=copy.deepcopy(a);b['cross_check']['cells'][0]['cpu']='c'*16
    assert va.commitment_digest(b,nonce)!=original
    b=copy.deepcopy(a);b['cross_check']['requested_lanes'].append('ridge')
    assert va.commitment_digest(b,nonce)!=original
    b=copy.deepcopy(a);b['cross_check']['cells'].reverse();b['cross_check']['cells'][0]['seconds']=999
    assert va.commitment_digest(b,nonce)==original


def test_portable_models_add_their_nonbase_input_fingerprints_without_running_models(tmp_path):
    path=tmp_path/va.vref.TABLE_DIR/va.MODELS_DIR/va.MODELS_MANIFEST
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'models':[{'fixture':'base'},{'fixture':'denormal'}]}))
    calls=[]
    h=SimpleNamespace(FIXTURES=['base','denormal'],
                      fixture=lambda f:calls.append(('fixture',f)) or ('X','yc','yr'),
                      heldout=lambda f:calls.append(('heldout',f)) or 'held')
    data,held=va.portable_contract_inputs(h,{'base':('X','yc','yr')},{'base':'held'},str(tmp_path))
    assert set(data)==set(held)=={'base','denormal'}
    assert calls==[('fixture','denormal'),('heldout','denormal')]


def test_compare_can_save_its_own_json_report(tmp_path,capsys):
    a,b=tmp_path/'a.json',tmp_path/'b.json';output=tmp_path/'comparison.json'
    a.write_text(json.dumps(document()));b.write_text(json.dumps(document('nvidia')))
    assert va._cmd_compare(SimpleNamespace(compare=[str(a),str(b)],json=True,json_out=str(output)))==0
    assert json.loads(output.read_text())['verdict']=='AGREE'
    assert json.loads(capsys.readouterr().out)['agree']==5


def test_incompatible_batch_does_not_discard_train_and_inference_matches():
    a, b = document(), document('nvidia')
    b['verification_contract']['protocols']['batch'] = {'test': 2}
    r = va.compare_documents(a, b)
    assert (r['verdict'], r['agree'], r['incomparable']) == ('INCOMPLETE', 3, 2)
    assert all(row['part'] == 'batch' for row in r['incomparable_cells'])
    assert '3 shared cell parts match' in va.format_compare(r)


def test_real_mismatch_outranks_unrelated_incompatible_cells():
    a, b = document(), document('nvidia')
    b['verification_contract']['protocols']['batch'] = {'test': 2}
    b['cells'][0]['value'] = 'c' * 16
    r = va.compare_documents(a, b)
    assert (r['verdict'], r['differ'], r['agree'], r['incomparable']) == ('MISMATCH', 1, 2, 2)


def test_different_scopes_keep_matches_and_name_both_missing_sides():
    a, b = document(cross=False), document('nvidia', cross=False)
    a['cells'].append(dict(a['cells'][0], lane='ridge'))
    b['cells'].append(dict(b['cells'][0], lane='lasso'))
    r = va.compare_documents(a, b)
    assert r['verdict'] == 'INCOMPLETE' and r['agree'] == 1
    assert r['only_in_a'] == [['ridge', 'base', 'train']]
    assert r['only_in_b'] == [['lasso', 'base', 'train']]


def test_no_compatible_cells_remains_incomparable():
    a, b = document(), document('nvidia')
    b['verification_contract']['harness_sha256'] = 'c' * 64
    r = va.compare_documents(a, b)
    assert (r['verdict'], r['agree'], r['incomparable']) == ('INCOMPARABLE', 0, 5)
