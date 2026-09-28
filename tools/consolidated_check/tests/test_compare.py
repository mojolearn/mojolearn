"""Saved evidence comparisons and fault controls; no package/native import."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

DRIVER = Path(__file__).resolve().parents[1] / 'compare.py'
spec = importlib.util.spec_from_file_location('saved_crossvendor', DRIVER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def record(backend, lane='x', value='a'*16, commit='1'*40):
    cell = dict(verdict='STABLE', hashes=[value], infer=[value], infer_verdict='STABLE',
                model=[value], model_verdict='STABLE', reload=[value])
    for part in ('batch', 'batchgrad', 'batchscale', 'ragged', 'stepfull'):
        cell[part] = ['n/a:no-operation']
        cell[part+'_verdict'] = 'N/A'
    j = dict(mode='identical', complete=True, repeats=1, commit=commit,
             vendor={'metal':'apple-test', 'hip':'amd-test', 'cuda':'nvidia-test'}[backend],
             cells={lane+'/base':cell},
             fixtures={'base':dict(X='b'*16,y_clf='c'*16,y_reg='d'*16)},
             heldout={'base':dict(X='e'*16)}, lane_revisions={})
    for part in ('batch', 'batchgrad', 'batchscale', 'ragged', 'stepfull', 'rlpair'):
        j[part+'_protocol'] = dict(enabled=True)
    return j


def write(root, backend, j, lane='x'):
    directory=root/backend
    directory.mkdir(parents=True, exist_ok=True)
    p=directory/(lane+'.gpu.json')
    p.write_text(json.dumps(j))
    return p


def pair(tmp_path):
    for backend in ('metal','hip'):
        write(tmp_path, backend, record(backend))
    return [('metal',tmp_path/'metal'), ('hip',tmp_path/'hip')]


def test_complete_match_records_sha_and_does_not_require_uniform_commits_across_lanes(tmp_path):
    cols=pair(tmp_path)
    for backend in ('metal','hip'):
        write(tmp_path,backend,record(backend,lane='y',commit='2'*40),lane='y')
    out=module.compare({'lanes':['x','y']},cols)
    assert out['verdict']=='AGREE' and out['exit']==0
    assert out['counts']['numeric_matches']==8  # train/infer/model/reload
    assert out['source_commits']==['1'*40,'2'*40]
    assert all(row['sha256']==hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() for row in out['records'])


@pytest.mark.parametrize('part', ['train','infer','model','reload','batch','batchgrad','batchscale','ragged','stepfull','rlpair'])
def test_any_changed_numerical_part_fails(tmp_path,part):
    cols=pair(tmp_path)
    for backend in ('metal','hip'):
        j=record(backend); c=j['cells']['x/base'];field='hashes' if part=='train' else part
        c[field]=['a'*16 if backend=='metal' else 'f'*16]
        if part!='reload':c['verdict' if part=='train' else part+'_verdict']='STABLE'
        write(tmp_path,backend,j)
    out=module.compare({'lanes':['x']},cols)
    assert out['verdict']=='DIFFERENT' and out['counts']['different_parts']>=1


@pytest.mark.parametrize('problem', ['missing_lane','missing_fixture','bad_commit','bad_inputs','no_inputs',
    'refused','undeclared','bad_backend','incomplete_record','bad_protocol','missing_repeat','missing_model'])
def test_missing_or_invalid_evidence_never_passes(tmp_path,problem):
    cols=pair(tmp_path);j=record('hip');c=j['cells']['x/base'];fixtures=('base',)
    if problem=='missing_lane': (tmp_path/'hip/x.gpu.json').unlink()
    elif problem=='missing_fixture':fixtures=('base','odd')
    elif problem=='bad_commit':j['commit']='2'*40
    elif problem=='bad_inputs':j['fixtures']['base']['X']='f'*16
    elif problem=='no_inputs':j['fixtures']={}
    elif problem=='refused':c['verdict']='REFUSED';c['hashes']=[None]
    elif problem=='undeclared':c['batch']=['n/a:UNDECLARED']
    elif problem=='bad_backend':j['vendor']='cpu-test'
    elif problem=='incomplete_record':j['complete']=False
    elif problem=='bad_protocol':j['batch_protocol']['enabled']=False
    elif problem=='missing_repeat':j['repeats']=2
    elif problem=='missing_model':del c['model']
    if problem!='missing_lane':write(tmp_path,'hip',j)
    out=module.compare({'lanes':['x']},cols,fixtures,progress=True)
    assert out['verdict']=='INCOMPLETE' and out['exit']==2


def test_within_column_instability_fails_even_if_other_column_is_stable(tmp_path):
    cols=pair(tmp_path);j=record('hip');j['repeats']=2
    for c in j['cells'].values():
        for key,value in c.items():
            if isinstance(value,list):value.append(value[0])
        c['hashes'][1]='b'*16
    write(tmp_path,'hip',j)
    assert module.compare({'lanes':['x']},cols)['verdict']=='DIFFERENT'


def test_all_na_has_no_numerical_coverage_but_numeric_infer_can_stand_alone(tmp_path):
    cols=pair(tmp_path)
    for backend in ('metal','hip'):
        j=record(backend);c=j['cells']['x/base']
        for part in ('train','infer','model'):
            c['hashes' if part=='train' else part]=['n/a:no-operation']
            c['verdict' if part=='train' else part+'_verdict']='N/A'
        del c['reload'];write(tmp_path,backend,j)
    assert module.compare({'lanes':['x']},cols)['verdict']=='INCOMPLETE'
    for backend in ('metal','hip'):
        p=tmp_path/backend/'x.gpu.json';j=json.loads(p.read_text())
        j['cells']['x/base'].update(infer=['a'*16],infer_verdict='STABLE');write(tmp_path,backend,j)
    assert module.compare({'lanes':['x']},cols)['verdict']=='AGREE'


def test_explicit_overlay_uses_new_commit_only_when_all_columns_match(tmp_path):
    cols=pair(tmp_path);overlay=tmp_path/'overlay'
    write(overlay,'metal',record('metal',commit='2'*40))
    cols.append(('metal',overlay/'metal'))
    assert module.compare({'lanes':['x']},cols)['verdict']=='INCOMPLETE'
    write(overlay,'hip',record('hip',commit='2'*40))
    cols.append(('hip',overlay/'hip'))
    assert module.compare({'lanes':['x']},cols)['verdict']=='AGREE'
    j=record('hip',commit='2'*40);j['complete']=False;write(overlay,'hip',j)
    assert module.compare({'lanes':['x']},cols)['verdict']=='INCOMPLETE'


def test_empty_plan_or_one_backend_cannot_pass(tmp_path):
    cols=pair(tmp_path)
    with pytest.raises(ValueError): module.compare({'lanes':[]},cols)
    with pytest.raises(ValueError): module.compare({'lanes':['x']},cols[:1])
    out=module.compare({'lanes':['absent']},cols)
    assert out['verdict']=='INCOMPLETE' and out['counts']['numeric_matches']==0


def test_cli_uses_full_plan_scope_and_writes_progress_artifact(tmp_path):
    pair(tmp_path);plan=tmp_path/'plan.json';plan.write_text(json.dumps({'lanes':['x','y']}))
    report=tmp_path/'out.json'
    code=module.main(['--plan',str(plan),'--column','metal='+str(tmp_path/'metal'),
        '--column','hip='+str(tmp_path/'hip'),'--progress','--json-out',str(report)])
    assert code==2
    got=json.loads(report.read_text())
    assert got['verdict']=='INCOMPLETE' and got['lane_results']['y']=='INCOMPLETE'
    assert got['plan']['sha256']==hashlib.sha256(plan.read_bytes()).hexdigest()


def test_reload_mismatch_cannot_hide_behind_matching_broken_vendors(tmp_path):
    cols=pair(tmp_path)
    for backend in ('metal','hip'):
        j=record(backend);j['cells']['x/base']['reload']=['f'*16];write(tmp_path,backend,j)
    out=module.compare({'lanes':['x']},cols)
    assert out['verdict']=='DIFFERENT'
    assert any('reloaded model' in (r.get('reason') or '') for r in out['parts'])


def test_different_na_declarations_are_scope_incomplete(tmp_path):
    cols=pair(tmp_path);j=record('hip');j['cells']['x/base']['batch']=['n/a:different-scope'];write(tmp_path,'hip',j)
    assert module.compare({'lanes':['x']},cols)['verdict']=='INCOMPLETE'


@pytest.mark.parametrize('left,right,expected,verdict', [
    (None, None, 'v2', 'INCOMPLETE'),
    ('v1', 'v1', 'v2', 'INCOMPLETE'),
    ('v2', None, None, 'INCOMPLETE'),
    ('v1', 'v2', None, 'INCOMPLETE'),
    ('v2', 'v2', 'v2', 'AGREE'),
    (None, None, None, 'AGREE'),  # explicitly historical plan
])
def test_batch_revision_is_checked_without_invalidating_other_parts(tmp_path,left,right,expected,verdict):
    cols=pair(tmp_path)
    for backend,revision in [('metal',left),('hip',right)]:
        j=record(backend)
        if revision is not None: j['batch_revisions']={'x':revision}
        write(tmp_path,backend,j)
    plan={'lanes':['x']}
    if expected is not None: plan['batch_revisions']={'x':expected}
    out=module.compare(plan,cols)
    assert out['verdict']==verdict
    assert out['counts']['numeric_matches']==4
    batch=next(p for p in out['parts'] if p['part']=='batch')
    assert batch['state']==('NA' if verdict=='AGREE' else 'INCOMPLETE')
    assert out['expected_batch_revisions']==plan.get('batch_revisions',{})


def test_invalid_plan_batch_revision_rejected(tmp_path):
    with pytest.raises(ValueError,match='batch_revisions'):
        module.compare({'lanes':['x'],'batch_revisions':{'x':None}},pair(tmp_path))
