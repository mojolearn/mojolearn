"""Installed report transport/admission only; no product execution or device fit."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import identity_columns as C
import identity_verify_launch as L
import release_identity_evidence as E
from test_identity_verify_record import example


def setup_report(tmp_path):
    report, h = example(tmp_path)
    report.update(repeats=1, binding_provenance_protocol='loaded-native-v2',
                  selection=dict(profile=dict(name='routine'), include_pending=True), models_checked=0)
    report['device']['platform'] = 'Linux-6.8-x86_64-with-glibc2.39'
    package = tmp_path / 'installed' / 'mojolearn'
    report['bindings'][0]['file'] = str(package / 'native.so')
    files = {'native.so': dict(size=16, sha256='b'*64),
             '_identity_break.py': dict(size=30, sha256=report['harness']['sha256'])}
    options = dict(lanes=['lane'], profile='routine', repeats=1, cpu_threads=8,
                   fixtures=['base'], parts=list(C.vref().PARTS+C.vref().OPTIONAL_PARTS))
    options.update({'all': True, 'include_pending': True, 'no_models': True})
    source = dict(root='/clean/frozen', commit='a'*40, dirty_tracked='')
    command = ['/env/bin/python', '-m', 'mojolearn', 'verify', '--all', '--include-pending', '--no-models', '--lanes', 'lane',
               '--repeats', '1', '--cpu-threads', '8', '--json-out', '/out/verify.json', '--json']
    raw = json.dumps(report).encode()
    receipt = dict(schema=L.SCHEMA, phase='before-execution', status='completed', exit_code=1,
                   source=source, after_source=copy.deepcopy(source), options=options, command=command,
                   environment=dict(L.DEFAULT_POLICY),
                   install=dict(package=str(package), commit='a'*40, files=files, manifest_sha256=L.digest(files)),
                   after_install_manifest_sha256=L.digest(files), report_sha256=hashlib.sha256(raw).hexdigest())
    report_path, launch_path = tmp_path/'verify.json', tmp_path/'run.json'
    report_path.write_bytes(raw); launch_path.write_text(json.dumps(receipt))
    return report, h, receipt, report_path, launch_path


def convert(tmp_path, monkeypatch, damage=None):
    report, h, receipt, report_path, launch_path = setup_report(tmp_path)
    if damage: damage(report, receipt)
    raw = json.dumps(report).encode(); report_path.write_bytes(raw)
    receipt['report_sha256'] = hashlib.sha256(raw).hexdigest()
    launch_path.write_text(json.dumps(receipt))
    monkeypatch.setattr(C, '_load', lambda *args: h)
    target = tmp_path/'nvidia-test.identical.json'
    C.cmd_from_verify(SimpleNamespace(report=report_path, harness=h.__file__, out=target, launch=launch_path))
    return target, json.loads(target.read_text())


def test_converted_report_enters_strict_record_admission_without_refit(tmp_path, monkeypatch):
    path, record = convert(tmp_path, monkeypatch)
    cls, errors = E.record_errors(record, path, E.reference(path), 'a'*40)
    assert cls == 'nvidia' and errors == []
    assert 'resume_signature' not in record
    assert E.requested_lanes(record) == ['lane']
    assert record['cells']['lane/base']['verdict'] == 'STABLE'  # old-reference mismatch retained
    assert record['verifier_observation']['reported_repeats'] == 1
    assert C.validate_verifier_record(record, path)['all'] is True


@pytest.mark.parametrize('damage', ['dirty','changed_source','changed_install','binding','protocol','repeat','profile','command','fixtures','parts','pending','models'])
def test_missing_or_mismatched_producer_witness_is_refused(tmp_path, monkeypatch, damage):
    def break_it(report, receipt):
        if damage == 'dirty': receipt['source']['dirty_tracked']=' M binding.mojo'; receipt['after_source']['dirty_tracked']=' M binding.mojo'
        elif damage == 'changed_source': receipt['after_source']['commit']='f'*40
        elif damage == 'changed_install': receipt['after_install_manifest_sha256']='0'*64
        elif damage == 'binding': report['bindings'][0]['sha256']='e'*64
        elif damage == 'protocol': report.pop('binding_provenance_protocol')
        elif damage == 'repeat': receipt['options']['repeats']=2
        elif damage == 'profile': receipt['options']['profile']='neural-training'
        elif damage == 'command': receipt['command'][4]='--quick'
        elif damage == 'fixtures': receipt['options']['fixtures']=[]
        elif damage == 'parts': receipt['options']['parts'].pop()
        elif damage == 'pending': report['selection']['include_pending']=False
        elif damage == 'models': report['models_checked']=1
    with pytest.raises(ValueError): convert(tmp_path, monkeypatch, break_it)


def test_legacy_conversion_does_not_claim_unobserved_clean_launch(tmp_path, monkeypatch):
    report,h,_,report_path,_=setup_report(tmp_path)
    monkeypatch.setattr(C,'_load',lambda *args:h)
    target=tmp_path/'nvidia-legacy.identical.json'
    C.cmd_from_verify(SimpleNamespace(report=report_path,harness=h.__file__,out=target,launch=None))
    record=json.loads(target.read_text())
    _, errors=E.record_errors(record,target,E.reference(target),'a'*40)
    assert any('bound harness options' in x for x in errors)
    assert any('provenance unavailable' in x for x in errors)


@pytest.mark.parametrize('damage',['receipt','report','options','cell','removed_receipt'])
def test_admission_rechecks_retained_evidence(tmp_path,monkeypatch,damage):
    path,record=convert(tmp_path,monkeypatch)
    obs=record['verifier_observation']
    if damage=='receipt': (tmp_path/obs['launch']).write_text('{}')
    elif damage=='report': (tmp_path/obs['report']).write_bytes(b'bad gzip')
    elif damage=='options': obs['options']['lanes']=['other']
    elif damage=='cell': record['cells']['lane/base']['hashes']=['0'*16]
    elif damage=='removed_receipt': (tmp_path/obs['launch']).unlink()
    _,errors=E.record_errors(record,path,E.reference(path),'a'*40)
    assert any('verifier producer witness invalid' in x for x in errors)


def test_launch_writes_before_execution_and_preserves_nonzero_report(tmp_path,monkeypatch):
    report,h,_,_,_=setup_report(tmp_path)
    root=tmp_path/'source'; (root/'tools').mkdir(parents=True)
    (root/'tools/identity_break.py').write_bytes(Path(h.__file__).read_bytes())
    package=tmp_path/'installed'/'mojolearn'; (package/'identity_columns').mkdir(parents=True)
    (package/'identity_columns/COMMIT').write_text('a'*40)
    (package/'_identity_break.py').write_bytes(Path(h.__file__).read_bytes())
    (package/'native.so').write_bytes(b'native fixture')
    monkeypatch.setattr(L,'locate',lambda *a:package)
    monkeypatch.setattr(L,'source_state',lambda r:dict(root=str(r),commit='a'*40,dirty_tracked=''))
    monkeypatch.setattr(C,'harness',lambda:h)
    monkeypatch.setattr(C,'profile_lanes',lambda *a:['lane'])
    out=tmp_path/'run'
    monkeypatch.setenv('MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32', '1')
    monkeypatch.setenv('MOJOLEARN_EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE', '1')
    def runner(command,**kwargs):
        before=json.loads((out/'verifier-run.json').read_text())
        assert before['phase']=='before-execution' and before['status']=='running'
        assert before['environment'] == L.DEFAULT_POLICY
        assert before['producer']['sha256'] == L.sha(L.__file__)
        assert kwargs['env']['MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32'] == '0'
        assert kwargs['env']['MOJOLEARN_EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE'] == '0'
        assert kwargs['cwd']==out and 'PYTHONPATH' not in kwargs['env']
        (out/'verify.json').write_text(json.dumps(report))
        return SimpleNamespace(returncode=1)
    assert L.execute(root,'/env/python',out,['lane'],'routine',8,runner=runner)==1
    receipt=json.loads((out/'verifier-run.json').read_text())
    assert receipt['status']=='completed' and receipt['exit_code']==1
    assert receipt['install']['manifest_sha256']==receipt['after_install_manifest_sha256']
    with pytest.raises(ValueError,match='output already exists'):
        L.execute(root,'/env/python',out,['lane'],'routine',8,runner=runner)


def test_harness_may_order_the_explicit_lane_set(tmp_path):
    report, _, receipt, _, _ = setup_report(tmp_path)
    receipt['options']['lanes'] = ['lane', 'other']
    receipt['command'][receipt['command'].index('--lanes') + 1] = 'lane,other'
    report['lanes'] = ['other', 'lane']
    raw = json.dumps(report).encode()
    receipt['report_sha256'] = hashlib.sha256(raw).hexdigest()
    assert L.validate_run(receipt, report, receipt['report_sha256'])['lanes'] == ['lane', 'other']


def test_optional_absence_is_checked_against_installed_manifest(tmp_path, monkeypatch):
    path, record = convert(tmp_path, monkeypatch)
    contract={'_mojolearn_tokenizer_fast':dict(present=False,predicate=dict(basename='_mojolearn_tokenizer_fast.so'))}
    C.validate_optional_install_files(record,path,contract)
    contract['_mojolearn_tokenizer_fast']['present']=True
    with pytest.raises(ValueError,match='presence differs'):
        C.validate_optional_install_files(record,path,contract)
    present={'native':dict(present=False,predicate=dict(basename='native.so'))}
    with pytest.raises(ValueError,match='presence differs'):
        C.validate_optional_install_files(record,path,present)


def test_optional_absence_cannot_use_unwitnessed_legacy_column(tmp_path,monkeypatch):
    path, record = convert(tmp_path,monkeypatch)
    record['verifier_observation'].pop('launch')
    with pytest.raises(ValueError,match='retained verifier filename'):
        C.validate_optional_install_files(record,path,{'optional':dict(present=False,predicate=dict(basename='optional.so'))})


@pytest.mark.parametrize('key,value', [('MOJOLEARN_IDENTITY_N','16'), ('FIXTURES','base'), ('REPEATS','2'), ('MOJOLEARN_TEST_SABOTAGE','1')])
def test_launch_reuses_existing_full_fixture_environment_policy(tmp_path,monkeypatch,key,value):
    monkeypatch.setenv(key,value)
    with pytest.raises(ValueError):
        L.execute(tmp_path,'/env/python',tmp_path/'refused',['lane'],'routine',8,
                  runner=lambda *a,**k: pytest.fail('narrowed fixtures must never execute'))


@pytest.mark.parametrize('damage', ['candidate', 'missing', 'upload_candidate', 'legacy'])
def test_default_column_requires_explicit_current_policy(tmp_path, monkeypatch, damage):
    def break_policy(report, receipt):
        if damage == 'legacy':
            receipt['schema'] = L.LEGACY_SCHEMA
            receipt['environment'].pop('resident_sampling_fp32')
        elif damage == 'missing':
            receipt['environment'].pop('resident_sampling_fp32')
        elif damage == 'upload_candidate':
            receipt['environment']['sampling_kv_upload_reuse'] = '1'
        else:
            receipt['environment']['resident_sampling_fp32'] = '1'
    with pytest.raises(ValueError, match='sampling policy'):
        convert(tmp_path, monkeypatch, break_policy)


def test_legacy_policy_refusal_does_not_rewrite_observation(tmp_path):
    report, _, receipt, _, _ = setup_report(tmp_path)
    receipt['schema'] = L.LEGACY_SCHEMA
    receipt.pop('environment')
    original = copy.deepcopy(receipt)
    with pytest.raises(ValueError, match='legacy.*unrecorded'):
        L.validate_run(receipt, report, receipt['report_sha256'])
    assert receipt == original
